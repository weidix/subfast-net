# ROI Balanced Training Design

## Goal

Make ROI presence and embedding training receive controlled, observable positive/negative distributions without discarding source samples or weakening held-out validation quality.

## Current behavior and root causes

- `negative_ratio` is passed to dataset limiting. It has no effect when `max_train_samples` is unset or is not smaller than the dataset. With the current configured data, the requested `0.35` empty ratio therefore becomes the source distribution of `6211 / 9247 = 0.672`.
- The training loader uses unrestricted shuffle, so individual batches have no presence-class guarantee.
- Empty ROI samples do not participate in embedding loss. Embedding positives and negatives mean same-segment and different-segment pairs among subtitle-present samples; they are distinct from the presence `negative_ratio` concept.
- Random batches almost never contain two samples from the same segment. Across three measured shuffles, embedding pairs were more than 99.96% negative and some batches contained no usable embedding pair.

## Configuration semantics

- Keep `negative_ratio` as the target fraction of empty ROI samples in each training batch. It controls presence supervision only.
- Add `embedding_negative_ratio`, defaulting to `0.5`, as the target fraction of different-segment pairs contributing to embedding pair loss.
- Validate both ratios in `[0, 1]`. Batch construction must reject a configuration that cannot allocate at least one presence-positive slot and one presence-negative slot when both classes exist.
- Validation data remains unbalanced and unmodified so metrics reflect the configured held-out distribution.

## Presence-balanced batch sampler

Add a focused ROI batch sampler operating on dataset metadata rather than changing dataset contents.

For each batch it will:

1. Allocate empty and subtitle-present slots from `batch_size` using `negative_ratio`, with deterministic rounding and at least one slot per available class.
2. Fill subtitle-present slots with same-segment groups before singleton positives so usable embedding-positive pairs are created.
3. Cycle shuffled class queues deterministically from `seed + epoch` when a class is exhausted.
4. Set epoch length so every original sample is visited at least once. Minority-class samples may repeat; no source sample is deleted.
5. Expose `set_epoch(epoch)` so ordering changes reproducibly between epochs.

When only one class exists, the sampler uses all slots for that class and reports that the requested ratio cannot be achieved. A final partial batch is not emitted because it would break the declared distribution; queue cycling fills a complete batch instead.

The `DataLoader` uses this object through `batch_sampler`, replacing `batch_size`, `shuffle`, and `drop_last` arguments only for training. Validation loading remains unchanged.

## Embedding pair balancing

Pair discovery remains responsible for identifying valid local and OCR-derived pairs. After similarities are computed, pair reduction will:

1. Retain all available positive pairs.
2. Select the highest-similarity negative pairs first, because these are the hard false-match cases that determine embedding quality.
3. Limit the selected negative count to the configured `embedding_negative_ratio` relative to the retained positive count.
4. If a batch has no positive pair, retain a bounded hard-negative contribution rather than producing an unbounded all-negative loss; the segment-aware sampler should make this an exceptional, observable case.
5. Compute positive and negative loss means separately before combining them, so a large class cannot dilute the other class.

The existing uncommitted margin-loss, embedding-memory, and local-contrast work is preserved. Pair balancing applies to the in-batch pair loss; memory-pair behavior remains separately reported so stale memory entries do not silently determine the configured in-batch ratio.

## Metrics and observability

Training step and epoch metrics will include:

- presence positive/negative sample counts and realized negative ratio;
- embedding positive/negative selected-pair counts and realized negative ratio;
- embedding candidate counts before balancing;
- batches without an embedding-positive pair;
- existing local-positive, local-negative, OCR-negative, skipped-pair, presence, and embedding metrics.

The startup summary will print both configured ratio meanings explicitly.

## Quality protection

The change is accepted only when all of the following hold:

1. Focused sampler tests prove batch presence ratios, complete source-sample coverage, reproducibility, epoch reshuffling, and same-segment positive pairing.
2. Focused loss tests prove pair-ratio limiting and hard-negative selection.
3. The ROI training smoke test proves the sampler and loss integrate through one epoch.
4. A same-seed baseline/new A/B run uses the configured held-out validation source. The new run must stay within the existing `0.01` stability tolerance for global/normal/short presence F1, global/normal/style-hard-negative embedding accuracy, hard-negative similarity P90, and hard margin.
5. Dataset and training metrics must demonstrate that the configured ratios were actually realized; loss curves alone are not quality evidence.

If the configured validation root overlaps the training roots, the command must report that fact and the run cannot be used to claim held-out non-regression. Implementation correctness can still be tested, but quality completion requires a non-overlapping configured validation source.

## Scope

This change is limited to ROI dataset metadata sampling, embedding pair reduction, configuration, training metrics, and focused tests. It does not alter OCR, subtitle parsing, detector geometry, validation distribution, or unrelated model architecture.
