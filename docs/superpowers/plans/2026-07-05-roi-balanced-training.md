# ROI Balanced Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make negative_ratio control presence composition in every ROI training batch and independently balance embedding positive/negative pairs without losing source samples or degrading held-out metrics.

**Architecture:** Add a metadata-only, segment-aware batch sampler for presence balance and valid same-segment pair availability. Keep pair discovery unchanged, then apply hard-negative selection and class-separated reduction inside embedding loss using embedding_negative_ratio. Integrate both controls into train-roi, expose realized distributions, and protect quality with held-out validation metrics.

**Tech Stack:** Python 3.13, PyTorch Sampler/DataLoader, Pydantic, unittest, uv.

---

## File structure

- Create src/roi_sampler.py for deterministic presence-balanced, segment-aware batches.
- Modify src/roi_config.py for the independent embedding ratio.
- Modify src/roi_loss.py for hard-negative pair limiting and pair-count metrics.
- Modify src/train_roi.py for integration, metrics, and validation-overlap reporting.
- Modify tests/test_roi_presence_embedding.py with focused tests.
- Modify docs/roi_presence_embedding_training_commands.md with exact ratio semantics.

### Task 0: Capture the pre-sampler held-out baseline

**Files:**
- Runtime outputs only: outputs/roi_balance_baseline

- [ ] **Step 1: Run the current preserved implementation before changing production code**

~~~bash
uv run subfast-net train-roi \
  --train-root data/roi_samples1 --train-root data/roi_samples2 \
  --train-root data/roi_samples4 --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_balance_baseline \
  --resize-roi 256x64 --batch-size 32 --epochs 3 --lr 0.0003 \
  --negative-ratio 0.35 --device auto
~~~

Expected: summary.json and best.pt exist. Preserve this output until Task 6 completes.

### Task 1: Presence-balanced, segment-aware batch sampler

**Files:**
- Create: src/roi_sampler.py
- Modify: tests/test_roi_presence_embedding.py

- [ ] **Step 1: Write the failing sampler tests**

Add imports for Counter and RoiBalancedBatchSampler, a metadata-only sample factory, and these behaviors:

~~~python
def make_sampler_samples(
    *,
    positives: int,
    negatives: int,
    repeated_segments: bool = True,
) -> list[RoiSample]:
    samples = []
    for index in range(positives):
        segment_number = index // 2 if repeated_segments else index
        samples.append(RoiSample(
            image_path=Path(f"positive-{index}.jpg"),
            label_path=Path(f"positive-{index}.txt"),
            sample_id=f"positive-{index}",
            root=Path("root"),
            has_subtitle=True,
            segment_id=f"segment-{segment_number}",
            video_id="video",
            frame_index=segment_number * 300 + index % 2 * 30,
            ocr_text=f"subtitle-{segment_number}",
            annotation={},
        ))
    for index in range(negatives):
        samples.append(RoiSample(
            image_path=Path(f"negative-{index}.jpg"),
            label_path=Path(f"negative-{index}.txt"),
            sample_id=f"negative-{index}",
            root=Path("root"),
            has_subtitle=False,
            segment_id=f"empty-{index}",
            video_id="video",
            frame_index=10000 + index * 30,
            ocr_text="",
            annotation={},
        ))
    return samples

def test_balanced_batch_sampler_realizes_presence_ratio_and_covers_every_sample(self):
    samples = make_sampler_samples(positives=6, negatives=10)
    sampler = RoiBalancedBatchSampler(
        samples, batch_size=4, negative_ratio=0.5, frame_window=90, seed=7
    )
    batches = list(sampler)
    self.assertTrue(all(
        sum(not samples[index].has_subtitle for index in batch) == 2
        for batch in batches
    ))
    self.assertEqual(
        {index for batch in batches for index in batch},
        set(range(len(samples))),
    )

def test_balanced_batch_sampler_is_reproducible_and_changes_by_epoch(self):
    samples = make_sampler_samples(positives=8, negatives=8)
    left = RoiBalancedBatchSampler(samples, batch_size=4, negative_ratio=0.5, frame_window=90, seed=11)
    right = RoiBalancedBatchSampler(samples, batch_size=4, negative_ratio=0.5, frame_window=90, seed=11)
    self.assertEqual(list(left), list(right))
    left.set_epoch(1)
    self.assertNotEqual(list(left), list(right))

def test_balanced_batch_sampler_places_valid_same_segment_pair_in_every_batch(self):
    samples = make_sampler_samples(positives=8, negatives=8, repeated_segments=True)
    sampler = RoiBalancedBatchSampler(samples, batch_size=4, negative_ratio=0.5, frame_window=90, seed=13)
    for batch in sampler:
        positives = [samples[index] for index in batch if samples[index].has_subtitle]
        self.assertTrue(any(
            left.root == right.root
            and left.video_id == right.video_id
            and left.segment_id == right.segment_id
            and abs(int(left.frame_index) - int(right.frame_index)) <= 90
            for offset, left in enumerate(positives)
            for right in positives[offset + 1:]
        ))
~~~

- [ ] **Step 2: Run the tests and verify RED**

~~~bash
uv run python -m unittest \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_balanced_batch_sampler_realizes_presence_ratio_and_covers_every_sample \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_balanced_batch_sampler_is_reproducible_and_changes_by_epoch \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_balanced_batch_sampler_places_valid_same_segment_pair_in_every_batch -v
~~~

Expected: import failure because src.roi_sampler does not exist.

- [ ] **Step 3: Implement the sampler**

Create RoiBalancedBatchSampler as Sampler[list[int]]. Its constructor accepts samples, batch_size, negative_ratio, frame_window, and seed. It exposes set_epoch(epoch), __len__(), and __iter__().

Compute negative_slots by rounded batch_size times negative_ratio and use the remainder as positive_slots. Require at least two positive slots and one negative slot when both classes exist. Build positive pair candidates from matching root, video_id, and segment_id groups whose non-null frame indices differ by at most frame_window. Shuffle queues with random.Random(seed + epoch), start each mixed batch with a valid pair, fill remaining slots from cycling class queues without duplicates inside a batch, and choose the batch count so every original index is visited.

- [ ] **Step 4: Run the Step 2 command and verify GREEN**

Expected: all three tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add src/roi_sampler.py tests/test_roi_presence_embedding.py
git commit -m "feat(roi): add balanced segment-aware batches"
~~~

### Task 2: Embedding pair ratio and hard-negative selection

**Files:**
- Modify: src/roi_loss.py
- Modify: tests/test_roi_presence_embedding.py

- [ ] **Step 1: Write failing hard-negative tests**

~~~python
def test_balance_embedding_pairs_keeps_all_positives_and_hardest_negatives(self):
    similarities = torch.tensor([0.8, 0.7, 0.9, 0.6, 0.2, -0.1])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    selection = balance_embedding_pairs(similarities, targets, negative_ratio=0.5)
    self.assertEqual(selection.indices.tolist(), [0, 1, 2, 3])
    self.assertEqual(selection.candidate_positive_pairs, 2)
    self.assertEqual(selection.candidate_negative_pairs, 4)
    self.assertEqual(selection.selected_positive_pairs, 2)
    self.assertEqual(selection.selected_negative_pairs, 2)

def test_balance_embedding_pairs_bounds_negative_only_batch(self):
    similarities = torch.tensor([0.1, 0.9, 0.4])
    targets = torch.zeros(3)
    selection = balance_embedding_pairs(similarities, targets, negative_ratio=0.5)
    self.assertEqual(selection.indices.tolist(), [1])
~~~

- [ ] **Step 2: Run both tests and verify RED**

~~~bash
uv run python -m unittest \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_balance_embedding_pairs_keeps_all_positives_and_hardest_negatives \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_balance_embedding_pairs_bounds_negative_only_batch -v
~~~

Expected: import failure for balance_embedding_pairs.

- [ ] **Step 3: Implement selection and metrics**

Add a frozen BalancedEmbeddingPairs dataclass and balance_embedding_pairs. For ratio r below 1, retain all P positives and at most round(P * r / (1 - r)) negatives, choosing negatives by descending similarity. A negative-only batch keeps one hardest negative. Ratio 1 retains every negative. Extend RoiLossBreakdown with candidate and selected positive/negative counts. Pass embedding_negative_ratio through metric_embedding_loss and roi_presence_embedding_loss, and apply selection before embedding_margin_loss.

- [ ] **Step 4: Run focused loss tests and verify GREEN**

~~~bash
uv run python -m unittest \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_balance_embedding_pairs_keeps_all_positives_and_hardest_negatives \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_balance_embedding_pairs_bounds_negative_only_batch \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_embedding_margin_loss_balances_positive_and_negative_pairs -v
~~~

Expected: three tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add src/roi_loss.py tests/test_roi_presence_embedding.py
git commit -m "feat(roi): balance embedding pair supervision"
~~~

### Task 3: Configuration, loader, and metric integration

**Files:**
- Modify: src/roi_config.py
- Modify: src/train_roi.py
- Modify: tests/test_roi_presence_embedding.py

- [ ] **Step 1: Write failing CLI and smoke assertions**

~~~python
def test_parse_args_accepts_embedding_negative_ratio(self):
    settings = parse_args([
        "--negative-ratio", "0.5",
        "--embedding-negative-ratio", "0.4",
    ])
    self.assertEqual(settings.negative_ratio, 0.5)
    self.assertEqual(settings.embedding_negative_ratio, 0.4)
~~~

Change the existing smoke batch size from 2 to 4 so the configured ratio can allocate at least two subtitle-present slots. Require train_presence_negative_ratio, train_embedding_negative_ratio, train_embedding_positive_pairs, and train_embedding_negative_pairs.

- [ ] **Step 2: Run and verify RED**

~~~bash
uv run python -m unittest \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_parse_args_accepts_embedding_negative_ratio \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_one_epoch_roi_training_smoke -v
~~~

Expected: CLI argument rejection and missing metrics.

- [ ] **Step 3: Wire settings and training**

Add embedding_negative_ratio with default 0.5. Add --embedding-negative-ratio validation in [0, 1]. When negative_ratio is set, construct RoiBalancedBatchSampler from train_dataset.samples and use DataLoader batch_sampler instead of batch_size and shuffle. Call set_epoch(epoch) before each epoch. Pass embedding_negative_ratio into ROI loss.

Accumulate and emit these epoch metrics:

~~~text
train_presence_positive_samples
train_presence_negative_samples
train_presence_negative_ratio
train_embedding_candidate_positive_pairs
train_embedding_candidate_negative_pairs
train_embedding_positive_pairs
train_embedding_negative_pairs
train_embedding_negative_ratio
train_embedding_batches_without_positive_pairs
~~~

Print presence_negative_ratio and embedding_negative_ratio separately at startup.

- [ ] **Step 4: Run the Step 2 command and verify GREEN**

Expected: both tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add src/roi_config.py src/train_roi.py tests/test_roi_presence_embedding.py
git commit -m "feat(roi): apply independent training ratios"
~~~

### Task 4: Validation overlap and documentation

**Files:**
- Modify: src/train_roi.py
- Modify: tests/test_roi_presence_embedding.py
- Modify: docs/roi_presence_embedding_training_commands.md

- [ ] **Step 1: Write failing overlap test**

~~~python
def test_validation_overlap_detects_same_resolved_root(self):
    settings = RoiTrainSettings(
        train_roots=[Path("data/roi_samples6")],
        val_root=Path("data/roi_samples6"),
    )
    self.assertTrue(validation_overlaps_training(settings))
    separate = settings.model_copy(update={
        "val_root": Path("data/roi_validation_samples"),
    })
    self.assertFalse(validation_overlaps_training(separate))
~~~

- [ ] **Step 2: Run and verify RED**

~~~bash
uv run python -m unittest \
  tests.test_roi_presence_embedding.RoiPresenceEmbeddingTests.test_validation_overlap_detects_same_resolved_root -v
~~~

Expected: import failure for validation_overlaps_training.

- [ ] **Step 3: Implement and document**

Resolve roots, print a warning when validation overlaps training, and include validation_overlaps_training in validation metrics. Update the command document: negative-ratio is per-training-batch empty fraction; embedding-negative-ratio is selected different-segment pair fraction; validation ratio remains a capped-dataset setting. Add --embedding-negative-ratio 0.5 to the full, effect-check, and resume commands.

- [ ] **Step 4: Run the overlap test and verify GREEN**

Expected: pass.

- [ ] **Step 5: Commit**

~~~bash
git add src/train_roi.py tests/test_roi_presence_embedding.py docs/roi_presence_embedding_training_commands.md
git commit -m "docs(roi): clarify presence and embedding ratios"
~~~

### Task 5: Focused verification and real-data distribution audit

- [ ] **Step 1: Run focused tests**

~~~bash
uv run python -m unittest tests.test_roi_presence_embedding -v
~~~

Expected: all tests pass.

- [ ] **Step 2: Audit full training metadata**

Use the five documented training roots with batch_size 32, negative_ratio 0.35, frame_window 90, and seed 2026. Iterate RoiBalancedBatchSampler without loading images and assert:

~~~python
assert covered == set(range(len(dataset)))
assert abs(total_negative / total_samples - 0.34375) < 1e-12
assert batches_with_valid_positive_pair == len(sampler)
~~~

Expected: all source samples are covered; every full batch has 11 empties and a valid embedding-positive pair.

- [ ] **Step 3: Check preservation and formatting**

~~~bash
git status --short
git diff --check
git diff -- src/roi_config.py src/roi_loss.py src/roi_model.py src/train_roi.py src/roi_sampler.py tests/test_roi_presence_embedding.py docs/roi_presence_embedding_training_commands.md
~~~

Expected: no whitespace errors and the pre-existing local-contrast, margin, and memory changes remain.

### Task 6: Held-out candidate quality gate

Use data/roi_samples1, 2, 4, 5, and 6 for training; data/roi_validation_samples for validation; resize 256x64; batch 32; three epochs; seed 2026.

- [ ] **Step 1: Run the candidate**

~~~bash
uv run subfast-net train-roi \
  --train-root data/roi_samples1 --train-root data/roi_samples2 \
  --train-root data/roi_samples4 --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_balance_candidate \
  --resize-roi 256x64 --batch-size 32 --epochs 3 --lr 0.0003 \
  --negative-ratio 0.35 --embedding-negative-ratio 0.5 --device auto
~~~

Expected: summary.json, best.pt, and realized ratio metrics exist.

- [ ] **Step 2: Enforce non-regression**

~~~python
higher_is_better = (
    "global_presence_f1", "normal_presence_f1", "short_presence_f1",
    "global_embedding_acc", "normal_embedding_acc",
    "style_hard_negative_embedding_acc", "hard_margin",
)
for name in higher_is_better:
    assert candidate[name] + 0.01 >= baseline[name]
assert candidate["hard_negative_sim_p90"] <= baseline["hard_negative_sim_p90"] + 0.01
assert candidate["validation_overlaps_training"] is False
~~~

Expected: every assertion passes. If one fails, keep the goal incomplete and adjust only sampler or pair-balance behavior supported by that failed metric.

- [ ] **Step 3: Rerun focused tests after any evidence-driven correction**

~~~bash
uv run python -m unittest tests.test_roi_presence_embedding -v
git status --short
~~~

Expected: all focused tests pass and runtime outputs are not staged.
