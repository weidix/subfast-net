from __future__ import annotations

import sys

from . import export_safetensors, export_unified_model, train, train_presence, train_roi, train_roi_pair


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "export-unified":
        export_unified_model.main(args[1:])
        return
    if args and args[0] == "export-coreml":
        export_unified_model.main_coreml(args[1:])
        return
    if args and args[0] == "export-safetensors":
        export_safetensors.main(args[1:])
        return
    if args and args[0] == "train":
        train.main(args[1:])
        return
    if args and args[0] == "train-presence":
        train_presence.main(args[1:])
        return
    if args and args[0] == "benchmark-presence":
        train_presence.main_benchmark(args[1:])
        return
    if args and args[0] == "train-roi":
        train_roi.main(args[1:])
        return
    if args and args[0] == "validate-roi":
        train_roi.main_validate(args[1:])
        return
    if args and args[0] == "train-roi-pair":
        train_roi_pair.main(args[1:])
        return
    if args and args[0] == "validate-roi-pair":
        train_roi_pair.main_validate(args[1:])
        return
    train.main(args)


if __name__ == "__main__":
    main()
