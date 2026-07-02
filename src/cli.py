from __future__ import annotations

import sys

from . import export_unified_model, train, train_roi


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "export-unified":
        export_unified_model.main(args[1:])
        return
    if args and args[0] == "export-coreml":
        export_unified_model.main_coreml(args[1:])
        return
    if args and args[0] == "train":
        train.main(args[1:])
        return
    if args and args[0] == "train-roi":
        train_roi.main(args[1:])
        return
    if args and args[0] == "validate-roi":
        train_roi.main_validate(args[1:])
        return
    train.main(args)


if __name__ == "__main__":
    main()
