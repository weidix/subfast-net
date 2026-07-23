"""Command-line dispatcher for the subtitle model package.

The individual command modules own their argument parsing.  Keeping this
module as a small lazy dispatcher means ``subfast-net --help`` and unrelated
commands do not import PyTorch, OCR, or Core ML optional dependencies.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable, Sequence


_HELP = """usage: subfast-net <group> <command> [options]

Groups:
  train       Train one of the detector or ROI models
  validate    Validate a trained model
  benchmark   Measure inference performance
  export      Export a checkpoint for deployment

Run ``subfast-net <group> <command> --help`` for command-specific options.

Dataset and review utilities use the independent ``subfast-tools`` command.

Each command has one canonical grouped spelling; use ``--help`` to discover
its options.
"""


_ROUTES: dict[tuple[str, str], tuple[str, str]] = {
    # Training families.
    ("train", "detector"): ("subfast_detector.train", "main"),
    ("train", "frame-presence"): ("subfast_frame_presence.train", "main"),
    ("train", "presence"): ("subfast_roi_presence.train", "main"),
    ("train", "matcher"): ("subfast_roi_matcher.train", "main"),
    ("train", "embedding"): ("subfast_roi_embedding.train", "main"),
    # Validation and benchmarks.
    ("validate", "matcher"): ("subfast_roi_matcher.train", "main_validate"),
    ("validate", "embedding"): ("subfast_roi_embedding.train", "main_validate"),
    ("benchmark", "frame-presence"): ("subfast_frame_presence.train", "main_benchmark"),
    ("benchmark", "presence"): ("subfast_roi_presence.train", "main_benchmark"),
    # Export formats.
    ("export", "unified"): ("subfast_export.unified", "main"),
    ("export", "coreml"): ("subfast_export.unified", "main_coreml"),
    ("export", "safetensors"): ("subfast_export.safetensors", "main"),
}

_GROUP_COMMANDS: dict[str, tuple[str, ...]] = {
    "train": ("detector", "frame-presence", "presence", "embedding", "matcher"),
    "validate": ("embedding", "matcher"),
    "benchmark": ("frame-presence", "presence"),
    "export": ("unified", "coreml", "safetensors"),
}


def _show_help() -> int:
    print(_HELP.rstrip())
    return 0


def _show_group_help(group: str) -> int:
    print(f"usage: subfast-net {group} <command> [options]\n")
    print(f"{group} commands:")
    for command in _GROUP_COMMANDS[group]:
        print(f"  {command}")
    print(f"\nRun 'subfast-net {group} <command> --help' for command options.")
    return 0


def _fail(message: str) -> int:
    print(f"subfast-net: error: {message}", file=sys.stderr)
    print("Run 'subfast-net --help' for usage.", file=sys.stderr)
    return 2


def _invoke(
    module_name: str,
    function_name: str,
    argv: list[str],
    *,
    prog: str | None = None,
) -> int:
    module = import_module(module_name)
    function: Callable[..., object] = getattr(module, function_name)
    previous_program = sys.argv[0]
    if prog is not None:
        sys.argv[0] = prog
    try:
        result = function(argv)
    finally:
        sys.argv[0] = previous_program
    return int(result) if isinstance(result, int) else 0


def _dispatch(args: list[str]) -> int:
    if not args:
        return _show_help()

    command = args[0]
    if command in {"-h", "--help"}:
        return _show_help()
    if command == "--version":
        from subfast_shared import __version__

        print(__version__)
        return 0

    if command in {"validate", "benchmark", "export", "train"}:
        if len(args) < 2 or args[1] in {"-h", "--help"}:
            return _show_group_help(command)
        subgroup = args[1]
        route = _ROUTES.get((command, subgroup))
        if route is None:
            return _fail(f"unknown {command} subcommand '{subgroup}'")
        return _invoke(*route, args[2:], prog=f"subfast-net {command} {subgroup}")

    return _fail(f"unknown command '{command}'")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a command and return its process status."""

    args = list(sys.argv[1:] if argv is None else argv)
    return _dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
