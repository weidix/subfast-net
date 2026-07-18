"""Command-line dispatcher for the independent tools package."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable, Sequence


_HELP = """usage: subfast-tools <command> [options]

Commands:
  build-samples       Extract and label subtitle detector samples
  synthesize-samples  Generate synthetic subtitle samples
  prepare-roi         Build fixed-size ROI datasets
  extract-craft       Save CRAFT score maps for ROI samples
  labels-to-via       Convert YOLO labels to VIA JSON
  via-to-labels       Convert VIA JSON to YOLO labels
  review-labels       Review detector labels in a browser
  review-roi          Review ROI subtitle segments in a browser

Run ``subfast-tools <command> --help`` for command-specific options.
"""

_ROUTES: dict[str, tuple[str, str]] = {
    "build-samples": ("tools.build_samples", "main"),
    "synthesize-samples": ("tools.synthesize_samples", "main"),
    "prepare-roi": ("tools.prepare_roi", "main"),
    "extract-craft": ("tools.craft", "main"),
    "review-labels": ("tools.review_labels", "main"),
    "review-roi": ("tools.review_roi", "main"),
}
_VIA_COMMANDS = frozenset({"labels-to-via", "via-to-labels"})


def _invoke(
    module_name: str,
    function_name: str,
    argv: list[str],
    *,
    prog: str,
) -> int:
    module = import_module(module_name)
    function: Callable[..., object] = getattr(module, function_name)
    previous_program = sys.argv[0]
    sys.argv[0] = prog
    try:
        result = function(argv)
    finally:
        sys.argv[0] = previous_program
    return int(result) if isinstance(result, int) else 0


def _fail(message: str) -> int:
    print(f"subfast-tools: error: {message}", file=sys.stderr)
    print("Run 'subfast-tools --help' for usage.", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a tools command and return its process status."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_HELP.rstrip())
        return 0

    command = args[0]
    if command in _VIA_COMMANDS:
        return _invoke(
            "tools.via_labels",
            "main",
            [command, *args[1:]],
            prog="subfast-tools",
        )

    route = _ROUTES.get(command)
    if route is None:
        return _fail(f"unknown command '{command}'")
    return _invoke(*route, args[1:], prog=f"subfast-tools {command}")


if __name__ == "__main__":
    raise SystemExit(main())
