"""Standalone command dispatcher for deployment exports."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


_ROUTES: dict[str, tuple[str, str]] = {
    "unified": ("subfast_export.unified", "main"),
    "coreml": ("subfast_export.unified", "main_coreml"),
    "safetensors": ("subfast_export.safetensors", "main"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Export one checkpoint using ``unified``, ``coreml``, or ``safetensors``."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: subfast-export <unified|coreml|safetensors> [options]")
        return 0
    command = arguments.pop(0)
    try:
        module_name, function_name = _ROUTES[command]
    except KeyError:
        print(f"subfast-export: error: unknown export format '{command}'", file=sys.stderr)
        return 2
    from importlib import import_module

    function: Callable[[list[str]], object] = getattr(
        import_module(module_name), function_name
    )
    result = function(arguments)
    return int(result) if isinstance(result, int) else 0
