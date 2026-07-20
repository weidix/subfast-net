"""Focused CLI for compressed-domain H.264 streaming timing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser(prog: str = "h264-compressed-stream-timing") -> argparse.ArgumentParser:
    """Build the focused compressed-only streaming command parser."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Prepare, train, or run compressed-only H.264 causal timing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Build compressed-only feature caches")
    subparsers.add_parser("train", help="Train the compressed-only causal model")
    subparsers.add_parser("infer", help="Run compressed-only causal inference")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run compressed feature preparation, training, or inference.

    The command surface is intentionally small while retaining the established
    command implementations and their checkpoint contracts.
    """

    if argv is None:
        import sys

        argv = sys.argv[1:]
    arguments = list(argv)

    if not arguments or arguments[0] in {"-h", "--help"}:
        build_parser().print_help()
        return
    command = build_parser().parse_args([arguments.pop(0)]).command
    command_map = {
        "prepare": "prepare-compressed-stream",
        "train": "train-compressed-stream",
        "infer": "compressed-stream-infer",
    }
    mapped_command = command_map[command]
    from h264_timing.cli import main as h264_timing_main

    h264_timing_main([mapped_command, *arguments], prog="h264-compressed-stream-timing")
