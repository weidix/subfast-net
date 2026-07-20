"""Focused visual H.264 streaming CLI backed by the compatibility router."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser(prog: str = "h264-stream-timing") -> argparse.ArgumentParser:
    """Build the focused visual H.264 streaming command parser."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train or run visual H.264 causal subtitle timing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train the causal H.264 streaming model")
    subparsers.add_parser("infer", help="Run causal H.264 streaming inference")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run visual H.264 streaming training or inference.

    ``train`` and ``infer`` are mapped to the established H.264 compatibility
    commands, preserving their options and checkpoint contracts.
    """

    if argv is None:
        import sys

        argv = sys.argv[1:]
    arguments = list(argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        build_parser().print_help()
        return
    command = build_parser().parse_args([arguments.pop(0)]).command
    command_map = {"train": "train-stream", "infer": "stream-infer"}
    mapped_command = command_map[command]
    from h264_timing.cli import main as h264_timing_main

    h264_timing_main([mapped_command, *arguments], prog="h264-stream-timing")
