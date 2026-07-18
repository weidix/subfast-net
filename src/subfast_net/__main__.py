"""Allow ``python -m subfast_net`` to use the unified CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
