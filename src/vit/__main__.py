"""Enable ``python -m vit ...`` to invoke the CLI."""

from __future__ import annotations

from vit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
