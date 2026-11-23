"""Command-line entry point. Argument parsing only; delegates to ``raw_pipeline.runtime``."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Run the ``raw-pipeline`` command and return the process exit code."""
    del argv
    msg = "raw-pipeline CLI is not implemented yet (work package 'runtime-cli')"
    raise SystemExit(msg)
