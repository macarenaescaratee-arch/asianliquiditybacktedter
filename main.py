"""
Entry point for AsianLiquidityBacktester CLI workflows.

Future use: argparse/subcommands to run ingestion, a single backtest, batch jobs,
or launch the Streamlit dashboard. For now this module wires the package skeleton
and validates that configuration resolves correctly.
"""

from __future__ import annotations

import sys

from config import PROJECT_ROOT, DATA_DIR, REPORTS_DIR


def main() -> int:
    """Bootstrap check; extend with CLI commands in later phases."""
    print("AsianLiquidityBacktester")
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  data dir:     {DATA_DIR}")
    print(f"  reports dir:  {REPORTS_DIR}")
    print("Skeleton ready — add CLI commands in upcoming phases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
