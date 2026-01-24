from __future__ import annotations

import argparse
import subprocess
import sys


def _run_step(label: str, args: list[str]) -> None:
    print(f"\n--- {label} ---")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the daily update pipeline and (optionally) the backtests.",
    )
    parser.add_argument(
        "--skip-backtests",
        action="store_true",
        help="Only refresh data/signals (skip backtest generation).",
    )
    args = parser.parse_args()

    python = sys.executable
    _run_step("Daily data + signals", [python, "updater/update_parquet.py"])

    if not args.skip_backtests:
        _run_step("Backtests", [python, "updater/run_backtest_systems.py"])


if __name__ == "__main__":
    main()
