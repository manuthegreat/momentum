"""
Seed backtest history (GitHub Actions job: seed-history)

This version is designed to align with the current core/ layout:
core/
  __init__.py
  data.py
  data_utils.py
  features.py
  selection.py
  backtest.py

Key goals:
- DO NOT depend on non-existent `core.momentum_utils`
- Prefer calling a single high-level "seed"/"run_backtest" entrypoint if core provides it
- Provide clear diagnostics if core APIs changed
"""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
# Ensure repo root is on sys.path (Actions does export PYTHONPATH=$PWD,
# but this makes the script robust when run locally too.)
# -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _first_attr(module, names: list[str]):
    """Return the first attribute found on module from a list of candidates."""
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None


def _call_with_supported_kwargs(fn, **kwargs):
    """
    Call fn with only the kwargs that are accepted by fn.
    Helps keep compatibility as function signatures evolve.
    """
    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(**accepted)


def main() -> None:
    # -----------------------------
    # Config (override via env vars)
    # -----------------------------
    output_path = os.getenv("BACKTEST_HISTORY_PATH", "").strip()
    if not output_path:
        # sensible default used by many repos
        output_path = str(REPO_ROOT / "data" / "backtest_history.parquet")

    # optional knobs (only used if core functions support them)
    start_date = os.getenv("BACKTEST_START_DATE", "").strip()  # e.g. "2018-01-01"
    end_date = os.getenv("BACKTEST_END_DATE", "").strip()      # e.g. "2025-12-15"
    top_n = int(os.getenv("BACKTEST_TOP_N", "20"))
    universe = os.getenv("BACKTEST_UNIVERSE", "").strip()      # e.g. "sp500" / "all"

    # Parse dates if provided
    def _parse(d: str):
        return datetime.strptime(d, "%Y-%m-%d").date() if d else None

    start_date_d = _parse(start_date)
    end_date_d = _parse(end_date)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Imports from core/*
    # -----------------------------
    try:
        import core  # noqa: F401
        from core import backtest as core_backtest
    except Exception as e:
        raise RuntimeError(
            f"Failed to import core/backtest modules. "
            f"Repo root: {REPO_ROOT}\n"
            f"Error: {e}"
        ) from e

    # Optional modules (only used if needed)
    core_data = None
    core_features = None
    core_selection = None
    try:
        from core import data as core_data  # type: ignore
    except Exception:
        pass
    try:
        from core import features as core_features  # type: ignore
    except Exception:
        pass
    try:
        from core import selection as core_selection  # type: ignore
    except Exception:
        pass

    # ---------------------------------------------------------
    # Preferred: call a single high-level seed/backtest function
    # ---------------------------------------------------------
    preferred_entrypoints = [
        "seed_backtest_history",
        "generate_backtest_history",
        "build_backtest_history",
        "run_backtest_history",
        "run_backtest",
        "backtest",
    ]

    entry = _first_attr(core_backtest, preferred_entrypoints)

    if entry is not None and callable(entry):
        print(f"[seed-history] Using core.backtest.{entry.__name__}()")

        result = _call_with_supported_kwargs(
            entry,
            output_path=output_path,
            out_path=output_path,
            path=output_path,
            start_date=start_date_d,
            end_date=end_date_d,
            top_n=top_n,
            universe=universe,
        )

        # Some implementations may return a DF; if so, try saving it.
        if result is not None:
            try:
                import pandas as pd  # noqa: F401
                if hasattr(result, "to_parquet"):
                    result.to_parquet(output_path, index=False)
                    print(f"[seed-history] Saved backtest history -> {output_path}")
            except Exception:
                # If core already saves internally, that’s fine.
                pass

        print("[seed-history] Done.")
        return

    # ---------------------------------------------------------
    # Fallback: attempt to assemble via data/features/selection
    # (Only if core.backtest doesn't expose a usable entrypoint)
    # ---------------------------------------------------------
    # We keep this fallback conservative: we only proceed if we can
    # find obvious functions. Otherwise we fail with diagnostics.
    loader_candidates = []
    if core_data is not None:
        loader_candidates += [
            "load_all_market_data",
            "load_market_data",
            "load_price_data",
            "load_price_data_parquet",
        ]
    feature_candidates = []
    if core_features is not None:
        feature_candidates += [
            "apply_technical_indicators",
            "add_features",
            "compute_features",
            "apply_features",
        ]
    selector_candidates = []
    if core_selection is not None:
        selector_candidates += [
            "select_portfolio",
            "select_top_momentum",
            "build_portfolio",
            "pick_top",
        ]

    loader = _first_attr(core_data, loader_candidates) if core_data else None
    featurizer = _first_attr(core_features, feature_candidates) if core_features else None
    selector = _first_attr(core_selection, selector_candidates) if core_selection else None

    if not (callable(loader) and callable(featurizer) and callable(selector)):
        # Give a very actionable error with what we *did* find
        available = {
            "core.backtest": [n for n in dir(core_backtest) if not n.startswith("_")],
            "core.data": [n for n in dir(core_data) if core_data and not n.startswith("_")] if core_data else [],
            "core.features": [n for n in dir(core_features) if core_features and not n.startswith("_")] if core_features else [],
            "core.selection": [n for n in dir(core_selection) if core_selection and not n.startswith("_")] if core_selection else [],
        }
        raise RuntimeError(
            "core.backtest does not expose a recognized history/backtest entrypoint, "
            "and fallback wiring could not find the expected functions.\n\n"
            f"Expected one of these in core.backtest: {preferred_entrypoints}\n"
            f"Fallback requires:\n"
            f" - data loader in core.data: {loader_candidates}\n"
            f" - feature fn in core.features: {feature_candidates}\n"
            f" - selector fn in core.selection: {selector_candidates}\n\n"
            f"Discovered symbols:\n{available}\n"
        )

    print("[seed-history] Using fallback wiring via core.data/core.features/core.selection")
    print(f" - loader: {loader.__name__}")
    print(f" - featurizer: {featurizer.__name__}")
    print(f" - selector: {selector.__name__}")

    # Minimal generic fallback implementation:
    # load -> feature -> select per-date -> write history
    import pandas as pd

    df = _call_with_supported_kwargs(loader, universe=universe)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("Loader returned no data (empty or non-DataFrame).")

    df = _call_with_supported_kwargs(featurizer, df=df) or df

    # Require Date/Ticker columns for generic history creation
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise RuntimeError("Fallback requires df to include 'Date' and 'Ticker' columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if start_date_d:
        df = df[df["Date"].dt.date >= start_date_d]
    if end_date_d:
        df = df[df["Date"].dt.date <= end_date_d]

    all_dates = sorted(df["Date"].dt.normalize().unique())
    rows = []

    for d in all_dates:
        day_df = df[df["Date"].dt.normalize() == d]
        picks = _call_with_supported_kwargs(selector, df=day_df, top_n=top_n)

        # Normalize picks into a DataFrame
        if picks is None:
            continue
        if isinstance(picks, pd.DataFrame):
            out = picks.copy()
        elif isinstance(picks, (list, tuple)):
            out = pd.DataFrame({"Ticker": list(picks)})
        else:
            # e.g. dict
            out = pd.DataFrame(picks)

        if "Date" not in out.columns:
            out["Date"] = d

        rows.append(out)

    if not rows:
        raise RuntimeError("Fallback produced no selections; nothing to write.")

    hist = pd.concat(rows, ignore_index=True)
    hist.to_parquet(output_path, index=False)
    print(f"[seed-history] Saved backtest history -> {output_path}")
    print("[seed-history] Done.")


if __name__ == "__main__":
    main()
