import numpy as np
import pandas as pd


def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1.0
    return df

def compute_index_momentum(idx_df: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
    """
    Computes rolling momentum for indices using daily returns.
    Robust to different column names in index parquet.
    """

    df = idx_df.copy()

    # -----------------------------------------
    # Normalize column names
    # -----------------------------------------
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # normalize index column
    if "index" not in df.columns:
        for c in df.columns:
            if c in ("index_name", "benchmark", "symbol", "ticker"):
                df = df.rename(columns={c: "index"})
                break

    # normalize daily return column
    ret_candidates = [
        "idx_ret_1d",
        "index_ret_1d",
        "ret_1d",
        "return_1d",
        "daily_return",
    ]

    ret_col = None
    for c in ret_candidates:
        if c in df.columns:
            ret_col = c
            break

    if ret_col is None:
        raise ValueError(
            f"compute_index_momentum: no daily return column found. Columns={df.columns.tolist()}"
        )

    if ret_col != "idx_ret_1d":
        df = df.rename(columns={ret_col: "idx_ret_1d"})

    # -----------------------------------------
    # Compute momentum
    # -----------------------------------------
    df = df.sort_values(["index", "date"])

    out = df[["date", "index"]].copy()

    for w in windows:
        out[f"idx_{w}D"] = (
            df.groupby("index")["idx_ret_1d"]
              .rolling(w)
              .apply(lambda x: np.prod(1 + x) - 1, raw=False)
              .reset_index(level=0, drop=True)
        )

    return out.dropna()


def calculate_momentum_features(df: pd.DataFrame, windows=(5, 10, 30, 45, 60, 90)) -> pd.DataFrame:
    df = df.copy()

    for w in windows:
        r = f"{w}D Return"
        z = f"{w}D zscore"

        df[r] = (
            df.groupby("Ticker")["1D Return"]
            .rolling(w, min_periods=w)
            .apply(np.prod, raw=True)
            .reset_index(level=0, drop=True) - 1
        )

        mean = df.groupby("Date")[r].transform("mean")
        std = df.groupby("Date")[r].transform("std").replace(0, np.nan)
        df[z] = ((df[r] - mean) / std).fillna(0.0)

    return df.fillna(0.0)

def add_relative_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    True relative regime momentum:
    - Build relative returns vs index momentum (benchmark subtraction)
    - Cross-sectional z-score AFTER subtraction
    - Define Momentum_Fast/Mid/Slow from RELATIVE z-scores
    - Define Momentum Score from those regimes

    Requires:
      - Stock columns: 5D Return, 10D Return, 30D Return, 45D Return, 60D Return, 90D Return
      - Index columns: idx_5D, idx_10D, idx_30D, idx_45D, idx_60D, idx_90D
    """
    out = df.copy()

    required_stock = {"5D Return", "10D Return", "30D Return", "45D Return", "60D Return", "90D Return"}
    required_idx = {"idx_5D", "idx_10D", "idx_30D", "idx_45D", "idx_60D", "idx_90D"}
    missing = (required_stock | required_idx) - set(out.columns)
    if missing:
        raise ValueError(f"add_relative_regime_momentum_score missing columns: {sorted(missing)}")

    # --- Relative returns (stock minus index) using your local weighting ---
    out["Rel_Slow"] = (0.5 * out["60D Return"] + 0.5 * out["90D Return"]) - (0.5 * out["idx_60D"] + 0.5 * out["idx_90D"])
    out["Rel_Mid"]  = (0.5 * out["30D Return"] + 0.5 * out["45D Return"]) - (0.5 * out["idx_30D"] + 0.5 * out["idx_45D"])
    out["Rel_Fast"] = (0.6 * out["5D Return"]  + 0.4 * out["10D Return"]) - (0.6 * out["idx_5D"]  + 0.4 * out["idx_10D"])

    # --- Cross-sectional z-scores AFTER benchmark subtraction ---
    for col in ["Rel_Slow", "Rel_Mid", "Rel_Fast"]:
        mean = out.groupby("Date")[col].transform("mean")
        std = out.groupby("Date")[col].transform("std").replace(0, np.nan)
        out[col + "_z"] = ((out[col] - mean) / std).fillna(0.0)

    # --- Define regimes from relative z-scores ---
    out["Momentum_Slow"] = out["Rel_Slow_z"]
    out["Momentum_Mid"]  = out["Rel_Mid_z"]
    out["Momentum_Fast"] = out["Rel_Fast_z"]

    out["Momentum Score"] = (
        0.5 * out["Momentum_Slow"] +
        0.3 * out["Momentum_Mid"] +
        0.2 * out["Momentum_Fast"]
    )

    return out.fillna(0.0)

def add_regime_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Momentum_Fast"] = 0.6 * df["5D zscore"] + 0.4 * df["10D zscore"]
    df["Momentum_Mid"] = 0.5 * df["30D zscore"] + 0.5 * df["45D zscore"]
    df["Momentum_Slow"] = 0.5 * df["60D zscore"] + 0.5 * df["90D zscore"]

    df["Momentum Score"] = (
        0.5 * df["Momentum_Slow"]
        + 0.3 * df["Momentum_Mid"]
        + 0.2 * df["Momentum_Fast"]
    )

    return df.fillna(0.0)


def add_regime_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Momentum_Fast", "Momentum_Mid", "Momentum_Slow"]:
        df[f"Accel_{col}"] = df.groupby("Ticker")[col].diff()

    def zscore(x):
        s = x.std()
        return ((x - x.mean()) / s).fillna(0.0) if s and s > 0 else x.fillna(0.0)

    df["Accel_Score"] = (
        0.5 * df.groupby("Date")["Accel_Momentum_Fast"].transform(zscore)
        + 0.3 * df.groupby("Date")["Accel_Momentum_Mid"].transform(zscore)
        + 0.2 * df.groupby("Date")["Accel_Momentum_Slow"].transform(zscore)
    )

    return df.fillna(0.0)


def add_regime_early_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Early Momentum Score"] = (
        0.6 * df["Accel_Score"] + 0.4 * df["Momentum Score"]
    )

    return df.fillna(0.0)
