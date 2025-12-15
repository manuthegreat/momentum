import numpy as np
import pandas as pd


def add_absolute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["1D Return"] = df.groupby("Ticker")["Price"].pct_change() + 1.0
    return df


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
