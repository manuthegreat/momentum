import pandas as pd


def build_daily_lists(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    records = []

    for d in sorted(df["Date"].unique()):
        snap = df[df["Date"] == d].sort_values("Momentum Score", ascending=False).head(top_n)
        for _, r in snap.iterrows():
            records.append({
                "Date": d,
                "Ticker": r["Ticker"],
                "Momentum Score": r["Momentum Score"],
                "Early Momentum Score": r["Early Momentum Score"],
            })

    return pd.DataFrame(records)


def final_selection_from_daily(
    daily_df: pd.DataFrame,
    lookback_days: int = 10,
    w_momentum: float = 0.5,
    w_early: float = 0.3,
    w_consistency: float = 0.2,
    as_of_date=None,
    top_n: int = 10
) -> pd.DataFrame:

    if as_of_date is None:
        as_of_date = daily_df["Date"].max()

    dates = sorted(
        daily_df[daily_df["Date"] <= as_of_date]["Date"].unique(),
        reverse=True
    )[:lookback_days]

    window = daily_df[daily_df["Date"].isin(dates)]

    agg = (
        window.groupby("Ticker")
        .agg(
            Momentum_Score=("Momentum Score", "mean"),
            Early_Momentum_Score=("Early Momentum Score", "mean"),
            Appearances=("Date", "count"),
        )
        .reset_index()
    )

    agg["Consistency"] = agg["Appearances"] / len(dates)
    agg["Weighted_Score"] = (
        w_momentum * agg["Momentum_Score"]
        + w_early * agg["Early_Momentum_Score"]
        + w_consistency * agg["Consistency"]
    )

    return agg.sort_values("Weighted_Score", ascending=False).head(top_n)
