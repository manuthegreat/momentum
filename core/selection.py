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

def build_unified_target(
    dailyA,
    dailyB,
    as_of_date=None,
    lookback_days=10,
    w_momentum=0.5,
    w_early=0.3,
    w_consistency=0.2,
    top_n=10,
    total_capital=100_000,
    weight_A=0.2,
    weight_B=0.8,
):
    """
    Combine Bucket A + B into a single executable target:
    RETURNS: Ticker | Position_Size
    """

    if as_of_date is None:
        as_of_date = max(dailyA["Date"].max(), dailyB["Date"].max())

    frames = []

    def select_bucket(daily_df, weight):
        if daily_df is None or daily_df.empty:
            return None

        window = daily_df[daily_df["Date"] <= as_of_date].tail(lookback_days)
        if window.empty:
            return None

        agg = (
            window.groupby("Ticker")
            .agg(
                momentum=("Momentum Score", "mean"),
                early=("Early Momentum Score", "mean"),
                appearances=("Date", "count"),
            )
            .reset_index()
        )

        agg["consistency"] = agg["appearances"] / lookback_days
        agg["score"] = (
            w_momentum * agg["momentum"]
            + w_early * agg["early"]
            + w_consistency * agg["consistency"]
        )

        sel = agg.sort_values("score", ascending=False).head(top_n)
        if sel.empty:
            return None

        dollars_per_name = (total_capital * weight) / len(sel)

        out = sel[["Ticker"]].copy()
        out["Position_Size"] = dollars_per_name
        return out

    partA = select_bucket(dailyA, weight_A)
    partB = select_bucket(dailyB, weight_B)

    if partA is not None:
        frames.append(partA)
    if partB is not None:
        frames.append(partB)

    if not frames:
        return pd.DataFrame(columns=["Ticker", "Position_Size"])

    combined = pd.concat(frames, ignore_index=True)

    # Consolidate overlaps (CRITICAL)
    return (
        combined
        .groupby("Ticker", as_index=False)["Position_Size"]
        .sum()
    )
