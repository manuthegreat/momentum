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
    Combine Bucket A + B daily rankings into a single capital-weighted portfolio
    """

    if as_of_date is None:
        as_of_date = max(dailyA["Date"].max(), dailyB["Date"].max())

    start = as_of_date - pd.Timedelta(days=lookback_days)

    A = dailyA[dailyA["Date"].between(start, as_of_date)].copy()
    B = dailyB[dailyB["Date"].between(start, as_of_date)].copy()

    def score(df):
        agg = (
            df.groupby("Ticker")
            .agg(
                momentum=("Momentum Score", "mean"),
                early=("Early Momentum Score", "mean"),
                consistency=("Ticker", "count"),
            )
            .reset_index()
        )

        agg["Score"] = (
            w_momentum * agg["momentum"]
            + w_early * agg["early"]
            + w_consistency * (agg["consistency"] / lookback_days)
        )

        return agg.sort_values("Score", ascending=False)

    A_rank = score(A).head(top_n)
    B_rank = score(B).head(top_n)

    A_rank["Bucket"] = "A"
    B_rank["Bucket"] = "B"

    A_rank["Capital"] = total_capital * weight_A / len(A_rank)
    B_rank["Capital"] = total_capital * weight_B / len(B_rank)

    final = pd.concat([A_rank, B_rank], ignore_index=True)
    final["AsOf"] = as_of_date

    return final

