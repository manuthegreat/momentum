import pandas as pd
import requests
from io import StringIO
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================
# 1. UNIVERSE BUILDERS
# ============================================================

def get_sp500_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(r.text))

    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break

    df["Ticker"] = df["Symbol"].str.replace(".", "-", regex=False)
    df["Name"] = df["Security"]
    df["Sector"] = df["GICS Sector"]

    return df[["Ticker", "Name", "Sector"]]


def get_hsi_universe():
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(x in cols for x in ["ticker", "constituent", "sub-index", "code"]):
            df = t.copy()
            break

    if df is None:
        raise ValueError("No HSI table found")

    df.columns = [str(c).lower() for c in df.columns]

    ticker_col = next(
        (c for c in df.columns if "ticker" in c or "code" in c or "sehk" in c),
        None
    )
    if ticker_col is None:
        raise ValueError("No HSI ticker column")

    df["Ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)")
        .iloc[:, 0]
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )

    name_col = "name" if "name" in df.columns else df.columns[0]
    df["Name"] = df[name_col]

    if "sub-index" in df.columns:
        df["Sector"] = df["sub-index"]
    elif "industry" in df.columns:
        df["Sector"] = df["industry"]
    else:
        df["Sector"] = None

    return df[["Ticker", "Name", "Sector"]]


def get_sti_universe():
    data = [
        ("D05.SI","DBS Group Holdings","Financials"),
        ("U11.SI","United Overseas Bank","Financials"),
        ("O39.SI","OCBC","Financials"),
        ("C07.SI","Jardine Matheson","Conglomerate"),
        ("C09.SI","City Developments","Real Estate"),
        ("C38U.SI","CICT","Real Estate"),
        ("Z74.SI","Singtel","Telecom"),
    ]
    return pd.DataFrame(data, columns=["Ticker", "Name", "Sector"])


# ============================================================
# 2. DOWNLOAD CONSTITUENT OHLC (5Y)
# ============================================================

def download_5yr_ohlc(tickers, label):
    print(f"\nDownloading {label} ({len(tickers)} tickers)")
    frames = []
    batch_size = 40

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        data = yf.download(
            batch,
            period="5y",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False
        )

        for t in batch:
            try:
                df = data[t].dropna()
                if df.empty:
                    continue
                df = df.reset_index()
                df["Ticker"] = t
                df["Index"] = label
                frames.append(df)
            except Exception:
                continue

    return frames


# ============================================================
# 3. DOWNLOAD INDEX RETURNS (5Y)
# ============================================================

def download_index_5y(ticker, label):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.dropna().reset_index()

    df["index_name"] = label
    df["ticker"] = ticker

    df["ret_1d"]  = df["close"].pct_change()
    df["ret_5d"]  = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)
    df["ret_60d"] = df["close"].pct_change(60)

    return df


# ============================================================
# 4. MAIN
# ============================================================

def main():

    # ---------- Build universes ----------
    sp500 = get_sp500_universe()
    hsi   = get_hsi_universe()
    sti   = get_sti_universe()

    # ---------- Download constituents ----------
    frames = []
    frames += download_5yr_ohlc(sp500["Ticker"].tolist(), "SP500")
    frames += download_5yr_ohlc(hsi["Ticker"].tolist(), "HSI")
    frames += download_5yr_ohlc(sti["Ticker"].tolist(), "STI")

    full_constituents = pd.concat(frames, ignore_index=True)
    full_constituents.to_parquet(
        "artifacts/index_constituents_5yr.parquet",
        index=False
    )

    print("Saved index_constituents_5yr.parquet")

    # ---------- Download index returns ----------
    index_map = {
        "^GSPC": "SP500",
        "^HSI":  "HSI",
        "^STI":  "STI",
        "^VIX":  "VIX",
    }

    idx_frames = []
    for t, lbl in index_map.items():
        df = download_index_5y(t, lbl)
        if df is not None:
            idx_frames.append(df)

    full_index = pd.concat(idx_frames, ignore_index=True)
    full_index.to_parquet(
        "artifacts/index_returns_5y.parquet",
        index=False
    )

    print("Saved index_returns_5y.parquet")


if __name__ == "__main__":
    main()
