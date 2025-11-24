import pandas as pd
import requests
import os
from datetime import datetime
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
# Load tickers from a local file
with open("sp500_constituents_since_1995.csv", "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

API_KEY = os.getenv("FMP_API_KEY")  # Read API from environment variable
if not API_KEY:
    raise RuntimeError("FMP_API_KEY environment variable is not set.")

BASE_V3 = "https://financialmodelingprep.com/api/v3"
START_DATE = "1995-01-02"
END_DATE = "2025-10-10"
OUTPUT_FOLDER = "C:/ThesisResearch/thesis_project/data/raw/sp500_quarterly_1995_2025"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "sp500-enrichment-script/1.1"})

# -----------------------------
# Helpers
# -----------------------------
def _json_get(url: str):
    """GET json from URL with simple error-handling."""
    try:
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"HTTP {r.status_code} for {url}")
            return None
    except Exception as e:
        print(f"Error GET {url}: {e}")
        return None

def _within_window(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    mask = (df[date_col] >= pd.to_datetime(START_DATE)) & (df[date_col] <= pd.to_datetime(END_DATE))
    return df.loc[mask].sort_values(date_col).reset_index(drop=True)

# -----------------------------
# Fetchers
# -----------------------------
def fetch_income_data(ticker: str) -> pd.DataFrame:
    url = f"{BASE_V3}/income-statement/{ticker}?period=quarter&limit=1000&apikey={API_KEY}"
    js = _json_get(url)
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    if df.empty:
        return df
    df = _within_window(df, "date")
    return df

def fetch_balance_sheet_data(ticker: str) -> pd.DataFrame:
    """Fetch quarterly balance sheet; return date, totalAssets, totalEquity."""
    url = f"{BASE_V3}/balance-sheet-statement/{ticker}?period=quarter&limit=1000&apikey={API_KEY}"
    js = _json_get(url)
    if not js:
        return pd.DataFrame()

    df = pd.DataFrame(js)
    if df.empty:
        return df

    df = _within_window(df, "date")

    # Normalize equity field name (FMP may use totalStockholdersEquity)
    equity_col = None
    for cand in ["totalStockholdersEquity", "totalEquity"]:
        if cand in df.columns:
            equity_col = cand
            break

    # Keep only relevant columns when present
    keep = ["date"]
    if "totalAssets" in df.columns:
        keep.append("totalAssets")
    if equity_col is not None:
        keep.append(equity_col)

    df = df[keep].copy()

    if equity_col and equity_col != "totalEquity":
        df = df.rename(columns={equity_col: "totalEquity"})
    elif "totalEquity" not in df.columns:
        df["totalEquity"] = pd.NA  # if neither key existed

    if "totalAssets" not in df.columns:
        df["totalAssets"] = pd.NA

    return df

def fetch_profile(ticker: str) -> dict:
    """Return {sector, industry, exchange, ipoDate} using /profile."""
    url = f"{BASE_V3}/profile/{ticker}?apikey={API_KEY}"
    js = _json_get(url)
    if js and isinstance(js, list) and len(js):
        row = js[0]
        return {
            'sector': row.get('sector'),
            'industry': row.get('industry'),
            'exchangeShortName': row.get('exchangeShortName'),
            'ipoDate': row.get('ipoDate'),
        }
    return {'sector': None, 'industry': None, 'exchangeShortName': None, 'ipoDate': None}

# -----------------------------
# Merge helpers
# -----------------------------
def merge_income_balance_on_date(inc: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """
    Try strict merge on exact 'date'. If some income rows don't find a match,
    attempt an asof merge within a 7-day tolerance.
    """
    if bs.empty:
        inc["totalAssets"] = pd.NA
        inc["totalEquity"] = pd.NA
        return inc

    # First try exact date join
    merged = inc.merge(bs, on="date", how="left", suffixes=("", "_bs"))

    # If there are any missing assets/equity, try an asof merge for those rows
    missing_mask = merged["totalAssets"].isna() & merged["totalEquity"].isna()
    if missing_mask.any():
        inc_missing = merged.loc[missing_mask, ["date"]].copy()
        # Prepare asof frames
        inc_asof = inc_missing.rename(columns={"date": "date_inc"}).sort_values("date_inc")
        bs_asof = bs.sort_values("date")
        asof = pd.merge_asof(
            inc_asof, bs_asof, left_on="date_inc", right_on="date",
            direction="nearest", tolerance=pd.Timedelta(days=7)
        )
        # Stitch back
        merged.loc[missing_mask, "totalAssets"] = asof["totalAssets"].values
        merged.loc[missing_mask, "totalEquity"] = asof["totalEquity"].values

    return merged

# -----------------------------
# Main
# -----------------------------
summary = []
all_rows = []

for ticker in tqdm(tickers, desc="Fetching & enriching income statements with balance sheet"):
    inc = fetch_income_data(ticker)
    if inc.empty:
        summary.append((ticker, 0, None, None))
        continue

    bs = fetch_balance_sheet_data(ticker)

    # Add profile fields
    profile = fetch_profile(ticker)

    # Merge balance sheet fields onto income rows (by date, with fallback)
    enriched = merge_income_balance_on_date(inc, bs)

    # Attach profile
    enriched["sector"] = profile.get("sector")
    enriched["industry"] = profile.get("industry")
    enriched["exchangeShortName"] = profile.get("exchangeShortName")
    enriched["ipoDate"] = profile.get("ipoDate")

    # Persist per-ticker
    out_path = f"{OUTPUT_FOLDER}/{ticker}.csv"
    enriched.to_csv(out_path, index=False)

    summary.append((ticker, len(enriched), profile.get('sector'), profile.get('industry')))
    all_rows.append(enriched.assign(ticker=ticker))

# Save summary
summary_df = pd.DataFrame(summary, columns=["Ticker", "Records", "Sector", "Industry"])
summary_df.to_csv(f"{OUTPUT_FOLDER}/summary.csv", index=False)

# Save a combined master (optional but handy for modeling)
if all_rows:
    combined = pd.concat(all_rows, ignore_index=True)
    cols = combined.columns.tolist()
    if 'ticker' in cols:
        cols = ['ticker'] + [c for c in cols if c != 'ticker']
        combined = combined[cols]
    combined.to_csv(f"{OUTPUT_FOLDER}/sp500_income_enriched_master.csv", index=False)

print("✅ All done. Enriched data saved (now with totalAssets & totalEquity).")
print(f"✅ Output folder: {OUTPUT_FOLDER}")
