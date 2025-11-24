# batch_sarima_forecast.py
import os
import re
import time
import json
import math
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

# ==============
# Global paths
# ==============
# << EDIT THIS ONE FOLDER ONLY >>
features_folder = r"C:\ThesisResearch\thesis_project\data\features_since1995"
results_folder = r'C:\ThesisResearch\thesis_project\results\SARIMA'
summary_path    = os.path.join(features_folder, "sp500_constituents_since_1995.csv")
os.makedirs(features_folder, exist_ok=True)

# =========================
# SARIMA model configuration
# =========================
# Quarterly data -> seasonal period s=4
ARIMA_ORDER           = (4, 1, 3)
SEASONAL_ARIMA        = (0, 1, 0, 4)
TRAIN_SPLIT_FRACTION  = 0.8
ENFORCE_STATIONARITY  = False
ENFORCE_INVERTIBILITY = False
RANDOM_SEED           = 42
MAXITER_1             = 200
MAXITER_2             = 300

# Silence verbose warnings you already understand (optional)
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", ValueWarning)
warnings.filterwarnings("ignore", message="No supported index is available.*")

# ======================
# Robust metric helpers
# ======================
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    """Robust MAPE (%) with scale-aware epsilon in the denominator."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if eps is None:
        scale = np.nanmedian(np.abs(y_true))
        eps = max(1e-8, 1e-3 * scale)  # 0.1% of median(|y|)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def mdape(y_true: np.ndarray, y_pred: np.ndarray, eps: float | None = None) -> float:
    """Median APE (%), robust to outliers; same epsilon rule as safe_mape."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if eps is None:
        scale = np.nanmedian(np.abs(y_true))
        eps = max(1e-8, 1e-3 * scale)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.median(np.abs((y_true - y_pred) / denom)) * 100.0)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE (%) in [0, 200], stable at zeros."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return float("nan")
    return float(100.0 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))

def accuracy_from_mape(mape_pct: float) -> float:
    """Accuracy (%) = 100 - MAPE (%), clipped to [0, 100]."""
    if math.isnan(mape_pct):
        return float("nan")
    return float(np.clip(100.0 - mape_pct, 0.0, 100.0))

def trimmed_mean(x: np.ndarray, p: float = 0.05) -> float:
    """p-fraction trimmed mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    x.sort()
    k = int(len(x) * p)
    if len(x) - 2 * k <= 0:
        return float("nan")
    return float(np.mean(x[k:len(x) - k]))

# ============================
# Sector extraction helpers
# ============================
SECTOR_CANDIDATE_COLS = [
    "sector", "Sector", "gics_sector", "GICS_Sector",
    "GICS", "gics", "gicsSector", "GICSSector", "gics_sectors"
]

def load_sector_map(path: str) -> Dict[str, str]:
    """Load ticker to sector mapping from CSV file."""
    if not os.path.exists(path):
        print(f"[info] Sector map not found at: {path}")
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker")
    sector_col = None
    for cand in SECTOR_CANDIDATE_COLS + ["sector"]:
        if cand in df.columns:
            sector_col = cand
            break
        if cand.lower() in cols:
            sector_col = cols[cand.lower()]
            break
    if ticker_col is None or sector_col is None:
        print(f"[warn] Could not find ticker/sector columns in {path}")
        return {}
    sector_map = {str(t): str(s) for t, s in zip(df[ticker_col], df[sector_col])}
    print(f"[info] Loaded sector map with {len(sector_map)} entries")
    return sector_map

def extract_sector_from_feature_csv(ticker: str) -> Optional[str]:
    """Extract sector information from feature CSV file."""
    fpath = os.path.join(features_folder, f"{ticker}_feature.csv")
    if not os.path.exists(fpath):
        return None
    try:
        fdf = pd.read_csv(fpath, nrows=5)
        for col in SECTOR_CANDIDATE_COLS:
            if col in fdf.columns:
                val = fdf[col].dropna()
                if len(val) > 0:
                    sector = str(val.iloc[0])
                    print(f"[info] Found sector '{sector}' for {ticker} from column '{col}'")
                    return sector
        # Try case-insensitive matching
        lower_map = {c.lower(): c for c in fdf.columns}
        for cand in SECTOR_CANDIDATE_COLS:
            if cand.lower() in lower_map:
                col = lower_map[cand.lower()]
                val = fdf[col].dropna()
                if len(val) > 0:
                    sector = str(val.iloc[0])
                    print(f"[info] Found sector '{sector}' for {ticker} from column '{col}' (case-insensitive)")
                    return sector
    except Exception as e:
        print(f"[warn] Error reading sector from {fpath}: {e}")
    return None

# ==========================
# Quarter index coercion
# ==========================
_quarter_rx = re.compile(r"^\s*(\d{4})\s*Q\s*([1-4])\s*$", re.IGNORECASE)

def to_quarter_end_index(qseries: pd.Series) -> pd.DatetimeIndex:
    """
    Convert a Series of strings like '2018Q1' / '2018 Q1' / '2018q1'
    to a quarterly PeriodIndex, then to Quarter-end timestamps.
    Falls back to as_datetime if parsing fails.
    """
    vals = qseries.astype(str).fillna("")
    years, qs = [], []
    ok = True
    for s in vals:
        m = _quarter_rx.match(s.replace("-", "").replace("_", ""))
        if m:
            years.append(int(m.group(1)))
            qs.append(int(m.group(2)))
        else:
            ok = False
            break
    if ok and len(years) == len(vals):
        periods = pd.PeriodIndex(year=years, quarter=qs, freq="Q-DEC")
        return periods.to_timestamp(how="end")  # quarter-end dates
    # Fallback: try pandas auto parse, then infer freq
    dt = pd.to_datetime(vals, errors="coerce")
    if dt.isna().all():
        # last resort: integer index
        return pd.date_range(start="2000-03-31", periods=len(vals), freq="Q")
    return pd.DatetimeIndex(dt)

# =====================
# Core per-ticker run
# =====================
def fit_sarima_and_score(revenue: pd.Series) -> Tuple[float, float, float, float, float, float]:
    """
    revenue: Series indexed by time (ascending, DateTimeIndex), numeric revenue per quarter.
    Returns: RMSE, MAE, NRMSE, MAPE(%), MdAPE(%), sMAPE(%)
    """
    # Log transform for modeling; guard for non-positives
    if (revenue <= 0).any():
        shift = 1.0 - float(revenue.min())
        log_series = np.log(revenue + shift)
        use_shift = shift
    else:
        log_series = np.log(revenue)
        use_shift = 0.0

    n = len(log_series)
    if n < 8:
        return (float("nan"),) * 6

    train_size = max(1, int(n * TRAIN_SPLIT_FRACTION))
    train, test = log_series.iloc[:train_size], log_series.iloc[train_size:]

    # First attempt: L-BFGS
    model = SARIMAX(
        train,
        order=ARIMA_ORDER,
        seasonal_order=SEASONAL_ARIMA,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY,
        initialization="approximate_diffuse",
    )
    try:
        fitted = model.fit(disp=False, method="lbfgs", maxiter=MAXITER_1, tol=1e-6)
    except Exception:
        # Retry with Powell if lbfgs fails
        fitted = model.fit(disp=False, method="powell", maxiter=MAXITER_2)

    # Forecast on the test horizon
    forecast_log = fitted.forecast(steps=len(test))
    if use_shift != 0.0:
        forecast = np.exp(forecast_log) - use_shift
        actual = np.exp(test) - use_shift
    else:
        forecast = np.exp(forecast_log)
        actual = np.exp(test)

    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(forecast, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    nrmse = float(rmse / max(np.nanmedian(np.abs(y_true)), 1e-8))
    mape_pct  = float(safe_mape(y_true, y_pred))
    mdape_pct = float(mdape(y_true, y_pred))
    smape_pct = float(smape(y_true, y_pred))
    return rmse, mae, nrmse, mape_pct, mdape_pct, smape_pct

def summarize_metrics(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Generate overall summary statistics for all metrics."""
    rows = []
    for m in metric_cols:
        col = pd.to_numeric(df[m], errors="coerce")
        valid_count = col.count()
        rows.append({
            "Metric": m,
            "count": int(valid_count),
            "mean": float(col.mean()) if valid_count > 0 else float("nan"),
            "median": float(col.median()) if valid_count > 0 else float("nan"),
            "std": float(col.std(ddof=1)) if valid_count > 1 else float("nan"),
            "min": float(col.min()) if valid_count > 0 else float("nan"),
            "max": float(col.max()) if valid_count > 0 else float("nan"),
            "trimmed_mean_5%": trimmed_mean(col.to_numpy(), p=0.05),
        })
    return pd.DataFrame(rows)

def summarize_by_sector(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Generate sector-wise summary statistics for all metrics."""
    rows = []
    sector_groups = df.groupby("Sector", dropna=False)
    
    print(f"\n[info] Generating sector stats for {len(sector_groups)} sectors")
    
    for sector, g in sector_groups:
        sector_name = sector if pd.notna(sector) else "Unknown"
        print(f"[info] Processing sector: {sector_name} ({len(g)} companies)")
        
        for m in metric_cols:
            col = pd.to_numeric(g[m], errors="coerce")
            valid_count = col.count()
            rows.append({
                "Sector": sector_name,
                "Metric": m,
                "count": int(valid_count),
                "mean": float(col.mean()) if valid_count > 0 else float("nan"),
                "median": float(col.median()) if valid_count > 0 else float("nan"),
                "std": float(col.std(ddof=1)) if valid_count > 1 else float("nan"),
                "min": float(col.min()) if valid_count > 0 else float("nan"),
                "max": float(col.max()) if valid_count > 0 else float("nan"),
                "trimmed_mean_5%": trimmed_mean(col.to_numpy(), p=0.05),
            })
    return pd.DataFrame(rows)

def main():
    np.random.seed(RANDOM_SEED)
    ts = time.strftime("%Y%m%d-%H%M%S")

    # Load tickers from sp500_constituents_since_1995.csv
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"sp500_constituents_since_1995.csv not found at: {summary_path}")
    constituents_df = pd.read_csv(summary_path)
    if "Ticker" not in constituents_df.columns:
        candidates = [c for c in constituents_df.columns if c.lower() == "ticker"]
        if not candidates:
            raise ValueError("Cannot find 'Ticker' column in sp500_constituents_since_1995.csv")
        constituents_df.rename(columns={candidates[0]: "Ticker"}, inplace=True)
    tickers = sorted(set(str(t).strip() for t in constituents_df["Ticker"].dropna().unique()))

    print(f"[info] Found {len(tickers)} tickers to process")

    # Load fallback sector map
    per_ticker_rows = []
    processed_count = 0
    
    for ticker in tickers:
        processed_csv = os.path.join(features_folder, f"{ticker}_feature.csv")
        if not os.path.exists(processed_csv):
            print(f"[skip] Processed file missing for {ticker}: {processed_csv}")
            continue

        try:
            df = pd.read_csv(processed_csv)
            if "revenue" not in df.columns:
                print(f"[warn] Missing 'revenue' in {processed_csv}, skipping {ticker}")
                continue

            # Build a proper quarterly DateTimeIndex to avoid index warnings
            if "Year_Quarter" in df.columns:
                idx = to_quarter_end_index(df["Year_Quarter"])
            else:
                # Fallback to a simple quarterly range
                idx = pd.date_range(start="2000-03-31", periods=len(df), freq="Q")

            # Sort by index and align revenue
            rev = pd.to_numeric(df["revenue"], errors="coerce")
            s = pd.Series(rev.values, index=idx).sort_index()
            s = s.dropna()
            if s.empty:
                print(f"[warn] Empty/NaN revenue for {ticker}, skipping")
                continue

            print(f"[info] Fitting SARIMA for {ticker} ({len(s)} quarters)")
            rmse, mae, nrmse, mape_pct, mdape_pct, smape_pct = fit_sarima_and_score(s)

            # Sector detection: feature csv first, else fallback
            sector = extract_sector_from_feature_csv(ticker)

            acc_pct = accuracy_from_mape(mape_pct)

            per_ticker_rows.append({
                "Ticker": ticker,
                "Sector": sector if sector is not None else "Unknown",
                "RMSE": rmse,
                "MAE": mae,
                "NRMSE": nrmse,
                "MAPE (%)": mape_pct,
                "MdAPE (%)": mdape_pct,
                "sMAPE (%)": smape_pct,
                "Accuracy (%)": acc_pct,
            })
            processed_count += 1
            print(f"[success] Processed {ticker}: MAPE={mape_pct:.2f}%, Accuracy={acc_pct:.2f}%")
            
        except Exception as e:
            print(f"[error] {ticker}: {e}")
            continue

    if not per_ticker_rows:
        print("No results were generated. Check inputs.")
        return

    results_df = pd.DataFrame(per_ticker_rows)
    print(f"\n[success] Processed {processed_count}/{len(tickers)} tickers successfully")

    # Outlier flags
    try:
        outlier_mask = (results_df['MAPE (%)'] > 500) | (results_df['sMAPE (%)'] > 100)
        if outlier_mask.any():
            flagged_outliers = results_df.loc[outlier_mask].copy()
            flagged_path = os.path.join(features_folder, f"flagged_outliers_{ts}.csv")
            flagged_outliers.to_csv(flagged_path, index=False)
            print(f"⚠️  Outliers flagged to: {flagged_path}")
    except Exception as e:
        print(f"[warn] Could not write flagged outliers: {e}")

    # Summaries
    metric_cols = ["RMSE", "MAE", "NRMSE", "MAPE (%)", "MdAPE (%)", "sMAPE (%)", "Accuracy (%)"]
    
    print("\n[info] Generating overall statistics...")
    overall_df = summarize_metrics(results_df, metric_cols)
    
    print("[info] Generating sector-wise statistics...")
    sector_df  = summarize_by_sector(results_df, metric_cols)

    # Save results
    per_ticker_path  = os.path.join(results_folder, f"per_ticker_results_{ts}.csv")
    overall_csv_path = os.path.join(results_folder, f"overall_stats_{ts}.csv")
    overall_json_path= os.path.join(results_folder, f"overall_stats_{ts}.json")
    sector_csv_path  = os.path.join(results_folder, f"sector_stats_{ts}.csv")

    results_df.to_csv(per_ticker_path, index=False)
    overall_df.to_csv(overall_csv_path, index=False)
    sector_df.to_csv(sector_csv_path, index=False)
    
    # Save JSON with proper formatting
    with open(overall_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved:")
    print(f"✅ Per-ticker results: {per_ticker_path}")
    print(f"✅ Overall stats CSV:  {overall_csv_path}")
    print(f"✅ Overall stats JSON: {overall_json_path}")
    print(f"✅ Sector stats CSV:   {sector_csv_path}")
    
    # Print sector distribution
    sector_counts = results_df["Sector"].value_counts()
    print(f"\n📊 Sector distribution:")
    for sector, count in sector_counts.items():
        print(f"   {sector}: {count} companies")

if __name__ == "__main__":
    main()