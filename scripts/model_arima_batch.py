import os
import time
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
data_folder = r'C:\ThesisResearch\thesis_project\data\features_since1995'
results_folder = r'C:\ThesisResearch\thesis_project\results\ARIMA'

# -----------------------------
# Helpers
# -----------------------------
def get_sector_from_feature_df(df: pd.DataFrame) -> str:
    """
    Get sector directly from the per-ticker *_feature.csv.
    Priority order:
      1) 'gics_sectors'
      2) 'gics_sector'
      3) 'sector' / 'Sector'
      4) 'gics' / 'GICS'
      5) 'sector_name' / 'SectorName'
    Returns 'Unknown' if none found or all-NA.
    """
    if df is None or df.empty:
        return "Unknown"
    cols_lower = {c.lower(): c for c in df.columns}
    candidates = [
        "gics_sectors", "gics_sector", "sector", "gics",
        "sector_name", "sectorname"
    ]
    for key in candidates:
        if key in cols_lower:
            col = cols_lower[key]
            val = df[col].dropna()
            if len(val):
                return str(val.iloc[-1])
    return "Unknown"

def smape(a, f):
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    denom = (np.abs(a) + np.abs(f))
    mask = denom > 0
    if not np.any(mask):
        return np.nan
    return 100.0 * np.mean(2.0 * np.abs(f[mask] - a[mask]) / denom[mask])

def _compute_metrics(actual, forecast):
    """Return RMSE, MAE, MAPE, Accuracy, sMAPE (all floats)."""
    rmse = float(np.sqrt(mean_squared_error(actual, forecast)))
    mae  = float(mean_absolute_error(actual, forecast))
    actual_np = np.asarray(actual, dtype=float)
    forecast_np = np.asarray(forecast, dtype=float)
    mape_mask = actual_np != 0
    if np.any(mape_mask):
        mape = float(np.mean(np.abs((actual_np[mape_mask] - forecast_np[mape_mask]) / actual_np[mape_mask])) * 100.0)
    else:
        mape = np.nan
    acc = float(100.0 - mape) if pd.notna(mape) else np.nan
    s_map = float(smape(actual_np, forecast_np))
    return rmse, mae, mape, acc, s_map

# -----------------------------
# Discover tickers from *_feature.csv
# -----------------------------
feature_files = [f for f in os.listdir(data_folder) if f.endswith('_feature.csv')]
tickers = sorted([f[:-len('_feature.csv')] for f in feature_files])
if not tickers:
    raise FileNotFoundError(
        f"No *_feature.csv files found under: {data_folder}. "
        "Expected files like AAPL_feature.csv."
    )
print(f"Discovered {len(tickers)} tickers from feature files.")

# Fixed time windows
TRAIN_START, TRAIN_END = '1995Q1', '2016Q1'
VAL_START,   VAL_END   = '2016Q2', '2020Q3'
TEST_START,  TEST_END  = '2020Q4', '2025Q2'

def _slice_by_yq_index(series: pd.Series, start: str, end: str) -> pd.Series:
    """Inclusive slice on Year_Quarter string index (format YYYYQ{1-4})."""
    idx = series.index.astype(str)
    mask = (idx >= start) & (idx <= end)
    return series[mask]

# Collect results
results = []

for ticker in tickers:
    csv_path = os.path.join(data_folder, f"{ticker}_feature.csv")
    if not os.path.exists(csv_path):
        print(f"File not found for {ticker}, skipping...")
        continue

    try:
        # Load company data
        df = pd.read_csv(csv_path)
        if 'Year_Quarter' not in df.columns:
            raise ValueError(f"'Year_Quarter' column missing in {csv_path}")
        df = df.sort_values('Year_Quarter').reset_index(drop=True)

        # Target series
        if 'revenue' not in df.columns:
            raise ValueError(f"'revenue' column missing in {csv_path}")
        revenue_series = pd.Series(df['revenue'].astype(float).values,
                                   index=df['Year_Quarter'].astype(str))

        # Sector from feature file
        sector = get_sector_from_feature_df(df)

        # Stationarity diagnostics (on whole available series)
        dropna_rev = revenue_series.dropna()
        if len(dropna_rev) < 8:
            print(f"Insufficient data overall for {ticker}, skipping...")
            continue
        adf_stat, adf_p, adf_lags, adf_obs, adf_crit, _ = adfuller(
            dropna_rev, autolag='AIC'
        )
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
            dropna_rev, regression='c', nlags='auto'
        )

        # Preprocess: make strictly positive -> log
        safe_rev = revenue_series.copy()
        safe_rev[safe_rev <= 0] = np.nan
        safe_rev = safe_rev.fillna(method='ffill')
        log_series = np.log(safe_rev)

        # --- Explicit splits (inclusive ranges) ---
        train_log = _slice_by_yq_index(log_series, TRAIN_START, TRAIN_END)
        val_log   = _slice_by_yq_index(log_series, VAL_START,   VAL_END)
        test_log  = _slice_by_yq_index(log_series, TEST_START,  TEST_END)

        # Basic length checks
        if len(train_log) < 5 or len(val_log) < 1 or len(test_log) < 1:
            print(f"[{ticker}] Not enough data in one of the splits "
                  f"(train={len(train_log)}, val={len(val_log)}, test={len(test_log)}). Skipping.")
            continue

        ORDER = (1, 1, 1)

        # ====== Stage 1: fit on TRAIN, validate on VAL ======
        model_train = ARIMA(train_log, order= ORDER)
        fitted_train = model_train.fit()
        val_forecast_log = fitted_train.forecast(steps=len(val_log))

        # Back-transform
        val_forecast = np.exp(val_forecast_log)
        val_actual   = np.exp(val_log)

        val_rmse, val_mae, val_mape, val_acc, val_smap = _compute_metrics(val_actual, val_forecast)

        # ====== Stage 2: refit on TRAIN+VAL, test on TEST ======
        trainval_log = pd.concat([train_log, val_log])
        model_trainval = ARIMA(trainval_log, order= ORDER)
        fitted_trainval = model_trainval.fit()
        test_forecast_log = fitted_trainval.forecast(steps=len(test_log))

        test_forecast = np.exp(test_forecast_log)
        test_actual   = np.exp(test_log)

        rmse, mae, mape, acc, s_map = _compute_metrics(test_actual, test_forecast)

        # Store per-ticker results (Validation + Test)
        results.append({
            'Ticker': ticker,
            'Sector': sector,

            # Diagnostics (whole series)
            'ADF Statistic': adf_stat,
            'ADF p-value': adf_p,
            'ADF Lags Used': adf_lags,
            'ADF Observations Used': adf_obs,
            'ADF CV 1%': adf_crit['1%'],
            'ADF CV 5%': adf_crit['5%'],
            'ADF CV 10%': adf_crit['10%'],
            'KPSS Statistic': kpss_stat,
            'KPSS p-value': kpss_p,
            'KPSS Lags Used': kpss_lags,
            'KPSS CV 10%': kpss_crit['10%'],
            'KPSS CV 5%': kpss_crit['5%'],
            'KPSS CV 2.5%': kpss_crit['2.5%'],
            'KPSS CV 1%': kpss_crit['1%'],

            # Validation metrics (Train -> Validate)
            'Val RMSE': val_rmse,
            'Val MAE': val_mae,
            'Val MAPE (%)': None if pd.isna(val_mape) else round(val_mape, 4),
            'Val sMAPE (%)': None if pd.isna(val_smap) else round(val_smap, 4),
            'Val Accuracy (%)': None if pd.isna(val_acc) else round(val_acc, 4),

            # TEST metrics (Train+Val -> Test)  [keeps your original column names]
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': None if pd.isna(mape) else round(mape, 4),
            'sMAPE (%)': None if pd.isna(s_map) else round(s_map, 4),
            'Accuracy (%)': None if pd.isna(acc) else round(acc, 4),

            # Sizes (useful for debugging)
            'Train N': int(len(train_log)),
            'Val N': int(len(val_log)),
            'Test N': int(len(test_log)),
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

# -----------------------------
# Save all results safely
# -----------------------------
os.makedirs(results_folder, exist_ok=True)
timestamp = int(time.time())
output_file = os.path.join(results_folder, f'results_of_sp500_{timestamp}.csv')

overall_stats_file_json = os.path.join(results_folder, f'overall_stats_{timestamp}.json')
overall_stats_file_csv  = os.path.join(results_folder, f'overall_stats_{timestamp}.csv')
sector_stats_file_csv   = os.path.join(results_folder, f'sector_stats_{timestamp}.csv')

try:
    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise RuntimeError("No successful ARIMA fits; result table is empty.")

    # (1) Save per-ticker results
    result_df.to_csv(output_file, index=False)
    print(f"\n ✅ All results saved to: {output_file}")

    # (2) Overall statistics across companies (TEST metrics)
    metric_cols = ['RMSE', 'MAE', 'MAPE (%)', 'sMAPE (%)', 'Accuracy (%)']
    overall_stats = {}
    for col in metric_cols:
        s = pd.to_numeric(result_df[col], errors='coerce')
        overall_stats[col] = {
            'count': int(s.count()),
            'mean': float(s.mean()),
            'median': float(s.median()),
            'std': float(s.std(ddof=1)),
            'min': float(s.min()),
            'max': float(s.max()),
        }
    with open(overall_stats_file_json, "w") as f:
        json.dump(overall_stats, f, indent=2)
    pd.DataFrame(overall_stats).T.to_csv(overall_stats_file_csv)
    print(f" ✅ Overall stats saved to: {overall_stats_file_json} and {overall_stats_file_csv}")

    # (3) Sector-wise statistics (TEST metrics)
    sector_stats = (
        result_df
        .groupby('Sector')[metric_cols]
        .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    )
    sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns.values]
    sector_stats = sector_stats.reset_index()
    sector_stats.to_csv(sector_stats_file_csv, index=False)
    print(f" ✅ Sector-wise stats saved to: {sector_stats_file_csv}")

except PermissionError:
    print(f"\n❌ Permission denied: Please close '{output_file}' if it's open and rerun the script.")
except Exception as e:
    print(f"\n❌ Failed to save outputs: {e}")
