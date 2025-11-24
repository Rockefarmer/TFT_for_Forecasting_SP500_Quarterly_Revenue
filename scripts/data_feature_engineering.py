import os
import glob
import numpy as np
import pandas as pd

# === Paths (edit to your environment) ===
processed_dir = r"C:\ThesisResearch\thesis_project\data\preprocessed_since1995"
features_dir  = r"C:\ThesisResearch\thesis_project\data\features_since1995"
os.makedirs(features_dir, exist_ok=True)

_QMAP = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer / denom.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if {"rnd","revenue"}.issubset(df.columns):
        df["rnd_to_rev_ratio"] = _safe_div(df["rnd"], df["revenue"])
    if {"snaExpenses","revenue"}.issubset(df.columns):
        df["sna_to_rev_ratio"] = _safe_div(df["snaExpenses"], df["revenue"])
    if {"grossProfit","revenue"}.issubset(df.columns):
        df["grossProfitRatio"] = _safe_div(df["grossProfit"], df["revenue"])
    if {"operatingIncome","revenue"}.issubset(df.columns):
        df["operatingIncomeRatio"] = _safe_div(df["operatingIncome"], df["revenue"])
    if {"netIncome","revenue"}.issubset(df.columns):
        df["netIncomeRatio"] = _safe_div(df["netIncome"], df["revenue"])
    return df

def add_lags(df: pd.DataFrame, col: str, max_lag: int = 12) -> pd.DataFrame:
    if col not in df.columns: 
        return df
    for k in range(1, max_lag+1):
        df[f"{col}_lag{k}"] = df[col].shift(k)
    return df

def add_single_lag(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convenience: add exactly one lag column 'col_lag1' if base column exists."""
    if col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)
    return df

def avg_qoq_growth(s: pd.Series, n: int = 4) -> float:
    """Average quarter-over-quarter growth over the last n quarters from history.
    Returns 0.0 if not enough data or all-NaN."""
    if s is None or s.empty:
        return 0.0
    growths = s.astype(float).pct_change().dropna().tail(n)
    if growths.empty:
        return 0.0
    return float(growths.mean())

def add_yoy_growth(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[f"{c}_yoy"] = _safe_div(df[c], df[c].shift(4)) - 1.0
    return df

def one_hot_quarter(df: pd.DataFrame) -> pd.DataFrame:
    if "quarter" not in df.columns:
        raise ValueError("Expected 'quarter' column with values like 'Q1'..'Q4'.")
    q_int = df["quarter"].map(_QMAP)
    for i in [1,2,3,4]:
        df[f"Q{i}"] = (q_int == i).astype(int)
    return df

def _next_year_quarter(y: int, q_int: int) -> tuple[int, str, int]:
    qn = q_int + 1
    yn = y
    if qn == 5:
        qn = 1
        yn = y + 1
    return yn, f"Q{qn}", qn

def _ensure_year_quarter_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add Year_Quarter like '2024Q1' if missing."""
    if "Year_Quarter" not in df.columns:
        df["Year_Quarter"] = df["year"].astype(str) + df["quarter"].astype(str)
    return df

def _roll_forward_lags(last_base: float, last_lag_series: dict[int, float], horizon: int, max_k: int) -> dict[int, float]:
    """
    Produce a dictionary {k: value} for lag k at (t + horizon) given:
    - last_base = base at time t
    - last_lag_series = {1: base_{t-1}, 2: base_{t-2}, ...}
    - For horizon h: lag_k(t+h) = lag_{k-h+1}(t) for k>=h, and lag_h(t+h) = base_t.
      Values whose source index <=0 are NaN.
    """
    out = {}
    for k in range(1, max_k+1):
        src = k - (horizon - 1)  # where to read from time t
        if src == 1:  # becomes last_base when src==1 after shifting once
            # careful: when horizon==1 and k==1 => src=1 -> we want base_t in lag1 at t+1? No:
            # we handle below with if k==horizon then base -> correct TFT horizon behavior.
            pass
        if k == horizon:
            out[k] = last_base
        elif src > 1:
            out[k] = last_lag_series.get(src - 1, np.nan)  # because lag_series is indexed from 1
        else:
            out[k] = np.nan
    return out

def append_future_quarters(df: pd.DataFrame, n_future: int = 4) -> pd.DataFrame:
    """
    Append n_future rows with future Year/Quarter and fill:
      - seasonal one-hot (Q1..Q4)
      - 'rnd_lag*' and 'rnd_to_rev_ratio_lag*' rolled forward for each horizon
      - 'totalAssets_lag1' & 'totalEquity_lag1':
          * t+1  = last known actual (lag1)
          * t+2..t+4 = prior future value * (1 + avg_qoq_growth over last 4 hist quarters)
      - leave unknown targets (revenue, netIncome, rnd, ratios...) as NaN
    """
    if df.empty:
        return df

    # Identify max lag depth already present for rnd, ratio
    rnd_lag_cols = sorted([c for c in df.columns if c.startswith("rnd_lag") and c[len("rnd_lag"):].isdigit()],
                          key=lambda c: int(c.split("rnd_lag")[1]))
    rratio_lag_cols = sorted([c for c in df.columns if c.startswith("rnd_to_rev_ratio_lag") and c.split("_lag")[-1].isdigit()],
                             key=lambda c: int(c.split("_lag")[1]))

    max_rnd_k = int(rnd_lag_cols[-1].split("rnd_lag")[1]) if rnd_lag_cols else 0
    max_rr_k  = int(rratio_lag_cols[-1].split("_lag")[1]) if rratio_lag_cols else 0

    # Last historical timepoint and seasonal
    last_hist = df.iloc[-1].copy()
    last_year = int(last_hist["year"])
    last_qint = int(_QMAP[last_hist["quarter"]])

    # Last known R&D base and ratio (for rolling lag logic)
    last_rnd = last_hist["rnd"] if "rnd" in df.columns else np.nan
    last_rr  = last_hist["rnd_to_rev_ratio"] if "rnd_to_rev_ratio" in df.columns else np.nan
    last_rnd_lags = {int(c.split("rnd_lag")[1]): last_hist[c] for c in rnd_lag_cols}
    last_rr_lags  = {int(c.split("_lag")[1]):  last_hist[c] for c in rratio_lag_cols}

    # Assets/Equity last actuals + average QoQ growth over past 4 hist quarters
    ta_hist = df["totalAssets"] if "totalAssets" in df.columns else None
    te_hist = df["totalEquity"] if "totalEquity" in df.columns else None

    ta_last = float(ta_hist.iloc[-1]) if ta_hist is not None and not pd.isna(ta_hist.iloc[-1]) else np.nan
    te_last = float(te_hist.iloc[-1]) if te_hist is not None and not pd.isna(te_hist.iloc[-1]) else np.nan

    ta_avg_g = avg_qoq_growth(ta_hist) if ta_hist is not None else 0.0
    te_avg_g = avg_qoq_growth(te_hist) if te_hist is not None else 0.0

    # We’ll carry forward future values of the *lag1* columns using the avg growth
    ta_future = None  # will hold the last projected 'lag1' value to multiply forward
    te_future = None

    frames = [df]

    for h in range(1, n_future + 1):
        last_year, qstr, qint = _next_year_quarter(last_year, last_qint)
        last_qint = qint

        # start as all-NaN to avoid leakage by default
        row = {k: np.nan for k in df.columns}
        row["year"] = last_year
        row["quarter"] = qstr

        tmp = pd.DataFrame([row])

        # seasonals + Year_Quarter key
        tmp = one_hot_quarter(tmp)
        tmp = _ensure_year_quarter_key(tmp)

        # roll forward R&D lags (never invent unknown base values)
        if max_rnd_k > 0:
            rolled = _roll_forward_lags(last_rnd, last_rnd_lags, h, max_rnd_k)
            for k in range(1, max_rnd_k + 1):
                col = f"rnd_lag{k}"
                tmp[col] = rolled[k]

        if max_rr_k > 0:
            rolled_rr = _roll_forward_lags(last_rr, last_rr_lags, h, max_rr_k)
            for k in range(1, max_rr_k + 1):
                col = f"rnd_to_rev_ratio_lag{k}"
                tmp[col] = rolled_rr[k]

        # --- totalAssets_lag1 / totalEquity_lag1 rule you requested ---
        if "totalAssets_lag1" in df.columns:
            if h == 1:
                ta_future = ta_last  # set base for rolling
                tmp["totalAssets_lag1"] = ta_future
            else:
                if pd.notna(ta_future) and ta_avg_g is not None:
                    ta_future = ta_future * (1.0 + float(ta_avg_g))
                    tmp["totalAssets_lag1"] = ta_future
                else:
                    tmp["totalAssets_lag1"] = np.nan

        if "totalEquity_lag1" in df.columns:
            if h == 1:
                te_future = te_last
                tmp["totalEquity_lag1"] = te_future
            else:
                if pd.notna(te_future) and te_avg_g is not None:
                    te_future = te_future * (1.0 + float(te_avg_g))
                    tmp["totalEquity_lag1"] = te_future
                else:
                    tmp["totalEquity_lag1"] = np.nan

        # keep ticker if present
        if "ticker" in df.columns:
            tmp["ticker"] = df["ticker"].iloc[-1]

        frames.append(tmp[df.columns])  # align to original column order

    out = pd.concat(frames, ignore_index=True)
    return out


def engineer_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # ensure ordering
    if {"year","quarter"}.issubset(df.columns) is False:
        raise ValueError(f"{ticker}: input must include 'year' and 'quarter'.")
    df["quarter_int"] = df["quarter"].map(_QMAP)
    df = df.sort_values(["year","quarter_int"]).reset_index(drop=True)

    # add Year_Quarter key
    df = _ensure_year_quarter_key(df)

    # ratios
    df = add_ratios(df)

    # net_plus_rnd
    if {"netIncome", "rnd"}.issubset(df.columns):
        df["net_plus_rnd"] = df["netIncome"] + df["rnd"]

    # lags we always want
    df = add_lags(df, "rnd", max_lag=12)
    if "rnd_to_rev_ratio" in df.columns:
        df = add_lags(df, "rnd_to_rev_ratio", max_lag=12)

    # NEW: add single-lag for assets & equity (if available)
    df = add_single_lag(df, "totalAssets")
    df = add_single_lag(df, "totalEquity")

    # seasonal one-hot
    df = one_hot_quarter(df)

    # YoY targets
    base_candidates = [
        "revenue","netIncome","grossProfit","rnd","ebitda","operatingIncome",
        "incomeBeforeTax","costOfRevenue","snaExpenses","operatingExpenses",
        "rnd_to_rev_ratio","sna_to_rev_ratio","grossProfitRatio",
        "operatingIncomeRatio","netIncomeRatio","net_plus_rnd"
    ]
    yoy_targets = [c for c in base_candidates if c in df.columns]
    df = add_yoy_growth(df, yoy_targets)

    # add ticker
    df["ticker"] = ticker

    # ---- APPEND 4 FUTURE QUARTERS (with lag roll-forward) ----
    df = append_future_quarters(df, n_future=4)

    # tidy column order
    front_cols = [c for c in ["ticker","Year_Quarter","year","quarter"] if c in df.columns]
    seasonal_cols = [f"Q{i}" for i in [1,2,3,4] if f"Q{i}" in df.columns]
    core_order = [
        "revenue","grossProfit","costOfRevenue","operatingExpenses",
        "snaExpenses","ebitda","operatingIncome","incomeBeforeTax",
        "netIncome","rnd","net_plus_rnd","totalAssets","totalEquity",
        "totalAssets_lag1","totalEquity_lag1"
    ]
    ratio_order = [
        "rnd_to_rev_ratio","sna_to_rev_ratio","grossProfitRatio",
        "operatingIncomeRatio","netIncomeRatio"
    ]
    rnd_lags    = [c for c in df.columns if c.startswith("rnd_lag")]
    rr_lags     = [c for c in df.columns if c.startswith("rnd_to_rev_ratio_lag")]
    yoy_cols    = [c for c in df.columns if c.endswith("_yoy")]

    cols = front_cols + seasonal_cols
    cols += [c for c in core_order if c in df.columns]
    cols += [c for c in ratio_order if c in df.columns]
    cols += sorted(rnd_lags, key=lambda c: int(c.split("rnd_lag")[1]))
    cols += sorted(rr_lags, key=lambda c: int(c.split("_lag")[1]))
    cols += yoy_cols
    if "sector" in df.columns: cols.append("sector")
    remaining = [c for c in df.columns if c not in cols]
    return df[cols + remaining]

def engineer_file(in_path: str, out_path: str, ticker_guess: str | None = None) -> None:
    df = pd.read_csv(in_path)
    if ticker_guess is None:
        base = os.path.basename(in_path)
        ticker_guess = base.split("_")[0].upper()
    df_out = engineer_df(df, ticker_guess)
    df_out.to_csv(out_path, index=False)

def main():
    found = sorted(glob.glob(os.path.join(processed_dir, "*_processed.csv")))
    if found:
        print(f"Scanning by glob: found {len(found)} processed files.")
        for fp in found:
            ticker = os.path.basename(fp).split("_")[0].upper()
            outp = os.path.join(features_dir, f"{ticker}_feature.csv")
            try:
                engineer_file(fp, outp, ticker_guess=ticker)
                print(f"[OK] {ticker}: {os.path.basename(outp)}")
            except Exception as e:
                print(f"[Error] {ticker}: {e}")

    # optional pass over explicit list if you keep one; omitted for brevity

if __name__ == "__main__":
    main()
