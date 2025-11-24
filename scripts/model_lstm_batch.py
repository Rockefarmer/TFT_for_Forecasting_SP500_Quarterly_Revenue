# 1995_2025_batch_LSTM.py
# Company-wise next-quarter revenue prediction from per-ticker feature files.
# Features: log-space revenue features + business metrics; Target: revenue (RobustScaled).
# Lookback search: {12,14,16}. Per-firm outputs + combined predictions, overall and sector stats.

import os, glob, time, math
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------- CONFIG -----------------------------
FEATURES_DIR = r"C:\ThesisResearch\thesis_project\data\features_since1995"
RESULTS_DIR  = r"C:\ThesisResearch\thesis_project\results\LSTM"

# ensure directories
os.makedirs(RESULTS_DIR, exist_ok=True)

# CPU throttling (keeps Windows responsive)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Model / training
LOOKBACK_CANDS = [16]   # best lookbacks to try
EPOCHS         = 30               # best lookbacks to try
PATIENCE       = 5
BATCH_SIZE     = 32
HIDDEN         = 64
LAYERS         = 1
DROPOUT        = 0.10
LR             = 5e-3
WEIGHT_DECAY   = 1e-5
HUBER_DELTA    = 1.0
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed time splits (inclusive)
SPLIT_TRAIN = ("1995Q1", "2016Q1")
SPLIT_VAL   = ("2016Q2", "2020Q3")
SPLIT_TEST  = ("2020Q4", "2025Q2")

# Column candidates
ID_CANDS  = ["ticker","symbol","firm_id","gvkey","permno","cusip","company_id"]
REV_CANDS = ["revenue","total_revenue","sales","revt","revenue_t"]

RND_CANDS = ["rnd_to_rev_ratio","rnd_to_revenue_ratio","rnd_to_rev"]
SGA_CANDS = ["sga_to_rev_ratio","sna_to_rev_ratio","sga_to_revenue_ratio"]
GP_CANDS  = ["grossProfitRatio","gross_profit_ratio"]

YQ_CANDS  = ["Year_Quarter","year_quarter","yr_qtr","fiscal_quarter","period"]

# Extra business features to include if available (levels + lags 1..4)
BUSINESS_FEATURES = ["profit_margin", "asset_turnover", "gross_profit_margin", "rnd_intensity"]

EPS_LOG = 1e-6   # to avoid log(0)

# ----------------------------- HELPERS -----------------------------
def pick_col(cands: List[str], cols: List[str]) -> Optional[str]:
    for c in cands:
        for col in cols:
            if col.lower() == c.lower():
                return col
    return None

def parse_yq_string(s: str) -> Optional[Tuple[int,int]]:
    if not isinstance(s, str):
        return None
    z = s.strip().upper().replace(" ", "")
    try:
        if "Q" in z:
            if z.startswith("Q"):     # Q3-2020
                q = int(z[1]); y = int(z.split("-")[-1])
            else:                     # 2020Q3 or 2020-Q3
                parts = z.replace("-", "").split("Q")
                y = int(parts[0]); q = int(parts[1])
            return (y, q)
    except Exception:
        return None
    return None

def yq_to_int(y: int, q: int) -> int:
    return int(y) * 4 + int(q)

def bounds_to_int(bounds_tuple):
    lo_s, hi_s = bounds_tuple
    yq_lo = parse_yq_string(lo_s)
    yq_hi = parse_yq_string(hi_s)
    assert yq_lo and yq_hi, f"Invalid YQ bounds: {bounds_tuple}"
    return yq_to_int(*yq_lo), yq_to_int(*yq_hi)

def build_yq_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = df.columns

    if "year" in cols and "quarter" in cols:
        def to_year_int(y):
            if pd.isna(y): return np.nan
            s = str(y).strip()
            digits = ''.join(ch for ch in s if ch.isdigit())
            return int(digits[:4]) if digits else np.nan

        def to_quarter_int(q):
            if pd.isna(q): return np.nan
            s = str(q).strip().upper().replace(" ", "")
            if s.startswith("Q"): s = s[1:]
            digits = ''.join(ch for ch in s if ch.isdigit())
            try:
                qi = int(digits)
                return qi if 1 <= qi <= 4 else np.nan
            except:
                return np.nan

        df["year_int"] = df["year"].apply(to_year_int)
        df["quarter_int"] = df["quarter"].apply(to_quarter_int)

        if any(df["year_int"].isna() | df["quarter_int"].isna()):
            yq_col = pick_col(YQ_CANDS, cols)
            if yq_col:
                mask_bad = df["year_int"].isna() | df["quarter_int"].isna()
                yq_parsed = df.loc[mask_bad, yq_col].apply(parse_yq_string)
                df.loc[mask_bad, "year_int"]    = yq_parsed.apply(lambda t: t[0] if isinstance(t, tuple) else np.nan)
                df.loc[mask_bad, "quarter_int"] = yq_parsed.apply(lambda t: t[1] if isinstance(t, tuple) else np.nan)

        df = df.dropna(subset=["year_int","quarter_int"]).copy()
        df["year_int"] = df["year_int"].astype(int)
        df["quarter_int"] = df["quarter_int"].astype(int)
        df["yq_int"] = df["year_int"]*4 + df["quarter_int"]
        return df

    yq_col = pick_col(YQ_CANDS, cols)
    if yq_col:
        yq = df[yq_col].apply(parse_yq_string)
        df = df[yq.notna()].copy()
        df["yq_int"] = yq[yq.notna()].apply(lambda t: t[0]*4 + t[1])
        return df

    for guess in ["quarter_int", "time_idx", "quarter_idx"]:
        if guess in cols:
            df["yq_int"] = pd.to_numeric(df[guess], errors="coerce")
            df = df.dropna(subset=["yq_int"]).copy()
            df["yq_int"] = df["yq_int"].astype(int)
            return df

    raise ValueError("Cannot build yq_int. Need (year & quarter), or Year_Quarter, or an integer-like time index.")

def wape(y, yhat, eps=1e-8):
    return float(np.sum(np.abs(y - yhat)) / np.clip(np.sum(np.abs(y)), eps, None))

def smape(y, yhat, eps=1e-8):
    return float(np.mean(2*np.abs(y-yhat)/np.clip(np.abs(y)+np.abs(yhat), eps, None))*100)

def get_sector_from_feature_df(df: pd.DataFrame) -> str:
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

# ---------------------- FEATURE ENGINEERING ----------------------
def add_quarter_features(df: pd.DataFrame):
    # q in {1,2,3,0} since yq_int % 4 (we want {1..4})
    df["q"] = (df["yq_int"] % 4) + 1
    df["q_sin"] = np.sin(2*np.pi*df["q"]/4.0)
    df["q_cos"] = np.cos(2*np.pi*df["q"]/4.0)

def create_log_features(df: pd.DataFrame, rev_col: str) -> pd.DataFrame:
    df = df.copy()
    df["log_rev"] = np.log(df[rev_col] + EPS_LOG)

    # Revenue lags in log space
    for k in range(1, 13):
        df[f"log_rev_lag{k}"] = df["log_rev"].shift(k)

    # QoQ / YoY in log domain
    df["log_rev_qoq"] = df["log_rev"] - df["log_rev_lag1"]
    df["log_rev_yoy"] = df["log_rev"] - df["log_rev_lag4"]
    return df

def add_optional_ratios(df: pd.DataFrame, all_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    used = {}

    rnd_col = pick_col(RND_CANDS, all_cols)
    sga_col = pick_col(SGA_CANDS, all_cols)
    gp_col  = pick_col(GP_CANDS,  all_cols)
    if rnd_col:
        used["rnd"] = rnd_col
        for k in range(1, 13):
            df[f"{rnd_col}_lag{k}"] = df[rnd_col].shift(k)
    if sga_col: used["sga"] = sga_col
    if gp_col:  used["gp"]  = gp_col
    return df, used

def add_business_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}
    added = []
    for name in BUSINESS_FEATURES:
        if name in lower_map:
            col = lower_map[name]
            added.append(col)
            for k in range(1,5):
                df[f"{col}_lag{k}"] = df[col].shift(k)
    return df, added

def select_feature_columns(df: pd.DataFrame, base_ratios: Dict[str,str], business_added: List[str]) -> List[str]:
    feat = [f"log_rev_lag{k}" for k in range(1,13)] + ["log_rev_qoq","log_rev_yoy","q_sin","q_cos"]

    if "rnd" in base_ratios:
        rnd_col = base_ratios["rnd"]
        feat += [rnd_col] + [f"{rnd_col}_lag{k}" for k in range(1,13)]
    if "sga" in base_ratios:
        feat += [base_ratios["sga"]]
    if "gp" in base_ratios:
        feat += [base_ratios["gp"]]

    for col in business_added:
        feat += [col] + [f"{col}_lag{k}" for k in range(1,5)]

    feat = [c for c in feat if c in df.columns]
    return feat

# ----------------------------- DATASETS -----------------------------
class SeqDS(Dataset):
    def __init__(self, X, y, lookback):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.lookback = lookback
        self.idxs = np.arange(lookback-1, len(X))
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        e = self.idxs[i]; s = e - self.lookback + 1
        return torch.from_numpy(self.X[s:e+1]), torch.tensor(self.y[e])

# ------------------------------ MODEL ------------------------------
class TinyLSTM(nn.Module):
    def __init__(self, nfeat, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(nfeat, hidden, num_layers=layers,
                            dropout=dropout if layers>1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.lstm(x)  # [B,T,H]
        return self.fc(out[:,-1,:]).squeeze(-1)

def create_loaders(ds_tr, ds_va, ds_te, batch_size=BATCH_SIZE):
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=False) if ds_va is not None else None
    test_loader  = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader

def train_model(nfeat, train_loader, val_loader, ticker):
    model = TinyLSTM(nfeat=nfeat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=HUBER_DELTA)  # Huber

    best_val = math.inf
    best_state = None
    pat = 0
    hist = {"train_loss": [], "val_loss": []}

    for ep in range(1, EPOCHS+1):
        # train
        model.train()
        tot, n = 0.0, 0
        for xb,yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p = model(xb)
            loss = loss_fn(p, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        trl = tot / max(1,n)
        hist["train_loss"].append(trl)

        # val
        if val_loader is not None:
            model.eval()
            totv, nv = 0.0, 0
            with torch.no_grad():
                for xb,yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pv = model(xb)
                    lv = loss_fn(pv, yb)
                    totv += lv.item() * xb.size(0)
                    nv += xb.size(0)
            vall = totv / max(1,nv)
            hist["val_loss"].append(vall)
        else:
            vall = trl
            hist["val_loss"].append(vall)

        if ep == 1 or ep % 2 == 0:
            print(f"  {ticker} | Epoch {ep:02d} | train MAE={trl:.4f} | val MAE={vall:.4f}")

        if vall < best_val:
            best_val = vall
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist

# ------------------------------ CORE ------------------------------
def train_one_firm(path_csv: str):
    # read
    try:
        raw = pd.read_csv(path_csv)
    except Exception as e:
        print(f"[SKIP] {os.path.basename(path_csv)}: cannot read ({e})")
        return None, None

    sector = get_sector_from_feature_df(raw)

    cols = raw.columns.tolist()
    id_col = pick_col(ID_CANDS, cols) or "ticker"
    if id_col not in raw.columns:
        base = os.path.basename(path_csv)
        ticker = os.path.splitext(base)[0].split("_")[0].upper()
        raw[id_col] = ticker
    ticker = str(raw[id_col].astype(str).str.upper().iloc[0])

    rev_col = pick_col(REV_CANDS, cols)
    if not rev_col:
        print(f"[SKIP] {os.path.basename(path_csv)}: no revenue-like column")
        return None, None

    try:
        g = build_yq_int(raw)
    except Exception as e:
        print(f"[SKIP] {os.path.basename(path_csv)}: cannot build yq_int ({e})")
        return None, None

    g = g.sort_values("yq_int").dropna(subset=[rev_col]).copy()
    if g.empty:
        print(f"[SKIP] {ticker}: empty after revenue dropna")
        return None, None
    g[id_col] = g[id_col].astype(str).str.upper()

    # seasonality helpers
    add_quarter_features(g)

    # log-space revenue features
    g = create_log_features(g, rev_col)

    # optional ratios + business metrics
    g, used_ratios = add_optional_ratios(g, g.columns.tolist())
    g, biz_added   = add_business_features(g)

    # target in original units (next quarter revenue)
    g["target_rev"] = g[rev_col].shift(-1)

    # REQUIRED columns for modeling
    need = [f"log_rev_lag{k}" for k in range(1, 13)] + ["log_rev_qoq","log_rev_yoy","target_rev"]
    g = g.dropna(subset=need).copy()
    if len(g) < 40:
        print(f"[SKIP] {ticker}: not enough rows after feature/target construction ({len(g)})")
        return None, None

    # time splits
    tr_lo,tr_hi = bounds_to_int(SPLIT_TRAIN)
    va_lo,va_hi = bounds_to_int(SPLIT_VAL)
    te_lo,te_hi = bounds_to_int(SPLIT_TEST)

    tr = g[(g["yq_int"]>=tr_lo)&(g["yq_int"]<=tr_hi)].copy()
    va = g[(g["yq_int"]>=va_lo)&(g["yq_int"]<=va_hi)].copy()
    te = g[(g["yq_int"]>=te_lo)&(g["yq_int"]<=te_hi)].copy()

    if len(tr) < 24 or len(te) < 8:
        print(f"[SKIP] {ticker}: insufficient split sizes (train={len(tr)}, test={len(te)})")
        return None, None

    # Feature list
    feature_cols = select_feature_columns(g, used_ratios, biz_added)

    # Impute features by train-median & scale features
    for c in feature_cols:
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
        va[c] = pd.to_numeric(va[c], errors="coerce")
        te[c] = pd.to_numeric(te[c], errors="coerce")
        med = tr[c].median()
        tr[c] = tr[c].fillna(med)
        va[c] = va[c].fillna(med)
        te[c] = te[c].fillna(med)

    x_scaler = RobustScaler(quantile_range=(25, 75))
    x_scaler.fit(tr[feature_cols])

    trX = x_scaler.transform(tr[feature_cols])
    vaX = x_scaler.transform(va[feature_cols]) if len(va) else np.zeros((0, len(feature_cols)))
    teX = x_scaler.transform(te[feature_cols])

    # Target scaling on original revenue
    y_scaler = RobustScaler(quantile_range=(25, 75))
    trY_raw = tr["target_rev"].to_numpy().reshape(-1, 1)
    y_scaler.fit(trY_raw)
    trY = y_scaler.transform(trY_raw).ravel()
    vaY = y_scaler.transform(va["target_rev"].to_numpy().reshape(-1, 1)).ravel() if len(va) else np.zeros((0,))
    teY = y_scaler.transform(te["target_rev"].to_numpy().reshape(-1, 1)).ravel()

    # try lookbacks and keep the best
    candidates = []
    for lb in LOOKBACK_CANDS:
        ds_tr = SeqDS(trX, trY, lb)
        ds_va = SeqDS(vaX, vaY, lb) if len(vaY) >= lb else None
        ds_te = SeqDS(teX, teY, lb)

        if len(ds_te.idxs) == 0:
            continue

        train_loader, val_loader, test_loader = create_loaders(ds_tr, ds_va, ds_te)
        model, hist = train_model(len(feature_cols), train_loader, val_loader, ticker)

        # choose by min val loss (fallback to train)
        if hist["val_loss"]:
            score = min(hist["val_loss"])
        else:
            score = min(hist["train_loss"]) if hist["train_loss"] else math.inf

        candidates.append({
            "lookback": lb, "model": model,
            "ds_tr": ds_tr, "ds_va": ds_va, "ds_te": ds_te,
            "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
            "score": score
        })

    if not candidates:
        print(f"[SKIP] {ticker}: no test windows for lookbacks {LOOKBACK_CANDS}")
        return None, None

    best = min(candidates, key=lambda d: d["score"])
    lookback = best["lookback"]
    model    = best["model"]
    ds_te    = best["ds_te"]
    test_loader = best["test_loader"]
    print(f"  {ticker} | Selected lookback = {lookback}")

    # ---- Predict on test ----
    preds = []
    with torch.no_grad():
        model.eval()
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())

    if not preds:
        print(f"[SKIP] {ticker}: test loader produced zero batches")
        return None, None

    y_pred_s = np.concatenate(preds)                 # scaled predictions
    idxs = ds_te.idxs
    y_true_s = ds_te.y[idxs]                         # scaled truths
    yq_test_aligned = te["yq_int"].to_numpy()[idxs]

    # inverse-transform to original revenue
    y_true = y_scaler.inverse_transform(y_true_s.reshape(-1,1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()

    # metrics (linear space) with denominator floor to stabilize % metrics
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    # Denominator floor from TRAIN target_rev in original space
    train_median = float(np.median(tr["target_rev"].dropna().to_numpy())) if len(tr) else 1.0
    den_floor = max(train_median * 0.10, 1.0)  # 10% of train median or $1
    eps = 1e-6

    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), den_floor, None))) * 100.0)
    wape_pct  = wape(y_true, y_pred) * 100.0
    smape_pct = float(np.mean(2*np.abs(y_true - y_pred) / np.clip(np.abs(y_true) + np.abs(y_pred), den_floor, None)) * 100.0)

    # ---- Save per-firm outputs ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    firm_dir = os.path.join(RESULTS_DIR, "per_company", ticker)
    os.makedirs(firm_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        "ticker": ticker,
        "yq_int": yq_test_aligned,
        "y_true": y_true,
        "y_pred": y_pred
    })
    pred_path = os.path.join(firm_dir, f"{ticker}_lstm_predictions_{ts}.csv")
    pred_df.to_csv(pred_path, index=False)

    readme_path = os.path.join(firm_dir, f"{ticker}_README_{ts}.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            f"Ticker: {ticker}\n"
            f"Sector: {sector}\n"
            f"File  : {os.path.basename(path_csv)}\n"
            f"Features: {len(feature_cols)} (log lags, log growth, seasonality, ratios/metrics if present)\n"
            f"Lookback selected: {lookback}\n"
            f"Splits: Train {SPLIT_TRAIN[0]}–{SPLIT_TRAIN[1]}, "
            f"Val {SPLIT_VAL[0]}–{SPLIT_VAL[1]}, Test {SPLIT_TEST[0]}–{SPLIT_TEST[1]}\n"
            f"Test MAE  : {mae:.4f}\n"
            f"Test RMSE : {rmse:.4f}\n"
            f"Test MAPE : {mape:.2f}%\n"
            f"Test WAPE : {wape_pct:.2f}%\n"
            f"Test sMAPE: {smape_pct:.2f}%\n"
            f"Pred CSV  : {pred_path}\n"
        )

    metrics = {
        "ticker": ticker,
        "Sector": sector,
        "lookback": lookback,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape_pct,
        "sMAPE": smape_pct,
        "features": len(feature_cols),
        "file": os.path.basename(path_csv)
    }
    return metrics, pred_df

# ------------------------------ MAIN ------------------------------
def main():
    print(f"Scanning folder: {FEATURES_DIR}")
    patterns = ["*_feature.csv", "*_features.csv"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(FEATURES_DIR, p)))
    files = sorted(list(dict.fromkeys(files)))  # de-dup + stable order
    print(f"Found {len(files)} CSV files.")

    ts = int(time.time())
    summary_rows = []
    all_pred_rows = []

    for i, path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(path)}")
        res, pred_df = train_one_firm(path)
        if res:
            summary_rows.append(res)
        if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
            all_pred_rows.append(pred_df)

    # Combined predictions
    if all_pred_rows:
        all_preds = pd.concat(all_pred_rows, ignore_index=True)
        all_preds_path = os.path.join(RESULTS_DIR, f"_ALL_predictions_{ts}.csv")
        all_preds.to_csv(all_preds_path, index=False)
        print("\nSaved ALL-firm predictions ->", all_preds_path)

        # Panel metrics
        y_true = all_preds["y_true"].to_numpy()
        y_pred = all_preds["y_pred"].to_numpy()
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        eps  = 1e-6
        # crude panel denominator floor using median of abs truths
        den_floor = max(np.median(np.abs(y_true)) * 0.10, 1.0)
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), den_floor, None))) * 100.0)
        wape_pct  = wape(y_true, y_pred) * 100.0
        smape_pct = float(np.mean(2*np.abs(y_true - y_pred) / np.clip(np.abs(y_true)+np.abs(y_pred), den_floor, None)) * 100.0)

        panel_metrics = pd.DataFrame([{
            "firms_predicted": int(all_preds["ticker"].nunique()),
            "rows_predicted":  len(all_preds),
            "PANEL_MAE": mae,
            "PANEL_RMSE": rmse,
            "PANEL_MAPE": mape,
            "PANEL_WAPE": wape_pct,
            "PANEL_sMAPE": smape_pct
        }])
        panel_metrics_path = os.path.join(RESULTS_DIR, f"_ALL_metrics_{ts}.csv")
        panel_metrics.to_csv(panel_metrics_path, index=False)
        print("Saved ALL-firm panel metrics ->", panel_metrics_path)

    # Per-firm metrics & stats
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        perfirm_csv = os.path.join(RESULTS_DIR, f"lstm_results_per_firm_{ts}.csv")
        summary.to_csv(perfirm_csv, index=False)
        print(f"Saved per-firm LSTM results -> {perfirm_csv}")

        # overall stats (across companies)
        metric_cols = ["RMSE", "MAE", "MAPE", "WAPE", "sMAPE"]
        overall = {}
        for col in metric_cols:
            s = pd.to_numeric(summary[col], errors="coerce")
            overall[col] = {
                "count": int(s.count()),
                "mean":  float(s.mean()),
                "median":float(s.median()),
                "std":   float(s.std(ddof=1)),
                "min":   float(s.min()),
                "max":   float(s.max()),
            }
        overall_csv = os.path.join(RESULTS_DIR, f"lstm_overall_stats_{ts}.csv")
        pd.DataFrame(overall).T.to_csv(overall_csv)
        print(f"Saved overall stats -> {overall_csv}")

        # sector-wise stats
        if "Sector" not in summary.columns:
            summary["Sector"] = "Unknown"
        summary["Sector"] = summary["Sector"].fillna("Unknown").astype(str)
        sector_stats = (
            summary
            .groupby("Sector")[metric_cols]
            .agg(["count", "mean", "median", "std", "min", "max"])
        )
        sector_stats.columns = ["_".join(col).strip() for col in sector_stats.columns.values]
        sector_csv = os.path.join(RESULTS_DIR, f"lstm_sector_stats_{ts}.csv")
        sector_stats.reset_index().to_csv(sector_csv, index=False)
        print(f"Saved sector-wise stats -> {sector_csv}")

        print("\nQuick medians across firms:")
        print(summary[["MAE","RMSE","MAPE","WAPE","sMAPE"]].median())

    else:
        print("No firms processed (check files/columns/splits).")

if __name__ == "__main__":
    main()
