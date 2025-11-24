# -*- coding: utf-8 -*-
"""
TFT batch training with FIXED splits (aligned with LSTM):
  TRAIN: 1995Q1–2016Q1  (inclusive)
  VAL:   2016Q2–2020Q3  (inclusive)
  TEST:  2020Q4–2025Q2  (inclusive)

Inputs (engineered company-wise files):
  C:\ThesisResearch\thesis_project\data\re-feature_1995\*_feature.csv

Outputs (to C:\ThesisResearch\thesis_project\results\TFT\):
  - tft_test_predictions.csv
  - tft_test_agg_by_ticker.csv
  - tft_test_overall_stats.csv
  - best checkpoint + interpretability arrays
"""

import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch  # add this near the top (after numpy/pandas)

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# for a tiny speed boost on your 4060
torch.set_float32_matmul_precision("high")

# ------------------ USER CONFIG ------------------
DATA_DIR  = r"C:\ThesisResearch\thesis_project\data\re-feature_1995"
OUT_DIR   = r"C:\ThesisResearch\thesis_project\results\TFT"

# Fixed splits (inclusive)
SPLIT_TRAIN = ("1995Q1", "2016Q1")
SPLIT_VAL   = ("2016Q2", "2020Q3")
SPLIT_TEST  = ("2020Q4", "2025Q2")

SEED            = 42
MAX_EPOCHS      = 50
BATCH_SIZE      = 64
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
HIDDEN_SIZE     = 64
ATTN_HEADS      = 4
DROPOUT         = 0.15
MAX_ENCODER_LEN = 12     # 12 quarters of context
MAX_PRED_LEN    = 1      # next quarter
# -------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ HELPERS ------------------
def _safe_log1p(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))

def _parse_yq_str(yq: str):
    """Parse '1995Q1', '1995-Q1', '1995 Q1' (any case) -> (year, quarter)."""
    s = str(yq).strip().upper()
    m = re.search(r'(\d{4})\s*[-/ ]?\s*Q\s*([1-4])', s)
    if not m:
        raise ValueError(f"Cannot parse quarter string: {yq!r}")
    return int(m.group(1)), int(m.group(2))

def _yq_scalar(y, q) -> int:
    """Scalar year/quarter -> ordinal key."""
    return int(y) * 4 + (int(q) - 1)

def _yq_key_vec(year_series, quarter_series) -> pd.Series:
    """Vectorized ordinal key for Series/arrays (nullable int)."""
    y = pd.to_numeric(year_series, errors="coerce")
    q = pd.to_numeric(quarter_series, errors="coerce")
    return (y * 4 + (q - 1)).astype("Int64")

def _ensure_year_quarter(df: pd.DataFrame):
    """
    Ensure df has 'year' (int) and 'quarter_int' (1..4).
    Accepts any of: year/quarter_int, Year_Quarter, date, quarter.
    """
    if "year" not in df.columns:
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = dt.dt.year
        elif "Year_Quarter" in df.columns:
            y = df["Year_Quarter"].astype(str).str.extract(r"(\d{4})", expand=False)
            df["year"] = pd.to_numeric(y, errors="coerce")
        else:
            raise ValueError("Missing 'year' and no convertible 'date' or 'Year_Quarter'.")
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "quarter_int" not in df.columns:
        if "quarter" in df.columns:
            q = pd.to_numeric(df["quarter"], errors="coerce")
            if q.dropna().between(0, 3).all():
                q = q.replace({0:1, 1:2, 2:3, 3:4})
            df["quarter_int"] = q
        elif "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            df["quarter_int"] = dt.dt.quarter
        elif "Year_Quarter" in df.columns:
            q = df["Year_Quarter"].astype(str).str.extract(r"[Qq]([1-4])", expand=False)
            df["quarter_int"] = pd.to_numeric(q, errors="coerce")
        else:
            raise ValueError("Missing 'quarter_int' and no convertible 'date' or 'Year_Quarter'.")
    else:
        df["quarter_int"] = pd.to_numeric(df["quarter_int"], errors="coerce")
        if df["quarter_int"].dropna().between(0, 3).all():
            df["quarter_int"] = df["quarter_int"].replace({0:1, 1:2, 2:3, 3:4})
    return df

def _attach_time_idx(df: pd.DataFrame):
    """Sort by (ticker, year, quarter_int) and attach time_idx per ticker."""
    sort_keys = ["ticker", "year", "quarter_int"]
    df = df.sort_values(sort_keys).reset_index(drop=True)
    df["time_idx"] = df.groupby("ticker").cumcount()
    return df

def _apply_fixed_splits(df: pd.DataFrame):
    (ty1, tq1) = _parse_yq_str(SPLIT_TRAIN[0])
    (ty2, tq2) = _parse_yq_str(SPLIT_TRAIN[1])
    (vy1, vq1) = _parse_yq_str(SPLIT_VAL[0])
    (vy2, vq2) = _parse_yq_str(SPLIT_VAL[1])
    (sy1, sq1) = _parse_yq_str(SPLIT_TEST[0])
    (sy2, sq2) = _parse_yq_str(SPLIT_TEST[1])

    train_lo, train_hi = _yq_scalar(ty1, tq1), _yq_scalar(ty2, tq2)
    val_lo,   val_hi   = _yq_scalar(vy1, vq1), _yq_scalar(vy2, vq2)
    test_lo,  test_hi  = _yq_scalar(sy1, sq1), _yq_scalar(sy2, sq2)

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["quarter_int"] = pd.to_numeric(df["quarter_int"], errors="coerce")
    df = df[df["year"].notna() & df["quarter_int"].notna()]

    df["yq_key"] = _yq_key_vec(df["year"], df["quarter_int"])

    df["split"] = np.where((df["yq_key"] >= train_lo) & (df["yq_key"] <= train_hi), "train",
                   np.where((df["yq_key"] >= val_lo) & (df["yq_key"] <= val_hi), "val",
                   np.where((df["yq_key"] >= test_lo) & (df["yq_key"] <= test_hi), "test", "drop")))
    df = df[df["split"] != "drop"].copy()
    return df

def _load_all(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*_feature.csv"))
    if not files:
        raise RuntimeError(f"No *_feature.csv files found in {data_dir}")

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
    if not frames:
        raise RuntimeError("No readable feature files.")
    data = pd.concat(frames, ignore_index=True)

    # normalize ticker col
    if "ticker" not in data.columns:
        if "TICKER" in data.columns:
            data = data.rename(columns={"TICKER": "ticker"})
        else:
            raise ValueError("No 'ticker' column found.")

    data = _ensure_year_quarter(data)
    data = _attach_time_idx(data)

    # target
    if "revenue" not in data.columns:
        raise ValueError("Column 'revenue' is required.")
    if "revenue_log" not in data.columns:
        data["revenue_log"] = _safe_log1p(data["revenue"])

    # preferred log-lags (create if missing)
    if "revenue_log_lag1" not in data.columns:
        data["revenue_log_lag1"] = data.groupby("ticker")["revenue_log"].shift(1)
    if "revenue_log_lag4" not in data.columns:
        data["revenue_log_lag4"] = data.groupby("ticker")["revenue_log"].shift(4)

    # apply fixed splits (inclusive)
    data = _apply_fixed_splits(data)

    return data

def choose_known_future_cols(df: pd.DataFrame):
    """Calendar + lag1 equity (prefer *_log if present)."""
    known_reals = ["year", "quarter_int"]
    # REMOVED: totalAssets features
    eqty_log, eqty_lvl = "totalEquity_lag1_log", "totalEquity_lag1"

    if eqty_log in df.columns:
        known_reals.append(eqty_log)
    elif eqty_lvl in df.columns:
        known_reals.append(eqty_lvl)

    return known_reals

def build_feature_lists(df: pd.DataFrame):
    static_categoricals = [c for c in ["ticker", "gics_sectors"] if c in df.columns]
    known_reals = choose_known_future_cols(df)
    known_categoricals = []  # calendar numeric

    # Base unknown features - REMOVED R&D FEATURES and totalAssets
    base_unknown = [
        "revenue_log",
        "revenue_log_lag1", "revenue_log_lag4",
        "grossProfit", "costOfRevenue", "operatingExpenses", "snaExpenses",
        "ebitda", "operatingIncome", "incomeBeforeTax", "netIncome",
        "totalEquity",  # REMOVED: "totalAssets"
        "grossProfitRatio", "operatingIncomeRatio", "netIncomeRatio",
        # REMOVED: "rnd", "rnd_to_rev_ratio"
    ]
    unknown_reals = [c for c in base_unknown if c in df.columns]
    unknown_reals += [c for c in df.columns if c.endswith("_yoy")]
    # REMOVED R&D lag features:
    # unknown_reals += [c for c in df.columns if c.startswith("rnd_lag")]
    # unknown_reals += [c for c in df.columns if c.startswith("rnd_to_rev_ratio_lag")]

    # Deduplicate and avoid overlap with knowns
    unknown_reals = [c for c in dict.fromkeys(unknown_reals) if c not in known_reals]

    return static_categoricals, known_reals, known_categoricals, unknown_reals

def _sanitize_encoder_features(df: pd.DataFrame, encoder_cols: list) -> tuple[pd.DataFrame, list]:
    """
    Replace inf with NaN, then impute NaNs in encoder features.
    - For *_yoy: fill 0.0 (approx "no change"), add <col>_nanflag=1 where imputed.
    - For others: groupwise median by ticker, then 0.0 fallback; add <col>_nanflag.
    Returns: (df, list_of_added_flags)
    """
    df = df.copy()
    added_flags = []

    # replace +-inf with NaN first
    df[encoder_cols] = df[encoder_cols].replace([np.inf, -np.inf], np.nan)

    for c in encoder_cols:
        na = df[c].isna()
        if not na.any():
            continue
        flag = f"{c}_nanflag"
        df[flag] = na.astype(np.int8)
        added_flags.append(flag)
        if c.endswith("_yoy"):
            # conservative: treat missing YoY as 0 (no change)
            df.loc[na, c] = 0.0
        else:
            # groupwise median, then 0.0 fallback
            med = df.groupby("ticker")[c].transform("median")
            df.loc[na, c] = med[na]
            na2 = df[c].isna()
            if na2.any():
                df.loc[na2, c] = 0.0
    return df, added_flags

def inverse_log1p(x): return np.expm1(x)


def _extract_preds_and_index(predict_out):
    """
    Handle pytorch-forecasting 1.5.x variations:
      - (preds, index)
      - (preds, x, index)
      - (preds, x, out, index)
      - preds (no index)
    Returns: (preds_tensor, index_df_or_None)
    """
    preds, index_df = predict_out, None
    if isinstance(predict_out, (list, tuple)):
        # first item is always predictions
        preds = predict_out[0]
        # locate a pandas DataFrame among the rest as the index
        for obj in predict_out[1:]:
            if isinstance(obj, pd.DataFrame):
                index_df = obj
                break
            # sometimes it's a Series of indices
            if isinstance(obj, pd.Series):
                index_df = obj.to_frame()
                break
    return torch.as_tensor(preds), index_df

@torch.no_grad()
def evaluate_predictions(model, loader, tag: str):
    """
    Robust evaluation:
      - extracts preds + index no matter how many items predict(...) returns
      - supports dataloader yielding (x, y) or (x, (y, weight)) or dict-like
      - returns (y_true_lvl, y_pred_lvl, index_df_or_None)
    """
    # ---- 1) predictions (+ optional index)
    out = model.predict(loader, mode="prediction", return_index=True)
    preds_t, index_df = _extract_preds_and_index(out)
    preds = preds_t.detach().cpu().numpy()
    preds = np.squeeze(preds)

    # ---- 2) collect ground truth from loader
    ys = []
    for _x, y in loader:
        # y can be (target, weight)
        if isinstance(y, (tuple, list)) and len(y) >= 1:
            y = y[0]
        if isinstance(y, dict):
            y = y.get("target", y.get("y", y))
        y = torch.as_tensor(y).detach().cpu().numpy()
        ys.append(y)
    y_true = np.squeeze(np.concatenate(ys, axis=0))

    # ---- 3) align lengths defensively
    n = min(len(y_true), len(preds))
    if len(y_true) != len(preds):
        print(f"[WARN] pred/true length mismatch: pred={len(preds)} true={len(y_true)}; truncating to {n}")
    y_true = y_true[:n]
    preds  = preds[:n]
    if index_df is not None and len(index_df) != n:
        index_df = index_df.iloc[:n].copy()

    # ---- 4) invert log1p -> level
    y_true_lvl = np.expm1(y_true)
    y_pred_lvl = np.expm1(preds)

    # ---- 5) metrics
    denom = np.clip(np.abs(y_true_lvl), 1e-8, None)
    mape = np.mean(np.abs((y_true_lvl - y_pred_lvl) / denom)) * 100.0
    rmse = np.sqrt(np.mean((y_true_lvl - y_pred_lvl) ** 2))
    mae  = np.mean(np.abs(y_true_lvl - y_pred_lvl))
    print(f"[{tag}] MAPE: {mape:.2f}% | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    return y_true_lvl, y_pred_lvl, index_df


# ------------------ MAIN ------------------
def main():
    seed_everything(SEED, workers=True)

    df = _load_all(DATA_DIR)
    static_categoricals, known_reals, known_categoricals, unknown_reals = build_feature_lists(df)

    # ---- enforce encoder completeness in every split (drop early lag-missing rows)
    REQ_LAGS = [c for c in ["revenue_log", "revenue_log_lag1", "revenue_log_lag4"] if c in df.columns]

    def _prune_split(dfsplit: pd.DataFrame, name: str) -> pd.DataFrame:
        before = len(dfsplit)
        out = dfsplit.dropna(subset=REQ_LAGS).copy()
        dropped = before - len(out)
        if dropped > 0:
            print(f"[INFO] {name}: dropped {dropped} rows lacking {REQ_LAGS}")
        return out

    df_train = _prune_split(df.loc[df["split"].eq("train")], "TRAIN")
    df_val   = _prune_split(df.loc[df["split"].eq("val")],   "VAL")
    df_test  = _prune_split(df.loc[df["split"].eq("test")],  "TEST")  # OK to drop earliest quarters in TEST

    # ---- sanitize remaining encoder features (fill NaNs/Infs + add flags)
    encoder_cols = list(dict.fromkeys(["revenue_log"] + unknown_reals))  # target is also fed as unknown_real

    # sanitize each split independently (creates *_nanflag columns only where needed)
    df_train, flags_train = _sanitize_encoder_features(df_train, encoder_cols)
    df_val,   flags_val   = _sanitize_encoder_features(df_val,   encoder_cols)
    df_test,  flags_test  = _sanitize_encoder_features(df_test,  encoder_cols)

    # ---- ALIGN COLUMNS ACROSS SPLITS ----
    # union of all flags created in any split
    flag_cols = sorted(set(flags_train) | set(flags_val) | set(flags_test))

    # ensure every split has every flag col; if missing, create and fill with 0
    for col in flag_cols:
        if col not in df_train.columns:
            df_train[col] = 0
        if col not in df_val.columns:
            df_val[col] = 0
        if col not in df_test.columns:
            df_test[col] = 0

    # also ensure every split has all encoder columns referenced
    for col in encoder_cols:
        if col not in df_train.columns:
            df_train[col] = 0.0
        if col not in df_val.columns:
            df_val[col] = 0.0
        if col not in df_test.columns:
            df_test[col] = 0.0

    # extend unknown_reals with flags (informative but harmless)
    unknown_reals_extended = list(dict.fromkeys(unknown_reals + flag_cols))
 
    # union of all flags
    flag_cols = sorted(set(flags_train) | set(flags_val) | set(flags_test))
    # extend unknown_reals with flags (they are informative but harmless)
    unknown_reals_extended = list(dict.fromkeys(unknown_reals + flag_cols))

    # ---- Build datasets
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="revenue_log",
        group_ids=["ticker"],
        max_encoder_length=MAX_ENCODER_LEN,
        max_prediction_length=MAX_PRED_LEN,

        static_categoricals=static_categoricals,
        static_reals=[],

        time_varying_known_categoricals=known_categoricals,
        time_varying_known_reals=known_reals,

        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=list(dict.fromkeys(["revenue_log"] + unknown_reals_extended)),

        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = training.from_dataset(training, df_val)
    testing    = training.from_dataset(training, df_test)

    train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    test_loader  = testing.to_dataloader(train=False,  batch_size=BATCH_SIZE, num_workers=0)

    # ---- Trainer (GPU + AMP for RTX 4060)
    early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    ckpt = ModelCheckpoint(
        dirpath=OUT_DIR,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", save_top_k=1, mode="min"
    )
    lr_mon = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",   # ← was "16-mixed"; BF16 avoids the -1e9 overflow
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        callbacks=[early_stop, ckpt, lr_mon],
        default_root_dir=OUT_DIR,
        log_every_n_steps=50,
    )


    loss = QuantileLoss()

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTN_HEADS,
        dropout=DROPOUT,
        loss=loss,
        optimizer="AdamW",
        weight_decay=WEIGHT_DECAY,
        reduce_on_plateau_patience=4,
    )

    print(f"Model size: {tft.size()/1e3:.1f}k parameters")
    trainer.fit(tft, train_loader, val_loader)

    best_path = ckpt.best_model_path
    print("Best checkpoint:", best_path)
    best = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # ---- Evaluate
    val_true,  val_pred,  val_idx  = evaluate_predictions(best, val_loader,  "VAL")
    test_true, test_pred, test_idx = evaluate_predictions(best, test_loader, "TEST")

    # ---- Save row-level TEST predictions
    # ---- Build a tidy TEST predictions DataFrame aligned to predicted timestamps
    # test_idx typically has columns like ['time_idx', 'ticker', ...] (group ids)
    if test_idx is None:
        # Fallback: build a simple sequential index if PF didn't return one (rare)
        test_idx = pd.DataFrame({
            "time_idx": testing.data[testing.index.time][:len(test_true)],  # best-effort
            "ticker":   testing.data[testing.index.group_ids[0]][:len(test_true)],
        })

    pred_df = test_idx.copy()
    # normalize column names if needed
    if "ticker" not in pred_df.columns:
        # find a column that corresponds to group id, often named exactly "ticker"
        # If your group id uses a different name, set it explicitly here:
        # pred_df = pred_df.rename(columns={"<your_group_col>": "ticker"})
        pass

    pred_df["y_true"] = test_true
    pred_df["y_pred"] = test_pred
    pred_df["APE_%"]  = np.where(np.abs(pred_df["y_true"]) < 1e-8, np.nan,
                                np.abs(pred_df["y_true"] - pred_df["y_pred"]) / np.abs(pred_df["y_true"]) * 100.0)

    # Keep only the predicted timestamps from df_test and attach truth/pred
    merge_keys = ["ticker", "time_idx"]
    df_test_out = pd.merge(
        df_test.reset_index(drop=True),
        pred_df[merge_keys + ["y_true", "y_pred", "APE_%"]],
        on=merge_keys,
        how="inner"
    )

    test_path = os.path.join(OUT_DIR, "tft_test_predictions.csv")
    df_test_out.to_csv(test_path, index=False)

    # ---- Per-ticker metrics (use aligned df_test_out)
    def _agg(g):
        rmse = np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2))
        mae  = np.mean(np.abs(g["y_true"] - g["y_pred"]))
        mape = np.mean(np.abs((g["y_true"] - g["y_pred"]) / np.clip(np.abs(g["y_true"]), 1e-8, None))) * 100.0
        return pd.Series({"RMSE": rmse, "MAE": mae, "MAPE": mape, "N": len(g)})

    by_ticker = df_test_out.groupby("ticker", as_index=False).apply(_agg).reset_index(drop=True)
    agg_path  = os.path.join(OUT_DIR, "tft_test_agg_by_ticker.csv")
    by_ticker.to_csv(agg_path, index=False)

    # ---- Overall stats
    overall = pd.DataFrame({
        "metric": ["RMSE", "MAE", "MAPE(%)"],
        "mean":   [by_ticker["RMSE"].mean(), by_ticker["MAE"].mean(), by_ticker["MAPE"].mean()],
        "std":    [by_ticker["RMSE"].std(),  by_ticker["MAE"].std(),  by_ticker["MAPE"].std()],
        "N_tickers": [len(by_ticker)] * 3
    })
    ov_path = os.path.join(OUT_DIR, "tft_test_overall_stats.csv")
    overall.to_csv(ov_path, index=False)

    print("Saved:")
    print(" - per-row test predictions:", test_path)
    print(" - per-ticker metrics:",      agg_path)
    print(" - overall mean/std:",        ov_path)

    # ---- Interpretability snapshot
    try:
        interpret = best.interpret_output(best.predict(val_loader, mode="raw"))
        np.save(os.path.join(OUT_DIR, "val_attention.npy"),    interpret.get("attention", np.array([])))
        np.save(os.path.join(OUT_DIR, "val_encoder_vars.npy"), interpret.get("encoder_variables", np.array([])))
        np.save(os.path.join(OUT_DIR, "val_decoder_vars.npy"), interpret.get("decoder_variables", np.array([])))
        print("Interpretability arrays saved.")
    except Exception as e:
        print(f"[WARN] interpret_output failed: {e}")

if __name__ == "__main__":
    main()