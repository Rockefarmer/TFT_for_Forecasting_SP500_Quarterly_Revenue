🚀 Forecasting S&P 500 Quarterly Revenue with TFT

Reproducible deep learning pipeline for firm-level quarterly revenue forecasting using the Temporal Fusion Transformer (TFT) on a long-horizon S&P 500 panel (1995Q1–2025Q2).

This project benchmarks TFT against LSTM and classical time series models (ARIMA/SARIMA) under strict chronological splits and leakage-free design.

🔥 Key Results
-
- Next-quarter (h=1) forecasting: ~9.3% MAPE on 155 S&P 500 firms.
- Strict evaluation:Fully leakage-free pipeline;Chronological train/val/test splits.
- Model comparison:TFT (panel model) vs LSTM vs ARIMA/SARIMA (per-ticker).
- Interpretability:Feature importance via TFT variable selection & attention


🧠 Methodology Overview
-
- Raw Financial Data 
- Feature Engineering (lags, growth, fundamentals, calendar) 
- TFT / LSTM / ARIMA Models
- Multi-horizon Forecasting (h = 1, ..., H) 
- Evaluation (MAPE, MAE, RMSE).

📊 Dataset
-
- Universe: 155 continuously listed S&P 500 firms
- Period: 1995Q1 – 2025Q2
- Frequency: Quarterly
- Target: Revenue (log-transformed during training)

⚠️ Data is not included due to licensing.
Please place your dataset under data/ following the expected structure.

📌 Highlights
 -
- Unified chronological split (70–15–15).
- Leakage-safe feature engineering.
- Panel modeling with TFT.
- Per-ticker classical baselines.
- Fully reproducible pipeline (fixed seeds, saved configs)

📂 Repository Structure. \
 -
├─ src/                     # Python modules (dataset, models, training, utils) \
├─ notebooks/               # EDA and experiment runs \
├─ figures/                 # Plots exported to paper \
├─ results/                 # Metrics, tables, predictions (small text files) \
├─ data/                    # (gitignored) raw/processed data, per-ticker CSVs \
├─ scripts/                 # (gitignored) checkpoints (TFT/LSTM) \
├─ requirements.txt         # Python deps (or use environment.yml) \
├─ .gitignore               # Python/Jupyter/LaTeX/data artifacts \
└─ README.md

🔧 Setup
 -
Option A — venv (Windows/macOS/Linux)

python -m venv .venv

- Windows: .venv\Scripts\activate
- macOS/Linux: source .venv/bin/activate
  
pip install --upgrade pip

pip install -r requirements.txt

▶️ Quick Start
 -

- Prepare data
python scripts/prepare_data.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --split_scheme 70-15-15 \
  --seed 2025

- Train TFT
python scripts/train_tft.py \
  --data_dir data/processed \
  --save_dir models/tft_baseline \
  --max_epochs 50 \
  --batch_size 256 \
  --hidden_size 128 \
  --lr 3e-4 \
  --seed 2025

- Evaluate & export tables

- python scripts/eval.py \
  --data_dir data/processed \
  --model_dir models/tft_baseline \
  --out_dir results/tft_baseline


- Train baselines \
  --LSTM\
python scripts/train_lstm.py --data_dir data/processed --save_dir models/lstm_baseline

  --ARIMA/SARIMA (per-ticker)\
python scripts/train_arima.py  --data_dir data/processed --out_dir results/arima \
python scripts/train_sarima.py --data_dir data/processed --out_dir results/sarima

🧱 Feature Taxonomy
 -
- Static s_i: ticker/CIK (group id), GICS sector (embedded).
- Observed–past o_{i,t}: revenue lags, growth/volatility, rolling stats, lagged fundamentals.
- Known–future k_{i,t+h}: calendar (year/quarter), lag1 assets/equity treated as deterministic at t.
- Exact columns and encodings are defined in src/data/schema.py (to be finalized).

📊 Metrics & Reporting
 -
- Primary metric: MAPE (%); secondary: MAE, RMSE, MdAPE, and accuracy=100−MAPE.

🔬 Experiments
 -
- Ablations: remove groups of features (e.g., drop totalAssets), compare deltas.
- Robustness: sector-wise breakdown, error distribution, sensitivity to horizon/split.
- Interpretability: TFT variable selection / attention diagnostics.
- Reproduce any experiment with a single YAML config (see configs/) and:
- python scripts/run_experiment.py --config configs/tft_baseline.yml

🛡️ Reproducibility
 -
- Fixed seeds, deterministic cuDNN where possible.
- Frozen train/val/test indices saved under data/processed/splits/.
- Environment export:
- pip freeze > requirements-lock.txt

📜 License
-
- Code: MIT (recommended) — see LICENSE.
- Data: Not included (subject to original provider terms).

📣 Citation
-
If you use this repo, please cite:

@misc{tft_sp500_quarterly_revenue_2025,
  title        = {Forecasting S\&P 500 Quarterly Revenue with Temporal Fusion Transformer},
  author       = {Wu, Qiping and Collaborators},
  year         = {2025},
  howpublished = {\url{https://github.com/Rockefarmer/Forecasting_SP500_Quarterly_Revenue_with_TFT}}
}

🙌 Acknowledgments
-
- Lim et al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.
- PyTorch
- statsmodels
- Open-source community

💡 About This Project
-
This repository is part of a Master’s thesis focused on:
- Financial time-series forecasting
- Deep learning (TFT)
- Multimodal extensions (future work with NLP: FinBERT, Llama).
