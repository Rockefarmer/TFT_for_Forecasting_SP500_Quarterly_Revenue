import os
import pandas as pd

# === Paths (edit these to match your repo layout) ===
input_dir = r"C:\ThesisResearch\thesis_project\data\raw\sp500_quarterly_1995_2025"
output_dir = r"C:\ThesisResearch\thesis_project\data\preprocessed_since1995"
mapping_csv_path = r"C:\ThesisResearch\thesis_project\data\sectors\fmp_to_gics_sectors_1995.csv"
tickers_csv_path = r"C:\ThesisResearch\thesis_project\data\raw\sp500_constituents_since_1995.csv"

os.makedirs(output_dir, exist_ok=True)

# === Load FMP -> GICS sector mapping once ===
# Expect columns: fmp_sector, gics_sector, gics_sector_code
sector_map = pd.read_csv(mapping_csv_path)
sector_map['fmp_sector'] = sector_map['fmp_sector'].astype(str).str.strip()

# Build handy dicts (for fast map without merge)
fmp_to_gics = dict(zip(sector_map['fmp_sector'], sector_map['gics_sector']))
fmp_to_code = dict(zip(sector_map['fmp_sector'], sector_map['gics_sector_code']))

# === Universe from CSV (NEW) ===
# Accepts any of: symbol/Symbol/ticker/Ticker; ignores blanks/dupes; uppercases & strips.
def _read_universe(csv_path: str) -> list[str]:
    dfu = pd.read_csv(csv_path)
    col = None
    for cand in ['symbol', 'Symbol', 'ticker', 'Ticker']:
        if cand in dfu.columns:
            col = cand
            break
    if col is None:
        raise ValueError(
            f"Could not find a ticker column in {csv_path}. "
            "Expected one of: symbol, Symbol, ticker, Ticker."
        )
    tickers_list = (
        dfu[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({'': pd.NA})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(tickers_list)

tickers = _read_universe(tickers_csv_path)
print(f"Universe loaded from {tickers_csv_path}: {len(tickers)} tickers")

# Shift used to correct 1–2 day “edge” dates (e.g., 04-01, 07-01, 10-01, 01-01)
ANCHOR_SHIFT_DAYS = 5  # was 2, changed to 5 for robustness

# === Process each ticker ===
missing_files = []
processed_count = 0

for ticker in tickers:
    file_path = os.path.join(input_dir, f"{ticker}.csv")
    if not os.path.exists(file_path):
        missing_files.append(ticker)
        continue

    try:
        df = pd.read_csv(file_path)

        # Ensure date is datetime and build Year_Quarter
        df['date'] = pd.to_datetime(df['date'])
        anchor = df['date'] - pd.Timedelta(days=ANCHOR_SHIFT_DAYS)

        df['calendarYear'] = anchor.dt.year
        df['period'] = anchor.dt.quarter.apply(lambda x: f"Q{x}")
        df['Year_Quarter'] = df['calendarYear'].astype(str) + df['period']

        # Add required "year" and "quarter" columns
        df['year'] = df['calendarYear']
        df['quarter'] = df['period']

        # ----- Downscale dollar-valued metrics to millions -----
        dollar_cols = [
            'revenue', 'costOfRevenue', 'grossProfit',
            'researchAndDevelopmentExpenses', 'sellingGeneralAndAdministrativeExpenses',
            'operatingExpenses', 'ebitda', 'operatingIncome',
            'incomeBeforeTax', 'netIncome', 'totalAssets', 'totalEquity'
        ]
        for col in dollar_cols:
            if col in df.columns:
                df[col] = df[col] / 1_000_000

        # Standardize names
        df = df.rename(columns={
            'researchAndDevelopmentExpenses': 'rnd',
            'sellingGeneralAndAdministrativeExpenses': 'snaExpenses'
        })

        # ----- Map FMP sector -> GICS -----
        if 'sector' in df.columns:
            fmp_series = df['sector'].astype(str).str.strip()
            df['gics_sectors'] = fmp_series.map(fmp_to_gics)
            df['gics_sector_code'] = fmp_series.map(fmp_to_code)
            # If not mapped, fall back to original FMP name in gics_sectors
            df['gics_sectors'] = df['gics_sectors'].fillna(fmp_series)
        else:
            df['gics_sectors'] = pd.NA
            df['gics_sector_code'] = pd.NA

        # ----- Select output columns -----
        keep_cols = ['Year_Quarter', 'year', 'quarter']

        # Core targets/known features
        for c in [
            'revenue', 'costOfRevenue', 'grossProfit', 'rnd', 'snaExpenses',
            'operatingExpenses', 'ebitda', 'operatingIncome', 'incomeBeforeTax',
            'netIncome', 'totalAssets', 'totalEquity'
        ]:
            if c in df.columns:
                keep_cols.append(c)

        # New sector columns (string / code). Keep but do NOT require for dropna.
        keep_cols.extend(['gics_sectors', 'gics_sector_code'])

        # Build output; drop rows only if missing essential fields
        essential_cols = [c for c in keep_cols if c not in ['gics_sectors', 'gics_sector_code']]
        df_out = df[keep_cols].dropna(subset=essential_cols)

        # Save processed file
        out_path = os.path.join(output_dir, f"{ticker}_processed.csv")
        df_out.to_csv(out_path, index=False)
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} files... latest: {ticker}")

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

print(f"Done. Processed: {processed_count} | Missing source files: {len(missing_files)}")
if missing_files:
    # Save a quick log to help you fetch any missing raw CSVs
    pd.Series(missing_files, name="missing_tickers").to_csv(
        os.path.join(output_dir, "_missing_source_files.csv"), index=False
    )
    print(f"Missing tickers list saved to {os.path.join(output_dir, '_missing_source_files.csv')}")

