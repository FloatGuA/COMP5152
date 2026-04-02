import pandas as pd
from pathlib import Path
from src.data_cleaner import clean_series

DATA_DIR = Path("C:/Coding/COMP5152/selected_data")

SELECTED = [
    # Original 6
    ("stocks", "CLI"),
    ("stocks", "ALCO"),
    ("stocks", "ACCO"),
    ("etfs",   "DWM"),
    ("etfs",   "CHII"),
    ("etfs",   "BND"),
    # Extended set — diverse sectors
    ("stocks", "AAPL"),   # Technology
    ("stocks", "AMD"),    # Semiconductor
    ("stocks", "AMAT"),   # Semiconductor equipment
    ("stocks", "ADBE"),   # Software
    ("stocks", "ADP"),    # Business services
    ("stocks", "ABT"),    # Healthcare
    ("stocks", "AMGN"),   # Biotech
    ("stocks", "AFL"),    # Insurance
    # AIG excluded: catastrophic 2008 crisis (~$70→$1), severe distribution shift
    # AON excluded: anomalous results, removing from experiment
    ("stocks", "AEP"),    # Utilities
    ("stocks", "AN"),     # Auto retail
    ("etfs",   "BIV"),    # Intermediate bond
    ("etfs",   "ACWX"),   # International equity ex-US
]


def load_close(category: str, symbol: str) -> pd.Series:
    path = DATA_DIR / category / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    series = df["Close"].sort_index().dropna()
    series.name = symbol
    series = clean_series(series)
    return series


def train_val_test_split(
    series: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(series)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return series.iloc[:train_end], series.iloc[train_end:val_end], series.iloc[val_end:]
