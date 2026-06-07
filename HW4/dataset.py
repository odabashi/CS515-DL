"""
dataset.py

Data download, preprocessing, and PyTorch Dataset classes

Imports structural constants (T, D, FEATURES, ...) directly from parameters.py.
Tunable values (batch_size, tickers) arrive as function arguments so callers
can pass them straight from the cfg dict returned by get_params().
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from parameters import (FEATURES, START_DATE, END_DATE, TRAIN_END, VAL_END, T, D, ROLLING_WINDOW, ROLLING_WEIGHTS,
                        GAMMA)

# Default tickers used when the caller does not supply them.
# In a full run main.py passes cfg["tickers"] explicitly.
_DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]


# =============================================================================
# Download & split raw data
# =============================================================================

def _generate_synthetic_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fallback: generate realistic-looking OHLC data with a geometric Brownian
    motion (GBM) model. Used automatically when yfinance is unavailable (e.g. in sandboxed / offline environments).

    GBM recap: price evolves as  S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1). This is the standard Black-Scholes price model.

    NOTE: On our local machine with internet access, the real yfinance data will be used instead. This function
    is never called in that case.
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2 ** 31))  # ticker-specific seed

    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    # Starting prices differ per ticker to make the data look distinct.
    s0 = {"AAPL": 150.0, "MSFT": 200.0, "GOOGL": 2800.0}.get(ticker, 100.0)
    mu = 0.10 / 252  # ~10 % annual drift, scaled to daily
    sigma = 0.20 / np.sqrt(252)  # ~20 % annual volatility, scaled to daily

    # Simulate closing prices
    shocks = rng.standard_normal(n)
    log_returns = (mu - 0.5 * sigma ** 2) + sigma * shocks
    close = s0 * np.exp(np.cumsum(log_returns))

    # Derive Open / High / Low from Close with realistic intraday noise
    daily_range = close * rng.uniform(0.005, 0.025, n)  # 0.5 – 2.5 % daily range
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    high = np.maximum(close, open_) + rng.uniform(0, 1, n) * daily_range
    low = np.minimum(close, open_) - rng.uniform(0, 1, n) * daily_range

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=dates
    )
    return df.astype(np.float32)


def _load_clean_csv(path: str) -> pd.DataFrame | None:
    """
    Load a cached OHLC CSV and return a clean DataFrame, or None if the file is in the stale multi-index format
    written by newer yfinance.

    Newer yfinance (>=0.2) serializes its MultiIndex columns so that the CSV looks like:
        Price,Open,High,Low,Close      - column header row (index name = 'Price')
        Ticker,AAPL,AAPL,AAPL,AAPL     - ticker metadata row
        Date,,,,                       - date label row
        2020-01-02,296.24,...          - actual data starts here

    Reading this with index_col=0 makes the first index value the string 'Ticker' rather than a date.
    We detect that and return None so the caller knows to discard the file and re-download.
    """
    raw = pd.read_csv(path, index_col=0)

    # Stale multi-index CSV: first index value is a metadata string, not a date
    if len(raw) == 0 or str(raw.index[0]).strip() in ("Ticker", "Date"):
        return None

    # Clean CSV: convert index to DatetimeIndex and sort
    raw.index = pd.to_datetime(raw.index, format="%Y-%m-%d", errors="coerce")
    raw = raw[raw.index.notna()]  # drop any rows that failed to parse
    raw.sort_index(inplace=True)
    raw = raw[FEATURES]  # keep only the four OHLC columns
    return raw


def _fetch_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLC data for one ticker and return a clean single-level DataFrame.

    Handles the MultiIndex columns that newer yfinance returns by flattening them before any column selection,
    so the CSV written to disk is always in the simple format that _load_clean_csv expects.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Newer yfinance (>=0.2) returns MultiIndex columns like ('Close', 'AAPL').
    # When saved to CSV as-is, pandas writes extra 'Ticker' / 'Date' header rows that break a plain read_csv.
    # Flatten immediately after download.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[FEATURES].copy()
    df.dropna(inplace=True)

    if len(df) == 0:
        raise ValueError("yfinance returned an empty DataFrame")

    return df


def download_data(tickers=None, start=START_DATE, end=END_DATE, data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """
    Download daily OHLC data from Yahoo Finance for each ticker; cache to CSV.
    Falls back to synthetic GBM data if yfinance is unavailable.

    Stale CSVs written by older versions of this code (multi-index format)
    are detected automatically, deleted, and re-downloaded.

    Args:
        tickers: list of ticker strings; defaults to _DEFAULT_TICKERS
        start: start date for data
        end: end date for data
        data_dir: directory for CSV cache files

    Returns:
        dict: {ticker: pd.DataFrame with columns [Open, High, Low, Close]}
    """
    tickers = tickers or _DEFAULT_TICKERS

    os.makedirs(data_dir, exist_ok=True)
    data = {}

    for ticker in tickers:
        cache_path = os.path.join(data_dir, f"{ticker}.csv")
        df = None

        if os.path.exists(cache_path):
            df = _load_clean_csv(cache_path)
            if df is not None:
                print(f"[cache] Loaded {ticker} from '{cache_path}' ({len(df)} rows)")
            else:
                # Stale multi-index format — delete and fall through to download
                print(f"[cache] {ticker}: stale CSV format detected; re-downloading")
                os.remove(cache_path)

        # Download (fresh or after stale-cache deletion)
        if df is None:
            print(f"[download] Fetching {ticker} from Yahoo Finance ...")
            try:
                df = _fetch_from_yfinance(ticker, start, end)
                df.to_csv(cache_path)
                print(f"[download] Saved {ticker} -> '{cache_path}' with {len(df)} rows")
            except Exception as e:
                print(f"[warning] yfinance failed for {ticker}: {e}")
                print(f"[fallback] Generating synthetic OHLC data for {ticker} ...")
                df = _generate_synthetic_ohlc(ticker, start, end)
                df.to_csv(cache_path)
                print(f"[fallback] Saved synthetic {ticker} -> '{cache_path}' ({len(df)} rows)")

        data[ticker] = df

    return data


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split.

    train: 	Jan 2020 - Jul 2024
    val: 	Aug 2024 - Dec 2024
    test: 	Jan 2025 - Dec 2025

    IMPORTANT: We never shuffle here. In time-series work, shuffling would cause data leakage. The model would "see
    the future" during training.
    """
    train = df.loc[:TRAIN_END]
    val = df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1): VAL_END]
    test = df.loc[pd.Timestamp(VAL_END) + pd.Timedelta(days=1):]
    return train, val, test


# =============================================================================
# Normalization
# =============================================================================

class Normalizer:
    """
    Per-stock Min-max normalization fitted ONLY on the training set.

    Why only the training set?
    --------------------------
    If we computed statistics (min, max) over the whole dataset including the future (val/test), the model would
    indirectly "see" future price levels during training and this is called data leakage and inflates performance.

    We store the per-feature min and max from training, then apply the same transformation to val and test.

    Formula:  x_norm = (x - min) / (max - min + eps)
    Inverse:  x = x_norm * (max - min + eps) + min
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        """Compute min/max from training data (call once on train split)."""
        self.min_ = df.values.min(axis=0)  # shape (F,)
        self.max_ = df.values.max(axis=0)  # shape (F,)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply normalization; returns numpy array of shape (T, F)."""
        return (df.values - self.min_) / (self.max_ - self.min_ + self.eps)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


# =============================================================================
# Target builders
# =============================================================================

def build_exact_targets(close: np.ndarray, t: int) -> np.ndarray:
    """
    Part (b): exact d-day return ratio for d = 1 ... D.

        y[d-1] = (close[t+d] - close[t]) / close[t]

    Args:
        close: 1-D array of raw (un-normalized) closing prices for one stock
        t: current time index (last day of the input window)

    Returns:
        np.ndarray of shape (D, )
    """
    p_t = close[t]
    targets = np.array([(close[t + d] - p_t) / p_t for d in range(1, D + 1)], dtype=np.float32)
    return targets


def build_rolling_targets(close: np.ndarray, t: int) -> np.ndarray:
    """
    Part (c): weighted rolling-average return ratio.

        ŷ[d-1] = (Σ_{j=0}^{l} w_j * close[t+d-j]  - close[t]) / close[t]

    The rolling average smooths the target, which can stabilize training
    (less noise in the loss gradient).

    Args:
        close: 1-D array of raw closing prices
        t: current time index
    """
    w = ROLLING_WEIGHTS.numpy()  # convert to numpy for vectorized ops
    p_t = close[t]
    targets = []
    for d in range(1, D + 1):
        # Weighted sum of close[t+d], close[t+d-1], ..., close[t+d-l+1]
        # Guard against going below index 0.
        prices = np.array([close[max(0, t + d - j)] for j in range(ROLLING_WINDOW)])
        avg = np.dot(w, prices)
        targets.append((avg - p_t) / p_t)
    return np.array(targets, dtype=np.float32)


def build_turning_point_targets(high: np.ndarray, close: np.ndarray, t: int) -> tuple[np.ndarray, int]:
    """
    Part (d): return ratios using the MAX (high) price, plus a binary label.

        r[d-1] = (high[t+d] - close[t]) / close[t]

    Binary label = 1 if ANY r[d-1] > gamma, else 0.
    This becomes the "BUY" signal.

    Note: we use 'high' for the numerator (maximum price on day t+d) and
    'close' at time t as the baseline.
    """
    p_t = close[t]
    returns = np.array([(high[t + d] - p_t) / p_t for d in range(1, D + 1)], dtype=np.float32)
    label = int(np.any(returns > GAMMA))  # 1 = BUY, 0 = PASS
    return returns, label


# =============================================================================
# PyTorch SlidingWindowDataset
# =============================================================================

class SlidingWindowDataset(Dataset):
    """
    Generates overlapping (input, target) pairs using a sliding window of length T.

    For each valid time index t (T-1 ≤ t ≤ len - D - 1):
        X = normalized features for days [t-T+1 ... t]   shape (T, F)
        y = return ratio targets for days [t+1 ... t+D]  shape (D,)

    The 'mode' argument selects which target builder to use:
        "exact": Part (b) exact d-day return ratios
        "rolling": Part (c) rolling-average return ratios
        "turning_point": Part (d) max-price returns + binary label
    """
    def __init__(self,
                 norm_features: np.ndarray,   # (N_days, F) normalized
                 raw_close: np.ndarray,        # (N_days,)  raw close prices
                 raw_high:  np.ndarray,        # (N_days,)  raw high prices
                 mode: str = "exact"):
        """
        Args:
            norm_features: Normalized OHLC array, shape (N_days, F). We use this as the model input X.
            raw_close:  Un-normalized closing prices (used to compute return ratios). Return ratios must be computed
                        on raw prices. If we used normalized prices the ratios would be distorted.
            raw_high: Un-normalized high prices (only used in turning_point mode).
            mode: One of {"exact", "rolling", "turning_point"}.
        """
        assert mode in {"exact", "rolling", "turning_point"}, \
            f"Unknown mode '{mode}'. Choose from: exact, rolling, turning_point."

        self.features = norm_features.astype(np.float32)
        # .flatten() converts (N, 1) to (N,) which newer yfinance may produce via MultiIndex columns as target builders
        # require a 1-D array.
        self.raw_close = raw_close.flatten().astype(np.float32)
        self.raw_high = raw_high.flatten().astype(np.float32)
        self.mode = mode

        # Valid starting positions: we need T days of history before t,
        # and D days of future after t.
        self.indices = list(range(T - 1, len(norm_features) - D))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]

        # Input window with shape: (T, F) where T: time steps, F: features.
        X = torch.tensor(self.features[t - T + 1 : t + 1])   # (T, F)

        # Target
        if self.mode == "exact":
            y = torch.tensor(build_exact_targets(self.raw_close, t))   # (D,)
            return X, y

        elif self.mode == "rolling":
            y = torch.tensor(build_rolling_targets(self.raw_close, t))  # (D,)
            return X, y

        else:  # turning_point
            returns, label = build_turning_point_targets(self.raw_high, self.raw_close, t)
            y_ret = torch.tensor(returns)                 # (D,)
            y_label = torch.tensor(label, dtype=torch.float32)  # scalar
            return X, y_ret, y_label


# =============================================================================
# build_dataloaders
# =============================================================================

def build_dataloaders(mode: str = "exact", batch_size: int = 64, tickers: list[str] | None = None,
                      data_dir: str = "data") -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download data for all tickers, build SlidingWindowDatasets for each split,
    concatenate across stocks into three ConcatDatasets (train / val / test), and wrap in DataLoaders.

    Each stock is normalized independently (its own Normalizer fitted on its own training slice) to avoid
    cross-stock leakage.

    In DataLoader we set shuffle=True on the training loader: although the windows come from a time series,
    each (X, y) window is a self-contained sample. Shuffling the windows across batches reduces gradient correlation
    and usually helps training. We never shuffle val/test.

    Args:
        mode: "exact" | "rolling" | "turning_point"
        batch_size: mini-batch size
        tickers: list of ticker symbols; defaults to _DEFAULT_TICKERS
        data_dir: CSV cache directory

    Returns:
        train_loader, val_loader, test_loader
    """
    tickers = tickers or _DEFAULT_TICKERS
    all_data = download_data(tickers=tickers, data_dir=data_dir)

    train_sets, val_sets, test_sets = [], [], []

    for ticker in tickers:
        df = all_data[ticker]

        # Chronological split
        train_df, val_df, test_df = split_data(df)

        # Fit normalizer on training slice only
        norm = Normalizer()
        train_norm = norm.fit_transform(train_df)
        val_norm = norm.transform(val_df)
        test_norm = norm.transform(test_df)

        def _ds(norm_arr, raw_df):
            return SlidingWindowDataset(
                norm_features=norm_arr,
                raw_close=raw_df["Close"].values,
                raw_high=raw_df["High"].values,
                mode=mode,
            )

        train_sets.append(_ds(train_norm, train_df))
        val_sets.append(_ds(val_norm, val_df))
        test_sets.append(_ds(test_norm, test_df))

        print(f"{ticker}: train={len(train_sets[-1])} val={len(val_sets[-1])} test={len(test_sets[-1])}")

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    test_ds = ConcatDataset(test_sets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 55)
    for m in ("exact", "rolling", "turning_point"):
        tl, vl, tel = build_dataloaders(mode=m, batch_size=64)
        batch = next(iter(tl))
        X = batch[0]
        print(f"mode={m:<15}  X={tuple(X.shape)}  "
              f"y={tuple(batch[1].shape)}  batches={len(tl)}")
    print("\n✓  dataset.py smoke test passed.")