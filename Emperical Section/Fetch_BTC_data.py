"""


Extraction and transformation of the Bitcoin daily return series for the
BLADE empirical illustration.

Pipeline
--------
    1. Download daily close prices (auto-adjusted).
    2. Log-returns r_t = 100 * (log P_t - log P_{t-1}).
    3. Scale by 100 (returns in percent)
    4. Median-centre (robust to the very outliers under study; cf. Muler &
       Yohai 2008, who centre by the median for the same reason).
    5. Drop non-trading days / NaNs (no interpolation -- interpolation invents
       artificial low-volatility days and biases the variance dynamics).

Outputs
-------
    blade_returns_Bitcoin.csv    tidy file (date, ret)

Usage
-----
    pip install yfinance pandas numpy
    python blade_extract_data.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

START = "2000-01-01"
END = "2024-12-31"

TICKER = "BTC-USD"
NAME = "Bitcoin"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def fetch_prices(ticker):
    """Daily auto-adjusted close prices as a Series, or None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        print("  ERROR: yfinance not installed (pip install yfinance).")
        return None
    try:
        df = yf.download(ticker, start=START, end=END,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"  WARNING: no data returned for {ticker}.")
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.name = ticker
        return close
    except Exception as exc:
        print(f"  ERROR downloading {ticker}: {exc!r}")
        return None


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #

def to_returns(price):
    """
    Apply the minimal transformation pipeline (log-returns, %, median-centred).
    Returns (ret_series_percent_median_centred, info_dict).
    """
    price = price.dropna()
    n_price = len(price)

    ret = 100.0 * np.log(price).diff()
    ret = ret.dropna()                 # drop the first (undefined) return + NaNs
    median = float(ret.median())
    ret_centred = ret - median         # robust centring

    info = {
        "n_price": n_price,
        "n_returns": len(ret_centred),
        "median_subtracted": median,
        "start": ret_centred.index[0].date().isoformat(),
        "end": ret_centred.index[-1].date().isoformat(),
    }
    ret_centred.name = "ret"
    return ret_centred, info


def describe(ret):
    """Descriptive stats; the fat-tail columns motivate a robust gamma<2."""
    r = ret.values.astype(float)
    mu, sd = r.mean(), r.std(ddof=1)
    return {
        "n": len(r),
        "mean": mu,                       # ~0 by construction (median-centred)
        "sd": sd,
        "min": r.min(),
        "max": r.max(),
        "skew": ((r - mu) ** 3).mean() / sd ** 3,
        "ex_kurt": ((r - mu) ** 4).mean() / sd ** 4 - 3.0,
        ">5sd_%": 100.0 * np.mean(np.abs(r - mu) > 5 * sd),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print(f"\n{NAME} ({TICKER}) [log returns]")
    price = fetch_prices(TICKER)
    if price is None or len(price) < 50:
        print("  -> no usable data, aborting.")
        return 1

    ret, info = to_returns(price)
    print(f"  prices: {info['n_price']}")
    print(f"  returns: {info['n_returns']} log, "
          f"{info['start']} -> {info['end']}")
    print(f"  median subtracted: {info['median_subtracted']:.4f}")

    out = f"blade_returns_{NAME}.csv"
    ret.to_frame().to_csv(out, index_label="date")
    print(f"  saved -> {out}")

    stats = describe(ret)
    pd.set_option("display.float_format", lambda x: f"{x:8.3f}")
    print("\n" + "=" * 76)
    print("Descriptive statistics of (median-centred, %) returns")
    print("=" * 76)
    print(pd.Series(stats).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())