"""Universe construction for the research panel.

The default research universe is the current S&P 500 membership list. A
snapshot ships with the package so runs are reproducible offline; passing
``refresh=True`` re-fetches the live membership and rewrites the snapshot.

Survivorship bias
-----------------
The snapshot records *current* membership, so applying it to a historical
window silently excludes companies that were dropped from the index during
that window. Backtests built on it are therefore biased upward, and the size
of that bias grows with the length of the sample. Point-in-time membership
data is the fix; treat results produced here as a prototype, not an estimate
of realisable performance.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

_RESOURCES = Path(__file__).parent / "resources"
_SNAPSHOT = _RESOURCES / "sp500_constituents.csv"
_COMPOSITE_SNAPSHOT = _RESOURCES / "sp1500_constituents.csv"
_SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_AGENT = "alpha-research/1.0 (research use)"

#: Index membership sources making up the S&P Composite 1500.
_COMPOSITE_SOURCES: tuple[tuple[str, str], ...] = (
    ("large", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    ("mid", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
    ("small", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
)

#: Compact mega-cap universe retained for fast smoke tests.
MEGA_CAP_UNIVERSE: tuple[str, ...] = (
    "AAPL", "ABBV", "ABT", "AMGN", "AMZN", "AVGO", "AXP", "BA", "BAC", "CAT",
    "COST", "CRM", "CSCO", "CVX", "DIS", "GOOGL", "GS", "HD", "HON", "IBM",
    "INTC", "JNJ", "JPM", "KO", "LIN", "LLY", "LOW", "MA", "MCD", "META",
    "MMM", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP",
    "PFE", "PG", "QCOM", "RTX", "T", "TMO", "TSLA", "UNH", "UNP", "UPS",
    "USB", "V", "VZ", "WFC", "WMT", "XOM",
)


@dataclass(frozen=True, slots=True)
class Universe:
    """A set of tickers with their GICS sector labels and size-tier membership."""

    tickers: tuple[str, ...]
    sectors: dict[str, str]
    size_tiers: dict[str, str] | None = None

    def __len__(self) -> int:
        return len(self.tickers)


def _fetch_live() -> pd.DataFrame:
    """Fetch current index membership from the public source table."""
    response = requests_get(_SOURCE_URL)
    table = pd.read_html(io.StringIO(response))[0]
    frame = pd.DataFrame(
        {
            "ticker": table["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False),
            "sector": table["GICS Sector"].astype(str).str.strip(),
        }
    )
    return frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def requests_get(url: str) -> str:
    """Thin wrapper so the network call is easy to stub in tests."""
    import requests

    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def refresh_snapshot() -> Path:
    """Re-fetch index membership and overwrite the bundled snapshot."""
    frame = _fetch_live()
    _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(_SNAPSHOT, index=False)
    load_snapshot.cache_clear()
    return _SNAPSHOT


@lru_cache(maxsize=1)
def load_snapshot() -> pd.DataFrame:
    """Load the bundled membership snapshot."""
    if not _SNAPSHOT.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(
            f"Universe snapshot missing at {_SNAPSHOT}. Run universe.refresh_snapshot()."
        )
    return pd.read_csv(_SNAPSHOT)


def sp500(refresh: bool = False) -> Universe:
    """Return the S&P 500 research universe.

    Parameters
    ----------
    refresh:
        When ``True``, re-fetch live membership before returning. Requires
        network access; the bundled snapshot is used otherwise.
    """
    if refresh:
        refresh_snapshot()
    frame = load_snapshot()
    return Universe(
        tickers=tuple(frame["ticker"]),
        sectors=dict(zip(frame["ticker"], frame["sector"])),
    )


def _fetch_composite() -> pd.DataFrame:
    """Fetch large, mid, and small cap membership and stack them into one frame."""
    frames = []
    for tier, url in _COMPOSITE_SOURCES:
        table = pd.read_html(io.StringIO(requests_get(url)))[0]
        symbol_column = "Symbol" if "Symbol" in table.columns else "Ticker"
        frames.append(
            pd.DataFrame(
                {
                    "ticker": table[symbol_column].astype(str).str.strip().str.replace(".", "-", regex=False),
                    "sector": table["GICS Sector"].astype(str).str.strip(),
                    "size_tier": tier,
                }
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def refresh_composite_snapshot() -> Path:
    """Re-fetch S&P Composite 1500 membership and overwrite the bundled snapshot."""
    frame = _fetch_composite()
    _COMPOSITE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(_COMPOSITE_SNAPSHOT, index=False)
    load_composite_snapshot.cache_clear()
    return _COMPOSITE_SNAPSHOT


@lru_cache(maxsize=1)
def load_composite_snapshot() -> pd.DataFrame:
    """Load the bundled S&P Composite 1500 membership snapshot."""
    if not _COMPOSITE_SNAPSHOT.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(
            f"Composite snapshot missing at {_COMPOSITE_SNAPSHOT}. "
            "Run universe.refresh_composite_snapshot()."
        )
    return pd.read_csv(_COMPOSITE_SNAPSHOT)


def sp1500(refresh: bool = False) -> Universe:
    """Return the S&P Composite 1500 research universe.

    Roughly three times the name count of the S&P 500 alone, and the addition is
    entirely mid and small cap. That matters for a cross-sectional study: the
    large-cap segment is the most efficiently priced part of the market, and both
    the breadth term in the fundamental law and the documented strength of most
    cross-sectional effects improve markedly once smaller names are included.
    The trade-off is execution cost, which rises as liquidity falls, so results
    on this universe must be read against the cost sensitivity analysis.
    """
    if refresh:
        refresh_composite_snapshot()
    frame = load_composite_snapshot()
    return Universe(
        tickers=tuple(frame["ticker"]),
        sectors=dict(zip(frame["ticker"], frame["sector"])),
        size_tiers=dict(zip(frame["ticker"], frame["size_tier"])),
    )


def mega_cap() -> Universe:
    """Return the compact mega-cap universe used for smoke tests.

    Sector labels are sourced from the S&P 500 snapshot where available.
    """
    sectors = sp500().sectors
    return Universe(
        tickers=MEGA_CAP_UNIVERSE,
        sectors={t: sectors.get(t, "Unknown") for t in MEGA_CAP_UNIVERSE},
    )
