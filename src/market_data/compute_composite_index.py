"""
Compute the composite market sentiment index from raw indicators.

The composite index is modelled after the Fear & Greed Index and ranges from
0 (extreme fear) to 100 (extreme greed).  Each raw indicator is first
normalised to a 0-100 scale using a rolling window percentile rank (similar
to CNN's methodology), and then the individual scores are combined with equal
weights.

Indicator polarity mapping (higher raw value → higher score means more greed):

+----------------------+---------------------------------------------+
| Indicator            | Polarity                                    |
+======================+=============================================+
| turnover             | positive (high volume = optimism / greed)   |
+----------------------+---------------------------------------------+
| trin                 | negative (TRIN > 1 = bearish = fear)        |
+----------------------+---------------------------------------------+
| put_call_ratio       | negative (PCR > 1 = more puts = fear)       |
+----------------------+---------------------------------------------+
| pct_advancing        | positive (more stocks up = greed)           |
+----------------------+---------------------------------------------+
| cdi_rate             | negative (high rates = restrictive = fear)  |
+----------------------+---------------------------------------------+
| consumer_confidence  | positive (higher confidence = greed)        |
+----------------------+---------------------------------------------+
| cds_brasil_5y        | negative (higher CDS = more risk = fear)    |
+----------------------+---------------------------------------------+

Label thresholds (same as standard Fear & Greed):

- 0 – 20:  Medo Extremo
- 21 – 40: Medo
- 41 – 60: Neutro
- 61 – 80: Ganância
- 81 – 100: Ganância Extrema
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weights assigned to each component.  All weights must sum to 1.0.
# Components with no data for a given date are excluded and the remaining
# weights are re-normalised on the fly.
_WEIGHTS: Dict[str, float] = {
    "turnover": 0.15,
    "trin": 0.20,
    "put_call_ratio": 0.20,
    "pct_advancing": 0.20,
    "cdi_rate": 0.10,
    "consumer_confidence": 0.10,
    "cds_brasil_5y": 0.05,
}

# Indicators whose score is inverted before weighting (high raw = fear)
_INVERTED: set = {"trin", "put_call_ratio", "cdi_rate", "cds_brasil_5y"}

# Score labels
_LABELS = [
    (20, "Medo Extremo"),
    (40, "Medo"),
    (60, "Neutro"),
    (80, "Ganância"),
    (100, "Ganância Extrema"),
]

# Minimum historical records required for percentile normalisation
_MIN_HISTORY = 10

# Rolling window size (number of past observations) for percentile rank
_WINDOW = 252  # approximately 1 trading year


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _percentile_rank(value: float, history: List[float]) -> float:
    """
    Return the percentile rank of *value* within *history* (0.0 – 100.0).

    Percentile rank = (number of values in history that are ≤ value) /
                      len(history) * 100.

    Args:
        value:   The observation to rank.
        history: List of historical observations (need not be sorted).

    Returns:
        Float in [0.0, 100.0].
    """
    if not history:
        return 50.0
    count_lte = sum(1 for h in history if h <= value)
    return count_lte / len(history) * 100.0


def _score_from_raw(
    raw: float,
    history: List[float],
    inverted: bool = False,
) -> float:
    """
    Normalise a raw indicator value to a 0-100 sentiment score.

    Args:
        raw:      Current value of the indicator.
        history:  Past values (used to compute percentile rank).
        inverted: When True, higher raw value → lower (more fearful) score.

    Returns:
        Sentiment score in [0.0, 100.0].
    """
    score = _percentile_rank(raw, history)
    if inverted:
        score = 100.0 - score
    return score


def _label(score: float) -> str:
    """Return the sentiment label for *score* (0-100)."""
    for threshold, label in _LABELS:
        if score <= threshold:
            return label
    return "Ganância Extrema"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_composite_index(
    raw_indicators: List[Dict],
) -> List[Dict]:
    """
    Compute the composite sentiment index from a list of raw indicator records.

    The records should cover a sufficiently long historical window so that the
    percentile rank normalisation is meaningful.  Each record is a dict with
    keys ``date`` (ISO string), ``indicator`` (name), and ``value`` (float).

    Args:
        raw_indicators: List of raw indicator dicts as returned by
            :func:`~src.market_data.fetch_sentiment_indicators.fetch_bcb_indicators`
            or converted from
            :func:`~src.market_data.fetch_sentiment_indicators.fetch_market_indicators`.

    Returns:
        List of composite index dicts, one per date, ready to be stored via
        :meth:`~src.market_data.database_market.MarketDatabase.upsert_composite_index`.
    """
    if not raw_indicators:
        return []

    # ------------------------------------------------------------------
    # 1. Pivot: {date: {indicator: value}}
    # ------------------------------------------------------------------
    pivot: Dict[str, Dict[str, float]] = {}
    for rec in raw_indicators:
        date = rec.get("date")
        indicator = rec.get("indicator")
        value = rec.get("value")
        if date is None or indicator is None or value is None:
            continue
        pivot.setdefault(date, {})[indicator] = value

    sorted_dates = sorted(pivot.keys())

    # ------------------------------------------------------------------
    # 2. Build rolling history per indicator
    # ------------------------------------------------------------------
    history: Dict[str, List[float]] = {k: [] for k in _WEIGHTS}

    results: List[Dict] = []

    for date in sorted_dates:
        day_values = pivot[date]

        # Compute per-indicator scores
        component_scores: Dict[str, Optional[float]] = {}
        for indicator in _WEIGHTS:
            raw = day_values.get(indicator)
            if raw is None:
                component_scores[indicator] = None
                continue
            hist = history[indicator][-_WINDOW:]
            if len(hist) < _MIN_HISTORY:
                component_scores[indicator] = None
            else:
                component_scores[indicator] = _score_from_raw(
                    raw, hist, inverted=(indicator in _INVERTED)
                )

        # Update history after scoring (so current value is not included in
        # its own percentile calculation)
        for indicator in _WEIGHTS:
            raw = day_values.get(indicator)
            if raw is not None:
                history[indicator].append(raw)

        # Re-normalise weights for components with data
        available = {k: w for k, w in _WEIGHTS.items() if component_scores.get(k) is not None}
        if not available:
            continue

        total_weight = sum(available.values())
        composite_score = sum(
            component_scores[k] * w / total_weight
            for k, w in available.items()
        )

        results.append({
            "date": date,
            "score": round(composite_score, 2),
            "label": _label(composite_score),
            "turnover_score": component_scores.get("turnover"),
            "trin_score": component_scores.get("trin"),
            "put_call_score": component_scores.get("put_call_ratio"),
            "pct_advancing_score": component_scores.get("pct_advancing"),
            "cdi_score": component_scores.get("cdi_rate"),
            "consumer_confidence_score": component_scores.get("consumer_confidence"),
            "cds_score": component_scores.get("cds_brasil_5y"),
        })

    logger.info("Computed composite index for %d dates", len(results))
    return results


def indicators_to_raw_records(market_indicators: List[Dict]) -> List[Dict]:
    """
    Convert the dicts returned by
    :func:`~src.market_data.fetch_sentiment_indicators.fetch_market_indicators`
    into the flat ``(date, indicator, value)`` format expected by
    :meth:`~src.market_data.database_market.MarketDatabase.upsert_indicators`
    and :func:`compute_composite_index`.

    Args:
        market_indicators: List of dicts with keys ``date``, ``turnover``,
            ``trin``, ``put_call_ratio``, ``pct_advancing``.

    Returns:
        List of flat dicts with keys ``date``, ``indicator``, ``value``.
    """
    records: List[Dict] = []
    for row in market_indicators:
        date = row.get("date")
        if not date:
            continue
        for key in ("turnover", "trin", "put_call_ratio", "pct_advancing"):
            value = row.get(key)
            if value is not None:
                records.append({"date": date, "indicator": key, "value": value})
    return records
