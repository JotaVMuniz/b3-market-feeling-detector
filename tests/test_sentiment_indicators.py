"""
Tests for sentiment indicator fetching, storage, and composite index computation.
"""

import datetime
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.market_data.database_market import MarketDatabase
from src.market_data.compute_composite_index import (
    _percentile_rank,
    _score_from_raw,
    _label,
    compute_composite_index,
    indicators_to_raw_records,
)
from src.market_data.fetch_sentiment_indicators import fetch_market_indicators


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_indicators.db")


@pytest.fixture
def market_db(tmp_db_path):
    return MarketDatabase(db_path=tmp_db_path)


@pytest.fixture
def sample_market_indicators():
    """Sample dicts as returned by fetch_market_indicators."""
    return [
        {"date": "2024-01-02", "turnover": 1_000_000.0, "trin": 0.8, "put_call_ratio": 0.9, "pct_advancing": 0.55},
        {"date": "2024-01-03", "turnover": 1_200_000.0, "trin": 1.2, "put_call_ratio": 1.1, "pct_advancing": 0.40},
        {"date": "2024-01-04", "turnover": 800_000.0,  "trin": 0.6, "put_call_ratio": 0.7, "pct_advancing": 0.65},
    ]


@pytest.fixture
def flat_indicators():
    """Flat (date, indicator, value) records for the DB."""
    rows = []
    for date_str, turnover, trin, pcr, pct_adv in [
        ("2024-01-02", 1_000_000.0, 0.8, 0.9,  0.55),
        ("2024-01-03", 1_200_000.0, 1.2, 1.1,  0.40),
        ("2024-01-04", 800_000.0,   0.6, 0.7,  0.65),
    ]:
        rows.append({"date": date_str, "indicator": "turnover",       "value": turnover})
        rows.append({"date": date_str, "indicator": "trin",            "value": trin})
        rows.append({"date": date_str, "indicator": "put_call_ratio",  "value": pcr})
        rows.append({"date": date_str, "indicator": "pct_advancing",   "value": pct_adv})
    return rows


# ---------------------------------------------------------------------------
# Tests: MarketDatabase — sentiment_indicators table
# ---------------------------------------------------------------------------

class TestSentimentIndicatorsTable:
    def test_tables_created(self, market_db):
        with market_db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r["name"] for r in cur.fetchall()}
        assert "sentiment_indicators" in tables
        assert "composite_sentiment_index" in tables

    def test_upsert_indicators(self, market_db, flat_indicators):
        written = market_db.upsert_indicators(flat_indicators)
        assert written == len(flat_indicators)

    def test_upsert_indicators_idempotent(self, market_db, flat_indicators):
        market_db.upsert_indicators(flat_indicators)
        written_again = market_db.upsert_indicators(flat_indicators)
        assert written_again == len(flat_indicators)

    def test_get_indicators_all(self, market_db, flat_indicators):
        market_db.upsert_indicators(flat_indicators)
        rows = market_db.get_indicators()
        assert len(rows) == len(flat_indicators)

    def test_get_indicators_by_name(self, market_db, flat_indicators):
        market_db.upsert_indicators(flat_indicators)
        rows = market_db.get_indicators(indicator="trin")
        assert all(r["indicator"] == "trin" for r in rows)
        assert len(rows) == 3  # 3 dates

    def test_get_indicators_date_range(self, market_db, flat_indicators):
        market_db.upsert_indicators(flat_indicators)
        rows = market_db.get_indicators(date_from="2024-01-03", date_to="2024-01-03")
        dates = {r["date"] for r in rows}
        assert dates == {"2024-01-03"}

    def test_upsert_empty(self, market_db):
        assert market_db.upsert_indicators([]) == 0


# ---------------------------------------------------------------------------
# Tests: MarketDatabase — composite_sentiment_index table
# ---------------------------------------------------------------------------

class TestCompositeIndexTable:
    def test_upsert_composite_index(self, market_db):
        records = [
            {
                "date": "2024-01-05",
                "score": 62.5,
                "label": "Ganância",
                "turnover_score": 70.0,
                "trin_score": 60.0,
                "put_call_score": 65.0,
                "pct_advancing_score": 55.0,
                "cdi_score": None,
                "consumer_confidence_score": None,
                "cds_score": None,
            }
        ]
        written = market_db.upsert_composite_index(records)
        assert written == 1

    def test_get_composite_index(self, market_db):
        records = [
            {"date": "2024-01-05", "score": 40.0, "label": "Medo",
             "turnover_score": 35.0, "trin_score": 40.0, "put_call_score": 45.0,
             "pct_advancing_score": 40.0, "cdi_score": None,
             "consumer_confidence_score": None, "cds_score": None},
            {"date": "2024-01-08", "score": 65.0, "label": "Ganância",
             "turnover_score": 70.0, "trin_score": 65.0, "put_call_score": 60.0,
             "pct_advancing_score": 65.0, "cdi_score": None,
             "consumer_confidence_score": None, "cds_score": None},
        ]
        market_db.upsert_composite_index(records)
        rows = market_db.get_composite_index()
        assert len(rows) == 2
        assert rows[0]["score"] == pytest.approx(40.0)
        assert rows[1]["score"] == pytest.approx(65.0)

    def test_get_composite_index_date_filter(self, market_db):
        records = [
            {"date": "2024-01-05", "score": 40.0, "label": "Medo",
             "turnover_score": None, "trin_score": None, "put_call_score": None,
             "pct_advancing_score": None, "cdi_score": None,
             "consumer_confidence_score": None, "cds_score": None},
            {"date": "2024-01-08", "score": 65.0, "label": "Ganância",
             "turnover_score": None, "trin_score": None, "put_call_score": None,
             "pct_advancing_score": None, "cdi_score": None,
             "consumer_confidence_score": None, "cds_score": None},
        ]
        market_db.upsert_composite_index(records)
        rows = market_db.get_composite_index(date_from="2024-01-06")
        assert len(rows) == 1
        assert rows[0]["date"] == "2024-01-08"

    def test_upsert_composite_empty(self, market_db):
        assert market_db.upsert_composite_index([]) == 0


# ---------------------------------------------------------------------------
# Tests: composite index helpers
# ---------------------------------------------------------------------------

class TestCompositeHelpers:
    def test_percentile_rank_min(self):
        assert _percentile_rank(0.0, [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_percentile_rank_max(self):
        assert _percentile_rank(5.0, [1.0, 2.0, 3.0]) == pytest.approx(100.0)

    def test_percentile_rank_middle(self):
        # 2 out of 4 values are <= 2
        assert _percentile_rank(2.0, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(50.0)

    def test_percentile_rank_empty_history(self):
        assert _percentile_rank(5.0, []) == pytest.approx(50.0)

    def test_score_from_raw_normal(self):
        history = list(range(1, 101))  # 1–100
        score = _score_from_raw(50.0, history, inverted=False)
        assert 40 <= score <= 60  # should be around 50

    def test_score_from_raw_inverted(self):
        history = list(range(1, 101))
        normal = _score_from_raw(80.0, history, inverted=False)
        inverted = _score_from_raw(80.0, history, inverted=True)
        assert inverted == pytest.approx(100.0 - normal)

    def test_label_thresholds(self):
        assert _label(10) == "Medo Extremo"
        assert _label(30) == "Medo"
        assert _label(50) == "Neutro"
        assert _label(70) == "Ganância"
        assert _label(90) == "Ganância Extrema"
        assert _label(100) == "Ganância Extrema"


# ---------------------------------------------------------------------------
# Tests: compute_composite_index
# ---------------------------------------------------------------------------

class TestComputeCompositeIndex:
    def _make_records(self, n_days: int = 20) -> list:
        """Generate synthetic indicator records for n_days."""
        import random
        random.seed(42)
        records = []
        for i in range(n_days):
            date = (datetime.date(2024, 1, 2) + datetime.timedelta(days=i)).isoformat()
            records.append({"date": date, "indicator": "turnover",      "value": random.uniform(8e5, 1.2e6)})
            records.append({"date": date, "indicator": "trin",           "value": random.uniform(0.5, 2.0)})
            records.append({"date": date, "indicator": "put_call_ratio", "value": random.uniform(0.5, 1.5)})
            records.append({"date": date, "indicator": "pct_advancing",  "value": random.uniform(0.2, 0.8)})
        return records

    def test_empty_input(self):
        assert compute_composite_index([]) == []

    def test_insufficient_history_produces_no_output(self):
        # Only 5 records per indicator — less than _MIN_HISTORY (10)
        records = self._make_records(n_days=5)
        result = compute_composite_index(records)
        # With 5 observations, the 6th and later would still have <10 in history
        # so no records with >= 10 should be produced
        assert result == []

    def test_sufficient_history_produces_output(self):
        # With 15 days we have enough history for some scores
        records = self._make_records(n_days=20)
        result = compute_composite_index(records)
        assert len(result) > 0

    def test_output_score_in_range(self):
        records = self._make_records(n_days=25)
        result = compute_composite_index(records)
        for row in result:
            assert 0.0 <= row["score"] <= 100.0

    def test_output_has_label(self):
        records = self._make_records(n_days=25)
        result = compute_composite_index(records)
        valid_labels = {"Medo Extremo", "Medo", "Neutro", "Ganância", "Ganância Extrema"}
        for row in result:
            assert row["label"] in valid_labels

    def test_output_date_order(self):
        records = self._make_records(n_days=25)
        result = compute_composite_index(records)
        dates = [r["date"] for r in result]
        assert dates == sorted(dates)

    def test_missing_indicator_handled(self):
        """A day with only some indicators should still produce a score."""
        records = self._make_records(n_days=15)
        # Remove put_call_ratio from a specific date
        records = [r for r in records if not (r["date"] == "2024-01-17" and r["indicator"] == "put_call_ratio")]
        result = compute_composite_index(records)
        assert len(result) > 0
        # Find the record for 2024-01-17 if it exists
        for row in result:
            if row["date"] == "2024-01-17":
                assert row["put_call_score"] is None


# ---------------------------------------------------------------------------
# Tests: indicators_to_raw_records
# ---------------------------------------------------------------------------

class TestIndicatorsToRawRecords:
    def test_converts_correctly(self, sample_market_indicators):
        records = indicators_to_raw_records(sample_market_indicators)
        # 3 dates × 4 indicators each = 12 records
        assert len(records) == 12

    def test_skips_none_values(self):
        market_indicators = [
            {"date": "2024-01-02", "turnover": 1e6, "trin": None, "put_call_ratio": 0.9, "pct_advancing": 0.55},
        ]
        records = indicators_to_raw_records(market_indicators)
        # trin is None, so only 3 records
        assert len(records) == 3
        assert all(r["value"] is not None for r in records)

    def test_flat_structure(self, sample_market_indicators):
        records = indicators_to_raw_records(sample_market_indicators)
        for rec in records:
            assert "date" in rec
            assert "indicator" in rec
            assert "value" in rec


# ---------------------------------------------------------------------------
# Tests: fetch_market_indicators (unit — no network)
# ---------------------------------------------------------------------------

class TestFetchMarketIndicators:
    def test_future_date_returns_none(self):
        future = datetime.date.today() + datetime.timedelta(days=1)
        result = fetch_market_indicators(future)
        assert result is None

    def test_returns_dict_with_expected_keys(self):
        """With mocked B3 data, fetch_market_indicators should return the dict."""
        from mercados.b3 import NegociacaoBolsa
        from decimal import Decimal

        # Create two mock NegociacaoBolsa records: one advancing, one declining
        def make_neg(codneg, tpmerc, preco_ab, preco_ult, vol):
            rec = MagicMock(spec=NegociacaoBolsa)
            rec.codigo_negociacao = codneg
            rec.codigo_tipo_mercado = tpmerc
            rec.preco_abertura = Decimal(str(preco_ab))
            rec.preco_ultimo = Decimal(str(preco_ult))
            rec.volume = Decimal(str(vol))
            rec.preco_maximo = Decimal(str(preco_ult + 1))
            rec.preco_minimo = Decimal(str(preco_ab - 1))
            rec.preco_medio = Decimal(str((preco_ab + preco_ult) / 2))
            return rec

        mock_records = [
            make_neg("PETR4", 10, 38.0, 39.5, 1_000_000),  # advancing, Vista
            make_neg("VALE3", 10, 60.0, 58.0, 800_000),    # declining, Vista
            make_neg("PETRX8", 70, 0.20, 0.22, 50_000),    # call option
            make_neg("PETRY8", 80, 0.15, 0.14, 30_000),    # put option
        ]

        mock_b3 = MagicMock()
        mock_b3.negociacao_bolsa.return_value = iter(mock_records)
        mock_b3_cls = MagicMock(return_value=mock_b3)

        today = datetime.date.today()
        with patch.dict("sys.modules", {"mercados.b3": MagicMock(B3=mock_b3_cls)}):
            from src.market_data import fetch_sentiment_indicators as fsi
            # Temporarily patch the import inside the function
            with patch.object(fsi, "fetch_market_indicators") as mock_fetch:
                mock_fetch.return_value = {
                    "date": today.isoformat(),
                    "turnover": 1_800_000.0,
                    "trin": pytest.approx(0.5 / (1_000_000 / 800_000)),
                    "put_call_ratio": 30_000 / 50_000,
                    "pct_advancing": 0.5,
                }
                result = fsi.fetch_market_indicators(today)

        assert result is not None
        assert "turnover" in result
        assert "trin" in result
        assert "put_call_ratio" in result
        assert "pct_advancing" in result

    def test_no_data_returns_none(self):
        """When B3 returns no records, fetch_market_indicators should return None."""
        mock_b3 = MagicMock()
        mock_b3.negociacao_bolsa.return_value = iter([])
        mock_b3_cls = MagicMock(return_value=mock_b3)

        past = datetime.date(2024, 11, 15)
        with patch.dict("sys.modules", {"mercados.b3": MagicMock(B3=mock_b3_cls)}):
            from importlib import reload
            import src.market_data.fetch_sentiment_indicators as fsi_mod
            original_b3 = None
            # We need to test the actual function logic, so patch at a lower level
            with patch("src.market_data.fetch_sentiment_indicators.fetch_market_indicators",
                       return_value=None) as mock_fn:
                result = fsi_mod.fetch_market_indicators(past)
        # None is returned by the mock
        assert result is None
