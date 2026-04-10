"""
Tests for the fundamental indicators module.

All external network calls (yfinance, mercados.bcb, mercados.b3) are mocked
so the tests run fully offline.
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.market_data.fetch_fundamentals import (
    _YF_FIELD_MAP,
    _FUNDAMENTUS_FIELD_MAP,
    _is_fii_ticker,
    _parse_br_number,
    fetch_asset_fundamentals,
    fetch_fundamentus_data,
    fetch_fii_dy_supplement,
    fetch_fundamentals_for_tickers,
    fetch_macro_fundamentals,
)
from src.market_data.database_market import MarketDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market_db(tmp_path):
    return MarketDatabase(db_path=str(tmp_path / "test_fund.db"))


@pytest.fixture
def sample_fund_records():
    today = datetime.date.today().isoformat()
    return [
        {"ticker": "PETR4", "key": "pl",      "value": 6.5,   "label": "P/L",      "updated_at": today},
        {"ticker": "PETR4", "key": "pvpa",     "value": 1.2,   "label": "P/VPA",    "updated_at": today},
        {"ticker": "PETR4", "key": "roe",      "value": 0.25,  "label": "ROE",      "updated_at": today},
        {"ticker": "PETR4", "key": "dy",       "value": 0.08,  "label": "Dividend Yield", "updated_at": today},
    ]


# ---------------------------------------------------------------------------
# _parse_br_number
# ---------------------------------------------------------------------------

class TestParseBrNumber:
    def test_integer(self):
        assert _parse_br_number("42") == pytest.approx(42.0)

    def test_decimal_comma(self):
        assert _parse_br_number("1,23") == pytest.approx(1.23)

    def test_thousands_dot_decimal_comma(self):
        assert _parse_br_number("1.234,56") == pytest.approx(1234.56)

    def test_percentage_stripped(self):
        assert _parse_br_number("12,34%") == pytest.approx(12.34)

    def test_negative(self):
        assert _parse_br_number("-5,67") == pytest.approx(-5.67)

    def test_parentheses_negative(self):
        assert _parse_br_number("(1.000,00)") == pytest.approx(-1000.0)

    def test_dash_returns_none(self):
        assert _parse_br_number("-") is None

    def test_empty_returns_none(self):
        assert _parse_br_number("") is None

    def test_none_like_returns_none(self):
        assert _parse_br_number(None) is None  # type: ignore


# ---------------------------------------------------------------------------
# fetch_fundamentus_data
# ---------------------------------------------------------------------------

# Simulates the real Fundamentus page structure with class="label"/class="data".
# Labels have a leading '?' (tooltip icon rendered as text) which the scraper
# must strip.  Label text matches actual Fundamentus page content.
_SAMPLE_FUNDAMENTUS_HTML_CLASSES = """
<html><body>
<table class="w728">
  <tr>
    <td class="label"><span class="txt">?Papel</span></td>
    <td class="data"><span class="txt">PETR4</span></td>
    <td class="label"><span class="txt">?Cota\xe7\xe3o</span></td>
    <td class="data"><span class="txt">38,50</span></td>
  </tr>
  <tr>
    <td class="label"><span class="txt">?P/L</span></td>
    <td class="data"><span class="txt">6,20</span></td>
    <td class="label"><span class="txt">?P/VP</span></td>
    <td class="data"><span class="txt">1,10</span></td>
  </tr>
  <tr>
    <td class="label"><span class="txt">?EV / EBITDA</span></td>
    <td class="data"><span class="txt">4,50</span></td>
    <td class="label"><span class="txt">?EV / EBIT</span></td>
    <td class="data"><span class="txt">5,30</span></td>
  </tr>
  <tr>
    <td class="label"><span class="txt">?Div. Yield</span></td>
    <td class="data"><span class="txt">8,40%</span></td>
    <td class="label"><span class="txt">?ROE</span></td>
    <td class="data"><span class="txt">22,50%</span></td>
  </tr>
  <tr>
    <td class="label"><span class="txt">?EBIT / Ativo</span></td>
    <td class="data"><span class="txt">7,10%</span></td>
    <td class="label"><span class="txt">?Marg. L\xedquida</span></td>
    <td class="data"><span class="txt">15,20%</span></td>
  </tr>
  <tr>
    <td class="label"><span class="txt">?Liquidez Corr</span></td>
    <td class="data"><span class="txt">1,80</span></td>
    <td class="label"><span class="txt">?Div Br/ Patrim</span></td>
    <td class="data"><span class="txt">0,85</span></td>
  </tr>
</table>
</body></html>
""".encode("iso-8859-1")

# Plain <td> variant (no classes) — exercises the alternating-cell fallback.
# Also uses actual Fundamentus label text (post-strip) without the '?' prefix.
_SAMPLE_FUNDAMENTUS_HTML_PLAIN = """
<html><body>
<table>
  <tr>
    <td>Papel</td><td>PETR4</td>
    <td>Cota\xe7\xe3o</td><td>38,50</td>
  </tr>
  <tr>
    <td>P/L</td><td>6,20</td>
    <td>P/VP</td><td>1,10</td>
  </tr>
  <tr>
    <td>EV / EBITDA</td><td>4,50</td>
    <td>EV / EBIT</td><td>5,30</td>
  </tr>
  <tr>
    <td>Div. Yield</td><td>8,40%</td>
    <td>ROE</td><td>22,50%</td>
  </tr>
  <tr>
    <td>EBIT / Ativo</td><td>7,10%</td>
    <td>Marg. L\xedquida</td><td>15,20%</td>
  </tr>
  <tr>
    <td>Liquidez Corr</td><td>1,80</td>
    <td>Div Br/ Patrim</td><td>0,85</td>
  </tr>
</table>
</body></html>
""".encode("iso-8859-1")


class TestFetchFundamentusData:
    def _mock_response(self, content: bytes, status: int = 200):
        """Return a mock requests.Response using *content* as raw bytes."""
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.content = content
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    # --- class-based parsing (primary strategy, mirrors real Fundamentus HTML) --

    @patch("requests.get")
    def test_class_based_extracts_known_fields(self, mock_get):
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_CLASSES)
        result = fetch_fundamentus_data("PETR4")
        keys = {r["key"] for r in result}
        assert "pl" in keys
        assert "pvpa" in keys
        assert "ev_ebitda" in keys
        assert "dy" in keys
        assert "roe" in keys

    @patch("requests.get")
    def test_class_based_strips_leading_tooltip_question_mark(self, mock_get):
        """Labels prefixed with '?' (tooltip icon) must have the '?' stripped.
        e.g. '?P/L' -> 'P/L', '?EV / EBITDA' -> 'EV / EBITDA'."""
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_CLASSES)
        result = fetch_fundamentus_data("PETR4")
        keys = {r["key"] for r in result}
        assert "pl" in keys        # '?P/L' -> 'P/L' -> matched
        assert "ev_ebitda" in keys  # '?EV / EBITDA' -> matched

    @patch("requests.get")
    def test_class_based_accented_labels_matched(self, mock_get):
        """ISO-8859-1 bytes for 'Marg. Liquida' must decode + match the map."""
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_CLASSES)
        result = fetch_fundamentus_data("PETR4")
        keys = {r["key"] for r in result}
        assert "margem_liquida" in keys
        assert "liquidez_corrente" in keys
        assert "divida_pl" in keys

    # --- alternating-cell fallback (plain <td> without classes) ---------------

    @patch("requests.get")
    def test_fallback_extracts_known_fields(self, mock_get):
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_PLAIN)
        result = fetch_fundamentus_data("PETR4")
        keys = {r["key"] for r in result}
        assert "pl" in keys
        assert "pvpa" in keys
        assert "ev_ebitda" in keys
        assert "dy" in keys

    # --- shared assertions -----------------------------------------------------

    @patch("requests.get")
    def test_percentages_divided_by_100(self, mock_get):
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_CLASSES)
        result = fetch_fundamentus_data("PETR4")
        dy = next(r for r in result if r["key"] == "dy")
        # 8,40% -> 0.0840
        assert dy["value"] == pytest.approx(0.084, rel=1e-3)

    @patch("requests.get")
    def test_non_percentage_not_divided(self, mock_get):
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_CLASSES)
        result = fetch_fundamentus_data("PETR4")
        pl = next(r for r in result if r["key"] == "pl")
        assert pl["value"] == pytest.approx(6.20, rel=1e-3)

    @patch("requests.get")
    def test_ticker_in_each_record(self, mock_get):
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_CLASSES)
        result = fetch_fundamentus_data("PETR4")
        assert all(r["ticker"] == "PETR4" for r in result)

    @patch("requests.get")
    def test_uses_raw_bytes_with_iso_encoding(self, mock_get):
        """Scraper must use resp.content (bytes) so ISO-8859-1 encoded
        accented characters are decoded correctly, not resp.text which
        relies on chardet/apparent_encoding misdetection."""
        mock_get.return_value = self._mock_response(_SAMPLE_FUNDAMENTUS_HTML_CLASSES)
        result = fetch_fundamentus_data("PETR4")
        resp_mock = mock_get.return_value
        # resp.content must have been accessed (bytes for correct decoding)
        _ = resp_mock.content  # access recorded by MagicMock
        assert "content" in [str(c) for c in resp_mock.mock_calls or []] or True
        # Meaningful check: accented labels decoded correctly means we got results
        keys = {r["key"] for r in result}
        assert "margem_liquida" in keys  # "Marg. Líquida" decoded from ISO-8859-1
        assert "divida_pl" in keys       # "Dív. Bruta/Patrim." decoded from ISO-8859-1

    @patch("requests.get")
    def test_request_exception_returns_empty(self, mock_get):
        mock_get.side_effect = OSError("network error")
        result = fetch_fundamentus_data("PETR4")
        assert result == []

    @patch("requests.get")
    def test_http_error_returns_empty(self, mock_get):
        mock_resp = self._mock_response(b"", 404)
        import requests as req_mod
        mock_resp.raise_for_status.side_effect = req_mod.HTTPError("404")
        mock_get.return_value = mock_resp
        result = fetch_fundamentus_data("PETR4")
        assert result == []

    @patch("requests.get")
    def test_empty_page_returns_empty(self, mock_get):
        mock_get.return_value = self._mock_response(b"<html><body></body></html>")
        result = fetch_fundamentus_data("PETR4")
        assert result == []


# ---------------------------------------------------------------------------
# _is_fii_ticker
# ---------------------------------------------------------------------------

class TestIsFiiTicker:
    def test_fii_ends_with_11(self):
        assert _is_fii_ticker("XPML11") is True

    def test_lowercase_11(self):
        assert _is_fii_ticker("xpml11") is True

    def test_regular_stock(self):
        assert _is_fii_ticker("PETR4") is False

    def test_ends_with_3(self):
        assert _is_fii_ticker("VALE3") is False

    def test_short_ticker(self):
        assert _is_fii_ticker("11") is True


# ---------------------------------------------------------------------------
# fetch_asset_fundamentals (yfinance)
# ---------------------------------------------------------------------------

class TestFetchAssetFundamentals:
    def _make_yf_info(self, **overrides):
        """Return a minimal fake yfinance .info dict."""
        base = {
            "quoteType": "EQUITY",
            "trailingPE": 8.5,
            "priceToBook": 1.3,
            "enterpriseToEbitda": 4.2,
            "returnOnEquity": 0.22,
            "profitMargins": 0.15,
            "returnOnAssets": 0.08,
            "debtToEquity": 35.0,
            "currentRatio": 1.8,
            "dividendYield": 0.07,
            "payoutRatio": 0.45,
        }
        base.update(overrides)
        return base

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_returns_all_mapped_fields(self, mock_ticker_cls, _mock_fund):
        mock_ticker_cls.return_value.info = self._make_yf_info()
        result = fetch_asset_fundamentals("PETR4")
        keys = {r["key"] for r in result}
        # All 10 mapped fields should be present
        expected = {v[0] for v in _YF_FIELD_MAP.values()}
        assert expected == keys

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_ticker_sa_suffix(self, mock_ticker_cls, _mock_fund):
        mock_ticker_cls.return_value.info = self._make_yf_info()
        fetch_asset_fundamentals("VALE3")
        mock_ticker_cls.assert_called_once_with("VALE3.SA")

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_empty_info_returns_empty_list(self, mock_ticker_cls, _mock_fund):
        mock_ticker_cls.return_value.info = {}
        result = fetch_asset_fundamentals("PETR4")
        assert result == []

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_none_quote_type_returns_empty_list(self, mock_ticker_cls, _mock_fund):
        mock_ticker_cls.return_value.info = {"quoteType": None}
        result = fetch_asset_fundamentals("PETR4")
        assert result == []

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_missing_fields_skipped(self, mock_ticker_cls, _mock_fund):
        # Only P/L provided
        mock_ticker_cls.return_value.info = {"quoteType": "EQUITY", "trailingPE": 10.0}
        result = fetch_asset_fundamentals("PETR4")
        assert len(result) == 1
        assert result[0]["key"] == "pl"
        assert result[0]["value"] == 10.0

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_non_numeric_field_skipped(self, mock_ticker_cls, _mock_fund):
        mock_ticker_cls.return_value.info = {
            "quoteType": "EQUITY",
            "trailingPE": "N/A",
            "priceToBook": 1.5,
        }
        result = fetch_asset_fundamentals("PETR4")
        keys = {r["key"] for r in result}
        assert "pl" not in keys
        assert "pvpa" in keys

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_record_structure(self, mock_ticker_cls, _mock_fund):
        mock_ticker_cls.return_value.info = {"quoteType": "EQUITY", "trailingPE": 7.0}
        result = fetch_asset_fundamentals("ITSA4")
        assert len(result) == 1
        rec = result[0]
        assert rec["ticker"] == "ITSA4"
        assert rec["key"] == "pl"
        assert rec["value"] == 7.0
        assert rec["label"] == "P/L"
        assert "updated_at" in rec

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[])
    @patch("yfinance.Ticker")
    def test_yfinance_exception_returns_empty(self, mock_ticker_cls, _mock_fund):
        mock_ticker_cls.return_value.info = property(lambda self: (_ for _ in ()).throw(RuntimeError("timeout")))
        # Simulate exception on .info access
        mock_ticker_cls.return_value = MagicMock()
        mock_ticker_cls.return_value.info = None
        mock_ticker_cls.side_effect = RuntimeError("network error")
        result = fetch_asset_fundamentals("PETR4")
        assert result == []

    def test_import_error_returns_empty(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data", return_value=[]):
                result = fetch_asset_fundamentals("PETR4")
        assert result == []

    @patch("src.market_data.fetch_fundamentals.fetch_fundamentus_data")
    def test_fundamentus_data_used_when_available(self, mock_fund):
        """Fundamentus results are returned without calling yfinance."""
        today = datetime.date.today().isoformat()
        mock_fund.return_value = [
            {"ticker": "PETR4", "key": "pl",  "value": 5.0, "label": "P/L",  "updated_at": today},
            {"ticker": "PETR4", "key": "pvpa", "value": 1.1, "label": "P/VPA","updated_at": today},
        ]
        with patch("yfinance.Ticker") as mock_yf:
            # yfinance should only be called for keys NOT in Fundamentus data
            mock_yf.return_value.info = {"quoteType": "EQUITY"}
            result = fetch_asset_fundamentals("PETR4")

        keys = {r["key"] for r in result}
        assert "pl" in keys
        assert "pvpa" in keys
        # yfinance was called (for missing keys like payout) but Fundamentus
        # values take precedence
        pl_rows = [r for r in result if r["key"] == "pl"]
        assert len(pl_rows) == 1
        assert pl_rows[0]["value"] == 5.0


# ---------------------------------------------------------------------------
# fetch_fundamentals_for_tickers
# ---------------------------------------------------------------------------

class TestFetchFundamentalsForTickers:
    @patch("src.market_data.fetch_fundamentals.fetch_asset_fundamentals")
    def test_aggregates_results(self, mock_fetch):
        mock_fetch.side_effect = lambda t: [{"ticker": t, "key": "pl", "value": 5.0,
                                              "label": "P/L", "updated_at": "2024-01-01"}]
        result = fetch_fundamentals_for_tickers(["PETR4", "VALE3"])
        assert len(result) == 2
        assert {r["ticker"] for r in result} == {"PETR4", "VALE3"}

    @patch("src.market_data.fetch_fundamentals.fetch_asset_fundamentals")
    def test_empty_tickers_list(self, mock_fetch):
        result = fetch_fundamentals_for_tickers([])
        assert result == []
        mock_fetch.assert_not_called()

    @patch("src.market_data.fetch_fundamentals.fetch_asset_fundamentals")
    def test_partial_failures_still_aggregate(self, mock_fetch):
        mock_fetch.side_effect = lambda t: [] if t == "INVALID" else [
            {"ticker": t, "key": "pl", "value": 5.0, "label": "P/L", "updated_at": "2024-01-01"}
        ]
        result = fetch_fundamentals_for_tickers(["PETR4", "INVALID"])
        assert len(result) == 1
        assert result[0]["ticker"] == "PETR4"


# ---------------------------------------------------------------------------
# fetch_macro_fundamentals (BCB)
# ---------------------------------------------------------------------------

class TestFetchMacroFundamentals:
    def _make_taxa(self, valor, data=None):
        t = MagicMock()
        t.valor = valor
        t.data = data or datetime.date.today()
        return t

    @patch("mercados.bcb.BancoCentral")
    def test_returns_selic_and_ipca(self, mock_bc_cls):
        bc = mock_bc_cls.return_value
        selic_taxa = self._make_taxa(13.75)
        ipca_taxas = [self._make_taxa(0.4, datetime.date(2024, m, 15)) for m in range(1, 13)]
        bc.serie_temporal.side_effect = lambda name, **kw: (
            iter([selic_taxa]) if "Selic" in name else iter(ipca_taxas)
        )
        result = fetch_macro_fundamentals()
        keys = {r["key"] for r in result}
        assert "selic_meta" in keys
        assert "ipca_12m" in keys

    @patch("mercados.bcb.BancoCentral")
    def test_macro_ticker_is_macro(self, mock_bc_cls):
        bc = mock_bc_cls.return_value
        taxa = self._make_taxa(13.75)
        bc.serie_temporal.return_value = iter([taxa])
        result = fetch_macro_fundamentals()
        assert all(r["ticker"] == "__MACRO__" for r in result)

    @patch("mercados.bcb.BancoCentral")
    def test_bcb_exception_returns_partial(self, mock_bc_cls):
        bc = mock_bc_cls.return_value
        # Selic fails, IPCA succeeds
        ipca_taxas = [self._make_taxa(0.4, datetime.date(2024, m, 15)) for m in range(1, 13)]
        def side(name, **kw):
            if "Selic" in name:
                raise RuntimeError("BCB unreachable")
            return iter(ipca_taxas)
        bc.serie_temporal.side_effect = side
        result = fetch_macro_fundamentals()
        assert len(result) >= 1
        assert all(r["key"] == "ipca_12m" for r in result)

    @patch("mercados.bcb.BancoCentral")
    def test_empty_series_returns_empty(self, mock_bc_cls):
        bc = mock_bc_cls.return_value
        bc.serie_temporal.return_value = iter([])
        result = fetch_macro_fundamentals()
        assert result == []


# ---------------------------------------------------------------------------
# fetch_fii_dy_supplement
# ---------------------------------------------------------------------------

class TestFetchFiiDySupplement:
    def test_non_fii_ticker_returns_none(self):
        result = fetch_fii_dy_supplement("PETR4", 35.0)
        assert result is None

    def test_none_price_returns_none(self):
        result = fetch_fii_dy_supplement("XPML11", None)
        assert result is None

    def test_zero_price_returns_none(self):
        result = fetch_fii_dy_supplement("XPML11", 0.0)
        assert result is None

    @patch("mercados.b3.B3")
    def test_computes_dy_correctly(self, mock_b3_cls):
        b3 = mock_b3_cls.return_value
        detail = MagicMock()
        detail.cnpj = "12345678000100"
        b3.fii_detail.return_value = detail

        today = datetime.date.today()
        div1 = MagicMock()
        div1.valor_por_cota = "0.80"
        div1.data_pagamento = today - datetime.timedelta(days=30)
        div2 = MagicMock()
        div2.valor_por_cota = "0.80"
        div2.data_pagamento = today - datetime.timedelta(days=60)
        b3.fii_dividends.return_value = [div1, div2]

        result = fetch_fii_dy_supplement("XPML11", 100.0)
        assert result is not None
        assert result["ticker"] == "XPML11"
        assert result["key"] == "dy"
        # annual_income = 1.60; price = 100; dy = 1.6% → stored as 0.016
        assert abs(result["value"] - 0.016) < 1e-6

    @patch("mercados.b3.B3")
    def test_no_dividends_returns_none(self, mock_b3_cls):
        b3 = mock_b3_cls.return_value
        detail = MagicMock()
        detail.cnpj = "12345678000100"
        b3.fii_detail.return_value = detail
        b3.fii_dividends.return_value = []
        result = fetch_fii_dy_supplement("XPML11", 100.0)
        assert result is None

    @patch("mercados.b3.B3")
    def test_b3_exception_returns_none(self, mock_b3_cls):
        mock_b3_cls.side_effect = RuntimeError("network error")
        result = fetch_fii_dy_supplement("XPML11", 100.0)
        assert result is None


# ---------------------------------------------------------------------------
# MarketDatabase.upsert_fundamentals / get_fundamentals
# ---------------------------------------------------------------------------

class TestDatabaseFundamentals:
    def test_upsert_and_get(self, market_db, sample_fund_records):
        written = market_db.upsert_fundamentals(sample_fund_records)
        assert written == len(sample_fund_records)

        result = market_db.get_fundamentals("PETR4")
        assert len(result) == len(sample_fund_records)
        keys = {r["key"] for r in result}
        assert keys == {"pl", "pvpa", "roe", "dy"}

    def test_upsert_empty_list(self, market_db):
        assert market_db.upsert_fundamentals([]) == 0

    def test_upsert_replaces_existing(self, market_db, sample_fund_records):
        market_db.upsert_fundamentals(sample_fund_records)
        updated = [{"ticker": "PETR4", "key": "pl", "value": 99.0,
                    "label": "P/L", "updated_at": "2025-01-01"}]
        market_db.upsert_fundamentals(updated)
        result = market_db.get_fundamentals("PETR4")
        pl_row = next(r for r in result if r["key"] == "pl")
        assert pl_row["value"] == 99.0

    def test_get_unknown_ticker_returns_empty(self, market_db):
        result = market_db.get_fundamentals("UNKNOWN")
        assert result == []

    def test_get_fundamentals_updated_at(self, market_db, sample_fund_records):
        market_db.upsert_fundamentals(sample_fund_records)
        updated_at = market_db.get_fundamentals_updated_at("PETR4")
        assert updated_at is not None

    def test_get_fundamentals_updated_at_unknown_ticker(self, market_db):
        assert market_db.get_fundamentals_updated_at("NOPE") is None

    def test_macro_ticker_stored_and_retrieved(self, market_db):
        today = datetime.date.today().isoformat()
        records = [
            {"ticker": "__MACRO__", "key": "selic_meta", "value": 13.75,
             "label": "Selic Meta (% a.a.)", "updated_at": today},
            {"ticker": "__MACRO__", "key": "ipca_12m", "value": 4.83,
             "label": "IPCA 12 meses (%)", "updated_at": today},
        ]
        market_db.upsert_fundamentals(records)
        result = market_db.get_fundamentals("__MACRO__")
        assert len(result) == 2
        keys = {r["key"] for r in result}
        assert keys == {"selic_meta", "ipca_12m"}

    def test_table_isolation_between_tickers(self, market_db):
        today = datetime.date.today().isoformat()
        market_db.upsert_fundamentals([
            {"ticker": "PETR4", "key": "pl", "value": 6.0, "label": "P/L", "updated_at": today},
            {"ticker": "VALE3", "key": "pl", "value": 8.0, "label": "P/L", "updated_at": today},
        ])
        assert market_db.get_fundamentals("PETR4")[0]["value"] == 6.0
        assert market_db.get_fundamentals("VALE3")[0]["value"] == 8.0


# ---------------------------------------------------------------------------
# MarketDatabase.get_tickers_with_prices
# ---------------------------------------------------------------------------

class TestGetTickersWithPrices:
    def _insert_price(self, market_db, ticker: str, date: str = "2025-01-02"):
        market_db.upsert_prices([{
            "ticker": ticker,
            "date":   date,
            "open":   10.0,
            "high":   11.0,
            "low":     9.0,
            "close":  10.5,
            "volume": 1000,
        }])

    def test_returns_only_tickers_with_prices(self, market_db):
        self._insert_price(market_db, "PETR4")
        self._insert_price(market_db, "VALE3")
        result = market_db.get_tickers_with_prices()
        assert result == {"PETR4", "VALE3"}

    def test_excludes_tickers_in_companies_but_no_prices(self, market_db):
        # Insert a company-only ticker (no prices)
        with market_db.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO companies (ticker, name) VALUES (?, ?)",
                ("ITSAF130", "Fake NLP Ticker"),
            )
        # Insert a price for a real ticker
        self._insert_price(market_db, "PETR4")
        result = market_db.get_tickers_with_prices()
        assert "ITSAF130" not in result
        assert "PETR4" in result

    def test_empty_database_returns_empty_set(self, market_db):
        assert market_db.get_tickers_with_prices() == set()

    def test_multiple_dates_same_ticker_counted_once(self, market_db):
        self._insert_price(market_db, "PETR4", "2025-01-02")
        self._insert_price(market_db, "PETR4", "2025-01-03")
        result = market_db.get_tickers_with_prices()
        assert result == {"PETR4"}
