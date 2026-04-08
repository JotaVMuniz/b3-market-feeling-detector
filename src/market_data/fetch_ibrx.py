"""
Fetch the IBrX 100 index composition from B3.

The ticker list is retrieved from B3's public JSON API.  When the API is
unreachable a hardcoded fallback list (accurate as of early 2026) is returned
so the pipeline continues to operate offline or when the B3 site is down.

The resulting list is the canonical universe of tickers processed by the
entire pipeline — prices, fundamentals, analytics — so that only liquid,
index-eligible assets are tracked.
"""

import base64
import json
import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B3 API
# ---------------------------------------------------------------------------

_B3_API_BASE = (
    "https://sistemaswebb3-listados.b3.com.br"
    "/indexProxy/indexCall/GetPortfolioDay/"
)

# ---------------------------------------------------------------------------
# Hardcoded fallback list (IBrX 100 as of early 2026)
# Updated quarterly; kept here so the pipeline works without internet access.
# ---------------------------------------------------------------------------

_IBRX100_FALLBACK: List[str] = [
    "ABEV3", "ALPA4", "AMBP3", "AMER3", "ANIM3", "ARZZ3", "ASAI3",
    "AURE3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3",
    "BEEF3", "BPAC11", "BRAP4", "BRFS3", "BRKM5", "CASH3", "CCRO3",
    "CIEL3", "CMIG4", "COGN3", "CPFE3", "CPLE6", "CRFB3", "CSAN3",
    "CSNA3", "CVCB3", "CYRE3", "DXCO3", "ECOR3", "EGIE3", "ELET3",
    "ELET6", "EMBR3", "ENEV3", "ENGI11", "EQTL3", "EZTC3", "FLRY3",
    "GGBR4", "GOAU4", "GOLL4", "GRND3", "HAPV3", "HYPE3", "IGTA3",
    "INTB3", "IRBR3", "ITSA4", "ITUB4", "JBSS3", "JHSF3", "KLBN11",
    "LCAM3", "LREN3", "LWSA3", "MDIA3", "MGLU3", "MRFG3", "MRVE3",
    "MULT3", "NTCO3", "PCAR3", "PETR3", "PETR4", "PETZ3", "PRIO3",
    "QUAL3", "RADL3", "RAIL3", "RAIZ4", "RDOR3", "RENT3", "RRRP3",
    "SANB11", "SBSP3", "SLCE3", "SMTO3", "SOMA3", "SULA11", "SUZB3",
    "TAEE11", "TASA4", "TEND3", "TIMS3", "TOTS3", "UGPA3", "UNIP6",
    "USIM5", "VALE3", "VBBR3", "VIVT3", "WEGE3", "YDUQ3",
]


def fetch_ibrx100_tickers() -> List[str]:
    """
    Return the current IBrX 100 composition as a list of B3 ticker codes.

    Queries B3's public portfolio API.  When the request fails for any
    reason (network error, unexpected response format, etc.) the hardcoded
    fallback list is returned instead so the pipeline can continue running.

    The API endpoint accepts a base64-encoded JSON payload that specifies
    the index name and pagination parameters:

    ``{"language":"pt-br","pageNumber":1,"pageSize":120,"index":"IBRX100","segment":"1"}``

    Returns:
        List of B3 ticker codes (e.g. ``["PETR4", "VALE3", ...]``).
        Always non-empty (falls back to the hardcoded list).
    """
    try:
        import requests
    except ImportError:
        logger.error("requests not installed — returning fallback IBrX 100 list")
        return list(_IBRX100_FALLBACK)

    payload = {
        "language": "pt-br",
        "pageNumber": 1,
        "pageSize": 120,
        "index": "IBRX100",
        "segment": "1",
    }
    encoded = base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    url = f"{_B3_API_BASE}{encoded}"

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        # The API may return results under different keys depending on version
        results = (
            data.get("results")
            or data.get("components")
            or []
        )
        tickers = []
        for item in results:
            code = (
                item.get("cod")
                or item.get("codISIN")
                or item.get("ticker")
                or item.get("codigo")
            )
            if code:
                tickers.append(code.strip().upper())

        if tickers:
            logger.info("Fetched %d IBrX 100 tickers from B3 API", len(tickers))
            return tickers

        logger.warning(
            "B3 API returned empty results for IBrX 100 — using fallback list"
        )

    except Exception as exc:
        logger.warning(
            "Could not fetch IBrX 100 from B3 API (%s) — using fallback list",
            exc,
        )

    logger.info(
        "Using hardcoded IBrX 100 fallback list (%d tickers)",
        len(_IBRX100_FALLBACK),
    )
    return list(_IBRX100_FALLBACK)
