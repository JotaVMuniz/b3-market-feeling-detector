"""
Streamlit Dashboard — B3 Market Feeling Detector

Three-tab layout:
  📊 Visão Geral  – Aggregated metrics and sentiment by market segment.
  📈 Por Ativo    – Asset-centric view: price history + related news.
  📰 Notícias     – Full news feed with sentiment filters.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/news.db")

SENTIMENT_COLORS = {
    "positivo": "#2ca02c",
    "negativo": "#d62728",
    "neutro":   "#ff7f0e",
}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _conn(db_path: str = DB_PATH):
    return sqlite3.connect(db_path)


def _table_exists(table: str, db_path: str = DB_PATH) -> bool:
    try:
        conn = _conn(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        exists = cur.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False


def _news_columns(db_path: str = DB_PATH) -> List[str]:
    try:
        conn = _conn(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(news)")
        cols = [r[1] for r in cur.fetchall()]
        conn.close()
        return cols
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_news(
    source: Optional[str] = None,
    sentiment: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    segment: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 2000,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Load enriched news from the database with optional filters."""
    cols = _news_columns(db_path)

    def has(c: str) -> bool:
        return c in cols

    select = (
        "id, title, content, source, published_at, url, collected_at, "
        + ("sentiment, confidence, " if has("sentiment") else "'neutro' AS sentiment, 0.0 AS confidence, ")
        + ("segments, " if has("segments") else "'[]' AS segments, ")
        + ("tickers" if has("tickers") else "'[]' AS tickers")
    )

    query = f"SELECT {select} FROM news WHERE 1=1"
    params: list = []

    if source and source != "Todas":
        query += " AND source = ?"
        params.append(source)
    if sentiment and sentiment != "Todos" and has("sentiment"):
        query += " AND sentiment = ?"
        params.append(sentiment)
    if date_from:
        query += " AND published_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND published_at <= ?"
        params.append(date_to + " 23:59:59")
    query += " ORDER BY published_at DESC LIMIT ?"
    params.append(limit)

    try:
        conn = _conn(db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    except Exception as e:
        st.error(f"Erro ao carregar notícias: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")

    for col in ("segments", "tickers"):
        df[col] = df[col].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x else []
        )

    # post-filter by segment
    if segment and segment not in ("Todos", "Sem segmento"):
        df = df[df["segments"].apply(lambda s: segment in s if isinstance(s, list) else False)]

    # post-filter by ticker
    if ticker:
        df = df[df["tickers"].apply(lambda t: ticker in t if isinstance(t, list) else False)]

    return df.reset_index(drop=True)


def load_asset_prices(ticker: str, db_path: str = DB_PATH) -> pd.DataFrame:
    if not _table_exists("asset_prices", db_path):
        return pd.DataFrame()
    query = """
        SELECT date, open, close, high, low, avg_price, volume
        FROM asset_prices WHERE ticker = ? ORDER BY date ASC
    """
    try:
        conn = _conn(db_path)
        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar preços para {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_tickers(db_path: str = DB_PATH) -> List[str]:
    if not _table_exists("asset_prices", db_path):
        return []
    try:
        conn = _conn(db_path)
        df = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM asset_prices ORDER BY ticker", conn
        )
        conn.close()
        return df["ticker"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=300)
def load_companies(db_path: str = DB_PATH) -> pd.DataFrame:
    if not _table_exists("companies", db_path):
        return pd.DataFrame()
    try:
        conn = _conn(db_path)
        df = pd.read_sql_query("SELECT ticker, name, tipo_papel FROM companies", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_correlations(limit: int = 2000, db_path: str = DB_PATH) -> pd.DataFrame:
    if not _table_exists("news_price_correlation", db_path):
        return pd.DataFrame()
    query = """
        SELECT c.news_id, c.ticker, c.news_date, c.sentiment, c.confidence,
               c.d0_var, c.d1_var, c.d5_var, n.title, n.source
        FROM news_price_correlation c
        JOIN news n ON n.id = c.news_id
        ORDER BY c.news_date DESC LIMIT ?
    """
    try:
        conn = _conn(db_path)
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        df["news_date"] = pd.to_datetime(df["news_date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar correlações: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_sentiment_indicators(
    indicator: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Load sentiment indicator rows from the database."""
    if not _table_exists("sentiment_indicators", db_path):
        return pd.DataFrame()

    query = "SELECT * FROM sentiment_indicators WHERE 1=1"
    params: list = []
    if indicator:
        query += " AND indicator = ?"
        params.append(indicator)
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
    query += " ORDER BY date ASC"

    try:
        conn = _conn(db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar indicadores: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_composite_index(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Load composite sentiment index rows from the database."""
    if not _table_exists("composite_sentiment_index", db_path):
        return pd.DataFrame()

    query = "SELECT * FROM composite_sentiment_index WHERE 1=1"
    params: list = []
    if date_from:
        query += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND date <= ?"
        params.append(date_to)
    query += " ORDER BY date ASC"

    try:
        conn = _conn(db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar índice composto: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_fundamentals(ticker: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load fundamental indicators for *ticker* from the database."""
    if not _table_exists("asset_fundamentals", db_path):
        return pd.DataFrame()
    try:
        conn = _conn(db_path)
        df = pd.read_sql_query(
            "SELECT key, value, label, updated_at FROM asset_fundamentals "
            "WHERE ticker = ? ORDER BY key",
            conn,
            params=(ticker,),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def expand_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "segments" not in df.columns:
        return df.copy()
    rows = []
    for _, row in df.iterrows():
        segs = row["segments"]
        if isinstance(segs, list) and segs:
            for seg in segs:
                r = row.copy()
                r["segment"] = seg
                rows.append(r)
        else:
            r = row.copy()
            r["segment"] = "Sem segmento"
            rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def get_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    expanded = expand_segments(df)
    if expanded.empty:
        return pd.DataFrame()
    stats = (
        expanded.groupby("segment")
        .agg(
            total=("id", "count"),
            positivo=("sentiment", lambda x: (x == "positivo").sum()),
            negativo=("sentiment", lambda x: (x == "negativo").sum()),
            neutro=("sentiment", lambda x: (x == "neutro").sum()),
            confianca_media=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("total", ascending=False)
    )
    stats["confianca_media"] = stats["confianca_media"].round(3)
    # Use max(total, 1) to avoid division by zero for segments with no news
    safe_total = stats["total"].clip(lower=1)
    stats["pct_positivo"] = (stats["positivo"] / safe_total * 100).round(1)
    stats["pct_negativo"] = (stats["negativo"] / safe_total * 100).round(1)
    return stats


def get_segment_price_stats(corr_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """Merge segment info from news with price variation from correlations."""
    if corr_df.empty or news_df.empty:
        return pd.DataFrame()

    # Build ticker → segments lookup from news
    ticker_seg_rows = []
    for _, row in news_df.iterrows():
        tickers = row.get("tickers", [])
        segs = row.get("segments", [])
        if isinstance(tickers, list) and isinstance(segs, list):
            for ticker in tickers:
                for seg in segs:
                    ticker_seg_rows.append({"ticker": ticker, "segment": seg})

    if not ticker_seg_rows:
        return pd.DataFrame()

    ticker_seg = pd.DataFrame(ticker_seg_rows).drop_duplicates()
    merged = corr_df.merge(ticker_seg, on="ticker", how="left")
    merged = merged.dropna(subset=["segment"])

    if merged.empty:
        return pd.DataFrame()

    stats = (
        merged.groupby("segment")
        .agg(
            avg_d0=(  "d0_var", lambda x: (x.dropna() * 100).mean()),
            avg_d1=(  "d1_var", lambda x: (x.dropna() * 100).mean()),
            avg_d5=(  "d5_var", lambda x: (x.dropna() * 100).mean()),
            n_pares=( "news_id", "count"),
        )
        .reset_index()
    )
    for col in ("avg_d0", "avg_d1", "avg_d5"):
        stats[col] = stats[col].round(2)
    return stats.sort_values("n_pares", ascending=False)


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def compute_asset_sentiment(news_df: pd.DataFrame) -> Dict:
    """Compute a weighted sentiment score from a news dataframe.

    Returns a dict with keys:
    - label: "positivo" | "negativo" | "neutro"
    - score: float in [-1, 1]
    - count: int (total news items)
    - pos: int (positive count)
    - neg: int (negative count)
    - neu: int (neutral count)
    """
    empty = {"label": "neutro", "score": 0.0, "count": 0, "pos": 0, "neg": 0, "neu": 0}
    if news_df.empty or "sentiment" not in news_df.columns:
        return empty

    sentiment_map = {"positivo": 1.0, "negativo": -1.0, "neutro": 0.0}
    scores: List[float] = []
    for _, row in news_df.iterrows():
        sent = (row.get("sentiment") or "neutro").lower()
        conf = float(row.get("confidence") or 0.5)
        scores.append(sentiment_map.get(sent, 0.0) * conf)

    if not scores:
        return empty

    avg_score = sum(scores) / len(scores)
    pos = int((news_df["sentiment"] == "positivo").sum())
    neg = int((news_df["sentiment"] == "negativo").sum())
    neu = int((news_df["sentiment"] == "neutro").sum())

    if avg_score > 0.1:
        label = "positivo"
    elif avg_score < -0.1:
        label = "negativo"
    else:
        label = "neutro"

    return {"label": label, "score": avg_score, "count": len(news_df), "pos": pos, "neg": neg, "neu": neu}


def _render_sentiment_indicator(sentiment_data: Dict) -> None:
    """Render a visual sentiment indicator for an asset."""
    label = sentiment_data["label"]
    score = sentiment_data["score"]
    count = sentiment_data["count"]

    emoji_map = {"positivo": "🟢", "negativo": "🔴", "neutro": "🟡"}
    label_pt = {"positivo": "Positivo", "negativo": "Negativo", "neutro": "Neutro"}
    emoji = emoji_map.get(label, "⚪")

    st.markdown(
        f"""
        <div style="
            background: {'#d4edda' if label == 'positivo' else '#f8d7da' if label == 'negativo' else '#fff3cd'};
            border-radius: 12px;
            padding: 16px 24px;
            display: inline-block;
            margin-bottom: 8px;
        ">
            <span style="font-size: 2rem;">{emoji}</span>
            <span style="font-size: 1.4rem; font-weight: bold; margin-left: 10px;">
                Sentimento Geral: {label_pt[label]}
            </span>
            <br/>
            <span style="color: #555; font-size: 0.9rem;">
                Score: {score:+.2f} &nbsp;|&nbsp;
                Baseado em {count} {'notícia' if count == 1 else 'notícias'}
                &nbsp;({sentiment_data['pos']} positivas, {sentiment_data['neg']} negativas, {sentiment_data['neu']} neutras)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _segment_bar(df: pd.DataFrame):
    stats = get_segment_stats(df)
    if stats.empty:
        return None
    fig = px.bar(
        stats, x="segment", y="total",
        title="Notícias por Segmento",
        color="total", color_continuous_scale="Blues",
        labels={"segment": "Segmento", "total": "Notícias"},
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    return fig


def _segment_sentiment_heatmap(df: pd.DataFrame):
    expanded = expand_segments(df)
    if expanded.empty or "sentiment" not in expanded.columns:
        return None
    pivot = pd.crosstab(expanded["segment"], expanded["sentiment"])
    fig = px.imshow(
        pivot, title="Sentimento por Segmento",
        color_continuous_scale="Blues", aspect="auto",
        labels={"x": "Sentimento", "y": "Segmento", "color": "Contagem"},
    )
    return fig


def _sentiment_over_time(df: pd.DataFrame):
    if df.empty or "published_at" not in df.columns:
        return None
    tmp = df.copy()
    tmp["date"] = tmp["published_at"].dt.date
    series = tmp.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig = px.line(
        series, x="date", y="count", color="sentiment",
        title="Evolução de Sentimentos ao Longo do Tempo",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"date": "Data", "count": "Notícias", "sentiment": "Sentimento"},
    )
    return fig


def _price_chart(price_df: pd.DataFrame, corr_df: pd.DataFrame, ticker: str):
    if price_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df["date"], y=price_df["close"],
        mode="lines", name="Fechamento",
        line=dict(color="#1f77b4", width=2),
    ))
    if not corr_df.empty:
        ticker_news = corr_df[corr_df["ticker"] == ticker].copy()
        price_lookup = price_df.set_index(price_df["date"].dt.normalize())["close"].to_dict()
        for sentiment, color, symbol in [
            ("positivo", "#2ca02c", "triangle-up"),
            ("negativo", "#d62728", "triangle-down"),
            ("neutro",   "#ff7f0e", "circle"),
        ]:
            subset = ticker_news[ticker_news["sentiment"] == sentiment].copy()
            if subset.empty:
                continue
            subset["_dk"] = subset["news_date"].dt.normalize()
            subset["_y"] = subset["_dk"].map(price_lookup)
            subset = subset.dropna(subset=["_y"])
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["news_date"], y=subset["_y"],
                mode="markers",
                name=f"Notícia {sentiment}",
                marker=dict(color=color, size=10, symbol=symbol),
                text=subset["title"],
                hovertemplate="%{text}<extra></extra>",
            ))
    fig.update_layout(
        title=f"Preço de Fechamento — {ticker}",
        xaxis_title="Data", yaxis_title="Preço (R$)",
    )
    return fig


def _scatter_sentiment_return(corr_df: pd.DataFrame, horizon: str = "d1_var"):
    if corr_df.empty or horizon not in corr_df.columns:
        return None
    label_map = {"d0_var": "D0 (intraday)", "d1_var": "D+1", "d5_var": "D+5"}
    plot_df = corr_df[corr_df[horizon].notna()].copy()
    if plot_df.empty:
        return None
    plot_df[horizon] = plot_df[horizon] * 100
    fig = px.strip(
        plot_df, x="sentiment", y=horizon, color="sentiment",
        hover_data=["ticker", "title", "news_date"],
        color_discrete_map=SENTIMENT_COLORS,
        title=f"Sentimento vs Retorno {label_map.get(horizon, horizon)} (%)",
        labels={"sentiment": "Sentimento", horizon: f"Retorno (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def _candlestick_chart(price_df: pd.DataFrame, ticker: str):
    if price_df.empty:
        return None
    fig = go.Figure(go.Candlestick(
        x=price_df["date"],
        open=price_df["open"], high=price_df["high"],
        low=price_df["low"], close=price_df["close"],
        name=ticker,
    ))
    fig.update_layout(
        title=f"Candlestick — {ticker}",
        xaxis_title="Data", yaxis_title="Preço (R$)",
        xaxis_rangeslider_visible=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _render_overview_tab():
    """📊 Home: aggregated metrics by segment."""
    st.header("📊 Visão Geral do Mercado")

    df = load_news(limit=5000)
    corr_df = load_correlations()

    if df.empty:
        st.info(
            "🔄 Nenhum dado disponível ainda. Execute o pipeline primeiro:\n"
            "```\npython main.py --stage all\n```"
        )
        return

    has_sentiment = (
        "sentiment" in df.columns
        and df["sentiment"].notna().any()
        and not df["sentiment"].eq("neutro").all()
    )

    # --- KPIs ---------------------------------------------------------------
    total = len(df)
    pos = int((df["sentiment"] == "positivo").sum()) if has_sentiment else 0
    neg = int((df["sentiment"] == "negativo").sum()) if has_sentiment else 0
    neu = int((df["sentiment"] == "neutro").sum()) if has_sentiment else total

    tickers_count = len(load_tickers())
    segs_count = len(
        {s for segs in df["segments"] for s in (segs if isinstance(segs, list) else [])}
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total de Notícias", total)
    c2.metric("Positivas", pos, delta=f"{pos/total*100:.1f}%" if total else "0%")
    c3.metric("Negativas", neg, delta=f"{neg/total*100:.1f}%" if total else "0%")
    c4.metric("Neutras", neu)
    c5.metric("Ativos Rastreados", tickers_count)

    if not has_sentiment:
        st.warning(
            "⚠️ Análise de sentimento ainda não executada. "
            "Configure `OPENAI_API_KEY` e rode `python main.py --stage trusted`."
        )

    st.divider()

    # --- Segment table + price variations -----------------------------------
    st.subheader("📦 Métricas por Segmento")

    seg_stats = get_segment_stats(df)
    seg_price = get_segment_price_stats(corr_df, df)

    if not seg_stats.empty:
        if not seg_price.empty:
            seg_stats = seg_stats.merge(seg_price[["segment", "avg_d0", "avg_d1", "avg_d5"]], on="segment", how="left")

        display = seg_stats.rename(columns={
            "segment": "Segmento", "total": "Notícias",
            "positivo": "Positivas", "negativo": "Negativas", "neutro": "Neutras",
            "pct_positivo": "% Pos", "pct_negativo": "% Neg",
            "confianca_media": "Confiança Méd.",
            "avg_d0": "Var. D0 (%)", "avg_d1": "Var. D+1 (%)", "avg_d5": "Var. D+5 (%)",
        })
        # select columns that actually exist
        show_cols = [c for c in [
            "Segmento", "Notícias", "Positivas", "% Pos", "Negativas", "% Neg",
            "Confiança Méd.", "Var. D0 (%)", "Var. D+1 (%)", "Var. D+5 (%)"
        ] if c in display.columns]
        st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

    # --- Charts row ---------------------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        bar = _segment_bar(df)
        if bar:
            st.plotly_chart(bar, use_container_width=True)
    with col_b:
        hm = _segment_sentiment_heatmap(df)
        if hm:
            st.plotly_chart(hm, use_container_width=True)
        elif not has_sentiment:
            st.info("Mapa de calor disponível após análise de sentimento.")

    # --- Sentiment over time ------------------------------------------------
    ot = _sentiment_over_time(df)
    if ot:
        st.plotly_chart(ot, use_container_width=True)


def _render_asset_tab():
    """📈 Asset-centric view: price chart + related news."""
    st.header("📈 Por Ativo")

    tickers = load_tickers()
    companies_df = load_companies()
    corr_df = load_correlations()

    if not tickers:
        st.info(
            "🔄 Nenhum dado de preço encontrado ainda. Execute:\n"
            "```\n# Backfill histórico desde 2025-01-01\n"
            "python main.py --stage backfill\n```"
        )
        return

    # Build ticker → company name lookup for enriched selectbox options
    ticker_to_name: Dict[str, str] = {}
    if not companies_df.empty and "ticker" in companies_df.columns and "name" in companies_df.columns:
        ticker_to_name = dict(zip(companies_df["ticker"], companies_df["name"]))

    options = [
        f"{t} — {ticker_to_name.get(t)}" if ticker_to_name.get(t) else t
        for t in tickers
    ]

    selected_option = st.selectbox(
        "🔍 Selecione o ativo (ticker ou nome da empresa)",
        options=options,
        key="asset_select",
    )
    if not selected_option:
        return

    # Extract the ticker from the combined label
    selected = selected_option.split(" — ")[0].strip()

    # Company info caption
    company_info = ticker_to_name.get(selected, "")
    if not company_info and not companies_df.empty:
        row_info = companies_df[companies_df["ticker"] == selected]
        if not row_info.empty:
            info = row_info.iloc[0]
            company_info = info.get("name", "")
            tipo = info.get("tipo_papel", "")
            if tipo:
                st.caption(f"**{company_info}** — {tipo}")
    elif company_info:
        tipo = ""
        if not companies_df.empty:
            row_info = companies_df[companies_df["ticker"] == selected]
            if not row_info.empty:
                tipo = row_info.iloc[0].get("tipo_papel", "")
        caption = f"**{company_info}**" + (f" — {tipo}" if tipo else "")
        st.caption(caption)

    price_df = load_asset_prices(selected)

    # --- Related news (load early to compute sentiment indicator) -----------
    news_df = load_news(ticker=selected, limit=100)

    # Detect the segments associated with this ticker from its news
    ticker_segments: List[str] = []
    if not news_df.empty and "segments" in news_df.columns:
        for segs in news_df["segments"]:
            if isinstance(segs, list):
                ticker_segments.extend(segs)
    ticker_segments = sorted(set(ticker_segments))

    # --- KPIs ---------------------------------------------------------------
    ticker_corr = corr_df[corr_df["ticker"] == selected] if not corr_df.empty else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    if not price_df.empty:
        last_close = price_df["close"].iloc[-1]
        prev_close = price_df["close"].iloc[-2] if len(price_df) > 1 else last_close
        delta_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0
        c1.metric("Último Fechamento", f"R$ {last_close:.2f}", delta=f"{delta_pct:+.2f}%")
        c2.metric("Mínima Histórica (período)", f"R$ {price_df['low'].min():.2f}")
        c3.metric("Máxima Histórica (período)", f"R$ {price_df['high'].max():.2f}")
    else:
        c1.metric("Último Fechamento", "N/A")
        c2.metric("Mínima", "N/A")
        c3.metric("Máxima", "N/A")

    news_count = len(ticker_corr) if not ticker_corr.empty else 0
    c4.metric("Notícias Relacionadas", news_count)

    # --- Sentiment indicator ------------------------------------------------
    st.subheader("🎯 Sentimento Geral do Ativo")
    sentiment_data = compute_asset_sentiment(news_df)
    if sentiment_data["count"] > 0:
        _render_sentiment_indicator(sentiment_data)
    else:
        st.info("Sem notícias suficientes para calcular o sentimento do ativo.")

    st.divider()

    # --- Price chart --------------------------------------------------------
    chart_type = st.radio(
        "Tipo de gráfico", ["Linha (Fechamento)", "Candlestick"],
        horizontal=True, key="chart_type",
    )
    if chart_type == "Candlestick":
        fig = _candlestick_chart(price_df, selected)
    else:
        fig = _price_chart(price_df, corr_df, selected)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    elif price_df.empty:
        st.info(f"Sem dados de preço para {selected}.")

    # --- Correlation metrics ------------------------------------------------
    if not ticker_corr.empty:
        st.subheader("📉 Correlação Notícias × Retorno de Preço")
        horizon = st.selectbox(
            "Horizonte", ["d1_var", "d0_var", "d5_var"],
            format_func=lambda x: {"d0_var": "D0 (intraday)", "d1_var": "D+1", "d5_var": "D+5"}[x],
            key="horizon_asset",
        )
        scatter = _scatter_sentiment_return(ticker_corr, horizon)
        if scatter:
            st.plotly_chart(scatter, use_container_width=True)

        avg_d1 = ticker_corr["d1_var"].dropna().mean() * 100 if "d1_var" in ticker_corr.columns else None
        avg_d5 = ticker_corr["d5_var"].dropna().mean() * 100 if "d5_var" in ticker_corr.columns else None
        cc1, cc2 = st.columns(2)
        if avg_d1 is not None:
            cc1.metric("Retorno Médio D+1", f"{avg_d1:.2f}%")
        if avg_d5 is not None:
            cc2.metric("Retorno Médio D+5", f"{avg_d5:.2f}%")

    # --- Fundamental indicators ---------------------------------------------
    st.divider()
    st.subheader("💹 Indicadores Fundamentalistas")

    fund_df = load_fundamentals(selected)
    macro_df = load_fundamentals("__MACRO__")

    # Helper: format a value according to its key
    def _fmt_fund(key: str, value: float) -> str:
        pct_keys = {"roe", "roa", "margem_liquida", "dy", "payout"}
        if key in pct_keys:
            return f"{value * 100:.2f}%"
        return f"{value:.2f}"

    if not fund_df.empty:
        # Group indicators by category
        _CATEGORIES = {
            "📊 Preço e Valuation": ["pl", "pvpa", "ev_ebitda"],
            "💰 Rentabilidade e Eficiência": ["roe", "roa", "margem_liquida"],
            "🏦 Saúde Financeira": ["divida_pl", "liquidez_corrente"],
            "💸 Remuneração ao Acionista": ["dy", "payout"],
        }

        fund_by_key = dict(zip(fund_df["key"], zip(fund_df["value"], fund_df["label"])))
        updated = fund_df["updated_at"].dropna().max() or "N/A"

        st.caption(f"Fonte: Yahoo Finance (yfinance) · Atualizado em: {updated}")

        for cat_name, keys in _CATEGORIES.items():
            items = [(key, fund_by_key[key]) for key in keys if key in fund_by_key]
            if not items:
                continue
            st.markdown(f"**{cat_name}**")
            cols = st.columns(len(items))
            for col, (key, (value, label)) in zip(cols, items):
                col.metric(label, _fmt_fund(key, value))

    else:
        st.info(
            "Sem dados fundamentalistas para este ativo. Execute:\n"
            "```\npython main.py --stage fundamentals\n```"
        )

    # --- Macro context (Selic / IPCA) ---------------------------------------
    if not macro_df.empty:
        st.markdown("**🌐 Contexto Macroeconômico**")
        macro_by_key = dict(zip(macro_df["key"], zip(macro_df["value"], macro_df["label"])))
        macro_items = [
            ("selic_meta", "Selic Meta (% a.a.)", lambda v: f"{v:.2f}%"),
            ("ipca_12m",   "IPCA 12 meses (%)",   lambda v: f"{v:.2f}%"),
        ]
        visible = [(label, fmt(macro_by_key[k][0]))
                   for k, label, fmt in macro_items if k in macro_by_key]
        if visible:
            mc = st.columns(len(visible))
            for col, (label, val_str) in zip(mc, visible):
                col.metric(label, val_str)

    # --- Direct ticker news -------------------------------------------------
    st.divider()
    st.subheader(f"📰 Notícias sobre {selected}")

    def _render_news_rows(df: pd.DataFrame, max_rows: int = 30) -> None:
        for _, row in df.head(max_rows).iterrows():
            pub = row["published_at"].strftime("%Y-%m-%d") if pd.notna(row["published_at"]) else "N/A"
            sent = row.get("sentiment", "neutro") or "neutro"
            color = {"positivo": "🟢", "negativo": "🔴", "neutro": "🟡"}.get(sent, "⚪")
            with st.expander(f"{color} {row['title']} — {row['source']} ({pub})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(row.get("content") or "")
                    st.markdown(f"[🔗 Leia mais]({row['url']})")
                with col2:
                    st.markdown(f"**Sentimento:** {sent}")
                    conf = row.get("confidence", 0)
                    if conf:
                        st.markdown(f"**Confiança:** {conf:.2f}")
                    segs = row.get("segments", [])
                    if segs:
                        st.markdown(f"**Segmentos:** {', '.join(segs)}")

    if news_df.empty:
        st.info("Nenhuma notícia encontrada para este ativo no período.")
    else:
        _render_news_rows(news_df)

    # --- Segment news -------------------------------------------------------
    if ticker_segments:
        st.divider()
        st.subheader(f"🏭 Notícias do Segmento: {', '.join(ticker_segments)}")

        # Collect the IDs of news already shown to avoid duplication
        shown_ids = set(news_df["id"].tolist()) if not news_df.empty else set()

        for segment in ticker_segments:
            seg_df = load_news(segment=segment, limit=50)
            # Exclude already shown news
            if not seg_df.empty and shown_ids:
                seg_df = seg_df[~seg_df["id"].isin(shown_ids)]

            if seg_df.empty:
                continue

            with st.expander(f"📂 Segmento: {segment} ({len(seg_df)} notícias)", expanded=False):
                seg_sentiment = compute_asset_sentiment(seg_df)
                if seg_sentiment["count"] > 0:
                    _render_sentiment_indicator(seg_sentiment)
                _render_news_rows(seg_df, max_rows=20)
    else:
        st.caption("ℹ️ Nenhum segmento identificado para este ativo com base nas notícias disponíveis.")


def _render_news_tab():
    """📰 Full news and X posts feed with sentiment filters."""
    st.header("📰 Notícias & Posts — Sentimento do Mercado")

    # --- Inline filters (previously in sidebar) ---------------------------------
    st.subheader("🔍 Filtros")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    try:
        conn = _conn()
        sources = pd.read_sql_query("SELECT DISTINCT source FROM news ORDER BY source", conn)["source"].tolist()
        conn.close()
        sources.insert(0, "Todas")
    except Exception:
        sources = ["Todas"]
    with filter_col1:
        source_filter = st.selectbox("Fonte", sources, key="news_source")

    try:
        conn = _conn()
        seg_df = pd.read_sql_query(
            "SELECT segments FROM news WHERE segments IS NOT NULL AND segments != '[]'", conn
        )
        conn.close()
        all_segs: List[str] = []
        for raw in seg_df["segments"]:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, list):
                    all_segs.extend(parsed)
            except Exception:
                pass
        unique_segs = ["Todos"] + sorted(set(all_segs))
    except Exception:
        unique_segs = ["Todos"]
    with filter_col2:
        segment_filter = st.selectbox("Segmento", unique_segs, key="news_seg")

    with filter_col3:
        sentiment_filter = st.selectbox(
            "Sentimento", ["Todos", "positivo", "negativo", "neutro"], key="news_sent"
        )

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        date_from = st.date_input("De", value=datetime.now() - timedelta(days=30), key="nf_from")
    with date_col2:
        date_to = st.date_input("Até", value=datetime.now(), key="nf_to")

    st.divider()

    # --- Load & metrics -----------------------------------------------------
    df = load_news(
        source=source_filter, sentiment=sentiment_filter,
        date_from=str(date_from), date_to=str(date_to),
        segment=segment_filter,
    )

    total = len(df)
    has_sent = total > 0 and "sentiment" in df.columns and df["sentiment"].notna().any()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    if has_sent:
        c2.metric("Positivas", int((df["sentiment"] == "positivo").sum()))
        c3.metric("Negativas", int((df["sentiment"] == "negativo").sum()))
        avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
        c4.metric("Confiança Méd.", f"{avg_conf:.2f}")
    else:
        c2.metric("Positivas", "N/A")
        c3.metric("Negativas", "N/A")
        c4.metric("Confiança Méd.", "N/A")

    # --- Charts -------------------------------------------------------------
    if not df.empty and has_sent:
        col_a, col_b = st.columns(2)
        with col_a:
            counts = df["sentiment"].value_counts()
            fig_pie = px.pie(
                values=counts.values, names=counts.index,
                title="Distribuição de Sentimentos",
                color=counts.index, color_discrete_map=SENTIMENT_COLORS,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            ot = _sentiment_over_time(df)
            if ot:
                st.plotly_chart(ot, use_container_width=True)

    # --- News list ----------------------------------------------------------
    st.divider()
    if df.empty:
        st.warning("Nenhuma notícia encontrada para os filtros aplicados.")
        return

    for _, row in df.head(50).iterrows():
        pub = row["published_at"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["published_at"]) else "N/A"
        sent = row.get("sentiment", "neutro") or "neutro"
        color = {"positivo": "🟢", "negativo": "🔴", "neutro": "🟡"}.get(sent, "⚪")
        source = row.get("source", "") or ""
        source_icon = "🐦" if source == "X (Twitter)" else "📄"
        link_label = "Ver post no X" if source == "X (Twitter)" else "Leia mais"
        with st.expander(f"{color} {source_icon} {row['title']} — {source} ({pub})"):
            col1, col2 = st.columns([3, 1])
            with col1:
                content = row.get("content") or ""
                if content and content != row["title"]:
                    st.markdown(content)
                st.markdown(f"[🔗 {link_label}]({row['url']})")
            with col2:
                st.markdown(f"**Sentimento:** {sent}")
                conf = row.get("confidence", 0)
                if conf:
                    st.markdown(f"**Confiança:** {conf:.2f}")
                segs = row.get("segments", [])
                if segs:
                    st.markdown(f"**Segmentos:** {', '.join(segs)}")
                tks = row.get("tickers", [])
                if tks:
                    st.markdown(f"**Tickers:** {', '.join(tks)}")
                st.markdown(f"**Publicado:** {pub}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_LABEL_COLORS = {
    "Medo Extremo":     "#d62728",
    "Medo":             "#ff7f0e",
    "Neutro":           "#aec7e8",
    "Ganância":         "#2ca02c",
    "Ganância Extrema": "#1a7f2e",
}


def _gauge_chart(score: float, label: str):
    """Render a gauge chart for the composite sentiment score."""
    color = _LABEL_COLORS.get(label, "#aec7e8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 20],  "color": "#f8d7da"},
                {"range": [20, 40], "color": "#ffe8cc"},
                {"range": [40, 60], "color": "#e8f4fd"},
                {"range": [60, 80], "color": "#d4edda"},
                {"range": [80, 100],"color": "#b8dfc5"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
        title={"text": f"Índice de Sentimento<br><b>{label}</b>"},
        number={"suffix": " / 100"},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=20, r=20))
    return fig


def _render_indicators_tab():
    """🧭 Sentiment indicators and composite Fear & Greed index."""
    st.header("🧭 Indicadores de Sentimento do Mercado")

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        date_from = st.date_input(
            "De", value=datetime.now() - timedelta(days=90), key="ind_from"
        )
    with col_date2:
        date_to = st.date_input("Até", value=datetime.now(), key="ind_to")

    date_from_str = str(date_from)
    date_to_str = str(date_to)

    # Load all composite index data (for the current-score gauge)
    composite_all_df = load_composite_index()
    # Load filtered composite index data (for historical charts)
    composite_df = load_composite_index(date_from=date_from_str, date_to=date_to_str)
    indicators_df = load_sentiment_indicators(date_from=date_from_str, date_to=date_to_str)

    if composite_all_df.empty and indicators_df.empty:
        st.info(
            "🔄 Nenhum dado de indicadores disponível ainda. Execute o pipeline:\n"
            "```\npython main.py --stage indicators\n```\n\n"
            "Para carregar histórico completo (≈1 ano necessário para o índice composto):\n"
            "```\npython main.py --stage indicators --from 2024-01-01\n```"
        )
        return

    # ------------------------------------------------------------------
    # Composite index gauge + time-series
    # ------------------------------------------------------------------
    if not composite_all_df.empty:
        # Gauge always reflects the most recent available score
        latest = composite_all_df.iloc[-1]
        score = latest.get("score", 50.0)
        label = latest.get("label", "Neutro")

        col_gauge, col_info = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(_gauge_chart(score, label), use_container_width=True)
        with col_info:
            st.subheader("Score Atual")
            st.metric("Índice Composto", f"{score:.1f} / 100", delta=label)
            st.caption(
                f"Última atualização: **{latest['date'].strftime('%d/%m/%Y') if pd.notna(latest['date']) else 'N/A'}**"
            )
            st.markdown("""
**Escala:**
- 🔴 **0–20** — Medo Extremo
- 🟠 **21–40** — Medo
- 🔵 **41–60** — Neutro
- 🟢 **61–80** — Ganância
- 🌿 **81–100** — Ganância Extrema
""")

        st.divider()

        # Time-series of composite index — uses date-filtered data
        chart_df = composite_df if not composite_df.empty else composite_all_df
        fig_idx = px.line(
            chart_df, x="date", y="score",
            title="Evolução do Índice de Sentimento (0 = Medo Extremo | 100 = Ganancia Extrema)",
            labels={"date": "Data", "score": "Score"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_idx.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutro")
        fig_idx.add_hrect(y0=0,  y1=20,  fillcolor="#f8d7da", opacity=0.15, line_width=0)
        fig_idx.add_hrect(y0=20, y1=40,  fillcolor="#ffe8cc", opacity=0.15, line_width=0)
        fig_idx.add_hrect(y0=60, y1=80,  fillcolor="#d4edda", opacity=0.15, line_width=0)
        fig_idx.add_hrect(y0=80, y1=100, fillcolor="#b8dfc5", opacity=0.15, line_width=0)
        st.plotly_chart(fig_idx, use_container_width=True)

        # Component scores — uses date-filtered data
        score_cols = [c for c in chart_df.columns if c.endswith("_score") and chart_df[c].notna().any()]
        if score_cols:
            st.subheader("📊 Componentes do Índice")
            fig_comp = go.Figure()
            labels_map = {
                "turnover_score": "Turnover",
                "trin_score": "TRIN (Arms)",
                "put_call_score": "Put/Call",
                "pct_advancing_score": "% Ações em Alta",
                "cdi_score": "CDI",
                "consumer_confidence_score": "Conf. Consumidor",
                "cds_score": "CDS Brasil",
            }
            for col in score_cols:
                name = labels_map.get(col, col)
                fig_comp.add_trace(go.Scatter(
                    x=chart_df["date"], y=chart_df[col],
                    mode="lines", name=name,
                ))
            fig_comp.add_hline(y=50, line_dash="dash", line_color="gray")
            fig_comp.update_layout(
                title="Scores por Componente (0–100)",
                xaxis_title="Data", yaxis_title="Score",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.divider()

    # ------------------------------------------------------------------
    # Raw indicators
    # ------------------------------------------------------------------
    if not indicators_df.empty:
        st.subheader("📈 Indicadores Brutos")

        indicator_labels = {
            "turnover":             "Turnover (Volume Total)",
            "trin":                 "TRIN – Arms Index",
            "put_call_ratio":       "Proporção Put/Call (PCR)",
            "pct_advancing":        "% Ações em Alta",
            "cdi_rate":             "Taxa CDI",
            "consumer_confidence":  "Confiança do Consumidor (ICC)",
            "cds_brasil_5y":        "CDS Brasil 5Y",
        }

        all_indicators = sorted(indicators_df["indicator"].unique())
        selected_inds = st.multiselect(
            "Indicadores",
            options=all_indicators,
            default=all_indicators[:4],
            format_func=lambda x: indicator_labels.get(x, x),
        )

        for ind in selected_inds:
            sub = indicators_df[indicators_df["indicator"] == ind].copy()
            if sub.empty:
                continue
            label = indicator_labels.get(ind, ind)
            fig_raw = px.line(
                sub, x="date", y="value",
                title=label,
                labels={"date": "Data", "value": label},
            )
            st.plotly_chart(fig_raw, use_container_width=True)

        # Raw data table
        with st.expander("📋 Dados brutos", expanded=False):
            pivot = indicators_df.pivot_table(
                index="date", columns="indicator", values="value"
            ).reset_index()
            pivot.columns.name = None
            pivot = pivot.rename(columns=indicator_labels)
            st.dataframe(pivot, use_container_width=True)

    st.divider()
    st.subheader("ℹ️ Sobre os Indicadores")
    st.markdown("""
Os indicadores são calculados a partir dos dados públicos disponíveis na biblioteca
[mercados](https://github.com/PythonicCafe/mercados):

| Indicador | Fonte | Interpretação |
|-----------|-------|---------------|
| **Turnover** | B3 – `negociacao_bolsa` | Volume total negociado. Alto volume sugere otimismo. |
| **TRIN (Arms Index)** | B3 – `negociacao_bolsa` | >1 = mais volume em ações caindo (pessimismo). |
| **Put/Call (PCR)** | B3 – `negociacao_bolsa` (opções) | >1 = mais opções de venda que de compra (pessimismo). |
| **% Ações em Alta** | B3 – `negociacao_bolsa` | Percentual de ações cujo fechamento > abertura. |
| **Taxa CDI** | BCB – SGS | Proxy para amplitude do DI. Taxas altas = restrição. |
| **Confiança Consumidor** | BCB – SGS (ICC, cód. 4393) | Índice FGV/Fecomercio. Maior = otimismo. |
| **CDS Brasil 5Y** | BCB – SGS (cód. 28229) | Risco-país. Maior = mais medo de calote. |

> **Nota metodológica:** o índice composto utiliza o método de *percentile rank* em janela
> móvel de 252 pregões (≈1 ano) para normalizar cada indicador a uma escala 0–100. São
> necessários pelo menos 10 observações históricas por componente para que ele entre no cálculo.
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="B3 Market Feeling Detector",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 B3 Market Feeling Detector")
    st.caption("Análise de sentimento de notícias financeiras brasileiras com dados de mercado da B3.")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "⚠️ `OPENAI_API_KEY` não configurada — análise de sentimento desabilitada. "
            "Configure o arquivo `.env` e reinicie."
        )

    tab_overview, tab_asset, tab_indicators, tab_news = st.tabs([
        "📊 Visão Geral",
        "📈 Por Ativo",
        "🧭 Indicadores",
        "📰 Notícias & Posts",
    ])

    with tab_overview:
        _render_overview_tab()

    with tab_asset:
        _render_asset_tab()

    with tab_indicators:
        _render_indicators_tab()

    with tab_news:
        _render_news_tab()


if __name__ == "__main__":
    main()
