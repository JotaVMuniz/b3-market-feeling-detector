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

    # Sidebar selector
    selected = st.selectbox("Selecione o ativo", options=tickers, key="asset_select")
    if not selected:
        return

    # Company info
    if not companies_df.empty:
        row = companies_df[companies_df["ticker"] == selected]
        if not row.empty:
            info = row.iloc[0]
            st.caption(f"**{info.get('name', selected)}** — {info.get('tipo_papel', '')}")

    price_df = load_asset_prices(selected)

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

    # --- Related news -------------------------------------------------------
    st.subheader(f"📰 Notícias sobre {selected}")
    news_df = load_news(ticker=selected, limit=100)

    if news_df.empty:
        st.info("Nenhuma notícia encontrada para este ativo no período.")
    else:
        for _, row in news_df.head(30).iterrows():
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
                    st.markdown(f"**Confiança:** {conf:.2f}" if conf else "")
                    segs = row.get("segments", [])
                    if segs:
                        st.markdown(f"**Segmentos:** {', '.join(segs)}")


def _render_news_tab():
    """📰 Full news feed with sentiment filters."""
    st.header("📰 Notícias & Sentimento")

    # --- Sidebar filters ----------------------------------------------------
    st.sidebar.header("🔍 Filtros")

    try:
        conn = _conn()
        sources = pd.read_sql_query("SELECT DISTINCT source FROM news ORDER BY source", conn)["source"].tolist()
        conn.close()
        sources.insert(0, "Todas")
    except Exception:
        sources = ["Todas"]
    source_filter = st.sidebar.selectbox("Fonte", sources, key="news_source")

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
    segment_filter = st.sidebar.selectbox("Segmento", unique_segs, key="news_seg")

    sentiment_filter = st.sidebar.selectbox(
        "Sentimento", ["Todos", "positivo", "negativo", "neutro"], key="news_sent"
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        date_from = st.date_input("De", value=datetime.now() - timedelta(days=30), key="nf_from")
    with col2:
        date_to = st.date_input("Até", value=datetime.now(), key="nf_to")

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
                tks = row.get("tickers", [])
                if tks:
                    st.markdown(f"**Tickers:** {', '.join(tks)}")
                st.markdown(f"**Publicado:** {pub}")


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

    tab_overview, tab_asset, tab_news = st.tabs([
        "📊 Visão Geral",
        "📈 Por Ativo",
        "📰 Notícias",
    ])

    with tab_overview:
        _render_overview_tab()

    with tab_asset:
        _render_asset_tab()

    with tab_news:
        _render_news_tab()


if __name__ == "__main__":
    main()
