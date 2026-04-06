"""
Streamlit Dashboard for Financial News Sentiment Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# Database connection
def get_db_connection(db_path: str = "data/news.db"):
    """Get database connection."""
    return sqlite3.connect(db_path)


def expand_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Expand rows with multiple segments into one row per segment."""
    if df.empty or 'segments' not in df.columns:
        return df.copy()

    rows = []
    for _, row in df.iterrows():
        segs = row['segments']
        if isinstance(segs, list) and segs:
            for seg in segs:
                new_row = row.copy()
                new_row['segment'] = seg
                rows.append(new_row)
        else:
            new_row = row.copy()
            new_row['segment'] = 'Sem segmento'
            rows.append(new_row)

    if not rows:
        return pd.DataFrame()

    expanded = pd.DataFrame(rows).reset_index(drop=True)
    return expanded


def get_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregated statistics per market segment."""
    if df.empty or 'segments' not in df.columns:
        return pd.DataFrame()

    expanded = expand_segments(df)
    if expanded.empty:
        return pd.DataFrame()

    stats = (
        expanded.groupby('segment')
        .agg(
            total=('id', 'count'),
            positivo=('sentiment', lambda x: (x == 'positivo').sum()),
            negativo=('sentiment', lambda x: (x == 'negativo').sum()),
            neutro=('sentiment', lambda x: (x == 'neutro').sum()),
            confianca_media=('confidence', 'mean'),
        )
        .reset_index()
        .sort_values('total', ascending=False)
    )
    stats['confianca_media'] = stats['confianca_media'].round(3)
    return stats


def create_segment_bar_chart(df: pd.DataFrame):
    """Create bar chart showing news distribution by segment."""
    stats = get_segment_stats(df)
    if stats.empty:
        return None

    fig = px.bar(
        stats,
        x='segment',
        y='total',
        title="Distribuição de Notícias por Segmento",
        color='total',
        color_continuous_scale='Blues',
        labels={'segment': 'Segmento', 'total': 'Número de Notícias'},
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    return fig


def create_segment_sentiment_heatmap(df: pd.DataFrame):
    """Create heatmap of segment vs sentiment."""
    if df.empty or 'segments' not in df.columns:
        return None

    expanded = expand_segments(df)
    if expanded.empty:
        return None

    pivot = pd.crosstab(expanded['segment'], expanded['sentiment'])

    fig = px.imshow(
        pivot,
        title="Segmento vs Sentimento",
        color_continuous_scale="Blues",
        aspect="auto",
        labels={'x': 'Sentimento', 'y': 'Segmento', 'color': 'Contagem'},
    )
    fig.update_layout(xaxis_title="Sentimento", yaxis_title="Segmento")
    return fig


def create_segment_sentiment_over_time(df: pd.DataFrame, segment: Optional[str] = None):
    """Create time-series of sentiment counts by segment (or for a specific segment)."""
    if df.empty or 'segments' not in df.columns or 'published_at' not in df.columns:
        return None

    expanded = expand_segments(df)
    if expanded.empty:
        return None

    if segment and segment != "Todos":
        expanded = expanded[expanded['segment'] == segment]

    if expanded.empty:
        return None

    expanded = expanded.copy()
    expanded['date'] = expanded['published_at'].dt.date

    series = (
        expanded.groupby(['date', 'segment', 'sentiment'])
        .size()
        .reset_index(name='count')
    )

    fig = px.line(
        series,
        x='date',
        y='count',
        color='segment',
        line_dash='sentiment',
        title="Evolução de Sentimentos por Segmento ao Longo do Tempo",
        labels={'date': 'Data', 'count': 'Número de Notícias', 'segment': 'Segmento'},
    )
    fig.update_layout(xaxis_title="Data", yaxis_title="Número de Notícias", showlegend=True)
    return fig


def load_news_data(
    source_filter: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    segment_filter: Optional[str] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """Load news data from database with filters."""

    # Check if sentiment columns exist
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(news)")
    columns = [row[1] for row in cursor.fetchall()]
    has_sentiment = 'sentiment' in columns
    has_confidence = 'confidence' in columns
    has_segments = 'segments' in columns
    has_tickers = 'tickers' in columns
    conn.close()

    if has_sentiment and has_confidence and has_segments and has_tickers:
        query = """
        SELECT
            id,
            title,
            content,
            source,
            published_at,
            url,
            collected_at,
            sentiment,
            confidence,
            segments,
            tickers
        FROM news
        WHERE 1=1
        """
    elif has_sentiment and has_confidence:
        query = """
        SELECT
            id,
            title,
            content,
            source,
            published_at,
            url,
            collected_at,
            sentiment,
            confidence,
            '[]' as segments,
            '[]' as tickers
        FROM news
        WHERE 1=1
        """
    else:
        query = """
        SELECT
            id,
            title,
            content,
            source,
            published_at,
            url,
            collected_at,
            'neutro' as sentiment,
            0.0 as confidence,
            '[]' as segments,
            '[]' as tickers
        FROM news
        WHERE 1=1
        """

    params = []

    if source_filter and source_filter != "Todas":
        query += " AND source = ?"
        params.append(source_filter)

    if sentiment_filter and sentiment_filter != "Todos" and has_sentiment:
        query += " AND sentiment = ?"
        params.append(sentiment_filter)

    if date_from:
        query += " AND published_at >= ?"
        params.append(date_from)

    if date_to:
        query += " AND published_at <= ?"
        params.append(date_to)

    query += " ORDER BY published_at DESC LIMIT ?"
    params.append(limit)

    try:
        conn = get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert dates
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df['collected_at'] = pd.to_datetime(df['collected_at'], errors='coerce')
        
        # Parse JSON fields
        if 'segments' in df.columns:
            df['segments'] = df['segments'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
        if 'tickers' in df.columns:
            df['tickers'] = df['tickers'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])

        # Apply segment filter (post-filter since segments is a JSON column)
        if segment_filter and segment_filter not in ("Todos", "Sem segmento") and 'segments' in df.columns:
            df = df[df['segments'].apply(lambda segs: segment_filter in segs if isinstance(segs, list) else False)]

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()


def get_sentiment_stats(df: pd.DataFrame) -> Dict:
    """Calculate sentiment statistics."""
    if df.empty:
        return {
            'total': 0,
            'positivo': 0,
            'negativo': 0,
            'neutro': 0,
            'avg_confidence': 0.0,
            'has_sentiment': False
        }

    # Check if sentiment data exists (real data, not just default 'neutro')
    has_sentiment = 'sentiment' in df.columns and df['sentiment'].notna().any() and not df['sentiment'].eq('neutro').all()
    has_confidence = 'confidence' in df.columns and df['confidence'].notna().any() and (df['confidence'] > 0).any()

    if has_sentiment:
        stats = {
            'total': len(df),
            'positivo': len(df[df['sentiment'] == 'positivo']),
            'negativo': len(df[df['sentiment'] == 'negativo']),
            'neutro': len(df[df['sentiment'] == 'neutro']),
            'avg_confidence': df['confidence'].mean() if has_confidence else 0.0,
            'has_sentiment': True
        }
    else:
        stats = {
            'total': len(df),
            'positivo': 0,
            'negativo': 0,
            'neutro': len(df),  # All are neutral by default
            'avg_confidence': 0.0,
            'has_sentiment': False
        }

    return stats


def check_sentiment_availability() -> bool:
    """Check if sentiment data is available in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(news)")
        columns = [row[1] for row in cursor.fetchall()]
        has_sentiment = 'sentiment' in columns

        if has_sentiment:
            # Check if there's any non-null sentiment data
            cursor.execute("SELECT COUNT(*) FROM news WHERE sentiment IS NOT NULL")
            count = cursor.fetchone()[0]
            has_sentiment = count > 0

        conn.close()
        return has_sentiment
    except:
        return False


def create_sentiment_pie_chart(df: pd.DataFrame):
    """Create sentiment distribution pie chart."""
    if df.empty:
        return None

    sentiment_counts = df['sentiment'].value_counts()

    colors = {
        'positivo': '#00ff00',
        'negativo': '#ff0000',
        'neutro': '#ffff00'
    }

    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Distribuição de Sentimentos",
        color=sentiment_counts.index,
        color_discrete_map=colors
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_confidence_histogram(df: pd.DataFrame):
    """Create confidence score histogram."""
    if df.empty or 'confidence' not in df.columns:
        return None

    fig = px.histogram(
        df,
        x='confidence',
        nbins=20,
        title="Distribuição de Confiança",
        color='sentiment',
        color_discrete_map={
            'positivo': '#00ff00',
            'negativo': '#ff0000',
            'neutro': '#ffff00'
        }
    )

    fig.update_layout(
        xaxis_title="Pontuação de Confiança",
        yaxis_title="Contagem",
        showlegend=True
    )

    return fig


def create_sentiment_over_time(df: pd.DataFrame):
    """Create sentiment trend over time."""
    if df.empty or 'published_at' not in df.columns:
        return None

    # Group by date and sentiment
    df_time = df.copy()
    df_time['date'] = df_time['published_at'].dt.date

    sentiment_over_time = df_time.groupby(['date', 'sentiment']).size().reset_index(name='count')

    fig = px.line(
        sentiment_over_time,
        x='date',
        y='count',
        color='sentiment',
        title="Evolução de Sentimentos ao Longo do Tempo",
        color_discrete_map={
            'positivo': '#00ff00',
            'negativo': '#ff0000',
            'neutro': '#ffff00'
        }
    )

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Número de Notícias",
        showlegend=True
    )

    return fig


def create_source_sentiment_heatmap(df: pd.DataFrame):
    """Create source vs sentiment heatmap."""
    if df.empty:
        return None

    # Create pivot table
    pivot = pd.crosstab(df['source'], df['sentiment'])

    fig = px.imshow(
        pivot,
        title="Fonte vs Sentimento",
        color_continuous_scale="Blues",
        aspect="auto"
    )

    fig.update_layout(
        xaxis_title="Sentimento",
        yaxis_title="Fonte"
    )

    return fig


def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="B3 Market Feeling Detector",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 B3 Market Feeling Detector")
    st.markdown("Dashboard de Análise de Sentimento de Notícias Financeiras")

    # API key warning
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ A variável de ambiente OPENAI_API_KEY não está configurada. Configure o arquivo .env e reinicie o dashboard para habilitar análise de sentimento.")

    # Sidebar filters
    st.sidebar.header("🔍 Filtros")

    # Check if sentiment data is available
    has_sentiment_data = check_sentiment_availability()

    # Source filter
    try:
        conn = get_db_connection()
        sources = pd.read_sql_query("SELECT DISTINCT source FROM news ORDER BY source", conn)['source'].tolist()
        conn.close()
        sources.insert(0, "Todas")
    except:
        sources = ["Todas"]

    source_filter = st.sidebar.selectbox("Fonte", sources)

    # Sentiment filter
    if has_sentiment_data:
        sentiment_options = ["Todos", "positivo", "negativo", "neutro"]
        sentiment_filter = st.sidebar.selectbox("Sentimento", sentiment_options)
    else:
        st.sidebar.info("🎯 Filtro de sentimento disponível após análise completa")
        sentiment_filter = "Todos"

    # Segment filter (populated from all available segments in the database)
    try:
        conn = get_db_connection()
        seg_df = pd.read_sql_query("SELECT segments FROM news WHERE segments IS NOT NULL AND segments != '[]'", conn)
        conn.close()
        all_segments: List[str] = []
        for raw in seg_df['segments']:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, list):
                    all_segments.extend(parsed)
            except Exception:
                pass
        unique_segments = sorted(set(all_segments))
        unique_segments.insert(0, "Todos")
    except Exception:
        unique_segments = ["Todos"]

    segment_filter = st.sidebar.selectbox("Segmento", unique_segments)

    # Date filters
    col1, col2 = st.sidebar.columns(2)

    with col1:
        date_from = st.date_input(
            "Data Inicial",
            value=datetime.now() - timedelta(days=7),
            key="date_from"
        )

    with col2:
        date_to = st.date_input(
            "Data Final",
            value=datetime.now(),
            key="date_to"
        )

    # Convert dates to string format
    date_from_str = date_from.strftime("%Y-%m-%d") if date_from else None
    date_to_str = date_to.strftime("%Y-%m-%d") if date_to else None

    # Load data
    df = load_news_data(
        source_filter=source_filter,
        sentiment_filter=sentiment_filter,
        date_from=date_from_str,
        date_to=date_to_str,
        segment_filter=segment_filter,
    )

    # Statistics
    stats = get_sentiment_stats(df)

    # Main content
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total de Notícias", stats['total'])

    with col2:
        if stats['has_sentiment']:
            st.metric("Positivo", stats['positivo'], delta=f"{stats['positivo']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%")
        else:
            st.metric("Positivo", "N/A", delta="Análise pendente")

    with col3:
        if stats['has_sentiment']:
            st.metric("Negativo", stats['negativo'], delta=f"{stats['negativo']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%")
        else:
            st.metric("Negativo", "N/A", delta="Análise pendente")

    with col4:
        if stats['has_sentiment']:
            st.metric("Neutro", stats['neutro'], delta=f"{stats['neutro']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%")
        else:
            st.metric("Neutro", stats['total'], delta="Dados básicos")

    with col5:
        if stats['has_sentiment']:
            st.metric("Confiança Média", f"{stats['avg_confidence']:.2f}")
        else:
            st.metric("Confiança Média", "N/A", delta="Análise pendente")

    # Warning if no sentiment data
    if not stats['has_sentiment']:
        st.warning("⚠️ **Análise de sentimento não executada ainda.** Configure a API key do OpenAI e execute o pipeline completo para ver as análises de sentimento.")

    # Charts
    if not df.empty and stats['has_sentiment']:
        st.header("📊 Visualizações")

        col1, col2 = st.columns(2)

        with col1:
            pie_chart = create_sentiment_pie_chart(df)
            if pie_chart:
                st.plotly_chart(pie_chart, use_container_width=True)

        with col2:
            hist_chart = create_confidence_histogram(df)
            if hist_chart:
                st.plotly_chart(hist_chart, use_container_width=True)

        # Time series chart
        time_chart = create_sentiment_over_time(df)
        if time_chart:
            st.plotly_chart(time_chart, use_container_width=True)

        # Source vs sentiment heatmap
        heatmap = create_source_sentiment_heatmap(df)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
    elif not df.empty:
        st.header("📊 Visualizações")
        st.info("📈 Execute a análise de sentimento para ver gráficos e visualizações detalhadas.")

    # Segment analysis section
    st.header("📦 Análise por Segmento")

    segment_stats = get_segment_stats(df)

    if segment_stats.empty:
        st.info("Nenhum dado de segmento disponível para as notícias filtradas.")
    else:
        # Summary cards
        num_segments = len(segment_stats)
        top_segment = segment_stats.iloc[0]['segment']
        avg_conf_seg = segment_stats['confianca_media'].mean()

        card_col1, card_col2, card_col3 = st.columns(3)
        with card_col1:
            st.metric("Segmentos Identificados", num_segments)
        with card_col2:
            st.metric("Segmento Mais Frequente", top_segment)
        with card_col3:
            if avg_conf_seg > 0:
                st.metric("Confiança Média (segmentos)", f"{avg_conf_seg:.3f}")
            else:
                st.metric("Confiança Média (segmentos)", "N/A")

        # Charts row
        seg_col1, seg_col2 = st.columns(2)

        with seg_col1:
            bar_chart = create_segment_bar_chart(df)
            if bar_chart:
                st.plotly_chart(bar_chart, use_container_width=True)

        with seg_col2:
            seg_heatmap = create_segment_sentiment_heatmap(df)
            if seg_heatmap:
                st.plotly_chart(seg_heatmap, use_container_width=True)
            elif not stats['has_sentiment']:
                st.info("Mapa de calor disponível após análise de sentimento.")

        # Temporal series
        time_seg_chart = create_segment_sentiment_over_time(df, segment=segment_filter)
        if time_seg_chart:
            st.plotly_chart(time_seg_chart, use_container_width=True)

        # Aggregated statistics table
        st.subheader("📋 Estatísticas por Segmento")
        display_stats = segment_stats.rename(columns={
            'segment': 'Segmento',
            'total': 'Total',
            'positivo': 'Positivo',
            'negativo': 'Negativo',
            'neutro': 'Neutro',
            'confianca_media': 'Confiança Média',
        })
        st.dataframe(display_stats, use_container_width=True, hide_index=True)

    # News table
    st.header("📰 Notícias")

    if df.empty:
        st.warning("Nenhuma notícia encontrada com os filtros aplicados.")
    else:
        # Format dataframe for display
        display_df = df.copy()
        display_df['published_at'] = display_df['published_at'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['collected_at'] = display_df['collected_at'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['confidence'] = display_df['confidence'].round(3)

        # Add sentiment colors
        def color_sentiment(val):
            if val == 'positivo':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'negativo':
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'

        styled_df = display_df.style.apply(
            lambda x: [color_sentiment(val) if col == 'sentiment' else '' for col, val in x.items()],
            axis=1
        )

        # Display table with expandable content
        for idx, row in display_df.iterrows():
            with st.expander(f"📰 {row['title']} - {row['source']} ({row['sentiment']})"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Conteúdo:** {row['content'] or 'N/A'}")
                    st.markdown(f"**URL:** [{row['url']}]({row['url']})")

                with col2:
                    st.markdown(f"**Fonte:** {row['source']}")
                    st.markdown(f"**Sentimento:** {row['sentiment']}")
                    st.markdown(f"**Confiança:** {row['confidence']:.3f}")
                    if 'segments' in row and row['segments']:
                        st.markdown(f"**Segmentos:** {', '.join(row['segments'])}")
                    if 'tickers' in row and row['tickers']:
                        st.markdown(f"**Tickers:** {', '.join(row['tickers'])}")
                    st.markdown(f"**Publicado:** {row['published_at']}")
                    st.markdown(f"**Coletado:** {row['collected_at']}")

    # Footer
    st.markdown("---")
    st.markdown("💡 **Dicas:**")
    st.markdown("- Use os filtros na barra lateral para explorar os dados")
    st.markdown("- Clique nas notícias para ver o conteúdo completo")
    st.markdown("- Os gráficos mostram tendências e distribuições dos sentimentos")


if __name__ == "__main__":
    main()