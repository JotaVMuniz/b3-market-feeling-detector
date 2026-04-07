# B3 Market Feeling Detector

Pipeline de ingestão e análise de sentimento de notícias financeiras brasileiras e posts do X (Twitter), com dados históricos de preços da B3 e dashboard interativo orientado por ativos.

---

## Funcionalidades

- **Ingestão** de notícias de RSS (InfoMoney, Valor Econômico, Exame)
- **Ingestão de posts do X (Twitter)** via API v2 — busca os últimos 7 dias sobre o mercado financeiro brasileiro em português
- **Enriquecimento NLP** via OpenAI GPT-4o-mini: sentimento, tickers, segmentos
- **Limpeza automática** de notícias e posts com sentimento neutro e mais de 7 dias
- **Preços históricos B3** desde 2025-01-01 (todos os ativos)
- **Correlações** notícia × variação de preço (D0, D+1, D+5)
- **Dashboard** orientado por ativo com visão geral por segmento — notícias e posts do X na mesma aba

---

## Estrutura do Projeto

```
b3-market-feeling-detector/
├── src/
│   ├── ingestion/          # Busca RSS e posts do X (Twitter)
│   ├── processing/         # Limpeza e normalização de texto
│   ├── nlp/                # Sentimento e enriquecimento (OpenAI)
│   ├── storage/            # SQLite + JSON raw
│   └── market_data/        # Preços B3, empresas, correlações
├── tests/                  # Suite de testes (pytest)
├── main.py                 # Orquestrador do pipeline
├── dashboard.py            # Dashboard Streamlit
├── Dockerfile              # Imagem para Cloud Run
└── requirements.txt
```

---

## Instalação

```bash
pip install -r requirements.txt
cp .env.example .env   # preencha OPENAI_API_KEY e TWITTER_BEARER_TOKEN
```

---

## Variáveis de Ambiente

| Variável | Descrição |
|----------|-----------|
| `OPENAI_API_KEY` | Chave OpenAI (obrigatória para NLP) |
| `TWITTER_BEARER_TOKEN` | Bearer Token da API do X — obrigatório para ingestão de posts do X |
| `DB_PATH` | Caminho do banco SQLite (padrão: `data/news.db`) |

---

## Configuração da API do X (Twitter)

Para ativar a ingestão de posts do X (Twitter):

1. Acesse o [X Developer Portal](https://developer.twitter.com/) e crie uma conta de desenvolvedor gratuita.
2. Crie um novo projeto e um app dentro dele.
3. Vá em **Keys and Tokens** do seu app e gere o **Bearer Token**.
4. Adicione o token ao arquivo `.env`:
   ```
   TWITTER_BEARER_TOKEN=seu_bearer_token_aqui
   ```

### Limites do plano gratuito

| Métrica | Limite (Free) |
|---------|---------------|
| Requisições ao endpoint de busca recente | 100/mês |
| Resultados por requisição | Até 100 tweets |
| Janela de tempo disponível | Últimos 7 dias |

> **Nota:** Sem `TWITTER_BEARER_TOKEN`, a etapa `tweets` é ignorada com um aviso no log e o restante do pipeline continua normalmente.

---

## Pipeline

```bash
# Backfill histórico de preços (executar uma vez)
python main.py --stage backfill --from 2025-01-01

# Execução diária completa
python main.py --stage all

# Estágios individuais
python main.py --stage raw          # busca notícias RSS
python main.py --stage tweets       # busca posts do X (Twitter)
python main.py --stage trusted      # sentimento/NLP (notícias + posts)
python main.py --stage cleanup      # remove notícias neutras com > 7 dias
python main.py --stage prices       # preços do dia
python main.py --stage analytics    # correlações
```

### Sequência completa do pipeline

```
raw → tweets → trusted → cleanup → prices → indicators → analytics
```

---

## Dashboard

```bash
streamlit run dashboard.py
```

Quatro abas:

| Aba | Conteúdo |
|-----|----------|
| **📊 Visão Geral** | Métricas agregadas por segmento, variação média de preço, heatmap sentimento × segmento |
| **📈 Por Ativo** | Selector de ticker, gráfico de preços (linha ou candlestick), notícias relacionadas, retornos D0/D+1/D+5 |
| **🧭 Indicadores** | Fear & Greed Index composto e indicadores brutos (TRIN, PCR, CDI, etc.) |
| **📰 Notícias & Posts** | Feed unificado de notícias RSS e posts do X com filtros de fonte, segmento, sentimento e período. Posts do X são identificados com o ícone 🐦. |

---

## Arquitetura GCP (execução diária)

```mermaid
flowchart TD
    subgraph Agendamento
        CS[Cloud Scheduler\nCron diário]
    end

    subgraph Pipeline["Cloud Run Job — pipeline"]
        R[Stage: raw\nRSS → limpeza]
        TW[Stage: tweets\nX API → posts]
        T[Stage: trusted\nNLP / OpenAI]
        CL[Stage: cleanup\nneutro > 7 dias]
        P[Stage: prices\nB3 cotações]
        A[Stage: analytics\ncorrelações]
        R --> TW --> T --> CL --> P --> A
    end

    subgraph Armazenamento
        GCS[Cloud Storage\nJSON raw diário]
        SQL[Cloud SQL\nPostgreSQL\nnews + prices]
    end

    subgraph Dashboard
        CR[Cloud Run\nStreamlit]
    end

    subgraph Segurança
        SM[Secret Manager\nOPENAI_API_KEY\nTWITTER_BEARER_TOKEN]
    end

    CS -->|trigger| Pipeline
    R -->|salva JSON| GCS
    A -->|upsert| SQL
    SM -->|inject env| Pipeline
    CR -->|lê| SQL
    Usuario([Usuário]) --> CR
```

### Componentes

| Componente | Serviço GCP | Papel |
|------------|-------------|-------|
| Agendamento diário | Cloud Scheduler | Aciona o job às 7h |
| Execução do pipeline | Cloud Run Job | Container `main.py --stage all` |
| Armazenamento raw | Cloud Storage | Snapshots JSON diários |
| Banco de dados | Cloud SQL (PostgreSQL) | Notícias, posts, preços, correlações |
| Dashboard | Cloud Run (Streamlit) | Servido via HTTPS |
| Segredos | Secret Manager | `OPENAI_API_KEY`, `TWITTER_BEARER_TOKEN`, conexão DB |
| Imagem Docker | Artifact Registry | Versões do container |

> Para migrar de SQLite para Cloud SQL, substitua a string de conexão em
> `src/storage/database.py` e `src/market_data/database_market.py` por uma
> URL PostgreSQL injetada via Secret Manager.

---

## Testes

```bash
pytest
```

---

## Licença

MIT — © 2025 JotaVMuniz
