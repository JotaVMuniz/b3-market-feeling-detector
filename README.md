# B3 Market Feeling Detector

Pipeline de ingestão e análise de sentimento de notícias financeiras brasileiras, com dados históricos de preços da B3, indicadores fundamentalistas e dashboard interativo orientado por ativos.

---

## Funcionalidades

- **Ingestão** de notícias de RSS (InfoMoney, Valor Econômico, Exame)
- **Enriquecimento NLP** via OpenAI GPT-4o-mini: sentimento, relevância de mercado (0–1), tickers e segmentos
- **Limpeza automática** de notícias com sentimento neutro e mais de 7 dias
- **Preços históricos B3** desde 2025-01-01 (todos os ativos)
- **Indicadores fundamentalistas** (P/L, P/VPA, EV/EBITDA, ROE, Dividend Yield, etc.) via Fundamentus + yfinance — universo IBrX 100
- **Indicadores macroeconômicos** (Selic meta e IPCA 12m) via Banco Central
- **Correlações** notícia × variação de preço (D0, D+1, D+5)
- **Dashboard** com rank geral das ações por dados fundamentalistas, top 10 notícias mais relevantes, visão por ativo e feed completo de notícias

---

## Estrutura do Projeto

```
b3-market-feeling-detector/
├── src/
│   ├── ingestion/          # Busca RSS
│   ├── processing/         # Limpeza e normalização de texto
│   ├── nlp/                # Sentimento e enriquecimento (OpenAI)
│   ├── storage/            # SQLite + JSON raw
│   └── market_data/        # Preços B3, empresas, correlações, fundamentalistas
├── scripts/                # Ferramentas auxiliares de análise e validação (ver seção própria)
├── tests/                  # Suite de testes (pytest)
├── main.py                 # Orquestrador do pipeline
├── dashboard.py            # Dashboard Streamlit
├── Dockerfile              # Imagem para rodar o dashboard Streamlit em contêiner
└── requirements.txt
```

---

## Instalação

```bash
pip install -r requirements.txt
cp .env.example .env   # preencha OPENAI_API_KEY
```

---

## Variáveis de Ambiente

| Variável | Descrição |
|----------|-----------|
| `OPENAI_API_KEY` | Chave OpenAI (obrigatória para NLP) |
| `DB_PATH` | Caminho do banco SQLite (padrão: `data/news.db`) |

---

## Pipeline

O pipeline é organizado em três camadas — **raw**, **trusted** e **analytics** —
no espírito de uma arquitetura medallion/ETL estendida. Cada camada é um
`--stage`; a flag `--only` seleciona uma única sub-etapa dentro de `trusted`
ou `analytics` para execução granular.

```bash
# Backfill histórico de preços (executar uma vez)
python main.py --stage backfill --from 2025-01-01

# Execução diária completa (raw → trusted → analytics)
python main.py --stage all

# Camada RAW — ingestão bruta de notícias via RSS
python main.py --stage raw

# Camada TRUSTED — validação, curadoria e enriquecimento
python main.py --stage trusted                        # todas as sub-etapas abaixo
python main.py --stage trusted --only enrichment       # sentimento/NLP + relevância de mercado
python main.py --stage trusted --only cleanup          # remove notícias neutras com > 7 dias
python main.py --stage trusted --only ibrx             # universo IBrX 100
python main.py --stage trusted --only prices           # preços do dia
python main.py --stage trusted --only fundamentals     # indicadores fundamentalistas (universo IBrX 100)
python main.py --stage trusted --only fundamentals --tickers PETR4,VALE3
python main.py --stage trusted --only indicators       # série bruta dos indicadores de sentimento

# Camada ANALYTICS — métricas derivadas do dado trusted
python main.py --stage analytics                       # todas as sub-etapas abaixo
python main.py --stage analytics --only composite-index  # recalcula o índice Fear & Greed (sem rede)
python main.py --stage analytics --only correlation      # correlações notícia × preço
```

### Camadas do pipeline

| Camada | Sub-etapa | Descrição |
|---|---|---|
| **raw** | `raw` | Coleta bruta de notícias financeiras via feeds RSS |
| **trusted** | `enrichment` | Enriquecimento semântico das notícias via LLM (sentimento, tickers, relevância) |
| **trusted** | `cleanup` | Remove notícias de sentimento neutro após o período de retenção (7 dias) |
| **trusted** | `ibrx` | Atualização dos constituintes do índice IBrX 100 |
| **trusted** | `prices` | Coleta dos preços históricos diários dos ativos |
| **trusted** | `fundamentals` | Coleta dos indicadores fundamentalistas e macroeconômicos |
| **trusted** | `indicators` | Coleta da série bruta dos indicadores de sentimento de mercado |
| **analytics** | `composite-index` | Cálculo do índice composto de sentimento (Fear & Greed), a partir do dado já coletado |
| **analytics** | `correlation` | Cálculo das variações de preço associadas a notícias (D0/D+1/D+5) |

Separar `indicators` (trusted, faz chamadas de rede à B3/BCB) de
`composite-index` (analytics, apenas recalcula sobre o que já está no banco)
permite recalcular o índice — por exemplo após ajustar os pesos — sem repetir
a coleta.

---

## Dashboard

```bash
streamlit run dashboard.py
```

Quatro abas:

| Aba | Conteúdo |
|-----|----------|
| **📊 Visão Geral** | KPIs agregados, rank geral das ações por dados fundamentalistas (ROE, DY, P/L, etc.) e top 10 notícias mais relevantes para o mercado |
| **📈 Por Ativo** | Selector de ticker, gráfico de preços (linha ou candlestick), indicadores fundamentalistas, notícias relacionadas, retornos D0/D+1/D+5 |
| **🧭 Indicadores** | Fear & Greed Index composto e indicadores brutos (TRIN, PCR, CDI, etc.) |
| **📰 Notícias** | Feed de notícias RSS com filtros de fonte, segmento, sentimento e período |

### Enriquecimento de Notícias

A sub-etapa `trusted --only enrichment` enriquece cada notícia com os seguintes campos via OpenAI GPT-4o-mini:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `is_relevant` | bool | Se a notícia é financeiramente relevante |
| `market_relevance` | float 0–1 | Índice de relevância para o mercado financeiro brasileiro |
| `sentiment` | string | `positivo`, `negativo` ou `neutro` |
| `confidence` | float 0–1 | Confiança do sentimento atribuído |
| `segments` | list | Segmentos de mercado identificados |
| `tickers` | list | Tickers de ações brasileiras mencionados |

---

## Execução Diária

O pipeline roda localmente uma vez por dia via um agendador do próprio sistema
operacional, executando `python main.py --stage all` e registrando log em
`logs/pipeline_daily.log`. Não há, no momento, uma arquitetura de implantação em
nuvem — o projeto tem escopo acadêmico e roda em ambiente local.

### macOS (launchd)

1. Copie o template `deploy/com.b3marketfeeling.dailypipeline.plist.example` para
   `~/Library/LaunchAgents/com.b3marketfeeling.dailypipeline.plist`.
2. Nele, substitua os placeholders `/ABSOLUTE/PATH/TO/...` pelo caminho absoluto
   do seu clone e pelo Python do seu virtualenv (`.venv/bin/python3`, já com
   `requirements.txt` instalado). `WorkingDirectory` precisa apontar para a raiz
   do repositório — é de lá que `load_dotenv()` lê a `OPENAI_API_KEY` do `.env`.
3. Carregue o agente:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.b3marketfeeling.dailypipeline.plist
   ```
4. Para testar imediatamente, sem esperar o horário agendado (7h por padrão):
   ```bash
   launchctl start com.b3marketfeeling.dailypipeline
   ```
5. Para verificar se está carregado e ver o último código de saída:
   ```bash
   launchctl list | grep b3marketfeeling
   ```
6. Para parar e remover o agendamento:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.b3marketfeeling.dailypipeline.plist
   ```

### Linux (cron)

```bash
crontab -e
```

Adicione uma linha (ajustando os caminhos para o seu clone e virtualenv):

```cron
0 7 * * * cd /ABSOLUTE/PATH/TO/b3-market-feeling-detector && /ABSOLUTE/PATH/TO/b3-market-feeling-detector/.venv/bin/python3 main.py --stage all >> logs/pipeline_daily.log 2>&1
```

### Observações

- `logs/pipeline_daily.log` não tem rotação configurada — em execução contínua por
  muitos meses, o arquivo cresce indefinidamente; truncar ou configurar `logrotate`
  periodicamente é responsabilidade do operador.
- Um checkpoint por etapa (raw, prices, indicators) evita reprocessamento e
  duplicação em execuções diárias consecutivas — ver seção "Camadas do pipeline"
  acima.

---

## Scripts Auxiliares

Ferramentas de análise, validação e backfill que rodam à parte do pipeline principal
(`main.py`), para checagens de qualidade e análises complementares sob demanda:

| Script | Uso |
|---|---|
| `scripts/backfill_historical_news.py` | Reconstrói notícias históricas via snapshots do Internet Archive (Wayback Machine), já que feeds RSS só expõem os itens mais recentes. `python scripts/backfill_historical_news.py --from 2025-01-01 --to 2025-12-31` |
| `scripts/check_ticker_consistency.py` | Checagem heurística pós-hoc: cruza cada ticker atribuído pelo LLM com o nome oficial da empresa e busca evidência textual na notícia. `python scripts/check_ticker_consistency.py [--persist]` |
| `scripts/validate_composite_index.py` | Valida os pesos do índice composto de sentimento via PCA e via correlação com o retorno real de mercado (IBrX 100). `python scripts/validate_composite_index.py` |
| `scripts/news_price_report.py` | Relatório notícia × preço × fundamentos, incluindo teste de hipótese (Mann-Whitney U) entre sentimento e variação de preço. `python scripts/news_price_report.py [--exclude-flagged]` |
| `scripts/manual_llm_validation.py` | Validação manual semântica da saída do LLM: sorteia uma amostra fixa de notícias já enriquecidas e conduz o revisor por um julgamento cego (sentimento/relevância) seguido de confirmação dos tickers propostos. `python scripts/manual_llm_validation.py` para rotular, `--report` para ver as taxas de concordância. |

Todos são independentes do `main.py` e somente leitura sobre o banco, exceto os modos
`--persist`/interativo explicitamente indicados em cada um.

---

## Testes

```bash
pytest
```

---

## Licença

MIT — © 2025 JotaVMuniz
