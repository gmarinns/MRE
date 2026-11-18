# FinProv / MRE

TCC: dados financeiros, risco de mercado e rastreabilidade. A fase atual usa um ETL único com PostgreSQL: fontes → raw → curated → resultados. A carteira contém 12 ações com pesos iguais; a previsão experimental tem horizonte de um pregão.

## Documentação atual

- [Modelo de dados e decisões](docs/modelo-dados.md)
- [Executar, consultar o PostgreSQL e abrir o dashboard](docs/teste-fase1.md)

## Executar

Na raiz MRE, configure DATABASE_URL para seu PostgreSQL e execute:

```powershell
uv sync --locked --dev
docker compose -f infra/docker-compose.yml up -d postgres
$env:DATABASE_URL = 'postgresql+psycopg2://finprov:finprov@localhost:5432/finprov'
uv run finprov db-init
uv run finprov collect --config config/carteira.toml
uv run finprov validate latest --recalculate
uv run finprov export latest --data-dir data/postgres
```

Não execute reset para uso cotidiano. Em uma instalação nova, db-init cria o modelo e collect popula o banco. `latest` seleciona a execução ready mais recente. O comando export apenas gera cópias legíveis dos resultados do banco.

## Interface

O dashboard mantém o visual e a navegação do motor de risco: Painel, VaR, ES, Volatilidade, Drawdown, Carteira, Stress, Backtesting e a proposta de Proveniência. Os cálculos interativos usam o pacote `motor_calculo`, com preços da última execução ready no PostgreSQL. Somente o ETL acessa as fontes externas; o dashboard não usa Yahoo diretamente nem cópias locais de cotações. Os comandos estão no guia de execução acima.

## Estado do trabalho

Implementados: PostgreSQL raw/curated/controle/resultados, ingestão Yahoo/BACEN/Investidor10 para as 12 ações, tratamento de dados, carteira, VaR e ES históricos, classificação experimental e dashboard integrado ao motor de risco. Há testes de domínio, integração com PostgreSQL e interface sem acesso HTTP ao mercado.

O pacote motor-calculo é desenvolvido separadamente para a Fase 2 e foi preservado. O dashboard utiliza esse motor sem alterar suas fórmulas. Grafo PROV-O/Neo4j, API FastAPI, Kubernetes/AkôFlow e avaliação final de proveniência continuam em fases posteriores. A arquitetura atual é Python; o antigo planejamento em Rust não é a implementação vigente.

A retirada anterior dos snapshots em disco foi seguida pelo pedido de preservar RAW no banco: o novo fluxo guarda respostas brutas no PostgreSQL, sem a antiga pasta de objetos nem cópias de código. O ETL legado e as tabelas genéricas de public foram removidos. Arquivos antigos em data não alimentam este modelo.

Os resultados são experimentais. O sistema não identifica ações sem risco nem promete prever altas. Este repositório não certifica conformidade regulatória.
