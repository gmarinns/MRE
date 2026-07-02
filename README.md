# MRE

# FinProv

> Motor de risco de mercado com **proveniência de dados fim-a-fim** — para que cada número de risco possa ser auditado, explicado e reproduzido.

**Status:** 🚧 Início de desenvolvimento (Fase 0 — Fundação). Este projeto é um Trabalho de Conclusão de Curso (Sistemas de Informação) em andamento.

---

## O problema

Gestores de risco precisam responder todos os dias: *"quanto esta carteira pode perder amanhã, no pior cenário razoável?"*. A resposta padrão da indústria é o **VaR (Value at Risk)**.

Os sistemas existentes calculam esse número, mas o tratam como uma caixa-preta. Se o VaR de uma carteira foi de R$ 9.800 ontem para R$ 12.400 hoje, o analista não sabe se a causa foi:

- um dado de entrada que mudou;
- a janela de tempo que foi alterada;
- o modelo que foi atualizado;
- ou a volatilidade do mercado que aumentou de verdade.

Descobrir isso hoje exige re-executar tudo manualmente. Pior: reproduzir um cálculo do passado, com exatamente os mesmos dados e parâmetros, é quase impossível.

## A proposta

O **FinProv** calcula o risco de uma carteira por três métodos complementares — **Histórico**, **Monte Carlo** e **GARCH** — e adiciona o que os sistemas atuais não têm: uma camada de **proveniência de dados**.

Cada resultado nasce com uma "certidão de nascimento" completa: quais dados entraram (versionados por hash), qual versão do código foi usada, quais parâmetros foram aplicados e em que ambiente rodou. Esse rastro é armazenado como um grafo aderente ao padrão **W3C PROV** e pode ser auditado, comparado entre execuções e consultado em linguagem natural.

Isso resolve três dores reais:

- **Auditoria** — reguladores podem inspecionar qualquer cálculo histórico.
- **Explicabilidade** — o analista entende *por que* o risco mudou, sem re-executar nada.
- **Reprodutibilidade** — qualquer resultado do passado pode ser reproduzido de forma determinística.

## Stack planejada

| Camada | Tecnologia |
|---|---|
| Ingestão, ETL e modelos de risco | Python 3.12 (pandas, numpy, scipy, `arch`) |
| API REST + kernel de Monte Carlo | Rust (Axum, PyO3) |
| Séries temporais e resultados | PostgreSQL + TimescaleDB |
| Grafo de proveniência | Neo4j (W3C PROV-O) |
| Cache | Redis |
| Snapshots imutáveis de dados | MinIO / S3 (versionados por SHA-256) |
| Orquestração de workflows (batch) | [AkôFlow](https://uffescience.github.io/akoflow/) |
| Orquestração de serviços | Kubernetes |
| Camada de explicação | Agente LLM (GraphRAG / text-to-Cypher) |
| Dashboard | Streamlit |

> A arquitetura é poliglota por decisão de projeto: Python para o núcleo quantitativo, Rust para performance na API e na simulação. As decisões técnicas são registradas em [`docs/adr/`](docs/adr/).

## Estrutura do repositório

```
finprov/
├── ingest/       # conectores de dados (Yahoo Finance, brapi, BACEN/SGS)
├── etl/          # transformação e versionamento de dados
├── risk/         # modelos de risco (VaR histórico, Monte Carlo, GARCH, ES)
├── provenance/   # camada FinProv — grafo de proveniência (Neo4j / PROV-O)
├── api/          # serviço REST stateless (Rust / Axum)
├── agent/        # agente LLM de explicação
├── dash/         # dashboard (Streamlit)
├── infra/        # docker-compose, manifests Kubernetes, workflows AkôFlow
├── docs/         # documentação e ADRs
└── paper/        # monografia
```

## Roadmap (visão macro)

- [ ] **Fase 0 — Fundação:** repositório, CI, infraestrutura local. *(em andamento)*
- [ ] **Fase 1 — Pipeline + VaR Histórico:** primeiro cálculo fim-a-fim.
- [ ] **Fase 2 — Motor completo:** Monte Carlo, GARCH, Expected Shortfall e backtesting.
- [ ] **Fase 3 — FinProv:** grafo de proveniência e replay determinístico.
- [ ] **Fase 4 — API em Rust:** serviço REST stateless + benchmark.
- [ ] **Fase 5 — Orquestração:** Kubernetes + AkôFlow.
- [ ] **Fase 6 — Agente LLM:** explicação de variações de risco.
- [ ] **Fase 7 — Dashboard e experimentos finais.**
- [ ] **Fase 8 — Monografia e defesa.**

## Como executar

🚧 Instruções de setup serão adicionadas assim que o ambiente local estiver estável (Fase 0). O objetivo é que `docker compose up` suba toda a infraestrutura de desenvolvimento.

## Aviso

Este é um projeto acadêmico. O sistema é *inspirado* em conceitos regulatórios (Basileia / FRTB), mas **não é certificado** para uso em produção ou para tomada de decisão financeira real.

## Licença

🚧 A definir.
