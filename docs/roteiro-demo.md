# Roteiro da demonstração

Sequência para apresentar o painel ao orientador, em cerca de oito minutos.

## Antes de começar

A partir de `dash/motor-risco`, com o ambiente virtual ativo:

```bash
streamlit run app.py
```

- Abrir uma vez com rede, na véspera: cada coleta fica gravada em `.cache/` e
  serve de reserva se a rede falhar na hora.
- Conferir no rodapé da barra lateral se a origem é o Yahoo Finance ou a cópia
  local.
- Durante o pregão, a última barra é a cotação do momento. Os números do painel
  ficam um pouco diferentes dos slides, que usam a coleta de 08/10/2026.
- Tema claro projeta melhor. O escuro está em Configurações, no menu do canto
  superior direito.

## Sequência

| # | Página | O que mostrar | O que dizer |
|---|---|---|---|
| 1 | Painel | Os quatro indicadores e o quadro "Leitura" | O risco em percentual e em reais, para uma posição de R$ 100.000 |
| 2 | Value at Risk | Trocar entre os cinco métodos | Cada método tem um gráfico próprio porque parte de uma hipótese diferente |
| 3 | Value at Risk · Monte Carlo | Mudar a distribuição para t de Student | A cauda mais pesada aumenta o VaR; o leque mostra as trajetórias |
| 4 | Expected Shortfall | O gráfico da cauda ampliada | O VaR marca onde a cauda começa; o ES é a média dentro dela |
| 5 | Volatilidade | GARCH contra a realizada; a previsão de 21 dias | A volatilidade muda no tempo e o modelo acompanha |
| 6 | Carteira | Alterar o peso de um ativo | A cascata e as contribuições mudam; a soma das parcelas fecha com o total |
| 7 | Backtesting | Trocar o estimador e a janela | Violações no tempo, semáforo de Basileia e os três testes |
| 8 | Proveniência (proposta) | O registro real de uma execução | O gancho já existe no motor; falta decidir como persistir |

## Se sobrar tempo

- **Stress:** mover os controles de "Monte o seu cenário".
- **Drawdown:** a tabela dos cinco maiores episódios.
- Mudar o nível de confiança para 99% e voltar ao Backtesting.

## Perguntas para levar ao orientador

1. Um registro de proveniência por execução do motor, ou um por ativo e por dia?
2. Grafo em Neo4j ou tabelas relacionais no Postgres?
3. O ETL deve passar a guardar as versões de uma cotação revisada?
4. Até onde a reprodutibilidade bit a bit precisa ser garantida?
5. A proveniência do AkôFlow e a do motor ficam no mesmo grafo?
6. Como avaliar o ganho? Proposta: um experimento em que o VaR muda por três
   causas e o sistema aponta qual foi.
