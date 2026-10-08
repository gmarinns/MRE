# Painel do motor de risco

Interface Streamlit do FinProv. Só apresenta: todo cálculo vem do pacote
`motor-calculo`, sempre por intermédio de `servicos.py`.

## Como rodar

A partir desta pasta, com o ambiente virtual ativo:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Abre em `http://localhost:8501`.

## Páginas

| Página | Pergunta que responde |
|---|---|
| Painel | Quanto a posição pode perder? |
| Value at Risk | Qual o limite de perda, por cinco métodos? |
| Expected Shortfall | Quando o limite é rompido, de quanto é a perda? |
| Volatilidade | O quanto o preço oscila, e isso está mudando? |
| Drawdown | Qual a maior queda acumulada e quanto durou? |
| Carteira | Quanto a diversificação reduz o risco? Quem puxa o risco? |
| Stress | E se o mercado ficar muito mais turbulento? |
| Backtesting | O VaR previsto no passado se confirmou? |
| Proveniência (proposta) | Onde a proveniência entra. Ainda não implementada |

## Estrutura

```
app.py            navegação e barra lateral
paginas/          uma página por arquivo
servicos.py       chama o motor e guarda o cache
dados.py          cotações do Yahoo Finance + cópia local
ui/tema.py        paleta, layout dos gráficos e formatação pt-BR
ui/graficos.py    figuras Plotly
ui/componentes.py cabeçalho, indicadores e cartão de gráfico
```

## Dados

As cotações vêm do Yahoo Finance e ficam em cache por uma hora. Cada coleta
bem-sucedida é gravada em `.cache/`. Sem rede, o painel usa a última coleta e
avisa a data na barra lateral; sem rede e sem cópia, mostra um erro. Nenhum
dado é simulado.

Durante o pregão, a última barra é a cotação do momento: os números mudam ao
longo do dia até o fechamento.
