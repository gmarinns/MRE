"""Drawdown: a perda acumulada desde o último topo."""

import pandas as pd
import streamlit as st

import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import brl, pct

ctx = servicos.atual()
frame = ctx.cotacoes.frame
precos = frame["close"].to_numpy(dtype=float)
queda = servicos.serie_drawdown(precos)
episodios = servicos.episodios_drawdown(precos, frame["date"], top=5)

ui.cabecalho(
    "Drawdown",
    "Quem comprou no topo, quanto chegou a perder e quanto tempo levou para recuperar?",
)

atual = float(queda[-1])
pior = float(queda.min())
itens = [
    {
        "label": "Drawdown máximo",
        "value": pct(pior, 1),
        "delta": brl(pior * ctx.posicao),
        "delta_color": "off",
        "delta_arrow": "off",
    },
    {
        "label": "Drawdown atual",
        "value": pct(atual, 1),
        "help": "Distância entre o último preço e o maior preço do período.",
    },
]
if not episodios.empty:
    mais_longo = episodios.loc[episodios["days_total"].idxmax()]
    itens += [
        {
            "label": "Queda mais longa até o fundo",
            "value": f"{int(episodios['days_to_trough'].max())} pregões",
        },
        {
            "label": "Episódio mais longo",
            "value": f"{int(mais_longo['days_total'])} pregões",
            "help": "Do topo até a recuperação (ou até hoje, se ainda em aberto).",
        },
    ]
ui.indicadores(itens)

ui.grafico(
    graficos.underwater(frame["date"], queda),
    f"{ctx.ticker} · distância do preço ao último topo",
    "Zero significa preço em máxima histórica do período.",
    chave="dd_underwater",
)

with st.container(border=True):
    st.markdown("**Os cinco maiores episódios**")
    if episodios.empty:
        st.info("O preço não ficou abaixo de um topo anterior neste período.")
    else:
        st.dataframe(
            pd.DataFrame(
                {
                    "Topo": [f"{d:%d/%m/%Y}" for d in episodios["peak"]],
                    "Fundo": [f"{d:%d/%m/%Y}" for d in episodios["trough"]],
                    "Recuperação": [
                        f"{d:%d/%m/%Y}" if ok else "em aberto"
                        for d, ok in zip(
                            episodios["recovery"], episodios["recovered"], strict=True
                        )
                    ],
                    "Profundidade": [pct(v, 1) for v in episodios["depth"]],
                    "Em reais": [brl(v * ctx.posicao) for v in episodios["depth"]],
                    "Pregões até o fundo": episodios["days_to_trough"],
                    "Pregões no total": episodios["days_total"],
                }
            ),
            width="stretch",
            hide_index=True,
        )

ui.como_ler(
    """
- O drawdown mede a perda **acumulada**, que o VaR de 1 dia não vê: uma
  sequência de dias ruins, nenhum deles extremo, pode somar uma queda grande.
- **Profundidade** é a maior distância ao topo dentro do episódio;
  **recuperação** é o primeiro pregão em que o preço volta ao topo.
- Um episódio "em aberto" ainda não recuperou o topo até a última cotação.
"""
)
