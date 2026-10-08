"""Expected Shortfall: o tamanho da perda quando o VaR é ultrapassado."""

import numpy as np
import pandas as pd
import streamlit as st

import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import brl, nivel, num, pct, prazo

ctx = servicos.atual()
conf, dias = nivel(ctx.confianca), prazo(ctx.horizonte)
ret = ctx.retornos
FONTES = {
    "historical": "Histórico",
    "parametric": "Paramétrico",
    "montecarlo": "Monte Carlo",
}

ui.cabecalho(
    "Expected Shortfall",
    "Quando o VaR é ultrapassado, de quanto é a perda, em média?",
)

var_h, es_h = ctx.var["historical"].value, ctx.es["historical"].value
ui.indicadores(
    [
        {"label": f"ES {conf} · {FONTES[f]}", **ui.perda(ctx.es[f].value, ctx.posicao)}
        for f in FONTES
    ]
    + [
        {
            "label": "Razão ES / VaR",
            "value": num(es_h / var_h) + "×",
            "help": "Em uma normal, a 95%, a razão é 1,25. Acima disso, a cauda "
            "é mais pesada que a da normal.",
        }
    ]
)

# O gráfico usa retornos diários; os limiares de 1 dia vêm do quantil empírico.
quantil = ctx.var["historical"].params["quantile"]
es_diario = es_h / np.sqrt(ctx.horizonte)
cauda = np.sort(ret[ret <= quantil])
ui.grafico(
    graficos.distribuicao(
        ret,
        quantil,
        [
            (quantil, f"VaR {conf} · {pct(-quantil)}"),
            (-es_diario, f"ES {conf} · {pct(es_diario)}"),
        ],
        faixa_x=(float(ret.min()) * 1.05, float(np.percentile(ret, 40))),
        altura=420,
    ),
    "A cauda de perdas, ampliada",
    f"O VaR marca onde a cauda começa; o ES é a média dos {len(cauda)} dias "
    "que ficam dentro dela (valores de 1 dia).",
    dados=pd.DataFrame({"Retornos na cauda": [pct(v) for v in cauda]}),
    chave="es_cauda",
)

esquerda, direita = st.columns([3, 2], gap="medium")
with esquerda:
    curva = servicos.curva_de_confianca(ret, ctx.horizonte)
    ui.grafico(
        graficos.linhas(
            curva["confianca"],
            [("VaR", curva["var"]), ("Expected Shortfall", curva["es"])],
            formato_y=".1%",
            formato_x=".1%",
            titulo_x="Nível de confiança",
            marcadores=True,
            altura=360,
        ),
        f"VaR e ES por nível de confiança ({dias})",
        "A distância entre as curvas cresce com a confiança: é o peso da cauda.",
        dados=pd.DataFrame(
            {
                "Confiança": [nivel(c) for c in curva["confianca"]],
                "VaR": [pct(v) for v in curva["var"]],
                "ES": [pct(v) for v in curva["es"]],
            }
        ),
        chave="es_curva",
    )
with direita, st.container(border=True):
    st.markdown(f"**ES {conf} em {dias}, por fonte**")
    st.dataframe(
        pd.DataFrame(
            {
                "Fonte": list(FONTES.values()),
                "ES": [pct(ctx.es[f].value) for f in FONTES],
                "Em reais": [brl(ctx.em_reais(ctx.es[f].value)) for f in FONTES],
                "VaR da mesma fonte": [pct(ctx.var[f].value) for f in FONTES],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.latex(r"\mathrm{ES}_c = -\,\mathbb{E}\,[\,r \mid r \le -\mathrm{VaR}_c\,]")
    st.caption(
        "Basileia III (FRTB) trocou o VaR 99% pelo ES 97,5% como medida de "
        "capital para risco de mercado."
    )

ui.como_ler(
    f"""
- O VaR responde *"qual o limite?"*; o ES responde *"e se o limite for
  rompido?"*. Aqui: nos {nivel(1 - ctx.confianca)} piores cenários, a perda
  média em {dias} é de **{pct(es_h)}**, contra um VaR de {pct(var_h)}.
- O ES é sempre maior ou igual ao VaR. Quanto maior a razão ES/VaR, mais
  pesada a cauda.
- Diferente do VaR, o ES é uma medida *coerente*: o ES de uma carteira nunca
  é maior que a soma dos ES das partes.
"""
)
