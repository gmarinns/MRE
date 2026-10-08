"""Painel: os números principais do ativo em foco."""

import numpy as np
import pandas as pd
import streamlit as st

import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import ROTULO_METODO, brl, nivel, pct, prazo

ctx = servicos.atual()
dias = prazo(ctx.horizonte)
conf = nivel(ctx.confianca)

ui.cabecalho(
    f"Risco de mercado · {ctx.ticker}",
    f"Quanto uma posição de {brl(ctx.posicao)} pode perder em {dias}?",
)

precos = ctx.cotacoes.frame["close"].to_numpy(dtype=float)
queda = servicos.serie_drawdown(precos)
vol_anual = float(np.std(ctx.retornos, ddof=1) * np.sqrt(252))
var_h = ctx.var["historical"].value
es_h = ctx.es["historical"].value

ui.indicadores(
    [
        {
            "label": f"VaR {conf} · {dias}",
            **ui.perda(var_h, ctx.posicao),
            "help": "Perda que só deve ser superada nos piores "
            f"{nivel(1 - ctx.confianca)} dos casos (método histórico).",
        },
        {
            "label": f"Expected Shortfall {conf}",
            **ui.perda(es_h, ctx.posicao),
            "help": "Perda média quando o VaR é ultrapassado.",
        },
        {
            "label": "Volatilidade anualizada",
            "value": pct(vol_anual, 1),
            "delta": pct(vol_anual / np.sqrt(252)) + " ao dia",
            "delta_color": "off",
            "delta_arrow": "off",
            "help": "Desvio-padrão dos retornos diários × √252.",
        },
        {
            "label": "Drawdown máximo",
            "value": pct(float(queda.min()), 1),
            "delta": brl(float(queda.min()) * ctx.posicao),
            "delta_color": "off",
            "delta_arrow": "off",
            "help": "Maior queda do preço em relação a um topo anterior.",
        },
    ]
)

ui.grafico(
    graficos.candlestick(ctx.cotacoes.frame),
    f"Preço de {ctx.ticker}",
    "Fechamento ajustado por proventos · Yahoo Finance",
    chave="painel_preco",
)

esquerda, direita = st.columns([3, 2], gap="medium")
with esquerda:
    tabela = pd.DataFrame(
        {
            "Método": [ROTULO_METODO[m] for m in servicos.METODOS],
            "VaR": [pct(ctx.var[m].value) for m in servicos.METODOS],
            "Em reais": [brl(ctx.em_reais(ctx.var[m].value)) for m in servicos.METODOS],
        }
    )
    ui.grafico(
        graficos.barras_horizontais(
            [ROTULO_METODO[m] for m in servicos.METODOS],
            [ctx.var[m].value for m in servicos.METODOS],
        ),
        f"VaR {conf} pelos cinco métodos",
        "Mesmos dados, hipóteses diferentes. A dispersão entre eles é o risco "
        "de modelo.",
        dados=tabela,
        chave="painel_metodos",
    )
with direita, st.container(border=True):
    st.markdown("**Leitura**")
    st.markdown(
        f"Com **{conf}** de confiança, a perda em {dias} não passa de "
        f"**{brl(ctx.em_reais(var_h))}** ({pct(var_h)})."
    )
    st.markdown(
        f"Nos {nivel(1 - ctx.confianca)} piores cenários, a perda média é de "
        f"**{brl(ctx.em_reais(es_h))}** ({pct(es_h)})."
    )
    valores = [ctx.var[m].value for m in servicos.METODOS]
    st.markdown(
        f"Entre os métodos, o VaR vai de **{pct(min(valores))}** a "
        f"**{pct(max(valores))}**."
    )
    if ctx.garch["available"]:
        amanha = ctx.garch["sigma_next"] * np.sqrt(252)
        longo = ctx.garch["sigma_long_run"] * np.sqrt(252)
        frase = f"O GARCH prevê volatilidade de **{pct(amanha, 1)}** ao ano"
        if np.isfinite(longo):
            relacao = "acima" if amanha > longo else "abaixo"
            frase += f", {relacao} do nível de longo prazo ({pct(longo, 1)})"
        st.markdown(frase + " para o próximo pregão.")
    st.caption(f"{ctx.cotacoes.pregoes} pregões · {len(ctx.retornos)} retornos diários")
