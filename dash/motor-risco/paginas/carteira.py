"""Carteira: risco conjunto dos ativos e o efeito da diversificação."""

import pandas as pd
import streamlit as st

import dados
import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import brl, nivel, pct, prazo

ctx = servicos.atual()
conf, dias = nivel(ctx.confianca), prazo(ctx.horizonte)

ui.cabecalho(
    "Carteira",
    "Juntos, os ativos arriscam menos do que a soma de cada um? Quem puxa o risco?",
)

if len(ctx.ativos) < 2:
    st.info(
        "Selecione pelo menos dois ativos na barra lateral para montar a carteira.",
        icon=":material/info:",
    )
    st.stop()

matriz, faltaram = dados.matriz_de_retornos(tuple(ctx.ativos), ctx.periodo)
if faltaram:
    st.warning(
        "Sem cotações para: " + ", ".join(faltaram) + ".", icon=":material/warning:"
    )
if matriz.shape[1] < 2 or len(matriz) < 60:
    st.error("Não há pregões em comum suficientes entre os ativos escolhidos.")
    st.stop()

ativos = list(matriz.columns)
controle, _ = st.columns([2, 3])
with controle:
    editado = st.data_editor(
        pd.DataFrame(
            {"Ativo": ativos, "Peso (%)": [round(100 / len(ativos), 1)] * len(ativos)}
        ),
        hide_index=True,
        width="stretch",
        disabled=["Ativo"],
        column_config={
            "Peso (%)": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=5.0, format="%.1f"
            )
        },
        key="carteira_pesos_" + "_".join(ativos),
    )
pesos = editado["Peso (%)"].fillna(0).to_numpy(dtype=float)
if pesos.sum() <= 0:
    st.error("A soma dos pesos precisa ser maior que zero.")
    st.stop()
if abs(pesos.sum() - 100) > 0.5:
    st.caption(
        f"Os pesos somam {pesos.sum():.1f}%".replace(".", ",")
        + " e foram reescalados para 100%."
    )

r = servicos.carteira(matriz, pesos, ctx.confianca, ctx.horizonte)
comp = r["components"]

ui.indicadores(
    [
        {
            "label": f"VaR {conf} da carteira",
            **ui.perda(r["var_parametric"], ctx.posicao),
            "help": "Paramétrico, pela matriz de covariância.",
        },
        {
            "label": "VaR histórico da carteira",
            **ui.perda(r["var_historical"], ctx.posicao),
            "help": "Quantil dos retornos da carteira com os pesos atuais.",
        },
        {
            "label": "Expected Shortfall",
            **ui.perda(r["es_historical"], ctx.posicao),
        },
        {
            "label": "Ganho de diversificação",
            **ui.perda(r["diversification_benefit"], ctx.posicao),
            "help": "Soma ponderada dos VaR individuais menos o VaR da carteira.",
        },
    ]
)

esquerda, direita = st.columns([3, 2], gap="medium")
with esquerda:
    ui.grafico(
        graficos.cascata_diversificacao(
            r["undiversified_var"], r["diversification_benefit"], r["var_parametric"]
        ),
        f"Do risco somado ao risco da carteira · VaR {conf} em {dias}",
        "Se os ativos andassem sempre juntos, o risco seria a primeira barra. "
        "A correlação imperfeita retira a segunda.",
        dados=pd.DataFrame(
            {
                "Etapa": [
                    "Soma dos VaR individuais",
                    "Diversificação",
                    "VaR da carteira",
                ],
                "Valor": [
                    pct(r["undiversified_var"]),
                    pct(-r["diversification_benefit"], sinal=True),
                    pct(r["var_parametric"]),
                ],
            }
        ),
        chave="cart_cascata",
    )
with direita:
    ui.grafico(
        graficos.correlacao(r["correlation"], altura=380),
        "Correlação entre os retornos diários",
        f"{len(matriz)} pregões em comum, alinhados por data.",
        chave="cart_correlacao",
    )

ui.grafico(
    graficos.barras_horizontais(list(comp["asset"]), list(comp["component_var"])),
    "Quanto cada ativo contribui para o VaR da carteira",
    "As contribuições somam exatamente o VaR da carteira.",
    dados=pd.DataFrame(
        {
            "Ativo": comp["asset"],
            "Peso": [pct(v, 1) for v in comp["weight"]],
            "VaR isolado": [pct(v) for v in comp["individual_var"]],
            "Contribuição": [pct(v) for v in comp["component_var"]],
            "Parcela do risco": [pct(v, 1) for v in comp["share"]],
            "Em reais": [brl(v * ctx.posicao) for v in comp["component_var"]],
        }
    ),
    chave="cart_componentes",
)

ui.como_ler(
    r"""
- O VaR da carteira usa a matriz de covariância: $\sigma_p = \sqrt{w^\top
  \Sigma\, w}$. Quanto menor a correlação entre os ativos, menor $\sigma_p$.
- A **contribuição** de um ativo depende do peso *e* de como ele se move com o
  resto da carteira. Um ativo pode ter peso de 25% e responder por 40% do
  risco.
- Um ativo com contribuição negativa funcionaria como proteção: aumentar sua
  posição reduziria o risco total.
"""
)
