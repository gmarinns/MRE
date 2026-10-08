"""Volatilidade: realizada, EWMA e condicional GARCH(1,1)."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import num, pct, pp

ctx = servicos.atual()
ret, garch = ctx.retornos, ctx.garch
ANUAL = np.sqrt(252)

ui.cabecalho(
    "Volatilidade",
    "O quanto o preço oscila, e esse nível está mudando?",
)

amostral = float(np.std(ret, ddof=1)) * ANUAL
realizada = pd.Series(ret).rolling(21).std(ddof=1).to_numpy() * ANUAL
ewma = servicos.sigma_ewma(ret, 0.94) * ANUAL

itens = [
    {
        "label": "Volatilidade do período",
        "value": pct(amostral, 1),
        "help": "Desvio-padrão de todos os retornos diários × √252.",
    },
    {
        "label": "Últimos 21 pregões",
        "value": pct(float(realizada[-1]), 1),
        "delta": pp(float(realizada[-1]) - amostral) + " vs. período",
        "delta_color": "inverse",
    },
]
if garch["available"]:
    itens += [
        {
            "label": "GARCH · próximo pregão",
            "value": pct(garch["sigma_next"] * ANUAL, 1),
            "help": "Previsão um passo à frente, anualizada.",
        },
        {
            "label": "Persistência (α + β)",
            "value": num(garch["persistence"], 3),
            "help": "Perto de 1, um choque de volatilidade demora a se dissipar.",
        },
    ]
ui.indicadores(itens)

series = [("Realizada (21 pregões)", realizada), ("EWMA (λ = 0,94)", ewma)]
if garch["available"]:
    series.insert(1, ("Condicional GARCH(1,1)", garch["conditional_vol"] * ANUAL))
else:
    st.warning(f"GARCH não estimado: {garch['note']}.", icon=":material/warning:")
ui.grafico(
    graficos.linhas(ctx.datas, series, formato_y=".0%", altura=420),
    "Volatilidade anualizada ao longo do tempo",
    "Os modelos condicionais reagem a um choque antes da janela de 21 pregões.",
    chave="vol_series",
)

esquerda, direita = st.columns([3, 2], gap="medium")
with esquerda:
    if garch["available"]:
        previsao = servicos.previsao_garch(garch, 21) * ANUAL
        longo = garch["sigma_long_run"] * ANUAL
        ui.grafico(
            graficos.linha_com_referencia(
                np.arange(1, 22),
                previsao,
                "Volatilidade prevista",
                longo if np.isfinite(longo) else float(previsao[-1]),
                "nível de longo prazo",
                titulo_x="Dias úteis à frente",
                marcadores=True,
            ),
            "Previsão do GARCH para os próximos 21 pregões",
            "A previsão converge para o nível de longo prazo à taxa α + β.",
            dados=pd.DataFrame(
                {
                    "Dia": np.arange(1, 22),
                    "Vol. prevista": [pct(v, 1) for v in previsao],
                }
            ),
            chave="vol_previsao",
        )
        with st.container(border=True):
            st.markdown("**Parâmetros estimados do GARCH(1,1)**")
            st.latex(
                r"\sigma_t^2 = \omega + \alpha\,\epsilon_{t-1}^2"
                r" + \beta\,\sigma_{t-1}^2"
            )
            ui.indicadores(
                [
                    {
                        "label": "ω",
                        "value": num(garch["omega"], 4),
                        "help": "Estimado com retornos em pontos percentuais.",
                    },
                    {
                        "label": "α · reação ao choque",
                        "value": num(garch["alpha"], 3),
                    },
                    {"label": "β · memória", "value": num(garch["beta"], 3)},
                ]
            )
            st.caption(
                "Estimado por máxima verossimilhança (biblioteca arch) · "
                f"log-verossimilhança {num(garch['loglik'], 1)} · "
                f"AIC {num(garch['aic'], 1)}"
            )
with direita, st.container(border=True):
    st.markdown("**Estatísticas dos retornos diários**")
    jb = stats.jarque_bera(ret)
    st.dataframe(
        pd.DataFrame(
            {
                "Medida": [
                    "Observações",
                    "Média",
                    "Desvio-padrão",
                    "Assimetria",
                    "Curtose em excesso",
                    "Pior dia",
                    "Melhor dia",
                ],
                "Valor": [
                    str(len(ret)),
                    pct(float(ret.mean()), 3),
                    pct(float(ret.std(ddof=1))),
                    num(float(stats.skew(ret))),
                    num(float(stats.kurtosis(ret))),
                    pct(float(ret.min())),
                    pct(float(ret.max())),
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    ui.estado(
        jb.pvalue >= 0.05,
        "Jarque-Bera não rejeita a normalidade dos retornos.",
        "Jarque-Bera rejeita a normalidade dos retornos "
        f"(p-valor {jb.pvalue:.1e}".replace(".", ",")
        + ").",
    )
    st.caption(
        "Rejeitar a normalidade é o esperado em séries financeiras, e é o que "
        "justifica ir além do VaR paramétrico."
    )

ui.como_ler(
    """
- **Realizada**: desvio-padrão dos últimos 21 pregões. Simples, mas reage com
  atraso e "esquece" um choque de uma vez quando ele sai da janela.
- **EWMA**: média ponderada em que o peso de cada dia cai 6% em relação ao
  dia seguinte (λ = 0,94).
- **GARCH(1,1)**: estima do próprio histórico o quanto a volatilidade reage a
  um choque (α) e o quanto ela persiste (β). É a volatilidade usada na
  simulação de Monte Carlo.
- Períodos de alta volatilidade aparecem agrupados: é o *volatility
  clustering*, o fenômeno que esses modelos existem para capturar.
"""
)
