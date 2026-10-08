"""Value at Risk: os cinco estimadores, cada um com o gráfico que o explica."""

import numpy as np
import pandas as pd
import streamlit as st

import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import ROTULO_METODO, brl, inteiro, nivel, num, pct, prazo

ctx = servicos.atual()
conf, dias = nivel(ctx.confianca), prazo(ctx.horizonte)
ret = ctx.retornos

ui.cabecalho(
    "Value at Risk",
    f"Qual a perda que só é superada em {nivel(1 - ctx.confianca)} dos casos, "
    f"em {dias}?",
)
ui.indicadores(
    [
        {"label": ROTULO_METODO[m], **ui.perda(ctx.var[m].value, ctx.posicao)}
        for m in servicos.METODOS
    ]
)

metodo = st.segmented_control(
    "Método",
    servicos.METODOS,
    default="historical",
    format_func=ROTULO_METODO.get,
    key="var_metodo",
    label_visibility="collapsed",
)
metodo = metodo or "historical"
resultado = ctx.var[metodo]
if ctx.horizonte > 1 and metodo != "montecarlo":
    st.caption(
        f"Os gráficos mostram a distribuição diária; o VaR de {dias} é o valor "
        f"diário × √{ctx.horizonte}."
    )


def explicacao(como: str, formula: str, limite: str) -> None:
    with st.container(border=True):
        st.markdown("**Como funciona**")
        st.markdown(como)
        st.latex(formula)
        st.markdown("**Limitação**")
        st.markdown(limite)


grafico_col, texto_col = st.columns([3, 2], gap="medium")

if metodo == "historical":
    quantil = resultado.params["quantile"]
    with grafico_col:
        ui.grafico(
            graficos.distribuicao(
                ret, quantil, [(quantil, f"VaR {conf} · {pct(-quantil)}")]
            ),
            "Distribuição dos retornos observados",
            f"A cauda destacada reúne os {nivel(1 - ctx.confianca)} piores dias "
            f"dos {len(ret)} observados.",
            chave="var_hist",
        )
    with texto_col:
        explicacao(
            "Ordena os retornos que de fato ocorreram e lê o percentil. Não "
            "supõe nenhuma distribuição.",
            r"\mathrm{VaR}_c = -\,Q_{1-c}(r_1,\dots,r_n)",
            "Só enxerga o que está na janela: se o período não contém uma "
            "crise, o VaR não a antecipa.",
        )

elif metodo == "parametric":
    mu, sigma, z = (resultado.params[k] for k in ("mu", "sigma", "z"))
    limiar = mu + z * sigma
    with grafico_col:
        ui.grafico(
            graficos.distribuicao(
                ret,
                limiar,
                [(limiar, f"VaR {conf} · {pct(-limiar)}")],
                normal=(mu, sigma),
            ),
            "Retornos observados e a normal ajustada",
            "O VaR é o quantil da curva, não do histograma.",
            chave="var_param",
        )
        ui.grafico(
            graficos.qq_normal(ret),
            "Os retornos são normais?",
            "Pontos fora da reta nas extremidades indicam caudas mais pesadas "
            "que as da normal.",
            chave="var_qq",
        )
    with texto_col:
        explicacao(
            "Supõe retornos normais e calcula o quantil a partir da média e do "
            "desvio-padrão.",
            r"\mathrm{VaR}_c = -(\mu + z_{1-c}\,\sigma)",
            "Retornos financeiros têm caudas mais pesadas que a normal; em "
            "níveis altos de confiança o método subestima a perda.",
        )
        ui.indicadores(
            [
                {"label": "Média diária (μ)", "value": pct(mu, 3)},
                {"label": "Desvio-padrão (σ)", "value": pct(sigma)},
            ]
        )

elif metodo == "ewma":
    with texto_col:
        lam = st.slider(
            "Fator de decaimento (λ)",
            0.90,
            0.99,
            0.94,
            0.01,
            key="var_lambda",
            help="0,94 é o valor do RiskMetrics para dados diários.",
        )
        resultado = servicos.var_ewma(ret, ctx.confianca, lam, ctx.horizonte)
        meia_vida = np.log(0.5) / np.log(lam)
        ui.indicadores(
            [
                {
                    "label": f"VaR com λ = {num(lam)}",
                    **ui.perda(resultado.value, ctx.posicao),
                },
                {
                    "label": "Meia-vida do peso",
                    "value": f"{num(meia_vida, 1)} dias",
                    "help": "Dias até um retorno pesar metade do mais recente.",
                },
            ]
        )
        explicacao(
            "Como o paramétrico, mas a volatilidade dá mais peso aos dias "
            "recentes, reagindo rápido a mudanças de regime.",
            r"\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2",
            "Continua supondo normalidade, e λ é escolhido, não estimado.",
        )
    with grafico_col:
        n = min(120, len(ret))
        pesos = servicos.pesos_ewma(len(ret), lam)[:n]
        ui.grafico(
            graficos.linha_com_referencia(
                np.arange(n),
                pesos,
                "Peso EWMA",
                1 / len(ret),
                "pesos iguais (1/n)",
                formato_y=".2%",
                titulo_x="Dias atrás",
            ),
            "Quanto pesa cada dia no cálculo da volatilidade",
            "O dia mais recente está à esquerda.",
            chave="var_pesos",
        )
        sigma_serie = servicos.sigma_ewma(ret, lam) * np.sqrt(252)
        ui.grafico(
            graficos.linha_com_referencia(
                ctx.datas,
                sigma_serie,
                "Volatilidade EWMA",
                float(np.std(ret, ddof=1) * np.sqrt(252)),
                "desvio-padrão da amostra inteira",
                datas=True,
            ),
            "Volatilidade EWMA ao longo do tempo (anualizada)",
            chave="var_sigma_ewma",
        )

elif metodo == "cornish_fisher":
    p = resultado.params
    normal_q = p["mu"] + p["z"] * p["sigma"]
    corrigido_q = p["mu"] + p["z_cf"] * p["sigma"]
    with grafico_col:
        ui.grafico(
            graficos.distribuicao(
                ret,
                corrigido_q,
                [
                    (corrigido_q, f"Quantil corrigido · {pct(-corrigido_q)}"),
                    (normal_q, f"Quantil normal · {pct(-normal_q)}"),
                ],
                normal=(p["mu"], p["sigma"]),
            ),
            "O quanto a assimetria e a curtose deslocam o quantil",
            chave="var_cf",
        )
    with texto_col:
        explicacao(
            "Parte do quantil normal e o corrige pela assimetria (S) e pela "
            "curtose (K) medidas na amostra.",
            r"z_{cf} = z + \tfrac{(z^2-1)S}{6} + \tfrac{(z^3-3z)(K-3)}{24}"
            r" - \tfrac{(2z^3-5z)S^2}{36}",
            "A expansão só é confiável para desvios moderados da normal; com "
            "curtose muito alta o quantil corrigido pode se comportar mal.",
        )
        ui.indicadores(
            [
                {"label": "Assimetria (S)", "value": num(p["skew"])},
                {"label": "Curtose (K)", "value": num(p["kurtosis"])},
            ]
        )

else:  # montecarlo
    with texto_col:
        with st.container(border=True):
            st.markdown("**Parâmetros da simulação**")
            st.select_slider(
                "Simulações",
                [1_000, 5_000, 10_000, 50_000, 100_000],
                key="mc_sims",
                format_func=inteiro,
            )
            st.segmented_control(
                "Distribuição dos choques",
                ["normal", "t"],
                key="mc_dist",
                format_func={"normal": "Normal", "t": "t de Student (5 g.l.)"}.get,
            )
            st.toggle(
                "Volatilidade estocástica (GARCH)",
                key="mc_vol",
                help="Com a opção ativa, a recursão do GARCH roda dentro de "
                "cada trajetória simulada.",
            )
            st.number_input("Semente", 0, 9_999, key="mc_semente", step=1)
        explicacao(
            "Simula milhares de trajetórias de retorno com a volatilidade "
            "prevista pelo GARCH e lê o percentil da distribuição simulada.",
            r"r_t = \mu + \sigma_t\,\varepsilon_t,\quad"
            r"\sigma_t^2 = \omega + \alpha\,\epsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2",
            "O resultado depende do modelo escolhido para os choques, e tem "
            "erro amostral: muda um pouco com a semente.",
        )
    dist = st.session_state["mc_dist"] or "normal"
    n_dias = max(21, ctx.horizonte)
    percentis = servicos.leque_montecarlo(
        ret,
        ctx.confianca,
        min(st.session_state["mc_sims"], 20_000),
        n_dias,
        dist,
        st.session_state["mc_vol"],
        st.session_state["mc_semente"],
    )
    erro = resultado.params["mc_std_error"]
    with grafico_col:
        ui.grafico(
            graficos.leque(percentis, ctx.horizonte),
            f"Trajetórias simuladas para os próximos {n_dias} dias úteis",
            "As faixas mostram onde ficam 50% e 90% das trajetórias.",
            dados=percentis,
            chave="var_leque",
        )
        ui.grafico(
            graficos.distribuicao(
                ctx.simulacoes,
                -resultado.value,
                [(-resultado.value, f"VaR {conf} · {pct(resultado.value)}")],
                rotulo_corpo="Cenários simulados",
                titulo_x=f"Retorno simulado em {dias}",
            ),
            f"Distribuição dos {inteiro(len(ctx.simulacoes))} cenários em {dias}",
            f"Erro padrão do VaR: ± {pct(erro, 3)}.",
            chave="var_mc_dist",
        )
    if ctx.horizonte == 1:
        tabela_conv = servicos.convergencia(
            ret, ctx.confianca, dist, st.session_state["mc_semente"]
        )
        ui.grafico(
            graficos.convergencia(tabela_conv),
            "Quantas simulações são necessárias?",
            "VaR estimado e intervalo de 95% conforme cresce o número de simulações.",
            dados=tabela_conv,
            chave="var_conv",
        )

st.subheader("Comparação entre os métodos", divider="gray")
comparacao = pd.DataFrame(
    {
        "Método": [ROTULO_METODO[m] for m in servicos.METODOS],
        f"VaR {conf}": [pct(ctx.var[m].value) for m in servicos.METODOS],
        "Em reais": [brl(ctx.em_reais(ctx.var[m].value)) for m in servicos.METODOS],
        "Hipótese central": [
            "O futuro se parece com a janela observada",
            "Retornos normais, volatilidade constante",
            "Retornos normais, volatilidade recente pesa mais",
            "Normal corrigida por assimetria e curtose",
            "Choques simulados com volatilidade GARCH",
        ],
    }
)
ui.grafico(
    graficos.barras_horizontais(
        [ROTULO_METODO[m] for m in servicos.METODOS],
        [ctx.var[m].value for m in servicos.METODOS],
        destaque=ROTULO_METODO[metodo],
    ),
    f"VaR {conf} em {dias}",
    dados=comparacao,
    chave="var_comparacao",
)

ui.como_ler(
    f"""
- Um VaR de **{pct(ctx.var["historical"].value)}** a {conf} significa que, em
  {dias}, a perda só deve passar desse valor em {nivel(1 - ctx.confianca)} dos
  casos. O VaR **não** diz quanto se perde quando o limite é ultrapassado; isso
  é o Expected Shortfall.
- Métodos diferentes dão números diferentes porque partem de hipóteses
  diferentes. A página **Backtesting** mostra qual deles acertou mais no
  passado.
- Para horizontes acima de 1 dia, os métodos de fórmula fechada multiplicam o
  VaR diário por √h (regra da raiz do tempo); o Monte Carlo simula o horizonte
  inteiro.
"""
)
