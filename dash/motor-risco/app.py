"""
FinProv — Tela de Proposta do Motor de Risco
============================================
Demonstração viva (Streamlit) da proposta de engenharia do Motor de Risco
Python-first: VaR por cinco estimadores, Expected Shortfall, GARCH(1,1),
Monte Carlo vetorizado, carteira multi-ativo, backtesting formal, stress
testing, benchmark de performance e proveniência W3C PROV real.

    streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from finprov_demo import __version__, backtesting as bt, charts, provenance as prov, risk
from finprov_demo.config import METHOD_LABELS, PALETTE, badge, inject_css
from finprov_demo.data import YFINANCE_AVAILABLE, load_prices, load_returns_matrix, log_returns

try:
    from streamlit_agraph import Config, Edge, Node, agraph

    AGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    AGRAPH_AVAILABLE = False

STRETCH = "stretch"  # substitui use_container_width=True (depreciado no Streamlit >= 1.49)

st.set_page_config(
    page_title="FinProv — Motor de Risco",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"about": f"FinProv — proposta do Motor de Risco · v{__version__}"},
)
inject_css()

TICKER_UNIVERSE = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "WEGE3.SA",
    "B3SA3.SA", "ABEV3.SA", "^BVSP", "BRL=X", "^GSPC",
]

# ── Estado inicial (a proposta pede persistência via session_state) ────────
DEFAULTS = {
    "tickers": ["PETR4.SA", "VALE3.SA", "^BVSP"],
    "focus": "PETR4.SA",
    "period": "2y",
    "confidence": 0.95,
    "horizon": 1,
    "n_sims": 10_000,
    "mc_dist": "normal",
    "vol_dynamics": True,
    "seed": 42,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 FinProv")
    st.caption(f"Proposta do Motor de Risco · v{__version__}")
    st.progress(0.28, text="F0 concluída · F1–F3 prototipadas nesta tela")
    st.divider()

    st.markdown("**Carteira**")
    tickers = st.multiselect("Ativos", TICKER_UNIVERSE, key="tickers")
    if not tickers:
        st.warning("Selecione ao menos um ativo.", icon="⚠️")
        st.stop()

    if st.session_state["focus"] not in tickers:
        st.session_state["focus"] = tickers[0]
    focus = st.selectbox("Ativo em foco", tickers, key="focus")
    period = st.select_slider("Período histórico", ["6mo", "1y", "2y", "5y"], key="period")

    st.divider()
    st.markdown("**Parâmetros de risco**")
    confidence = st.select_slider("Nível de confiança", [0.90, 0.95, 0.975, 0.99],
                                  key="confidence", format_func=lambda c: f"{c:.1%}")
    horizon = st.select_slider("Horizonte (dias úteis)", [1, 5, 10, 21], key="horizon")

    with st.expander("Monte Carlo", expanded=False):
        n_sims = st.select_slider("Simulações", [1_000, 5_000, 10_000, 50_000, 100_000], key="n_sims")
        mc_dist = st.radio("Distribuição das inovações", ["normal", "t"], key="mc_dist",
                           horizontal=True, format_func=lambda d: "Normal" if d == "normal" else "t de Student")
        vol_dynamics = st.toggle("Volatilidade estocástica (recursão GARCH na simulação)",
                                 key="vol_dynamics")
        seed = st.number_input("Semente (reprodutibilidade)", 0, 10_000, key="seed", step=1)

    st.divider()
    st.markdown("**Ambiente**")
    dep_rows = [
        ("yfinance", YFINANCE_AVAILABLE), ("arch (GARCH)", risk.ARCH_AVAILABLE),
        ("streamlit-agraph", AGRAPH_AVAILABLE),
    ]
    for name, ok in dep_rows:
        st.caption(("✅ " if ok else "⚠️ ") + name + ("" if ok else " — usando fallback"))
    if st.button("🧹 Limpar cache de dados", width=STRETCH):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  CARGA + CÁLCULO (uma vez, compartilhado por todas as abas)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def compute_all(ticker: str, period: str, confidence: float, horizon: int,
                n_sims: int, dist: str, vol_dynamics: bool, seed: int):
    series = load_prices(ticker, period)
    returns = log_returns(series.frame)
    t0 = time.perf_counter()
    results = {
        "historical": risk.historical_var(returns, confidence, horizon),
        "parametric": risk.parametric_var(returns, confidence, horizon),
        "ewma": risk.ewma_var(returns, confidence, horizon_days=horizon),
        "cornish_fisher": risk.cornish_fisher_var(returns, confidence, horizon),
    }
    mc, sims = risk.montecarlo_var(returns, confidence, n_sims=n_sims, n_days=horizon,
                                   dist=dist, seed=int(seed), vol_dynamics=vol_dynamics)
    results["montecarlo"] = mc
    es = {
        "historical": risk.expected_shortfall(returns, confidence, "historical"),
        "parametric": risk.expected_shortfall(returns, confidence, "parametric"),
        "montecarlo": risk.expected_shortfall(returns, confidence, "montecarlo", sims=sims),
    }
    elapsed = (time.perf_counter() - t0) * 1000
    return series, returns, results, es, sims, elapsed


with st.spinner(f"Carregando {focus} e calculando o motor de risco…"):
    series, returns, var_results, es_results, sims, elapsed_ms = compute_all(
        focus, period, confidence, horizon, n_sims,
        st.session_state["mc_dist"], st.session_state["vol_dynamics"], st.session_state["seed"],
    )
dates = pd.to_datetime(series.frame["date"]).iloc[1:].reset_index(drop=True)
garch = risk.fit_garch(returns)

# ══════════════════════════════════════════════════════════════════════════
#  CABEÇALHO
# ══════════════════════════════════════════════════════════════════════════
st.title("📊 FinProv — Motor de Risco de Mercado")
source_badge = (
    badge("dados reais · Yahoo Finance", "ok") if series.is_real
    else badge("dados sintéticos (GBM determinístico)", "warn")
)
st.markdown(
    source_badge
    + badge(f"{series.n_obs} pregões", "info")
    + badge(f"{focus} · {confidence:.1%} · {horizon}d", "info")
    + badge(f"motor: {elapsed_ms:.0f} ms", "info")
    + (badge(f"GARCH(1,1) ajustado · persistência {garch['persistence']:.3f}", "ok")
       if garch["available"] else badge("GARCH indisponível — fallback 21d", "warn")),
    unsafe_allow_html=True,
)
st.caption(
    "Proposta de engenharia Python-first · VaR (5 estimadores) · Expected Shortfall · "
    "GARCH(1,1) · Monte Carlo vetorizado · Backtesting formal · Proveniência W3C PROV"
)

tabs = st.tabs([
    "🏠 Visão Geral", "📈 Motor de Risco", "🌊 Volatilidade", "💼 Carteira",
    "🧪 Backtesting", "🔥 Stress", "🔗 Proveniência", "⚡ Performance",
    "🏗️ Arquitetura", "🗺️ Roadmap",
])


# ══════════════════════════════════════════════════════════════════════════
#  1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("O problema")
        st.markdown(
            """
Gestores de risco respondem diariamente: **"quanto esta carteira pode perder amanhã,
no pior cenário razoável?"** A resposta padrão é o **VaR (Value at Risk)** e, sob
FRTB/Basileia III, o **Expected Shortfall (ES)**.

Os sistemas atuais tratam esse cálculo como **caixa-preta**: não explicam variações,
não são auditáveis retroativamente e não são reprodutíveis bit-a-bit.

O **FinProv** trata **proveniência como cidadã de primeira classe** — cada número de
risco nasce com uma certidão de nascimento (insumos, hash, parâmetros, código e
ambiente), navegável como grafo W3C PROV no Neo4j.
            """
        )
        st.subheader("O que esta tela demonstra")
        st.markdown(
            """
- **F1–F2** — cinco estimadores de VaR e três fontes de ES rodando sobre dados reais.
- **F2.1** — GARCH(1,1) estimado por máxima verossimilhança (`arch`).
- **F2.2** — Monte Carlo vetorizado com volatilidade estocástica e cauda t de Student.
- **F3.1–F3.3** — Kupiec, Christoffersen, cobertura condicional, semáforo de Basileia
  e VaR de carteira por matriz de covariância.
- **F4** — benchmark ao vivo: laço Python vs. NumPy vetorizado.
- **F5** — grafo de proveniência **gerado da execução real**, exportável em Cypher e PROV-JSON.
            """
        )
    with right:
        st.info(
            "**Decisão de linguagem**\n\nMotor de risco 100% Python (NumPy vetorizado); "
            "API em **FastAPI**. Rust/Go ficam num experimento de benchmark opcional, "
            "fora do caminho crítico.", icon="🐍",
        )
        st.success("**Dados**\n\nPostgreSQL (séries) · Neo4j (proveniência) · Redis (cache)", icon="🗄️")
        st.warning(
            "**Escopo**\n\nBatch diário (não intraday), carteira pequena, um usuário — "
            "é isso que dispensa uma segunda linguagem.", icon="🎯",
        )

    st.divider()
    st.subheader("Objetivos específicos")
    st.dataframe(
        pd.DataFrame({
            "Código": [f"OE{i}" for i in range(1, 8)],
            "Objetivo": [
                "Pipeline de ingestão/ETL com versionamento imutável de datasets",
                "Implementar e validar (backtesting) três estimadores de VaR + ES",
                "Modelar grafo FinProv em Neo4j, aderente ao PROV-O",
                "Integrar proveniência do AkôFlow ao grafo de domínio",
                "Construir serviço REST stateless (FastAPI) com cache Redis",
                "Agente LLM que explica variações a partir do grafo de proveniência",
                "Demonstrar reprodutibilidade: replay determinístico de cálculo histórico",
            ],
            "Status": ["🟡 Em andamento", "🟢 Prototipado nesta tela", "🟡 Prototipado (mock Cypher)",
                       "⚪ Planejado", "🟡 Em andamento", "⚪ Planejado", "🟢 Prototipado nesta tela"],
        }),
        width=STRETCH, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════
#  2 — MOTOR DE RISCO
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader(f"{focus} · VaR e ES a {confidence:.1%} · horizonte {horizon} dia(s)")

    base = var_results["historical"].value
    cols = st.columns(5)
    for col, key in zip(cols, ["historical", "parametric", "ewma", "cornish_fisher", "montecarlo"]):
        r = var_results[key]
        delta = None if key == "historical" else f"{r.value - base:+.2%} vs. histórico"
        col.metric(
            METHOD_LABELS[key], f"{r.value:.2%}", delta=delta,
            delta_color="inverse" if delta else "normal",  # VaR maior = pior = vermelho
            help=f"run_id {r.run_id[:8]} · {', '.join(f'{k}={v}' for k, v in list(r.params.items())[:3])}",
        )

    es_cols = st.columns(4)
    for col, key in zip(es_cols, ["historical", "parametric", "montecarlo"]):
        col.metric(f"ES · {key}", f"{es_results[key].value:.2%}",
                   help="Perda média condicional a exceder o VaR (Basileia III/FRTB).")
    ratio = es_results["historical"].value / base if base else float("nan")
    es_cols[3].metric("ES / VaR (histórico)", f"{ratio:.2f}×",
                      help="Quanto a cauda é mais pesada que o próprio VaR. Normal ≈ 1,15–1,30.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.plotly_chart(charts.candlestick(series.frame, focus), width=STRETCH)
        st.plotly_chart(charts.method_comparison_bar(
            list(var_results.values()) + [es_results["historical"]]), width=STRETCH)
    with c2:
        st.plotly_chart(charts.returns_histogram(
            returns, list(var_results.values()) + [es_results["historical"]]), width=STRETCH)
        st.plotly_chart(charts.qq_plot(returns), width=STRETCH)

    with st.expander("📐 Sensibilidade ao nível de confiança"):
        curve = pd.DataFrame([
            {"confianca": c,
             "var": risk.historical_var(returns, c, horizon).value,
             "es": risk.expected_shortfall(returns, c, "historical").value}
            for c in (0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
        ])
        st.plotly_chart(charts.confidence_curve(curve), width=STRETCH)

    with st.expander("🔬 Metadados de proveniência desta execução", expanded=False):
        meta = pd.DataFrame([
            {"método": r.method, "valor": f"{r.value:.6f}", "confiança": r.confidence_level,
             "horizonte": r.horizon_days, "input_hash": r.input_hash,
             "run_id": r.run_id, "params": str(r.params)}
            for r in list(var_results.values()) + list(es_results.values())
        ])
        st.dataframe(meta, width=STRETCH, hide_index=True)

    export = pd.DataFrame([
        {"ticker": focus, "fonte": series.source, "periodo": period, **r.to_dict()}
        for r in list(var_results.values()) + list(es_results.values())
    ])
    st.download_button(
        "⬇️ Baixar resultados + proveniência (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name=f"finprov_{focus}_{datetime.now(timezone.utc):%Y%m%d}.csv",
        mime="text/csv",
    )

    if len(tickers) > 1:
        st.divider()
        st.subheader("Correlação entre os ativos selecionados")
        matrix, sources = load_returns_matrix(tuple(tickers), period)
        synthetic = [t for t, s in sources.items() if s != "yahoo"]
        if synthetic:
            st.caption(f"⚠️ Séries sintéticas nesta matriz: {', '.join(synthetic)}")
        st.plotly_chart(charts.correlation_heatmap(matrix), width=STRETCH)


# ══════════════════════════════════════════════════════════════════════════
#  3 — VOLATILIDADE
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("F1.3 / F2.1 — volatilidade rolante e modelos condicionais")

    if garch["available"]:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("ω (omega)", f"{garch['omega']:.5f}")
        g2.metric("α (choque)", f"{garch['alpha']:.4f}")
        g3.metric("β (persistência)", f"{garch['beta']:.4f}")
        g4.metric("α+β", f"{garch['persistence']:.4f}",
                  help="Próximo de 1 → choques de volatilidade demoram a dissipar.")
        st.caption(
            f"Log-verossimilhança {garch['loglik']:.1f} · AIC {garch['aic']:.1f} · "
            f"σ previsto para o próximo dia: **{garch['sigma_next']:.2%}** ao dia "
            f"({garch['sigma_next'] * np.sqrt(252):.1%} anualizado)"
        )
    else:
        st.warning(garch["note"], icon="⚠️")

    realized = risk.rolling_volatility(returns, 21)
    ewma_series = np.array([
        risk.ewma_sigma(returns[max(0, i - 120):i + 1]) if i >= 21 else np.nan
        for i in range(len(returns))
    ]) * np.sqrt(252)
    garch_series = (garch["conditional_vol"] * np.sqrt(252)) if garch["available"] else None
    st.plotly_chart(
        charts.volatility_chart(dates, realized, garch_series, ewma_series), width=STRETCH
    )

    v1, v2 = st.columns(2, gap="medium")
    with v1:
        st.plotly_chart(
            charts.drawdown_chart(
                series.frame["date"], series.frame["close"].to_numpy(),
                risk.drawdown_series(series.frame["close"].to_numpy()),
            ),
            width=STRETCH,
        )
    with v2:
        st.markdown("**Estatísticas descritivas dos log-retornos**")
        from scipy import stats as _st

        desc = pd.DataFrame({
            "métrica": ["observações", "média diária", "vol. diária", "vol. anualizada",
                        "assimetria", "curtose (excesso)", "mínimo", "máximo",
                        "Jarque-Bera (p-valor)"],
            "valor": [
                f"{len(returns)}", f"{returns.mean():.4%}", f"{returns.std(ddof=1):.4%}",
                f"{returns.std(ddof=1) * np.sqrt(252):.2%}", f"{_st.skew(returns):.3f}",
                f"{_st.kurtosis(returns):.3f}", f"{returns.min():.2%}", f"{returns.max():.2%}",
                f"{_st.jarque_bera(returns).pvalue:.2e}",
            ],
        })
        st.dataframe(desc, width=STRETCH, hide_index=True)
        st.caption(
            "Curtose em excesso > 0 e Jarque-Bera rejeitando normalidade justificam, "
            "na monografia, por que o VaR paramétrico puro subestima a cauda — e por que "
            "Cornish-Fisher e Monte Carlo com t de Student entram no comparativo."
        )


# ══════════════════════════════════════════════════════════════════════════
#  4 — CARTEIRA
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("F3.3 — VaR de carteira multi-ativo (matriz de covariância)")
    if len(tickers) < 2:
        st.info("Selecione pelo menos dois ativos na barra lateral para montar a carteira.", icon="💡")
    else:
        matrix, sources = load_returns_matrix(tuple(tickers), period)
        st.caption(f"{len(matrix)} pregões com cotação para **todos** os ativos (interseção por data).")

        weight_input = pd.DataFrame({"ativo": list(matrix.columns),
                                     "peso (%)": [round(100 / len(matrix.columns), 2)] * len(matrix.columns)})
        edited = st.data_editor(
            weight_input, width=STRETCH, hide_index=True, key="weights_editor",
            column_config={"peso (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0)},
            disabled=["ativo"],
        )
        weights = edited["peso (%)"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            st.error("A soma dos pesos precisa ser positiva.")
        else:
            port = risk.portfolio_var(matrix, weights, confidence, horizon)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("VaR paramétrico (covariância)", f"{port['var_parametric']:.2%}")
            p2.metric("VaR histórico da carteira", f"{port['var_historical']:.2%}")
            p3.metric("ES histórico da carteira", f"{port['es_historical']:.2%}")
            p4.metric("Benefício de diversificação", f"{port['diversification_benefit']:.2%}",
                      help="VaR não diversificado (soma ponderada dos VaR individuais) − VaR da carteira.",
                      delta_color="off")

            cc1, cc2 = st.columns([1.15, 1], gap="medium")
            with cc1:
                comp = port["component_var"].copy()
                comp.columns = ["ativo", "peso", "VaR individual", "VaR componente", "contribuição"]
                st.dataframe(
                    comp, width=STRETCH, hide_index=True,
                    column_config={
                        "peso": st.column_config.NumberColumn(format="percent"),
                        "VaR individual": st.column_config.NumberColumn(format="percent"),
                        "VaR componente": st.column_config.NumberColumn(format="percent"),
                        "contribuição": st.column_config.ProgressColumn(
                            format="percent", min_value=0.0, max_value=1.0),
                    },
                )
                st.caption(
                    "O **VaR componente** decompõe o risco: a soma das componentes reproduz o "
                    "VaR da carteira, o que responde *qual ativo está puxando o risco*."
                )
            with cc2:
                st.plotly_chart(charts.correlation_heatmap(matrix, height=360), width=STRETCH)

            st.plotly_chart(
                charts.returns_histogram(
                    port["portfolio_returns"],
                    [port["result"],
                     risk.expected_shortfall(port["portfolio_returns"], confidence, "historical")],
                    height=380, show_normal=True,
                ),
                width=STRETCH,
            )


# ══════════════════════════════════════════════════════════════════════════
#  5 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("F3.1 / F3.2 — backtesting formal")
    b1, b2 = st.columns([1, 1])
    window = b1.slider("Janela de calibração (dias)", 21, 252, 63, step=21, key="bt_window")
    bt_method = b2.selectbox("Estimador testado", ["historical", "parametric", "ewma"],
                             format_func=lambda m: METHOD_LABELS[m], key="bt_method")

    var_series = risk.rolling_var_series(returns, window, confidence, bt_method)
    valid = ~np.isnan(var_series)

    if valid.sum() < 30:
        st.warning("Histórico insuficiente para essa janela — reduza a janela ou amplie o período.",
                   icon="⚠️")
    else:
        rets_v = returns[valid]
        var_v = var_series[valid]
        dates_v = dates[valid].reset_index(drop=True)
        report = bt.full_backtest(rets_v, var_v, confidence)
        k, ind, cc, basel = report["kupiec"], report["independence"], report["conditional"], report["basel"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Violações observadas", f"{k['violations']} / {k['n_obs']}")
        m2.metric("Violações esperadas", f"{k['expected_violations']:.1f}")
        m3.metric("Taxa de violação", f"{k['violation_rate']:.2%}",
                  delta=f"{k['violation_rate'] - k['expected_rate']:+.2%} vs. esperado",
                  delta_color="inverse")
        m4.metric("Zona de Basileia", basel["zone"], help=basel["note"], delta_color="off")

        st.divider()
        t1, t2, t3 = st.columns(3, gap="medium")
        for col, res, desc in (
            (t1, k, "H₀: a taxa de violação é igual a 1 − c (cobertura incondicional)."),
            (t2, ind, "H₀: violações são independentes no tempo (sem clustering)."),
            (t3, cc, "H₀: cobertura correta **e** independência, conjuntamente."),
        ):
            with col:
                st.markdown(f"**{res['test']}**")
                st.caption(desc)
                st.write(f"LR = `{res['lr_stat']:.4f}` · gl = {res['df']} · p-valor = `{res['p_value']:.4f}`")
                if res["reject_h0"]:
                    st.error("H₀ rejeitada (p < 0,05)", icon="🚨")
                else:
                    st.success("H₀ não rejeitada", icon="✅")

        st.plotly_chart(
            charts.var_backtest_chart(dates_v, rets_v, var_v, report["mask"]), width=STRETCH
        )
        st.markdown("**Linha do tempo das violações** — agrupamento visível sugere dependência temporal.")
        st.plotly_chart(charts.violation_clustering(dates_v, report["mask"]), width=STRETCH)

        with st.expander("Matriz de transição de Christoffersen e diagnóstico adicional"):
            tr = ind["transitions"]
            st.dataframe(
                pd.DataFrame(
                    [[tr["00"], tr["01"]], [tr["10"], tr["11"]]],
                    index=["sem violação em t−1", "violação em t−1"],
                    columns=["sem violação em t", "violação em t"],
                ),
                width=STRETCH,
            )
            st.write(
                f"π₀₁ = `{ind['pi01']:.4f}` · π₁₁ = `{ind['pi11']:.4f}` — sob independência "
                "as duas probabilidades deveriam ser iguais."
            )
            st.write(f"VaR médio no período: `{report['avg_var']:.2%}` · "
                     f"perda média nos dias de violação: `{report['loss_when_violated']:.2%}`")
            st.caption(
                "Nota metodológica: cada violação é comparada com o VaR **previsto para aquele dia** "
                "(janela rolante, sem look-ahead), e não com a média da série — é o que garante que "
                "a contagem dos testes e os marcadores do gráfico sejam o mesmo evento."
            )

        st.divider()
        st.markdown("**Comparação entre estimadores na mesma janela**")
        rows = []
        for m in ("historical", "parametric", "ewma"):
            s = risk.rolling_var_series(returns, window, confidence, m)
            v = ~np.isnan(s)
            rep = bt.full_backtest(returns[v], s[v], confidence)
            rows.append({
                "estimador": METHOD_LABELS[m],
                "VaR médio": rep["avg_var"],
                "violações": rep["kupiec"]["violations"],
                "taxa": rep["kupiec"]["violation_rate"],
                "p (Kupiec)": rep["kupiec"]["p_value"],
                "p (independência)": rep["independence"]["p_value"],
                "p (condicional)": rep["conditional"]["p_value"],
                "Basileia": rep["basel"]["zone"],
            })
        st.dataframe(
            pd.DataFrame(rows), width=STRETCH, hide_index=True,
            column_config={
                "VaR médio": st.column_config.NumberColumn(format="percent"),
                "taxa": st.column_config.NumberColumn(format="percent"),
                "p (Kupiec)": st.column_config.NumberColumn(format="%.4f"),
                "p (independência)": st.column_config.NumberColumn(format="%.4f"),
                "p (condicional)": st.column_config.NumberColumn(format="%.4f"),
            },
        )


# ══════════════════════════════════════════════════════════════════════════
#  6 — STRESS
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Stress testing e análise de cenários")
    s1, s2 = st.columns(2)
    vol_mult = s1.slider("Multiplicador de volatilidade", 1.0, 5.0, 2.0, 0.25, key="stress_vol")
    mean_shift = s2.slider("Deslocamento da média diária (p.p.)", -2.0, 1.0, -0.5, 0.1,
                           key="stress_mean") / 100

    stressed = risk.stressed_var(returns, confidence, vol_mult, mean_shift, horizon)
    base_var = var_results["historical"].value
    k1, k2, k3 = st.columns(3)
    k1.metric("VaR base", f"{base_var:.2%}")
    k2.metric("VaR sob estresse", f"{stressed.value:.2%}",
              delta=f"{stressed.value - base_var:+.2%}", delta_color="inverse")
    k3.metric("Fator de amplificação", f"{stressed.value / base_var:.2f}×" if base_var else "—",
              delta_color="off")

    st.markdown("**Cenários pré-definidos** — choques aplicados sobre a série observada")
    scenarios = [
        ("Base (observado)", 1.0, 0.0),
        ("Vol ×1,5 (tensão moderada)", 1.5, -0.002),
        ("Vol ×2 (crise cambial 2015)", 2.0, -0.004),
        ("Vol ×3 (choque COVID mar/2020)", 3.0, -0.010),
        ("Vol ×4 (cauda extrema)", 4.0, -0.015),
    ]
    st.dataframe(
        pd.DataFrame([
            {
                "cenário": name,
                "VaR": risk.stressed_var(returns, confidence, vm, ms, horizon).value,
                "ES": risk.expected_shortfall(
                    (returns - returns.mean()) * vm + returns.mean() + ms, confidence, "historical"
                ).value,
                "vol. anual implícita": returns.std(ddof=1) * vm * np.sqrt(252),
            }
            for name, vm, ms in scenarios
        ]),
        width=STRETCH, hide_index=True,
        column_config={
            "VaR": st.column_config.NumberColumn(format="percent"),
            "ES": st.column_config.NumberColumn(format="percent"),
            "vol. anual implícita": st.column_config.NumberColumn(format="percent"),
        },
    )

    st.divider()
    st.markdown("**Piores janelas observadas no período** (stress histórico, sem simulação)")
    st.dataframe(
        risk.worst_windows(returns, dates), width=STRETCH, hide_index=True,
        column_config={"pior retorno acumulado": st.column_config.NumberColumn(format="percent")},
    )
    st.caption(
        "Essas janelas são a base do *stressed VaR* exigido por Basileia II.5: recalibrar o "
        "modelo sobre o pior período histórico em vez da janela recente."
    )


# ══════════════════════════════════════════════════════════════════════════
#  7 — PROVENIÊNCIA
# ══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("F5 — grafo de proveniência (W3C PROV-O) desta execução")
    st.caption(
        "Este grafo **não é ilustrativo**: os nós carregam o hash real dos insumos, os "
        "parâmetros efetivamente usados e os valores calculados na aba Motor de Risco."
    )

    all_results = list(var_results.values()) + [es_results["historical"]]
    graph = prov.build_graph(
        ticker=focus, source=series.source, period=period, n_prices=series.n_obs,
        returns_hash=var_results["historical"].input_hash, results=all_results,
    )

    gc1, gc2, gc3 = st.columns([1, 1, 1])
    g_height = gc1.slider("Altura do grafo", 320, 900, 520, 20, key="graph_h")
    g_width = gc2.slider("Largura do grafo", 400, 1400, 900, 50, key="graph_w",
                         help="O componente agraph exige largura em pixels; ajuste para a sua tela.")
    physics = gc3.toggle("Física (nós se reorganizam)", value=True, key="graph_physics")

    if AGRAPH_AVAILABLE:
        nodes = [
            Node(id=n["id"], label=n["label"], size=30 if n["kind"] == "Result" else 24,
                 color=prov.PROV_COLORS[n["kind"]], shape="dot",
                 title=" · ".join(f"{k}={v}" for k, v in n["props"].items())[:400])
            for n in graph.nodes
        ]
        edges = [Edge(source=e["source"], target=e["target"], label=e["relation"])
                 for e in graph.edges]
        agraph(nodes=nodes, edges=edges,
               config=Config(width=g_width, height=g_height, directed=True,
                             physics=physics, hierarchical=False,
                             nodeHighlightBehavior=True, highlightColor=PALETTE["primary"],
                             collapsible=False))
        legend = " ".join(
            f'<span class="fp-badge" style="background:{c}22;color:{c};border-color:{c}66">{k}</span>'
            for k, c in prov.PROV_COLORS.items()
        )
        st.markdown(legend, unsafe_allow_html=True)
    else:
        st.warning("Instale `streamlit-agraph` para o grafo interativo. Segue a versão tabular:",
                   icon="⚠️")
        st.dataframe(pd.DataFrame(graph.edges), width=STRETCH, hide_index=True)

    st.markdown(
        "**Leitura:** a `Entity` de preços é consumida (`used`) pela `Activity` de ETL, que gera "
        "(`wasGeneratedBy`) a `Entity` de retornos identificada pelo hash SHA-256. Cada método de "
        "risco é uma `Activity` que consome essa mesma entidade e gera um resultado. O `Agent` "
        "carrega o ambiente (versões de Python/NumPy/SciPy/arch) — é o que permite responder "
        "*\"por que o VaR mudou?\"* comparando dois `run_id` em vez de reprocessar tudo."
    )

    st.divider()
    e1, e2 = st.columns(2, gap="medium")
    with e1:
        st.markdown("**Exportar para o Neo4j (Cypher)**")
        cypher = prov.to_cypher(graph)
        st.code(cypher[:1400] + ("\n…" if len(cypher) > 1400 else ""), language="cypher")
        st.download_button("⬇️ Baixar .cypher", cypher.encode("utf-8"),
                           file_name=f"finprov_prov_{focus}.cypher", mime="text/plain")
    with e2:
        st.markdown("**Exportar em PROV-JSON (W3C)**")
        pj = prov.to_prov_json(graph)
        st.code(pj[:1400] + ("\n…" if len(pj) > 1400 else ""), language="json")
        st.download_button("⬇️ Baixar .json", pj.encode("utf-8"),
                           file_name=f"finprov_prov_{focus}.json", mime="application/json")

    st.divider()
    st.subheader("OE7 — replay determinístico")
    st.caption(
        "Recalcula tudo do zero e compara os `run_id`. Iguais ⇒ mesmo insumo, mesmos parâmetros, "
        "mesmo resultado — a propriedade que a monografia precisa demonstrar."
    )
    if st.button("▶️ Executar replay e conferir hashes", width=STRETCH):
        replay = {
            "historical": risk.historical_var(returns, confidence, horizon),
            "parametric": risk.parametric_var(returns, confidence, horizon),
            "ewma": risk.ewma_var(returns, confidence, horizon_days=horizon),
            "cornish_fisher": risk.cornish_fisher_var(returns, confidence, horizon),
            "montecarlo": risk.montecarlo_var(
                returns, confidence, n_sims=n_sims, n_days=horizon,
                dist=st.session_state["mc_dist"], seed=int(st.session_state["seed"]),
                vol_dynamics=st.session_state["vol_dynamics"])[0],
        }
        check = pd.DataFrame([
            {"método": m, "run_id original": var_results[m].run_id[:18],
             "run_id do replay": replay[m].run_id[:18],
             "Δ valor": abs(var_results[m].value - replay[m].value),
             "idêntico": var_results[m].run_id == replay[m].run_id
                         and np.isclose(var_results[m].value, replay[m].value)}
            for m in replay
        ])
        st.dataframe(check, width=STRETCH, hide_index=True)
        if bool(check["idêntico"].all()):
            st.success("Replay determinístico confirmado em todos os métodos.", icon="✅")
        else:
            st.error("Divergência detectada — investigar fonte de não determinismo.", icon="🚨")

    with st.expander("Ambiente de execução registrado no PROV Agent"):
        st.json(prov.environment_fingerprint())


# ══════════════════════════════════════════════════════════════════════════
#  8 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("F4 — a pergunta certa não é 'Python é lento', é 'está vetorizado?'")
    st.markdown(
        "O único trecho realmente intensivo em CPU é o Monte Carlo. Abaixo, a **mesma** "
        "simulação em três implementações, medida ao vivo nesta máquina."
    )
    bench_sims = st.select_slider("Simulações no benchmark", [1_000, 5_000, 10_000, 25_000],
                                  value=5_000, key="bench_sims")
    bench_days = st.slider("Dias no horizonte", 1, 22, 10, key="bench_days")

    if st.button("⚡ Rodar benchmark agora", width=STRETCH):
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        rows = []

        t0 = time.perf_counter()
        rng = np.random.default_rng(42)
        acc = []
        for _ in range(bench_sims):          # laço Python explícito (o anti-padrão)
            total = 0.0
            for _ in range(bench_days):
                total += mu + sigma * rng.standard_normal()
            acc.append(total)
        rows.append({"implementação": "Laço Python puro", "tempo_ms": (time.perf_counter() - t0) * 1000})

        t0 = time.perf_counter()
        rng = np.random.default_rng(42)
        z = rng.standard_normal((bench_sims, bench_days))
        _ = (mu + sigma * z).sum(axis=1)
        rows.append({"implementação": "NumPy vetorizado", "tempo_ms": (time.perf_counter() - t0) * 1000})

        t0 = time.perf_counter()
        _ = risk.simulate_paths(mu, garch, n_sims=bench_sims, n_days=bench_days, seed=42)
        rows.append({"implementação": "NumPy + recursão GARCH", "tempo_ms": (time.perf_counter() - t0) * 1000})

        bench = pd.DataFrame(rows)
        speedup = bench.loc[0, "tempo_ms"] / bench.loc[1, "tempo_ms"]
        st.session_state["bench"] = (bench, speedup, bench_sims, bench_days)

    if "bench" in st.session_state:
        bench, speedup, bs, bd = st.session_state["bench"]
        st.plotly_chart(charts.benchmark_bar(bench), width=STRETCH)
        st.success(
            f"Vetorizar deu **{speedup:.0f}×** de ganho com {bs:,} simulações × {bd} dias — "
            "sem trocar de linguagem.".replace(",", "."), icon="⚡",
        )
        st.dataframe(bench, width=STRETCH, hide_index=True,
                     column_config={"tempo_ms": st.column_config.NumberColumn(format="%.2f ms")})

    st.divider()
    st.markdown("**Convergência do estimador** — quantas simulações são realmente necessárias")
    conv = risk.montecarlo_convergence(returns, confidence, (500, 1_000, 5_000, 10_000, 50_000, 100_000),
                                       dist=st.session_state["mc_dist"], seed=int(st.session_state["seed"]))
    st.plotly_chart(charts.convergence_chart(conv, var_results["historical"].value), width=STRETCH)
    st.plotly_chart(
        charts.montecarlo_distribution(sims, var_results["montecarlo"].value,
                                       es_results["montecarlo"].value, confidence),
        width=STRETCH,
    )
    st.caption(
        f"Erro padrão do VaR Monte Carlo com {n_sims:,} simulações: "
        f"±{var_results['montecarlo'].params['mc_std_error']:.3%} — abaixo da terceira casa "
        "decimal, o número deixa de ser sensível à semente.".replace(",", ".")
    )

    st.divider()
    st.markdown(
        """
| Abordagem | 100k simulações × 10 dias | Veredito |
|---|---|---|
| Laço Python puro | ~30–90 s | ❌ Evitar |
| **NumPy vetorizado** | **~50–300 ms** | ✅ **Adotado** |
| NumPy + Numba (`@njit`) | ~5–30 ms | Reforço opcional (F4) |
| Go/Rust nativo | ~2–15 ms | Só como experimento de avaliação |

O escopo é **batch diário**, não intraday: não existe requisito de latência
sub-segundo que justifique uma segunda linguagem no caminho crítico.
        """
    )


# ══════════════════════════════════════════════════════════════════════════
#  9 — ARQUITETURA
# ══════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader("Stack tecnológica decidida")
    st.dataframe(
        pd.DataFrame({
            "Camada": ["Linguagem", "API", "Motor de risco", "Banco relacional", "Grafo de proveniência",
                       "Cache", "Frontend", "Gráficos", "Grafo (UI)", "Orquestração", "Ambiente", "Testes"],
            "Tecnologia": ["Python 3.12", "FastAPI", "NumPy + SciPy + arch", "PostgreSQL 16",
                           "Neo4j 5 Community", "Redis 7", "Streamlit", "Plotly", "streamlit-agraph",
                           "AkôFlow (K8s)", "uv + uv.lock", "pytest + ruff + CI"],
            "Justificativa": [
                "Zona de força do autor; ecossistema científico maduro",
                "Async nativo, OpenAPI automático, baixa curva de aprendizado",
                "Monte Carlo vetorizado dispensa uma segunda linguagem",
                "Séries tabulares exigem ACID, joins e agregações temporais",
                "Lineage é intrinsecamente um grafo — Cypher > SQL recursivo",
                "Cache de features de baixa latência",
                "Alinhado ao stack Python; foco no motor, não no produto",
                "Único com candlestick nativo + melhor integração Streamlit",
                "Renderização nativa sem iframe",
                "Já captura proveniência W3C PROV de workflows",
                "Lockfile + reprodutibilidade de builds",
                "Testes com valor de referência independente (nunca circular)",
            ],
        }),
        width=STRETCH, hide_index=True,
    )

    st.divider()
    st.subheader("Fronteiras de responsabilidade")
    st.code(
        "Streamlit (dash/) --httpx--> FastAPI (/risk/var, /provenance/{run_id})\n"
        "                                   |\n"
        "                                   +--> finprov.risk   (NumPy/SciPy/arch)\n"
        "                                   +--> PostgreSQL     (séries e resultados)\n"
        "                                   +--> Neo4j          (grafo PROV)\n"
        "                                   +--> Redis          (cache de features)\n\n"
        "Regra: o dashboard NUNCA acessa Postgres/Neo4j direto — a API é a única porta de entrada.",
        language="text",
    )

    st.subheader("Onde este protótipo encaixa no repositório MRE")
    st.dataframe(
        pd.DataFrame({
            "Módulo desta tela": ["finprov_demo/risk.py", "finprov_demo/risk.py",
                                  "finprov_demo/risk.py", "finprov_demo/backtesting.py",
                                  "finprov_demo/data.py", "finprov_demo/provenance.py",
                                  "finprov_demo/charts.py"],
            "Destino em src/finprov/": ["risk/var.py", "risk/es.py + risk/volatility.py",
                                        "risk/portfolio.py", "risk/backtesting.py",
                                        "etl/extract_yahoo.py + etl/transform.py",
                                        "provenance/tracker.py", "dash/components/charts.py"],
            "Observação": [
                "5 estimadores já no contrato RiskResult",
                "ES nas 3 fontes; GARCH via arch",
                "Covariância + VaR componente",
                "Kupiec, Christoffersen, LR_cc, Basileia",
                "Trocar fallback sintético por carga do Postgres",
                "Trocar exportação Cypher por driver Neo4j",
                "Reaproveitável quase sem alteração",
            ],
        }),
        width=STRETCH, hide_index=True,
    )

    st.divider()
    st.subheader("Contrato de API planejado (FastAPI)")
    st.dataframe(
        pd.DataFrame({
            "Método": ["POST", "POST", "GET", "GET", "GET"],
            "Rota": ["/risk/var", "/risk/es", "/portfolio/{id}/risk-history",
                     "/provenance/{run_id}", "/health"],
            "Função": [
                "VaR pelo método escolhido (historical | parametric | montecarlo)",
                "Expected Shortfall",
                "Série histórica de VaR/ES de uma carteira",
                "Grafo de proveniência de uma execução (proxy Cypher)",
                "Healthcheck de Postgres, Neo4j e Redis",
            ],
            "Execução": ["run_in_threadpool se montecarlo", "async", "async", "async", "async"],
        }),
        width=STRETCH, hide_index=True,
    )
    st.caption(
        "Endpoints de leitura são `async def` puros; o Monte Carlo é CPU-bound e vai para "
        "`run_in_threadpool` — rodá-lo direto na rota bloquearia o event loop."
    )


# ══════════════════════════════════════════════════════════════════════════
#  10 — ROADMAP
# ══════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.subheader("Roadmap de implementação")
    roadmap = pd.DataFrame({
        "Etapa": ["F1.1", "F1.2", "F1.3", "F2.1", "F2.2", "F2.3", "F3.1", "F3.2", "F3.3", "F4", "F5"],
        "Entrega": [
            "VaR Histórico", "VaR Paramétrico (Delta-Normal)", "Retornos e volatilidade rolante",
            "Modelo GARCH(1,1)", "VaR Monte Carlo (GBM + GARCH)", "Expected Shortfall",
            "Backtesting — Kupiec (POF)", "Backtesting — Christoffersen",
            "VaR de carteira multi-ativo", "Otimização/benchmark de performance",
            "Camada de proveniência (Neo4j)",
        ],
        "Módulo alvo": [
            "risk/var.py", "risk/var.py", "etl/transform.py", "risk/volatility.py",
            "risk/var.py", "risk/es.py", "risk/backtesting.py", "risk/backtesting.py",
            "risk/portfolio.py", "risk/_kernels.py", "provenance/tracker.py",
        ],
        "No MRE": ["✅ Concluído", "🟡 Próximo", "⚪ Planejado", "⚪ Planejado", "⚪ Planejado",
                   "⚪ Planejado", "⚪ Planejado", "⚪ Planejado", "⚪ Planejado", "⚪ Planejado",
                   "⚪ Planejado"],
        "Nesta tela": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅ (sem Numba)", "✅ (export, sem driver)"],
    })
    st.dataframe(roadmap, width=STRETCH, hide_index=True)

    st.divider()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Próximos passos no MRE")
        st.markdown(
            """
1. **Portar `RiskResult`** para `src/finprov/risk/types.py` e migrar `historical_var`
   para retorná-lo — sem isso, cada função nova gera retrabalho na proveniência.
2. **`parametric_var`** com teste de valor de referência independente (nunca circular).
3. **Esqueleto FastAPI** (`main.py` + `/health`) rodando no ambiente `uv`, CI verde.
4. **GARCH + Monte Carlo** — a etapa mais complexa, já com fundação testada.
5. **Neo4j**: trocar a exportação Cypher desta tela pelo driver oficial.
            """
        )
    with c2:
        st.subheader("O que esta tela já resolveu de risco de projeto")
        st.markdown(
            """
- Provou que o **motor inteiro cabe em Python** (benchmark reprodutível na aba Performance).
- Fechou o **contrato `RiskResult`** com `run_id` determinístico — testado por replay.
- Validou que **`arch` estima GARCH(1,1)** sobre dados reais da B3 sem tuning manual.
- Definiu o **formato de exportação PROV** (Cypher + PROV-JSON) antes de subir o Neo4j.
- Mostrou que **Plotly + Streamlit** entregam o dashboard sem front-end dedicado.
            """
        )
    st.caption(
        "Esta tela é um protótipo de proposta: a lógica de `finprov_demo/` é a mesma que será "
        "portada para `src/finprov/` — com testes — quando a Fase 2/3 começar no repositório MRE."
    )
