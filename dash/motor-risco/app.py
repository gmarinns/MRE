"""FinProv — painel do motor de risco de mercado.

    streamlit run app.py

Este arquivo monta a navegação e a barra lateral; cada página fica em
``paginas/``. Os cálculos vêm do pacote ``motor_calculo``, sempre por
intermédio de ``servicos.py``.
"""

from __future__ import annotations

import streamlit as st

import dados
import servicos

st.set_page_config(
    page_title="FinProv · Motor de Risco",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ATIVOS = {
    "AGRO3.SA": "BrasilAgro ON",
    "VALE3.SA": "Vale ON",
    "BBSE3.SA": "BB Seguridade ON",
    "FLRY3.SA": "Fleury ON",
    "SAPR4.SA": "Sanepar PN",
    "GOAU3.SA": "Metalúrgica Gerdau ON",
    "KLBN3.SA": "Klabin ON",
    "PETR4.SA": "Petrobras PN",
    "ABEV3.SA": "Ambev ON",
    "CPFE3.SA": "CPFL Energia ON",
    "BBAS3.SA": "Banco do Brasil ON",
    "VLID3.SA": "Valid ON",
    "^BVSP": "Ibovespa",
}
PERIODOS = {"1y": "1 ano", "2y": "2 anos", "5y": "5 anos"}

PADRAO = {
    "ativos": [ticker for ticker in ATIVOS if ticker != "^BVSP"],
    "foco": "PETR4.SA",
    "periodo": "2y",
    "confianca": 0.95,
    "horizonte": 1,
    "posicao": 100_000,
    "mc_sims": 10_000,
    "mc_dist": "normal",
    "mc_vol": True,
    "mc_semente": 42,
}
for chave, valor in PADRAO.items():
    st.session_state.setdefault(chave, valor)
    # Reatribuir mantém o valor quando o widget não é desenhado na página atual.
    st.session_state[chave] = st.session_state[chave]

GRUPOS = {
    "": [
        st.Page("paginas/painel.py", title="Painel", icon=":material/dashboard:"),
    ],
    "Medidas de risco": [
        st.Page("paginas/var.py", title="Value at Risk", icon=":material/percent:"),
        st.Page(
            "paginas/es.py",
            title="Expected Shortfall",
            icon=":material/trending_down:",
        ),
        st.Page(
            "paginas/volatilidade.py",
            title="Volatilidade",
            icon=":material/show_chart:",
        ),
        st.Page("paginas/drawdown.py", title="Drawdown", icon=":material/south_east:"),
        st.Page("paginas/carteira.py", title="Carteira", icon=":material/pie_chart:"),
    ],
    "Validação": [
        st.Page("paginas/stress.py", title="Stress", icon=":material/bolt:"),
        st.Page(
            "paginas/backtesting.py", title="Backtesting", icon=":material/fact_check:"
        ),
    ],
    "Próxima etapa": [
        st.Page(
            "paginas/proveniencia.py",
            title="Proveniência (proposta)",
            icon=":material/account_tree:",
        ),
    ],
}
pagina = st.navigation(GRUPOS)

with st.sidebar:
    st.subheader("Parâmetros", divider="gray")
    ativos = st.multiselect(
        "Ativos da carteira",
        list(ATIVOS),
        key="ativos",
        format_func=lambda t: f"{t} · {ATIVOS[t]}",
    )
    if not ativos:
        st.warning("Selecione ao menos um ativo.", icon=":material/warning:")
        st.stop()
    if st.session_state["foco"] not in ativos:
        st.session_state["foco"] = ativos[0]
    st.selectbox(
        "Ativo em foco",
        ativos,
        key="foco",
        help="As páginas de medida analisam este ativo. A página Carteira usa todos.",
    )
    st.segmented_control(
        "Histórico",
        list(PERIODOS),
        key="periodo",
        format_func=PERIODOS.get,
        selection_mode="single",
    )
    st.select_slider(
        "Nível de confiança",
        [0.90, 0.95, 0.975, 0.99],
        key="confianca",
        format_func=lambda c: f"{c * 100:g}%".replace(".", ","),
    )
    st.select_slider(
        "Horizonte (dias úteis)",
        [1, 5, 10, 21],
        key="horizonte",
        help="O VaR de 1 dia é levado ao horizonte pela regra da raiz do tempo.",
    )
    st.number_input(
        "Valor da posição (R$)",
        min_value=1_000,
        step=10_000,
        key="posicao",
        help="Converte as perdas percentuais em reais.",
    )

try:
    with st.spinner("Carregando cotações e calculando…"):
        dados.selecionar_execucao()
        ctx = servicos.contexto()
except dados.DadosIndisponiveis as erro:
    st.error(str(erro), icon=":material/cloud_off:")
    st.stop()

with st.sidebar:
    cot = ctx.cotacoes
    st.caption(
        f":material/storage: PostgreSQL · {cot.pregoes} pregões até "
        f"{cot.frame['date'].iloc[-1]:%d/%m/%Y}"
    )
    st.caption(f"Execução: {cot.run_id}")

st.session_state["_ctx"] = ctx
pagina.run()
