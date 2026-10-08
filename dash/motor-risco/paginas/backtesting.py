"""Backtesting: o VaR previsto no passado se confirmou?"""

import pandas as pd
import streamlit as st

import servicos
from ui import componentes as ui
from ui import graficos
from ui.tema import ROTULO_METODO, nivel, num, pct, pp

ctx = servicos.atual()
conf = nivel(ctx.confianca)
ret = ctx.retornos

ui.cabecalho(
    "Backtesting",
    "Se o VaR tivesse sido calculado todos os dias no passado, com que "
    "frequência a perda real o teria ultrapassado?",
)

c1, c2 = st.columns([2, 3])
metodo = c1.segmented_control(
    "Estimador testado",
    servicos.METODOS_ROLANTES,
    default="historical",
    format_func=ROTULO_METODO.get,
    key="bt_metodo",
)
metodo = metodo or "historical"
janela = c2.select_slider(
    "Janela de estimação (pregões)",
    [42, 63, 126, 252],
    value=126,
    key="bt_janela",
    help="Quantos pregões anteriores alimentam o VaR de cada dia.",
)

rel = servicos.backtest(ret, janela, ctx.confianca, metodo)
if rel is None:
    st.warning(
        "Histórico insuficiente para essa janela. Reduza a janela ou amplie o "
        "período na barra lateral.",
        icon=":material/warning:",
    )
    st.stop()

kupiec, indep, conj = rel["kupiec"], rel["independence"], rel["conditional"]
basileia, limites = rel["basel"], rel["thresholds"]
datas = ctx.datas[rel["validos"]].reset_index(drop=True)

ui.indicadores(
    [
        {
            "label": "Violações observadas",
            "value": f"{kupiec['violations']} em {kupiec['n_obs']}",
        },
        {
            "label": "Violações esperadas",
            "value": num(kupiec["expected_violations"], 1),
            "help": f"{nivel(1 - ctx.confianca)} dos dias testados.",
        },
        {
            "label": "Taxa de violação",
            "value": pct(kupiec["violation_rate"]),
            "delta": pp(kupiec["violation_rate"] - kupiec["expected_rate"], 2)
            + " vs. esperado",
            "delta_color": "inverse",
        },
        {
            "label": "Zona de Basileia",
            "value": basileia["zone"],
            "help": basileia["note"],
        },
    ]
)

ui.grafico(
    graficos.backtest(datas, rel["returns"], rel["var_series"], rel["mask"]),
    f"Retorno realizado e VaR {conf} previsto para cada dia",
    f"VaR {ROTULO_METODO[metodo].lower()} de 1 dia, estimado com os {janela} "
    "pregões anteriores. Nenhum dia usa informação do futuro.",
    chave="bt_serie",
)
if rel["mask"].any():
    ui.grafico(
        graficos.faixa_de_violacoes(datas, rel["mask"]),
        "Quando as violações aconteceram",
        "Violações agrupadas indicam que o modelo demora a reagir a mudanças "
        "de volatilidade.",
        dados=pd.DataFrame(
            {
                "Data": [f"{d:%d/%m/%Y}" for d in datas[rel["mask"]]],
                "Retorno": [pct(v) for v in rel["returns"][rel["mask"]]],
                "VaR previsto": [pct(v) for v in rel["var_series"][rel["mask"]]],
            }
        ),
        chave="bt_faixa",
    )

ui.grafico(
    graficos.regua_basileia(kupiec["violations"], limites),
    f"Semáforo de Basileia · zona {basileia['zone'].lower()}",
    basileia["note"]
    + " As zonas seguem a regra de Basileia (definida para VaR 99% em 250 "
    "dias), aplicada a esta amostra pela mesma probabilidade acumulada.",
    chave="bt_regua",
)

st.subheader("Testes estatísticos", divider="gray")
testes = (
    (
        "Kupiec · cobertura",
        "A taxa de violação é igual à esperada?",
        kupiec,
        "Taxa compatível com o nível de confiança.",
        "Taxa incompatível com o nível de confiança.",
    ),
    (
        "Christoffersen · independência",
        "As violações são independentes entre si?",
        indep,
        "Sem evidência de agrupamento.",
        "Violações agrupadas no tempo.",
    ),
    (
        "Cobertura condicional",
        "As duas coisas ao mesmo tempo?",
        conj,
        "Modelo não rejeitado no teste conjunto.",
        "Modelo rejeitado no teste conjunto.",
    ),
)
for coluna, (nome, pergunta, teste, bom, ruim) in zip(
    st.columns(3, gap="medium"), testes, strict=True
):
    with coluna, st.container(border=True):
        st.markdown(f"**{nome}**")
        st.caption(pergunta)
        st.markdown(
            f"LR = {num(teste['lr_stat'], 3)} · {teste['df']} g.l. · "
            f"p-valor = {num(teste['p_value'], 3)}"
        )
        ui.estado(not teste["reject_h0"], bom, ruim)

linhas = []
for m in servicos.METODOS_ROLANTES:
    outro = servicos.backtest(ret, janela, ctx.confianca, m)
    if outro is None:
        continue
    linhas.append(
        {
            "Estimador": ROTULO_METODO[m],
            "VaR médio": pct(outro["avg_var"]),
            "Violações": outro["kupiec"]["violations"],
            "Taxa": pct(outro["kupiec"]["violation_rate"]),
            "p-valor Kupiec": num(outro["kupiec"]["p_value"], 3),
            "p-valor independência": num(outro["independence"]["p_value"], 3),
            "p-valor conjunto": num(outro["conditional"]["p_value"], 3),
            "Basileia": outro["basel"]["zone"],
        }
    )
with st.container(border=True):
    st.markdown(f"**Os três estimadores na mesma janela de {janela} pregões**")
    st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True)
    st.caption(
        "p-valor abaixo de 0,05 rejeita a hipótese de que o modelo está correto."
    )

ui.como_ler(
    f"""
- Uma **violação** é um dia em que a perda real foi maior que o VaR previsto
  para aquele dia. A {conf}, espera-se violação em
  {nivel(1 - ctx.confianca)} dos dias: nem mais, nem menos.
- **Violações demais**: o modelo subestima o risco. **Violações de menos**:
  superestima, e imobiliza capital à toa. Kupiec testa os dois lados.
- **Christoffersen** olha a ordem: mesmo com a taxa certa, violações em
  sequência mostram que o modelo não acompanha a mudança de volatilidade.
- O **semáforo de Basileia** é o critério do regulador: verde aceita o modelo,
  amarelo aumenta o capital exigido, vermelho o rejeita.
"""
)
