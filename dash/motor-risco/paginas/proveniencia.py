"""Proveniência: proposta de onde ela entra. Ainda não implementada."""

import pandas as pd
import streamlit as st

import servicos
from ui import componentes as ui
from ui.tema import ROTULO_METODO, cores, pct

ctx = servicos.atual()
c = cores()

ui.cabecalho(
    "Proveniência",
    "Se o VaR de hoje é diferente do de ontem, o que mudou: o dado, o "
    "parâmetro, o código ou o mercado?",
)
st.info(
    "**Proposta para validação.** A camada de proveniência ainda não foi "
    "implementada. Esta página mostra onde ela se encaixa no que já existe e "
    "o que precisa ser decidido antes de construí-la.",
    icon=":material/draft:",
)

# ── 1. Onde capturar ──────────────────────────────────────────────────────
st.subheader("Onde a proveniência seria capturada", divider="gray")

ETAPAS = (
    ("dado", "", "Yahoo Finance", "fonte externa"),
    ("etapa", "1", "Ingestão", "ingest/yahoo.py"),
    ("dado", "", "Cotações brutas", "raw.cotacoes"),
    ("etapa", "2", "Transformação", "etl/precos.py"),
    ("dado", "", "Preços tratados", "curated.precos"),
    ("etapa", "3", "Cálculo de risco", "motor_calculo"),
    ("dado", "4", "Resultado", "VaR, ES, backtest"),
)
ESTILO = f"""
<style>
.prov-fluxo {{ display: flex; align-items: stretch; gap: 4px; }}
.prov-caixa {{ flex: 1 1 0; min-width: 0; padding: 12px 6px; text-align: center;
  border: 1px solid {c["eixo"]}; color: {c["tinta"]}; position: relative; }}
.prov-dado {{ border-radius: 14px; background: {c["serie"][0]}26; }}
.prov-etapa {{ border-radius: 3px; background: {c["serie"][1]}30; }}
.prov-caixa b {{ display: block; font-size: 0.86rem; line-height: 1.25; }}
.prov-caixa span {{ font-size: 0.72rem; color: {c["tinta2"]};
  font-family: ui-monospace, monospace; overflow-wrap: anywhere; }}
.prov-num {{ position: absolute; top: -10px; left: -6px; width: 22px; height: 22px;
  border-radius: 50%; background: {c["tinta"]}; color: {c["superficie"]};
  font-size: 0.78rem; font-weight: 700; line-height: 22px; }}
.prov-seta {{ align-self: center; color: {c["tinta3"]}; font-size: 1rem; }}
.prov-agente {{ margin-top: 12px; padding: 9px 12px; text-align: center;
  border: 1.5px dashed {c["tinta3"]}; border-radius: 10px;
  color: {c["tinta2"]}; font-size: 0.88rem; }}
@media (max-width: 900px) {{
  .prov-fluxo {{ flex-direction: column; }}
  .prov-seta {{ transform: rotate(90deg); }}
}}
</style>
"""
caixas = []
for tipo, numero, titulo, detalhe in ETAPAS:
    selo = f'<div class="prov-num">{numero}</div>' if numero else ""
    caixas.append(
        f'<div class="prov-caixa prov-{tipo}">{selo}<b>{titulo}</b>'
        f"<span>{detalhe}</span></div>"
    )
fluxo = '<div class="prov-seta">&rarr;</div>'.join(caixas)
with st.container(border=True):
    st.html(
        ESTILO
        + f'<div class="prov-fluxo">{fluxo}</div>'
        + '<div class="prov-agente">Código e ambiente (commit, versões das '
        "bibliotecas) executam as etapas 1, 2 e 3</div>"
    )
    st.caption(
        "Caixas arredondadas são dados (entidades); caixas retas são etapas "
        "que os transformam (atividades); a faixa tracejada é quem executa "
        "(agente). É o vocabulário do padrão W3C PROV. Os números marcam os "
        "quatro pontos de captura."
    )

st.table(
    pd.DataFrame(
        {
            "Ponto de captura": [
                "1 · Ingestão",
                "2 · Transformação",
                "3 · Cálculo de risco",
                "4 · Resultado",
            ],
            "O que seria registrado": [
                "Fonte, ticker, intervalo pedido, horário da coleta, hash do "
                "retorno bruto",
                "Qual coleta foi usada, regras aplicadas (ajuste, limpeza), "
                "hash da tabela gerada",
                "Método, parâmetros, hash dos retornos de entrada, versão do "
                "código, ambiente",
                "Valor, data de referência, identificador da execução que o gerou",
            ],
            "Pergunta que passa a ter resposta": [
                "O dado de entrada mudou entre duas coletas?",
                "O número mudou por causa de uma regra de tratamento?",
                "O que foi diferente entre o cálculo de ontem e o de hoje?",
                "De onde veio este número exibido no painel?",
            ],
            "Já existe hoje": [
                "Tabela raw.cotacoes com payload e horário",
                "Tabela curated.precos (sem vínculo com a coleta)",
                "Hash, parâmetros e identificador devolvidos pelo motor",
                "Nada persistido",
            ],
        }
    ).set_index("Ponto de captura")
)

# ── 2. O gancho que já existe ─────────────────────────────────────────────
st.subheader("O gancho que já existe no motor", divider="gray")
esquerda, direita = st.columns([2, 3], gap="medium")
with esquerda:
    st.markdown(
        "Toda função de `motor_calculo` devolve um `RiskResult`: o número e "
        "os metadados da execução. Ao lado, o registro **real** do VaR que "
        "está no painel agora."
    )
    escolha = st.selectbox(
        "Execução",
        servicos.METODOS,
        format_func=ROTULO_METODO.get,
        key="prov_metodo",
    )
    r = ctx.var[escolha]
    st.markdown(
        f"""
- **`input_hash`** identifica a série de {len(ctx.retornos)} retornos usada.
  Se uma cotação for revisada, o hash muda.
- **`run_id`** é determinístico: mesmo insumo, método e parâmetros geram o
  mesmo identificador. Recalcular reproduz o resultado.
- **`params`** guarda tudo o que influenciou o número ({pct(r.value)}).
"""
    )
with direita, st.container(border=True):
    st.json(r.to_dict(), expanded=2)

# ── 3. Onde entra no código ───────────────────────────────────────────────
st.subheader("Onde isso entra no código", divider="gray")
st.table(
    pd.DataFrame(
        {
            "Local": [
                "motor-calculo/motor_calculo/types.py",
                "dash/motor-risco/servicos.py",
                "src/finprov/ingest/yahoo.py",
                "src/finprov/etl/precos.py",
                "src/finprov/provenance/",
            ],
            "Papel": [
                "Contrato RiskResult: o que cada cálculo declara sobre si",
                "Ponto único por onde todo cálculo do painel passa",
                "Grava a coleta bruta em raw.cotacoes",
                "Gera curated.precos a partir da coleta mais recente",
                "Pacote já reservado no repositório, hoje vazio",
            ],
            "Mudança proposta": [
                "Nenhuma: já carrega hash, parâmetros e identificador",
                "Após cada cálculo, enviar o RiskResult ao registro de proveniência",
                "Registrar o hash do payload e devolver o id da coleta",
                "Gravar em cada linha o id da coleta de origem, em vez de sobrescrever",
                "Receber os registros e gravá-los como grafo",
            ],
        }
    ).set_index("Local")
)

# ── 4. Decisões em aberto ─────────────────────────────────────────────────
st.subheader("Pontos para decidir com o orientador", divider="gray")
perguntas = (
    (
        "Granularidade",
        "Um registro por execução do motor, ou um por ativo e por dia? O "
        "backtesting de 2 anos gera centenas de VaR: cada um é uma execução?",
    ),
    (
        "Armazenamento",
        "Grafo em Neo4j (já previsto no docker-compose) ou tabelas "
        "relacionais no Postgres com consultas recursivas? O que o trabalho "
        "ganha com o grafo?",
    ),
    (
        "Dados revisados",
        "O ETL atual sobrescreve o preço quando o Yahoo revisa uma cotação. "
        "Para explicar mudanças no VaR é preciso guardar as duas versões. "
        "Vale mudar o modelo de dados agora?",
    ),
    (
        "Reprodutibilidade",
        "Reproduzir bit a bit exige fixar versões de bibliotecas e a semente "
        "do Monte Carlo. Até onde o trabalho precisa garantir isso?",
    ),
    (
        "AkôFlow",
        "A proveniência do workflow (AkôFlow) e a do domínio (este motor) "
        "ficam no mesmo grafo ou são ligadas por identificador?",
    ),
    (
        "Avaliação",
        "Como demonstrar o ganho? Proposta: um experimento em que o VaR muda "
        "por três causas diferentes e o sistema aponta qual foi.",
    ),
)
for inicio in range(0, len(perguntas), 3):
    colunas = st.columns(3, gap="medium")
    for coluna, (tema, texto) in zip(
        colunas, perguntas[inicio : inicio + 3], strict=True
    ):
        with coluna, st.container(border=True):
            st.markdown(f"**{tema}**")
            st.markdown(texto)
