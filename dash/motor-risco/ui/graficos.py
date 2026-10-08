"""Figuras Plotly do painel.

Convenções (ver ui/tema.py): dado observado na série 1 (azul); modelo ou
ajuste teórico na série 2 (laranja); vermelho e verde só indicam estado
(perda extrema, violação, zona do semáforo). Limiares de VaR e ES são
réguas tracejadas em tinta neutra, com rótulo ao lado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from ui.tema import com_alfa, cores, layout, pct


# ── Peças comuns ──────────────────────────────────────────────────────────
def _regua_vertical(
    fig: go.Figure, x: float, rotulo: str, nivel: int = 0, tracejado: str = "dash"
) -> None:
    """Régua vertical em tinta com rótulo; ``nivel`` escalona rótulos vizinhos."""
    c = cores()
    fig.add_vline(x=x, line={"color": c["tinta"], "width": 1.5, "dash": tracejado})
    fig.add_annotation(
        x=x,
        y=0.98 - 0.13 * nivel,
        yref="paper",
        text=rotulo,
        showarrow=False,
        xanchor="right",
        xshift=-6,
        font={"color": c["tinta"], "size": 12},
        bgcolor=com_alfa(c["superficie"], 0.85),
        borderpad=3,
    )


def _eixo_de_datas(fig: go.Figure) -> None:
    """Datas no formato brasileiro, sem nomes de mês em inglês."""
    fig.update_xaxes(tickformat="%m/%Y", hoverformat="%d/%m/%Y")


def _histograma(valores: np.ndarray, limiar: float, n_bins: int = 60):
    """Densidade em faixas que têm ``limiar`` como borda exata."""
    valores = np.asarray(valores, dtype=float)
    largura = (valores.max() - valores.min()) / n_bins
    esquerda = np.arange(limiar, valores.min() - largura, -largura)[::-1]
    direita = np.arange(limiar, valores.max() + largura, largura)
    bordas = np.concatenate([esquerda[:-1], direita])
    densidade, bordas = np.histogram(valores, bins=bordas, density=True)
    centros = (bordas[:-1] + bordas[1:]) / 2
    return centros, densidade, largura


# ── Distribuições ─────────────────────────────────────────────────────────
def distribuicao(
    valores: np.ndarray,
    limiar: float,
    reguas: list[tuple[float, str]],
    rotulo_corpo: str = "Retornos observados",
    rotulo_cauda: str = "Cauda de perdas",
    normal: tuple[float, float] | None = None,
    faixa_x: tuple[float, float] | None = None,
    titulo_x: str = "Log-retorno diário",
    altura: int = 400,
) -> go.Figure:
    """Histograma com a cauda (valores abaixo de ``limiar``) destacada."""
    c = cores()
    centros, densidade, largura = _histograma(valores, limiar)
    cauda = centros < limiar
    fig = go.Figure()
    for mascara, nome, cor in (
        (~cauda, rotulo_corpo, c["serie"][0]),
        (cauda, rotulo_cauda, c["critico"]),
    ):
        fig.add_bar(
            x=centros[mascara],
            y=densidade[mascara],
            width=largura * 0.92,
            name=nome,
            marker={"color": cor, "line": {"width": 0}},
            hovertemplate="%{x:.2%}<extra>" + nome + "</extra>",
        )
    if normal is not None:
        mu, sigma = normal
        grade = np.linspace(valores.min(), valores.max(), 300)
        fig.add_scatter(
            x=grade,
            y=stats.norm.pdf(grade, mu, sigma),
            mode="lines",
            name="Normal ajustada",
            line={"color": c["serie"][1], "width": 2},
            hoverinfo="skip",
        )
    for nivel, (x, rotulo) in enumerate(reguas):
        _regua_vertical(fig, x, rotulo, nivel, "dash" if nivel == 0 else "dot")
    fig.update_layout(**layout(altura, hovermode="closest", barmode="overlay"))
    fig.update_xaxes(title_text=titulo_x, tickformat=".1%")
    fig.update_yaxes(title_text="Densidade", showticklabels=False)
    if faixa_x is not None:
        fig.update_xaxes(range=list(faixa_x))
    return fig


def qq_normal(retornos: np.ndarray, altura: int = 360) -> go.Figure:
    """Quantis observados contra os de uma normal: desvio da reta = cauda pesada."""
    c = cores()
    observado = np.sort(stats.zscore(retornos))
    n = len(observado)
    teorico = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    limite = [float(teorico.min()), float(teorico.max())]
    fig = go.Figure()
    fig.add_scatter(
        x=limite,
        y=limite,
        mode="lines",
        name="Se fosse normal",
        line={"color": c["tinta3"], "width": 1.5, "dash": "dash"},
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=teorico,
        y=observado,
        mode="markers",
        name="Retornos observados",
        marker={"color": c["serie"][0], "size": 6, "opacity": 0.65},
        hovertemplate="teórico %{x:.2f} · observado %{y:.2f}<extra></extra>",
    )
    fig.update_layout(**layout(altura, hovermode="closest"))
    fig.update_xaxes(title_text="Quantil teórico (desvios-padrão)")
    fig.update_yaxes(title_text="Quantil observado (desvios-padrão)")
    return fig


# ── Séries no tempo ───────────────────────────────────────────────────────
def candlestick(frame: pd.DataFrame, altura: int = 420) -> go.Figure:
    c = cores()
    fig = go.Figure(
        go.Candlestick(
            x=frame["date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            increasing={"line": {"color": c["bom"], "width": 1}},
            decreasing={"line": {"color": c["critico"], "width": 1}},
            showlegend=False,
        )
    )
    fig.update_layout(**layout(altura, legenda=False, hovermode="x"))
    fig.update_xaxes(rangeslider_visible=False)
    _eixo_de_datas(fig)
    fig.update_yaxes(title_text="Preço ajustado", tickprefix="R$ ")
    return fig


def linhas(
    x,
    series: list[tuple[str, np.ndarray]],
    formato_y: str = ".1%",
    titulo_y: str | None = None,
    formato_x: str | None = None,
    titulo_x: str | None = None,
    marcadores: bool = False,
    altura: int = 380,
) -> go.Figure:
    """Até quatro séries no mesmo eixo, nas cores da paleta em ordem fixa.

    Sem ``formato_x`` o eixo horizontal é tratado como datas.
    """
    c = cores()
    fig = go.Figure()
    for i, (nome, y) in enumerate(series):
        fig.add_scatter(
            x=x,
            y=y,
            mode="lines+markers" if marcadores else "lines",
            name=nome,
            line={"color": c["serie"][i], "width": 2},
            marker={"size": 8, "line": {"color": c["superficie"], "width": 2}},
            hovertemplate="%{y:" + formato_y + "}<extra>" + nome + "</extra>",
        )
    fig.update_layout(**layout(altura, legenda=len(series) > 1))
    fig.update_yaxes(tickformat=formato_y, title_text=titulo_y)
    if formato_x is None:
        _eixo_de_datas(fig)
    else:
        fig.update_xaxes(tickformat=formato_x)
    fig.update_xaxes(title_text=titulo_x)
    return fig


def linha_com_referencia(
    x,
    y: np.ndarray,
    nome: str,
    referencia: float,
    rotulo_referencia: str,
    formato_y: str = ".1%",
    titulo_x: str | None = None,
    marcadores: bool = False,
    datas: bool = False,
    altura: int = 340,
) -> go.Figure:
    """Uma série e uma linha horizontal de referência rotulada."""
    c = cores()
    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=y,
        mode="lines+markers" if marcadores else "lines",
        name=nome,
        line={"color": c["serie"][0], "width": 2},
        marker={"size": 8, "line": {"color": c["superficie"], "width": 2}},
        hovertemplate="%{y:" + formato_y + "}<extra>" + nome + "</extra>",
    )
    fig.add_hline(
        y=referencia,
        line={"color": c["tinta3"], "width": 1.5, "dash": "dash"},
        annotation_text=rotulo_referencia,
        annotation_position="top right",
        annotation_font={"color": c["tinta"], "size": 12},
        annotation_bgcolor=com_alfa(c["superficie"], 0.85),
    )
    fig.update_layout(**layout(altura, legenda=False))
    fig.update_xaxes(title_text=titulo_x)
    fig.update_yaxes(tickformat=formato_y)
    if datas:
        _eixo_de_datas(fig)
    return fig


def underwater(datas, drawdown: np.ndarray, altura: int = 400) -> go.Figure:
    """Perda acumulada desde o último topo, com o pior ponto rotulado."""
    c = cores()
    pior = int(np.argmin(drawdown))
    datas = pd.to_datetime(pd.Series(datas)).reset_index(drop=True)
    fig = go.Figure()
    fig.add_scatter(
        x=datas,
        y=drawdown,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line={"color": c["serie"][0], "width": 1.5},
        fillcolor=com_alfa(c["serie"][0], 0.14),
        hovertemplate="%{y:.1%}<extra>abaixo do topo</extra>",
    )
    fig.add_scatter(
        x=[datas[pior]],
        y=[drawdown[pior]],
        mode="markers",
        marker={
            "color": c["serie"][0],
            "size": 10,
            "line": {"color": c["superficie"], "width": 2},
        },
        hoverinfo="skip",
    )
    fig.add_annotation(
        x=datas[pior],
        y=drawdown[pior],
        text=f"Pior ponto: {pct(drawdown[pior], 1)} em {datas[pior]:%d/%m/%Y}",
        showarrow=False,
        yshift=-16,
        font={"color": c["tinta"], "size": 12},
    )
    fig.update_layout(**layout(altura, legenda=False))
    _eixo_de_datas(fig)
    fig.update_yaxes(tickformat=".0%", title_text="Distância do topo")
    return fig


def backtest(
    datas, retornos: np.ndarray, var_series: np.ndarray, violou: np.ndarray
) -> go.Figure:
    """Retorno realizado contra o VaR previsto para cada dia."""
    c = cores()
    datas = np.asarray(datas)
    fig = go.Figure()
    fig.add_scatter(
        x=datas,
        y=retornos,
        mode="lines",
        name="Retorno realizado",
        line={"color": c["serie"][0], "width": 1.2},
        hovertemplate="%{y:.2%}<extra>retorno</extra>",
    )
    fig.add_scatter(
        x=datas,
        y=-var_series,
        mode="lines",
        name="Limite previsto (−VaR)",
        line={"color": c["serie"][1], "width": 2},
        hovertemplate="%{y:.2%}<extra>−VaR previsto</extra>",
    )
    if violou.any():
        fig.add_scatter(
            x=datas[violou],
            y=retornos[violou],
            mode="markers",
            name=f"Violação ({int(violou.sum())})",
            marker={
                "color": c["critico"],
                "size": 10,
                "symbol": "x",
                "line": {"color": c["superficie"], "width": 1},
            },
            hovertemplate="%{y:.2%}<extra>violação</extra>",
        )
    fig.update_layout(**layout(420))
    _eixo_de_datas(fig)
    fig.update_yaxes(tickformat=".1%")
    return fig


def faixa_de_violacoes(datas, violou: np.ndarray) -> go.Figure:
    """Um traço por violação ao longo do tempo: evidencia o agrupamento."""
    c = cores()
    datas = np.asarray(datas)
    fig = go.Figure()
    fig.add_scatter(
        x=datas[violou],
        y=np.zeros(int(violou.sum())),
        mode="markers",
        marker={
            "symbol": "line-ns",
            "size": 22,
            "line": {"color": c["critico"], "width": 2},
        },
        hovertemplate="%{x|%d/%m/%Y}<extra>violação</extra>",
    )
    fig.update_layout(**layout(110, legenda=False, hovermode="closest"))
    fig.update_xaxes(range=[datas[0], datas[-1]])
    _eixo_de_datas(fig)
    fig.update_yaxes(visible=False, range=[-1, 1])
    return fig


# ── Monte Carlo ───────────────────────────────────────────────────────────
def leque(percentis: pd.DataFrame, horizonte: int, altura: int = 420) -> go.Figure:
    """Faixas de percentil das trajetórias simuladas."""
    c = cores()
    dia = percentis["dia"]
    fig = go.Figure()
    faixas = (("p05", "p95", "Faixa 5%–95%", 0), ("p25", "p75", "Faixa 25%–75%", 2))
    for baixo, alto, nome, tom in faixas:
        fig.add_scatter(
            x=dia, y=percentis[alto], mode="lines", line={"width": 0},
            showlegend=False, hoverinfo="skip",
        )  # fmt: skip
        fig.add_scatter(
            x=dia,
            y=percentis[baixo],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=com_alfa(c["rampa"][tom], 0.55),
            name=nome,
            hoverinfo="skip",
        )
    fig.add_scatter(
        x=dia,
        y=percentis["p50"],
        mode="lines",
        name="Mediana",
        line={"color": c["serie"][0], "width": 2},
        hovertemplate="%{y:.2%}<extra>mediana</extra>",
    )
    fig.add_scatter(
        x=dia,
        y=percentis["var"],
        mode="lines",
        name="Percentil do VaR",
        line={"color": c["tinta"], "width": 1.5, "dash": "dash"},
        hovertemplate="%{y:.2%}<extra>percentil do VaR</extra>",
    )
    if 1 < horizonte <= int(dia.max()):
        fig.add_vline(x=horizonte, line={"color": c["eixo"], "width": 1})
        fig.add_annotation(
            x=horizonte,
            y=0.02,
            yref="paper",
            text=f"horizonte: {horizonte} dias",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font={"color": c["tinta2"], "size": 12},
        )
    fig.update_layout(**layout(altura))
    fig.update_xaxes(title_text="Dias úteis à frente")
    fig.update_yaxes(tickformat=".0%", title_text="Retorno acumulado simulado")
    return fig


def convergencia(tabela: pd.DataFrame, altura: int = 320) -> go.Figure:
    """VaR estimado e intervalo de 95% conforme cresce o número de simulações."""
    c = cores()
    fig = go.Figure(
        go.Scatter(
            x=tabela["n_sims"],
            y=tabela["var"],
            mode="lines+markers",
            line={"color": c["serie"][0], "width": 2},
            marker={"size": 8, "line": {"color": c["superficie"], "width": 2}},
            error_y={
                "type": "data",
                "array": 1.96 * tabela["std_error"],
                "color": c["tinta3"],
                "thickness": 1.2,
                "width": 4,
            },
            hovertemplate="%{x:,} simulações: %{y:.3%}<extra></extra>",
        )
    )
    fig.update_layout(**layout(altura, legenda=False, hovermode="closest"))
    fig.update_xaxes(
        type="log",
        title_text="Número de simulações (escala log)",
        tickmode="array",
        tickvals=list(tabela["n_sims"]),
        ticktext=[f"{n:,}".replace(",", ".") for n in tabela["n_sims"]],
    )
    fig.update_yaxes(tickformat=".2%", title_text="VaR estimado")
    return fig


# ── Barras ────────────────────────────────────────────────────────────────
def barras_horizontais(
    rotulos: list[str],
    valores: list[float],
    destaque: str | None = None,
    altura: int | None = None,
) -> go.Figure:
    """Barras de uma única série; ``destaque`` fica na cor, as demais em cinza."""
    c = cores()
    cor = [c["serie"][0] if destaque in (None, r) else c["neutro"] for r in rotulos]
    fig = go.Figure(
        go.Bar(
            x=valores,
            y=rotulos,
            orientation="h",
            width=0.5,
            marker={"color": cor, "cornerradius": 4},
            text=[pct(v) for v in valores],
            textposition="outside",
            textfont={"color": c["tinta"], "size": 13},
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    altura = altura or 70 + 44 * len(rotulos)
    fig.update_layout(**layout(altura, legenda=False, hovermode="closest"))
    fig.update_xaxes(tickformat=".1%", range=[0, max(valores) * 1.18])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def colunas_ordenadas(
    rotulos: list[str], valores: list[float], altura: int = 360
) -> go.Figure:
    """Colunas em rampa de um matiz: a ordem dos cenários é a severidade."""
    c = cores()
    tons = [c["rampa"][i] for i in np.linspace(1, 5, len(rotulos)).astype(int)]
    fig = go.Figure(
        go.Bar(
            x=rotulos,
            y=valores,
            width=0.3,
            marker={"color": tons, "cornerradius": 4},
            text=[pct(v) for v in valores],
            textposition="outside",
            textfont={"color": c["tinta"], "size": 13},
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(**layout(altura, legenda=False, hovermode="closest"))
    fig.update_yaxes(tickformat=".0%", range=[0, max(valores) * 1.15])
    fig.update_xaxes(showgrid=False)
    return fig


def cascata_diversificacao(
    soma_individual: float, beneficio: float, var_carteira: float, altura: int = 380
) -> go.Figure:
    """Do risco somado ao risco da carteira: quanto a diversificação retira."""
    c = cores()
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Soma dos VaR individuais", "Diversificação", "VaR da carteira"],
            y=[soma_individual, -beneficio, 0],
            width=0.3,
            text=[pct(soma_individual), pct(-beneficio, sinal=True), pct(var_carteira)],
            textposition="outside",
            textfont={"color": c["tinta"], "size": 13},
            cliponaxis=False,
            increasing={"marker": {"color": c["neutro"]}},
            decreasing={"marker": {"color": c["serie"][2]}},
            totals={"marker": {"color": c["serie"][0]}},
            connector={"line": {"color": c["eixo"], "width": 1}},
            hovertemplate="%{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(**layout(altura, legenda=False, hovermode="closest"))
    fig.update_yaxes(tickformat=".1%", range=[0, soma_individual * 1.18])
    fig.update_xaxes(showgrid=False)
    return fig


def correlacao(matriz: pd.DataFrame, altura: int = 380) -> go.Figure:
    """Mapa de correlação em escala divergente com cinza no zero."""
    c = cores()
    escala = [
        [0.0, c["divergente"][0]],
        [0.5, c["divergente"][1]],
        [1.0, c["divergente"][2]],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=matriz.to_numpy(),
            x=list(matriz.columns),
            y=list(matriz.index),
            zmin=-1,
            zmax=1,
            colorscale=escala,
            xgap=2,
            ygap=2,
            texttemplate="%{z:.2f}",
            textfont={"size": 13},
            colorbar={"thickness": 10, "len": 0.8, "tickfont": {"color": c["tinta3"]}},
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**layout(altura, legenda=False, hovermode="closest"))
    fig.update_xaxes(showgrid=False, ticks="")
    fig.update_yaxes(showgrid=False, autorange="reversed", ticks="")
    return fig


def regua_basileia(observado: int, limites: dict, altura: int = 170) -> go.Figure:
    """Zonas do semáforo de Basileia com a contagem observada marcada."""
    c = cores()
    amarela, vermelha = limites["yellow_from"], limites["red_from"]
    fim = max(vermelha * 1.35, observado * 1.1, vermelha + 3)
    zonas = (
        ("Verde", 0, amarela, c["bom"]),
        ("Amarela", amarela, vermelha, c["alerta"]),
        ("Vermelha", vermelha, fim, c["critico"]),
    )
    fig = go.Figure()
    for nome, inicio, termino, cor in zonas:
        fig.add_shape(
            type="rect", x0=inicio, x1=termino, y0=0, y1=1,
            fillcolor=com_alfa(cor, 0.28), line={"width": 0}, layer="below",
        )  # fmt: skip
        fig.add_annotation(
            x=(inicio + termino) / 2, y=0.5, text=nome, showarrow=False,
            font={"color": c["tinta"], "size": 13},
        )  # fmt: skip
    fig.add_scatter(
        x=[observado],
        y=[0.5],
        mode="markers",
        marker={
            "symbol": "diamond",
            "size": 16,
            "color": c["tinta"],
            "line": {"color": c["superficie"], "width": 2},
        },
        hovertemplate="%{x} violações observadas<extra></extra>",
    )
    fig.add_annotation(
        x=observado, y=1.0, yshift=12, text=f"observado: {observado}",
        showarrow=False, font={"color": c["tinta"], "size": 12},
    )  # fmt: skip
    fig.add_vline(
        x=limites["expected"],
        line={"color": c["tinta2"], "width": 1.5, "dash": "dot"},
    )
    fig.add_annotation(
        x=limites["expected"], y=1.0, yshift=30, showarrow=False,
        text="esperado: " + f"{limites['expected']:.1f}".replace(".", ","),
        font={"color": c["tinta2"], "size": 12},
    )  # fmt: skip
    extra = {"margin": {"l": 8, "r": 16, "t": 48, "b": 34}}
    fig.update_layout(**layout(altura, legenda=False, hovermode="closest", **extra))
    fig.update_xaxes(range=[0, fim], title_text="Número de violações", showgrid=False)
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig
