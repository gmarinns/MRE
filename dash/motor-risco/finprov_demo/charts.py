"""Gráficos Plotly — todos herdam o tema ativo e usam fundo transparente."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .config import (
    METHOD_COLORS, METHOD_LABELS, PALETTE, annotation_bg, base_layout, plotly_template,
)
from .risk import RiskResult


def _apply(fig: go.Figure, title: str | None, height: int) -> go.Figure:
    fig.update_layout(**base_layout(title, height))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


def candlestick(frame: pd.DataFrame, ticker: str, height: int = 420) -> go.Figure:
    """OHLC + volume em painéis sincronizados."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=frame["date"], open=frame["open"], high=frame["high"],
            low=frame["low"], close=frame["close"], name=ticker,
            increasing=dict(line=dict(color=PALETTE["prov"]), fillcolor=PALETTE["prov"]),
            decreasing=dict(line=dict(color=PALETTE["risk"]), fillcolor=PALETTE["risk"]),
            showlegend=False,
        ),
        row=1, col=1,
    )
    if "volume" in frame:
        up = frame["close"].to_numpy() >= frame["open"].to_numpy()
        fig.add_trace(
            go.Bar(
                x=frame["date"], y=frame["volume"], name="Volume", showlegend=False,
                marker_color=np.where(up, PALETTE["prov"], PALETTE["risk"]), opacity=0.45,
            ),
            row=2, col=1,
        )
    _apply(fig, f"{ticker} — preço (OHLC) e volume", height)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x")
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def returns_histogram(
    returns: np.ndarray, results: list[RiskResult], height: int = 420,
    show_normal: bool = True,
) -> go.Figure:
    """Distribuição empírica + linhas de VaR/ES.

    As anotações ficam **escalonadas na vertical**: no beta os três VaR caíam
    quase na mesma abscissa e os rótulos se sobrepunham (e o da ponta era
    cortado pela borda do gráfico).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns, nbinsx=70, name="Retornos diários", histnorm="probability density",
            marker=dict(color=PALETTE["accent"], line=dict(width=0)), opacity=0.55,
        )
    )
    if show_normal:
        mu, sigma = float(np.mean(returns)), float(np.std(returns, ddof=1))
        grid = np.linspace(returns.min(), returns.max(), 400)
        fig.add_trace(
            go.Scatter(
                x=grid, y=stats.norm.pdf(grid, mu, sigma), mode="lines",
                name="Normal ajustada", line=dict(color=PALETTE["muted"], width=1.5, dash="dot"),
            )
        )

    ordered = sorted(results, key=lambda r: r.value)
    for i, r in enumerate(ordered):
        color = METHOD_COLORS.get(r.method, "#FFFFFF")
        fig.add_vline(x=-r.value, line=dict(color=color, width=1.6, dash="dash"))
        fig.add_annotation(
            x=-r.value, xref="x", yref="paper",
            y=0.97 - 0.115 * i, showarrow=False,
            text=f"<b>{METHOD_LABELS.get(r.method, r.method)}</b>  {r.value:.2%}",
            font=dict(color=color, size=11),
            bgcolor=annotation_bg(), borderpad=3,
            xanchor="left" if i % 2 == 0 else "right",
            xshift=6 if i % 2 == 0 else -6,
        )
    _apply(fig, "Distribuição dos retornos diários · VaR e ES estimados", height)
    fig.update_layout(hovermode="closest", bargap=0.02)
    fig.update_xaxes(title_text="Log-retorno diário", tickformat=".1%")
    fig.update_yaxes(title_text="Densidade")
    return fig


def var_backtest_chart(
    dates, returns: np.ndarray, var_series: np.ndarray, mask: np.ndarray, height: int = 420,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=returns, mode="lines", name="Retorno diário",
                   line=dict(color=PALETTE["accent"], width=1))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=-var_series, mode="lines", name="Limiar −VaR",
                   line=dict(color=PALETTE["warn"], width=1.8, dash="dash"))
    )
    if np.any(mask):
        fig.add_trace(
            go.Scatter(
                x=np.asarray(dates)[mask], y=returns[mask], mode="markers",
                name=f"Violações ({int(mask.sum())})",
                marker=dict(color=PALETTE["risk"], size=9, symbol="x", line=dict(width=1.4)),
            )
        )
    _apply(fig, "Backtesting — retorno realizado vs. VaR previsto (janela rolante)", height)
    fig.update_yaxes(tickformat=".1%")
    return fig


def violation_clustering(dates, mask: np.ndarray, height: int = 170) -> go.Figure:
    """Faixa temporal das violações — evidencia visualmente o clustering do Christoffersen."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.asarray(dates)[mask], y=np.ones(int(mask.sum())), mode="markers",
            marker=dict(color=PALETTE["risk"], size=11, symbol="line-ns",
                        line=dict(width=2.4, color=PALETTE["risk"])),
            name="violação", showlegend=False,
        )
    )
    _apply(fig, None, height)
    fig.update_yaxes(visible=False, range=[0.5, 1.5])
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=8, t=6, b=24))
    return fig


def correlation_heatmap(matrix: pd.DataFrame, height: int = 400) -> go.Figure:
    corr = matrix.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(), x=list(corr.columns), y=list(corr.columns),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.to_numpy(), 2), texttemplate="%{text}",
            textfont=dict(size=12), colorbar=dict(thickness=12, len=0.85),
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        )
    )
    _apply(fig, f"Correlação dos log-retornos ({len(matrix)} pregões alinhados por data)", height)
    fig.update_layout(hovermode="closest")
    return fig


def volatility_chart(
    dates, realized: np.ndarray, garch_vol: np.ndarray | None, ewma_vol: np.ndarray | None,
    height: int = 400,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=realized, mode="lines", name="Realizada (21d, anualizada)",
                             line=dict(color=PALETTE["accent"], width=1.6)))
    if garch_vol is not None:
        fig.add_trace(go.Scatter(x=dates, y=garch_vol, mode="lines", name="Condicional GARCH(1,1)",
                                 line=dict(color=PALETTE["primary"], width=1.8)))
    if ewma_vol is not None:
        fig.add_trace(go.Scatter(x=dates, y=ewma_vol, mode="lines", name="EWMA (λ=0,94)",
                                 line=dict(color=PALETTE["warn"], width=1.4, dash="dot")))
    _apply(fig, "Volatilidade anualizada — realizada vs. modelos condicionais", height)
    fig.update_yaxes(tickformat=".0%")
    return fig


def montecarlo_distribution(
    sims: np.ndarray, var_value: float, es_value: float, confidence_level: float,
    height: int = 400,
) -> go.Figure:
    tail = sims[sims <= -var_value]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sims, nbinsx=90, name="Cenários simulados",
                               marker_color=PALETTE["accent"], opacity=0.55))
    if tail.size:
        fig.add_trace(go.Histogram(x=tail, nbinsx=25, name=f"Cauda ({(1-confidence_level):.1%})",
                                   marker_color=PALETTE["risk"], opacity=0.85))
    fig.add_vline(x=-var_value, line=dict(color=METHOD_COLORS["montecarlo"], width=2, dash="dash"))
    fig.add_vline(x=-es_value, line=dict(color=METHOD_COLORS["expected_shortfall"], width=2, dash="dot"))
    fig.add_annotation(x=-var_value, yref="paper", y=0.97, showarrow=False, xanchor="right", xshift=-6,
                       text=f"<b>VaR</b> {var_value:.2%}", font=dict(color=METHOD_COLORS["montecarlo"], size=11),
                       bgcolor=annotation_bg(), borderpad=3)
    fig.add_annotation(x=-es_value, yref="paper", y=0.84, showarrow=False, xanchor="right", xshift=-6,
                       text=f"<b>ES</b> {es_value:.2%}", font=dict(color=METHOD_COLORS["expected_shortfall"], size=11),
                       bgcolor=annotation_bg(), borderpad=3)
    _apply(fig, f"Monte Carlo — {len(sims):,} cenários simulados".replace(",", "."), height)
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(title_text="Retorno simulado no horizonte", tickformat=".1%")
    return fig


def convergence_chart(frame: pd.DataFrame, reference: float, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["n_sims"], y=frame["var"], mode="lines+markers", name="VaR Monte Carlo",
            line=dict(color=METHOD_COLORS["montecarlo"], width=2),
            error_y=dict(type="data", array=1.96 * frame["erro_padrao"], visible=True,
                         color=PALETTE["muted"], thickness=1),
        )
    )
    fig.add_hline(y=reference, line=dict(color=METHOD_COLORS["historical"], width=1.6, dash="dash"),
                  annotation_text="VaR histórico (referência)", annotation_position="top left",
                  annotation_font=dict(color=METHOD_COLORS["historical"], size=11))
    _apply(fig, "Convergência do estimador de Monte Carlo", height)
    fig.update_xaxes(type="log", title_text="Número de simulações (escala log)")
    fig.update_yaxes(tickformat=".2%", title_text="VaR estimado")
    fig.update_layout(hovermode="closest")
    return fig


def drawdown_chart(dates, prices: np.ndarray, dd: np.ndarray, height: int = 360) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Preço",
                             line=dict(color=PALETTE["accent"], width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=dd, mode="lines", name="Drawdown", fill="tozeroy",
                             line=dict(color=PALETTE["risk"], width=1.2),
                             fillcolor="rgba(239,68,68,.25)"), row=2, col=1)
    _apply(fig, "Preço e drawdown (perda acumulada desde o topo)", height)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return fig


def benchmark_bar(frame: pd.DataFrame, height: int = 340) -> go.Figure:
    colors = [PALETTE["risk"] if "laço" in impl.lower() or "loop" in impl.lower()
              else PALETTE["prov"] for impl in frame["implementação"]]
    fig = go.Figure(
        go.Bar(
            x=frame["tempo_ms"], y=frame["implementação"], orientation="h",
            marker_color=colors, text=[f"{t:,.1f} ms" for t in frame["tempo_ms"]],
            textposition="outside", cliponaxis=False,
        )
    )
    _apply(fig, "Tempo de execução do Monte Carlo por implementação", height)
    fig.update_xaxes(type="log", title_text="milissegundos (escala log)")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def qq_plot(returns: np.ndarray, height: int = 360) -> go.Figure:
    """Q-Q normal — mostra visualmente por que o VaR paramétrico subestima a cauda."""
    z = np.sort(stats.zscore(returns))
    theo = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo, y=z, mode="markers", name="Quantis empíricos",
                             marker=dict(color=PALETTE["accent"], size=4, opacity=0.7)))
    lim = [float(min(theo.min(), z.min())), float(max(theo.max(), z.max()))]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Normal teórica",
                             line=dict(color=PALETTE["risk"], width=1.5, dash="dash")))
    _apply(fig, "Q-Q plot vs. Normal", height)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Quantis teóricos")
    fig.update_yaxes(title_text="Quantis observados (z)")
    return fig


def method_comparison_bar(results: list[RiskResult], height: int = 340) -> go.Figure:
    ordered = sorted(results, key=lambda r: r.value)
    fig = go.Figure(
        go.Bar(
            x=[r.value for r in ordered],
            y=[METHOD_LABELS.get(r.method, r.method) for r in ordered],
            orientation="h",
            marker_color=[METHOD_COLORS.get(r.method, PALETTE["primary"]) for r in ordered],
            text=[f"{r.value:.2%}" for r in ordered], textposition="outside", cliponaxis=False,
        )
    )
    _apply(fig, "Comparação entre estimadores", height)
    fig.update_xaxes(tickformat=".1%")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def confidence_curve(curve: pd.DataFrame, height: int = 360) -> go.Figure:
    """VaR e ES em função do nível de confiança."""
    fig = go.Figure()
    for col, method in (("var", "historical"), ("es", "expected_shortfall")):
        fig.add_trace(
            go.Scatter(
                x=curve["confianca"], y=curve[col], mode="lines+markers",
                name=METHOD_LABELS.get(method, col),
                line=dict(color=METHOD_COLORS[method], width=2),
            )
        )
    _apply(fig, "Sensibilidade ao nível de confiança", height)
    fig.update_xaxes(tickformat=".1%", title_text="Nível de confiança")
    fig.update_yaxes(tickformat=".1%", title_text="Perda potencial")
    return fig


def template_name() -> str:
    return plotly_template()
