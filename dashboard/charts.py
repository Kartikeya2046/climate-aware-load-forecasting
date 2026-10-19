"""Plotly figure builders. Every figure is drawn only from values returned by the core service."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# validated categorical order (dataviz reference palette, light mode): never cycled, follows the entity
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
BAND = "rgba(42,120,214,0.16)"
SEQ = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]]
DIV = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e2"


def _style(fig: go.Figure, title: str, y: str | None = None, x: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=INK)), height=height, margin=dict(l=8, r=8, t=48, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1), hoverlabel=dict(font_size=12))
    fig.update_xaxes(title=x, gridcolor=GRID, zeroline=False, showline=True, linecolor=GRID)
    fig.update_yaxes(title=y, gridcolor=GRID, zeroline=False)
    return fig


def timeseries(df: pd.DataFrame, alternatives: dict[str, pd.DataFrame], names: dict[str, str], value: str, selected_name: str) -> go.Figure:
    """Predicted load (or energy) over time with the empirical 90 % band, observed load and computed alternatives."""
    fig = go.Figure()
    ylab = "MWh per interval" if value == "energy" else "MW (hourly average)"
    col = "energy_mwh" if value == "energy" and "energy_mwh" in df else "pred_mw"
    if "upper_mw" in df and df.upper_mw.notna().any() and col == "pred_mw":
        fig.add_trace(go.Scatter(x=df.index, y=df.upper_mw, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df.lower_mw, fill="tonexty", fillcolor=BAND, line=dict(width=0),
                                 name="90 % empirical interval", hovertemplate="%{y:,.0f}<extra>interval bound</extra>"))
    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=f"Predicted · {selected_name}", line=dict(color=BLUE, width=2),
                             hovertemplate="%{y:,.1f}<extra>predicted</extra>"))
    if "actual_mw" in df and df.actual_mw.notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df.actual_mw, name="Observed load", line=dict(color=ORANGE, width=2),
                                 hovertemplate="%{y:,.1f}<extra>observed</extra>"))
    for k, (mid, alt) in enumerate(alternatives.items()):
        fig.add_trace(go.Scatter(x=alt.index, y=alt.pred_mw, name=f"Alternative · {names.get(mid, mid)}",
                                 line=dict(color=AQUA, width=2, dash="dot"), hovertemplate="%{y:,.1f}<extra>alternative</extra>"))
    fig.update_layout(hovermode="x unified")
    return _style(fig, "Predicted electricity consumption", ylab, "Local standard time")


def cumulative(df: pd.DataFrame) -> go.Figure:
    c = df.energy_mwh.fillna(0).cumsum()
    fig = go.Figure(go.Scatter(x=df.index, y=c, line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor=BAND,
                               name="Cumulative predicted energy", hovertemplate="%{y:,.0f} MWh<extra></extra>"))
    observed = bool("actual_mw" in df and df.actual_mw.notna().all())
    if observed:
        fig.add_trace(go.Scatter(x=df.index, y=df.actual_mw.cumsum(), line=dict(color=ORANGE, width=2), name="Cumulative observed energy",
                                 hovertemplate="%{y:,.0f} MWh<extra></extra>"))
    fig.update_layout(hovermode="x unified", showlegend=observed)
    return _style(fig, "Cumulative predicted energy", "MWh", "Local standard time")


def hourly_profile(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(df.index.hour).pred_mw.mean()
    fig = go.Figure(go.Bar(x=g.index, y=g.values, marker=dict(color=BLUE, cornerradius=4), name="Average predicted load",
                           hovertemplate="hour %{x}: %{y:,.0f} MW<extra></extra>"))
    fig.update_layout(bargap=0.15)
    return _style(fig, "Average predicted load by hour of day", "MW", "Hour (local standard time)")


def hour_dow_heatmap(df: pd.DataFrame) -> go.Figure:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    t = df.assign(h=df.index.hour, d=df.index.dayofweek).pivot_table(index="d", columns="h", values="pred_mw", aggfunc="mean")
    t = t.reindex(index=range(7), columns=range(24))
    fig = go.Figure(go.Heatmap(z=t.values, x=list(range(24)), y=days, colorscale=SEQ, colorbar=dict(title="MW"),
                               hovertemplate="%{y} %{x}:00 · %{z:,.0f} MW<extra></extra>", xgap=2, ygap=2))
    return _style(fig, "Predicted load by hour and day of week (empty cells: not in the timeframe)", None, "Hour", height=300)


def weather_series(df: pd.DataFrame, var: str, label: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df.index, y=df[var], line=dict(color=BLUE, width=2), name=label,
                               hovertemplate="%{y:,.2f}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    return _style(fig, f"{label} used by the model", label, "Local standard time", height=300)


def scatter_vs(df: pd.DataFrame, var: str, label: str) -> go.Figure:
    d = df[[var, "pred_mw"]].dropna()
    fig = go.Figure(go.Scatter(x=d[var], y=d.pred_mw, mode="markers", name="Predicted hour",
                               marker=dict(color=BLUE, size=8, opacity=0.75, line=dict(color="#fcfcfb", width=1)),
                               hovertemplate=f"{label} %{{x:,.2f}} · %{{y:,.0f}} MW<extra></extra>", text=d.index.astype(str)))
    if len(d) > 2 and d[var].std() > 1e-9 and d.pred_mw.std() > 1e-9:
        r = float(np.corrcoef(d[var], d.pred_mw)[0, 1])
        fig.add_annotation(text=f"Pearson r = {r:+.2f}", xref="paper", yref="paper", x=0.01, y=0.98, showarrow=False, font=dict(color=MUTED))
    return _style(fig, f"{label} vs predicted load", "Predicted load (MW)", label, height=320)


def correlation(df: pd.DataFrame, cols: list[str], labels: dict[str, str], title: str) -> go.Figure | None:
    d = df[cols].dropna()
    cols = [c for c in cols if d[c].std() > 0]
    if len(cols) < 2:
        return None
    c = d[cols].corr()
    names = [labels.get(x, x) for x in cols]
    fig = go.Figure(go.Heatmap(z=c.values, x=names, y=names, zmin=-1, zmax=1, colorscale=DIV, colorbar=dict(title="r"),
                               text=np.round(c.values, 2), texttemplate="%{text}", hovertemplate="%{y} × %{x}: r = %{z:.2f}<extra></extra>",
                               xgap=2, ygap=2))
    return _style(fig, title, None, None, height=420)


def decomposition(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    a = go.Figure()
    a.add_trace(go.Scatter(x=df.index, y=df.baseline_mw, name="Calendar baseline", line=dict(color=ORANGE, width=2),
                           hovertemplate="%{y:,.0f} MW<extra>baseline</extra>"))
    a.add_trace(go.Scatter(x=df.index, y=df.pred_mw, name="Predicted (baseline + adjustment)", line=dict(color=BLUE, width=2),
                           hovertemplate="%{y:,.0f} MW<extra>predicted</extra>"))
    a.update_layout(hovermode="x unified")
    b = go.Figure(go.Bar(x=df.index, y=df.adjustment_mw, name="Weather-induced adjustment",
                         marker=dict(color=np.where(df.adjustment_mw >= 0, "#e34948", BLUE)),
                         hovertemplate="%{y:+,.0f} MW<extra>adjustment</extra>"))
    b.update_layout(bargap=0.1)
    return (_style(a, "Baseline vs predicted load", "MW", "Local standard time"),
            _style(b, "Weather-induced adjustment ΔE = predicted − baseline (red: above baseline, blue: below)", "MW", "Local standard time", height=300))


def actual_vs_predicted(df: pd.DataFrame) -> go.Figure:
    d = df[["actual_mw", "pred_mw"]].dropna()
    lo, hi = float(d.min().min()), float(d.max().max())
    fig = go.Figure([go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, width=1, dash="dash"), name="Perfect prediction",
                                hoverinfo="skip"),
                     go.Scatter(x=d.actual_mw, y=d.pred_mw, mode="markers", name="Hour", marker=dict(color=BLUE, size=8, opacity=0.7),
                                hovertemplate="observed %{x:,.0f} · predicted %{y:,.0f} MW<extra></extra>")])
    return _style(fig, "Observed vs predicted load", "Predicted (MW)", "Observed (MW)", height=360)


def fitness_bars(rows: list[dict]) -> go.Figure:
    rows = [r for r in rows if r["fitness"] is not None][::-1]
    fig = go.Figure(go.Bar(y=[r["name"] for r in rows], x=[r["fitness"] for r in rows], orientation="h",
                           marker=dict(color=[BLUE if r["selected"] else "#9ec5f4" for r in rows], cornerradius=4),
                           text=[f"{r['fitness']:.1f}{'  ✓ selected' if r['selected'] else ''}" for r in rows], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f} / 100<extra></extra>"))
    fig.update_xaxes(range=[0, 105])
    return _style(fig, "Model Fitness Score (0–100 rubric, not a probability)", None, "Score", height=110 + 50 * len(rows))


def components_chart(score) -> go.Figure:
    comps = [c for c in score.components][::-1]
    from core.scoring import LABELS
    fig = go.Figure(go.Bar(y=[LABELS[c.name] for c in comps], x=[c.score if c.score is not None else 0 for c in comps], orientation="h",
                           marker=dict(color=[BLUE if c.score is not None else GRID for c in comps], cornerradius=4),
                           text=[(f"{c.score:.0f} · weight {c.effective_weight:.0%}" if c.score is not None else "unavailable") for c in comps],
                           textposition="outside", hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
    fig.update_xaxes(range=[0, 125])
    return _style(fig, f"Component scores · {score.display_name}", None, "0–100", height=300)


def distributions(inp: pd.Series, ref: pd.Series, label: str, ref_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref, histnorm="probability density", name=ref_name, marker_color=ORANGE, opacity=0.55, nbinsx=50))
    fig.add_trace(go.Histogram(x=inp, histnorm="probability density", name="Input weather", marker_color=BLUE, opacity=0.65, nbinsx=30))
    fig.update_layout(barmode="overlay", bargap=0.05)
    return _style(fig, f"{label}: input vs model training distribution", "Density", label, height=320)


def pca_plot(points: pd.DataFrame, inp: pd.DataFrame | None, names: dict[str, str], evr) -> go.Figure:
    fig = go.Figure()
    greys = {c: g for c, g in zip(sorted(points.city.unique()), ["#b7b6b0", "#a3a29c", "#8f8e88", "#7b7a74", "#676660", "#53524d"])}
    for c in sorted(points.city.unique()):
        p = points[points.city == c]
        fig.add_trace(go.Scatter(x=p.PC1, y=p.PC2, mode="markers", name=names.get(c, c), marker=dict(size=4, opacity=0.35, color=greys[c]),
                                 hovertemplate=f"{names.get(c, c)}<extra></extra>"))
        fig.add_annotation(x=p.PC1.median(), y=p.PC2.median(), text=names.get(c, c).split(",")[0], showarrow=False,
                           font=dict(color=INK, size=11), bgcolor="rgba(252,252,251,0.8)")
    if inp is not None and len(inp):
        fig.add_trace(go.Scatter(x=inp.PC1, y=inp.PC2, mode="markers", name="Input weather",
                                 marker=dict(size=9, color=BLUE, line=dict(color="#fcfcfb", width=2)), hovertemplate="input<extra></extra>"))
    return _style(fig, f"Weather feature space, PCA of 8 standardised variables (PC1 {evr[0]:.0%}, PC2 {evr[1]:.0%} of variance)",
                  "PC2", "PC1", height=460)


def similarity_matrix(mat: dict, names: dict[str, str]) -> go.Figure:
    ids = list(names)
    z = np.full((len(ids), len(ids)), 100.0)
    for k, v in mat.items():
        a, b = k.split("|")
        i, j = ids.index(a), ids.index(b)
        z[i, j] = z[j, i] = v["overall"]
    lab = [names[i].split(",")[0] for i in ids]
    fig = go.Figure(go.Heatmap(z=z, x=lab, y=lab, colorscale=SEQ, zmin=0, zmax=100, text=np.round(z, 1), texttemplate="%{text}",
                               colorbar=dict(title="score"), hovertemplate="%{y} ↔ %{x}: %{z:.1f}/100<extra></extra>", xgap=2, ygap=2))
    return _style(fig, "Climate similarity between supported cities (project formula, full records)", None, None, height=420)


def daily_bars(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=daily.index, y=daily.energy_mwh, name="Predicted energy", marker=dict(color=BLUE, cornerradius=4),
                           hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f} MWh<extra>predicted</extra>"))
    if "actual_energy_mwh" in daily:
        fig.add_trace(go.Bar(x=daily.index, y=daily.actual_energy_mwh, name="Observed energy", marker=dict(color=ORANGE, cornerradius=4),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f} MWh<extra>observed</extra>"))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.08)
    return _style(fig, "Daily energy (sum of hourly MWh; partial days show fewer hours)", "MWh per day", "Day")
