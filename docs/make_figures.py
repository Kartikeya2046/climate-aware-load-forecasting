"""Regenerate the README figures (light + dark) from the real models, evidence and recorded data.

    python docs/make_figures.py        # from the repository root; writes docs/assets/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from core import climate  # noqa: E402
from core.data import CITIES, repository_record  # noqa: E402
from core.schema import PredictionRequest  # noqa: E402
from core.service import analog_ranking, run_prediction  # noqa: E402
from core.validation import process, propose_mapping, read_csv, suggest_unit  # noqa: E402

OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"dayton": "Dayton", "pittsburgh": "Pittsburgh", "atlanta": "Atlanta", "birmingham": "Birmingham",
         "las_vegas": "Las Vegas", "phoenix": "Phoenix"}
THEMES = {  # reference palette (dataviz skill), light and dark steps
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e7e6e2", muted="#c9c7c0", blue="#2a78d6",
                  orange="#eb6834", red="#e34948", mid="#f0efec",
                  seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835", muted="#5a5955", blue="#3987e5",
                 orange="#d95926", red="#e66767", mid="#383835",
                 seq=["#0d366b", "#104281", "#184f95", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"]),
}


def style(t):
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5, "figure.facecolor": t["bg"],
                         "axes.facecolor": t["bg"], "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink2"],
                         "xtick.color": t["ink2"], "ytick.color": t["ink2"], "text.color": t["ink"], "axes.grid": True, "axes.axisbelow": True,
                         "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlelocation": "left"})


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=170, bbox_inches="tight", pad_inches=0.25, facecolor=fig.get_facecolor())
    plt.close(fig)


def holdout(city, start, hours=168):
    s = pd.Timestamp(start)
    e = s + pd.Timedelta(hours=hours - 1)
    w = repository_record(city).loc[s - pd.Timedelta(hours=400):e + pd.Timedelta(hours=10)]
    r = run_prediction(PredictionRequest(city, s, e, "csv", w, prediction_mode="one_step", quality={"rows_hourly": len(w)}))
    assert r.status == "ok", r.message
    return r


def sample_frame(path):
    ing = read_csv(path.read_bytes(), path.name)
    m = {c: v["variable"] for c, v in propose_mapping(list(ing.raw.columns)).items()}
    units = {v: suggest_unit(v, ing.raw[c])[0] for c, v in m.items() if v and v != "timestamp"}
    return process(ing.raw, m, units, "", next(c for c, v in m.items() if v == "timestamp"), "local_standard").frame


# ------------------------------------------------------------------------------------- data (computed once)
print("predicting held-out weeks…")
WEEKS = {"birmingham": holdout("birmingham", "2023-07-10"), "phoenix": holdout("phoenix", "2023-07-10"),
         "pittsburgh": holdout("pittsburgh", "2015-07-06")}
EV = json.loads((ROOT / "dashboard" / "evidence" / "evidence.json").read_text())
print("climate matrix…")
MATRIX = climate.city_matrix()
print("analog rankings…")
SAMPLES = {"Kirk land": "kirkland_test_weather.csv", "Maple Falls": "maple_falls_test_weather_load.csv",
           "Cedar Ridge": "cedar_ridge_test_weather.csv"}
ANALOG = {n: analog_ranking(sample_frame(ROOT / "dashboard" / "samples" / f)) for n, f in SAMPLES.items()}


def fig_banner(t):
    p = WEEKS["phoenix"].predictions
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.fill_between(p.index, p.actual_mw, p.actual_mw.min() * 0.92, color=t["blue"], alpha=0.10, lw=0)
    ax.plot(p.index, p.actual_mw, color=t["blue"], lw=2, alpha=0.55)
    ax.set_ylim(p.actual_mw.min() * 0.92, p.actual_mw.max() * 2.0)
    ax.axis("off")
    ax.text(0.0, 0.93, "Climate-Aware Load Forecasting", transform=ax.transAxes, fontsize=27, weight="bold", va="top", color=t["ink"])
    ax.text(0.0, 0.58, "Hourly electricity demand from weather  ·  two-branch TCN + LSTM  ·  model selection by climate similarity",
            transform=ax.transAxes, fontsize=12.5, va="top", color=t["ink2"])
    return fig


def fig_transfer(t):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    titles = {"birmingham": "Birmingham · TCN trained for Atlanta", "phoenix": "Phoenix · TCN trained for Las Vegas",
              "pittsburgh": "Pittsburgh · LSTM trained on Dayton"}
    for ax, (c, r) in zip(axes, WEEKS.items()):
        p = r.predictions
        ax.plot(p.index, p.actual_mw, color=t["ink2"], lw=2, label="Observed load")
        ax.plot(p.index, p.pred_mw, color=t["blue"], lw=2, ls=(0, (4, 1.5)), label="Predicted (1 h ahead)")
        ax.set_title(titles[c], fontsize=11, pad=20)
        ax.text(0.0, 1.02, f"MAPE {r.summary['observed_MAPE_percent']:.2f} % on this held-out week", transform=ax.transAxes,
                fontsize=9.5, color=t["ink2"], va="bottom")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a"))
        ax.set_ylabel("MW" if ax is axes[0] else "")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, fontsize=10)
    fig.tight_layout()
    return fig


def fig_two_branch(t):
    p = WEEKS["phoenix"].predictions
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.12})
    a.plot(p.index, p.baseline_mw, color=t["orange"], lw=2, label="Branch 1 · calendar baseline (MLP)")
    a.plot(p.index, p.actual_mw, color=t["ink2"], lw=2, label="Observed load")
    a.plot(p.index, p.pred_mw, color=t["blue"], lw=2, ls=(0, (4, 1.5)), label="Final prediction = baseline × e^residual")
    a.set_ylabel("MW")
    a.legend(loc="upper left", ncol=3, fontsize=9.5, bbox_to_anchor=(0, 1.13))
    a.set_title("Phoenix, held-out week of 10 July 2023", pad=30)
    adj = p.adjustment_mw
    b.bar(p.index, adj, width=1 / 24 * 0.85, color=np.where(adj >= 0, t["red"], t["blue"]), lw=0)
    b.axhline(0, color=t["ink2"], lw=0.8)
    b.set_ylabel("MW")
    b.text(0.0, 1.03, "Branch 2 · weather-residual TCN adjustment (red raises load, blue lowers it)", transform=b.transAxes,
           fontsize=9.5, color=t["ink2"])
    b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d"))
    return fig


def fig_accuracy(t):
    rows = [("tcn_atlanta", "atlanta", "TCN Atlanta → Atlanta"), ("tcn_atlanta", "birmingham", "TCN Atlanta → Birmingham*"),
            ("tcn_las_vegas", "las_vegas", "TCN Las Vegas → Las Vegas"), ("tcn_las_vegas", "phoenix", "TCN Las Vegas → Phoenix*"),
            ("lstm_dayton", "dayton", "LSTM Dayton → Dayton"), ("lstm_dayton", "pittsburgh", "LSTM Dayton → Pittsburgh*")]
    mod = [EV["models"][m][c]["one_step"]["MAPE_percent"] for m, c, _ in rows]
    nai = [EV["models"][m][c]["one_step"]["naive"]["MAPE_percent"] for m, c, _ in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(y + 0.19, nai, height=0.34, color=t["muted"], label="Naive benchmark (same hour yesterday)")
    ax.barh(y - 0.19, mod, height=0.34, color=t["blue"], label="Model, one hour ahead")
    for yy, a, b in zip(y, mod, nai):
        ax.text(a + 0.12, yy - 0.19, f"{a:.2f} %", va="center", fontsize=9.5, color=t["ink"], weight="semibold")
        ax.text(b + 0.12, yy + 0.19, f"{b:.2f} %", va="center", fontsize=9, color=t["ink2"])
    ax.set_yticks(y, [r[2] for r in rows])
    ax.tick_params(axis="y", length=0, labelcolor=t["ink"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MAPE on held-out data (%)  ·  lower is better")
    ax.set_xlim(0, max(nai) * 1.15)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9.5)
    ax.set_title("Held-out accuracy, one hour ahead", pad=28)
    ax.text(0.0, -0.2, "* transfer city: the weather model was trained on the source city only (the two-branch TCN pairs it with the "
            "target city's own calendar baseline)", transform=ax.transAxes, fontsize=9, color=t["ink2"])
    return fig


def fig_matrix(t):
    ids = list(CITIES)
    m = np.full((6, 6), np.nan)
    for k, v in MATRIX.items():
        a, b = k.split("|")
        m[ids.index(a), ids.index(b)] = m[ids.index(b), ids.index(a)] = v["overall"]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", t["seq"])
    cmap.set_bad(t["mid"])
    im = ax.imshow(m, cmap=cmap, vmin=30, vmax=100)
    for i in range(6):
        for j in range(6):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=t["ink2"])
                continue
            r, g, b_, _ = cmap((m[i, j] - 30) / 70)
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=10.5,
                    color="#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b_ > 0.5 else "#ffffff")
    names = [SHORT[c] for c in ids]
    ax.set_xticks(range(6), names, rotation=35, ha="right")
    ax.set_yticks(range(6), names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelcolor=t["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["ink2"], length=0)
    ax.set_title("Climate similarity between trained cities (0–100)")
    return fig


def fig_analog(t):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharex=True)
    for ax, (name, rank) in zip(axes, ANALOG.items()):
        rank = rank[::-1]
        vals = [r["similarity"] for r in rank]
        cols = [t["muted"]] * 5 + [t["blue"]]
        y = np.arange(6)
        ax.barh(y, vals, color=cols, height=0.62)
        for yy, v in zip(y, vals):
            ax.text(v + 1.2, yy, f"{v:.0f}", va="center", fontsize=9.5, color=t["ink"] if yy == 5 else t["ink2"],
                    weight="semibold" if yy == 5 else "normal")
        ax.set_yticks(y, [SHORT[r["city"]] for r in rank])
        ax.tick_params(axis="y", length=0, labelcolor=t["ink"])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 100)
        ax.set_title(f"{name}  →  {SHORT[rank[-1]['city']]}", fontsize=11.5)
    axes[1].set_xlabel("Climate similarity of the uploaded weather to each trained city (0–100)")
    fig.text(0.0, -0.06, "Synthetic test cities from dashboard/samples (perturbed Phoenix, Pittsburgh and Birmingham weather). "
             "The closest city with an eligible model is used for the prediction.", fontsize=9, color=t["ink2"])
    fig.tight_layout()
    return fig


for theme, t in THEMES.items():
    style(t)
    for name, fn in [("banner", fig_banner), ("transfer", fig_transfer), ("two_branch", fig_two_branch),
                     ("accuracy", fig_accuracy), ("climate_matrix", fig_matrix), ("analog", fig_analog)]:
        save(fn(t), name, theme)
        print("wrote", name, theme)
