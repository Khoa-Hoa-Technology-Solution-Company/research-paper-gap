"""Print-ready version of the corpus saturation figure (Fig. 2 of the AMI 2027 paper).

Reads the canonical saturation series written by scratch/render_canonical_saturation.py
(outputs/corpus_saturation_report_canonical.json, same data as the original figure) and draws a
vector PDF sized for the LNCS text width. Series are distinguished by line style and grey level,
not by colour alone, and a legend is included.

Usage (from ESV-Gap/):
  python experiments/plot_saturation_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = Path("outputs/corpus_saturation_report_canonical.json")
OUT = Path("paper_v2/figures/corpus_saturation_curve.pdf")

data = json.load(open(SRC, encoding="utf-8"))
series, summary = data["series"], data["summary"]
x = np.array([s["paper_index"] for s in series])
cum = np.array([s["cumulative_entities"] for s in series])
new = np.array([s["new_entities"] for s in series])
k, beta = summary["heaps_law"]["K"], summary["heaps_law"]["beta"]
w = 7
roll = np.convolve(new, np.ones(w) / w, mode="valid")

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "font.serif": ["STIXGeneral"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42})  # embedded TrueType, no Type 3 fonts
fig, ax1 = plt.subplots(figsize=(4.8, 2.3))
ax2 = ax1.twinx()
ax2.bar(x, new, width=0.8, color="0.80", label="New entities per paper (right)")
ax2.plot(x[w - 1:], roll, color="0.35", linestyle="--", linewidth=1.0, label=f"{w}-paper moving average (right)")
ax1.plot(x, cum, color="black", linewidth=1.6, label="Cumulative canonical entities $V(n)$ (left)")
ax1.plot(x, k * x ** beta, color="#1f77b4", linestyle=":", linewidth=1.4,
         label=f"Heaps' fit $V(n)={k:.2f}\\,n^{{{beta:.3f}}}$ (left)")
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.set_xlabel("Number of papers ingested $n$")
ax1.set_ylabel("Cumulative entities $V(n)$")
ax2.set_ylabel("New entities per paper")
ax1.set_xlim(0, len(x) + 1)
ax1.set_ylim(0, cum.max() * 1.05)
ax2.set_ylim(0, new.max() * 1.6)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, ncol=1)
fig.tight_layout(pad=0.3)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print("->", OUT)
