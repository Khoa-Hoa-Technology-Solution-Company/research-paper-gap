"""Generate the tables of the supplementary material (tables/supp_*.tex) from gapclose/outputs*/results."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "gapclose" / "outputs" / "results"
RESC = HERE.parent / "gapclose" / "outputs_climate" / "results"
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)

P = ["0.0", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]
ALPHA = 0.1


def load(path):
    return json.load(open(path, encoding="utf-8"))


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")


def bold_if(v, s=None):
    s = s or f"{v:.3f}"
    return r"\textbf{" + s + "}" if v > ALPHA + 0.005 else s


def tabular(spec, header, rows, sep="3pt"):
    return (rf"\setlength{{\tabcolsep}}{{{sep}}}" "\n" rf"\begin{{tabular}}{{{spec}}}" "\n" r"\toprule" "\n"
            + header + "\n" r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")


def tex(t):
    for a, b in (("\\", ""), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("µ", r"$\mu$"), ("−", "-"), ("–", "-"), ("—", ", "), ("’", "'"), ("“", "``"), ("”", "''"),
                 ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("±", r"$\pm$"), ("×", r"$\times$")):
        t = t.replace(a, b)
    return "".join(ch if ord(ch) < 128 else " " for ch in t)


def short(t, n):
    t = tex(t.strip())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + r" \ldots{}"


# ------------------------------------------------------------- S1: cross-validation grid (training split only)
cv = load(RES / "cv_dev_large.json")
rows = []
for a in ("0.0", "0.25", "0.5", "1.0"):
    cells = [f"{cv['esv_scope'][f'a={a},k={k}'][0]:.3f} $\\pm$ {cv['esv_scope'][f'a={a},k={k}'][1]:.3f}" for k in (3, 5, 10)]
    rows.append(f"ESV-Scope, $a={float(a):g}$ & " + " & ".join(cells) + r" \\")
cells = [f"{cv['esv_learned'][f'k={k}'][0]:.3f} $\\pm$ {cv['esv_learned'][f'k={k}'][1]:.3f}" for k in (3, 5, 10)]
rows.append(r"ESV-Learned & " + " & ".join(cells) + r" \\")
rows.append(r"\midrule")
bl = {"CoMention": "CoMention", "BM25-thr": "BM25", "Dense-thr": "Dense (MiniLM)", "NLI-verify": "NLI-verify ($k=5$)"}
for k, lab in bl.items():
    m, s = cv["baselines"][k]
    rows.append(f"{lab} & \\multicolumn{{3}}{{c}}{{{m:.3f} $\\pm$ {s:.3f}}}" + r" \\")
write("supp_cv.tex", tabular("lccc", r"Variant & $k=3$ & $k=5$ & $k=10$ \\", rows))

# ------------------------------------------------------------- S2: the two passes over the test split
r1 = load(RES / "main_run1_small_untuned.json")["methods"]
main = load(RES / "main_large.json")
r2 = main["methods"]
lab = {"CoMention": "CoMention", "BM25-thr": "BM25", "Dense-thr": "Dense (MiniLM)", "NLI-verify": "NLI-verify",
       "ESV-Scope": "ESV (run 1: Scope, $a=0.5$, $k=5$; run 2: Learned, $k=3$)"}
rows = []
for k, l in lab.items():
    rows.append(f"{l} & {r1[k]['macro_f1']:.3f} & {r1[k]['false_novelty_rate']:.3f} & "
                f"{r2[k]['macro_f1']:.3f} & {r2[k]['false_novelty_rate']:.3f}" + r" \\")
write("supp_runs.tex", tabular(r">{\raggedright\arraybackslash}p{4.1cm}cccc",
                               r" & \multicolumn{2}{c}{Run 1 (small)} & \multicolumn{2}{c}{Run 2 (large)} \\"
                               "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n" r"Method & F1 & FN & F1 & FN \\", rows))

# ------------------------------------------------------------- S3: paired comparison and full uncertified metrics
rows = []
for k, l in [("CoMention", "CoMention"), ("BM25-thr", "BM25"), ("Dense-thr", "Dense (MiniLM)"), ("NLI-verify", "NLI-verify")]:
    d = main["paired_vs_esv"][k]
    rows.append(f"{l} & {d['delta_macro_f1']:+.3f} & [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] & "
                f"{d['mcnemar_b_c'][0]} / {d['mcnemar_b_c'][1]}" + r" \\")
write("supp_paired.tex", tabular("lccc", r"Baseline & $\Delta$F1 & 95\% CI & McNemar $b$ / $c$ \\", rows))

rows = []
for k, l in [("CoMention", "CoMention"), ("BM25-thr", "BM25"), ("Dense-thr", "Dense (MiniLM)"),
             ("NLI-verify", "NLI-verify"), ("ESV-Scope", "ESV-Learned")]:
    m = r2[k]
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
    rows.append(f"{l} & {m['accuracy']:.3f} & [{m['false_novelty_ci'][0]:.2f}, {m['false_novelty_ci'][1]:.2f}] & "
                f"[{m['false_closure_ci'][0]:.2f}, {m['false_closure_ci'][1]:.2f}] & {fmt(m['witness_doc_precision'])} & "
                f"{fmt(m['grounded_closure_recall'])} & {fmt(m['false_closures_on_hard_negative'])}" + r" \\")
write("supp_unc.tex", tabular("lcccccc", r"Method & Acc. & FN 95\% CI & FC 95\% CI & Wit.\ doc & Gr.\ rec. & FC hard \\",
                              rows, sep="2.5pt"))

# ------------------------------------------------------------- S4: false novelty per score, rule and deletion rate
cert = load(RES / "certify_large_pooled_mcar.json")
G = cert["grid"]
order = ["CoMention", "BM25", "Dense", "BGE-dense", "BGE-rerank", "LLM-judge", "NLI-verify", "ESV-Scope",
         "CoMention+OA", "Dense+OA", "BGE-rerank+OA", "ESV-Scope+OA"]
order = [m for m in order if f"{m}|transfer" in G["0.0"]]
names = {"CoMention": "CoMention", "BM25": "BM25", "Dense": "Dense", "BGE-dense": "BGE dense", "BGE-rerank": "BGE rerank",
         "LLM-judge": "LLM judge", "NLI-verify": "NLI-verify", "ESV-Scope": "ESV-Scope", "CoMention+OA": "CoMention +OA",
         "Dense+OA": "Dense +OA", "BGE-rerank+OA": "BGE rerank +OA", "ESV-Scope+OA": "ESV-Scope +OA"}
head = "Score & " + " & ".join(f"${float(p):g}$" for p in P) + r" \\"
for rule in ("f1-frozen", "conformal", "in-situ", "transfer"):
    rows = []
    for m in order:
        if m == "CoMention+OA":
            rows.append(r"\midrule")
        rows.append(names[m] + " & " + " & ".join(bold_if(G[p][f"{m}|{rule}"]["fnov"]) for p in P) + r" \\")
    write(f"supp_fn_{rule}.tex", tabular("l" + "c" * len(P), head, rows, sep="2.5pt"))

# 95th percentile of false novelty over splits for the certified rules (ESV-Scope)
rows = []
for rule, l in (("in-situ", "In-situ"), ("transfer", "Transfer")):
    rows.append(l + " & " + " & ".join(f"{G[p][f'ESV-Scope|{rule}']['fnov']:.3f} / {G[p][f'ESV-Scope|{rule}']['fnov_q95']:.3f}"
                                         for p in ("0.0", "0.3", "0.5", "1.0")) + r" \\")
write("supp_q95.tex", tabular("lcccc", "Rule & " + " & ".join(f"$p={float(p):g}$" for p in ("0.0", "0.3", "0.5", "1.0")) + r" \\", rows))

# ------------------------------------------------------------- S5: power with standard deviation, coverage estimate
rows = []
for m in order:
    if m == "CoMention+OA":
        rows.append(r"\midrule")
    cells = [f"{G[p][f'{m}|transfer']['power']:.3f}" for p in P] + [f"{G['0.5'][f'{m}|worst-case']['power']:.3f}"]
    rows.append(names[m] + " & " + " & ".join(cells) + r" \\")
write("supp_power.tex", tabular("l" + "c" * (len(P) + 1),
                                "Score & " + " & ".join(f"${float(p):g}$" for p in P) + r" & $\hat p{=}1$ \\", rows, sep="2.5pt"))

rows = [f"${float(p):g}$ & {G[p]['ESV-Scope|transfer']['true_rate']:.3f} & {G[p]['ESV-Scope|transfer']['p_hat']:.3f} & "
        f"{G[p]['ESV-Scope|transfer']['power_sd']:.3f} & {100 * G[p]['ESV-Scope+OA|transfer']['ext_calls']:.0f}\\%" + r" \\" for p in P]
write("supp_phat.tex", tabular("ccccc", r"$p$ & realized rate & mean $\hat p$ & SD of power & ext.\ queries \\", rows))

# ------------------------------------------------------------- S6: robustness per score and setting
settings = [("pooled", "mcar"), ("pooled", "targeted"), ("holdout", "mcar"), ("holdout", "targeted")]
cs = {s: load(RES / f"certify_large_{s[0]}_{s[1]}.json")["grid"] for s in settings}
rows = []
for m in order:
    if m == "CoMention+OA":
        rows.append(r"\midrule")
    cells = []
    for s in settings:
        for rule in ("in-situ", "transfer"):
            cells.append(bold_if(max(cs[s][p][f"{m}|{rule}"]["fnov"] for p in P)))
    rows.append(names[m] + " & " + " & ".join(cells) + r" \\")
head = (r" & \multicolumn{2}{c}{Res., random} & \multicolumn{2}{c}{Res., targeted} & \multicolumn{2}{c}{T$\to$T, random}"
        r" & \multicolumn{2}{c}{T$\to$T, targeted} \\" "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
        "\n" r"Score & in-situ & transf. & in-situ & transf. & in-situ & transf. & in-situ & transf. \\")
write("supp_robust.tex", tabular("lcccccccc", head, rows, sep="2pt"))

# ------------------------------------------------------------- S7: Climate-FEVER per rate
cm = load(RESC / "certify_large_pooled_mcar.json")
Gc = cm["grid"]
corder = ["CoMention", "BM25", "Dense", "NLI-verify", "ESV-Scope"]
blocks = []
for rule, l in (("f1-frozen", "F1-optimal threshold"), ("conformal", "Split conformal"), ("in-situ", "In-situ (Thm.~1)"),
                ("transfer", r"Transfer, $\hat p$ (Thm.~2)")):
    blocks.append(rf"\multicolumn{{{len(P) + 1}}}{{l}}{{\emph{{False novelty, {l}}}}} \\")
    for m in corder:
        blocks.append(m + " & " + " & ".join(bold_if(Gc[p][f"{m}|{rule}"]["fnov"]) for p in P) + r" \\")
blocks.append(rf"\multicolumn{{{len(P) + 1}}}{{l}}{{\emph{{Power, transfer calibration}}}} \\")
for m in corder:
    blocks.append(m + " & " + " & ".join(f"{Gc[p][f'{m}|transfer']['power']:.3f}" for p in P) + r" \\")
write("supp_climate.tex", tabular("l" + "c" * len(P), header="Score & " + " & ".join(f"${float(p):g}$" for p in P) + r" \\",
                                  rows=blocks, sep="2.5pt"))

# ------------------------------------------------------------- S8: retrospective validation, all cutoffs
rt = load(RES / "retro.json")
rorder = [("Dense", "Dense"), ("BGE-rerank", "BGE rerank"), ("LLM-judge", "LLM judge"), ("ESV-Scope", "ESV-Scope"),
          ("BGE-rerank+OA", "BGE rerank +OA"), ("ESV-Scope+OA", "ESV-Scope +OA")]
rows = []
for y in sorted(rt["grid"]):
    g = rt["grid"][y]
    rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{Cutoff {y}: {100 * g['visible_share']:.0f}\% of documents visible, "
                rf"{g['n_closed']} closed, {g['n_filled']} later filled, {g['n_nei']} \textup{{\textsc{{nei}}}}}}}} \\")
    for m, l in rorder:
        if f"{m}|in-situ" not in g["methods"]:
            continue
        f, c = g["methods"][f"{m}|f1-frozen"], g["methods"][f"{m}|in-situ"]
        rows.append(f"{l} & {bold_if(f['fn'])} & {bold_if(c['fn'])} & {c['pow_filled']:.3f} & {c['pow_nei']:.3f} & {c['prec']:.3f}" + r" \\")
write("supp_retro.tex", tabular("lccccc", r"Score & F1-opt.\ FN & In-situ FN & Filled & \textsc{nei} & Prec. \\", rows))

# ------------------------------------------------------------- S9: more gap dossiers
dos = load(RES / "gap_dossiers.json")["2010"]
pick = [x for x in dos if x["nearest_local"] and x["filled_by"]["doi"] and len(x["question"]) < 110
        and not x["filled_by"]["rationale"][0].startswith("Highlights")]
pick = sorted(pick, key=lambda x: -x["nearest_local"]["score"])[:10]
rows = []
for x in pick:
    nl, fb = x["nearest_local"], x["filled_by"]
    rows.append(tex(x["question"]) + " & "
                + f"``{short(nl['sentence'], 120)}'' ({nl['year']}, {nl['score']:.3f}) & "
                + f"``{short(fb['rationale'][0], 160)}'' ({tex(fb['label'].lower())}, {fb['year']}, doi:{tex(fb['doi'])}) & "
                + f"{x['lag_years']}" + r" \\")
lags = sorted(x["lag_years"] for x in dos)
write("supp_dossier.tex",
      r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{4.7cm}>{\raggedright\arraybackslash}p{6.6cm}c}" "\n"
      r"\toprule" "\n" r"Certified open question & Closest local evidence (year, score) & Later answering paper & Lag \\" "\n"
      r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")

# ------------------------------------------------------------- macros for the supplementary prose
mac = {
    "SuppNdos": str(len(dos)),
    "SuppLagMin": str(lags[0]), "SuppLagMax": str(lags[-1]),
    "SuppTauDos": f"{pick[0]['tau']:.3f}" if pick else "n/a",
    "SuppClimN": f"{cm['n']:,}", "SuppClimCal": str(cm["n_calibration_pool"]), "SuppClimEval": str(cm["n_eval"]),
    "SuppR": str(cert["R"]), "SuppM": str(cert["m"]), "SuppDelta": f"{cert['delta']:g}",
    "SuppCVsel": f"{cv['selected']['cv_macro_f1']:.3f}",
}
write("supp_macros.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print("supplementary tables written:", sorted(p.name for p in OUT.glob("supp_*.tex")))
print(mac)
