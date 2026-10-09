"""Generate LaTeX tables, figure data and number macros from gapclose/outputs/results."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "gapclose" / "outputs" / "results"
OUT = HERE / "tables"
OUT.mkdir(exist_ok=True)

main = json.load(open(RES / "main_large.json", encoding="utf-8"))
def load_cert(protocol, deletion):
    f = RES / f"certify_large_{protocol}_{deletion}.json"
    if not f.exists() and (protocol, deletion) == ("pooled", "mcar"):
        f = RES / "certify_large.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


cert = load_cert("pooled", "mcar")
G = cert["grid"]
P = ["0.0", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]
ALPHA = cert["alpha"]


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------- Table: classification
names = {"CoMention": "CoMention", "BM25-thr": "BM25", "Dense-thr": "Dense (MiniLM)",
         "BGE-dense": "BGE-large dense", "BGE-rerank": "BGE reranker",
         "LLM-judge": "LLM judge", "NLI-verify": "NLI-verify", "ESV-Scope": "ESV-Learned"}
allm = dict(main["methods"])
for extra in ("bge_uncertified.json", "extra_uncertified.json"):
    if (RES / extra).exists():
        allm.update(json.load(open(RES / extra, encoding="utf-8")))
names = {k: v for k, v in names.items() if k in allm}
rows = []
best_f1 = max(allm[k]["macro_f1"] for k in names)
best_aurc = min(allm[k]["aurc"] for k in names)
for k, lab in names.items():
    m = allm[k]
    f1 = f"{m['macro_f1']:.3f}"
    if m["macro_f1"] == best_f1:
        f1 = r"\textbf{" + f1 + "}"
    ci = f"[{m['macro_f1_ci'][0]:.2f}, {m['macro_f1_ci'][1]:.2f}]"
    au = f"{m['aurc']:.3f}"
    if m["aurc"] == best_aurc:
        au = r"\textbf{" + au + "}"
    rows.append(f"{lab} & {f1} & {ci} & {m['false_novelty_rate']:.3f} & {m['false_closure_rate']:.3f} & "
                f"{f3(m['witness_sent_precision'])} & {au} \\\\")
(OUT / "cls.tex").write_text(
    "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lcccccc}\n\\toprule\n"
    "Method & F1 & 95\\% CI & FN & FC & Wit. & AURC \\\\\n\\midrule\n" + "\n".join(rows)
    + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- Table: false novelty (ESV-Scope)
rules = [("f1-frozen", "F1-optimal threshold"), ("conformal", "Split conformal (complete)"),
         ("in-situ", "In-situ conformal (Thm.~1)"), ("transfer", "Transfer, $\\hat p$ (Thm.~2)"),
         ("worst-case", "Transfer, $\\hat p=1$")]
cols = ["0.0", "0.1", "0.3", "0.5", "1.0"]
lines = []
for r, lab in rules:
    vals = []
    for p in cols:
        v = G[p][f"ESV-Scope|{r}"]["fnov"]
        s = f"{v:.3f}"
        vals.append(r"\textbf{" + s + "}" if v > ALPHA + 0.005 else s)
    lines.append(lab + " & " + " & ".join(vals) + " \\\\")
(OUT / "fn.tex").write_text(
    "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n"
    "Rule & " + " & ".join(f"$p={float(p):g}$" for p in cols) + " \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

# ---------------------------------------------------------------- Table: power
meths = [("CoMention", "CoMention"), ("Dense", "Dense"), ("BGE-dense", "BGE-large dense"),
         ("BGE-rerank", "BGE reranker"), ("LLM-judge", "LLM judge"), ("NLI-verify", "NLI-verify"), ("ESV-Scope", "ESV-Scope"),
         ("CoMention+OA", "CoMention +OA"), ("Dense+OA", "Dense +OA"), ("BGE-rerank+OA", "BGE reranker +OA"),
         ("ESV-Scope+OA", "ESV-Scope +OA")]
meths = [(m, lab) for m, lab in meths if f"{m}|transfer" in G["0.0"]]
pcols = ["0.0", "0.3", "0.5", "1.0"]
best = {p: max(G[p][f"{m}|transfer"]["power"] for m, _ in meths) for p in pcols}
bestw = max(G["0.5"][f"{m}|worst-case"]["power"] for m, _ in meths)
lines = []
for m, lab in meths:
    vals = []
    for p in pcols:
        v = G[p][f"{m}|transfer"]["power"]
        vals.append(r"\textbf{" + f"{v:.3f}" + "}" if abs(v - best[p]) < 1e-9 else f"{v:.3f}")
    w = G["0.5"][f"{m}|worst-case"]["power"]
    vals.append(r"\textbf{" + f"{w:.3f}" + "}" if abs(w - bestw) < 1e-9 else f"{w:.3f}")
    if m == "CoMention+OA":
        lines.append("\\midrule")
    lines.append(lab + " & " + " & ".join(vals) + " \\\\")
(OUT / "power.tex").write_text(
    "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
    " & \\multicolumn{4}{c}{Transfer, estimated $\\hat p$} & Worst case \\\\\n\\cmidrule(lr){2-5}\n"
    "Score & " + " & ".join(f"$p={float(p):g}$" for p in pcols) + " & any $p$ \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


# ---------------------------------------------------------------- Table: robustness of certified rules
settings = [("pooled", "mcar", r"Resampled, random deletion"), ("pooled", "targeted", r"Resampled, targeted deletion"),
            ("holdout", "mcar", r"Train$\to$test, random deletion"), ("holdout", "targeted", r"Train$\to$test, targeted deletion")]
rob, robmac = [], {}
for prot, dele, lab in settings:
    c = load_cert(prot, dele)
    if c is None:
        continue
    g = c["grid"]
    ms = sorted({k.split("|")[0] for k in g["0.0"]})

    def mx(rule, only=None):
        return max(g[p][f"{m}|{rule}"]["fnov"] for m in (only or ms) for p in P)

    vals = [mx("in-situ", ["ESV-Scope"]), mx("transfer", ["ESV-Scope"]), mx("in-situ"), mx("transfer"),
            max(g[p]["ESV-Scope|conformal"]["fnov"] for p in P)]
    robmac[f"{prot}{dele}"] = vals
    cells = [(r"\textbf{%.3f}" % v) if v > ALPHA + 0.005 else f"{v:.3f}" for v in vals]
    rob.append(lab + " & " + " & ".join(cells) + r" \\")
head = (r"\setlength{\tabcolsep}{2.5pt}" "\n" r"\begin{tabular}{lccccc}" "\n" r"\toprule" "\n"
        r" & \multicolumn{2}{c}{ESV-Scope} & \multicolumn{2}{c}{All scores} & Split conf. \\" "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}" "\n"
        r"Evaluation setting & in-situ & transfer & in-situ & transfer & ESV-Scope \\" "\n" r"\midrule" "\n")
tail = "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"
(OUT / "robust.tex").write_text(head + "\n".join(rob) + tail, encoding="utf-8")


# ---------------------------------------------------------------- Table: second domain (Climate-FEVER)
RESC = HERE.parent / "gapclose" / "outputs_climate" / "results"
clim, climmac = None, {}
if (RESC / "certify_large_pooled_mcar.json").exists():
    clim = json.load(open(RESC / "certify_large_pooled_mcar.json", encoding="utf-8"))
    climt = json.load(open(RESC / "certify_large_pooled_targeted.json", encoding="utf-8"))
    Gc, Gt = clim["grid"], climt["grid"]
    cm = [("BM25", "BM25"), ("Dense", "Dense"), ("CoMention", "CoMention"), ("NLI-verify", "NLI-verify"),
          ("ESV-Scope", "ESV-Scope")]
    lines = []
    for m, lab in cm:
        v = [Gc["0.0"][f"{m}|f1-frozen"]["fnov"], Gc["1.0"][f"{m}|f1-frozen"]["fnov"],
             max(Gc[p][f"{m}|conformal"]["fnov"] for p in P),
             max(max(Gc[p][f"{m}|{r}"]["fnov"], Gt[p][f"{m}|{r}"]["fnov"]) for p in P for r in ("in-situ", "transfer")),
             Gc["0.0"][f"{m}|transfer"]["power"], Gc["1.0"][f"{m}|transfer"]["power"]]
        cells = [(r"\textbf{%.3f}" % x) if i < 4 and x > ALPHA + 0.005 else f"{x:.3f}" for i, x in enumerate(v)]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    head = (r"\setlength{\tabcolsep}{2.5pt}" "\n" r"\begin{tabular}{lcccccc}" "\n" r"\toprule" "\n"
            r" & \multicolumn{2}{c}{F1-optimal FN} & Split conf. & Certified & \multicolumn{2}{c}{Power} \\" "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}" "\n"
            r"Score & $p=0$ & $p=1$ & max FN & max FN & $p=0$ & $p=1$ \\" "\n" r"\midrule" "\n")
    (OUT / "climate.tex").write_text(head + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n",
                                     encoding="utf-8")
    ms = [m for m, _ in cm]
    climmac = {
        "ClimN": f"{clim['n']:,}",
        "ClimFoneMax": f"{max(Gc['1.0'][f'{m}|f1-frozen']['fnov'] for m in ms):.2f}",
        "ClimFoneZeroMin": f"{min(Gc['0.0'][f'{m}|f1-frozen']['fnov'] for m in ms):.2f}",
        "ClimFoneZeroMax": f"{max(Gc['0.0'][f'{m}|f1-frozen']['fnov'] for m in ms):.2f}",
        "ClimConfMax": f"{max(Gc[p][f'{m}|conformal']['fnov'] for m in ms for p in P):.3f}",
        "ClimCertMax": f"{max(max(Gc[p][f'{m}|{r}']['fnov'], Gt[p][f'{m}|{r}']['fnov']) for m in ms for p in P for r in ('in-situ', 'transfer')):.3f}",
        "ClimDensePowZero": f"{Gc['0.0']['Dense|transfer']['power']:.3f}",
        "ClimDensePowOne": f"{Gc['1.0']['Dense|transfer']['power']:.3f}",
    }


# ---------------------------------------------------------------- Retrospective validation (natural incompleteness)
retro_f = RES / "retro.json"
retromac = {}
if retro_f.exists():
    rt = json.load(open(retro_f, encoding="utf-8"))
    RY = "2010" if "2010" in rt["grid"] else sorted(rt["grid"])[0]
    gy = rt["grid"][RY]["methods"]
    rlab = [("Dense", "Dense"), ("BGE-rerank", "BGE reranker"), ("LLM-judge", "LLM judge"), ("ESV-Scope", "ESV-Scope"),
            ("BGE-rerank+OA", "BGE reranker +OA"), ("ESV-Scope+OA", "ESV-Scope +OA")]
    rlab = [(m, l) for m, l in rlab if f"{m}|in-situ" in gy]
    lines = []
    for m, lab in rlab:
        f, c = gy[f"{m}|f1-frozen"], gy[f"{m}|in-situ"]
        cells = [f"{f['fn']:.3f}", f"{f['pow_filled']:.3f}", f"{c['fn']:.3f}", f"{c['pow_filled']:.3f}", f"{c['prec']:.3f}"]
        cells = [(r"\textbf{" + x + "}") if i in (0, 2) and float(x) > ALPHA + 0.005 else x for i, x in enumerate(cells)]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    head = (r"\setlength{\tabcolsep}{3pt}" "\n" r"\begin{tabular}{lccccc}" "\n" r"\toprule" "\n"
            r" & \multicolumn{2}{c}{F1-optimal} & \multicolumn{3}{c}{In-situ (Thm.~1)} \\" "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}" "\n"
            r"Score & FN & Filled & FN & Filled & Prec. \\" "\n" r"\midrule" "\n")
    (OUT / "retro.tex").write_text(head + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n", encoding="utf-8")
    best_m = max(rlab, key=lambda x: gy[f"{x[0]}|in-situ"]["pow_filled"])[0]
    allY = sorted(rt["grid"])
    retromac = {
        "RetroY": RY, "RetroYears": ", ".join(allY[:-1]) + " and " + allY[-1],
        "RetroNclosed": str(rt["grid"][RY]["n_closed"]), "RetroNfilled": str(rt["grid"][RY]["n_filled"]),
        "RetroNnei": str(rt["grid"][RY]["n_nei"]), "RetroVisible": f"{100 * rt['grid'][RY]['visible_share']:.0f}\\%",
        "RetroBestPow": f"{100 * gy[f'{best_m}|in-situ']['pow_filled']:.0f}\\%",
        "RetroBestPrec": f"{100 * gy[f'{best_m}|in-situ']['prec']:.0f}\\%",
        "RetroMaxFN": f"{max(rt['grid'][y]['methods'][k]['fn'] for y in allY for k in rt['grid'][y]['methods'] if k.endswith('in-situ')):.3f}",
        "RetroFoneMin": f"{100 * min(gy[k]['fn'] for k in gy if k.endswith('f1-frozen')):.0f}\\%",
        "RetroFoneMax": f"{100 * max(gy[k]['fn'] for k in gy if k.endswith('f1-frozen')):.0f}\\%",
    }
    dos_f = RES / "gap_dossiers.json"
    if dos_f.exists():
        dos = json.load(open(dos_f, encoding="utf-8")).get(RY, [])
        retromac["RetroNdossier"] = str(len(dos))
        lags = sorted(x["lag_years"] for x in dos)
        retromac["RetroLag"] = str(lags[len(lags) // 2]) if lags else "n/a"


        def tex(t):
            for a, b in (("\\", ""), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                         ("µ", r"$\mu$"), ("−", "-"), ("–", "-"), ("—", ", "), ("’", "'"), ("“", "``"), ("”", "''")):
                t = t.replace(a, b)
            return t

        def short(t, n=150):
            t = tex(t.strip())
            return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + r" \ldots{}"
        pick = [x for x in dos if x["nearest_local"] and x["filled_by"]["doi"] and len(x["question"]) < 95
                and not x["filled_by"]["rationale"][0].startswith("Highlights")]
        pick = sorted(pick, key=lambda x: -x["nearest_local"]["score"])[:3]
        rows = []
        for x in pick:
            nl, fb = x["nearest_local"], x["filled_by"]
            rows.append(
                tex(x["question"]) + " & "
                + f"``{short(nl['sentence'], 110)}'' ({nl['year']}, score {nl['score']:.3f} $<\\tau$={x['tau']:.3f}) & "
                + f"``{short(fb['rationale'][0], 150)}'' ({tex(fb['label'].lower())}; {fb['year']}, doi:{tex(fb['doi'])}) \\\\")
        (OUT / "dossier.tex").write_text(
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{5.6cm}>{\raggedright\arraybackslash}p{7.2cm}}" "\n"
            r"\toprule" "\n" rf"Certified open question (cutoff {RY}) & Closest evidence visible at the cutoff & Paper that later answered it \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n", encoding="utf-8")


# ---------------------------------------------------------------- Figure
def coords(key, metric):
    return " ".join(f"({float(p):g},{G[p][key][metric]:.4f})" for p in P)


fig = r"""\begin{tikzpicture}
\begin{axis}[name=left, width=0.47\textwidth, height=4.6cm, xmin=0, xmax=1, ymin=0, ymax=0.72,
  xlabel={deletion rate $p$}, ylabel={false novelty}, label style={font=\footnotesize},
  tick label style={font=\footnotesize}, title={(a) False novelty, ESV-Scope}, title style={font=\footnotesize},
  legend style={font=\scriptsize, at={(0.5,-0.32)}, anchor=north, legend columns=2, draw=none},
  legend cell align=left]
\addplot[thick, black, mark=square*, mark size=1.3pt] coordinates {%s};
\addplot[thick, gray, mark=triangle*, mark size=1.6pt] coordinates {%s};
\addplot[thick, blue, mark=*, mark size=1.3pt] coordinates {%s};
\addplot[thick, teal, mark=x, mark size=2pt] coordinates {%s};
\addplot[dashed, red, domain=0:1, samples=2] {0.1};
\legend{F1-optimal threshold, split conformal (complete), in-situ (Thm.~1), transfer $\hat p$ (Thm.~2), target $\alpha$}
\end{axis}
\begin{axis}[at={(left.east)}, anchor=west, xshift=1.4cm, width=0.47\textwidth, height=4.6cm, xmin=0, xmax=1, ymin=0, ymax=0.32,
  xlabel={deletion rate $p$}, ylabel={power}, label style={font=\footnotesize},
  tick label style={font=\footnotesize}, title={(b) Power at false novelty $\le 0.1$}, title style={font=\footnotesize},
  legend style={font=\scriptsize, at={(0.5,-0.32)}, anchor=north, legend columns=2, draw=none},
  legend cell align=left]
\addplot[thick, blue, mark=*, mark size=1.3pt] coordinates {%s};
\addplot[thick, blue, dashed, mark=o, mark size=1.3pt] coordinates {%s};
\addplot[thick, black!60, mark=diamond*, mark size=1.6pt] coordinates {%s};
\addplot[thick, orange, dotted, mark=star, mark size=1.6pt] coordinates {%s};
\legend{ESV-Scope +OA, ESV-Scope, Dense, ESV-Scope with $\hat p=1$}
\end{axis}
\end{tikzpicture}
""" % (coords("ESV-Scope|f1-frozen", "fnov"), coords("ESV-Scope|conformal", "fnov"),
       coords("ESV-Scope|in-situ", "fnov"), coords("ESV-Scope|transfer", "fnov"), coords("ESV-Scope+OA|transfer", "power"),
       coords("ESV-Scope|transfer", "power"), coords("Dense|transfer", "power"),
       coords("ESV-Scope|worst-case", "power"))
(OUT / "fig.tex").write_text(fig, encoding="utf-8")

# ---------------------------------------------------------------- macros
pct = lambda v: f"{100 * v:.0f}\\%"
mac = {
    "FNfoneZero": f"{G['0.0']['ESV-Scope|f1-frozen']['fnov']:.3f}",
    "FNfoneOne": f"{G['1.0']['ESV-Scope|f1-frozen']['fnov']:.3f}",
    "FNfoneOAOne": f"{G['1.0']['ESV-Scope+OA|f1-frozen']['fnov']:.3f}",
    "FNconfOne": f"{G['1.0']['ESV-Scope|conformal']['fnov']:.3f}",
    "phatThree": f"{G['0.3']['ESV-Scope|transfer']['p_hat']:.3f}",
    "PowESVZero": f"{G['0.0']['ESV-Scope|transfer']['power']:.3f}",
    "PowESVHalf": f"{G['0.5']['ESV-Scope|transfer']['power']:.3f}",
    "PowESVWorst": f"{G['0.5']['ESV-Scope|worst-case']['power']:.3f}",
    "PowOAHalf": f"{G['0.5']['ESV-Scope+OA|transfer']['power']:.3f}",
    "PowOAWorst": f"{G['0.5']['ESV-Scope+OA|worst-case']['power']:.3f}",
    "ExtCalls": pct(G["0.5"]["ESV-Scope+OA|transfer"]["ext_calls"]),
    "Nquestions": f"{cert['n']:,}",
}
hold = load_cert("holdout", "mcar")
if hold is not None:
    import math
    # the holdout evaluation set is fixed: binomial standard error of a false-novelty rate at alpha
    n_closed_eval = hold.get("n_eval_closed")
    if n_closed_eval is None:
        bench_test = json.load(open(HERE.parent / "gapclose" / "outputs" / "bench" / "test.json", encoding="utf-8"))
        ext_ok = json.load(open(HERE.parent / "gapclose" / "outputs" / "external" / "nli_test_large.json", encoding="utf-8"))
        n_closed_eval = sum(c["status"] == "CLOSED" and str(c["id"]) in ext_ok for c in bench_test)
    mac["NholdClosed"] = str(n_closed_eval)
    se = math.sqrt(ALPHA * (1 - ALPHA) / n_closed_eval)
    mac["SEhold"] = f"{se:.3f}"
    hmax = robmac["holdoutmcar"][3] if "holdoutmcar" in robmac else ALPHA  # same value as \RobHMAllTr
    mac["HoldNse"] = f"{(hmax - ALPHA) / se:.1f}"
scores_all = sorted({k.split("|")[0] for k in G["0.0"]})
viol01 = sum(G["0.1"][f"{m}|conformal"]["fnov"] > ALPHA + 0.005 for m in scores_all)
all_by = next((p for p in P if all(G[p][f"{m}|conformal"]["fnov"] > ALPHA + 0.005 for m in scores_all)), None)
words = {i: w for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
mac.update({
    "Nscores": words.get(len(scores_all), str(len(scores_all))),
    "NconfViolOne": words.get(viol01, str(viol01)),
    "ConfAllBy": f"{float(all_by):g}" if all_by else "1",
    "MaxInsitu": f"{max(G[p][f'{m}|in-situ']['fnov'] for m in scores_all for p in P):.3f}",
    "MaxTransfer": f"{max(G[p][f'{m}|transfer']['fnov'] for m in scores_all for p in P):.3f}",
})
if "BGE-rerank|transfer" in G["0.0"]:
    mac.update({
        "FNbgeZero": f"{G['0.0']['BGE-rerank|f1-frozen']['fnov']:.3f}",
        "FNbgeOne": f"{G['1.0']['BGE-rerank|f1-frozen']['fnov']:.3f}",
        "PowBGEZero": f"{G['0.0']['BGE-rerank|transfer']['power']:.3f}",
        "PowBGEHalf": f"{G['0.5']['BGE-rerank|transfer']['power']:.3f}",
        "PowBGEOne": f"{G['1.0']['BGE-rerank|transfer']['power']:.3f}",
    })
for key, vals in robmac.items():
    name = {"pooledmcar": "RobPM", "pooledtargeted": "RobPT", "holdoutmcar": "RobHM", "holdouttargeted": "RobHT"}[key]
    mac[name + "EsvIn"], mac[name + "EsvTr"], mac[name + "AllIn"], mac[name + "AllTr"], mac[name + "Conf"] = (f"{v:.3f}" for v in vals)
mac.update(climmac)
mac.update(retromac)
llm = allm.get("LLM-judge")
mac.update({"LLMfone": f"{llm['macro_f1']:.3f}" if llm else "n/a",
            "LLMfn": f"{llm['false_novelty_rate']:.3f}" if llm else "n/a",
            "LLMaucFull": f"{llm['auc_complete_calib']:.3f}" if llm else "n/a",
            "LLMaucDel": f"{llm['auc_after_deletion_calib']:.3f}" if llm else "n/a"})
if "LLM-judge|transfer" in G["0.0"]:
    mac.update({"LLMfoneOne": f"{G['1.0']['LLM-judge|f1-frozen']['fnov']:.3f}",
                "LLMpowZero": f"{G['0.0']['LLM-judge|transfer']['power']:.3f}"})
else:
    mac.update({"LLMfoneOne": "n/a", "LLMpowZero": "n/a"})
(OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()), encoding="utf-8")

# ---------------------------------------------------------------- checks for the prose
print("macros:", mac)
allm = sorted({k.split("|")[0] for k in G["0.0"]})
for rule in ("in-situ", "transfer"):
    worst = max((G[p][f"{m}|{rule}"]["fnov"], m, p) for m in allm for p in P)
    print(f"max mean FN under {rule}: {worst}")
viol = [(m, p) for m in allm for p in P if G[p][f"{m}|conformal"]["fnov"] > ALPHA + 0.005]
print("split conformal first violating p per method:",
      {m: min(float(p) for mm, p in viol if mm == m) for m in allm if any(mm == m for mm, _ in viol)})
for p in ("0.1", "0.3", "0.5"):
    print(f"ESV transfer/worst power ratio at p={p}:",
          round(G[p]["ESV-Scope|transfer"]["power"] / G[p]["ESV-Scope|worst-case"]["power"], 2))
print("f1-frozen FN range at p=0 over methods:",
      [round(G["0.0"][f"{m}|f1-frozen"]["fnov"], 3) for m in allm])
print("f1-frozen FN at p=1 over methods:", {m: round(G["1.0"][f"{m}|f1-frozen"]["fnov"], 3) for m in allm})
print("worst-case power ESV vs +OA:", G["0.5"]["ESV-Scope|worst-case"]["power"], G["0.5"]["ESV-Scope+OA|worst-case"]["power"])
