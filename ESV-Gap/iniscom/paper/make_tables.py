"""Generate LaTeX tables, pgfplots data and number macros from ../outputs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)

main = json.load(open(OUT / "results_main.json", encoding="utf-8"))
stats = json.load(open(OUT / "corpus_stats.json", encoding="utf-8"))
abl = json.load(open(OUT / "ablation.json", encoding="utf-8"))
CUT = list(main["cutoffs"])
NAMES = {"random": "Random", "expected": "Expected co-occurrence",
         "growth_min": "Concept growth (min)", "profile_cos": "PPMI profile cosine",
         "recent_cos": "Recent PPMI cosine", "adamic_adar": "Adamic--Adar index",
         "semantic": "Semantic similarity (MiniLM)",
         "lr_pop_growth": "Logistic, popularity + growth", "gbm": "Gradient boosting",
         "learned": "Learned forecaster"}
SINGLE = [n for n in NAMES if n not in ("lr_pop_growth", "gbm", "learned", "random")]

macros = {}
def mac(name, val):
    macros[name] = val


# ------------------------------------------------------------- corpus
mac("nWorks", f"{stats['works']:,}")
mac("nConcepts", str(stats["concepts"]))
mac("taggedShare", f"{100 * stats['tagged_share']:.1f}")
mac("meanConcepts", f"{stats['mean_concepts_per_work']:.2f}")
mac("nTopics", str(len(stats["per_topic"])))
mac("pairsSentShare", f"{100 * stats['pairs_sentence'] / stats['pairs_doc']:.0f}")
mac("pairsSurveyShare", f"{100 * (1 - stats['pairs_sentence_nosurvey'] / stats['pairs_sentence']):.1f}")

# ------------------------------------------------------------- main table
best = {m: max(main["mean"][n][m] for n in NAMES if n != "learned") for m in ("auc", "ap", "p@20", "open@20")}
lines = [r"\begin{tabular}{l" + "c" * len(CUT) + "cccc}", r"\toprule",
         r" & \multicolumn{" + str(len(CUT)) + r"}{c}{AUC at cutoff $T$} & \multicolumn{4}{c}{Mean over cutoffs}\\",
         r"\cmidrule(lr){2-" + str(1 + len(CUT)) + r"}\cmidrule(lr){" + str(2 + len(CUT)) + "-" + str(5 + len(CUT)) + "}",
         "Score & " + " & ".join(CUT) + r" & AUC & AP & P@20 & Open@20\\", r"\midrule"]
for n, label in NAMES.items():
    if n in ("lr_pop_growth", "learned"):
        lines.append(r"\midrule")
    cells = [f"{main['cutoffs'][T]['methods'][n]['auc']:.3f}" for T in CUT]
    mean = main["mean"][n]
    mc = []
    for m in ("auc", "ap", "p@20", "open@20"):
        v = f"{mean[m]:.3f}"
        mc.append(r"\textbf{" + v + "}" if n == "learned" and mean[m] >= best[m] else v)
    lines.append(f"{label} & " + " & ".join(cells) + " & " + " & ".join(mc) + r"\\")
lines += [r"\midrule", "Gaps / filled rate & " + " & ".join(
    f"{main['cutoffs'][T]['gaps']}/{main['cutoffs'][T]['filled_rate']:.2f}" for T in CUT) + r" & \multicolumn{4}{c}{}\\",
          r"\bottomrule", r"\end{tabular}"]
(TAB / "main.tex").write_text("\n".join(lines), encoding="utf-8")

L = main["mean"]["learned"]
mac("learnedAUC", f"{L['auc']:.3f}")
mac("learnedAP", f"{L['ap']:.3f}")
mac("learnedPtw", f"{L['p@20']:.2f}")
mac("learnedOpen", f"{L['open@20']:.2f}")
bn = max(SINGLE, key=lambda n: main["mean"][n]["auc"])
mac("lrpgAUC", f"{main['mean']['lr_pop_growth']['auc']:.3f}")
mac("gbmAUC", f"{main['mean']['gbm']['auc']:.3f}")
mac("aaAUC", f"{main['mean']['adamic_adar']['auc']:.3f}")
cc = [main["cutoffs"][T]["ci_cluster_learned_minus_expected"] for T in CUT]
cl = [main["cutoffs"][T]["ci_cluster_learned_minus_lrpg"] for T in CUT]
mac("ciClExpLoMin", f"{min(c[0] for c in cc):.3f}")
mac("ciClLrpgExclude", str(sum(c[0] > 0 for c in cl)))
mac("ciClExpExclude", str(sum(c[0] > 0 for c in cc)))
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}
mac("ciClLrpgExcludeW", WORDS[sum(c[0] > 0 for c in cl)])
mac("ciClExpExcludeW", WORDS[sum(c[0] > 0 for c in cc)])
mac("maxSingleCut", f"{max(main['cutoffs'][T]['methods'][n]['auc'] for T in CUT for n in SINGLE):.3f}")
mac("bestBaseName", NAMES[bn])
mac("bestBaseAUC", f"{main['mean'][bn]['auc']:.3f}")
mac("expectedAUC", f"{main['mean']['expected']['auc']:.3f}")
mac("semanticAUC", f"{main['mean']['semantic']['auc']:.3f}")
mac("minGaps", str(min(main["cutoffs"][T]["gaps"] for T in CUT)))
mac("maxGaps", str(max(main["cutoffs"][T]["gaps"] for T in CUT)))
fr = [main["cutoffs"][T]["filled_rate"] for T in CUT]
mac("fillMin", f"{100 * min(fr):.0f}")
mac("fillMax", f"{100 * max(fr):.0f}")
ci = [main["cutoffs"][T]["ci_learned_minus_expected"] for T in CUT]
mac("ciExpLoMin", f"{min(c[0] for c in ci):.3f}")
ci2 = [main["cutoffs"][T]["ci_learned_minus_recent"] for T in CUT]
mac("ciRecLoMin", f"{min(c[0] for c in ci2):.3f}")
mac("ciRecLoMinCut", CUT[min(range(len(CUT)), key=lambda i: ci2[i][0])])
ca = [main["cutoffs"][T]["ci_cluster_learned_minus_aa"] for T in CUT]
mac("ciClAaExcludeW", WORDS[sum(c[0] > 0 for c in ca)])
mac("ciClAaLoMin", f"{min(c[0] for c in ca):.3f}")
mac("ciClAaLoMinCut", CUT[min(range(len(CUT)), key=lambda i: ca[i][0])])
mac("aaGain", f"{main['mean']['learned']['auc'] - main['mean']['adamic_adar']['auc']:.3f}")
ciall = [main["cutoffs"][T]["ci_learned_minus_expected"] + main["cutoffs"][T]["ci_cluster_learned_minus_expected"]
         + main["cutoffs"][T]["ci_cluster_learned_minus_aa"] + main["cutoffs"][T]["ci_cluster_learned_minus_lrpg"]
         for T in CUT]
(TAB / "ci.tex").write_text("\n".join(
    [r"\begin{tabular}{lcccc}", r"\toprule",
     r"$T$ & \shortstack{Expected co-occ.\\pair bootstrap}"
     r" & \shortstack{Expected co-occ.\\concept bootstrap}"
     r" & \shortstack{Adamic--Adar\\concept bootstrap}"
     r" & \shortstack{Logistic P+G\\concept bootstrap}\\", r"\midrule"]
    + [f"{T} & [{c[0]:.3f}, {c[1]:.3f}] & [{c[2]:.3f}, {c[3]:.3f}] & [{c[4]:.3f}, {c[5]:.3f}]"
       f" & [{c[6]:.3f}, {c[7]:.3f}]" + r"\\" for T, c in zip(CUT, ciall)]
    + [r"\bottomrule", r"\end{tabular}"]), encoding="utf-8")

# coefficients (last cutoff)
FLAB = {"expected": "Expected co-occ.", "min_occ": "Min. support", "growth_min": "Growth (min)",
        "growth_max": "Growth (max)", "profile_cos": "PPMI profile cos.", "recent_cos": "Recent PPMI cos.",
        "semantic": "Semantic sim."}
co = main["cutoffs"][CUT[-1]]["coef"]
mac("coefGrowthMax", f"{co['growth_max']:+.2f}")
mac("coefGrowthMin", f"{co['growth_min']:+.2f}")
mac("coefExpected", f"{co['expected']:+.2f}")
mac("coefMinOcc", f"{co['min_occ']:+.2f}")
mac("coefProfile", f"{co['profile_cos']:+.2f}")
mac("coefRecent", f"{co['recent_cos']:+.2f}")
mac("coefSemantic", f"{co['semantic']:+.2f}")
(TAB / "coef.tex").write_text("\n".join(
    [r"\begin{tabular}{lr}", r"\toprule", r"Feature (standardised) & Weight\\", r"\midrule"]
    + [f"{FLAB[k]} & {v:+.2f}" + r"\\" for k, v in co.items() if k != "co_prior"]
    + [r"\bottomrule", r"\end{tabular}"]), encoding="utf-8")

# expected-quintile fill rates (averaged over cutoffs)
q = [sum(main["cutoffs"][T]["fill_rate_by_expected_quintile"][i] for T in CUT) / len(CUT) for i in range(5)]
mac("fillQlow", f"{100 * q[0]:.0f}")
mac("fillQhigh", f"{100 * q[4]:.0f}")
(TAB / "quintile.dat").write_text("q rate\n" + "\n".join(f"{i+1} {v:.4f}" for i, v in enumerate(q)), encoding="utf-8")

# ------------------------------------------------------------- survival
sv = main["survival"]
mac("survT", str(sv["T0"]))
mac("survGaps", str(sv["gaps"]))
fy = sv["filled_by_year"]
mac("survThree", f"{100 * fy[str(sv['T0'] + 3)]:.0f}")
mac("survLast", f"{100 * fy[max(fy)]:.0f}")
mac("survLastYear", max(fy))
(TAB / "survival.dat").write_text("year rate\n" + "\n".join(f"{y} {v:.4f}" for y, v in fy.items()), encoding="utf-8")

# ------------------------------------------------------------- ablation
ALAB = {"all": "All features", "-expected,min_occ": "without popularity", "-growth": "without growth",
        "-profile_cos": "without PPMI profile cosine", "-recent_cos": "without recent PPMI cosine",
        "-semantic": "without semantic similarity", "only growth": "growth only", "only popularity": "popularity only"}
rows = [r"\begin{tabular}{lcc}", r"\toprule", r"Feature set & AUC & AP\\", r"\midrule"]
for k, v in abl.items():
    if k == "examples":
        continue
    rows.append(f"{ALAB[k]} & {v['auc']:.3f} & {v['ap']:.3f}" + r"\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(TAB / "ablation.tex").write_text("\n".join(rows), encoding="utf-8")
mac("ablNoGrowth", f"{abl['-growth']['auc']:.3f}")
mac("ablOnlyGrowth", f"{abl['only growth']['auc']:.3f}")
mac("ablOnlyPop", f"{abl['only popularity']['auc']:.3f}")
mac("ablNoSem", f"{abl['-semantic']['auc']:.3f}")
mac("ablNoPop", f"{abl['-expected,min_occ']['auc']:.3f}")

# ------------------------------------------------------------- robustness
rob = [("main", r"main ($m{=}50$, $g{=}0$, $k{=}3$, $H{=}3$)"), ("g2", r"$g{=}2$ (sparse, not only absent)"),
       ("k1", r"$k{=}1$"), ("k5", r"$k{=}5$"), ("h2", r"$H{=}2$"), ("occ20", r"$m{=}20$"), ("occ100", r"$m{=}100$"),
       ("sent", "same-sentence closure label"), ("sentns", "same sentence, surveys/reviews excluded"),
       ("vocab2013", "concepts established by 2013 only")]
rr = [r"\begin{tabular}{lcccc}", r"\toprule", r"Setting & Gaps (mean) & Filled & Best single score & Learned\\",
      r"\midrule"]
lo, hi = 9, 0
for tag, label in rob:
    f = OUT / f"results_{tag}.json"
    if not f.exists():
        continue
    r = json.load(open(f, encoding="utf-8"))
    cs = list(r["cutoffs"])
    g = sum(r["cutoffs"][T]["gaps"] for T in cs) / len(cs)
    fr_ = sum(r["cutoffs"][T]["filled_rate"] for T in cs) / len(cs)
    b = max(r["mean"][n]["auc"] for n in SINGLE)
    la = r["mean"]["learned"]["auc"]
    if tag == "vocab2013":  # too few gaps per cutoff to enter the range quoted in the text
        mac("vocabGapsLo", str(min(r["cutoffs"][T]["gaps"] for T in cs)))
        mac("vocabGapsHi", str(max(r["cutoffs"][T]["gaps"] for T in cs)))
        mac("vocabLearned", f"{la:.2f}")
        mac("vocabBest", f"{b:.2f}")
    else:
        lo, hi = min(lo, la), max(hi, la)
    if tag == "sentns":
        mac("sentnsLearned", f"{la:.3f}")
        mac("sentnsBest", f"{b:.3f}")
        mac("sentnsFill", f"{100 * fr_:.0f}")
        mac("sentnsLrpg", f"{r['mean']['lr_pop_growth']['auc']:.3f}")
    if tag == "sent":
        mac("sentLearned", f"{la:.3f}")
        mac("sentBest", f"{b:.3f}")
        mac("sentFill", f"{100 * fr_:.0f}")
        mac("sentLrpg", f"{r['mean']['lr_pop_growth']['auc']:.3f}")
    rr.append(f"{label} & {g:.0f} & {fr_:.2f} & {b:.3f} & {la:.3f}" + r"\\")
rr += [r"\bottomrule", r"\end{tabular}"]
(TAB / "robust.tex").write_text("\n".join(rr), encoding="utf-8")
mac("robLo", f"{lo:.2f}")
mac("robHi", f"{hi:.2f}")

# ------------------------------------------------------------- forecast and examples
fc = main["forecast"]
mac("fcT", str(fc["T"]))
mac("fcGaps", str(fc["gaps"]))
def pair(p):
    return r" $\times$ ".join(p)
fl = [r"\begin{tabular}{lc@{\hspace{12pt}}lc}", r"\toprule",
      r"\multicolumn{2}{c}{Most likely to be filled by 2028} & \multicolumn{2}{c}{Most likely to stay open}\\",
      r"\cmidrule(lr){1-2}\cmidrule(lr){3-4}", r"Concept pair & $\hat p$ & Concept pair & $\hat p$\\", r"\midrule"]
for a, b in zip(fc["likely_filled"][:10], fc["persistent"][:10]):
    fl.append(f"{pair(a['pair'])} & {a['p_fill']:.2f} & {pair(b['pair'])} & {b['p_fill']:.3f}" + r"\\")
fl += [r"\bottomrule", r"\end{tabular}"]
(TAB / "forecast.tex").write_text("\n".join(fl), encoding="utf-8")

ex = abl["examples"]
mac("exT", str(ex["T"]))
mac("exTopFilled", str(sum(e["filled"] for e in ex["top"])))
mac("exBottomFilled", str(sum(e["filled"] for e in ex["bottom"])))
el = [r"\begin{tabular}{lr@{\hspace{12pt}}lr}", r"\toprule",
      r"\multicolumn{2}{c}{Top-ranked at $T{=}" + str(ex["T"]) + r"$} & \multicolumn{2}{c}{Bottom-ranked at $T{=}" + str(ex["T"]) + r"$}\\",
      r"\cmidrule(lr){1-2}\cmidrule(lr){3-4}", r"Concept pair & Co-occ. & Concept pair & Co-occ.\\", r"\midrule"]
for a, b in zip(ex["top"], ex["bottom"]):
    el.append(f"{pair(a['pair'])} & {a['future_co']} & {pair(b['pair'])} & {b['future_co']}" + r"\\")
el += [r"\bottomrule", r"\end{tabular}"]
(TAB / "examples.tex").write_text("\n".join(el), encoding="utf-8")

sb = json.load(open(OUT / "subsets.json", encoding="utf-8"))
mac("subN", str(sb["n"]))
mac("subK", str(sb["runs"][0]["concepts"]))
mac("subLearnedMed", f"{sb['learned_median']:.3f}")
mac("subLearnedMin", f"{sb['learned_min']:.3f}")
mac("subLearnedMax", f"{sb['learned_max']:.3f}")
mac("subBestMed", f"{sb['best_median']:.3f}")
mac("subBestMin", f"{sb['best_min']:.3f}")
mac("subBestMax", f"{sb['best_max']:.3f}")
mac("subWins", str(sb["learned_wins"]))
mac("subGapMin", f"{sb['gap_min']:.1f}")
mac("subGapMax", f"{sb['gap_max']:.1f}")
mac("subLrpgGainMed", f"{sb['learned_minus_lrpg_median']:.3f}")
st = [r"\begin{tabular}{lccc}", r"\toprule",
      r"Method & Median & Minimum & Maximum\\", r"\midrule"]
for label, prefix in [("Learned forecaster", "learned"),
                      ("Logistic, popularity + growth", "lrpg"),
                      ("Strongest single score in subset", "best")]:
    st.append(f"{label} & {sb[prefix + '_median']:.3f} & {sb[prefix + '_min']:.3f}"
              f" & {sb[prefix + '_max']:.3f}" + r"\\")
st += [r"\bottomrule", r"\end{tabular}"]
(TAB / "subsets.tex").write_text("\n".join(st), encoding="utf-8")
(TAB / "macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()), encoding="utf-8")
print(json.dumps(macros, indent=1))

# ------------------------------------------------------------- areas and calibration (src/extra.py)
ex2 = json.load(open(OUT / "extra.json", encoding="utf-8"))
AR = ["Wireless", "Industrial/intell.", "Security", "HW/SW", "Signal/data", "Quantum"]
na, fa = ex2["n"], ex2["filled"]
at = [r"\begin{tabular}{l" + "c" * len(AR) + "}", r"\toprule", " & " + " & ".join(AR) + r"\\", r"\midrule"]
for i, a in enumerate(AR):
    cells = []
    for j in range(len(AR)):
        if j < i:
            cells.append("")
        elif na[i][j] < 20:
            cells.append(r"\textcolor{gray}{(" + str(int(na[i][j])) + ")}")
        else:
            cells.append(f"{fa[i][j] / na[i][j]:.2f} ({int(na[i][j])})")
    at.append(f"{a} & " + " & ".join(cells) + r"\\")
at += [r"\bottomrule", r"\end{tabular}"]
(TAB / "areas.tex").write_text("\n".join(at), encoding="utf-8")
mac("withinRate", f"{100 * ex2['within_rate']:.0f}")
mac("acrossRate", f"{100 * ex2['across_rate']:.0f}")
mac("withinN", str(ex2["within_n"]))
mac("acrossN", f"{ex2['across_n']:,}")
mac("brier", f"{ex2['brier']:.3f}")
mac("brierBase", f"{ex2['brier_base']:.3f}")
mac("pooledN", f"{ex2['pooled_n']:,}")
mac("pooledRate", f"{100 * ex2['pooled_rate']:.0f}")
rel = ex2["reliability"]
mac("relTopP", f"{rel[-1]['mean_p']:.2f}")
mac("relTopObs", f"{rel[-1]['observed']:.2f}")
mac("relLowP", f"{rel[0]['mean_p']:.2f}")
mac("relLowObs", f"{rel[0]['observed']:.2f}")
rc = ex2["reliability_cal"]
mac("brierCal", f"{ex2['brier_cal']:.3f}")
mac("relCalTopP", f"{rc[-1]['mean_p']:.2f}")
mac("relCalTopObs", f"{rc[-1]['observed']:.2f}")
mac("relCalLowP", f"{rc[0]['mean_p']:.2f}")
mac("relCalLowObs", f"{rc[0]['observed']:.2f}")
rt = [r"\begin{tabular}{ccccc}", r"\toprule",
      r" & \multicolumn{2}{c}{Raw logistic output} & \multicolumn{2}{c}{Temporal Platt scaling}\\",
      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
      r"Quintile & Mean $\hat p$ & Observed & Mean $\hat p$ & Observed\\", r"\midrule"]
rt += [f"{i + 1} & {b['mean_p']:.3f} & {b['observed']:.3f} & {c['mean_p']:.3f} & {c['observed']:.3f}" + r"\\"
       for i, (b, c) in enumerate(zip(rel, rc))]
rt += [r"\midrule", f"Brier & \\multicolumn{{2}}{{c}}{{{ex2['brier']:.3f}}} & \\multicolumn{{2}}{{c}}{{{ex2['brier_cal']:.3f}}}"
       + r"\\"]
rt += [r"\bottomrule", r"\end{tabular}"]
(TAB / "calib.tex").write_text("\n".join(rt), encoding="utf-8")
(TAB / "macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()), encoding="utf-8")

# ------------------------------------------------------------- lexicon and corpus tables
import sys
sys.path.insert(0, str(HERE.parent / "src"))
from extra import AREA_NAMES, concept_areas  # noqa: E402
from lexicon import CONCEPTS  # noqa: E402
from fetch_corpus import TOPICS  # noqa: E402

ca = concept_areas()
lx = [r"\begin{tabular}{L{2.6cm}L{9.0cm}}", r"\toprule", r"Area & Concepts\\", r"\midrule"]
for i, a in enumerate(AREA_NAMES):
    names = sorted((c for c, k in zip(CONCEPTS, ca) if k == i), key=str.lower)
    lx.append(f"{a} ({len(names)}) & " + ", ".join(names) + r"\\")
lx += [r"\bottomrule", r"\end{tabular}"]
(TAB / "lexicon.tex").write_text("\n".join(lx), encoding="utf-8")
freq = stats["concept_freq"]
top5 = sorted(freq, key=freq.get, reverse=True)[:5]
mac("topConcepts", ", ".join(f"{c} ({freq[c]:,})" for c in top5))

tp = [r"\begin{tabular}{llr}", r"\toprule", r"ID & OpenAlex topic & Works\\", r"\midrule"]
for tid, n in sorted(stats["per_topic"].items(), key=lambda x: -x[1]):
    tp.append(f"{tid} & {TOPICS[tid]} & {n:,}" + r"\\")
tp += [r"\midrule", f" & Total & {stats['works']:,}" + r"\\", r"\bottomrule", r"\end{tabular}"]
(TAB / "topics.tex").write_text("\n".join(tp), encoding="utf-8")
(TAB / "macros.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()), encoding="utf-8")
