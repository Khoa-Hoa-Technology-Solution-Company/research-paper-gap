"""Print the paper tables from outputs/results/main<tag>.json."""
from __future__ import annotations

import argparse
import json

from common import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_large")
    args = ap.parse_args()
    r = json.load(open(OUT / "results" / f"main{args.tag}.json", encoding="utf-8"))
    print("ESV config:", r.get("esv_config"))
    print(f"{'method':11s} {'F1':>6s} {'CI':>15s} {'FNov':>6s} {'FClo':>6s} {'gRecall':>7s} {'wSent':>6s} {'AURC':>6s}")
    for n, m in r["methods"].items():
        ws = m["witness_sent_precision"]
        print(f"{n:11s} {m['macro_f1']:6.3f} [{m['macro_f1_ci'][0]:.3f},{m['macro_f1_ci'][1]:.3f}] "
              f"{m['false_novelty_rate']:6.3f} {m['false_closure_rate']:6.3f} {m['grounded_closure_recall']:7.3f} "
              f"{'   -  ' if ws is None else f'{ws:6.3f}'} {m['aurc']:6.3f}")
    print("\npaired (ESV minus other):")
    for n, d in r["paired_vs_esv"].items():
        print(f"  {n:11s} dF1={d['delta_macro_f1']:+.3f} CI=[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] b/c={d['mcnemar_b_c']}")
    print("\nstress test: false novelty (gold CLOSED declared OPEN) by deletion rate p")
    grid = r["stress"]["grid"]
    ps = list(grid)
    print(f"  {'config':22s} " + " ".join(f"p={p:<5s}" for p in ps) + "  abstainOPEN@p=0")
    for k in grid[ps[0]]:
        print(f"  {k:22s} " + " ".join(f"{grid[p][k]['false_novelty'][0]:7.3f}" for p in ps)
              + f"  {grid[ps[0]][k]['abstain_open'][0]:.3f}")


if __name__ == "__main__":
    main()
