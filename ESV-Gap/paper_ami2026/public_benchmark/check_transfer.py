"""Post-hoc cross-lab transfer and paired ablation diagnostics; no retuning."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1]))
from src.context_retrieval import context_prior, retrieve_context, word_overlap


def main():
    examples = json.loads((BASE/'context24/task2-train-dev.json').read_text(encoding='utf-8'))
    texts = json.loads((BASE/'context24/full_texts-2024-04-25-update.json').read_text(encoding='utf-8'))
    saved = json.loads((BASE/'results/context24_report.json').read_text(encoding='utf-8'))
    comparisons = {}
    for comparator in ['tfidf', 'prior_only', 'context']:
        by_paper = defaultdict(list)
        for row in saved['rows']:
            by_paper[row['citekey']].append(row['metrics']['context_diverse']['rougel_f1']-
                                          row['metrics'][comparator]['rougel_f1'])
        values = [sum(v)/len(v) for _, v in sorted(by_paper.items())]
        rng = random.Random(20261001)
        boot = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(2000))
        comparisons[comparator] = {'paper_macro_rougel_difference':sum(values)/len(values),
                                  'descriptive_95_percentile_interval':[boot[49],boot[1949]]}
    transfer = {}
    for lab in sorted({r['dataset'] for r in examples}):
        held = [r for r in examples if r['dataset']==lab]
        held_papers = {r['citekey'] for r in held}
        training = [r for r in examples if r['dataset']!=lab and r['citekey'] not in held_papers]
        assert not held_papers & {r['citekey'] for r in training}
        prior = context_prior(training, texts)
        scores = []
        for row in held:
            spans = retrieve_context(texts[row['citekey']],row['claim'],prior=prior)
            scores.append(word_overlap(' '.join(s['text'] for s in spans),' '.join(row['context'])))
        transfer[lab] = {'held_claims':len(held),'training_claims':len(training),
                         'training_papers':len({r['citekey'] for r in training}),
                         'held_papers_excluded':True,
                         'rouge1_f1':sum(s['rouge1_f1'] for s in scores)/len(scores),
                         'rouge2_f1':sum(s['rouge2_f1'] for s in scores)/len(scores),
                         'rougel_f1':sum(s['rougel_f1'] for s in scores)/len(scores),
                         'bm25_rougel_f1':saved['domain_metrics'][lab]['bm25']['rougel_f1']}
    result = {'posthoc':True,'weights_retuned':False,'word_budget':512,
              'paired_paper_ablation_comparisons':comparisons,'cross_lab_transfer':transfer,
              'interpretation':'Small auxiliary benchmark; no independent significance or gap-closure claim.'}
    (BASE/'results/transfer_and_ablations.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
