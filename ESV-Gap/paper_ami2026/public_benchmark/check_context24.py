"""Independent arithmetic, provenance, scorer, and budget robustness checks.

Budget sensitivity is post hoc; it does not select a new deployed weight.
"""
import hashlib
import json
import sys
from pathlib import Path

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]
sys.path.insert(0,str(PROJECT))
from src.context_retrieval import context_prior, retrieve_context, terms, word_overlap


def conventional_lcs(a,b):
    row=[0]*(len(b)+1)
    for x in a:
        new=[0]
        for j,y in enumerate(b,1):
            new.append(row[j-1]+1 if x==y else max(new[-1],row[j]))
        row=new
    return row[-1]


def main():
    result_dir=BASE/'results'
    report=json.loads((result_dir/'context24_report.json').read_text(encoding="utf-8"))
    examples=json.loads((BASE/'context24/task2-train-dev.json').read_text(encoding="utf-8"))
    texts=json.loads((BASE/'context24/full_texts-2024-04-25-update.json').read_text(encoding="utf-8"))
    by_id={row['id']:row for row in examples}
    checks=0
    for method,summary in report['macro_metrics'].items():
        predictions=json.loads((result_dir/f'predictions_{method}.json').read_text(encoding="utf-8"))
        assert set(p['id'] for p in predictions)==set(by_id)
        for prediction in predictions:
            source=texts[prediction['citekey']]
            spans=prediction['source_spans']
            assert all(s['text']==source[s['start']:s['end']] and s['reviewed'] is False for s in spans)
            assert sum(len(s['text'].split()) for s in spans)<=512
            checks+=1
        for metric,value in summary.items():
            mean=sum(row['metrics'][method][metric] for row in report['rows'])/len(report['rows'])
            assert abs(mean-value)<1e-12
    # Check optimized LCS against the conventional algorithm on bounded prefixes.
    for prediction in json.loads((result_dir/'predictions_context_diverse.json').read_text(encoding="utf-8"))[:10]:
        pred=' '.join(prediction['context']);gold=' '.join(by_id[prediction['id']]['context'])
        a,b=terms(pred)[:80],terms(gold)[:80]
        lcs=conventional_lcs(a,b)
        expected=2*lcs/(len(a)+len(b)) if a and b else 0.
        assert abs(word_overlap(' '.join(a),' '.join(b))['rougel_f1']-expected)<1e-12
    sensitivity={}
    for budget in [256,768]:
        scored={'bm25':[],'context_diverse':[]}
        priors={}
        for example in examples:
            paper=example['citekey']
            if paper not in priors:
                training=[r for r in examples if r['citekey']!=paper]
                priors[paper]=context_prior(training,texts)
            for method in scored:
                snippets=retrieve_context(texts[paper],example['claim'],prior=priors[paper],
                                          method=method,budget=budget)
                assert sum(s['word_count'] for s in snippets)<=budget
                scored[method].append(word_overlap(' '.join(s['text'] for s in snippets),
                                                  ' '.join(example['context'])))
        sensitivity[str(budget)]={method:{metric:sum(r[metric] for r in values)/len(values)
                                         for metric in ['rouge1_f1','rouge2_f1','rougel_f1']}
                                 for method,values in scored.items()}
    result={'status':'passed','verified_saved_predictions':checks,
            'macro_arithmetic_check':True,'source_and_budget_checks':True,
            'bit_lcs_vs_conventional_lcs_checks':10,
            'posthoc_word_budget_sensitivity':sensitivity,
            'interpretation':'Retrieval robustness only; no new closure labels or official-test claim.',
            'benchmark_report_sha256':hashlib.sha256((result_dir/'context24_report.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (result_dir/'verification_and_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
