"""Rerun the primary diagnostic audit with the entire MongoDB domain excluded.

Preserved historical inputs/code are read, never modified. This is post-hoc
domain exclusion, not provenance repair, expert validation, or new extraction.
"""
import hashlib
import json
import platform
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
sys.path.insert(0, str(PROJECT))
import yaml
from src.temporal_backtest import run_cutoff_backtest, evaluate_ranked_candidates
from paper_icai2026.expanded_run.run_multidomain_backtest import DOMAIN_TEXT, direct_limitation_baseline, paper_key
from paper_icai2026.expanded_run.run_bm25_lacks_baseline import bm25_scores, matched_ids

ARCHIVE = PROJECT/'paper_icai2026/expanded_run'
OUTPUT = BASE/'primary_audit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = read(ARCHIVE/'multidomain_summary.json')
    config_path = PROJECT/'config.yaml'
    assert digest(config_path) == frozen['fingerprints']['config']
    for label, filename in [('temporal_code','src/temporal_backtest.py'),
                            ('detector_code','src/detect_gaps.py')]:
        assert digest(PROJECT/filename) == frozen['fingerprints'][label]
    sources, rows, keys, unique_controls = [], [], [], set()
    for source in frozen['sources']:
        domain = source['domain']
        if domain == 'mongodb_security':
            continue
        run = PROJECT/'runs'/source['source_run']
        corpus_path, triples_path = run/'data/processed/corpus_filtered.jsonl', run/'data/triples/all_triples.json'
        hashes = {'corpus':digest(corpus_path),'triples':digest(triples_path)}
        assert hashes == frozen['fingerprints']['inputs'][domain]
        documents = [json.loads(line) for line in corpus_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        triples = read(triples_path)
        keys.extend(paper_key(d) for d in documents)
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        config['project']['domain'] = DOMAIN_TEXT[domain]
        sources.append({'domain':domain,'records':len(documents),'triples':len(triples),'input_sha256':hashes})
        for cutoff in (2022,2023,2024):
            pre_ids = {str(d.get('paperId') or d.get('paper_id') or '') for d in documents if int(d.get('year') or 0)<=cutoff}
            if len(pre_ids)<10 or len(documents)-len(pre_ids)<10:
                continue
            result = run_cutoff_backtest(documents,triples,config,cutoff,OUTPUT/domain)
            saved = read(ARCHIVE/domain/f'cutoff_{cutoff}/temporal_backtest.json')
            assert result['leakage_check_passed']
            # Test actual regeneration, not only saved aggregate arithmetic.
            for field in ['metrics','positive_controls','negative_controls','ranked_candidates']:
                assert result[field] == saved[field], (domain,cutoff,field)
            direct = direct_limitation_baseline(triples,pre_ids)
            query = DOMAIN_TEXT[domain]+' limitation unresolved challenge'
            scores = bm25_scores([r['subject']+' '+r['missing_capability'] for r in direct],query)
            ordered = sorted(zip(direct,scores),key=lambda p:(-p[1],-p[0]['mean_evidence_confidence'],p[0]['subject'],p[0]['missing_capability']))
            bm25 = [{**candidate,'candidate_quality':{'score':score}} for candidate,score in ordered]
            bmetrics = evaluate_ranked_candidates(bm25,result['positive_controls'],result['negative_controls'])
            cmetrics = evaluate_ranked_candidates(direct,result['positive_controls'],result['negative_controls'])
            unique_controls.update((domain,c['control_id']) for c in result['positive_controls'])
            kinds = dict(Counter(c['type'] for c in result['ranked_candidates']))
            rows.append({'domain':domain,'cutoff':cutoff,'pre':result['pre_cutoff_papers'],
                         'future':result['post_cutoff_papers'],'controls':len(result['positive_controls']),
                         'esv_candidates':len(result['ranked_candidates']),'candidate_types':kinds,
                         'lacks_pool':len(direct),'esv_recall':result['metrics']['candidate_recall_at_k'],
                         'confidence_recall':cmetrics['candidate_recall_at_k'],
                         'bm25_recall':bmetrics['candidate_recall_at_k'],
                         'bm25_matches20':len(matched_ids(bm25,result['positive_controls'])),
                         'esv_matches20':len(matched_ids(result['ranked_candidates'],result['positive_controls'])),
                         'regenerated_records_equal_archive':True,
                         'output_sha256':digest(OUTPUT/domain/f'cutoff_{cutoff}/temporal_backtest.json')})
    assert len(keys)==len(set(keys))
    scored = [r for r in rows if r['controls']]
    report = {'status':'posthoc_three_domain_offline_regeneration_passed',
              'exclusion_reason':'Whole MongoDB domain excluded because one matched record has unresolved bibliographic provenance/date.',
              'exclusion_preregistered':False,'provenance_repaired':False,'other_domains_expert_validated':False,
              'source_records':len(keys),'triples':sum(s['triples'] for s in sources),
              'completed_evaluations':len(rows),'scored_evaluations':len(scored),
              'control_rows_nonindependent':sum(r['controls'] for r in rows),
              'distinct_control_ids_within_domain':len(unique_controls),
              'esv_candidate_rows_nonindependent':sum(r['esv_candidates'] for r in rows),
              'lacks_pool_rows_nonindependent':sum(r['lacks_pool'] for r in rows),
              'macro_proxy_recall':{method:{str(k):sum(r[method+'_recall'][str(k)] for r in scored)/len(scored)
                                           for k in (5,10,20)} for method in ['esv','confidence','bm25']},
              'match_counts20':{method:sum(r[method+'_matches20'] for r in scored) for method in ['esv','bm25']},
              'all_regenerated_records_equal_archive':all(r['regenerated_records_equal_archive'] for r in rows),
              'sources':sources,'rows':rows,
              'fingerprints':{'script':digest(__file__),'config':digest(config_path),
                              'original_summary':digest(ARCHIVE/'multidomain_summary.json'),
                              'code':{str(p.relative_to(PROJECT)):digest(p) for p in [PROJECT/'src/temporal_backtest.py',
                                      PROJECT/'src/detect_gaps.py',PROJECT/'src/evidence_graph.py',
                                      ARCHIVE/'run_multidomain_backtest.py',ARCHIVE/'run_bm25_lacks_baseline.py']}},
              'python':platform.python_version(),
              'packages':{name:metadata.version(name) for name in ['networkx','PyYAML','numpy','python-louvain']},
              'interpretation':'Diagnostic proxy ranking only; no semantic labels, closure efficacy, or novelty measured.'}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'primary_audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','sources','fingerprints']},indent=2))


if __name__=='__main__':
    main()
