"""Freeze a manageable unlabelled pilot from pre-cutoff candidate inputs.

No future controls or reviewer labels consulted; do not overwrite completed CSVs.
"""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
PACKET = BASE/'review_packet'
OUT = BASE/'pilot_review'
SEED = 20261002


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mapping = read(PACKET/'organizer_manifest.json')
    contracts = read(PACKET/'scope_contracts_unannotated.json')
    with (PACKET/'reviewer_A.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        pairs = list(reader)
    assert all(not row.get('evidence_label') for row in pairs), 'Prepare from the original unlabelled packet only'
    candidate_domain = {r['candidate_id']:r['domain'] for r in mapping}
    groups = {}
    for row in contracts:
        domain = row['domain']
        if domain=='mongodb_security':
            continue
        key = (domain,' '.join(row['subject'].casefold().split()),' '.join(row['capability'].casefold().split()))
        if key not in groups or row['contract']['cutoff'] < groups[key]['contract']['cutoff']:
            groups[key] = row
    by_domain = defaultdict(list)
    for key,row in sorted(groups.items()):
        if row['candidate_id'] in candidate_domain:
            by_domain[key[0]].append(row)
    chosen = []
    for domain,rows in sorted(by_domain.items()):
        rows = list(rows)
        random.Random(f'{SEED}:{domain}').shuffle(rows)
        chosen.extend(rows[:4])
    candidate_ids = {r['candidate_id'] for r in chosen}
    selected_mapping = [r for r in mapping if r['candidate_id'] in candidate_ids]
    item_ids = {r['item_id'] for r in selected_mapping}
    selected_pairs = [r for r in pairs if r['item_id'] in item_ids]
    assert len(selected_pairs)==len(item_ids)
    assert all(candidate_domain[r['candidate_id']]!='mongodb_security' for r in selected_pairs)
    OUT.mkdir(exist_ok=True)
    for target in [OUT/'reviewer_A.csv',OUT/'reviewer_B.csv']:
        if target.exists():
            with target.open(encoding='utf-8-sig',newline='') as stream:
                assert all(not r.get('evidence_label') for r in csv.DictReader(stream)), 'Refuse to overwrite annotations'
    for reviewer in ['A','B']:
        ordered = list(selected_pairs)
        random.Random(f'{SEED}:reviewer:{reviewer}').shuffle(ordered)
        with (OUT/f'reviewer_{reviewer}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(ordered)
    (OUT/'scope_contracts_unannotated.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'organizer_manifest.json').write_text(json.dumps(selected_mapping,indent=2)+'\n',encoding='utf-8')
    report = {'status':'frozen_unlabelled_pilot_not_evaluation','seed':SEED,
              'selection':'Earliest cutoff per exact normalized subject/capability within domain, then seeded sample up to four per non-MongoDB domain. Retain all prepared evidence pairs.',
              'candidate_counts':{domain:sum(r['domain']==domain for r in chosen) for domain in sorted(by_domain)},
              'candidate_rows':len(chosen),'pairs':len(selected_pairs),
              'unique_evidence_papers':len({r['evidence_paper_id'] for r in selected_pairs}),
              'all_fields_unlabelled':True,'future_controls_consulted':False,
              'statistical_power_justified':False,'independent_annotation_performed':False,
              'source_sha256':{str(p.relative_to(BASE)):digest(p) for p in [PACKET/'organizer_manifest.json',
                               PACKET/'scope_contracts_unannotated.json',PACKET/'reviewer_A.csv']},
              'frozen_manifest_sha256':digest(OUT/'organizer_manifest.json'),
              'script_sha256':digest(__file__)}
    (OUT/'PILOT_PROTOCOL.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','selection']},indent=2))


if __name__=='__main__':
    main()
