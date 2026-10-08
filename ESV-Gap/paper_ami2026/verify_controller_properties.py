"""Bounded enumeration of conditional controller properties on constructed inputs.

This is software/formalization verification, not a real-literature experiment.
Expected witnesses use known symbolic facet states, not extracted text labels.
"""
import copy
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
sys.path.insert(0,str(PROJECT))
from src.scoped_verification import evaluate_scope


def fixture():
    contract = {'task':'synthetic task','domain':'synthetic domain','scope_reviewed':True,
                'cutoff':'2024-12-31','conditions':['condition A'],
                'requirements':[{'name':'quality','op':'>=','bound':.9,'unit':'fraction'},
                                {'name':'latency','op':'<=','bound':10,'unit':'ms'}]}
    ledger = {'cutoff':contract['cutoff'],'required_probe_ids':['p'],
              'probes':[{'probe_id':'p','provider':'constructed','query':'symbolic probe',
                         'snapshot_id':'constructed','status':'completed','pagination_complete':True,
                         'semantic_review_complete':True}]}
    source = 'synthetic task synthetic domain condition A quality 0.95 latency 8'
    def quoted(value,span):
        return {'value':value,'span':span,'reviewed':True}
    record = {'paper_id':'constructed-paper','experiment_id':'A','available_on':'2024-01-01',
              'metadata_reviewed':True,'experiment_reviewed':True,'source_text':source,
              'facets':{'task':quoted(contract['task'],'synthetic task'),
                        'domain':quoted(contract['domain'],'synthetic domain')},
              'conditions':{'condition A':quoted(True,'condition A')},
              'measurements':{'quality':{**quoted(.95,'quality 0.95'),'unit':'fraction'},
                              'latency':{**quoted(8,'latency 8'),'unit':'ms'}}}
    return contract,ledger,record


def make_record(template,state,eid):
    task,domain,condition,quality,latency,reviewed = state
    row = copy.deepcopy(template)
    row['experiment_id'] = eid
    row['facets']['task']['value'] = 'synthetic task' if task else 'other task'
    row['facets']['domain']['value'] = 'synthetic domain' if domain else 'other domain'
    row['conditions']['condition A']['value'] = condition
    row['measurements']['quality']['value'] = .95 if quality else .5
    row['measurements']['latency']['value'] = 8 if latency else 20
    row['experiment_reviewed'] = reviewed
    # Values are trusted symbolic inputs; changing them is not semantic annotation.
    return row


def main():
    contract,ledger,template = fixture()
    states = list(itertools.product([False,True],repeat=6))
    incomplete = copy.deepcopy(ledger)
    incomplete['probes'][0]['status'] = 'timeout'
    dispositions = Counter()
    checked = 0
    no_composition_cases = 0
    for left,right in itertools.product(states,repeat=2):
        rows = [make_record(template,left,'A'),make_record(template,right,'B')]
        has_witness = all(left) or all(right)
        if (left[-1] and right[-1] and not has_witness and
                all(a or b for a,b in zip(left[:-1],right[:-1]))):
            no_composition_cases += 1
        for protocol in [ledger,incomplete]:
            result = evaluate_scope(contract,rows,protocol)
            assert (result['disposition']=='CLOSED_WITHIN_SCOPE') == has_witness
            assert result['novelty_established'] is False
            if protocol is incomplete and not has_witness:
                assert result['action']=='ABSTAIN_AND_ESCALATE'
            dispositions[result['disposition']] += 1
            checked += 1
    # Insertion and permutation: semantic disposition/action invariant; traces/hash may differ.
    witness = make_record(template,(True,)*6,'witness')
    insertion_cases = permutation_cases = repeat_cases = 0
    for state in states:
        row = make_record(template,state,'added')
        base = evaluate_scope(contract,[row],ledger)
        added = evaluate_scope(contract,[row,witness],ledger)
        assert added['disposition']=='CLOSED_WITHIN_SCOPE'
        insertion_cases += 1
        reversed_result = evaluate_scope(contract,[witness,row],ledger)
        assert (added['disposition'],added['action']) == (reversed_result['disposition'],reversed_result['action'])
        permutation_cases += 1
        assert added == evaluate_scope(copy.deepcopy(contract),copy.deepcopy([row,witness]),copy.deepcopy(ledger))
        repeat_cases += 1
        if base['disposition']=='CLOSED_WITHIN_SCOPE':
            assert added['disposition']==base['disposition']
    # Strengthened valid requirements cannot create closure from a non-witness.
    strengthened = 0
    for quality,latency in itertools.product([.5,.9,.95,1.],[5,8,10,15]):
        row = copy.deepcopy(witness)
        row['measurements']['quality']['value'] = quality
        row['measurements']['latency']['value'] = latency
        original = evaluate_scope(contract,[row],ledger)
        for quality_bound,latency_bound in itertools.product([.9,.95,1.],[5,8,10]):
            tighter = copy.deepcopy(contract)
            tighter['requirements'][0]['bound'] = quality_bound
            tighter['requirements'][1]['bound'] = latency_bound
            strict = evaluate_scope(tighter,[row],ledger)
            if strict['disposition']=='CLOSED_WITHIN_SCOPE':
                assert original['disposition']=='CLOSED_WITHIN_SCOPE'
            strengthened += 1
    duplicate = copy.deepcopy(witness)
    duplicate['measurements']['latency']['value'] = 20
    assert evaluate_scope(contract,[witness,duplicate],ledger)['disposition']=='REVIEW_REQUIRED'
    result = {'status':'passed','constructed_inputs_only':True,'real_literature_effectiveness_measured':False,
              'enumerated_two_record_protocol_inputs':checked,'states_per_record':len(states),
              'disposition_counts':dict(dispositions),'cross_record_union_without_single_witness_states':no_composition_cases,
              'unique_key_insertion_checks':insertion_cases,'semantic_permutation_checks':permutation_cases,
              'repeatability_checks':repeat_cases,'threshold_strengthening_checks':strengthened,
              'duplicate_key_nonmonotonicity_counterexample_checked':True,
              'annotation_semantics_assumed':True,
              'sha256':{'controller':hashlib.sha256((PROJECT/'src/scoped_verification.py').read_bytes()).hexdigest(),
                        'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (BASE/'controller_properties_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
