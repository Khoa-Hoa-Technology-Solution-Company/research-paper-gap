"""Record current source fingerprints; do not compile or export a PDF."""
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
source = (BASE/'main.tex').read_text(encoding='utf-8')
citations = {key for match in re.findall(r'\\cite\{([^}]+)\}', source) for key in match.split(',')}
references = set(re.findall(r'\\bibitem\{([^}]+)\}', source))
labels = re.findall(r'\\label\{([^}]+)\}', source)
refs = set(re.findall(r'\\ref\{([^}]+)\}', source))
assert citations <= references and refs <= set(labels)
assert len(labels) == len(set(labels))
stack = []
for kind, name in re.findall(r'\\(begin|end)\{([^}]+)\}', source):
    if kind == 'begin':
        stack.append(name)
    else:
        assert stack and stack.pop() == name
assert not stack
# Check figure line breaks without pretending to render the document.
figure = source[source.index(r'\begin{picture}'):source.index(r'\end{picture}')]
stacks = re.findall(r'\\shortstack\{([^}]+)\}', figure)
assert len(stacks)==8 and all(r'\\' in block for block in stacks)
assert not re.search(r'(?<!\\)\\(?:empirical|contract|documents|semantic|witness|required|residual|Otherwise)\b',figure)
report = json.loads((BASE/'public_benchmark/results/context24_report.json').read_text(encoding='utf-8'))
assert report['claims'] == 42 and report['papers'] == 31
for method in report['macro_metrics']:
    for metric in ['rouge1_f1','rouge2_f1','rougel_f1']:
        rendered = f"{report['macro_metrics'][method][metric]:.4f}"[1:]
        assert rendered in source, (method, metric, rendered)
primary = json.loads((BASE/'primary_audit/primary_audit_report.json').read_text(encoding='utf-8'))
assert primary['source_records']==325 and primary['scored_evaluations']==7
assert all(row['domain']!='mongodb_security' for row in primary['rows'])
assert primary['all_regenerated_records_equal_archive']
for method in ['esv','bm25']:
    value = f"{primary['macro_proxy_recall'][method]['20']:.4f}"[1:]
    assert value in source
properties = json.loads((BASE/'controller_properties_report.json').read_text(encoding='utf-8'))
assert properties['status']=='passed' and properties['enumerated_two_record_protocol_inputs']==8192
assert '8,192' in source and '180' in source
pilot = json.loads((BASE/'pilot_review/PILOT_PROTOCOL.json').read_text(encoding='utf-8'))
assert pilot['candidate_rows']==12 and pilot['pairs']==53 and pilot['all_fields_unlabelled']
sanity = {'status':'static_checks_passed_not_compilation',
          'citation_keys_resolved':len(citations),'reference_labels_resolved':len(refs),
          'environment_nesting_check':True,'context_result_rounding_checks':18,
          'figure_line_break_checks':8,'primary_audit_result_checks':True,
          'property_enumeration_and_pilot_count_checks':True,
          'native_compiler_status':'unavailable_platform_directories',
          'source_sha256':hashlib.sha256((BASE/'main.tex').read_bytes()).hexdigest()}
(BASE/'source_consistency_report.json').write_text(json.dumps(sanity,indent=2)+'\n',encoding='utf-8')
qa = '''# Verification status — current revision, 1 October 2026

## Current manuscript

- Edited the existing open main.tex in place; kept its document editor.
- Added methodological-context retrieval, six baseline/ablation arms, a public expert-labelled evaluation, matched budgets, paper-level descriptive uncertainty, budget sensitivity, and negative cross-lab transfer results.
- Applied the pasted review to the current source: control-first positioning, a native LaTeX system figure, conditional control properties, and primary/secondary audit separation. The pasted review's 11-page PDF is an older export.
- Native compile_latex_document returned compile-failed: Unable to find standard directories for platform.
- Current compilation, page count, and rendered layout are UNVERIFIED. No separate PDF compilation/export was performed in this revision.
- Existing main.pdf and output/pdf/ESV-Gap_AMI2026_Scoped_Draft.pdf are the earlier 11-page draft and STALE relative to current source. Previous visual QA applies only to that version.
- Static citation/reference, environment-nesting and rounded-result checks pass. This is not a substitute for LaTeX compilation or rendered QA.

## New evidence

- Context24 author dataset at revision 457d3b5cb4bb8ade34e37458f4900c6eae0959bb, CC BY 4.0; six downloaded files fingerprinted.
- All 42 expert-labelled Task-2 examples/31 papers; no missing full texts. Held-paper claims and full text excluded from prior training.
- Same windows/512-word limit in all six arms; 252 predictions separately verified for source offsets, quote/budget integrity and aggregate arithmetic.
- Claim-macro ROUGE-L: BM25 0.1805, TF-IDF 0.1857, prior-only 0.2045, BM25-diverse 0.1805, hybrid 0.2102, hybrid-diverse 0.2122.
- Proposed-minus-BM25 paper-macro difference 0.0393, descriptive paired-paper interval [0.0131, 0.0666]; different from claim-macro difference 0.0317.
- Paired-paper intervals against prior-only/nondiversified hybrid include zero. No established incremental diversification advantage.
- Cross-lab transfer to MegaCog: 0.1504 versus BM25 0.1714. No domain-portability claim.
- Post-hoc 256/768-word sensitivities retain higher aggregate ROUGE-L than BM25; original weights and budget retained.
- Optimized LCS agrees with conventional DP on ten bounded prefixes. Six retrieval unit tests pass.
- Official test has 109 claims without gold contexts, and is not scored. Metrics differ from official stemmed snippet-wise ROUGE/BERTScore. No leaderboard/SOTA claim.

## Preserved evidence and boundaries

- Original source PDF and historical material preserved.
- Four audit corpora: 378 keyed records/3,188 triples; earlier recount and fingerprint checks retained.
- Twenty-five constructed controller tests pass; they check policy behavior, not scientific efficacy.
- Bounded property verification passes on 8,192 constructed two-record/protocol inputs, including 180 cross-record union states lacking an individual witness. Unique-key insertion, semantic permutation, repeatability, threshold strengthening and duplicate-key counterexample checks pass. This is not full state-space model checking or human efficacy evaluation.
- Historical ESV candidate yield 39 rows, all explicit limitations; typed branch zero. Silver-control R20 0.0139 versus BM25-LACKS 0.0933; non-independent controls and MongoDB provenance issue disclosed.
- The primary diagnostic analysis now regenerates candidates, controls and rankings in three domains with MongoDB excluded: 325 records/2,545 triples, eight completed/seven scored folds, 61 controls/29 distinct IDs, 38 candidate rows. Regenerated records equal archived counterparts; ESV proxy R20 0.0179 versus BM25 0.0485. Exclusion is post hoc, not provenance repair or expert validation of remaining domains.
- Review preparation remains 188 pairs/39 candidate rows/97 evidence-paper IDs. Human fields remain blank.
- A frozen pilot selects 12 candidate rows/53 pairs/40 evidence-paper IDs without future controls. Both reviewer packets remain blank. Scorer with the pilot manifest was checked to reject these unlabelled inputs; it created no agreement score. The user confirmed no reviewers/labels are currently available.
- No controller closure precision, witness-document recall, new independent annotation, gap-discovery efficacy, public artifact release, external submission or independent reproduction claimed.
- Read SELF_REVIEW_VI.md for the current readiness assessment and remaining evidence requirements.
'''
(BASE/'QA_STATUS.md').write_text(qa,encoding='utf-8')
manifest_path = BASE/'artifact_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['previous_export_pages'] = manifest.get('pages') or manifest.get('previous_export_pages',11)
manifest.update(pages=None,status='current_source_revised_native_compile_unavailable_not_submission_certified',
                current_source_compiled=False,previous_pdf_exports_stale=True,
                public_expert_labels_reused=True,new_human_annotation_performed=False)
paths = [entry['path'] for entry in manifest['files']]
paths += [str(p.relative_to(ROOT)) for p in [BASE/'SELF_REVIEW_VI.md',BASE/'source_consistency_report.json',
          ROOT/'ESV-Gap/src/context_retrieval.py',ROOT/'ESV-Gap/tests/test_context_retrieval.py',
          BASE/'public_benchmark/CONTEXT24_PROTOCOL.md',BASE/'public_benchmark/run_context24.py',
          BASE/'public_benchmark/check_context24.py',BASE/'public_benchmark/check_transfer.py',
          BASE/'public_benchmark/context24/DOWNLOAD_MANIFEST.json',
          BASE/'public_benchmark/results/context24_report.json',
          BASE/'public_benchmark/results/verification_and_sensitivity.json',
          BASE/'public_benchmark/results/transfer_and_ablations.json',
          BASE/'replay_primary_audit.py',BASE/'primary_audit/primary_audit_report.json',
          BASE/'verify_controller_properties.py',BASE/'controller_properties_report.json',
          BASE/'prepare_pilot.py',BASE/'pilot_review/PILOT_PROTOCOL.json',
          BASE/'pilot_review/README_VI.md',BASE/'pilot_review/organizer_manifest.json',
          BASE/'pilot_review/scope_contracts_unannotated.json',BASE/'pilot_review/reviewer_A.csv',
          BASE/'pilot_review/reviewer_B.csv',BASE/'REVIEW_RESPONSE_VI.md']]
manifest['files'] = []
for relative in dict.fromkeys(paths):
    path = ROOT/relative
    content = path.read_bytes()
    entry = {'path':relative,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    if path.suffix == '.pdf':
        entry['stale_relative_to_current_source'] = True
    manifest['files'].append(entry)
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(sanity,indent=2))
