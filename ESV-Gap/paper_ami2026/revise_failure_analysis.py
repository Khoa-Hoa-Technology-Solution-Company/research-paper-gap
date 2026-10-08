"""Apply the primary/secondary diagnostic boundary in the existing manuscript."""
from pathlib import Path

path = Path(__file__).resolve().parent/'main.tex'
source = path.read_text(encoding='utf-8')
start = source.index(r'\section{Retrospective Audit Protocol}')
end = source.index(r'\section{Implemented Extension and Verification Status}')
replacement = r'''\section{Retrospective Failure Analysis}
\subsection{Primary Diagnostic Scope and Protocol}
This analysis explains why proxy matching cannot authorize suppression; it evaluates the earlier retriever, not the new controller. The preserved four-domain archive contains 378 keyed records and 3,188 triples. One matched MongoDB record has unresolved source identity and publication year, so our primary diagnostic analysis excludes the entire 53-record MongoDB domain. This exclusion was decided after inspecting archived results. It is not a preregistered sampling decision, provenance repair, or certification that the remaining domains are clean.

We rerun candidate generation, silver-control construction, and confidence/BM25 rankings on 325 records and 2,545 triples: 150 IoT intrusion-detection, 143 microservice-security, and 32 handwritten-mathematical-expression papers. Cutoffs are 2022, 2023, and 2024, requiring at least ten records on each side. Eight folds qualify; seven have positive controls. Regenerated candidates, controls, and metrics equal archived outputs for every retained fold. Input, configuration, and code hashes are checked; historical files remain unchanged.

Pre-cutoff triples generate candidates; future triples generate silver controls through limitation-plus-action correspondence or newly instantiated typed relations. These are not expert-confirmed research questions or resolutions. Publication years are stored metadata, not reconstructed first-online dates. Support-ID checks exclude future candidate sources but do not reconstruct historical model training or search indexes.

Comparators rank direct \texttt{LACKS} triples by extractor confidence or BM25 over subject/capability text ($k_1=1.2$, $b=0.75$). BM25's domain-label query plus ``limitation unresolved challenge'' was chosen post hoc. It reranks the extracted pool and is not an independent raw-text discovery baseline. Matching uses capability token similarity at least 0.62 and subject similarity at least 0.25, or capability similarity at least 0.82; similarity combines containment and Jaccard overlap. Proxy recall counts matched positive controls under the same candidate budget, averaged across the seven scored folds.

\subsection{Diagnostic Findings and Excluded Secondary Results}
\begin{table}[t]
\caption{Primary three-domain diagnostic audit, seven scored folds. Controls are silver and non-independent. Scores concern the earlier candidate retriever, not grounding retrieval or scope-controller accuracy.}
\label{tab:macro}
\centering\small
\begin{tabular}{lrrrr}
\toprule
Ranking & R@5 & R@10 & R@20 & Matches@20 \\
\midrule
Earlier ESV-Gap & 0 & 0 & .0179 & 1/61 \\
Confidence--LACKS & 0 & 0 & 0 & 0/61 \\
BM25--LACKS & 0 & .0179 & .0485 & 4/61 \\
\bottomrule
\end{tabular}
\end{table}

The 61 control rows contain 29 distinct control IDs within domain; repeated cutoffs share papers. The retriever produces 38 candidate rows, all explicit limitations, while its typed evidence-map branch yields zero. The direct-limitation pool contains 396 rows across folds. Three folds yield no ESV-Gap candidates. Table~\ref{tab:macro} shows weak candidate utility under the proxy; it supplies no positive evidence of graph-specific discovery value.

The four BM25 matches are examined at abstract level with AI assistance. The IoT adversarial-attack match concerns a cybersecurity synthesis~\cite{jayawardena2026}; a microservice match concerns a systematic mapping study~\cite{rahaman2023}; and an API/service-mesh match concerns a Kubernetes overview~\cite{kampa2024}. None of those abstracts reports a matched same-scope resolution experiment. The remaining match reports quantitative DoS detection~\cite{olaya2024}, but its candidate, ``Retrofitting--Security,'' lacks that task and required conditions. These observations are not independent labels or four verified false positives. Full texts may contain evidence absent from abstracts.

For transparent secondary accounting, the original four-domain archive retains ten executable folds, nine scored folds, 64 control rows, and 39 ESV-Gap candidate rows. Its macro R20 is 0.0139 for ESV-Gap and 0.0933 for BM25--LACKS, with five BM25 matches; the fifth is the disputed MongoDB record. These values are excluded from the primary table. The three-domain score ordering persists without that domain. Exclusion addresses its influence on the primary comparison, while its identity/date and all silver labels still need verification. The regenerated three-domain analysis supersedes the earlier aggregate-only omission check.

'''
source = source[:start]+replacement+source[end:]
source = source.replace('The audit uses four retained project corpora of unequal size and noisy extracted controls. Recorded years are a coarse historical proxy, and the MongoDB record has an unresolved provenance issue. Repeated folds reuse papers; no independence-based confidence intervals or significance tests are justified. The BM25 limitation comparator and query sensitivity were post hoc.',
    'The primary audit uses three retained corpora of unequal size and noisy silver controls; the fourth domain is excluded post hoc because of unresolved provenance. Recorded years remain a coarse historical proxy. Repeated folds reuse papers; no independence-based confidence intervals or significance tests are justified. The BM25 limitation comparator and domain exclusion were post hoc.')
source = source.replace(r'The aggregate values in Tables~\ref{tab:folds}--\ref{tab:macro} can be checked from those retained reports; full-pipeline reproduction additionally requires the original inputs and environment.',
    r'Table~\ref{tab:macro} is regenerated from the retained inputs, and its local report includes per-fold metrics and code/environment hashes; reproducing it requires those inputs and dependencies.')
source = source.replace('The four-corpus diagnostic audit exposes weak earlier candidate availability and unreliable closure proxies. Twenty-five policy checks and 188 prepared evidence pairs support continued controller assessment,',
    'The primary three-domain failure analysis exposes weak earlier candidate availability and unreliable closure proxies. Conditional control properties, constructed checks, and a frozen unlabelled pilot support continued controller assessment,')
source = source.replace('across the four audit domains without evaluation.', 'across the audit domains without evaluation.')
source = source.replace(r'\clearpage'+'\n'+r'\begin{thebibliography}{18}',r'\begin{thebibliography}{17}')
old = r'\bibitem{singhmongo2024}'
if old in source:
    a = source.index(old)
    b = source.index('\n',a)
    source = source[:a]+source[b+1:]
path.write_text(source,encoding='utf-8')
print('Updated primary/secondary failure analysis in existing main.tex.')
