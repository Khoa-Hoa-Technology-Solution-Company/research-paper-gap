# Descriptions accompanying the revised submission

These descriptions supplement the manuscript's descriptive captions. The PDF is not
a tagged-accessibility PDF; these texts are available for the publisher's accessibility
workflow. Numbering follows the anonymous submission.

## Figure 1: Pipeline

Five boxes show OpenAlex article retrieval, tagging with 90 concepts, annual concept
and pair counts, structural gaps and features at cutoff T, and a logistic forecaster.
Solid arrows show feature processing. A dashed branch constructs future-window labels;
training sees those labels only for cutoffs whose windows have ended by T.

## Figure 2: Closure over time and expected-co-occurrence quintiles

The left line chart follows the 264 gaps open in 2013 and shows their cumulative
closure share increasing to 86 percent by 2025. The right bar chart shows three-year
closure rates averaged across four test cutoffs, increasing with expected-co-occurrence
quintile. The most expected absent combinations are more likely to cross the lexical
closure threshold. This figure does not measure problem resolution.

## Table 1: Corpus topics

Fifteen rows give OpenAlex primary-topic identifiers, topic names and article counts.
The total is 194,221 articles with abstracts, covering publication years 2010--2025.

## Table 2: Lexicon

Six rows list the concepts assigned to each INISCOM topic area. There are 90 concepts:
39 wireless, 25 industrial and intelligent systems, seven security and privacy, five
hardware/software, 13 signal/data and one quantum concept.

## Table 3: Forecasting results

Rows compare single scores, two multi-feature baselines and the full logistic model.
Columns give AUC for cutoffs 2019--2022 and mean AUC, average precision, P@20 and
Open@20. Full-model mean AUC is 0.809, Adamic--Adar 0.659, and popularity-and-growth
logistic regression 0.776. The reduced logistic model has the highest Open@20.

## Table 4: Bootstrap comparisons

Rows give four cutoff-specific, pointwise 95 percent intervals for learned AUC minus
each comparator. Pair and concept resampling are shown for expected co-occurrence;
concept resampling is shown for Adamic--Adar and the reduced logistic model. Against
Adamic--Adar, the 2021 interval includes zero; the other three are positive, with a
near-zero lower bound in 2019. Only the 2022 interval is positive against the reduced
logistic model. Models are not refitted during resampling.

## Table 5: Ablation and coefficients

The left panel reports mean AUC and average precision after removing or retaining
feature groups. The right panel gives standardised logistic coefficients for the
2022 cutoff. Combining popularity and growth accounts for most of the improvement;
coefficients describe associations and are not causal effects.

## Table 6: Ranked 2022 examples

Two panels show ten highest- and ten lowest-scoring concept pairs with their later
joint-mention article counts. All ten high-scoring pairs cross the three-paper closure
threshold; none of the ten low-scoring pairs does. Counts include incidental mentions
and surveys. Six specific research articles for three pairs are discussed in Section 5.7.

## Table 7: Topic-area closure rates

An upper-triangular matrix gives pooled three-year closure rates and pair--cutoff
observation counts for combinations of six INISCOM areas. Sparse cells show grey
counts. Within-area closure is 30 percent; cross-area closure is 18 percent. Pairs can
recur across cutoffs, and the single quantum concept has no within-area pair.

## Table 8: Calibration

Five score quintiles compare predicted and observed closure shares for the raw logistic
model and a temporally fitted Platt map. A final row compares Brier scores. Calibration
does not improve with the tested map. Raw scores should be used as rankings rather
than literal probabilities.

## Table 9: Definition sensitivity

Nine rows vary support, gap tolerance, closure threshold, horizon, same-sentence labels
or a vocabulary established by 2013. The full model has higher mean AUC than the
strongest single score in each row. The 2013 vocabulary has very few candidates and
closures; its high AUC is unstable and does not establish improved robustness.

## Table 10: Concept-subset sensitivity

Three rows give the median, minimum and maximum of mean AUC across 20 subsets of
60 concepts. Each run first averages four cutoff AUCs. Full-model median is 0.797,
reduced-logistic median 0.766, and retrospectively strongest single-score median 0.651.
The full-model range is 0.737--0.845. These are descriptive ranges, not confidence intervals.

## Table 11: Forward-window ranking

Two panels list the ten highest and ten lowest closure scores among 356 gaps at the
end of 2025 for the 2026--2028 outcome window. The ranking was constructed in October
2026 using no post-2025 publications. It is not a preregistered forecast and has no
complete outcome evaluation. Values are uncalibrated closure scores.
