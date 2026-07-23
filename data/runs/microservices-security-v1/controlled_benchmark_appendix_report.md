# Controlled Perturbation Benchmark: Appendix Report

Artifact version: `1.0`  
Run: `microservices-security-v1`  
Input summary SHA-256: `39d47dd699fd2f4ef24826ac91e027f9c2e43eaa0031f1d6861b5099ac103bc4`  
Input trials SHA-256: `312c21d6ea6f2bc4b8bcbd5266d62873897b9e54c800c9c158ba5ac86cd975b9`

## Scope

This is a conditional, offline structural-isolation perturbation check of the topology-stage orphan detector. It does not evaluate TABI generation, scientific novelty, evidence support, or research-gap usefulness.

## Analysis unit and labeling

The primary unit is one **target-community by unique selected-edge-mask instance**. Repeated seeds that produce the same selected edge set for the same target, condition, and mask level are deduplicated before the primary summaries and bootstrap intervals. Each targeted-boundary instance induces exactly one positive: its deliberately boundary-masked baseline community. After one-to-one Jaccard matching at the pre-specified threshold, a TP is one output matched to that target; an FN is a target with no such match; and every remaining orphan output is an FP, including an output matched to a non-target baseline community. Metrics pool these event units across unique instances at a mask level. A precision or F1 value is `n/a` when there are no outputs; in negative controls, P/R/F1 are not defined because there are no positive instances.

The intervals below are 2,000-resample percentile bootstraps over unique perturbation instances. They describe this fixed graph and are not population-generalization intervals.

## Primary end-to-end topology-stage measurement

| Masked boundary adjacencies | n | TP | FP | FN | Precision [95% bootstrap CI] | Recall [95% bootstrap CI] | F1 [95% bootstrap CI] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 83 | 0 | 0 | 83 | n/a | 0.000 [0.000, 0.000] | n/a |
| 50% | 86 | 19 | 3 | 67 | 0.864 [0.700, 1.000] | 0.221 [0.140, 0.314] | 0.352 [0.238, 0.462] |
| 75% | 75 | 73 | 6 | 2 | 0.924 [0.869, 0.974] | 0.973 [0.933, 1.000] | 0.948 [0.914, 0.980] |
| 100% | 3 | 3 | 1 | 0 | 0.750 [0.500, 1.000] | 1.000 [1.000, 1.000] | 0.857 [0.667, 1.000] |

End-to-end reruns Louvain after masking and then applies the orphan detector and matching procedure.

## Fixed-membership metric-only measurement

| Masked boundary adjacencies | n | TP | FP | FN | Precision [95% bootstrap CI] | Recall [95% bootstrap CI] | F1 [95% bootstrap CI] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 83 | 0 | 0 | 83 | n/a | 0.000 [0.000, 0.000] | n/a |
| 50% | 86 | 26 | 4 | 60 | 0.867 [0.731, 0.969] | 0.302 [0.209, 0.395] | 0.448 [0.330, 0.556] |
| 75% | 75 | 75 | 8 | 0 | 0.904 [0.852, 0.962] | 1.000 [1.000, 1.000] | 0.949 [0.920, 0.980] |
| 100% | 3 | 3 | 1 | 0 | 0.750 [0.500, 1.000] | 1.000 [1.000, 1.000] | 0.857 [0.667, 1.000] |

Metric-only retains the baseline Louvain membership and reapplies the size/bridge-ratio rule after masking; it isolates the detector metric from partition changes.

## Negative controls

Both controls contain no induced positive. `internal_edge_control` removes the same number of within-target adjacencies while preserving the target boundary; `global_random_edge_control` removes the same number of graph adjacencies without targeting a community boundary. Reported FP outputs are all outputs in these no-positive trials.

### End-to-end controls

| Control | Mask level | n | FP outputs | Any-output rate [95% Wilson CI] |
|---|---:|---:|---:|---:|
| internal_edge_control | 25% | 90 | 0 | 0.000 [0.000, 0.041] |
| internal_edge_control | 50% | 90 | 1 | 0.011 [0.002, 0.060] |
| internal_edge_control | 75% | 90 | 1 | 0.011 [0.002, 0.060] |
| internal_edge_control | 100% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 25% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 50% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 75% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 100% | 90 | 0 | 0.000 [0.000, 0.041] |

### Metric-only controls

| Control | Mask level | n | FP outputs | Any-output rate [95% Wilson CI] |
|---|---:|---:|---:|---:|
| internal_edge_control | 25% | 90 | 0 | 0.000 [0.000, 0.041] |
| internal_edge_control | 50% | 90 | 0 | 0.000 [0.000, 0.041] |
| internal_edge_control | 75% | 90 | 0 | 0.000 [0.000, 0.041] |
| internal_edge_control | 100% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 25% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 50% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 75% | 90 | 0 | 0.000 [0.000, 0.041] |
| global_random_edge_control | 100% | 90 | 0 | 0.000 [0.000, 0.041] |

## Threshold-free structural reference comparisons (metric-only only)

The baseline Louvain membership is fixed; each perturbation has one induced positive among the three prequalified communities. These are descriptive rank comparisons, not calibrated end-to-end detector tests.

These supplemental reference rankings were added after reviewer feedback. They use no fitted cutoff or tuned parameter and do not alter the benchmark's primary end-to-end results.

| Mask level | n | Bridge-ratio AUC / top-1 | Conductance AUC / top-1 | Normalized-cut AUC / top-1 | Size-only AUC / top-1 | Random-score expected AUC / top-1 | Size-only P/R/F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 83 | 0.669 / 0.446 | 0.669 / 0.446 | 0.669 / 0.446 | 0.542 / 0.361 | 0.500 / 0.333 | 0.333 / 1.000 / 0.500 |
| 50% | 86 | 0.886 / 0.674 | 0.886 / 0.674 | 0.868 / 0.674 | 0.523 / 0.349 | 0.500 / 0.333 | 0.333 / 1.000 / 0.500 |
| 75% | 75 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.580 / 0.400 | 0.500 / 0.333 | 0.333 / 1.000 / 0.500 |
| 100% | 3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.500 / 0.333 | 0.500 / 0.333 | 0.333 / 1.000 / 0.500 |

AUC is the within-perturbation, tie-aware probability that the induced target receives a higher score than an eligible negative; top-1 gives fractional credit for ties. The random-score column is the analytic iid-continuous-score expectation, not a noisy seed-dependent draw. `external-edge fraction` is not shown as a separate method because, with the unweighted internal-plus-external denominator used here, it is exactly the production bridge ratio. The size-only reference outputs all three size-gated communities, so it has fixed P=1/3, R=1, F1=1/2.

For every evaluated candidate score instance, vol(S) <= vol(not-S). Therefore conductance equals R_bridge / (2 - R_bridge), a monotonic transform of the bridge ratio; identical rank results are expected in this fixed candidate universe.

The random-score reference is the exact expectation for iid continuous scores independent of the candidate and perturbation: pairwise AUC = 0.5 and top-1 = 1/K for K candidates. An arbitrary seeded random draw is intentionally omitted because it would add Monte-Carlo noise without testing a different detector.

Neither a conductance nor normalized-cut cutoff was pre-specified or calibrated on an independent development set. Selecting either after inspecting these masks would create an optimistic comparison, so both are reported only as threshold-free rank scores.

After Louvain is rerun, a baseline community can split, merge, or disappear; a fair conductance detector would require a separately specified community proposal stage and a held-out calibration protocol.

## Validation

The generator verifies source hashes, seeded and deduplicated trial counts, TP+FP=output count, TP+FN=number of induced-positive units, control accounting, and reconstruction of every unique targeted mask from its recorded metadata.
