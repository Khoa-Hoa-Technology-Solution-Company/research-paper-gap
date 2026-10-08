# Certified research gaps under corpus incompleteness

## Setting

* A question (candidate gap) `c` is **CLOSED** if the literature contains a witness document for it, **OPEN** otherwise.
* The system only sees a local corpus `C_p`: every witness document of the full literature `C` is missing
  independently with probability `p` (the *incompleteness rate*), i.e. missing-completely-at-random over witnesses.
* A closure score `s(c; K)` is computed against a corpus `K`. The decision rule is
  `OPEN  iff  s(c; C_p) < tau`, otherwise CLOSED (or ABSTAIN in the three-way version).
* **False novelty** = declaring OPEN for a CLOSED question. Target: `P(OPEN | CLOSED) <= alpha`.

**Monotone score.** `s` is monotone if `K' ⊆ K  =>  s(c; K') <= s(c; K)` for every `c`.
Max-over-available-documents scores (co-mention, BM25/dense top-1, ESV-Scope over the full
candidate pool) are monotone; top-k-truncated or learned aggregates are in general not
(verified empirically: 0 violations vs 15–46 / 809 claims under nested deletions).

## Conformal threshold

Given `n` calibration questions known to be CLOSED with scores `S_1..S_n`, let
`k = floor(alpha (n+1))` and `tau = S_(k)` (the k-th smallest; `tau = -inf` if `k = 0`).

## Theorem 1 (in-situ calibration; p unknown)

If the calibration CLOSED questions are scored against the **same** incomplete corpus `C_p` as
the test question, and their labels do not come from `C_p` (e.g. expert / external labels), then
for an exchangeable CLOSED test question, `P(s(c; C_p) < tau) <= alpha`, **for any unknown p**.

*Proof.* Under MCAR deletion the calibration and test (question, deletion pattern of its
witnesses) pairs are exchangeable, so the scores are exchangeable; the rank of the test score
among the n+1 scores is uniform, hence `P(S_test < S_(k)) <= k/(n+1) <= alpha`. ∎

## Theorem 2 (transfer calibration with an upper bound on p)

Suppose the calibration set is scored on a **complete** benchmark, and we simulate deletion of
its witnesses at a rate `p_hat`. If `s` is monotone and `p_hat >= p`, then
`P(s(c; C_p) < tau_{p_hat}) <= alpha`.

*Proof.* Couple the two corpora: delete each witness with probability `p` to obtain `C_p`, then
delete each surviving witness with probability `(p_hat - p)/(1 - p)`; the result `C_{p_hat}` has
the correct law and `C_{p_hat} ⊆ C_p`. By monotonicity `s(c; C_{p_hat}) <= s(c; C_p)`, so
`P(s(c; C_p) < tau) <= P(s(c; C_{p_hat}) < tau) <= alpha`, the last step by exchangeability of the
test score at rate `p_hat` with the calibration scores at rate `p_hat` (Theorem 1). ∎

## Corollary (estimated p)

Draw `m` reference documents known to be relevant (e.g. witnesses of known-closed questions, or a
systematic review's reference list) and observe `x` of them missing from the local corpus. Let
`p_hat` be the one-sided `(1 - delta)` Clopper–Pearson upper bound for `x/m`. Then
`P(false novelty) <= alpha + delta`.

## Price of certainty

Power = `P(OPEN | OPEN)`. For `p_hat = 1` (no knowledge of coverage) any monotone evidence score
must place `tau` below the scores of CLOSED questions whose witnesses are *all* missing, which are
evidence-wise indistinguishable from OPEN questions; power then collapses (≈ 6–13 % on SciFact).
Estimating coverage (Corollary) recovers power while keeping the guarantee.
