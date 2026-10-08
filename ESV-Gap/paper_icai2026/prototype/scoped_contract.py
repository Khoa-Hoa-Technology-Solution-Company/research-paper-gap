"""Research prototype: scoped closure over supplied, checked evidence annotations.

This is NOT a literature extractor, novelty detector, or calibrated verifier.
Matching annotations and verified_span are trusted inputs requiring human review.
All requirements must hold in a single experiment: never combine unrelated runs.
"""
from dataclasses import dataclass
from datetime import date
from math import isfinite


@dataclass(frozen=True)
class Requirement:
    name: str
    op: str
    bound: float
    unit: str


@dataclass(frozen=True)
class Hypothesis:
    task: str
    conditions: frozenset[str]
    requirements: tuple[Requirement, ...]
    cutoff: date


@dataclass(frozen=True)
class Evidence:
    paper_id: str
    experiment_id: str
    available_on: date
    task: str
    conditions: frozenset[str]
    metrics: dict[str, tuple[float, str]]
    source_text: str
    span: str
    verified_span: bool


def evaluate(hypothesis: Hypothesis, records: list[Evidence], *, search_complete: bool):
    """Return a bounded disposition and per-record residual obligations.

    `search_complete` means the declared protocol completed, NOT world coverage.
    Absence, missing values and metric failures never establish scientific novelty.
    """
    if not hypothesis.task or not hypothesis.requirements:
        raise ValueError("A task and at least one predeclared requirement are needed")
    names = [r.name for r in hypothesis.requirements]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate requirement names")
    for requirement in hypothesis.requirements:
        if requirement.op not in {"<=", ">="} or not isfinite(requirement.bound):
            raise ValueError("Requirements need finite bounds and <= or >=")
    traces = []
    for record in records:
        if record.available_on > hypothesis.cutoff:
            traces.append({"paper_id": record.paper_id, "status": "excluded_future"})
            continue
        if (not record.paper_id or not record.experiment_id or not record.verified_span
                or not record.span.strip() or record.span not in record.source_text):
            traces.append({"paper_id": record.paper_id, "status": "unchecked_annotation"})
            continue
        if record.task != hypothesis.task:
            traces.append({"paper_id": record.paper_id, "status": "different_task"})
            continue
        missing_conditions = sorted(hypothesis.conditions - record.conditions)
        residuals = []
        for requirement in hypothesis.requirements:
            measurement = record.metrics.get(requirement.name)
            if measurement is None:
                state = "unreported"
            else:
                value, unit = measurement
                if unit != requirement.unit or not isfinite(value):
                    state = "incomparable"
                else:
                    satisfied = (value <= requirement.bound if requirement.op == "<="
                                 else value >= requirement.bound)
                    state = "satisfied" if satisfied else "not_satisfied"
            residuals.append({"requirement": requirement.name, "state": state})
        closed = not missing_conditions and all(r["state"] == "satisfied" for r in residuals)
        traces.append({"paper_id": record.paper_id, "experiment_id": record.experiment_id,
                       "status": "closes_scope" if closed else "partial",
                       "missing_conditions": missing_conditions, "requirements": residuals})
    statuses = {trace["status"] for trace in traces}
    if "closes_scope" in statuses:
        disposition = "CLOSED_WITHIN_SCOPE"
    elif not search_complete or "unchecked_annotation" in statuses:
        disposition = "REVIEW_REQUIRED"
    elif "partial" in statuses:
        disposition = "PARTIAL_EVIDENCE"
    else:
        disposition = "OPEN_FOR_REVIEW"
    return {"disposition": disposition, "traces": traces,
            "novelty_established": False, "search_protocol_complete": search_complete}
