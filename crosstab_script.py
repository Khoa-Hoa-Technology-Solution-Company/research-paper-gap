import json
import hashlib
from pathlib import Path
from collections import Counter

run_dir = (
    Path(__file__).resolve().parent
    / "ESV-Gap"
    / "runs"
    / "security_of_mongodb_20260806_135447"
    / "outputs"
)

# 1. Get the hash of gap_validation_audit.json
audit_path = run_dir / "gap_validation_audit.json"
h = hashlib.sha256()
h.update(audit_path.read_bytes())
print(f"gap_validation_audit.json SHA-256: {h.hexdigest()}")

# 2. Cross-tabulate gate outcome vs reviewer outcome
audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
gate_decisions = {}
for item in audit_data["candidates"]:
    gate_decisions[item["description"]] = item["validation"]["status"]

expert_reviews = json.loads((run_dir / "expert_reviews.json").read_text(encoding="utf-8"))
ranked_gaps = json.loads((run_dir / "gaps_ranked_top.json").read_text(encoding="utf-8"))

crosstab = Counter()
for i, gap in enumerate(ranked_gaps):
    desc = gap.get("description", "")
    review_key = f"review_{i+1}"
    decision = expert_reviews.get("reviews", {}).get(review_key, "Unknown")
    
    gate_status = gate_decisions.get(desc, "Not found")
    crosstab[(gate_status, decision)] += 1

print("\nCross-tab:")
gate_statuses = ["review_required", "rejected", "automatically_eligible"]
decisions = ["Accept", "Modify", "Reject", "Pending"]
print(f"{'Gate outcome':<25} | " + " | ".join(f"{d:<8}" for d in decisions))
print("-" * 75)
for gs in gate_statuses:
    row = []
    for d in decisions:
        row.append(f"{crosstab[(gs, d)]:<8}")
    print(f"{gs:<25} | " + " | ".join(row))
