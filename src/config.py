import os
import re
import sys
from dotenv import load_dotenv

# Reconfigure stdout/stderr encoding on Windows to prevent UnicodeEncodeErrors
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Fallback for Python < 3.7

# Load environment variables from .env
load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None
LLM_DELAY = float(os.getenv("LLM_DELAY", "2.0"))
# Optional provider/model fields are recorded verbatim in *future* call
# manifests.  They are deliberately nullable: guessing a revision or API date
# would make a provenance record less, not more, reliable.
LLM_MODEL_REVISION = os.getenv("LLM_MODEL_REVISION", "").strip() or None
LLM_MODEL_RELEASE_DATE = os.getenv("LLM_MODEL_RELEASE_DATE", "").strip() or None
LLM_API_VERSION = os.getenv("LLM_API_VERSION", "").strip() or None
LLM_API_DATE = os.getenv("LLM_API_DATE", "").strip() or None
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "7"))
LLM_RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "5.0"))
LLM_RETRY_NON_RATE_LIMIT_DELAY_SECONDS = float(
    os.getenv("LLM_RETRY_NON_RATE_LIMIT_DELAY_SECONDS", "3.0")
)
_llm_top_p = os.getenv("LLM_TOP_P", "").strip()
LLM_TOP_P = float(_llm_top_p) if _llm_top_p else None
# The default intentionally stores hashes, rather than raw prompts/responses,
# because source-derived content may be sensitive or redistribution-restricted.
# Set to "store" only for an approved rerun where retaining raw model output is
# allowed. "none" records only the policy and no response hash.
LLM_RAW_RESPONSE_POLICY = os.getenv("LLM_RAW_RESPONSE_POLICY", "hash-only").strip().lower()

# Semantic Scholar Configuration
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None
SEMANTIC_SCHOLAR_API_BASE_URL = os.getenv(
    "SEMANTIC_SCHOLAR_API_BASE_URL", "https://api.semanticscholar.org/graph/v1"
).rstrip("/")
SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS", "30")
)
SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS = float(
    os.getenv("SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS", "3.0")
)

# Version identifiers for prompts and deterministic procedures.  They are
# recorded with future run manifests and should be bumped when wording or logic
# changes in a way that can change outputs.
QUERY_EXPANSION_PROMPT_VERSION = "query-expansion-v1-provenance"
RELEVANCE_SCREENING_PROMPT_VERSION = "relevance-screening-v1-provenance"
CLOSURE_QUERY_VARIANT_PROMPT_VERSION = "closure-query-variants-v1-provenance"
CLOSURE_DETERMINISTIC_VARIANT_VERSION = "closure-deterministic-variants-v1"

# Business logic constraints for Software Engineering
ENTITY_TYPES = ["METHOD", "DATASET", "METRIC", "CONCEPT", "FINDING", "TOOL"]
RELATION_TYPES = [
    "USES", "IMPROVES", "ADDRESSES", "EVALUATES_ON", "PRODUCES",
    "CONTRADICTS", "EXTENDS", "LACKS", "COMBINES", "APPLIED_TO"
]

# Topology / Filtering Thresholds
RELEVANCE_THRESHOLD = 0.7
TRIPLE_CONFIDENCE_THRESHOLD = 0.3
# This is a model-reported extraction score used only for filtering until a
# human calibration study establishes whether it is probabilistic confidence.
TRIPLE_EXTRACTION_PROMPT_VERSION = "triple-extraction-v2-provenance"
TABI_INFERENCE_PROMPT_VERSION = "tabi-inference-v3-feasibility-labels"
FUZZY_MATCH_THRESHOLD = 85
COSINE_SIMILARITY_THRESHOLD = 0.85
LOUVAIN_MIN_SIZE_RATIO = 0.05
LOUVAIN_MAX_BRIDGE_RATIO = 0.10
LOUVAIN_RANDOM_STATE = 42
LOUVAIN_RESOLUTION = 1.0
TEMPORAL_DECAY_THRESHOLD = 0.30
TEMPORAL_MIN_EVENTS = 5
GENERIC_TEMPORAL_NODES = {
    "analysis", "conceptual framework", "framework", "knowledge gaps",
    "literature review", "proposed framework", "proposed methodology",
    "research gap", "research gaps", "review", "systematic review",
}

# Candidate-screening configuration. A cluster pair must pass this deterministic
# compatibility gate before it can be sent to the LLM for hypothesis generation.
COMPATIBILITY_THRESHOLD = 0.20
COMPATIBILITY_TOP_NODES = 12
COMPATIBILITY_MIN_CONTENT_OVERLAP = 0.05
SEMANTIC_COMPATIBILITY_THRESHOLD = 0.42
SEMANTIC_COMPATIBILITY_MODEL = os.getenv(
    "SEMANTIC_COMPATIBILITY_MODEL", "all-MiniLM-L6-v2"
)
TEMPORAL_MIN_DISTINCT_YEARS = 4
TEMPORAL_SIGNIFICANCE_LEVEL = 0.05
# Screen all eligible concepts before applying Benjamini--Hochberg correction.
# Keeping the FDR level separate from the raw-p reporting level makes the
# multiple-testing decision explicit in the run configuration.
TEMPORAL_FDR_SIGNIFICANCE_LEVEL = float(
    os.getenv("TEMPORAL_FDR_SIGNIFICANCE_LEVEL", "0.05")
)
# The recent period is defined by the last N *covered* complete years. Years
# with no relation-event coverage are treated as missing rather than as zero
# concept activity.
TEMPORAL_RECENT_WINDOW_YEARS = int(
    os.getenv("TEMPORAL_RECENT_WINDOW_YEARS", "2")
)
# Require a strictly negative Sen slope by default. A future preregistered run
# can set a positive minimum effect size without changing the implementation.
TEMPORAL_MIN_NEGATIVE_SLOPE = float(
    os.getenv("TEMPORAL_MIN_NEGATIVE_SLOPE", "0.0")
)
# The pilot was collected during calendar year 2026, so 2026 is incomplete and
# excluded from temporal inference. Override it with a later completed year in
# a future run.
TEMPORAL_CUTOFF_YEAR = int(os.getenv("TEMPORAL_CUTOFF_YEAR", "2025"))

if not 0 < TEMPORAL_SIGNIFICANCE_LEVEL <= 1:
    raise ValueError("TEMPORAL_SIGNIFICANCE_LEVEL must be in (0, 1].")
if not 0 < TEMPORAL_FDR_SIGNIFICANCE_LEVEL <= 1:
    raise ValueError("TEMPORAL_FDR_SIGNIFICANCE_LEVEL must be in (0, 1].")
if TEMPORAL_RECENT_WINDOW_YEARS < 1:
    raise ValueError("TEMPORAL_RECENT_WINDOW_YEARS must be at least 1.")
if TEMPORAL_MIN_NEGATIVE_SLOPE < 0:
    raise ValueError("TEMPORAL_MIN_NEGATIVE_SLOPE must be non-negative.")
if LLM_MAX_RETRIES < 1:
    raise ValueError("LLM_MAX_RETRIES must be at least 1.")
if LLM_RETRY_BASE_DELAY_SECONDS < 0 or LLM_RETRY_NON_RATE_LIMIT_DELAY_SECONDS < 0:
    raise ValueError("LLM retry delays must be non-negative.")
if LLM_TOP_P is not None and not 0 < LLM_TOP_P <= 1:
    raise ValueError("LLM_TOP_P must be in (0, 1] when configured.")
if LLM_RAW_RESPONSE_POLICY not in {"hash-only", "store", "none"}:
    raise ValueError("LLM_RAW_RESPONSE_POLICY must be one of: hash-only, store, none.")
if SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS <= 0:
    raise ValueError("SEMANTIC_SCHOLAR_REQUEST_TIMEOUT_SECONDS must be positive.")
if SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS < 0:
    raise ValueError("SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS must be non-negative.")

# File Paths
# A domain experiment can be isolated by setting KG_TABI_RUN_ID, e.g.
# KG_TABI_RUN_ID=microservices-security-v1. The legacy artifacts remain in
# data/, while new runs are written beneath data/runs/<run-id>/.
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RUN_ID = os.getenv("KG_TABI_RUN_ID", "").strip()
if RUN_ID and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", RUN_ID):
    raise ValueError("KG_TABI_RUN_ID may contain only letters, digits, '.', '_' and '-'.")
DATA_DIR = os.path.join(DATA_ROOT, "runs", RUN_ID) if RUN_ID else DATA_ROOT
RAW_PAPERS_DIR = os.path.join(DATA_DIR, "raw_papers")
TRIPLES_DIR = os.path.join(DATA_DIR, "triples")
GRAPH_DIR = os.path.join(DATA_DIR, "graph")
GAPS_DIR = os.path.join(DATA_DIR, "gaps")
PROVENANCE_DIR = os.path.join(DATA_DIR, "provenance")
LLM_RESPONSE_DIR = os.path.join(PROVENANCE_DIR, "llm_responses")
LLM_CALL_LOG_PATH = os.path.join(PROVENANCE_DIR, "llm_calls.jsonl")

# Ensure directories exist
for path in [RAW_PAPERS_DIR, TRIPLES_DIR, GRAPH_DIR, GAPS_DIR, PROVENANCE_DIR]:
    os.makedirs(path, exist_ok=True)
