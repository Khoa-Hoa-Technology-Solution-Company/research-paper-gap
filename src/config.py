import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

# Semantic Scholar Configuration
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None

# Business logic constraints for Software Engineering
ENTITY_TYPES = ["METHOD", "DATASET", "METRIC", "CONCEPT", "FINDING", "TOOL"]
RELATION_TYPES = [
    "USES", "IMPROVES", "ADDRESSES", "EVALUATES_ON", "PRODUCES",
    "CONTRADICTS", "EXTENDS", "LACKS", "COMBINES", "APPLIED_TO"
]

# Topology / Filtering Thresholds
RELEVANCE_THRESHOLD = 0.7
TRIPLE_CONFIDENCE_THRESHOLD = 0.3
FUZZY_MATCH_THRESHOLD = 85
COSINE_SIMILARITY_THRESHOLD = 0.85
LOUVAIN_MIN_SIZE_RATIO = 0.05
LOUVAIN_MAX_BRIDGE_RATIO = 0.10
TEMPORAL_DECAY_THRESHOLD = 0.30

# File Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_PAPERS_DIR = os.path.join(DATA_DIR, "raw_papers")
TRIPLES_DIR = os.path.join(DATA_DIR, "triples")
GRAPH_DIR = os.path.join(DATA_DIR, "graph")
GAPS_DIR = os.path.join(DATA_DIR, "gaps")

# Ensure directories exist
for path in [RAW_PAPERS_DIR, TRIPLES_DIR, GRAPH_DIR, GAPS_DIR]:
    os.makedirs(path, exist_ok=True)
