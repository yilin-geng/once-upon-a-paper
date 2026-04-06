"""Constants, venue definitions, model pricing, and defaults."""

from __future__ import annotations

# Year range defaults
DEFAULT_YEAR_MIN = 2016
DEFAULT_YEAR_MAX = 2026
DEFAULT_BATCH_SIZE = 50

# ACL Anthology bib URL (with abstracts)
ACL_BIB_URL = "https://aclanthology.org/anthology+abstracts.bib.gz"

# Semantic Scholar bulk search endpoint
S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "paperId,title,abstract,year,venue,authors,externalIds"

# Venue definitions: name -> {source, booktitle keywords for classification}
# ACL venues are identified by booktitle patterns in the ACL bib file
ACL_VENUES = {
    "ACL": {"booktitle_patterns": [
        "Annual Meeting of the Association for Computational Linguistics",
        "Proceedings of the ACL",
    ]},
    "EMNLP": {"booktitle_patterns": [
        "Conference on Empirical Methods in Natural Language Processing",
        "EMNLP",
    ]},
    "NAACL": {"booktitle_patterns": [
        "North American Chapter",
        "NAACL",
    ]},
    "EACL": {"booktitle_patterns": [
        "European Chapter",
        "EACL",
    ]},
    "COLING": {"booktitle_patterns": [
        "International Conference on Computational Linguistics",
        "COLING",
    ]},
    "Findings": {"booktitle_patterns": [
        "Findings of the Association for Computational Linguistics",
    ]},
}

# S2 venues: our name -> list of S2 venue query strings to try
S2_VENUES = {
    "NeurIPS": "NeurIPS",
    "ICML": "ICML",
    "ICLR": "ICLR",
    "AAAI": "AAAI",
    "IJCAI": "IJCAI",
}

ALL_VENUES = list(ACL_VENUES.keys()) + list(S2_VENUES.keys())

# Model pricing: USD per token
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6, "provider": "openai"},
    "gpt-4o": {"input": 2.50e-6, "output": 10.0e-6, "provider": "openai"},
    "gpt-4.1-mini": {"input": 0.40e-6, "output": 1.60e-6, "provider": "openai"},
    "gpt-4.1-nano": {"input": 0.10e-6, "output": 0.40e-6, "provider": "openai"},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00e-6, "output": 15.0e-6, "provider": "anthropic"},
    "claude-haiku-4-20250414": {"input": 0.80e-6, "output": 4.00e-6, "provider": "anthropic"},
    # Google Gemini (Vertex AI)
    "gemini-2.0-flash": {"input": 0.10e-6, "output": 0.40e-6, "provider": "google"},
    "gemini-2.5-flash-preview-04-17": {"input": 0.15e-6, "output": 0.60e-6, "provider": "google"},
    "gemini-2.5-pro-preview-03-25": {"input": 1.25e-6, "output": 10.0e-6, "provider": "google"},
    # Local (free)
    "local": {"input": 0.0, "output": 0.0, "provider": "huggingface"},
}

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_CATEGORIZE_BATCH_SIZE = 25
DEFAULT_REPORT_MAX_PAPERS_PER_SECTION = 30

# Data freshness: seconds before data is considered stale
DATA_FRESHNESS_DAYS = 30

# Paper categorization taxonomy
CATEGORIES = [
    "OVERLAP", "CONTRA", "SAME_PROBLEM", "SAME_METHOD",
    "FOUNDATION", "SUPPORT", "TANGENTIAL",
]

CATEGORY_DISPLAY_ORDER = [
    "OVERLAP", "CONTRA", "SAME_PROBLEM", "SAME_METHOD",
    "FOUNDATION", "SUPPORT", "TANGENTIAL",
]

CATEGORY_LABELS = {
    "OVERLAP": "Direct Overlap",
    "CONTRA": "Contradictory Evidence",
    "SAME_PROBLEM": "Shared Problem, Different Method",
    "SAME_METHOD": "Shared Method, Different Problem",
    "FOUNDATION": "Foundational / Landscape",
    "SUPPORT": "Supporting Evidence",
    "TANGENTIAL": "Tangential / Peripheral",
}

CATEGORY_ALERTS = {
    "OVERLAP": "NOVELTY RISK",
    "CONTRA": "ADDRESS REQUIRED",
}

CATEGORY_COLORS = {
    "OVERLAP": "bold red",
    "CONTRA": "bold yellow",
    "SAME_PROBLEM": "cyan",
    "SAME_METHOD": "blue",
    "FOUNDATION": "magenta",
    "SUPPORT": "green",
    "TANGENTIAL": "dim",
}
