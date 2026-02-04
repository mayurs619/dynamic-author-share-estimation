import json
import math
from typing import Any, Dict, List, Mapping, Union

CREDIT_ROLES = [
    "Conceptualization",
    "Methodology",
    "Software",
    "Validation",
    "Formal Analysis",
    "Investigation",
    "Resources",
    "Data Curation",
    "Writing – Original Draft",
    "Writing – Review & Editing",
    "Visualization",
    "Supervision",
    "Project Administration",
    "Funding Acquisition",
]

def softmax(logits: List[float]) -> List[float]:
    """
    Apply softmax to a list of logits.
    """
    exp_logits = [math.exp(logit) for logit in logits]
    return [exp_logit / sum(exp_logits) for exp_logit in exp_logits]