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

def normalize(points: List[float]) -> List[float]:
    total = 0.0
    for p in points:
        total += p

    if total > 0.0:
        return [p / total for p in points]

    # If there are no points at all, return uniform distribution.
    n = len(points)
    if n == 0:
        return []
    return [1.0 / n] * n

class AuthorShareModel:
    """
    Linear, rule-based share model.
    - Aggregates weighted CRediT role contributions into per-author points.
    - Normalizes points to a probability distribution that always sums to 1.
    """

    def __init__(self, role_weights: Mapping[str, float]):
        self.role_weights = dict(role_weights)
    
    def predict(self, input_json: str | Dict[str, Any]) -> List[float]:
        # Parse input
        if isinstance(input_json, str):
            payload: Dict[str, Any] = json.loads(input_json)
        else:
            payload = input_json
        
        # Read roles (must be a dict)
        if "roles" not in payload or payload["roles"] is None: # If no roles, set to empty dict
            roles_payload: Dict[str, Any] = {}
        else:
            roles_payload = payload["roles"] # If roles, set to roles

        if not isinstance(roles_payload, dict):
            raise ValueError("'roles' must be a dict mapping role -> list of contributors")
        
        # Determine author list
        all_authors = payload.get("all_authors")

        # If no all_authors field in the json, infer from roles (contributors under roles)   
        if all_authors is None:
            # Infer from role contributor entries
            inferred_authors: List[str] = []
            for role_name, contributors in roles_payload.items():
                if not isinstance(contributors, list):
                    continue

                for contributor in contributors:
                    if not isinstance(contributor, dict):
                        continue
                    if "author_id" not in contributor:
                        continue

                    inferred_authors.append(contributor["author_id"])

            # De-duplicate and make deterministic order
            all_authors = sorted(set(inferred_authors))
        
        if not isinstance(all_authors, list) or len(all_authors) == 0:
            raise ValueError("Provide non-empty 'all_authors' (or include author_id entries under roles)")
        
        # Build an index so we can store points in a list
        author_index: Dict[str, int] = {}
        for i, author_id in enumerate(all_authors):
            author_index[author_id] = i

        points: List[float] = [0.0] * len(all_authors)
        
        # Aggregate weighted points
        for role_name, weight in self.role_weights.items():
            contributors = roles_payload.get(role_name)

            # If role is missing or null, treat as empty
            if contributors is None:
                continue

            if not isinstance(contributors, list) or len(contributors) == 0:
                continue

             # Compute total contribution for this role
            total_contribution = 0.0
            cleaned: List[Dict[str, Any]] = []

            for item in contributors:
                if not isinstance(item, dict):
                    continue
                if "author_id" not in item:
                    continue

                raw_contribution = item.get("contribution", 1.0) # If no contribution weighting, set to 1.0 (likely the only role contributor)
                ccontribution = float(raw_contribution)
                if contribution < 0.0: # If contribution weighting is negative, set to 0.0
                    contribution = 0.0

                cleaned.append({"author_id": item["author_id"], "contribution": contribution})
                total_contribution += contribution

            if total_contribution <= 0.0:
                continue

            # Allocate this role's weight proportionally
            for item in cleaned:
                author_id = item["author_id"]
                contribution = item["contribution"]

                if contribution == 0.0:
                    continue
                if author_id not in author_index: # If author_id not in author_index, skip
                    continue

                idx = author_index[author_id]
                points[idx] += weight * (contribution / total_contribution)
        
        return normalize(points)



