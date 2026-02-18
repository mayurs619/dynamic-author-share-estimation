import json
import math
from typing import Any, Dict, List, Mapping, Union

ROLE_ALIASES = {
    # canonical snake_case -> canonical snake_case
    "conceptualization": "conceptualization",
    "methodology": "methodology",
    "software": "software",
    "validation": "validation",
    "formal_analysis": "formal_analysis",
    "investigation": "investigation",
    "resources": "resources",
    "data_curation": "data_curation",
    "writing_original_draft": "writing_original_draft",
    "writing_review_editing": "writing_review_editing",
    "visualization": "visualization",
    "supervision": "supervision",
    "project_administration": "project_administration",
    "funding_acquisition": "funding_acquisition",

    # Title Case / standard CRediT -> canonical snake_case
    "Conceptualization": "conceptualization",
    "Methodology": "methodology",
    "Software": "software",
    "Validation": "validation",
    "Formal Analysis": "formal_analysis",
    "Investigation": "investigation",
    "Resources": "resources",
    "Data Curation": "data_curation",
    "Writing - Original Draft": "writing_original_draft",
    "Writing – Original Draft": "writing_original_draft",
    "Writing - Review & Editing": "writing_review_editing",
    "Writing – Review & Editing": "writing_review_editing",
    "Visualization": "visualization",
    "Supervision": "supervision",
    "Project Administration": "project_administration",
    "Funding Acquisition": "funding_acquisition",
}

def normalize(points: List[float]) -> List[float]:
    """
    Normalize a list of points to a probability distribution that always sums to 1.
    """
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
        canonical_weights: Dict[str, float] = {}
        for k, v in role_weights.items():
            canonical = ROLE_ALIASES.get(k, k)  # no normalization, dictionary only
            canonical_weights[canonical] = canonical_weights.get(canonical, 0.0) + float(v) # If 2 different input keys map to the same canonical key, sum the weights
        self.role_weights = canonical_weights
    
    def predict(self, input_json: str | Dict[str, Any]) -> List[float]:
        # Parse input
        if isinstance(input_json, str):
            payload: Dict[str, Any] = json.loads(input_json)
        else:
            payload = input_json

        # Read roles (must be a dict)
        roles_payload_raw = payload.get("roles") or {}
        if not isinstance(roles_payload_raw, dict):
            raise ValueError("'roles' must be a dict mapping role -> list of contributors")

        # Normalize role keys via explicit alias dictionary (no canonicalizer function)
        normalized_roles_payload: Dict[str, List[Dict[str, Any]]] = {}
        for role_name, contributors in roles_payload_raw.items():
            canonical_role = ROLE_ALIASES.get(role_name, role_name)

            # Treat missing/null role entries as empty
            if contributors is None:
                continue

            if not isinstance(contributors, list):
                continue

            if canonical_role not in normalized_roles_payload:
                normalized_roles_payload[canonical_role] = []

            normalized_roles_payload[canonical_role].extend(contributors)

        # Determine author list
        all_authors = payload.get("all_authors")
        if all_authors is None:
            # Infer from role contributor entries
            inferred_authors: List[str] = []
            for contributors in normalized_roles_payload.values():
                for contributor in contributors:
                    if not isinstance(contributor, dict):
                        continue
                    author_id = contributor.get("author_id")
                    if author_id is None:
                        continue
                    inferred_authors.append(author_id)

            # De-duplicate and make deterministic order
            all_authors = sorted(set(inferred_authors))

        if not isinstance(all_authors, list) or len(all_authors) == 0:
            raise ValueError("Provide non-empty 'all_authors' (or include author_id entries under roles)")

        # Build author index
        author_index: Dict[str, int] = {}
        for i, author_id in enumerate(all_authors):
            author_index[author_id] = i

        points: List[float] = [0.0] * len(all_authors)

        # Aggregate weighted points
        for role_name, weight in self.role_weights.items():
            contributors = normalized_roles_payload.get(role_name)

            # If role is missing or empty, treat as empty
            if contributors is None or len(contributors) == 0:
                continue

            total_contribution = 0.0
            cleaned: List[Dict[str, Any]] = []

            for item in contributors:
                if not isinstance(item, dict):
                    continue

                author_id = item.get("author_id")
                if author_id is None:
                    continue

                # Default contribution is 1.0 when author is listed but no numeric score is provided
                raw_c = item.get("contribution", 1.0)
                try:
                    c = float(raw_c)
                except (TypeError, ValueError):
                    c = 1.0

                if c < 0.0:
                    c = 0.0

                cleaned.append({"author_id": author_id, "contribution": c})
                total_contribution += c

            if total_contribution <= 0.0:
                continue

            # Allocate this role's weight proportionally
            for item in cleaned:
                author_id = item["author_id"]
                c = item["contribution"]

                if c == 0.0:
                    continue
                if author_id not in author_index:
                    continue

                idx = author_index[author_id]
                points[idx] += weight * (c / total_contribution)

        return normalize(points)



