import json
import math
from typing import Any, Dict, List, Mapping

from model.role_aliases import ROLE_ALIASES


def softmax(logits: List[float], temperature: float) -> List[float]:
    """
    Temperature-scaled softmax:
    P_i = exp(z_i / T) / sum_j exp(z_j / T)
    """
    if len(logits) == 0:
        return []

    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    scaled = [z / temperature for z in logits]

    # Numerical stability
    max_scaled = max(scaled)
    exps = [math.exp(v - max_scaled) for v in scaled]
    denom = sum(exps)

    if denom <= 0.0:
        # extremely defensive fallback; should not happen in practice
        n = len(logits)
        return [1.0 / n] * n

    return [e / denom for e in exps]


class SoftmaxAuthorShare:
    """
    Role-weighted author share model using temperature-scaled softmax.
    - Build per-author logits from weighted CRediT contributions.
    - Convert logits to probabilities with softmax(T).
    """

    def __init__(self, role_weights: Mapping[str, float], temperature: float = 1.0):
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        self.temperature = float(temperature)

        canonical_weights: Dict[str, float] = {}
        for k, v in role_weights.items():
            canonical = ROLE_ALIASES.get(k, k)
            canonical_weights[canonical] = canonical_weights.get(canonical, 0.0) + float(v)

        self.role_weights = canonical_weights

    def predict(self, input_json: str | Dict[str, Any]) -> List[float]:
        # Parse input
        if isinstance(input_json, str):
            payload: Dict[str, Any] = json.loads(input_json)
        else:
            payload = input_json

        # Read roles (must be a dict)
        if "roles" not in payload or payload["roles"] is None:
            roles_payload_raw = {}
        else:
            roles_payload_raw = payload["roles"]

        if not isinstance(roles_payload_raw, dict):
            raise ValueError("'roles' must be a dict mapping role -> list of contributors")

        # Normalize role names using aliases
        normalized_roles_payload: Dict[str, List[Dict[str, Any]]] = {}
        for role_name, contributors in roles_payload_raw.items():
            canonical_role = ROLE_ALIASES.get(role_name, role_name)

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
            inferred_authors: List[str] = []
            for contributors in normalized_roles_payload.values():
                for contributor in contributors:
                    if not isinstance(contributor, dict):
                        continue
                    author_id = contributor.get("author_id")
                    if author_id is None:
                        continue
                    inferred_authors.append(author_id)

            all_authors = sorted(set(inferred_authors))

        if not isinstance(all_authors, list) or len(all_authors) == 0:
            raise ValueError("Provide non-empty 'all_authors' (or include author_id entries under roles)")

        # Build index
        author_index: Dict[str, int] = {}
        for i, author_id in enumerate(all_authors):
            author_index[author_id] = i

        logits: List[float] = [0.0] * len(all_authors)

        # Aggregate weighted logits
        for role_name, weight in self.role_weights.items():
            contributors = normalized_roles_payload.get(role_name)
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

            for item in cleaned:
                author_id = item["author_id"]
                c = item["contribution"]

                if c == 0.0:
                    continue
                if author_id not in author_index:
                    continue

                idx = author_index[author_id]
                logits[idx] += weight * (c / total_contribution)

        return softmax(logits, self.temperature)