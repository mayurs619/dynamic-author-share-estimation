"""
Generate Clean Training Set for Iurii
Uses DataValidator to ensure all papers pass validation
"""

import json
import uuid
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from data_validator import DataValidator

# -----------------------------
# Output directory
# -----------------------------

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# -----------------------------
# Role weights (Iurii priors)
# -----------------------------

ROLE_WEIGHTS: Dict[str, float] = {
    "Conceptualization": 150,
    "Methodology": 110,
    "Software": 90,
    "Validation": 60,
    "Formal Analysis": 120,
    "Investigation": 100,
    "Resources": 30,
    "Data Curation": 70,
    "Writing – Original Draft": 150,
    "Writing – Review & Editing": 50,
    "Visualization": 60,
    "Supervision": 40,
    "Project Administration": 30,
    "Funding Acquisition": 30,
}

ALL_ROLES = list(ROLE_WEIGHTS.keys())

# -----------------------------
# Iurii-style linear model
# -----------------------------

def normalize(points: List[float]) -> List[float]:
    s = sum(points)
    if s > 0:
        return [p / s for p in points]
    return [1.0 / len(points)] * len(points)


class AuthorShareModel:
    def __init__(self, role_weights: Dict[str, float]):
        self.role_weights = role_weights

    def predict(self, payload: Dict[str, Any]) -> List[float]:
        roles_payload = payload["roles"]
        authors = payload["all_authors"]

        index = {a: i for i, a in enumerate(authors)}
        points = [0.0] * len(authors)

        for role, weight in self.role_weights.items():
            contributors = roles_payload.get(role, [])
            if not contributors:
                continue

            total = sum(c["contribution"] for c in contributors)
            if total <= 0:
                continue

            for c in contributors:
                idx = index[c["author_id"]]
                points[idx] += weight * (c["contribution"] / total)

        return normalize(points)


MODEL = AuthorShareModel(ROLE_WEIGHTS)

# -----------------------------
# Synthetic paper generation
# -----------------------------

def generate_roles(authors: List[str]) -> Dict[str, List[Dict[str, float]]]:
    roles: Dict[str, List[Dict[str, float]]] = {}

    for role in ALL_ROLES:
        if np.random.rand() < 0.35:
            k = np.random.randint(1, min(3, len(authors)) + 1)
            chosen = np.random.choice(authors, size=k, replace=False)
            roles[role] = [
                {"author_id": a, "contribution": 1.0} for a in chosen
            ]

    return roles


def generate_paper() -> Dict[str, Any]:
    n = int(np.random.choice([1, 2, 3, 5, 10, 20],
                             p=[0.05, 0.10, 0.20, 0.30, 0.20, 0.15]))

    authors = [str(uuid.uuid4()) for _ in range(n)]
    roles = generate_roles(authors)

    payload = {
        "all_authors": authors,
        "roles": roles,
    }

    base_shares = MODEL.predict(payload)

    base_shares = np.array(base_shares, dtype=float)
    min_alpha = 1e-6  # negligible but non-zero mass for zero-signal authors
    scaled = base_shares * np.random.uniform(8, 20)
    alpha = np.maximum(scaled, min_alpha)

    shares = np.random.dirichlet(alpha)

    return {
        "paper_id": str(uuid.uuid4()),
        "all_authors": authors,
        "roles": roles,
        "base_shares": base_shares.tolist(),
        "shares": shares.tolist(),
    }


# -----------------------------
# Training set generation
# -----------------------------

def generate_training_set(
    target_size: int = 10000,
    max_attempts_multiplier: float = 1.2,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate a validated training set
    
    Args:
        target_size: Number of valid papers needed
        max_attempts_multiplier: Generate up to this many papers to reach target
        verbose: Print progress
    
    Returns:
        List of validated papers
    """
    validator = DataValidator(tolerance=1e-9)
    
    max_attempts = int(target_size * max_attempts_multiplier)
    
    if verbose:
        print(f"Generating up to {max_attempts} papers to get {target_size} valid ones...")
    
    # Generate papers
    papers = []
    for i in range(max_attempts):
        if verbose and (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{max_attempts} papers...")
        
        paper = generate_paper()
        papers.append(paper)
        
        # Early exit if we have enough
        if len(papers) >= target_size:
            # Check if we have enough valid papers
            temp_valid, temp_report = validator.filter_valid_papers(papers)
            if len(temp_valid) >= target_size:
                if verbose:
                    print(f"Reached {target_size} valid papers early at {i + 1} generated")
                papers = papers[:i + 1]
                break
    
    if verbose:
        print(f"\nValidating {len(papers)} generated papers...")
    
    # Filter to valid papers only
    valid_papers, report = validator.filter_valid_papers(papers)
    
    if verbose:
        print(report.summary())
    
    # Take exactly target_size papers
    if len(valid_papers) < target_size:
        raise ValueError(
            f"Only generated {len(valid_papers)} valid papers, "
            f"need {target_size}. Increase max_attempts_multiplier."
        )
    
    return valid_papers[:target_size]


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING SET GENERATOR v1.0")
    print("=" * 60)
    
    # Generate 10,000 valid papers
    training_set = generate_training_set(
        target_size=10000,
        max_attempts_multiplier=1.2,
        verbose=True
    )
    
    print(f"\n✅ Successfully generated {len(training_set)} valid papers")
    
    # Save to JSONL format
    output_path = OUT_DIR / "training_set_v1.jsonl"
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        for paper in training_set:
            f.write(json.dumps(paper) + '\n')
    
    print(f"✅ Saved {len(training_set)} papers to {output_path}")
    
    # Run final validation to confirm
    print("\nRunning final validation check...")
    validator = DataValidator(tolerance=1e-9)
    final_report = validator.validate_dataset(training_set, verbose=False)
    
    if final_report.invalid_papers == 0:
        print(f"✅ ALL {final_report.total_papers} PAPERS PASSED VALIDATION!")
    else:
        print(f"❌ ERROR: {final_report.invalid_papers} papers failed validation")
        print(final_report.summary())
    
    # Print sample papers
    print("\n" + "=" * 60)
    print("SAMPLE PAPERS (first 3)")
    print("=" * 60)
    for i, paper in enumerate(training_set[:3]):
        print(f"\nPaper {i+1}:")
        print(json.dumps({
            "paper_id": paper["paper_id"],
            "num_authors": len(paper["all_authors"]),
            "num_roles": len(paper["roles"]),
            "shares_sum": sum(paper["shares"]),
            "shares": paper["shares"]
        }, indent=2))
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
