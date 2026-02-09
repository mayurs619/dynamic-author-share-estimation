"""
Standalone Synthetic Authorship Share Generator
Empirically grounded priors + Dirichlet targets
"""

import json
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict

from pathlib import Path
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# -------------------------------------------------
# 1. EMPIRICAL POSITION PRIORS (FROM TABLE)
# -------------------------------------------------

POSITION_PRIORS = {
    3: {"first": 0.42, "middle": 0.17, "last": 0.41},
    5: {"first": 0.34, "second": 0.12, "middle": 0.08, "fourth": 0.07, "last": 0.38},
}

def extrapolate_position_priors(n: int) -> Dict[str, float]:
    priors = {"first": 0.30, "last": 0.30}
    middle_share = 0.40 / max(1, n - 2)
    for i in range(1, n - 1):
        priors[f"middle_{i}"] = middle_share
    return priors


# -------------------------------------------------
# 2. ROLE MULTIPLIERS (COARSE CRediT BRIDGE)
# -------------------------------------------------

ROLE_POSITION_MULTIPLIERS = {
    "Conceptualization": {"first": 1.1, "middle": 0.9, "last": 1.3},
    "Writing":           {"first": 1.3, "middle": 1.0, "last": 0.8},
    "DataCuration":      {"first": 1.0, "middle": 1.2, "last": 0.7},
    "Supervision":       {"first": 0.7, "middle": 0.8, "last": 1.4},
}

ALL_ROLES = list(ROLE_POSITION_MULTIPLIERS.keys())


# -------------------------------------------------
# 3. HELPERS
# -------------------------------------------------

def author_positions(n: int) -> List[str]:
    if n == 1:
        return ["solo"]
    if n == 2:
        return ["first", "last"]
    if n == 3:
        return ["first", "middle", "last"]
    if n == 5:
        return ["first", "second", "middle", "fourth", "last"]
    return ["first"] + [f"middle_{i}" for i in range(1, n - 1)] + ["last"]


def assign_roles(n: int) -> List[List[str]]:
    roles = []
    for _ in range(n):
        k = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25])
        roles.append(list(np.random.choice(ALL_ROLES, size=k, replace=False)))
    return roles


def base_position_score(n: int, pos: str) -> float:
    if n in POSITION_PRIORS:
        return POSITION_PRIORS[n].get(pos, POSITION_PRIORS[n].get("middle", 0.05))
    priors = extrapolate_position_priors(n)
    return priors.get(pos, priors["first"])


# -------------------------------------------------
# 4. CORE GENERATOR
# -------------------------------------------------

def generate_paper() -> Dict:
    n = int(np.random.choice(
        [1, 2, 3, 5, 10, 20],
        p=[0.05, 0.10, 0.20, 0.30, 0.20, 0.15]
    ))

    positions = author_positions(n)
    roles = assign_roles(n)

    scores = []
    role_labels = []

    for i, pos in enumerate(positions):
        pos_key = pos if pos in ["first", "last"] else "middle"
        s = base_position_score(n, pos_key)

        for r in roles[i]:
            s *= ROLE_POSITION_MULTIPLIERS[r][pos_key]
            role_labels.append(r)

        scores.append(s)

    scores = np.array(scores) ** 0.85  # diminishing returns

    alpha = scores * np.random.uniform(6, 15)
    shares = np.random.dirichlet(alpha)

    return {
        "paper_id": str(uuid.uuid4()),
        "num_authors": n,
        "positions": positions,
        "roles": roles,
        "raw_scores": scores.tolist(),
        "shares": shares.tolist(),
    }


# -------------------------------------------------
# 5. DATASET GENERATION
# -------------------------------------------------

def generate_dataset(N: int = 1200) -> pd.DataFrame:
    rows = [generate_paper() for _ in range(N)]
    df = pd.DataFrame(rows)
    return df


# -------------------------------------------------
# 6. DIAGNOSTICS
# -------------------------------------------------

def visualize_outputs(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import numpy as np

    # Flatten all shares
    all_shares = np.concatenate(df["shares"].values)

    # -------------------------------
    # 1. Histogram of all shares
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(all_shares, bins=50, range=(0, 1), color="steelblue", alpha=0.85)
    plt.title("Global Histogram of Authorship Shares")
    plt.xlabel("Share value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 2. Boxplot by role
    # -------------------------------
    role_shares = defaultdict(list)

    for _, row in df.iterrows():
        for roles, share in zip(row["roles"], row["shares"]):
            for r in roles:
                role_shares[r].append(share)

    labels = list(role_shares.keys())
    data = [role_shares[r] for r in labels]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title("Authorship Share Distribution by Role")
    plt.ylabel("Share")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 3. Mean share vs team size
    # -------------------------------
    team_means = defaultdict(list)

    for _, row in df.iterrows():
        team_means[row["num_authors"]].extend(row["shares"])

    sizes = sorted(team_means.keys())
    means = [np.mean(team_means[s]) for s in sizes]

    plt.figure(figsize=(7, 5))
    plt.plot(sizes, means, marker="o")
    plt.title("Mean Authorship Share vs Team Size")
    plt.xlabel("Number of Authors")
    plt.ylabel("Mean Share")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_diagnostics(df: pd.DataFrame):
    print("\n=== BASIC CHECKS ===")
    assert not df["shares"].isna().any(), "NaN found in shares column"

    all_shares = np.concatenate(df["shares"].values)

    print(f"Global share min: {all_shares.min():.4f}")
    print(f"Global share max: {all_shares.max():.4f}")
    print(f"Mean share: {all_shares.mean():.4f}")
    print(f"Variance: {all_shares.var():.4f}")

    role_shares = defaultdict(list)

    for _, row in df.iterrows():
        for roles, share in zip(row["roles"], row["shares"]):
            for r in roles:
                role_shares[r].append(share)

    print("\n=== MEAN SHARE PER ROLE ===")
    for r, vals in role_shares.items():
        print(f"{r:20s} mean={np.mean(vals):.4f} var={np.var(vals):.4f}")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(all_shares, bins=50, range=(0, 1), color="steelblue", alpha=0.8)
    plt.title("Histogram of Generated Authorship Shares")
    plt.xlabel("Share")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "share_histogram.png")
    plt.close()

    print("Saved histogram to:", (OUT_DIR / "share_histogram.png").resolve())


# -------------------------------------------------
# 7. MAIN
# -------------------------------------------------

if __name__ == "__main__":
    df = generate_dataset(N=1500)

    # Save outputs
    df_out = df.copy()
    for col in ["positions", "roles", "raw_scores", "shares"]:
        df_out[col] = df_out[col].apply(json.dumps)

    df_out.to_csv("synthetic_authorship_data.csv", index=False)
    df_out.to_json("synthetic_authorship_data.json", orient="records", indent=2)

    run_diagnostics(df)
    
# -------------------------------------------------
# 8. PRINT SAMPLE ROWS  ✅ ADD HERE
# -------------------------------------------------

for _, row in df.iterrows():
    print(json.dumps(row.to_dict(), indent=2))