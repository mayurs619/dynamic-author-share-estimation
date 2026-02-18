import json
from pathlib import Path

import pytest

from model.author_share_model import AuthorShareModel


def assert_distribution(shares, n, tol=1e-12):
    assert len(shares) == n
    assert all(x >= 0.0 for x in shares)
    assert sum(shares) == pytest.approx(1.0, abs=tol)


@pytest.fixture(scope="module")
def role_weights():
    repo_root = Path(__file__).resolve().parents[1]
    weights_path = repo_root / "data" / "weights.json"
    with weights_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert "weights" in payload
    assert isinstance(payload["weights"], dict)
    return payload["weights"]


@pytest.fixture
def model(role_weights):
    return AuthorShareModel(role_weights)


def test_weights_file_is_production_shape(role_weights):
    # Ensures tests are aligned to real snake_case production weights
    assert "writing_original_draft" in role_weights
    assert "software" in role_weights
    assert "Writing - Original Draft" not in role_weights


def test_title_case_payload_works_with_snake_case_weights(model):
    # Critical bug regression: payload role labels can be Title Case
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Conceptualization": [{"author_id": "a1", "contribution": 1.0}],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    assert shares == pytest.approx([1.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("role_label", ["Writing - Original Draft", "Writing – Original Draft"])
def test_hyphen_and_endash_role_labels_are_equivalent(model, role_label):
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            role_label: [
                {"author_id": "a1", "contribution": 1.0},
                {"author_id": "a2", "contribution": 1.0},
            ]
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    assert shares == pytest.approx([0.5, 0.5], abs=1e-12)


def test_missing_contribution_defaults_to_one(model):
    # Requirement: if author is listed under a role, default contribution is baseline 1.0
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Software": [{"author_id": "a1"}, {"author_id": "a2"}],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    assert shares == pytest.approx([0.5, 0.5], abs=1e-12)


def test_nonnumeric_contribution_falls_back_to_one(model):
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Software": [
                {"author_id": "a1", "contribution": "not-a-number"},
                {"author_id": "a2", "contribution": 1.0},
            ],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    assert shares == pytest.approx([0.5, 0.5], abs=1e-12)


def test_negative_contribution_clamped_to_zero(model):
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Software": [
                {"author_id": "a1", "contribution": -10.0},
                {"author_id": "a2", "contribution": 1.0},
            ],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    assert shares == pytest.approx([0.0, 1.0], abs=1e-12)


def test_infers_all_authors_when_missing(model):
    # Inferred authors are sorted(set(...)) => ["a", "b"]
    payload = {
        "roles": {
            "Conceptualization": [
                {"author_id": "b", "contribution": 3.0},
                {"author_id": "a", "contribution": 1.0},
            ]
        }
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    # With sorted order ["a", "b"], contribution ratio is 1:3
    assert shares == pytest.approx([0.25, 0.75], abs=1e-12)


def test_roles_must_be_dict(model):
    with pytest.raises(ValueError, match="'roles' must be a dict"):
        model.predict({"all_authors": ["a1"], "roles": []})


def test_requires_non_empty_all_authors_if_cannot_infer(model):
    with pytest.raises(ValueError, match="Provide non-empty 'all_authors'"):
        model.predict({"roles": {}})


def test_unknown_roles_are_ignored_and_result_is_uniform_if_no_points(model):
    payload = {
        "all_authors": ["a1", "a2", "a3"],
        "roles": {
            "Some Unknown Role": [{"author_id": "a1", "contribution": 1.0}],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 3)
    assert shares == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-12)


def test_accepts_json_string_input(model):
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Methodology": [
                {"author_id": "a1", "contribution": 2.0},
                {"author_id": "a2", "contribution": 1.0},
            ]
        },
    }
    shares = model.predict(json.dumps(payload))
    assert_distribution(shares, 2)
    assert shares == pytest.approx([2 / 3, 1 / 3], abs=1e-12)


def test_duplicate_alias_keys_in_weights_are_merged():
    model_local = AuthorShareModel(
        {
            "writing_original_draft": 10.0,
            "Writing - Original Draft": 5.0,
        }
    )
    assert model_local.role_weights["writing_original_draft"] == pytest.approx(15.0, abs=1e-12)