import json
import math
from pathlib import Path

import pytest

from model.softmax_author_share_model import SoftmaxAuthorShare, softmax


def assert_distribution(shares, n, tol=1e-12):
    assert len(shares) == n
    assert all(math.isfinite(x) for x in shares)
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
    return SoftmaxAuthorShare(role_weights, temperature=1.0)


def test_init_requires_positive_temperature(role_weights):
    with pytest.raises(ValueError, match="temperature must be > 0"):
        SoftmaxAuthorShare(role_weights, temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        SoftmaxAuthorShare(role_weights, temperature=-1.0)


def test_predict_returns_probability_distribution(model):
    payload = {
        "all_authors": ["a1", "a2", "a3"],
        "roles": {
            "Conceptualization": [{"author_id": "a1", "contribution": 1.0}],
            "Methodology": [
                {"author_id": "a2", "contribution": 1.0},
                {"author_id": "a3", "contribution": 1.0},
            ],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 3)


def test_title_case_payload_works_with_snake_case_weights(model):
    # Regression for snake_case weights.json vs title-case payload keys
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Conceptualization": [{"author_id": "a1", "contribution": 1.0}],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    assert shares[0] > shares[1]


@pytest.mark.parametrize("role_label", ["Writing - Original Draft", "Writing – Original Draft"])
def test_hyphen_and_endash_labels_are_equivalent(model, role_label):
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
    # Author listed under role but no explicit contribution -> baseline 1.0
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Software": [{"author_id": "a1"}, {"author_id": "a2"}],
        },
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    assert shares == pytest.approx([0.5, 0.5], abs=1e-12)


def test_temperature_controls_sharpness():
    # Use simple custom weights so effect is visible and not saturated
    custom_weights = {"software": 1.0}
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {"Software": [{"author_id": "a1", "contribution": 1.0}]},
    }

    low_t = SoftmaxAuthorShare(custom_weights, temperature=0.5).predict(payload)
    high_t = SoftmaxAuthorShare(custom_weights, temperature=5.0).predict(payload)

    assert_distribution(low_t, 2)
    assert_distribution(high_t, 2)

    # Lower T -> sharper winner-take-all
    assert low_t[0] > high_t[0]
    assert low_t[1] < high_t[1]

    # High T should be closer to uniform
    assert abs(high_t[0] - 0.5) < abs(low_t[0] - 0.5)


def test_numerical_stability_with_large_logits():
    # Very large weight should not overflow/NaN due to max-shift softmax
    huge_weights = {"conceptualization": 1_000_000.0}
    model = SoftmaxAuthorShare(huge_weights, temperature=1.0)

    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {"Conceptualization": [{"author_id": "a1", "contribution": 1.0}]},
    }
    shares = model.predict(payload)

    assert_distribution(shares, 2)
    assert shares[0] > 0.999999
    assert shares[1] < 0.000001


def test_infers_all_authors_if_missing(model):
    payload = {
        "roles": {
            "Methodology": [
                {"author_id": "b", "contribution": 3.0},
                {"author_id": "a", "contribution": 1.0},
            ]
        }
    }
    shares = model.predict(payload)
    assert_distribution(shares, 2)
    # inferred order is sorted(set(...)) => ["a", "b"], so b should have higher prob
    assert shares[1] > shares[0]


def test_roles_must_be_dict(model):
    with pytest.raises(ValueError, match="'roles' must be a dict"):
        model.predict({"all_authors": ["a1"], "roles": []})


def test_accepts_json_string_input(model):
    payload = {
        "all_authors": ["a1", "a2"],
        "roles": {
            "Software": [
                {"author_id": "a1", "contribution": 2.0},
                {"author_id": "a2", "contribution": 1.0},
            ]
        },
    }
    shares = model.predict(json.dumps(payload))
    assert_distribution(shares, 2)
    assert shares[0] > shares[1]


def test_softmax_helper_equal_logits_returns_uniform():
    probs = softmax([10.0, 10.0, 10.0], temperature=1.0)
    assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-12)


def test_duplicate_alias_weights_are_merged():
    model = SoftmaxAuthorShare(
        {
            "writing_original_draft": 10.0,
            "Writing - Original Draft": 5.0,
        },
        temperature=1.0,
    )
    assert model.role_weights["writing_original_draft"] == pytest.approx(15.0, abs=1e-12)

def test_softmax_matches_manual_formula_temperature_1():
    logits = [1.2, -0.7, 0.0]
    probs = softmax(logits, temperature=1.0) 

    exps = [math.exp(z) for z in logits]
    denom = sum(exps)
    expected = [e / denom for e in exps]

    assert probs == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_softmax_matches_manual_formula_with_temperature():
    logits = [2.0, 0.0, -1.0]
    T = 2.5
    probs = softmax(logits, temperature=T)

    exps = [math.exp(z / T) for z in logits]
    denom = sum(exps)
    expected = [e / denom for e in exps]

    assert probs == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_softmax_shift_invariance():
    # softmax(z) == softmax(z + c)
    logits = [0.3, 1.1, -2.0, 4.5]
    c = 123.456
    p1 = softmax(logits, temperature=1.0)
    p2 = softmax([z + c for z in logits], temperature=1.0)

    assert p1 == pytest.approx(p2, rel=1e-12, abs=1e-12)


def test_softmax_two_class_closed_form():
    # For two classes: p1 = 1 / (1 + exp((z2-z1)/T))
    z1, z2 = 3.0, 1.0
    T = 0.8
    probs = softmax([z1, z2], temperature=T)

    expected_p1 = 1.0 / (1.0 + math.exp((z2 - z1) / T))
    expected_p2 = 1.0 - expected_p1

    assert probs[0] == pytest.approx(expected_p1, rel=1e-12, abs=1e-12)
    assert probs[1] == pytest.approx(expected_p2, rel=1e-12, abs=1e-12)


def test_softmax_monotonic_with_logit_increase():
    logits = [0.0, 0.0, 0.0]
    base = softmax(logits, temperature=1.0)
    boosted = softmax([1.0, 0.0, 0.0], temperature=1.0)

    # Increasing one logit should increase only its probability
    assert boosted[0] > base[0]
    assert boosted[1] < base[1]
    assert boosted[2] < base[2]