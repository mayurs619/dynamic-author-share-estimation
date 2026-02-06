import pytest

from model.author_share_model import AuthorShareModel

# Dummy role weights
ROLE_WEIGHTS = {
    "Conceptualization": 20,
    "Methodology": 12,
    "Software": 8,
    "Validation": 6,
    "Formal Analysis": 10,
    "Investigation": 8,
    "Resources": 4,
    "Data Curation": 6,
    "Writing – Original Draft": 16,
    "Writing – Review & Editing": 5,
    "Visualization": 3,
    "Supervision": 1,
    "Project Administration": 0.5,
    "Funding Acquisition": 0.5,
}


def assert_distribution(shares, n, tol=1e-12):
    assert len(shares) == n
    assert all(x >= 0.0 for x in shares)
    assert sum(shares) == pytest.approx(1.0, abs=tol)


@pytest.fixture
def model():
    return AuthorShareModel(ROLE_WEIGHTS)


def test_3_authors_basic(model):
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


def test_10_authors_variable_length_uniform_when_identical(model):
    authors = [f"a{i}" for i in range(10)]
    payload = {
        "all_authors": authors,
        "roles": {"Software": [{"author_id": a, "contribution": 1.0} for a in authors]},
    }
    shares = model.predict(payload)
    assert_distribution(shares, 10)
    assert shares == pytest.approx([0.1] * 10, abs=1e-12)
    assert shares[1] == pytest.approx(1 / 10, abs=1e-12)