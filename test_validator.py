"""
Test suite demonstrating DataValidator capabilities
"""

import json
import numpy as np
from data_validator import DataValidator, run_diagnostics

def test_validator():
    """Demonstrate validator with various test cases"""
    
    validator = DataValidator(tolerance=1e-9)
    
    print("=" * 60)
    print("DATA VALIDATOR TEST SUITE")
    print("=" * 60)
    
    # Test 1: Valid paper
    print("\n[TEST 1] Valid paper:")
    valid_paper = {
        "paper_id": "test-001",
        "all_authors": ["author1", "author2", "author3"],
        "shares": [0.5, 0.3, 0.2]
    }
    result = validator.validate_paper(valid_paper)
    print(f"  Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    if result.errors:
        for error in result.errors:
            print(f"    Error: {error}")
    
    # Test 2: Shares don't sum to 1.0
    print("\n[TEST 2] Shares sum to 0.95 (should fail):")
    bad_sum_paper = {
        "paper_id": "test-002",
        "all_authors": ["author1", "author2"],
        "shares": [0.5, 0.45]  # sums to 0.95
    }
    result = validator.validate_paper(bad_sum_paper)
    print(f"  Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    if result.errors:
        for error in result.errors:
            print(f"    Error: {error}")
    
    # Test 3: Negative shares
    print("\n[TEST 3] Negative share value (should fail):")
    negative_paper = {
        "paper_id": "test-003",
        "all_authors": ["author1", "author2"],
        "shares": [1.2, -0.2]  # negative value
    }
    result = validator.validate_paper(negative_paper)
    print(f"  Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    if result.errors:
        for error in result.errors:
            print(f"    Error: {error}")
    
    # Test 4: Length mismatch
    print("\n[TEST 4] Author/share count mismatch (should fail):")
    mismatch_paper = {
        "paper_id": "test-004",
        "all_authors": ["author1", "author2", "author3"],
        "shares": [0.6, 0.4]  # only 2 shares for 3 authors
    }
    result = validator.validate_paper(mismatch_paper)
    print(f"  Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    if result.errors:
        for error in result.errors:
            print(f"    Error: {error}")
    
    # Test 5: Missing required field
    print("\n[TEST 5] Missing 'shares' field (should fail):")
    missing_field_paper = {
        "paper_id": "test-005",
        "all_authors": ["author1", "author2"]
        # shares field is missing
    }
    result = validator.validate_paper(missing_field_paper)
    print(f"  Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    if result.errors:
        for error in result.errors:
            print(f"    Error: {error}")
    
    # Test 6: Tolerance edge case (within tolerance)
    print("\n[TEST 6] Shares sum to 1.0 + 5e-10 (within tolerance):")
    edge_case_paper = {
        "paper_id": "test-006",
        "all_authors": ["author1", "author2"],
        "shares": [0.5, 0.5 + 5e-10]  # very small deviation
    }
    result = validator.validate_paper(edge_case_paper)
    print(f"  Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    print(f"  Sum: {sum(edge_case_paper['shares']):.15f}")
    print(f"  Deviation: {abs(sum(edge_case_paper['shares']) - 1.0):.2e}")
    
    # Test 7: Dataset validation
    print("\n[TEST 7] Validating a mixed dataset:")
    dataset = [
        valid_paper,
        bad_sum_paper,
        {
            "paper_id": "test-007",
            "all_authors": ["a1", "a2", "a3"],
            "shares": [0.4, 0.4, 0.2]
        },
        negative_paper,
    ]
    
    report = validator.validate_dataset(dataset, verbose=False)
    print(f"  Total: {report.total_papers}")
    print(f"  Valid: {report.valid_papers}")
    print(f"  Invalid: {report.invalid_papers}")
    print(f"  Pass rate: {report.pass_rate:.1%}")
    
    # Test 8: Filter valid papers
    print("\n[TEST 8] Filtering to valid papers only:")
    valid_only, report = validator.filter_valid_papers(dataset)
    print(f"  Original dataset: {len(dataset)} papers")
    print(f"  Filtered dataset: {len(valid_only)} papers")
    print(f"  Removed: {len(dataset) - len(valid_only)} papers")
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


def test_with_training_set():
    """Test validator on the generated training set"""
    
    print("\n" + "=" * 60)
    print("VALIDATING TRAINING SET")
    print("=" * 60)
    
    # Load training set
    with open("outputs/training_set_v1.jsonl", 'r') as f:
        papers = [json.loads(line) for line in f]
    
    print(f"\nLoaded {len(papers)} papers from training_set_v1.jsonl")
    
    # Run diagnostics (legacy function)
    run_diagnostics(papers)
    
    # Check some statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    share_sums = [sum(paper["shares"]) for paper in papers]
    deviations = [abs(s - 1.0) for s in share_sums]
    
    print(f"Max deviation from 1.0: {max(deviations):.2e}")
    print(f"Mean deviation: {np.mean(deviations):.2e}")
    print(f"Median deviation: {np.median(deviations):.2e}")
    
    author_counts = [len(paper["all_authors"]) for paper in papers]
    print(f"\nAuthor count distribution:")
    print(f"  Min: {min(author_counts)}")
    print(f"  Max: {max(author_counts)}")
    print(f"  Mean: {np.mean(author_counts):.1f}")
    print(f"  Median: {np.median(author_counts):.0f}")


if __name__ == "__main__":
    test_validator()
    test_with_training_set()
