"""
Standalone Data Validator for Authorship Attribution Datasets
Validates synthetic and gold-standard paper datasets
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Results from validating a single paper"""
    paper_id: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetValidationReport:
    """Aggregated validation report for entire dataset"""
    total_papers: int
    valid_papers: int
    invalid_papers: int
    results: List[ValidationResult]
    
    @property
    def pass_rate(self) -> float:
        return self.valid_papers / self.total_papers if self.total_papers > 0 else 0.0
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 60,
            "DATASET VALIDATION REPORT",
            "=" * 60,
            f"Total Papers: {self.total_papers}",
            f"Valid Papers: {self.valid_papers}",
            f"Invalid Papers: {self.invalid_papers}",
            f"Pass Rate: {self.pass_rate:.2%}",
            "=" * 60,
        ]
        
        if self.invalid_papers > 0:
            lines.append("\nINVALID PAPERS:")
            for result in self.results:
                if not result.is_valid:
                    lines.append(f"\nPaper ID: {result.paper_id}")
                    for error in result.errors:
                        lines.append(f"  ❌ {error}")
                    for warning in result.warnings:
                        lines.append(f"  ⚠️  {warning}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Validates authorship attribution datasets
    
    Validation rules:
    1. Shares must sum to 1.0 ± 1e-9
    2. All shares must be non-negative
    3. Number of shares must match number of authors
    4. All required fields must be present
    """
    
    def __init__(self, tolerance: float = 1e-9):
        """
        Args:
            tolerance: Maximum allowed deviation from 1.0 for share sum
        """
        self.tolerance = tolerance
    
    def validate_paper(self, paper: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single paper
        
        Args:
            paper: Dictionary containing paper data with keys:
                   'paper_id', 'all_authors', 'shares', etc.
        
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        paper_id = paper.get("paper_id", "UNKNOWN")
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["paper_id", "all_authors", "shares"]
        for field in required_fields:
            if field not in paper:
                errors.append(f"Missing required field: '{field}'")
        
        if errors:
            return ValidationResult(paper_id=paper_id, is_valid=False, errors=errors)
        
        # Extract data
        authors = paper["all_authors"]
        shares = paper["shares"]
        
        # Validate shares is a list
        if not isinstance(shares, (list, np.ndarray)):
            errors.append(f"'shares' must be a list or array, got {type(shares)}")
            return ValidationResult(paper_id=paper_id, is_valid=False, errors=errors)
        
        shares = np.array(shares, dtype=float)
        
        # Check length consistency
        if len(shares) != len(authors):
            errors.append(
                f"Length mismatch: {len(authors)} authors but {len(shares)} shares"
            )
        
        # Check non-negativity
        if np.any(shares < 0):
            negative_count = np.sum(shares < 0)
            errors.append(f"Found {negative_count} negative share value(s)")
        
        # Check sum equals 1.0
        share_sum = np.sum(shares)
        deviation = abs(share_sum - 1.0)
        
        if deviation > self.tolerance:
            errors.append(
                f"Shares sum to {share_sum:.15f}, deviation from 1.0 is {deviation:.2e} "
                f"(tolerance: {self.tolerance:.2e})"
            )
        
        # Additional warnings
        if len(authors) == 0:
            warnings.append("Paper has no authors")
        
        if np.any(shares == 0):
            zero_count = np.sum(shares == 0)
            warnings.append(f"Found {zero_count} author(s) with zero contribution")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            paper_id=paper_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def validate_dataset(
        self, 
        papers: List[Dict[str, Any]], 
        verbose: bool = True
    ) -> DatasetValidationReport:
        """
        Validate entire dataset
        
        Args:
            papers: List of paper dictionaries
            verbose: If True, print progress
        
        Returns:
            DatasetValidationReport with aggregated results
        """
        results = []
        
        for i, paper in enumerate(papers):
            if verbose and (i + 1) % 1000 == 0:
                print(f"Validated {i + 1}/{len(papers)} papers...")
            
            result = self.validate_paper(paper)
            results.append(result)
        
        valid_papers = sum(1 for r in results if r.is_valid)
        invalid_papers = len(papers) - valid_papers
        
        report = DatasetValidationReport(
            total_papers=len(papers),
            valid_papers=valid_papers,
            invalid_papers=invalid_papers,
            results=results
        )
        
        if verbose:
            print(report.summary())
        
        return report
    
    def filter_valid_papers(
        self, 
        papers: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], DatasetValidationReport]:
        """
        Filter dataset to only valid papers
        
        Args:
            papers: List of paper dictionaries
        
        Returns:
            Tuple of (valid_papers_list, validation_report)
        """
        report = self.validate_dataset(papers, verbose=False)
        
        valid_papers = [
            paper for paper, result in zip(papers, report.results)
            if result.is_valid
        ]
        
        return valid_papers, report


def run_diagnostics(papers: List[Dict[str, Any]]) -> None:
    """
    Legacy function for backwards compatibility
    Runs validation and prints diagnostics
    """
    validator = DataValidator()
    report = validator.validate_dataset(papers, verbose=True)
    
    if report.invalid_papers > 0:
        print(f"\n⚠️  WARNING: {report.invalid_papers} papers failed validation")
    else:
        print(f"\n✅ All {report.total_papers} papers passed validation!")


if __name__ == "__main__":
    # Example usage
    print("DataValidator - Standalone Validation Module")
    print("=" * 60)
    print("\nExample usage:")
    print("""
    from data_validator import DataValidator
    
    validator = DataValidator(tolerance=1e-9)
    
    # Validate single paper
    result = validator.validate_paper(paper_dict)
    
    # Validate entire dataset
    report = validator.validate_dataset(papers_list)
    print(report.summary())
    
    # Filter to only valid papers
    valid_papers, report = validator.filter_valid_papers(papers_list)
    """)
