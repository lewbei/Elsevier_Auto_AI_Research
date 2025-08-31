"""Test edge cases for the mutations module."""

import sys
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lab.mutations import propose_mutations


def test_mutations_with_invalid_types():
    """Test that mutations gracefully handle invalid input types."""
    
    # Test with invalid lr type
    spec = {"lr": "invalid"}
    result = propose_mutations(spec, max_k=2)
    assert isinstance(result, list), "Should return a list even with invalid lr"
    
    # Test with invalid max_train_steps type
    spec = {"max_train_steps": "abc"}
    result = propose_mutations(spec, max_k=2)
    assert isinstance(result, list), "Should return a list even with invalid max_train_steps"
    
    # Test with invalid input_size type
    spec = {"input_size": "xyz"}
    result = propose_mutations(spec, max_k=2)
    assert isinstance(result, list), "Should return a list even with invalid input_size"


def test_mutations_with_empty_spec():
    """Test that mutations work with empty or minimal specs."""
    
    result = propose_mutations({}, max_k=3)
    assert isinstance(result, list), "Should return a list for empty spec"
    assert len(result) <= 3, "Should respect max_k limit"


def test_mutations_with_none_values():
    """Test that mutations handle None values gracefully."""
    
    spec = {"lr": None, "optimizer": None, "max_train_steps": None}
    result = propose_mutations(spec, max_k=2)
    assert isinstance(result, list), "Should return a list even with None values"


def test_mutations_normal_case():
    """Test mutations with valid input values."""
    
    spec = {"lr": 0.001, "max_train_steps": 100, "input_size": 224, "optimizer": "adam"}
    result = propose_mutations(spec, max_k=3)
    assert isinstance(result, list), "Should return a list for valid spec"
    assert len(result) <= 3, "Should respect max_k limit"
    
    # Check that mutations actually contain modifications
    for mutation in result:
        assert isinstance(mutation, dict), "Each mutation should be a dict"
        # At least one parameter should be different from the original
        changed = any(mutation.get(k) != spec.get(k) for k in ["lr", "max_train_steps", "input_size", "optimizer"])
        assert changed, f"Mutation should change at least one parameter: {mutation}"