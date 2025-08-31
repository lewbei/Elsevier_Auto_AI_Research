"""Test prompt overrides functionality."""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lab.prompt_overrides import load_prompt, _truthy


def test_truthy_function():
    """Test the _truthy helper function."""
    
    # Test truthy values
    assert _truthy("1") is True
    assert _truthy("true") is True
    assert _truthy("TRUE") is True
    assert _truthy("yes") is True
    assert _truthy("on") is True
    
    # Test falsy values
    assert _truthy("0") is False
    assert _truthy("false") is False
    assert _truthy("no") is False
    assert _truthy("") is False
    assert _truthy(None) is False


def test_load_prompt_with_temp_files():
    """Test prompt loading with actual temporary files."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple prompt file
        simple_prompt = temp_path / "simple.md"
        simple_prompt.write_text("Simple prompt content", encoding="utf-8")
        
        # Create a hierarchical prompt
        category_dir = temp_path / "category"
        category_dir.mkdir()
        hier_prompt = category_dir / "subcategory.md"
        hier_prompt.write_text("Hierarchical prompt content", encoding="utf-8")
        
        # Test loading with custom base environment
        old_env = os.environ.get("TEST_PROMPTS_DIR")
        try:
            os.environ["TEST_PROMPTS_DIR"] = str(temp_path)
            
            # Test simple prompt loading
            result = load_prompt("simple", base_env="TEST_PROMPTS_DIR")
            assert result == "Simple prompt content", f"Expected 'Simple prompt content', got {result}"
            
            # Test hierarchical prompt loading
            result = load_prompt("category_subcategory", base_env="TEST_PROMPTS_DIR") 
            assert result == "Hierarchical prompt content", f"Expected 'Hierarchical prompt content', got {result}"
            
            # Test non-existent prompt
            result = load_prompt("nonexistent", base_env="TEST_PROMPTS_DIR")
            assert result is None, f"Expected None for non-existent prompt, got {result}"
            
        finally:
            # Restore environment
            if old_env is not None:
                os.environ["TEST_PROMPTS_DIR"] = old_env
            else:
                os.environ.pop("TEST_PROMPTS_DIR", None)


def test_load_prompt_security():
    """Test that load_prompt is secure against path traversal attacks."""
    
    # These should all return None safely without accessing sensitive files
    dangerous_paths = [
        "../../../etc/passwd",
        "../../../../etc/shadow", 
        "/etc/passwd",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "..\\..\\windows\\system32",
    ]
    
    for path in dangerous_paths:
        result = load_prompt(path)
        assert result is None, f"Path traversal attempt should return None: {path}"