"""Test stage manager functionality."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_stage_manager_imports():
    """Test that stage_manager imports correctly and has expected functions."""
    
    from agents.stage_manager import _read_json, _summary_goal, _load_novelty, run_stage, main
    
    # Test that functions exist and are callable
    assert callable(_read_json)
    assert callable(_summary_goal) 
    assert callable(_load_novelty)
    assert callable(run_stage)
    assert callable(main)


def test_read_json_helper():
    """Test the _read_json helper function."""
    
    from agents.stage_manager import _read_json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {"test": "value", "number": 42}
        json.dump(test_data, f)
        temp_path = Path(f.name)
    
    try:
        # Test reading valid JSON
        result = _read_json(temp_path)
        assert result == test_data, f"Expected {test_data}, got {result}"
        
        # Test reading non-existent file
        result = _read_json(Path("nonexistent.json"))
        assert result == {}, "Should return empty dict for non-existent file"
        
    finally:
        temp_path.unlink(missing_ok=True)


def test_summary_goal_function():
    """Test the _summary_goal function with mocked summary file."""
    
    from agents.stage_manager import _summary_goal, RUNS_DIR
    
    # Create temporary runs directory and summary file
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            runs_dir = Path("runs")
            runs_dir.mkdir(exist_ok=True)
            
            # Test with goal_reached=True
            summary_file = runs_dir / "summary.json"
            summary_file.write_text(json.dumps({"goal_reached": True}), encoding="utf-8")
            result = _summary_goal()
            assert result is True, "Should return True when goal_reached is True"
            
            # Test with goal_reached=False
            summary_file.write_text(json.dumps({"goal_reached": False}), encoding="utf-8")
            result = _summary_goal()
            assert result is False, "Should return False when goal_reached is False"
            
            # Test with missing file
            summary_file.unlink()
            result = _summary_goal()
            assert result is False, "Should return False when summary file is missing"
            
        finally:
            os.chdir(old_cwd)


def test_env_override_safety():
    """Test environment variable override and restoration."""
    
    # Test that environment variables are properly restored
    original_value = os.environ.get("TEST_VAR")
    test_key = "TEST_VAR"
    test_value = "test_value"
    
    try:
        # Set a test environment variable
        os.environ[test_key] = "original"
        
        # Create a backup dict like run_stage does
        bak = {test_key: os.getenv(test_key)}
        os.environ[test_key] = test_value
        
        # Verify change
        assert os.environ[test_key] == test_value
        
        # Restore like run_stage does
        for k, v in bak.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        
        # Verify restoration
        assert os.environ[test_key] == "original"
        
    finally:
        # Clean up
        if original_value is not None:
            os.environ[test_key] = original_value
        else:
            os.environ.pop(test_key, None)