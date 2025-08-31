from pathlib import Path
from lab.codegen_utils import write_generated_aug


def test_write_generated_aug(tmp_path: Path):
    out = tmp_path / "generated_aug.py"
    p = write_generated_aug("jitter rotate erase blur", out)
    assert p.exists()
    code = p.read_text(encoding="utf-8")
    assert "class GeneratedAug" in code
    assert "RandomRotation" in code

