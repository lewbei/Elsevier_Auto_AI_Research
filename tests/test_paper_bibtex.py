from pathlib import Path
from agents.write_paper import build_bibtex_from_csv


def test_build_bibtex_from_csv(tmp_path: Path):
    csv_path = tmp_path / "abstract_screen_deepseek.csv"
    csv_path.write_text(
        "title,year,coverDate,doi,pii,prism_url,openaccess,relevant,reason,pdf_path\n"
        "Paper A,2024,2024-01-01,10.1234/a,,url,true,true,,\n"
        "Paper B,2025,2025-02-02,10.5678/b,,url,true,true,,\n",
        encoding="utf-8",
    )
    bib = build_bibtex_from_csv(csv_path, tmp_path)
    assert bib is not None
    content = bib.read_text(encoding="utf-8")
    assert "@article{" in content
    assert "10.1234/a" in content
    assert "2024" in content
