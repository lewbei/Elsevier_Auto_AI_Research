import pathlib


def test_repo_has_paper_finder():
    assert pathlib.Path("paper_finder").exists()


def test_agents_scripts_present():
    assert pathlib.Path("agents_novelty.py").exists()
    assert pathlib.Path("llm_utils.py").exists()
    assert pathlib.Path("pdf_utils.py").exists()

