from agents_write_paper import render_mean_std_table_md, render_mean_std_table_tex


def test_render_mean_std_tables():
    agg = {
        "baseline": {"mean": 0.30, "std": 0.10, "n": 2},
        "novelty": {"mean": 0.60, "std": 0.05, "n": 3},
    }
    md = render_mean_std_table_md(agg)
    assert any("baseline" in line for line in md)
    assert any("novelty" in line for line in md)
    tex = render_mean_std_table_tex(agg)
    assert "tabular" in tex
    assert "baseline" in tex and "novelty" in tex

