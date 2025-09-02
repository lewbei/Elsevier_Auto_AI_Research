def test_render_md_handles_string_summary():
    from agents.write_paper import _render_md
    title = "Test Title"
    novelty = {}
    plan = {}
    # summary is intentionally a string (valid JSON could be a string)
    summary = "not a dict"
    md = _render_md(title, novelty, plan, summary)  # should not raise
    assert isinstance(md, str)
    assert md.startswith("# ")

