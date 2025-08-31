import llm_utils


def test_chat_text_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_utils, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm_utils, "LLM_CACHE_ENABLED", True)

    calls = {"count": 0}

    def fake_chat_text(messages, *, temperature=0.2, model=None, timeout=60, max_tries=4):
        calls["count"] += 1
        return f"reply {calls['count']}"

    monkeypatch.setattr(llm_utils, "chat_text", fake_chat_text)
    msgs = [{"role": "user", "content": "hi"}]

    assert llm_utils.chat_text_cached(msgs) == "reply 1"
    assert llm_utils.chat_text_cached(msgs) == "reply 1"
    assert calls["count"] == 1

    assert llm_utils.chat_text_cached(msgs, cache=False) == "reply 2"
    assert calls["count"] == 2
