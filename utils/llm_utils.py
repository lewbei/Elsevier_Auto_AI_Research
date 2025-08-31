"""Lightweight helpers for calling the DeepSeek chat API.

This module centralizes HTTP calls, basic retry logic, and a tiny on-disk
cache so other modules can interact with the LLM through a small surface
area. Functions here intentionally avoid advanced CLI or parser features to
keep usage simple and focused on programmatic access.
"""

import os
import time
import json
import requests
from typing import Any, Dict, List, Optional
import hashlib
import pathlib
import datetime
import atexit
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_URL = os.getenv("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

_UA = "elsevier-pipeline/agents (github.com/openai/codex-cli)"

LLM_CACHE_DIR = pathlib.Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE", "true").lower() in {"1", "true", "yes"}
LLM_LOG_ENABLED = os.getenv("LLM_LOG", "").lower() in {"1", "true", "yes"}
LLM_COST_INPUT_PER_1K = float(os.getenv("LLM_COST_INPUT_PER_1K", "0") or 0)
LLM_COST_OUTPUT_PER_1K = float(os.getenv("LLM_COST_OUTPUT_PER_1K", "0") or 0)

# In-process usage totals
_TOTAL_PROMPT_TOKENS = 0
_TOTAL_COMPLETION_TOKENS = 0
_TOTAL_COST = 0.0
_SESSION_EVENTS = 0


class LLMError(RuntimeError):
    """Raised when the LLM client encounters a fatal error."""


def _headers() -> Dict[str, str]:
    """Return HTTP headers for DeepSeek requests, validating the API key."""
    if not DEEPSEEK_API_KEY:
        raise LLMError("DEEPSEEK_API_KEY not set")
    return {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": _UA,
    }


def _is_verbose() -> bool:
    lvl = (os.getenv("LOG_LEVEL", "") or os.getenv("LOGLEVEL", "") or os.getenv("LEVEL", "")).strip().lower()
    if lvl in {"trace", "debug"}:
        return True
    if str(os.getenv("VERBOSE", "")).lower() in {"1", "true", "yes", "on"}:
        return True
    return False


def _llm_log(kind: str, payload: Dict[str, Any], response: Any) -> None:
    """Best-effort logging of LLM payload/response under logs/llm/.
    Avoids writing secrets by not including headers. No-op on errors.
    """
    try:
        if not (LLM_LOG_ENABLED or _is_verbose()):
            return
        logs_dir = pathlib.Path("logs") / "llm"
        logs_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fpath = logs_dir / f"{ts}_{kind}_{key}.json"
        rec = {"kind": kind, "ts": ts, "payload": payload, "response": response}
        fpath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        if _is_verbose():
            print(f"[LLM LOG] {fpath}")
    except Exception:
        return


def _usage_path() -> pathlib.Path:
    p = pathlib.Path("logs") / "llm"
    p.mkdir(parents=True, exist_ok=True)
    return p / "usage.jsonl"


def _record_usage(kind: str, payload: Dict[str, Any], usage: Dict[str, Any], cache_hit: bool = False) -> None:
    global _TOTAL_PROMPT_TOKENS, _TOTAL_COMPLETION_TOKENS, _TOTAL_COST, _SESSION_EVENTS
    try:
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        model = payload.get("model")
        # cost (best-effort) if env pricing provided
        cost = (pt / 1000.0) * LLM_COST_INPUT_PER_1K + (ct / 1000.0) * LLM_COST_OUTPUT_PER_1K
        rec = {
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,
            "model": model,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": int(usage.get("total_tokens") or (pt + ct)),
            "cost": round(cost, 6),
            "cache_hit": bool(cache_hit),
        }
        with _usage_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not cache_hit:
            _TOTAL_PROMPT_TOKENS += pt
            _TOTAL_COMPLETION_TOKENS += ct
            _TOTAL_COST += cost
            _SESSION_EVENTS += 1
    except Exception:
        return


def _extract_usage_from_response(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        u = data.get("usage") or {}
        pt = int(u.get("prompt_tokens") or 0)
        ct = int(u.get("completion_tokens") or 0)
        tt = int(u.get("total_tokens") or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def chat_json(system: str, user: str, *, temperature: float = 0.0, model: Optional[str] = None,
              timeout: int = 60, max_tries: int = 4) -> Dict[str, Any]:
    """
    Call DeepSeek chat with JSON response_format and return parsed dict.
    Retries on 429/5xx with simple backoff and honors Retry-After when present.
    """
    payload = {
        "model": model or DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    delay = 2
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.post(DEEPSEEK_CHAT_URL, headers=_headers(), json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_tries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue

        if r.status_code == 200:
            try:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                js = json.loads(content)
                try:
                    usage = _extract_usage_from_response(data)
                    _record_usage("chat_json", payload, usage, cache_hit=False)
                except Exception:
                    pass
                try:
                    _llm_log("chat_json", payload, js)
                except Exception:
                    pass
                return js
            except Exception as exc:
                raise LLMError(f"DeepSeek response parse error: {exc}")

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            sleep_for = int(ra) if ra and ra.isdigit() else delay
            time.sleep(max(sleep_for, 1))
            delay = min(delay * 2, 60)
            continue

        if 500 <= r.status_code < 600 and attempt < max_tries:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        # Non-retryable
        snippet = r.text[:300]
        raise LLMError(f"DeepSeek error {r.status_code}: {snippet}")

    raise LLMError(f"DeepSeek request failed after retries: {last_err}")


def _hash_payload(prefix: str, payload: Dict[str, Any]) -> str:
    """Create a deterministic hash of a payload for cache keying."""
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8"))
    h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def chat_json_cached(system: str, user: str, *, temperature: float = 0.0, model: Optional[str] = None,
                     timeout: int = 60, max_tries: int = 4) -> Dict[str, Any]:
    """Cached variant of :func:`chat_json` using a simple on-disk store."""
    payload = {
        "model": model or DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if not LLM_CACHE_ENABLED:
        return chat_json(system, user, temperature=temperature, model=model, timeout=timeout, max_tries=max_tries)
    key = _hash_payload("chat_json", payload)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fpath = LLM_CACHE_DIR / f"{key}.json"
    if fpath.exists():
        try:
            obj = json.loads(fpath.read_text(encoding="utf-8"))
            # Cache hits have zero cost; record for visibility
            try:
                _record_usage("chat_json", payload, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, cache_hit=True)
            except Exception:
                pass
            return obj
        except Exception:
            pass
    js = chat_json(system, user, temperature=temperature, model=model, timeout=timeout, max_tries=max_tries)
    try:
        fpath.write_text(json.dumps(js, ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception:
        pass
    return js


def chat_text(messages: List[Dict[str, str]], *, temperature: float = 0.2, model: Optional[str] = None,
              timeout: int = 60, max_tries: int = 4) -> str:
    """Return raw text from the chat API given a sequence of messages."""
    payload = {
        "model": model or DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    last_err = None
    delay = 2
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.post(DEEPSEEK_CHAT_URL, headers=_headers(), json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_tries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue

        if r.status_code == 200:
            try:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                try:
                    usage = _extract_usage_from_response(data)
                    _record_usage("chat_text", payload, usage, cache_hit=False)
                except Exception:
                    pass
                try:
                    _llm_log("chat_text", payload, text)
                except Exception:
                    pass
                return text
            except Exception as exc:
                raise LLMError(f"DeepSeek response parse error: {exc}")

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            sleep_for = int(ra) if ra and ra.isdigit() else delay
            time.sleep(max(sleep_for, 1))
            delay = min(delay * 2, 60)
            continue

        if 500 <= r.status_code < 600 and attempt < max_tries:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        snippet = r.text[:300]
        raise LLMError(f"DeepSeek error {r.status_code}: {snippet}")

    raise LLMError(f"DeepSeek request failed after retries: {last_err}")


def chat_text_cached(messages: List[Dict[str, str]], *, temperature: float = 0.2, model: Optional[str] = None,
                     timeout: int = 60, max_tries: int = 4, cache: bool = True) -> str:
    """Cached variant of :func:`chat_text` with optional per-call caching.

    The cache uses a hash of the request payload as the filename. Set
    ``cache=False`` to bypass disk writes for sensitive interactions.
    Exceptions during cache I/O are suppressed to favor successful LLM calls.
    """
    payload = {
        "model": model or DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if not (LLM_CACHE_ENABLED and cache):
        return chat_text(messages, temperature=temperature, model=model, timeout=timeout, max_tries=max_tries)
    key = _hash_payload("chat_text", payload)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fpath = LLM_CACHE_DIR / f"{key}.txt"
    if fpath.exists():
        try:
            with fpath.open("r", encoding="utf-8") as fh:
                txt = fh.read()
                try:
                    _record_usage("chat_text", payload, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, cache_hit=True)
                except Exception:
                    pass
                return txt
        except Exception:
            pass
    text = chat_text(messages, temperature=temperature, model=model, timeout=timeout, max_tries=max_tries)
    try:
        tmp_path = fpath.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
        tmp_path.replace(fpath)
    except Exception:
        pass
    return text


def _session_summary_path() -> pathlib.Path:
    base = pathlib.Path("logs") / "llm"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"session_{os.getpid()}_{ts}.json"


def _totals_path() -> pathlib.Path:
    base = pathlib.Path("logs") / "llm"
    base.mkdir(parents=True, exist_ok=True)
    return base / "totals.json"


def _write_session_summary() -> None:
    try:
        if (_TOTAL_PROMPT_TOKENS + _TOTAL_COMPLETION_TOKENS) <= 0 and _SESSION_EVENTS == 0:
            return
        summary = {
            "prompt_tokens": _TOTAL_PROMPT_TOKENS,
            "completion_tokens": _TOTAL_COMPLETION_TOKENS,
            "cost": round(_TOTAL_COST, 6),
            "events": _SESSION_EVENTS,
            "pid": os.getpid(),
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _session_summary_path().write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        # Update global totals (best-effort, not locked)
        tot_path = _totals_path()
        try:
            if tot_path.exists():
                agg = json.loads(tot_path.read_text(encoding="utf-8"))
            else:
                agg = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "sessions": 0}
        except Exception:
            agg = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "sessions": 0}
        agg["prompt_tokens"] = int(agg.get("prompt_tokens", 0)) + _TOTAL_PROMPT_TOKENS
        agg["completion_tokens"] = int(agg.get("completion_tokens", 0)) + _TOTAL_COMPLETION_TOKENS
        agg["cost"] = float(agg.get("cost", 0.0)) + _TOTAL_COST
        agg["sessions"] = int(agg.get("sessions", 0)) + 1
        tot_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


atexit.register(_write_session_summary)
