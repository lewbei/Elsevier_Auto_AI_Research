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
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_URL = os.getenv("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

_UA = "elsevier-pipeline/agents (github.com/openai/codex-cli)"

LLM_CACHE_DIR = pathlib.Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE", "true").lower() in {"1", "true", "yes"}


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
                return json.loads(content)
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
            return json.loads(fpath.read_text(encoding="utf-8"))
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
                return data["choices"][0]["message"]["content"]
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
                return fh.read()
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
