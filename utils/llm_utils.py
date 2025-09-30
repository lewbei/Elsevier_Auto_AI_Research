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
from lab.config import get, get_bool

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_URL = os.getenv("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

_CFG_PROVIDER = get("llm.provider", None)
_CFG_CHAT_URL = get("llm.chat_url", None)  # optional hard override; prefer provider defaults
_CFG_MODEL = get("llm.model", None)
_CFG_API_ENV = get("llm.api_key_env", None)
_CFG_DEFAULT = get("llm.default", None)  # e.g., "gpt-5-mini"
_CFG_CUSTOM = get("llm.custom", None)    # e.g., "deepseek"
_CFG_USE = get("llm.use", None)          # "default" or "custom"

LLM_PROVIDER = (str(_CFG_PROVIDER or os.getenv("LLM_PROVIDER", "")).strip().lower()) or "openai"

def _default_url_for(provider: str) -> str:
    if provider == "openai":
        return "https://api.openai.com/v1/chat/completions"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1/chat/completions"
    return DEEPSEEK_CHAT_URL

def _default_model_for(provider: str) -> str:
    if provider == "openai":
        # Default to GPT‑5 family as requested
        return "gpt-5-mini"
    if provider == "openrouter":
        return "openai/gpt-5-mini"
    return DEEPSEEK_MODEL

def _default_api_env(provider: str) -> str:
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    return "DEEPSEEK_API_KEY"

LLM_CHAT_URL = str(_CFG_CHAT_URL or _default_url_for(LLM_PROVIDER))
LLM_MODEL = str(_CFG_MODEL or _default_model_for(LLM_PROVIDER))
_API_ENV_NAME = str(_CFG_API_ENV or _default_api_env(LLM_PROVIDER))
LLM_API_KEY = os.getenv(_API_ENV_NAME) or DEEPSEEK_API_KEY


def _normalize_model(provider: str, model: Optional[str]) -> str:
    """Normalize model names per provider.

    For provider 'openai', allow non-GPT‑5 models when a custom chat_url is configured
    or when llm.allow_custom_openai_models is true. This supports OpenAI‑compatible
    endpoints (e.g., Z.AI GLM‑4.5) without requiring the 'custom' profile path.
    Otherwise restrict to GPT‑5 mini/nano and fall back to 'gpt-5-mini'.
    """
    from lab.config import get as _get  # lazy import to avoid cycles

    m = (model or LLM_MODEL) or ""
    m = str(m).strip()
    if provider == "openai":
        try:
            allow_custom = bool(_get("llm.allow_custom_openai_models", False))
        except Exception:
            allow_custom = False
        # If a custom chat_url is configured, allow arbitrary model (OpenAI‑compatible endpoint)
        custom_endpoint = (LLM_CHAT_URL != _default_url_for("openai"))
        if allow_custom or custom_endpoint:
            return m
        allowed = {"gpt-5-mini", "gpt-5-nano"}
        if m not in allowed:
            try:
                print(f"[LLM] OpenAI model '{m}' not allowed; using 'gpt-5-mini'.")
            except Exception:
                pass
            return "gpt-5-mini"
    return m


def _strict_guard(provider: str, chat_url: str, model: str) -> None:
    """Enforce strict YAML config if enabled to avoid unintended fallbacks."""
    try:
        strict = get_bool("llm.strict", False)
    except Exception:
        strict = False
    if not strict:
        return
    # Ensure API key exists for resolved provider
    api_env = _default_api_env(provider)
    if not (os.getenv(api_env) or LLM_API_KEY):
        raise LLMError(f"LLM strict: API key env '{api_env}' not set")
    # If using OpenAI provider with default endpoint, only allow GPT-5 unless custom allowed
    if provider == "openai" and chat_url.strip().lower() == _default_url_for("openai"):
        m = (model or "").lower()
        try:
            allow_custom = bool(get("llm.allow_custom_openai_models", False))
        except Exception:
            allow_custom = False
        if not (m.startswith("gpt-5-") or m == "gpt-5" or allow_custom):
            raise LLMError("LLM strict: default OpenAI endpoint requires GPT-5 or allow_custom_openai_models=true")


def _is_gpt5(provider: str, model: Optional[str]) -> bool:
    try:
        if provider != "openai":
            return False
        m = (model or "").lower()
        return m.startswith("gpt-5-") or m == "gpt-5"
    except Exception:
        return False


def _alias_to_profile(alias: Optional[str]) -> Dict[str, str]:
    """Map simple aliases (gpt-5-mini|gpt-5-nano|deepseek) to provider/model/api_env."""
    name = (alias or "").strip().lower()
    if not name:
        # fall back to global defaults
        return {"provider": LLM_PROVIDER, "model": LLM_MODEL, "api_env": _API_ENV_NAME, "chat_url": LLM_CHAT_URL}
    if name in {"gpt-5-mini", "gpt-5-nano"}:
        prov = "openai"
        mod = name
        env = "OPENAI_API_KEY"
        url = _default_url_for(prov)
        return {"provider": prov, "model": mod, "api_env": env, "chat_url": url}
    if name in {"deepseek", "deepseek-chat"}:
        prov = "deepseek"
        mod = "deepseek-chat"
        env = "DEEPSEEK_API_KEY"
        url = _default_url_for(prov)
        return {"provider": prov, "model": mod, "api_env": env, "chat_url": url}
    # OpenRouter variant: treat unknown starting with openrouter/ as openrouter
    if name.startswith("openrouter/") or name.startswith("openai/"):
        prov = "openrouter"
        mod = name
        env = "OPENROUTER_API_KEY"
        url = _default_url_for(prov)
        return {"provider": prov, "model": mod, "api_env": env, "chat_url": url}
    # Default to global
    return {"provider": LLM_PROVIDER, "model": LLM_MODEL, "api_env": _API_ENV_NAME, "chat_url": LLM_CHAT_URL}


def _resolve_profile(selected: Optional[str]) -> Dict[str, str]:
    """Resolve active profile using simple llm.default/custom mapping.

    selected: "default" | "custom" | None
    """
    name = (selected or str(_CFG_USE or "default")).strip().lower()
    if name == "custom":
        return _alias_to_profile(str(_CFG_CUSTOM or ""))
    # default
    return _alias_to_profile(str(_CFG_DEFAULT or ""))

_UA = "elsevier-pipeline/agents (github.com/openai/codex-cli)"

LLM_CACHE_DIR = pathlib.Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE", "true").lower() in {"1", "true", "yes"}
LLM_LOG_ENABLED = os.getenv("LLM_LOG", "").lower() in {"1", "true", "yes"}
LLM_COST_INPUT_PER_1K = float(os.getenv("LLM_COST_INPUT_PER_1K", "0") or 0)
LLM_COST_OUTPUT_PER_1K = float(os.getenv("LLM_COST_OUTPUT_PER_1K", "0") or 0)
LLM_COST_CACHED_INPUT_PER_1K = float(os.getenv("LLM_COST_CACHED_INPUT_PER_1K", "0") or 0)

# In-process usage totals
_TOTAL_PROMPT_TOKENS = 0
_TOTAL_COMPLETION_TOKENS = 0
_TOTAL_COST = 0.0
_SESSION_EVENTS = 0


class LLMError(RuntimeError):
    """Raised when the LLM client encounters a fatal error."""


def _headers(provider: str, api_key: Optional[str]) -> Dict[str, str]:
    """Return HTTP headers for requests, validating the API key."""
    if not api_key:
        raise LLMError("LLM API key not set")
    h = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _UA,
    }
    if provider == "openrouter":
        # Optional courtesy headers for OpenRouter
        ref = os.getenv("OPENROUTER_SITE", "")
        title = os.getenv("OPENROUTER_TITLE", "elsevier-pipeline")
        if ref:
            h["HTTP-Referer"] = ref
            if title:
                h["X-Title"] = title
    return h


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


def _price_table(provider: str, model: str) -> Dict[str, float]:
    """Return per-1K token pricing defaults for known models.

    Keys: input, cached_input, output
    """
    provider = (provider or "").lower()
    m = (model or "").lower()
    # OpenAI GPT‑5 family defaults
    if provider == "openai":
        if m.startswith("gpt-5-mini") or m == "gpt-5-mini" or m == "gpt-5":
            return {"input": 0.25, "cached_input": 0.025, "output": 2.00}
        if m.startswith("gpt-5-nano") or m == "gpt-5-nano":
            return {"input": 0.05, "cached_input": 0.005, "output": 0.40}
    # Fallback: zeros (no pricing info); caller may override via env
    return {"input": 0.0, "cached_input": 0.0, "output": 0.0}


def _resolve_rates(provider: str, model: str) -> Dict[str, float]:
    """Resolve effective rates, preferring env overrides; else model defaults."""
    table = _price_table(provider, model)
    inp = LLM_COST_INPUT_PER_1K if LLM_COST_INPUT_PER_1K > 0 else table["input"]
    outp = LLM_COST_OUTPUT_PER_1K if LLM_COST_OUTPUT_PER_1K > 0 else table["output"]
    cinp = LLM_COST_CACHED_INPUT_PER_1K if LLM_COST_CACHED_INPUT_PER_1K > 0 else table["cached_input"]
    return {"input": float(inp), "output": float(outp), "cached_input": float(cinp)}


def _record_usage(kind: str, payload: Dict[str, Any], usage: Dict[str, Any], cache_hit: bool = False, cache_key: Optional[str] = None) -> None:
    global _TOTAL_PROMPT_TOKENS, _TOTAL_COMPLETION_TOKENS, _TOTAL_COST, _SESSION_EVENTS
    try:
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        model = payload.get("model")
        # cost (best-effort) with per-model defaults and env overrides
        provider = str(payload.get("provider") or LLM_PROVIDER)
        model = str(payload.get("model") or LLM_MODEL)
        rates = _resolve_rates(provider, model)
        if cache_hit:
            cost = (pt / 1000.0) * rates["cached_input"] + (ct / 1000.0) * rates["output"]
            pricing = "cached_input"
        else:
            cost = (pt / 1000.0) * rates["input"] + (ct / 1000.0) * rates["output"]
            pricing = "standard"
        rec = {
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,
            "model": model,
            "provider": str(payload.get("provider") or ""),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": int(usage.get("total_tokens") or (pt + ct)),
            "cost": round(cost, 6),
            "cache_hit": bool(cache_hit),
            "run_id": os.getenv("LLM_RUN_ID", ""),
            "stage": os.getenv("LLM_STAGE", ""),
            "pricing": pricing,
            "input_rate": rates.get("input", 0.0),
            "cached_input_rate": rates.get("cached_input", 0.0),
            "output_rate": rates.get("output", 0.0),
            "cache_key": cache_key or "",
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
              profile: Optional[str] = None,
              timeout: int = 60, max_tries: int = 4) -> Dict[str, Any]:
    """
    Call DeepSeek chat with JSON response_format and return parsed dict.
    Retries on 429/5xx with simple backoff and honors Retry-After when present.
    """
    prof = _resolve_profile(profile)
    provider = prof["provider"]
    chat_url = prof["chat_url"]
    api_env = prof["api_env"]
    api_key = os.getenv(api_env) or LLM_API_KEY
    use_model = _normalize_model(provider, model or prof["model"])
    _strict_guard(provider, chat_url, use_model)
    payload = {
        "model": use_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        # provider/chat_url are kept only for local logging/caching; they are not
        # sent to the remote API to preserve OpenAI-compat compliance.
        "provider": provider,
        "chat_url": chat_url,
    }
    # GPT‑5 family (OpenAI) does not accept temperature; omit key
    if not _is_gpt5(provider, use_model):
        payload["temperature"] = temperature

    # Optional streaming for live token output (best-effort)
    stream_enabled = get_bool("llm.stream", False) or (str(os.getenv("LLM_STREAM", "")).lower() in {"1", "true", "yes", "on"})
    if stream_enabled:
        rf = payload.get("response_format", {})
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            stream_enabled = False
        else:
            payload["stream"] = True

    # Build API body without non-spec keys
    payload_api = {k: v for k, v in payload.items() if k not in {"provider", "chat_url"}}

    last_err = None
    delay = 2
    for attempt in range(1, max_tries + 1):
        try:
            if stream_enabled:
                r = requests.post(chat_url, headers=_headers(provider, api_key), json=payload_api, timeout=timeout, stream=True)
            else:
                r = requests.post(chat_url, headers=_headers(provider, api_key), json=payload_api, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_tries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue

        if r.status_code == 200:
            try:
                if stream_enabled:
                    full_text = ""
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            chunk = line[len("data:"):].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                js_line = json.loads(chunk)
                                delta = js_line.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content") or ""
                                if token:
                                    full_text += token
                                    try:
                                        print(token, end="", flush=True)
                                    except Exception:
                                        pass
                            except Exception:
                                continue
                    try:
                        print()
                    except Exception:
                        pass
                    try:
                        js = json.loads(full_text)
                    except Exception as exc:
                        raise LLMError(f"LLM stream parse error: {exc}")
                    try:
                        _llm_log("chat_json_stream", payload, js)
                    except Exception:
                        pass
                    return js
                else:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    js = json.loads(content)
                    try:
                        usage = _extract_usage_from_response(data)
                        key = _hash_payload("chat_json", payload)
                        _record_usage("chat_json", payload, usage, cache_hit=False, cache_key=key)
                    except Exception:
                        pass
                    try:
                        _llm_log("chat_json", payload, js)
                    except Exception:
                        pass
                    return js
            except Exception as exc:
                raise LLMError(f"LLM response parse error: {exc}")

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
        raise LLMError(f"LLM error {r.status_code}: {snippet}")

    raise LLMError(f"LLM request failed after retries: {last_err}")


def _hash_payload(prefix: str, payload: Dict[str, Any]) -> str:
    """Create a deterministic hash of a payload for cache keying."""
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8"))
    h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def chat_json_cached(system: str, user: str, *, temperature: float = 0.0, model: Optional[str] = None,
                     profile: Optional[str] = None,
                     timeout: int = 60, max_tries: int = 4) -> Dict[str, Any]:
    """Cached variant of :func:`chat_json` using a simple on-disk store."""
    prof = _resolve_profile(profile)
    provider = prof["provider"]
    chat_url = prof["chat_url"]
    api_env = prof["api_env"]
    api_key = os.getenv(api_env) or LLM_API_KEY
    cache_model = _normalize_model(provider, model or prof["model"])
    _strict_guard(provider, chat_url, cache_model)
    payload = {
        "model": cache_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "provider": provider,
        "chat_url": chat_url,
    }
    if not _is_gpt5(provider, cache_model):
        payload["temperature"] = temperature
    if not LLM_CACHE_ENABLED:
        return chat_json(system, user, temperature=temperature, model=cache_model, profile=profile, timeout=timeout, max_tries=max_tries)
    key = _hash_payload("chat_json", payload)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fpath = LLM_CACHE_DIR / f"{key}.json"
    if fpath.exists():
        try:
            obj = json.loads(fpath.read_text(encoding="utf-8"))
            # Cache hit: attempt to load prior usage by cache_key to estimate pricing
            try:
                key = key  # already computed
                usage = _find_usage_for_cache_key("chat_json", key)
                if not usage:
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                _record_usage("chat_json", payload, usage, cache_hit=True, cache_key=key)
            except Exception:
                pass
            return obj
        except Exception:
            pass
    js = chat_json(system, user, temperature=temperature, model=cache_model, profile=profile, timeout=timeout, max_tries=max_tries)
    try:
        fpath.write_text(json.dumps(js, ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception:
        pass
    return js


def chat_text(messages: List[Dict[str, str]], *, temperature: float = 0.2, model: Optional[str] = None,
              profile: Optional[str] = None,
              timeout: int = 60, max_tries: int = 4) -> str:
    """Return raw text from the chat API given a sequence of messages."""
    prof = _resolve_profile(profile)
    provider = prof["provider"]
    chat_url = prof["chat_url"]
    api_env = prof["api_env"]
    api_key = os.getenv(api_env) or LLM_API_KEY
    text_model = _normalize_model(provider, model or prof["model"])
    _strict_guard(provider, chat_url, text_model)
    payload = {
        "model": text_model,
        "messages": messages,
        # provider/chat_url are for local logging/caching only; not sent to API
        "provider": provider,
        "chat_url": chat_url,
    }
    if not _is_gpt5(provider, text_model):
        payload["temperature"] = temperature

    # Optional streaming for live token output (best-effort)
    stream_enabled = get_bool("llm.stream", False) or (str(os.getenv("LLM_STREAM", "")).lower() in {"1", "true", "yes", "on"})
    if stream_enabled:
        payload["stream"] = True

    # Build API-compliant body (exclude non-spec keys)
    payload_api = {k: v for k, v in payload.items() if k not in {"provider", "chat_url"}}

    last_err = None
    delay = 2
    for attempt in range(1, max_tries + 1):
        try:
            if stream_enabled:
                r = requests.post(chat_url, headers=_headers(provider, api_key), json=payload_api, timeout=timeout, stream=True)
            else:
                r = requests.post(chat_url, headers=_headers(provider, api_key), json=payload_api, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            if attempt == max_tries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue

        if r.status_code == 200:
            try:
                if stream_enabled:
                    # SSE-style streaming: accumulate tokens and print as they arrive
                    full_text = ""
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith(":"):
                            continue  # comment/keepalive
                        if line.startswith("data:"):
                            chunk = line[len("data:"):].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                js = json.loads(chunk)
                                delta = js.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content") or ""
                                if token:
                                    full_text += token
                                    try:
                                        print(token, end="", flush=True)
                                    except Exception:
                                        pass
                            except Exception:
                                # Ignore malformed lines
                                continue
                    # Newline after stream for readability
                    try:
                        print()
                    except Exception:
                        pass
                    try:
                        _llm_log("chat_text_stream", payload, full_text)
                    except Exception:
                        pass
                    return full_text
                else:
                    data = r.json()
                    text = data["choices"][0]["message"]["content"]
                    try:
                        usage = _extract_usage_from_response(data)
                        key = _hash_payload("chat_text", payload)
                        _record_usage("chat_text", payload, usage, cache_hit=False, cache_key=key)
                    except Exception:
                        pass
                    try:
                        _llm_log("chat_text", payload, text)
                    except Exception:
                        pass
                    return text
            except Exception as exc:
                raise LLMError(f"LLM response parse error: {exc}")

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
        raise LLMError(f"LLM error {r.status_code}: {snippet}")

    raise LLMError(f"LLM request failed after retries: {last_err}")


def chat_text_cached(messages: List[Dict[str, str]], *, temperature: float = 0.2, model: Optional[str] = None,
                     profile: Optional[str] = None,
                     timeout: int = 60, max_tries: int = 4, cache: bool = True) -> str:
    """Cached variant of :func:`chat_text` with optional per-call caching.

    The cache uses a hash of the request payload as the filename. Set
    ``cache=False`` to bypass disk writes for sensitive interactions.
    Exceptions during cache I/O are suppressed to favor successful LLM calls.
    """
    prof = _resolve_profile(profile)
    provider = prof["provider"]
    chat_url = prof["chat_url"]
    cache_model = _normalize_model(provider, model or prof["model"])
    _strict_guard(provider, chat_url, cache_model)
    payload = {
        "model": cache_model,
        "messages": messages,
        "provider": provider,
        "chat_url": chat_url,
    }
    if not _is_gpt5(provider, cache_model):
        payload["temperature"] = temperature
    if not (LLM_CACHE_ENABLED and cache):
        return chat_text(messages, temperature=temperature, model=cache_model, profile=profile, timeout=timeout, max_tries=max_tries)
    key = _hash_payload("chat_text", payload)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fpath = LLM_CACHE_DIR / f"{key}.txt"
    if fpath.exists():
        try:
            with fpath.open("r", encoding="utf-8") as fh:
                txt = fh.read()
                try:
                    key = key
                    usage = _find_usage_for_cache_key("chat_text", key)
                    if not usage:
                        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    _record_usage("chat_text", payload, usage, cache_hit=True, cache_key=key)
                except Exception:
                    pass
                return txt
        except Exception:
            pass
    text = chat_text(messages, temperature=temperature, model=cache_model, profile=profile, timeout=timeout, max_tries=max_tries)
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


def _find_usage_for_cache_key(kind: str, cache_key: str) -> Dict[str, Any]:
    """Lookup last recorded usage for a given cache key, if present."""
    try:
        path = _usage_path()
        if not path.exists():
            return {}
        last: Optional[Dict[str, Any]] = None
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") == kind and rec.get("cache_key") == cache_key:
                    last = rec
        if last is None:
            return {}
        return {
            "prompt_tokens": int(last.get("prompt_tokens") or 0),
            "completion_tokens": int(last.get("completion_tokens") or 0),
            "total_tokens": int(last.get("total_tokens") or 0),
        }
    except Exception:
        return {}
