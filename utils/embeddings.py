import os
import json
import math
import hashlib
import pathlib
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _safe_model_dir_name(model_name: str) -> str:
    s = str(model_name).strip().replace("/", "-").replace(":", "-")
    s = "".join(ch for ch in s if ch.isalnum() or ch in {"-", "_", "."})
    return s or "model"


def _get_cfg(key: str, default: Any = None) -> Any:
    try:
        from lab.config import get
        return get(key, default)
    except Exception:
        return default


def _emb_root(model_name: str) -> pathlib.Path:
    root = pathlib.Path("data") / "embeddings" / _safe_model_dir_name(model_name)
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_gpu() -> None:
    import torch  # lazy import
    if not os.getenv("ALLOW_CPU_FOR_TESTS"):
        assert torch.cuda.is_available(), (
            "no fallback: CUDA/GPU required. set ALLOW_CPU_FOR_TESTS=1 only for tests."
        )


def _pool_last_hidden_mean(last_hidden: Any, attn_mask: Any) -> Any:
    # Mean-pool over valid tokens
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _load_hf_model(model_name: str, dtype_str: str = "float16"):
    # Avoid top-level heavy imports
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    import torch

    dtype = torch.float16 if str(dtype_str).lower() == "float16" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_TOKEN"))
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_TOKEN"),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tok, model, device


def _embed_hf(texts: List[str], model_name: str, *, batch_size: int = 8, max_length: int = 1024, dtype_str: str = "float16") -> np.ndarray:
    ensure_gpu()
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    import torch

    tok = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_TOKEN"))
    # dtype & device
    dtype = torch.float16 if str(dtype_str).lower() == "float16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_TOKEN"),
    ).to(device)
    model.eval()

    out_vecs: List[np.ndarray] = []
    for i in range(0, len(texts)):
        # small batches to fit 4GB VRAM
        batch = texts[i : i + batch_size]
        if not batch:
            continue
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc, return_dict=True)
            # Prefer pooler_output when available; else mean of last_hidden_state
            pooled = getattr(out, "pooler_output", None)
            if pooled is None or (hasattr(pooled, "numel") and pooled.numel() == 0):
                pooled = _pool_last_hidden_mean(out.last_hidden_state, enc["attention_mask"])  # type: ignore
            # L2 normalize
            x = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            out_vecs.append(x.detach().float().cpu().numpy())
    if not out_vecs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(out_vecs, axis=0)


def _hash_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def embed_and_store(
    key: str,
    text: str,
    *,
    provider: str,
    model: str,
    batch_size: int = 8,
    max_length: int = 1024,
    dtype: str = "float16",
) -> pathlib.Path:
    """Compute a single-vector embedding for `text` and store it under data/embeddings/{model}/.

    Returns the vector path.
    """
    vec = None
    if provider == "huggingface":
        vec = _embed_hf([text], model, batch_size=batch_size, max_length=max_length, dtype_str=dtype)
    elif provider == "openai":
        raise RuntimeError("openai embedding provider not yet implemented (no fallback)")
    else:
        raise RuntimeError(f"unknown embeddings provider: {provider}")

    if vec.shape[0] != 1:
        raise RuntimeError("embedding produced unexpected shape")

    root = _emb_root(model)
    h = _hash_text(text)
    shard = h[:2]
    out_dir = root / shard
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}_{h[:10]}.npy"
    np.save(str(path), vec.astype(np.float32))

    # Append meta record
    meta = {
        "key": key,
        "hash": h,
        "model": model,
        "provider": provider,
        "path": str(path),
        "dim": int(vec.shape[1]),
        "ts": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }
    (root / "meta.jsonl").open("a", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False) + "\n")
    return path


def build_faiss_cpu_index(model: str) -> Tuple[Any, List[str]]:
    """Build a FAISS-CPU inner-product index from stored vectors.

    Returns (index, keys) aligned with index IDs.
    """
    import faiss  # type: ignore
    root = _emb_root(model)
    vec_paths = sorted([str(p) for p in root.rglob("*.npy")])
    if not vec_paths:
        raise RuntimeError(f"no vectors found under {root}")
    vecs = [np.load(p).astype(np.float32) for p in vec_paths]
    X = np.vstack(vecs)
    if X.ndim != 2:
        raise RuntimeError("invalid vector array shape")
    dim = int(X.shape[1])
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(X)
    try:
        faiss.write_index(cpu_index, str(root / "index.faiss"))
    except Exception:
        pass
    return cpu_index, vec_paths

def build_faiss_gpu_index(model: str) -> Tuple[Any, List[str]]:
    """(Optional) Build a FAISS-GPU index if GPU build is available."""
    import faiss  # type: ignore
    # Fallback to CPU function if GPU resources are not available
    if not (hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")):
        raise RuntimeError("FAISS GPU features not available; switch embeddings.retrieval.index to 'faiss-cpu'.")
    root = _emb_root(model)
    vec_paths = sorted([str(p) for p in root.rglob("*.npy")])
    if not vec_paths:
        raise RuntimeError(f"no vectors found under {root}")
    vecs = [np.load(p).astype(np.float32) for p in vec_paths]
    X = np.vstack(vecs)
    if X.ndim != 2:
        raise RuntimeError("invalid vector array shape")
    dim = int(X.shape[1])
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(X)
    try:
        faiss.write_index(cpu_index, str(root / "index.faiss"))
    except Exception:
        pass
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return gpu_index, vec_paths


def knn_search(index: Any, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    import faiss  # type: ignore
    if not isinstance(queries, np.ndarray) or queries.ndim != 2:
        raise RuntimeError("queries must be 2D numpy array")
    D, I = index.search(queries.astype(np.float32), int(top_k))
    return D, I


def compute_summary_text(summary: Dict[str, Any]) -> str:
    """Compose a compact text from summary JSON fields.

    Uses: title + novelty_claims + methods + limitations.
    """
    title = str(summary.get("title") or "")
    parts: List[str] = [title]
    for key in ("novelty_claims", "methods", "limitations"):
        vals = summary.get(key) or []
        for v in vals:
            s = str(v).strip()
            if s:
                parts.append(s)
    return "\n".join(parts)
