from __future__ import annotations
import json
import re
import pathlib
from typing import Dict, List, Any, Set

DATA_DIR = pathlib.Path("data")
FACETS_PATH = DATA_DIR / "facets.json"

# Lightweight lexicons; extend as needed
LEX = {
    "backbones": [
        "resnet18","resnet34","resnet50","resnet101","vit","swin","convnext","efficientnet",
        "yolov5","yolov7","yolov8","yolov9","yolox","detr","deformable detr","fcos","centernet",
        "unet","deeplab","hrnet","mamba","llama","clip","sdxl","sam","dinov2"
    ],
    "tasks": ["classification","segmentation","detection","captioning","retrieval","ocr","asr","sr","depth","pose","reid"],
    "datasets": ["imagenet","coco","cifar10","cifar100","cityscapes","voc","kitti","openimages","laion","flickr30k","pubmed","arxiv"],
    "losses": ["cross-entropy","focal","dice","triplet","contrastive","nt-xent","kl","l2","huber","giou","diou","ciou"],
    "modalities": ["image","video","audio","text","multimodal","point cloud"],
    "objectives": ["self-supervised","distillation","multitask","adversarial","rlhf","ranking","metric learning"],
    "augmentations": ["cutmix","mixup","randaug","autoaugment","mosaic","copy-paste"],
    "optimizers": ["sgd","adam","adamw","lamb","lion","adafactor"],
    "ops": [
        "deformable attention","depthwise conv","group conv","rope","flash attention",
        "squeeze-excitation","csp","panet","fpn","bi-fpn","linformer","low-rank","lora","qlora"
    ],
}

def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def _scan_line(line: str, keys: List[str]) -> Set[str]:
    found: Set[str] = set()
    for k in keys:
        if k in line:
            found.add(k)
    return found

def extract_facets(
    summaries: List[Dict[str, Any]],
    extra_lexicon: Dict[str, List[str]] | None = None,
    persist: bool = True,
) -> Dict[str, List[str]]:
    """Mine a controlled vocabulary from your summaries to anchor ideation."""
    lex = {**LEX, **(extra_lexicon or {})}

    buckets: Dict[str, Set[str]] = {k: set() for k in lex.keys()}
    for s in summaries:
        lines: List[str] = []
        for k in ("methods", "novelty_claims", "limitations", "title"):
            vals = s.get(k) or []
            if isinstance(vals, list):
                lines.extend([_norm(v) for v in vals])
            else:
                lines.append(_norm(vals))
        for ln in lines:
            for name, keys in lex.items():
                buckets[name].update(_scan_line(ln, keys))

    facets = {k: sorted(v) for k, v in buckets.items()}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if persist:
        FACETS_PATH.write_text(
            json.dumps(facets, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return facets

