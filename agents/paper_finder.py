import os
import csv
import time
import json
import pathlib
import requests
from typing import Optional, Tuple, List
from lab.config import get as cfg_get
from utils.llm_utils import chat_json, LLMError

# ---------------------------
# ENV / CONSTANTS
# ---------------------------
API_KEY = os.getenv("ELSEVIER_KEY")
INSTTOKEN = os.getenv("X_ELS_INSTTOKEN")  # optional; only if you actually have one
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # legacy; relevance now uses main LLM provider

# Note: Avoid failing at import time. Validate credentials at call-sites instead.

BASE_SD_SEARCH  = "https://api.elsevier.com/content/search/sciencedirect"
BASE_SD_ARTICLE = "https://api.elsevier.com/content/article"          # /pii/{PII}
BASE_SCOPUS_ABS = "https://api.elsevier.com/content/abstract"         # /doi/{DOI}
def _cfg_allowed_years() -> set[str]:
    try:
        yrs = cfg_get("pipeline.find_papers.allowed_years", ["2024", "2025"]) or ["2024", "2025"]
        return {str(y) for y in yrs}
    except Exception:
        return {"2024", "2025"}

ALLOWED_YEARS   = _cfg_allowed_years()

JSON_ACCEPT = {"Accept": "application/json"}
PDF_ACCEPT  = {"Accept": "application/pdf"}
USER_AGENT  = "elsevier-pipeline/1.0 (windows-gitbash)"

# DeepSeek (adjust if your endpoint/model differ)
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL    = "deepseek-chat"

# Policy: only download OA PDFs? (True = OA-only; False = try with Insttoken if you have one)
DOWNLOAD_OA_ONLY = bool(cfg_get("pipeline.find_papers.download_oa_only", True))

# Max number of relevant papers to keep before stopping
MAX_KEPT = int(cfg_get("pipeline.find_papers.max_kept", 40) or 40)

# Backup provider when Elsevier flow fails (explicit opt-in). Values: none|arxiv
BACKUP_PROVIDER = str(cfg_get("pipeline.find_papers.backup", "none") or "none").strip().lower()

# ---------------------------
# HELPERS
# ---------------------------
def _auth_headers(base: dict) -> dict:
    h = dict(base)
    if not API_KEY:
        raise RuntimeError("ELSEVIER_KEY not set")
    h["X-ELS-APIKey"] = API_KEY
    if INSTTOKEN:
        h["X-ELS-Insttoken"] = INSTTOKEN
    h["User-Agent"] = USER_AGENT
    return h

def _respect_rate_limit(resp: requests.Response) -> bool:
    if resp.status_code == 429:
        delay = int(resp.headers.get("Retry-After", "1"))
        print(f"[RATE-LIMIT] sleeping {delay}s …")
        time.sleep(delay)
        return True
    return False

def _year_ok(cover_date: Optional[str]) -> bool:
    """Return True if the cover_date year is allowed. Empty set disables filtering."""
    if not cover_date:
        return False
    y = cover_date.split("-", 1)[0].strip()
    if not ALLOWED_YEARS:  # empty set -> accept any year
        return True
    return y in ALLOWED_YEARS

def _bool(val) -> bool:
    if isinstance(val, bool): return val
    if isinstance(val, str):  return val.lower() == "true"
    return False

def _sanitize_filename(s: str, maxlen: int = 120) -> str:
    keep = "".join(c if c.isalnum() or c in " ._-()" else "_" for c in s)
    return keep[:maxlen].rstrip(" ._-")

def _elsevier_is_available(sample_query: str = "test") -> bool:
    """Preflight check: return True if Elsevier SD search responds 200 with credentials.

    No retries; treats 401/403/5xx as unavailable to avoid per-query failures.
    """
    if not API_KEY:
        return False
    try:
        params = {"query": sample_query or "test", "start": 0, "count": 1}
        r = requests.get(BASE_SD_SEARCH, headers=_auth_headers(JSON_ACCEPT), params=params, timeout=12)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------
# SEARCH (all pages)
# ---------------------------
def search_sciencedirect(query: str, page_size: int = 25):
    start = 0
    total = None
    page  = 0
    while True:
        params = {"query": query, "start": start, "count": page_size}
        while True:
            r = requests.get(BASE_SD_SEARCH, headers=_auth_headers(JSON_ACCEPT), params=params, timeout=45)
            if not _respect_rate_limit(r):
                break
        r.raise_for_status()
        data = r.json()
        sr   = data.get("search-results", {})
        if total is None:
            try: total = int(sr.get("opensearch:totalResults", "0"))
            except ValueError: total = 0

        entries = sr.get("entry", []) or []
        got = len(entries); page += 1
        print(f"[PAGE {page}] start={start}  got={got}  total={total}")

        for e in entries:
            yield e

        if got == 0:
            print("[DONE] No more entries.")
            break

        ipp   = int(sr.get("opensearch:itemsPerPage", str(page_size)))
        start = int(sr.get("opensearch:startIndex", str(start))) + ipp
        if total and start >= total:
            print("[DONE] Reached totalResults.")
            break

# ---------------------------
# ABSTRACT RETRIEVAL
# ---------------------------
def fetch_abstract_via_scopus_doi(doi: str) -> Optional[str]:
    url = f"{BASE_SCOPUS_ABS}/doi/{doi}"
    while True:
        resp = requests.get(url, headers=_auth_headers(JSON_ACCEPT), timeout=45)
        if not _respect_rate_limit(resp):
            break
    if resp.status_code != 200:
        return None
    try:
        js = resp.json()
    except Exception:
        return None

    arr  = js.get("abstracts-retrieval-response", {})
    core = arr.get("coredata", {})
    desc = core.get("dc:description")
    if desc: return desc

    try:
        paras = (
            arr.get("item", {})
               .get("bibrecord", {})
               .get("head", {})
               .get("abstracts", {})
               .get("abstract", {})
               .get("ce:para")
        )
        if isinstance(paras, list): return "\n".join(x for x in paras if isinstance(x, str))
        if isinstance(paras, str):  return paras
    except Exception:
        pass
    return None

def fetch_abstract_via_sciencedirect_pii(pii: str) -> Optional[str]:
    url = f"{BASE_SD_ARTICLE}/pii/{pii}"
    params = {"httpAccept": "application/json"}
    while True:
        resp = requests.get(url, headers=_auth_headers(JSON_ACCEPT), params=params, timeout=60)
        if not _respect_rate_limit(resp):
            break
    if resp.status_code != 200:
        return None
    try:
        js = resp.json()
    except Exception:
        return None

    ftr  = js.get("full-text-retrieval-response", {})
    core = ftr.get("coredata", {})
    desc = core.get("dc:description")
    if desc: return desc

    xocs_abs = ftr.get("xocs:abstracts") or ftr.get("xocs:abstract")
    if isinstance(xocs_abs, dict):
        blocks = []
        for k in ("xocs:abstract", "xocs:abs", "abstract"):
            node = xocs_abs.get(k)
            if isinstance(node, dict):
                para = node.get("xocs:para") or node.get("ce:para")
                if isinstance(para, list):
                    blocks.extend([p for p in para if isinstance(p, str)])
                elif isinstance(para, str):
                    blocks.append(para)
        if blocks:
            return "\n".join(blocks)

    for key in ("abstracts", "abstract", "dc:description"):
        val = ftr.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None

def get_abstract(entry: dict) -> Optional[str]:
    doi = entry.get("prism:doi") or (entry.get("dc:identifier") or "").replace("DOI:", "").strip()
    pii = entry.get("pii")
    if doi:
        a = fetch_abstract_via_scopus_doi(doi)
        if a and a.strip(): return a.strip()
    if pii:
        a = fetch_abstract_via_sciencedirect_pii(pii)
        if a and a.strip(): return a.strip()
    return None

# ---------------------------
def judge_relevance_llm(title: str, abstract: str, query: str) -> Tuple[bool, str]:
    """Use the main LLM provider to judge relevance.

    Returns (relevant, reason). On errors, returns (False, reason).
    """
    system = (
        "You are a strict research assistant. Given a user query and a paper's title+abstract, "
        "respond JSON with fields: 'relevant' (true/false) and 'reason' (short). "
        "Be conservative: only true if the abstract clearly addresses the query."
    )
    user = {"query": query, "title": title, "abstract": abstract}
    try:
        js = chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0)
        return bool(js.get("relevant")), str(js.get("reason", ""))
    except LLMError as exc:
        return False, f"LLM error: {exc}"


# ---------------------------
# Keyword-based query synthesis (deterministic)
# ---------------------------
_STOPWORDS = {
    "and","or","the","a","an","of","to","for","in","on","with","using","use","via","from",
    "by","into","at","as","is","are","be","being","been","this","that","these","those","we","i",
}

def _domain_keywords_from_goal(goal: str) -> List[str]:
    g = (goal or "").lower()
    kws: List[str] = []
    if any(x in g for x in ["human pose", "pose classification", "skeleton"]):
        kws += [
            "human pose classification",
            "skeleton-based",
            "pose estimation",
            "keypoint detection",
            "few-shot learning",
            "one-shot learning",
            "low-shot",
            "meta-learning",
            "prototypical networks",
            "relation networks",
            "metric learning",
            "graph neural network",
            "GNN",
            "ST-GCN",
            "transformer",
            "action recognition",
            "NTU RGB+D",
            "Human3.6M",
            "MPII",
            "COCO",
        ]
    return kws

def _generate_llm_keywords(goal: str, *, count: int, max_len: int, model: str | None = None, profile: str | None = None) -> List[str]:
    """Ask the active LLM for focused search keywords based on the project goal."""
    if not goal:
        return []
    try:
        payload = {
            "goal": goal,
            "count": max(1, count),
            "max_len": max(8, max_len),
            "format": "json",
        }
        system = (
            "You generate concise search keywords for academic literature retrieval.\n"
            "Return strictly JSON of the form {\"keywords\": [\"term\", ...]} with unique items.\n"
            "Each keyword must stay under max_len characters, avoid stopwords alone, and remain specific to the goal."
        )
        js = chat_json(system, json.dumps(payload, ensure_ascii=False), temperature=0.0, model=model, profile=profile)
    except LLMError:
        return []
    out: List[str] = []
    for item in (js.get("keywords") or []):
        term = str(item).strip()
        if not term:
            continue
        out.append(term[:max_len])
        if len(out) >= count:
            break
    return out


def _normalize_keywords(goal: str, include_terms: List[str], synonyms: List[str], cfg_keywords: List[str], llm_keywords: List[str]) -> List[str]:
    """Build a unique, ordered keyword list from config + domain mapping; avoid low-signal tokens."""
    seen = set()
    out: List[str] = []
    def add(s: str):
        s = str(s).strip()
        if not s:
            return
        if s.lower() in _STOPWORDS:
            return
        if s in seen:
            return
        seen.add(s); out.append(s)
    # 1) explicit keywords from config (if any)
    for s in (cfg_keywords or []):
        add(s)
    # 2) LLM-proposed keywords (preferred path when config is empty)
    for s in (llm_keywords or []):
        add(s)
    # 3) domain-mapped keywords from goal
    for s in _domain_keywords_from_goal(goal):
        add(s)
    # 4) include_terms and synonyms
    for s in (include_terms or []) + (synonyms or []):
        add(s)
    # Do NOT add single tokens from goal to avoid noise like "want"
    return out

def _chunked_keyword_queries(keywords: List[str], fields: List[str], terms_per_query: int, max_queries: int) -> List[str]:
    """Create one fielded query per keyword: (ti:"k" OR abs:"k"). Always 1 keyword/query."""
    out: List[str] = []
    if max_queries <= 0:
        max_queries = 5
    def expr(term: str) -> str:
        term = term.replace('"','').strip()
        alts = [f"{f}:\"{term}\"" for f in (fields or ["ti","abs"])]
        return "(" + " OR ".join(alts) + ")"
    for k in (keywords or []):
        if not k:
            continue
        out.append(expr(k))
        if len(out) >= max_queries:
            break
    return out

# ---------------------------
# PDF DOWNLOAD (triggered AFTER relevance == True)
# ---------------------------
def download_pdf_by_pii(pii: str, out_dir: pathlib.Path, title: str = "") -> Tuple[bool, str]:
    url = f"{BASE_SD_ARTICLE}/pii/{pii}"
    params = {"httpAccept": "application/pdf"}
    while True:
        r = requests.get(url, headers=_auth_headers(PDF_ACCEPT), params=params, timeout=90, stream=True)
        if not _respect_rate_limit(r):
            break

    if r.status_code == 200 and r.headers.get("content-type", "").lower().startswith("application/pdf"):
        out_dir.mkdir(parents=True, exist_ok=True)
        base = _sanitize_filename(title) if title else pii
        out_path = out_dir / f"{base or pii}.pdf"
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
        return True, str(out_path)

    if r.status_code == 401:
        return False, "401 Unauthorized"
    if r.status_code == 403:
        return False, "403 Forbidden (not OA or no entitlement)"
    if r.status_code == 404:
        return False, "404 Not Found"
    return False, f"{r.status_code}: {r.text[:300]}"

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Build queries from config and project goal (with optional LLM suggestions)
    project_goal = str(cfg_get("project.goal", "") or "").strip()
    base_query = str(cfg_get("pipeline.find_papers.query", "") or "").strip()
    try:
        include_terms = [str(x).strip() for x in (cfg_get("pipeline.find_papers.expand.include_terms", []) or []) if str(x).strip()]
        synonyms = [str(x).strip() for x in (cfg_get("pipeline.find_papers.expand.synonyms", []) or []) if str(x).strip()]
    except Exception:
        include_terms, synonyms = [], []
    try:
        llm_q_enable = bool(cfg_get("pipeline.find_papers.llm_query.enable", True))
        llm_q_count = int(cfg_get("pipeline.find_papers.llm_query.count", 4) or 4)
        llm_q_maxlen = int(cfg_get("pipeline.find_papers.llm_query.max_len", 120) or 120)
    except Exception:
        llm_q_enable, llm_q_count, llm_q_maxlen = True, 4, 120
    # Keyword-driven query synthesis (deterministic)
    try:
        kw_enable = bool(cfg_get("pipeline.find_papers.keyword_query.enable", True))
        kw_terms = int(cfg_get("pipeline.find_papers.keyword_query.terms_per_query", 4) or 4)
        kw_maxq = int(cfg_get("pipeline.find_papers.keyword_query.max_queries", 5) or 5)
        kw_fields = list(cfg_get("pipeline.find_papers.keyword_query.fields", ["ti","abs"]) or ["ti","abs"])  # type: ignore
        kw_list = list(cfg_get("pipeline.find_papers.keywords", []) or [])  # type: ignore
        kw_llm_count = int(cfg_get("pipeline.find_papers.keyword_query.llm_count", max(5, kw_maxq)) or max(5, kw_maxq))
        kw_llm_maxlen = int(cfg_get("pipeline.find_papers.keyword_query.llm_max_len", 60) or 60)
        kw_llm_profile = str(cfg_get("pipeline.find_papers.keyword_query.llm_profile", "") or "").strip() or None
        kw_llm_model = str(cfg_get("pipeline.find_papers.keyword_query.llm_model", "") or "").strip() or None
    except Exception:
        kw_enable, kw_terms, kw_maxq, kw_fields, kw_list = True, 4, 5, ["ti","abs"], []
        kw_llm_count, kw_llm_maxlen = 8, 60
        kw_llm_profile = kw_llm_model = None
    queries: List[str] = []
    # Build keywords first
    kw_llm: List[str] = []
    if kw_enable and not kw_list:
        kw_llm = _generate_llm_keywords(
            project_goal,
            count=kw_llm_count,
            max_len=kw_llm_maxlen,
            model=kw_llm_model,
            profile=kw_llm_profile,
        )
    kws = _normalize_keywords(project_goal, include_terms, synonyms, kw_list, kw_llm) if kw_enable else []
    try:
        simple_kw = bool(cfg_get("pipeline.find_papers.keyword_query.simple", True)) if kw_enable else False
    except Exception:
        simple_kw = True
    if kw_enable and kws:
        if simple_kw:
            queries = [k for k in kws][:kw_maxq]
        else:
            queries = _chunked_keyword_queries(kws, kw_fields, kw_terms, kw_maxq)
    # If not in simple keyword mode (or keywords missing), add goal/config/LLM queries
    if (not simple_kw) or not queries:
        if base_query:
            queries.append(base_query)
        elif project_goal:
            queries.append(project_goal)
        if include_terms:
            queries.append((project_goal or base_query) + " " + " ".join(include_terms))
        if synonyms:
            queries.extend([(project_goal or base_query) + " " + s for s in synonyms])
        if llm_q_enable and project_goal:
            try:
                system = (
                    "Given a research goal, propose concise ScienceDirect/arXiv search queries. "
                    "Return JSON: {\"queries\": [\"q1\", \"q2\", ...]} . Each query ≤ max_len, sorted from specific to broad."
                )
                user = {"goal": project_goal, "count": llm_q_count, "max_len": llm_q_maxlen}
                js = chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0)
                for q in (js.get("queries") or []):
                    s = str(q).strip()
                    if s:
                        queries.append(s[:llm_q_maxlen])
            except LLMError:
                pass
    # Deduplicate and ensure at least one query
    seen = set(); queries = [q for q in queries if not (q in seen or seen.add(q))]
    if not queries:
        queries = [project_goal or "deep learning"]
    # Show which keywords are used to build the search query
    try:
        print("[QUERY] source:")
        print(f"[QUERY]   base_query = {base_query!r}")
        print(f"[QUERY]   project_goal = {project_goal!r}")
        print(f"[QUERY]   include_terms = {include_terms}")
        print(f"[QUERY]   synonyms = {synonyms}")
    except Exception:
        pass
    # Log keywords if keyword_query enabled
    try:
        if kw_enable:
            kws_dbg = _normalize_keywords(project_goal, include_terms, synonyms, kw_list, kw_llm)
            print(f"[KEYWORDS] {kws_dbg}")
    except Exception:
        pass
    for i, q in enumerate(queries, start=1):
        try:
            print(f"[QUERY {i}/{len(queries)}] {q}")
        except Exception:
            pass
    # Persist the active query for audit
    try:
        _runs = pathlib.Path("runs"); _runs.mkdir(exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with (_runs / "find_papers_queries.txt").open("a", encoding="utf-8") as fh:
            for q in queries:
                fh.write(f"{ts}\t{q}\n")
    except Exception:
        pass
    try:
        page_size = int(cfg_get("pipeline.find_papers.page_size", 25) or 25)
    except Exception:
        page_size = 25  # SD cap
    out_csv  = pathlib.Path("abstract_screen_deepseek.csv")
    out_dir  = pathlib.Path("pdfs")

    # CSV header
    if not out_csv.exists():
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["title","year","coverDate","doi","pii","prism_url","openaccess","relevant","reason","pdf_path"])

    processed = kept = downloaded = 0
    skip_year = skip_dup = no_abs = 0
    # Track duplicates to avoid repeated LLM calls and CSV rows
    seen_dois = set()
    seen_piis = set()
    seen_urls = set()

    # Seed dedupe from existing CSV (if any) to support resuming without reprocessing
    prev_kept = 0
    prev_rows = 0
    try:
        if out_csv.exists():
            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    # Expected columns: title, year, coverDate, doi, pii, prism_url, ...
                    if len(row) >= 6:
                        doi_prev = (row[3] or "").strip().lower()
                        pii_prev = (row[4] or "").strip()
                        url_prev = (row[5] or "").strip()
                        if doi_prev:
                            seen_dois.add(doi_prev)
                        if pii_prev:
                            seen_piis.add(pii_prev)
                        if not doi_prev and not pii_prev and url_prev:
                            seen_urls.add(url_prev)
                        prev_rows += 1
                        # Col 7 (index 7) holds 'relevant' boolean per our header
                        try:
                            rel = row[7].strip().lower() if len(row) > 7 else ""
                            if rel in {"true", "1", "yes"}:
                                prev_kept += 1
                        except Exception:
                            pass
            print(f"[RESUME] Seeded dedupe from CSV: dois={len(seen_dois)} piis={len(seen_piis)} urls={len(seen_urls)}")
    except Exception as exc:
        print(f"[RESUME WARN] Failed to seed dedupe from CSV: {exc}")

    # Count existing PDFs (for download progress context)
    try:
        existing_pdfs = sum(1 for _ in out_dir.glob("*.pdf")) if out_dir.exists() else 0
    except Exception:
        existing_pdfs = 0

    # Baseline progress and remaining target
    left_total = max(MAX_KEPT - prev_kept, 0)
    print(f"[BASELINE] csv_rows={prev_rows} kept_csv={prev_kept} pdfs={existing_pdfs} target={MAX_KEPT} left_to_find={left_total}")

    def _run_elsevier(q: str):
        global processed, kept, downloaded, skip_year, skip_dup, no_abs, prev_kept
        try:
            print(f"[ELSEVIER] query = {q}")
        except Exception:
            pass
        for entry in search_sciencedirect(q, page_size):
            title = entry.get("dc:title") or ""
            cover_date = entry.get("prism:coverDate") or ""
            if not _year_ok(cover_date):
                y = cover_date.split("-", 1)[0] if cover_date else "NA"
                skip_year += 1
                print(f"[SKIP year] {title[:80]!r} year={y} | skip_year={skip_year}")
                continue

            doi       = (entry.get("prism:doi") or "").strip().lower() or None
            pii       = (entry.get("pii") or "").strip() or None
            prism_url = (entry.get("prism:url") or "").strip() or None
            oa        = _bool(entry.get("openaccess"))

            # Dedupe: skip if we've seen the same DOI or PII before
            is_dup = False
            if doi and doi in seen_dois:
                is_dup = True
            if pii and pii in seen_piis:
                is_dup = True
            if not doi and not pii and prism_url and prism_url in seen_urls:
                is_dup = True
            if is_dup:
                skip_dup += 1
                total_kept_so_far = prev_kept + kept
                left_now = max(MAX_KEPT - total_kept_so_far, 0)
                print(f"[SKIP DUP] {title[:80]!r} doi={doi or 'NA'} pii={pii or 'NA'} | skip_dup={skip_dup} | kept_total={total_kept_so_far}/{MAX_KEPT} left={left_now}")
                continue
            # Mark as seen so later duplicates won't be reprocessed
            if doi:
                seen_dois.add(doi)
            if pii:
                seen_piis.add(pii)
            if not doi and not pii and prism_url:
                seen_urls.add(prism_url)

            abstract = get_abstract(entry)
            if not abstract:
                no_abs += 1
                total_kept_so_far = prev_kept + kept
                left_now = max(MAX_KEPT - total_kept_so_far, 0)
                print(f"[NO ABSTRACT] {title[:80]!r} | no_abs={no_abs} | kept_total={total_kept_so_far}/{MAX_KEPT} left={left_now}")
                continue

            try:
                rel_enable = bool(cfg_get("pipeline.find_papers.relevance.enable", True))
            except Exception:
                rel_enable = True
            if rel_enable:
                relevant, reason = judge_relevance_llm(title, abstract, q)
            else:
                relevant, reason = True, "relevance disabled"
            processed += 1
            if relevant:
                kept += 1

            pdf_path = ""
            if relevant:
                # download AFTER relevance decision
                if DOWNLOAD_OA_ONLY and not oa:
                    print(f"[RELEVANT] (skip download, not OA) :: {title[:90]}")
                else:
                    if not pii:
                        print(f"[RELEVANT] (no PII available to download) :: {title[:90]}")
                    else:
                        ok, path_or_err = download_pdf_by_pii(pii, out_dir, title=title)
                        if ok:
                            downloaded += 1
                            pdf_path = path_or_err
                            print(f"[DOWNLOADED] {path_or_err}")
                        else:
                            print(f"[DL FAIL] {title[:80]!r} :: {path_or_err}")

            total_kept_so_far = prev_kept + kept
            left_now = max(MAX_KEPT - total_kept_so_far, 0)
            print(f"[JUDGED] relevant={relevant} | kept_run={kept}/{processed} | kept_total={total_kept_so_far}/{MAX_KEPT} left={left_now} | downloaded={downloaded} :: {title[:90]}")

            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                y = cover_date.split("-", 1)[0] if cover_date else ""
                w.writerow([title, y, cover_date, doi, pii, prism_url, oa, relevant, reason, pdf_path])

            # Stop once we have enough relevant papers
            if prev_kept + kept >= MAX_KEPT:
                print(f"[STOP] Reached MAX_KEPT={MAX_KEPT} (kept_total={prev_kept + kept}). Halting further processing.")
                break

    def _run_arxiv(q: str):
        """Fetch a single-page arXiv Atom query and process entries (download PDFs when relevant)."""
        global processed, kept, downloaded, skip_year, skip_dup, no_abs, prev_kept
        try:
            print(f"[ARXIV] query = {q}")
        except Exception:
            pass

        ARXIV_BASE = "http://export.arxiv.org/api/query"

        def _build_arxiv_search(term: str, cats: List[str], simple: bool = False) -> str:
            term = (term or "").strip()
            if simple and term:
                esc = term.replace('"', '')
                search = f'all:"{esc}"'
            else:
                has_ops = any(tok in term for tok in [" ti:", " abs:", " au:", " cat:", "AND", "OR", "ANDNOT", "(", ")", '"'])
                if has_ops:
                    search = term
                else:
                    esc = term.replace('"', '')
                    search = f'(ti:"{esc}" OR abs:"{esc}")'
            if cats:
                cat_expr = " OR ".join(f"cat:{c}" for c in cats if c)
                if cat_expr:
                    search = f'({search}) AND ({cat_expr})'
            return search

        def _arxiv_query_url(term: str, start: int, max_results: int, cats: List[str], simple: bool) -> str:
            from urllib.parse import urlencode
            search = _build_arxiv_search(term, cats, simple=simple)
            params = {
                "search_query": search,
                "start": int(start),
                "max_results": int(max_results),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            return f"{ARXIV_BASE}?{urlencode(params)}"

        import xml.etree.ElementTree as ET

        start = 0
        try:
            per_query = int(cfg_get("pipeline.find_papers.arxiv.max_results_per_query", 10) or 10)
        except Exception:
            per_query = 10
        page_sz = max(1, per_query)
        try:
            allow_cats = [str(x).strip() for x in (cfg_get("pipeline.find_papers.arxiv.categories", ["cs.CV"]) or ["cs.CV"]) if str(x).strip()]
        except Exception:
            allow_cats = ["cs.CV"]
        try:
            simple_kw = bool(cfg_get("pipeline.find_papers.keyword_query.simple", True))
        except Exception:
            simple_kw = True

        url = _arxiv_query_url(q, start, page_sz, allow_cats, simple=simple_kw)
        try:
            print(f"[ARXIV] page_start={start} url={url}")
        except Exception:
            pass

        r = requests.get(url, timeout=45, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            print(f"[ARXIV ERR] {r.status_code}: {r.text[:200]}")
            return

        try:
            root = ET.fromstring(r.text)
        except Exception as exc:
            print(f"[ARXIV ERR] failed to parse feed: {exc}")
            return
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        try:
            print(f"[ARXIV] fetched entries={len(entries)}")
        except Exception:
            pass
        if not entries:
            return

        for e in entries:
            title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (e.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            published = (e.findtext("atom:published", default="", namespaces=ns) or "").strip()

            # Category filter
            cats = set()
            for c in e.findall("atom:category", ns):
                term = c.get("term") or ""
                if term:
                    cats.add(term)
            if allow_cats and cats and not (cats.intersection(set(allow_cats))):
                continue

            year = (published.split("-", 1)[0] or "").strip()
            if year and year not in ALLOWED_YEARS:
                continue

            links = [(ln.get("rel"), ln.get("href"), ln.get("type", "")) for ln in e.findall("atom:link", ns)]
            pdf = ""
            for rel, href, typ in links:
                if (rel == "related" and typ == "application/pdf") or (href and href.endswith(".pdf")):
                    pdf = href
                    break
            if not pdf:
                entry_id = (e.findtext("atom:id", default="", namespaces=ns) or "").strip()
                if entry_id and "/abs/" in entry_id:
                    pdf = entry_id.replace("/abs/", "/pdf/") + ".pdf"

            if not summary:
                continue

            try:
                rel_enable = bool(cfg_get("pipeline.find_papers.relevance.enable", True))
            except Exception:
                rel_enable = True
            if rel_enable:
                relevant, reason = judge_relevance_llm(title, summary, q)
            else:
                relevant, reason = True, "relevance disabled"

            processed += 1
            if relevant:
                kept += 1

            pdf_path = ""
            if relevant and pdf:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    safe = _sanitize_filename(title) or title[:50]
                    dest = out_dir / f"{safe}.pdf"
                    rr = requests.get(pdf, stream=True, timeout=90, headers={"User-Agent": USER_AGENT})
                    if rr.status_code == 200 and rr.headers.get("content-type", "").lower().startswith("application/pdf"):
                        with open(dest, "wb") as f:
                            for chunk in rr.iter_content(chunk_size=1 << 15):
                                if chunk:
                                    f.write(chunk)
                        pdf_path = str(dest)
                        downloaded += 1
                        print(f"[ARXIV DOWNLOADED] {dest}")
                except Exception as exc:
                    print(f"[ARXIV DL FAIL] {title[:80]!r} :: {exc}")

            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([title, year, year, "", "", "", True, relevant, reason, pdf_path])

            total_kept_so_far = prev_kept + kept
            left_now = max(MAX_KEPT - total_kept_so_far, 0)
            print(f"[ARXIV JUDGED] relevant={relevant} | kept_run={kept}/{processed} | kept_total={total_kept_so_far}/{MAX_KEPT} left={left_now} | downloaded={downloaded} :: {title[:90]}")

            if prev_kept + kept >= MAX_KEPT:
                print(f"[ARXIV STOP] Reached MAX_KEPT={MAX_KEPT} (kept_total={prev_kept + kept}).")
                return

    # Decide provider once (auto/elsevier/arxiv) and do not retry per query
    try:
        prov_pref = str(cfg_get("pipeline.find_papers.provider", "auto") or "auto").strip().lower()
    except Exception:
        prov_pref = "auto"
    if prov_pref not in {"auto", "elsevier", "arxiv"}:
        prov_pref = "auto"
    if prov_pref == "arxiv":
        chosen = "arxiv"
    elif prov_pref == "elsevier":
        chosen = "elsevier" if _elsevier_is_available(queries[0]) else "arxiv"
    else:
        chosen = "elsevier" if _elsevier_is_available(queries[0]) else "arxiv"
    print(f"[PROVIDER] using: {chosen} (preference={prov_pref})")

    for idx, q in enumerate(queries, start=1):
        print(f"[RUN QUERY {idx}/{len(queries)}]")
        if chosen == "elsevier":
            _run_elsevier(q)
        else:
            _run_arxiv(q)
        if prev_kept + kept >= MAX_KEPT:
            break

    print(
        "\n[DONE] "
        f"processed={processed}, kept_run={kept}, kept_csv={prev_kept}, kept_total={prev_kept + kept}, "
        f"downloaded_run={downloaded}, skip_dup={skip_dup}, skip_year={skip_year}, no_abstract={no_abs}, "
        f"csv={out_csv.resolve()}, dir={out_dir.resolve()}"
    )
