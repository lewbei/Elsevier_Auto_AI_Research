import os
import csv
import time
import json
import pathlib
import requests
from typing import Optional, Tuple
from dotenv import load_dotenv

# ---------------------------
# ENV / CONSTANTS
# ---------------------------
load_dotenv()
API_KEY = os.getenv("ELSEVIER_KEY")
INSTTOKEN = os.getenv("X_ELS_INSTTOKEN")  # optional; only if you actually have one
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # required for LLM relevance

# Note: Avoid failing at import time. Validate credentials at call-sites instead.

BASE_SD_SEARCH  = "https://api.elsevier.com/content/search/sciencedirect"
BASE_SD_ARTICLE = "https://api.elsevier.com/content/article"          # /pii/{PII}
BASE_SCOPUS_ABS = "https://api.elsevier.com/content/abstract"         # /doi/{DOI}
ALLOWED_YEARS   = {"2024", "2025"}

JSON_ACCEPT = {"Accept": "application/json"}
PDF_ACCEPT  = {"Accept": "application/pdf"}
USER_AGENT  = "elsevier-pipeline/1.0 (windows-gitbash)"

# DeepSeek (adjust if your endpoint/model differ)
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL    = "deepseek-chat"

# Policy: only download OA PDFs? (True = OA-only; False = try with Insttoken if you have one)
DOWNLOAD_OA_ONLY = True

# Max number of relevant papers to keep before stopping
MAX_KEPT = 40

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
    if not cover_date:
        return False
    return cover_date.split("-", 1)[0].strip() in ALLOWED_YEARS

def _bool(val) -> bool:
    if isinstance(val, bool): return val
    if isinstance(val, str):  return val.lower() == "true"
    return False

def _sanitize_filename(s: str, maxlen: int = 120) -> str:
    keep = "".join(c if c.isalnum() or c in " ._-()" else "_" for c in s)
    return keep[:maxlen].rstrip(" ._-")

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
# DEEPSEEK RELEVANCE
# ---------------------------
def judge_relevance_deepseek(title: str, abstract: str, query: str) -> Tuple[bool, str]:
    if not DEEPSEEK_API_KEY:
        return False, "DEEPSEEK_API_KEY not set"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    system_msg = (
        "You are a strict research assistant. "
        "Given a user query and a paper's title+abstract, respond JSON with fields: "
        "'relevant' (true/false) and 'reason' (short). "
        "Be conservative: only true if the abstract clearly addresses the query."
    )
    user_msg = json.dumps({"query": query, "title": title, "abstract": abstract}, ensure_ascii=False)
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        return False, f"DeepSeek error {r.status_code}: {r.text[:200]}"
    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        js = json.loads(content)
        return bool(js.get("relevant")), js.get("reason", "")
    except Exception as exc:
        return False, f"DeepSeek parse error: {exc}"

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
    query = "deep learning skin cancer classification"
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

    for entry in search_sciencedirect(query, page_size):
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

        relevant, reason = judge_relevance_deepseek(title, abstract, query)
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

    print(
        "\n[DONE] "
        f"processed={processed}, kept_run={kept}, kept_csv={prev_kept}, kept_total={prev_kept + kept}, "
        f"downloaded_run={downloaded}, skip_dup={skip_dup}, skip_year={skip_year}, no_abstract={no_abs}, "
        f"csv={out_csv.resolve()}, dir={out_dir.resolve()}"
    )
