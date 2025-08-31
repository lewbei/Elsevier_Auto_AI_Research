import pathlib
from typing import Optional

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


def extract_text_from_pdf(path: pathlib.Path, max_pages: Optional[int] = 12,
                          max_chars: Optional[int] = 40000) -> str:
    """Extract text from first `max_pages` pages (default 12) up to `max_chars`.
    If pypdf is unavailable, raise ImportError to make the dependency explicit.
    """
    if PdfReader is None:
        raise ImportError("pypdf not installed. Please install with `pip install pypdf`. ")
    reader = PdfReader(str(path))
    pages = reader.pages
    buf = []
    count = 0
    limit = min(max_pages or len(pages), len(pages))
    for i in range(limit):
        try:
            txt = pages[i].extract_text() or ""
        except Exception:
            txt = ""
        if not txt:
            continue
        buf.append(txt)
        count += len(txt)
        if max_chars and count >= max_chars:
            break
    return "\n\n".join(buf)

