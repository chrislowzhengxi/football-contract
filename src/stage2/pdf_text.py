"""PDF text extraction for regulatory and financial filings.

The pilot discovered KAP (Borsa Istanbul) disclosures but could not read them:
they are served as PDFs and the HTML fetcher returned parsed binary garbage.
Regulated filings are the one Tier-1 class that can itemise transfer
consideration, so being unable to read them left the most promising hypothesis
untested.
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path

USER_AGENT = "football-contract-research/0.1 (academic research, MIT)"
PDF_MAGIC = b"%PDF-"


def looks_like_pdf(raw: bytes) -> bool:
    return PDF_MAGIC in raw[:1024]


def extract_pdf_text(raw: bytes, max_pages: int = 40) -> str:
    """Plain text from PDF bytes. Returns '' when the file is unreadable."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [(page.extract_text() or "") for page in reader.pages[:max_pages]]
    except Exception:                                   # noqa: BLE001
        return ""
    text = "\n".join(pages)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def fetch_binary(url: str, timeout: float = 30) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception:                                   # noqa: BLE001
        return None


def fetch_text(url: str, cache_dir: Path | None = None) -> tuple[str | None, str]:
    """(text, status) for a URL that may be a PDF. Binary bytes are cached so a
    re-read never re-downloads."""
    cache_file = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        import hashlib
        cache_file = cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".bin")
        if cache_file.exists():
            raw = cache_file.read_bytes()
            if looks_like_pdf(raw):
                text = extract_pdf_text(raw)
                return (text or None), ("pdf_ok" if text else "pdf_unreadable")
    raw = fetch_binary(url)
    if raw is None:
        return None, "fetch_failed"
    if cache_file:
        cache_file.write_bytes(raw)
    if not looks_like_pdf(raw):
        return None, "not_a_pdf"
    text = extract_pdf_text(raw)
    return (text or None), ("pdf_ok" if text else "pdf_unreadable")
