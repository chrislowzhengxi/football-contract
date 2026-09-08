from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import DEFAULT_OUTPUT_DIR
from .source_discovery import SourceCandidate


CACHE_DIR = DEFAULT_OUTPUT_DIR / "contract_research" / "official_page_cache"
USER_AGENT = "football-contract-known-source-retrieval/1.0"
MAX_TEXT_CHARS = 60000


@dataclass
class RetrievalResult:
    original_url: str
    final_url: str | None
    retrieval_status: str
    http_status: int | None
    extracted_text: str
    retrieval_timestamp: str
    retrieval_method: str = "urllib_get_htmlparser"
    retrieval_error: str | None = None
    content_length: int = 0
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}:
            self._skip_depth += 1
        if tag in {"p", "br", "div", "article", "section", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "article", "section", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        joined = html.unescape(" ".join(self.parts))
        joined = re.sub(r"\s+", " ", joined)
        boilerplate_patterns = (
            r"(?i)accept all cookies?",
            r"(?i)manage cookies?",
            r"(?i)privacy policy",
            r"(?i)terms and conditions",
            r"(?i)subscribe to our newsletter",
            r"(?i)all rights reserved",
        )
        for pattern in boilerplate_patterns:
            joined = re.sub(pattern, " ", joined)
        return re.sub(r"\s+", " ", joined).strip()[:MAX_TEXT_CHARS]


def extract_visible_text(raw_html: str) -> str:
    parser = VisibleTextParser()
    parser.feed(raw_html)
    return parser.text()


class OfficialPageFetcher:
    def __init__(self, cache_dir: Path = CACHE_DIR, timeout: float = 20, retry_delay: float = 1.0):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.network_fetches = 0

    def cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def fetch(self, url: str) -> RetrievalResult:
        cached = self._read_cache(url)
        if cached:
            return cached
        last_error: str | None = None
        for attempt in range(2):
            try:
                result = self._fetch_once(url)
                self._write_cache(result)
                return result
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                last_error = f"{type(error).__name__}: {error}"
                if attempt == 0:
                    time.sleep(self.retry_delay)
        result = RetrievalResult(
            original_url=url,
            final_url=None,
            retrieval_status="failed",
            http_status=None,
            extracted_text="",
            retrieval_timestamp=_timestamp(),
            retrieval_error=last_error,
            content_length=0,
        )
        self._write_cache(result)
        return result

    def apply_to_candidate(self, candidate: SourceCandidate) -> RetrievalResult:
        result = self.fetch(candidate.source_url)
        candidate.retrieved_text = result.extracted_text or None
        candidate.retrieval_status = result.retrieval_status
        candidate.retrieval_http_status = result.http_status
        candidate.retrieval_final_url = result.final_url
        candidate.retrieval_timestamp = result.retrieval_timestamp
        candidate.retrieval_method = result.retrieval_method
        candidate.retrieval_error = result.retrieval_error
        candidate.content_length = result.content_length
        return result

    def _fetch_once(self, url: str) -> RetrievalResult:
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
        self.network_fetches += 1
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read(2_000_000)
            final_url = response.geturl()
            status = getattr(response, "status", 200)
            charset = response.headers.get_content_charset() or "utf-8"
        text = extract_visible_text(raw.decode(charset, errors="replace"))
        return RetrievalResult(
            original_url=url,
            final_url=final_url,
            retrieval_status="success" if text else "empty",
            http_status=status,
            extracted_text=text,
            retrieval_timestamp=_timestamp(),
            content_length=len(text),
        )

    def _read_cache(self, url: str) -> RetrievalResult | None:
        path = self.cache_path(url)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return RetrievalResult(**{**payload, "from_cache": True})

    def _write_cache(self, result: RetrievalResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["from_cache"] = False
        self.cache_path(result.original_url).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
