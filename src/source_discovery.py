from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class SourceCandidate:
    source_url: str
    source_title: str | None
    publisher: str | None
    publication_date: str | None
    source_type: str
    evidence_text: str
    language: str
    evidence_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchProvider(Protocol):
    def search_transfer(self, event: dict[str, Any]) -> list[SourceCandidate]:
        ...


def load_source_candidates(path: Path) -> list[SourceCandidate]:
    payload = json.loads(path.read_text())
    records = payload.get("sources", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("source file must contain a JSON list or a {\"sources\": [...]} object")
    candidates = [SourceCandidate(**record) for record in records]
    for index, candidate in enumerate(candidates, start=1):
        if not candidate.evidence_id:
            candidate.evidence_id = f"s{index}"
    return candidates
