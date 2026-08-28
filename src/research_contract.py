from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .contract_schemas import ContractResearchResult, utc_now


class ResearchProvider(Protocol):
    def research(self, event: dict[str, Any]) -> ContractResearchResult:
        ...


class OpenAIWebResearchProvider:
    """Use an OpenAI Responses API model with web search enabled."""

    def __init__(self, api_key: str, model: str, prompt: str):
        self.api_key = api_key
        self.model = model
        self.prompt = prompt

    def research(self, event: dict[str, Any]) -> ContractResearchResult:
        request_body = {
            "model": self.model,
            "tools": [{"type": "web_search_preview"}],
            "input": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps(event, default=str)},
            ],
        }
        request = Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(request_body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request) as response:
                payload = json.load(response)
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"research provider returned HTTP {error.code}: {detail}") from error
        return ContractResearchResult.from_dict({**_response_json(payload), "event_id": event["event_id"]})


class FixtureResearchProvider:
    def __init__(self, path: Path):
        self.path = path

    def research(self, event: dict[str, Any]) -> ContractResearchResult:
        payload = json.loads(self.path.read_text())
        payload["event_id"] = event["event_id"]
        return ContractResearchResult.from_dict(payload)


def _response_json(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("output_text"), str):
        return json.loads(payload["output_text"])
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                return json.loads(text)
    raise ValueError("research provider response did not contain JSON output")


def load_event(path: Path, event_id: str) -> dict[str, Any]:
    events = pd.read_csv(path, dtype={"event_id": str})
    matches = events[events["event_id"] == event_id]
    if matches.empty:
        raise ValueError(f"event_id not found in {path}: {event_id}")
    if len(matches) > 1:
        raise ValueError(f"event_id is not unique in {path}: {event_id}")
    return {key: (None if pd.isna(value) else value) for key, value in matches.iloc[0].to_dict().items()}


def run(event_id: str, events_path: Path, output_dir: Path, provider: ResearchProvider) -> Path:
    event = load_event(events_path, event_id)
    result = provider.research(event)
    if result.event_id != event_id:
        raise ValueError("research provider returned a different event_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{event_id}.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Research contractual terms for one transfer event")
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--events", default=str(DEFAULT_OUTPUT_DIR / "structured_transfers.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "contract_research"))
    parser.add_argument("--fixture", help="Use a saved provider JSON response instead of live web research")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    args = parser.parse_args()
    if args.fixture:
        provider: ResearchProvider = FixtureResearchProvider(Path(args.fixture))
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for live research; use --fixture for offline validation")
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "contract_research.md"
        provider = OpenAIWebResearchProvider(api_key, args.model, prompt_path.read_text())
    output = run(args.event_id, Path(args.events), Path(args.output_dir), provider)
    print(f"Wrote contract research result to {output}")


if __name__ == "__main__":
    main()
