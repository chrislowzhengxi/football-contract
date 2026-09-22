"""Stage 2 research CLI, provider-selectable.

    python -m src.stage2.discovery.cli --search-providers cache,direct
    python -m src.stage2.discovery.cli --search-providers cache,direct,tavily --allow-network

No credential is required unless a provider that needs one is named. A
`cache,direct` run works on a machine with no API keys at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ..event_family import build_families, families_to_rows
from .pipeline import Stage2Pipeline
from .providers import DEFAULT_ORDER, PROVIDER_REGISTRY

DEFAULT_OUT = Path("data/outputs/rebuild/stage2")


def parse_providers(value: str) -> list[str]:
    names = [v.strip().lower() for v in value.split(",") if v.strip()]
    bad = [n for n in names if n not in PROVIDER_REGISTRY]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown provider(s) {bad}; choose from {sorted(PROVIDER_REGISTRY)}")
    # Keep the cheap-to-expensive ordering regardless of the order given.
    return sorted(names, key=lambda n: DEFAULT_ORDER.index(n)
                  if n in DEFAULT_ORDER else 99)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 provider-independent research")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--search-provider", type=parse_providers, dest="providers",
                   help="single provider: cache | direct | generic | tavily")
    g.add_argument("--search-providers", type=parse_providers, dest="providers",
                   help="comma-separated, tried cheapest first")
    p.add_argument("--events", help="CSV with an event_id column")
    p.add_argument("--event-id", action="append", default=[])
    p.add_argument("--canonical", default="data/outputs/rebuild/stage1c_canonical_transfers.csv")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--allow-network", action="store_true",
                   help="permit outbound HTTP for direct-source and retrieval")
    p.add_argument("--escalate", action="store_true", help="plan tier-2 queries too")
    p.add_argument("--max-docs", type=int, default=20)
    p.add_argument("--limit", type=int)
    p.add_argument("--list-providers", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    providers = tuple(args.providers or ("cache", "direct"))

    if args.list_providers:
        from .providers import build_providers
        usable, skipped = build_providers(PROVIDER_REGISTRY.keys(),
                                          allow_network=args.allow_network)
        for p in usable:
            print(f"  available  {p.provider_name:10s} {p.available()[1]}")
        for name, why in skipped:
            print(f"  skipped    {name:10s} {why}")
        return 0

    canonical = pd.read_csv(args.canonical, low_memory=False)
    ids = list(args.event_id)
    if args.events:
        ids += list(pd.read_csv(args.events).event_id)
    if not ids:
        print("no events given; use --events or --event-id", file=sys.stderr)
        return 2
    ids = list(dict.fromkeys(ids))[: args.limit] if args.limit else list(dict.fromkeys(ids))

    pipe = Stage2Pipeline(providers=providers, allow_network=args.allow_network)
    print(f"providers active: {pipe.provider_names()}")
    for name, why in pipe.skipped:
        print(f"  skipped {name}: {why}")
    if not pipe.providers:
        print("no usable providers", file=sys.stderr)
        return 3

    fams = build_families(canonical, ids)
    flat = families_to_rows(fams, canonical)
    out, rows = Path(args.output_dir), []
    out.mkdir(parents=True, exist_ok=True)
    for eid in ids:
        fam = fams.get(eid)
        if fam is None:
            continue
        fr = flat[flat.event_family_id == fam.family_id].copy()
        fr["is_family_anchor"] = fr.event_id == eid
        r = pipe.run_family(fr, escalate=args.escalate, max_docs=args.max_docs)
        rows.append({"event_id": eid, "event_family_id": r.event_family_id,
                     "player": r.player_name, "family": r.family_classification,
                     "evidence": len(r.evidence), "readable": len(r.readable),
                     "clause": len(r.with_clause), "tier1": len(r.tier1),
                     "search_need": r.search_need})
        print(f"  {r.player_name[:24]:26s} {r.family_classification:28s} "
              f"readable={len(r.readable):3d} clause={len(r.with_clause):2d} -> {r.search_need}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "stage2_cli_run.csv", index=False)
    print(f"\n{len(df)} families; wrote {out/'stage2_cli_run.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
