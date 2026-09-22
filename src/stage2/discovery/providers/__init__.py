"""Interchangeable search providers. Importing this package must never require
any API credential; each provider reports its own availability instead."""
from .cache import CacheProvider
from .direct import DirectSourceProvider
from .generic import GenericWebSearchProvider
from .tavily import TavilyProvider

__all__ = ["CacheProvider", "DirectSourceProvider", "GenericWebSearchProvider",
           "TavilyProvider", "build_providers", "PROVIDER_REGISTRY"]

PROVIDER_REGISTRY = {
    "cache": CacheProvider,
    "direct": DirectSourceProvider,
    "generic": GenericWebSearchProvider,
    "tavily": TavilyProvider,
}

# Cheapest and most authoritative first. Paid discovery is the last resort.
DEFAULT_ORDER = ("cache", "direct", "generic", "tavily")


def build_providers(names, **kwargs):
    """Construct providers by name, skipping any that cannot run.

    Returns (usable, skipped). A missing credential is a skip, never a crash:
    a `cache,direct` run must work on a machine with no keys at all.
    """
    usable, skipped = [], []
    for name in names:
        cls = PROVIDER_REGISTRY.get(name)
        if cls is None:
            skipped.append((name, "no such provider"))
            continue
        try:
            p = cls(**{k: v for k, v in kwargs.items()
                       if k in getattr(cls, "accepts", ())})
            ok, reason = p.available()
            (usable if ok else skipped).append(p if ok else (name, reason))
        except Exception as exc:                               # noqa: BLE001
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
    return usable, skipped
