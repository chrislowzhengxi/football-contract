"""Rewrite stage1c_validation_sample.csv from the frozen canonical file, without rebuilding Stage 1C.

    python -m src.stage1c.validation_sample

The sample is a view of `stage1c_canonical_transfers.csv` (`build.validation_sample`). The canonical file is read as
text, so every field is written exactly as it appears there; the canonical file and the original sample were written
from the same frame by `build.main`. The canonical file's SHA-256 is checked before and after.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from .build import CANONICAL_CSV, SAMPLE_CSV, validation_sample

STAGE1C_SHA256 = "92f383d72a3ed59925c922f0d1b5f98528bc66ad3a9f49bb990867ed1d1d50dd"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build() -> pd.DataFrame:
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    canonical = pd.read_csv(CANONICAL_CSV, dtype=str, keep_default_na=False)
    canonical["player_id"] = canonical.player_id.astype(int)      # selection and order are by numeric id
    return validation_sample(canonical)


def main() -> None:
    sample = build()
    sample.to_csv(SAMPLE_CSV, index=False)
    assert sha256(CANONICAL_CSV) == STAGE1C_SHA256
    print(f"wrote {SAMPLE_CSV.name}: {len(sample)} rows, {sample.player_id.nunique()} players (Stage 1C unchanged)")


if __name__ == "__main__":
    main()
