from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import duckdb

from .config import SOURCE_URL


def download_database(path: Path, url: str = SOURCE_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(url) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    return path


def connect_database(path: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=read_only)


def inspect_schema(connection: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    tables = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
    return {
        table: [row[0] for row in connection.execute(f'DESCRIBE "{table}"').fetchall()]
        for table in tables
    }


def require_columns(schema: dict[str, list[str]], table: str, columns: list[str]) -> None:
    if table not in schema:
        raise ValueError(f"Required source table is missing: {table}")
    missing = sorted(set(columns) - set(schema[table]))
    if missing:
        raise ValueError(f"Source table {table} is missing columns: {missing}")