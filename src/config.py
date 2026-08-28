from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "data" / "raw" / "transfermarkt-datasets.duckdb"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "outputs"
SOURCE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"