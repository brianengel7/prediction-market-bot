
#!/usr/bin/env python3
"""Read-only GFS/Open-Meteo inventory for Codespaces."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
from pathlib import Path

MATCH = re.compile(
    r"gfs|open[-_ ]?meteo|hrrr|forecast", re.I
)

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    ".next", "__pycache__", ".tox", "site-packages"
}

SKIP_NAMES = re.compile(
    r"\.env|secret|credential|private|token|api.?key|password",
    re.I,
)

PREVIEW_EXTS = {
    ".csv", ".tsv", ".json", ".jsonl",
    ".ndjson", ".parquet", ".db",
    ".sqlite", ".sqlite3",
}


def file_inventory(root: Path, limit: int):
    found = []

    for base in (root / "data", root / "src"):
        if not base.is_dir():
            continue

        for parent, folders, files in os.walk(base):
            folders[:] = [
                d for d in folders
                if d not in SKIP_DIRS
                and not SKIP_NAMES.search(d)
            ]

            for name in files:
                if SKIP_NAMES.search(name):
                    continue

                path = Path(parent) / name
                relative = path.relative_to(root)

                allowed = PREVIEW_EXTS | {
                    ".grib", ".grib2", ".nc",
                    ".pkl", ".joblib", ".py",
                    ".gz", ".zip"
                }

                if (
                    MATCH.search(str(relative))
                    and path.suffix.lower() in allowed
                ):
                    found.append(path)

                if len(found) >= limit:
                    return found

    return found


def preview(path: Path):
    ext = path.suffix.lower()

    if ext in {".csv", ".tsv"}:
        with path.open(
            "r", encoding="utf-8-sig",
            errors="replace", newline=""
        ) as fh:
            reader = csv.reader(
                fh,
                delimiter="\t" if ext == ".tsv" else ","
            )

            print(
                "  CSV columns:",
                next(reader, [])[:60]
            )
            print(
                "  First record:",
                [
                    str(x)[:55]
                    for x in next(reader, [])[:16]
                ]
            )

    elif ext in {".jsonl", ".ndjson"}:
        with path.open(
            "r", encoding="utf-8", errors="replace"
        ) as fh:
            line = fh.readline()

        first = json.loads(line)

        if isinstance(first, dict):
            print("  JSONL keys:", list(first)[:35])

            if isinstance(first.get("hourly"), dict):
                print(
                    "  Hourly keys:",
                    list(first["hourly"])[:40]
                )
        else:
            print("  JSONL type:", type(first).__name__)

    elif ext == ".json":
        if path.stat().st_size > 12_000_000:
            print("  Large JSON; preview skipped")
            return

        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)

        if isinstance(obj, dict):
            print("  JSON keys:", list(obj)[:40])

            if isinstance(obj.get("hourly"), dict):
                print(
                    "  Hourly keys:",
                    list(obj["hourly"])[:30]
                )

        elif isinstance(obj, list):
            print("  JSON list size:", len(obj))

            if obj and isinstance(obj[0], dict):
                print(
                    "  First object keys:",
                    list(obj[0])[:35]
                )

    elif ext == ".parquet":
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)

            print(
                "  Parquet rows:",
                pf.metadata.num_rows
            )
            print(
                "  Columns:",
                pf.schema.names[:60]
            )

        except ImportError:
            print("  Install pyarrow for schema preview")

    elif ext in {".db", ".sqlite", ".sqlite3"}:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True
        )

        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'"
            ).fetchall()

            print(
                "  Tables:",
                [r[0] for r in tables][:35]
            )

            for (table,) in tables[:8]:
                clean = table.replace('"', '""')

                cols = conn.execute(
                    f'PRAGMA table_info("{clean}")'
                ).fetchall()

                print(
                    f"  {table}:",
                    [x[1] for x in cols][:35]
                )

        finally:
            conn.close()

    elif ext == ".py":
        print("  Python source file")

    else:
        print("  Binary/other format")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    root = args.root.resolve()

    print("Repository:", root)

    found = file_inventory(
        root, max(1, args.limit)
    )

    if not found:
        print(
            "No GFS/Open-Meteo/forecast-named "
            "files found under data/ or src/."
        )
        print(
            "Generic filenames may still contain data."
        )
        return

    print("Candidate files:", len(found))

    for path in found:
        try:
            print(
                f"\n{path.relative_to(root)} "
                f"({path.stat().st_size:,} bytes)"
            )
            preview(path)

        except Exception as exc:
            print(
                "  Preview unavailable:",
                type(exc).__name__,
                str(exc)[:160],
            )

    print("\nSearch source references:")
    print(
        "rg -n -i "
        "'gfs|open.?meteo|single.runs' "
        "src | head -80"
    )


if __name__ == "__main__":
    main()
