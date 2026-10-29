#!/usr/bin/env python3
"""Repository-level integrity checks for the published POSEIDON scientific artefacts.

This check intentionally uses only the Python standard library. It validates
static repository invariants; it is not an end-to-end model retraining test.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCALER_SHA256 = (
    "9df8cdd9f1fd51e46d5e7a911fb35d7c72ef19d2f8b1e5275451d763c162c5f2"
)
EXPECTED_FEATURE_COUNT = 1330
EXPECTED_DATABASE_ROWS = 2371
EXPECTED_CELL_LINES = 42

FEATURE_BLOCK_SIZES = {
    "experimental": 8,
    "cargo": 62,
    "sequence": 18,
    "peptideR": 31,
    "sequence_encoding": 778,
    "genomics": 415,
    "anomalous": 18,
}

REQUIRED_FILES = [
    "data/CPP_dataset.csv",
    "data/GDSC_features.csv",
    "support/Peptides_deployment.R",
    "support/cargo_list.txt",
    "support/cargo_matcher.csv",
    "support/cell_line_name_encoder.csv",
    "support/cell_line_tissue_table.csv",
    "support/exclude_methods.txt",
    "support/exclude_units.txt",
    "support/features_header.txt",
    "support/scaler.pkl",
    "support/sequence_anomalies.csv",
]

PROHIBITED_WEB_PATHS = [
    "templates",
    "static",
    "poseidon_web_setup.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"FAIL: {message}")


def passed(message: str) -> None:
    print(f"PASS: {message}")


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if path.is_file():
            passed(f"required file exists: {rel}")
        else:
            fail(errors, f"missing required file: {rel}")

    for rel in PROHIBITED_WEB_PATHS:
        path = ROOT / rel
        if not path.exists():
            passed(f"web/deployment path absent from scientific repo: {rel}")
        else:
            fail(errors, f"web/deployment path must remain outside scientific repo: {rel}")

    header_path = ROOT / "support/features_header.txt"
    if header_path.is_file():
        features = [
            line.strip()
            for line in header_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(features) == EXPECTED_FEATURE_COUNT:
            passed(f"feature schema contains {EXPECTED_FEATURE_COUNT} entries")
        else:
            fail(
                errors,
                f"feature schema contains {len(features)} entries; "
                f"expected {EXPECTED_FEATURE_COUNT}",
            )

        if len(set(features)) == len(features):
            passed("feature schema contains no duplicate names")
        else:
            fail(errors, "feature schema contains duplicate names")

    block_total = sum(FEATURE_BLOCK_SIZES.values())
    if block_total == EXPECTED_FEATURE_COUNT:
        passed(f"feature-block sizes sum to {EXPECTED_FEATURE_COUNT}")
    else:
        fail(errors, f"feature-block sizes sum to {block_total}")

    dataset_path = ROOT / "data/CPP_dataset.csv"
    if dataset_path.is_file():
        with dataset_path.open("r", encoding="cp1252", newline="") as handle:
            reader = csv.reader(handle, delimiter=";")
            rows = list(reader)
        data_rows = max(0, len(rows) - 1)
        if data_rows == EXPECTED_DATABASE_ROWS:
            passed(f"CPP_dataset.csv contains {EXPECTED_DATABASE_ROWS} data rows")
        else:
            fail(
                errors,
                f"CPP_dataset.csv contains {data_rows} data rows; "
                f"expected {EXPECTED_DATABASE_ROWS}",
            )

        if rows and len(rows[0]) == 12:
            passed("CPP_dataset.csv has 12 published database columns")
        else:
            fail(errors, "CPP_dataset.csv header does not contain 12 columns")

    cell_path = ROOT / "support/cell_line_tissue_table.csv"
    if cell_path.is_file():
        with cell_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        data_rows = max(0, len(rows) - 1)
        if data_rows == EXPECTED_CELL_LINES:
            passed(f"cell-line tissue table contains {EXPECTED_CELL_LINES} rows")
        else:
            fail(
                errors,
                f"cell-line tissue table contains {data_rows} rows; "
                f"expected {EXPECTED_CELL_LINES}",
            )

    scaler_path = ROOT / "support/scaler.pkl"
    if scaler_path.is_file():
        actual = sha256(scaler_path)
        if actual == EXPECTED_SCALER_SHA256:
            passed(f"scaler SHA256 matches published deployment baseline: {actual}")
        else:
            fail(
                errors,
                f"scaler SHA256 mismatch: {actual}; "
                f"expected {EXPECTED_SCALER_SHA256}",
            )

    print()
    if errors:
        print(f"Repository self-check FAILED with {len(errors)} issue(s).")
        return 1

    print("Repository self-check PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
