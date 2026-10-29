#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "210c95cff2bb8b01fdf1e43bd2d98a887fdaf60f"


def git_show(path: str, out: Path) -> None:
    data = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)
    out.write_bytes(data)


def assert_frame_equal_csv(
    actual: Path,
    historical_path: str,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    check_column_order: bool = True,
    sort_by: str | None = None,
    atol: float = 1e-10,
) -> None:
    expected_path = Path("/tmp") / ("expected_" + actual.name)
    git_show(historical_path, expected_path)

    a = pd.read_csv(actual, sep=sep, encoding=encoding)
    e = pd.read_csv(expected_path, sep=sep, encoding=encoding)

    if sort_by is not None:
        a = a.sort_values(sort_by).reset_index(drop=True)
        e = e.sort_values(sort_by).reset_index(drop=True)

    if check_column_order:
        if list(a.columns) != list(e.columns):
            raise AssertionError(
                f"{actual}: column order differs\n"
                f"actual first={list(a.columns[:10])}\n"
                f"expected first={list(e.columns[:10])}"
            )
    else:
        if set(a.columns) != set(e.columns):
            raise AssertionError(f"{actual}: column sets differ")
        cols = sorted(a.columns)
        a = a[cols]
        e = e[cols]

    if a.shape != e.shape:
        raise AssertionError(f"{actual}: shape {a.shape} != historical {e.shape}")

    for col in a.columns:
        av, ev = a[col], e[col]
        if pd.api.types.is_numeric_dtype(av) and pd.api.types.is_numeric_dtype(ev):
            if not np.allclose(
                av.to_numpy(dtype=float),
                ev.to_numpy(dtype=float),
                rtol=1e-10,
                atol=atol,
                equal_nan=True,
            ):
                diff = np.nanmax(
                    np.abs(av.to_numpy(dtype=float) - ev.to_numpy(dtype=float))
                )
                raise AssertionError(f"{actual}: numeric mismatch in {col}; max abs={diff}")
        else:
            if not av.fillna("<NA>").astype(str).equals(
                ev.fillna("<NA>").astype(str)
            ):
                raise AssertionError(f"{actual}: text mismatch in {col}")

    print(f"PASS: {actual.relative_to(ROOT)} matches historical {historical_path}")


def compare_numeric_file(actual: Path, historical_path: str, atol: float = 1e-10) -> None:
    expected_path = Path("/tmp") / ("expected_" + actual.name)
    git_show(historical_path, expected_path)
    a = np.genfromtxt(actual, delimiter=",")
    e = np.genfromtxt(expected_path, delimiter=",")
    if a.shape != e.shape:
        raise AssertionError(f"{actual}: shape {a.shape} != {e.shape}")
    if not np.allclose(a, e, rtol=1e-10, atol=atol, equal_nan=True):
        raise AssertionError(
            f"{actual}: values differ; max abs={np.nanmax(np.abs(a-e))}"
        )
    print(f"PASS: {actual.relative_to(ROOT)} matches historical {historical_path}")


def compare_scaler() -> None:
    expected_path = Path("/tmp/expected_scaler.pkl")
    git_show("support/scaler.pkl", expected_path)
    with (ROOT / "support/scaler.pkl").open("rb") as f:
        a = pickle.load(f)
    with expected_path.open("rb") as f:
        e = pickle.load(f)

    for attr in ("mean_", "scale_", "var_"):
        av = getattr(a, attr)
        ev = getattr(e, attr)
        if not np.allclose(av, ev, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"scaler {attr} differs")
    if int(a.n_features_in_) != int(e.n_features_in_) != 1330:
        raise AssertionError("unexpected scaler feature count")
    print("PASS: regenerated scaler parameters match historical scaler")


def main(stage: str) -> None:
    if stage == "process":
        table = pd.read_csv(ROOT / "cargo_encoded.csv")
        if table.shape[0] != 1315:
            raise AssertionError(
                f"cargo_encoded.csv has {table.shape[0]} rows; expected 1315"
            )
        if len(table.columns) != len(set(table.columns)):
            raise AssertionError("cargo_encoded.csv contains duplicate column names")
        cargo_columns = [c for c in table.columns if c.startswith("cargo_")]
        if len(cargo_columns) != 96:
            raise AssertionError(
                f"cargo_encoded.csv has {len(cargo_columns)} cargo columns; expected 96"
            )
        required = {
            "target", "conc_uM", "time_min", "time_missing",
            "temperature_4", "temperature_23", "temperature_25",
            "temperature_37", "temperature_42", "cargo_tmr",
        }
        missing = sorted(required.difference(table.columns))
        if missing:
            raise AssertionError(f"cargo_encoded.csv missing required columns: {missing}")
        print(
            "PASS: raw CPP processing produced 1315 rows, "
            "96 unique cargo columns and the expected experimental fields"
        )
    elif stage == "merge":
        assert_frame_equal_csv(
            ROOT / "data/processed_dataset.csv",
            "data/processed_dataset.csv",
            sep=",",
            encoding="utf-8",
        )
    elif stage == "gdsc":
        # Legacy code builds gene columns via set(...), so compare values after
        # canonical column ordering rather than requiring historical hash order.
        assert_frame_equal_csv(
            ROOT / "data/GDSC_features.csv",
            "data/GDSC_features.csv",
            check_column_order=False,
            sort_by="cell_line",
        )
    elif stage == "features":
        assert_frame_equal_csv(
            ROOT / "data/data_with_features.csv",
            "data/data_with_features.csv",
            sep=",",
            encoding="utf-8",
            check_column_order=False,
            atol=1e-8,
        )
    elif stage == "processed":
        assert_frame_equal_csv(
            ROOT / "data/log_10_processed_data.csv",
            "data/log_10_processed_data.csv",
            sep=",",
            encoding="utf-8",
            atol=1e-8,
        )

        expected_header = subprocess.check_output(
            ["git", "show", f"{BASELINE}:support/features_header.txt"], cwd=ROOT
        ).decode().splitlines()
        actual_header = (ROOT / "support/features_header.txt").read_text().splitlines()
        if actual_header != expected_header:
            raise AssertionError("features_header.txt differs from historical baseline")
        if len(actual_header) != 1330:
            raise AssertionError(f"expected 1330 features, got {len(actual_header)}")
        print("PASS: features_header.txt matches historical 1330-feature schema")

        for name in [
            "train_features.csv",
            "test_features.csv",
            "train_target.csv",
            "test_taget.csv",
        ]:
            compare_numeric_file(ROOT / "data" / name, "data/" + name, atol=1e-8)

        for name in ["train_IDs.csv", "test_IDs.csv"]:
            assert_frame_equal_csv(
                ROOT / "data" / name,
                "data/" + name,
                sep=",",
                encoding="utf-8",
            )

        compare_scaler()
    else:
        raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    main(sys.argv[1])
