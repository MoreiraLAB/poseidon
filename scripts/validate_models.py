#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "210c95cff2bb8b01fdf1e43bd2d98a887fdaf60f"

ML_METHODS = ["linear", "svm", "sgd", "knn", "dt", "rf", "etr", "xgb"]


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        return {row[0]: float(row[1]) for row in csv.reader(handle) if row}


def historical_metrics(path: str) -> dict[str, float]:
    data = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)
    rows = csv.reader(data.decode("utf-8").splitlines())
    return {row[0]: float(row[1]) for row in rows if row}


def compare_metrics(actual: Path, historical_path: str, atol: float = 1e-8) -> None:
    a = read_metrics(actual)
    e = historical_metrics(historical_path)
    if set(a) != set(e):
        raise AssertionError(f"{actual}: metric keys differ")
    for key in a:
        if not math.isclose(a[key], e[key], rel_tol=1e-8, abs_tol=atol):
            raise AssertionError(
                f"{actual}: {key}={a[key]} differs from historical {e[key]}"
            )
    print(f"PASS: {actual.relative_to(ROOT)} matches historical metrics")


def validate_ml() -> None:
    for method in ML_METHODS:
        if method == "xgb":
            hist_train = "results/XGB_train.csv"
            hist_test = "results/XGB_test.csv"
        else:
            hist_train = f"results/{method}_train.csv"
            hist_test = f"results/{method}_test.csv"

        compare_metrics(ROOT / "results" / f"{method}_train.csv", hist_train, atol=1e-7)
        compare_metrics(ROOT / "results" / f"{method}_test.csv", hist_test, atol=1e-7)

        model = ROOT / "models" / f"{method}.h5"
        if not model.is_file() or model.stat().st_size == 0:
            raise AssertionError(f"missing or empty model: {model}")
        print(f"PASS: generated model exists: {model.relative_to(ROOT)}")


def validate_dl() -> None:
    for name in ["forked_train", "forked_test", "DL_train", "DL_test"]:
        path = ROOT / "results" / f"{name}.csv"
        if not path.is_file():
            raise AssertionError(f"missing DL metrics: {path}")
        metrics = read_metrics(path)
        for key, value in metrics.items():
            if not math.isfinite(value):
                raise AssertionError(f"{path}: non-finite {key}={value}")
        print(f"PASS: finite DL metrics: {path.relative_to(ROOT)}")

    for path in [
        ROOT / "models" / "DL_model.h5",
        ROOT / "models" / "forked_DL_model.h5py",
    ]:
        if not path.exists():
            raise AssertionError(f"missing DL model: {path}")
        print(f"PASS: generated DL model exists: {path.relative_to(ROOT)}")

    dl_test = read_metrics(ROOT / "results" / "DL_test.csv")
    if dl_test["Pearson"] < 0.75 or dl_test["Spearman"] < 0.75 or dl_test["r2"] < 0.55:
        raise AssertionError(f"DL test performance unexpectedly low: {dl_test}")
    print("PASS: regenerated DL model retains expected predictive performance")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ml":
        validate_ml()
    elif mode == "dl":
        validate_dl()
    else:
        raise SystemExit("usage: validate_models.py [ml|dl]")
