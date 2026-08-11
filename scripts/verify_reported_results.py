#!/usr/bin/env python3
"""Audit the archived result tables against the numerical claims in the paper."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mihic_repro.data import (  # noqa: E402
    PUBLISHED_MANIFEST_SHA256,
    manifest_digest,
)


TABLES = REPOSITORY_ROOT / "results" / "tables"
MANIFESTS = REPOSITORY_ROOT / "results" / "manifests"
PROVENANCE = REPOSITORY_ROOT / "provenance"
EXPECTED_MODELS = {
    "teacher_convnext_tiny",
    "proposed_nano_kd",
    "proposed_tiny_kd",
    "proposed_small_kd",
    "ablation_tiny_no_kd",
    "ablation_tiny_with_attention",
    "ablation_tiny_single_scale",
    "control_tinycnn_gap_clean",
    "baseline_efficientnet_b0_scratch",
    "baseline_mobilenet_v3_small_scratch",
    "baseline_mobilenetv4_conv_small_pretrained",
    "baseline_repvit_m0_9_pretrained",
    "baseline_shvit_s2_pretrained",
}
EXPECTED_SEEDS = {42, 123, 2026}


def close(actual: float, expected: float, tolerance: float = 5e-5) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"expected {expected}, got {actual}")


def verify_complete_runs(raw: pd.DataFrame) -> None:
    if raw.shape[0] != 39:
        raise AssertionError(f"expected 39 model-seed rows, found {raw.shape[0]}")
    if set(raw["model_id"]) != EXPECTED_MODELS:
        raise AssertionError("model inventory differs from the frozen 13-model registry")
    for model_id, group in raw.groupby("model_id"):
        if set(group["seed"].astype(int)) != EXPECTED_SEEDS:
            raise AssertionError(f"{model_id}: incomplete seed set")
        if group["dataset_manifest_sha256"].nunique() != 1:
            raise AssertionError(f"{model_id}: inconsistent dataset fingerprints")


def verify_numeric_summary(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    numeric = [
        column
        for column in raw.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    indexed = summary.set_index("model_id")
    for model_id, group in raw.groupby("model_id"):
        for column in numeric:
            expected_mean = group[column].mean()
            expected_std = group[column].std(ddof=1)
            actual_mean = indexed.loc[model_id, column]
            actual_std = indexed.loc[model_id, f"{column}_std"]
            if not np.isclose(actual_mean, expected_mean, equal_nan=True):
                raise AssertionError(f"{model_id}/{column}: mean mismatch")
            if not np.isclose(actual_std, expected_std, equal_nan=True):
                raise AssertionError(f"{model_id}/{column}: SD mismatch")


def verify_headline_claims(summary: pd.DataFrame) -> None:
    rows = summary.set_index("model_id")
    repvit = rows.loc["baseline_repvit_m0_9_pretrained"]
    small = rows.loc["proposed_small_kd"]
    teacher = rows.loc["teacher_convnext_tiny"]
    tiny = rows.loc["proposed_tiny_kd"]
    nano = rows.loc["proposed_nano_kd"]
    no_kd = rows.loc["ablation_tiny_no_kd"]
    single = rows.loc["ablation_tiny_single_scale"]

    close(repvit.test_macro_f1, 0.8179)
    close(repvit.test_macro_f1_std, 0.0086)
    close(small.test_macro_f1, 0.8104)
    close(small.test_macro_f1_std, 0.0057)
    close(teacher.test_macro_f1, 0.8080)
    close(teacher.test_macro_f1_std, 0.0021)
    close(nano.test_macro_f1, 0.8006)
    close(nano.test_macro_f1_std, 0.0107)
    close(tiny.test_macro_f1, 0.8016)

    close(teacher.params / small.params, 11.0, tolerance=0.05)
    close(teacher.gmacs / small.gmacs, 12.4, tolerance=0.05)
    close(
        teacher.checkpoint_size_mb / small.checkpoint_size_mb,
        10.8,
        tolerance=0.05,
    )
    close(teacher.params / nano.params, 92.8, tolerance=0.05)
    close(teacher.gmacs / nano.gmacs, 49.0511890922, tolerance=1e-8)
    close(
        teacher.checkpoint_size_mb / nano.checkpoint_size_mb,
        86.3452579854,
        tolerance=1e-8,
    )

    close(tiny.test_macro_f1 - no_kd.test_macro_f1, 0.0036)
    close(
        tiny.test_macro_roc_auc_ovr - no_kd.test_macro_roc_auc_ovr,
        0.0138,
    )
    close(tiny.test_ece_15bin - no_kd.test_ece_15bin, -0.0171)
    close(tiny.test_nll - no_kd.test_nll, -0.0694)
    close(
        tiny.val_macro_f1 - single.val_macro_f1,
        0.0011,
        tolerance=6e-5,
    )
    close(
        tiny.test_macro_f1 - single.test_macro_f1,
        -0.0060,
        tolerance=6e-5,
    )


def verify_per_class_and_stains() -> None:
    per_class = pd.read_csv(TABLES / "table_2_per_class_full.csv")
    piv = per_class.pivot(index="model_id", columns="class", values="f1_mean")
    differences = piv.loc["proposed_small_kd"] - piv.loc["teacher_convnext_tiny"]
    expected = {
        "Immune cells": 0.0319,
        "Necrosis": -0.0344,
        "Other": -0.0058,
        "Stroma": 0.0122,
        "Tumor": 0.0014,
        "alveoli": 0.0117,
        "background": -0.0002,
    }
    for class_name, value in expected.items():
        close(differences[class_name], value, tolerance=6e-5)

    subgroups = pd.read_csv(TABLES / "table_6_subgroup_performance.csv")
    stains = subgroups[subgroups["subgroup_type"] == "stain"]
    means = stains.groupby(["model_id", "subgroup"])["macro_f1"].mean().unstack(0)
    stain_diff = (
        means["proposed_small_kd"] - means["teacher_convnext_tiny"]
    )
    if int((stain_diff > 0).sum()) != 9:
        raise AssertionError("Small must exceed the teacher in 9 of 12 stains")
    for stain, expected_delta in {
        "SMA": 0.0464,
        "Ki67": 0.0339,
        "CD20": 0.0321,
        "CDK4": 0.0309,
        "CD34": 0.0292,
        "CD38": -0.0267,
        "CD68": -0.0127,
        "cyclin-D1": -0.0043,
    }.items():
        close(stain_diff[stain], expected_delta, tolerance=6e-5)


def verify_manifests_and_source() -> None:
    manifest = pd.read_csv(
        MANIFESTS / "dataset_manifest.csv.gz",
        dtype={"section_id_heuristic": "string"},
    )
    if len(manifest) != 309_698:
        raise AssertionError("portable dataset manifest must have 309,698 rows")
    digest = manifest_digest(manifest)
    if digest != PUBLISHED_MANIFEST_SHA256:
        raise AssertionError(f"dataset manifest digest mismatch: {digest}")
    if "absolute_path" in manifest.columns:
        raise AssertionError("public manifest must not disclose local absolute paths")

    expected_source_hash = (
        PROVENANCE / "executed_cell_source_sha256.txt"
    ).read_text(encoding="utf-8").strip()
    actual_source_hash = hashlib.sha256(
        (PROVENANCE / "executed_cell_source.py").read_bytes()
    ).hexdigest()
    if actual_source_hash != expected_source_hash:
        raise AssertionError("frozen executed source hash mismatch")


def verify_artifact_inventory() -> None:
    required_tables = {
        "raw_seed_results.csv",
        "summary_mean_std_numeric.csv",
        "per_class_raw_seed_results.csv",
        "table_1_main_performance.csv",
        "table_2_per_class_full.csv",
        "table_2b_per_class_f1_matrix.csv",
        "table_3_efficiency.csv",
        "table_4_ablation.csv",
        "table_5_architecture_inventory.csv",
        "table_6_subgroup_performance.csv",
        "table_dataset_distribution.csv",
    }
    actual_tables = {path.name for path in TABLES.glob("*.csv")}
    if required_tables - actual_tables:
        raise AssertionError(f"missing tables: {sorted(required_tables - actual_tables)}")
    figures = list((REPOSITORY_ROOT / "results" / "paper_figures").glob("*.png"))
    if len(figures) != 5:
        raise AssertionError("expected five paper-facing figures")


def main() -> int:
    raw = pd.read_csv(TABLES / "raw_seed_results.csv")
    summary = pd.read_csv(TABLES / "summary_mean_std_numeric.csv")
    verify_complete_runs(raw)
    verify_numeric_summary(raw, summary)
    verify_headline_claims(summary)
    verify_per_class_and_stains()
    verify_manifests_and_source()
    verify_artifact_inventory()
    print("PASS: 13 models x 3 seeds are complete.")
    print("PASS: archived means/SDs reproduce the seed-level exports.")
    print("PASS: headline compression, KD, class, and stain claims match the paper.")
    print("PASS: portable dataset manifest and frozen source hashes match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
