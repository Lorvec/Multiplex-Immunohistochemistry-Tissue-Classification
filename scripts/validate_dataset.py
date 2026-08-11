#!/usr/bin/env python3
"""Validate a local MIHIC extraction against the published portable manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mihic_repro.data import (  # noqa: E402
    CLASS_NAMES,
    EXPECTED_SPLIT_TOTALS,
    PUBLISHED_MANIFEST_SHA256,
    manifest_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check a MIHIC download without copying any image into this repository."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing train/, val/, and test/.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check folder names and image counts but skip per-file size/hash checks.",
    )
    return parser.parse_args()


def resolve_data_root(candidate: Path) -> Path:
    candidate = candidate.expanduser().resolve()
    if all((candidate / split).is_dir() for split in EXPECTED_SPLIT_TOTALS):
        return candidate
    matches = [
        path
        for path in candidate.glob("*/*")
        if path.is_dir()
        and all((path / split).is_dir() for split in EXPECTED_SPLIT_TOTALS)
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not identify one directory with train/, val/, and test/ below {candidate}"
    )


def actual_relative_paths(data_root: Path, split: str) -> list[str]:
    split_root = data_root / split
    class_names = tuple(sorted(path.name for path in split_root.iterdir() if path.is_dir()))
    if class_names != tuple(sorted(CLASS_NAMES)):
        raise AssertionError(
            f"{split}: class folders differ. Expected {CLASS_NAMES}; found {class_names}"
        )
    return sorted(
        path.relative_to(split_root).as_posix()
        for path in split_root.rglob("*.png")
        if path.is_file()
    )


def main() -> int:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    print(f"Resolved MIHIC root: {data_root}")

    expected_path = (
        REPOSITORY_ROOT / "results" / "manifests" / "dataset_manifest.csv.gz"
    )
    expected = pd.read_csv(
        expected_path,
        dtype={"section_id_heuristic": "string"},
    )

    for split, expected_total in EXPECTED_SPLIT_TOTALS.items():
        expected_split = expected[expected["split"] == split].sort_values(
            "sample_index"
        )
        actual_paths = actual_relative_paths(data_root, split)
        if len(actual_paths) != expected_total:
            raise AssertionError(
                f"{split}: expected {expected_total:,} PNG files, found {len(actual_paths):,}"
            )
        expected_paths = expected_split["relative_path"].tolist()
        if actual_paths != expected_paths:
            missing = sorted(set(expected_paths) - set(actual_paths))[:10]
            extra = sorted(set(actual_paths) - set(expected_paths))[:10]
            raise AssertionError(
                f"{split}: filenames differ; first missing={missing}, first extra={extra}"
            )
        print(f"  {split}: {len(actual_paths):,} filenames match")

    if args.quick:
        print("PASS: MIHIC folder names and split counts match the paper manifest.")
        return 0

    verified = expected.copy()
    actual_sizes: list[int] = []
    for row in verified.itertuples(index=False):
        actual_sizes.append(
            (data_root / row.split / row.relative_path).stat().st_size
        )
    verified["file_size_bytes"] = actual_sizes
    mismatches = verified[
        verified["file_size_bytes"].to_numpy()
        != expected["file_size_bytes"].to_numpy()
    ]
    if not mismatches.empty:
        examples = mismatches[
            ["split", "relative_path", "file_size_bytes"]
        ].head(10)
        raise AssertionError(
            "File sizes differ from the published release:\n"
            + examples.to_string(index=False)
        )

    digest = manifest_digest(verified)
    if digest != PUBLISHED_MANIFEST_SHA256:
        raise AssertionError(
            f"Manifest digest mismatch: expected {PUBLISHED_MANIFEST_SHA256}, got {digest}"
        )
    print(f"PASS: all 309,698 files and manifest SHA-256 match ({digest}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
