"""Dataset metadata helpers; no MIHIC image is distributed by this repository."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


CLASS_NAMES = (
    "Immune cells",
    "Necrosis",
    "Other",
    "Stroma",
    "Tumor",
    "alveoli",
    "background",
)

EXPECTED_SPLIT_TOTALS = {
    "train": 195_001,
    "val": 49_260,
    "test": 65_437,
}

PUBLISHED_MANIFEST_SHA256 = (
    "70d8f78674402f157adc21b7480b1458ee434477d41cfff4425ffc155b21d996"
)

STAIN_ALIASES = {
    "CD20": ("cd20",),
    "CD34": ("cd34",),
    "CD38": ("cd38",),
    "CD3": ("cd3",),
    "CD68": ("cd68",),
    "CDK4": ("cdk4",),
    "cyclin-D1": ("cyclin-d1", "cyclind1", "cyclin_d1"),
    "D2-40": ("d2-40", "d240", "d2_40"),
    "FAP": ("fap",),
    "Ki67": ("ki67", "ki-67", "ki_67"),
    "P53": ("p53",),
    "SMA": ("sma",),
}

STAIN_MATCHERS = sorted(
    [
        (
            len(alias),
            canonical,
            re.compile(
                rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])",
                flags=re.IGNORECASE,
            ),
        )
        for canonical, aliases in STAIN_ALIASES.items()
        for alias in aliases
    ],
    key=lambda item: item[0],
    reverse=True,
)


def infer_stain(path_text: str) -> str | None:
    for _, canonical, pattern in STAIN_MATCHERS:
        if pattern.search(path_text):
            return canonical
    return None


def infer_section(
    path_text: str,
    section_regex: str = r"(?<!\d)(\d{4})(?!\d)",
) -> str | None:
    match = re.search(section_regex, Path(path_text).name)
    return match.group(1) if match else None


def manifest_digest(frame: pd.DataFrame) -> str:
    """Hash the four portable fields used by the frozen experiment."""

    required = ["split", "relative_path", "class", "file_size_bytes"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    digest_source = frame[required].to_csv(index=False)
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
