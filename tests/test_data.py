from pathlib import Path

import pandas as pd

from mihic_repro.data import (
    PUBLISHED_MANIFEST_SHA256,
    infer_section,
    infer_stain,
    manifest_digest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_stain_matchers_avoid_cd3_substring_collisions():
    assert infer_stain("section/CD3_patch_001.png") == "CD3"
    assert infer_stain("section/CD34_patch_001.png") == "CD34"
    assert infer_stain("section/CD38_patch_001.png") == "CD38"


def test_section_inference():
    assert infer_section("Immune cells/0009-CD20-10-12.png") == "0009"


def test_public_manifest_is_portable_and_matches_published_digest():
    frame = pd.read_csv(
        REPOSITORY_ROOT
        / "results"
        / "manifests"
        / "dataset_manifest.csv.gz",
        dtype={"section_id_heuristic": "string"},
    )
    assert len(frame) == 309_698
    assert "absolute_path" not in frame.columns
    assert manifest_digest(frame) == PUBLISHED_MANIFEST_SHA256
