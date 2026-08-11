# Parameter-Efficient Knowledge-Distilled CNNs for MIHIC

[![Verify archived results](https://github.com/Lorvec/Multiplex-Immunohistochemistry-Tissue-Classification/actions/workflows/verify.yml/badge.svg)](https://github.com/Lorvec/Multiplex-Immunohistochemistry-Tissue-Classification/actions/workflows/verify.yml)
[![MIHIC dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10065510.svg)](https://doi.org/10.5281/zenodo.10065510)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-green.svg)](LICENSE)

Official reproducibility repository for **“Parameter-Efficient Knowledge-Distilled CNNs for Multiplex Immunohistochemistry Tissue Classification.”**

The study evaluates a validation-frozen Nano–Tiny–Small CNN family, offline knowledge distillation from an ImageNet-pretrained ConvNeXt-Tiny teacher, contemporary compact comparators, calibration, per-class and stain-level behavior, and measured hardware efficiency on the official MIHIC splits.

## What is included

- The exact source executed for the reported experiment, preserved unchanged in [`provenance/executed_cell_source.py`](provenance/executed_cell_source.py) with a SHA-256 checksum.
- A portable command-line runner, [`run_experiment.py`](run_experiment.py), driven by the frozen JSON configurations in [`configs/`](configs/).
- Reusable implementations of the proposed architectures and paper metrics in [`src/mihic_repro/`](src/mihic_repro/).
- All 39 model–seed metric rows (13 architectures × seeds 42, 123, and 2026), complete per-class and subgroup exports, paper-facing tables, figures, environment metadata, and the paper workbook in [`results/`](results/).
- A portable dataset manifest containing filenames, class labels, stain/section metadata, and file sizes—but no image data and no private absolute paths.
- Automated checks that recompute every archived mean/standard deviation and audit the paper’s main numerical comparisons.

The repository does **not** redistribute MIHIC images, model checkpoints, or the 7.6 GB dataset archive.

## Verify the published results without retraining

Create a Python 3.12 environment and install the lightweight reporting dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/reporting.txt
```

Audit the archived seed-level results and rebuild all five paper figures:

```bash
python scripts/verify_reported_results.py
python scripts/make_paper_figures.py --output-dir reproduced_figures
```

The verifier checks:

1. completeness of all 13 models × 3 seeds;
2. exact reconstruction of every numeric mean and sample standard deviation;
3. the reported compression, knowledge-distillation, ablation, class-specific, and stain-specific comparisons;
4. the 309,698-row portable manifest digest; and
5. the checksum of the frozen executed source.

## Obtain and validate MIHIC

Download `MIHIC_dataset.zip` from the [official Zenodo record](https://doi.org/10.5281/zenodo.10065510). The source record reports a 7.6 GB archive with MD5:

```text
bd718137f70427d530947862332f4cdd
```

Extract it outside this repository. Point the validator to the directory that contains—or contains a nested directory with—`train/`, `val/`, and `test/`:

```bash
python scripts/validate_dataset.py --data-root /path/to/extracted/MIHIC
```

The full check compares all filenames and file sizes and must produce this portable-manifest SHA-256:

```text
70d8f78674402f157adc21b7480b1458ee434477d41cfff4425ffc155b21d996
```

For a faster folder/count-only check, add `--quick`. See [`docs/DATASET.md`](docs/DATASET.md) for the expected directory structure and split counts.

## Reproduce the full experiment

The paper environment used Python 3.12.3, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, timm 1.0.24, CUDA runtime 12.8, and cuDNN 9.10.2. For the matching Linux/Windows CUDA 12.8 wheels:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/paper.txt
```

Then validate the dataset and launch the frozen FINAL configuration:

```bash
python scripts/validate_dataset.py --data-root /path/to/MIHIC

python run_experiment.py \
  --config configs/final.json \
  --data-root /path/to/MIHIC \
  --output-root outputs
```

FINAL trains and evaluates the complete 13-model registry for seeds 42, 123, and 2026. It downloads the declared ImageNet weights on first use. Runs are resumable: a completed model/seed directory is skipped unless `--force-rerun` is supplied.

The two pre-final modes remain available and do not load the official test directory:

```bash
python run_experiment.py --config configs/pilot.json --data-root /path/to/MIHIC
python run_experiment.py --config configs/freeze_check.json --data-root /path/to/MIHIC
```

Important: the paper results came from the frozen sequence PILOT → FREEZE_CHECK → FINAL. Do not use the test set to revise architectures or hyperparameters.

## Principal archived results

| Model | Parameters (M) | GMACs | Test Macro-F1 (mean ± SD) |
|---|---:|---:|---:|
| RepViT M0.9 (pretrained) | 4.725 | 0.2716 | 0.8179 ± 0.0086 |
| Proposed Small (KD) | 2.528 | 0.1174 | 0.8104 ± 0.0057 |
| ConvNeXt-Tiny teacher | 27.826 | 1.4595 | 0.8080 ± 0.0021 |
| MobileNetV4 Conv Small (pretrained) | 2.502 | 0.0625 | 0.8025 ± 0.0072 |
| Proposed Tiny (KD) | 0.819 | 0.0391 | 0.8016 ± 0.0100 |
| Proposed Nano (KD) | 0.300 | 0.0298 | 0.8006 ± 0.0107 |

The central result is performance preservation under compression, not universal benchmark dominance. In the recorded environment, latency rankings depended on device and batch size; hardware measurements should not be treated as portable constants.

## Repository map

```text
configs/                 Frozen PILOT, FREEZE_CHECK, and FINAL settings
docs/                    Dataset and reproduction details
provenance/              Unchanged executed source and checksum
requirements/            Exact paper and lightweight reporting dependencies
results/
  manifests/             Environment, methods snapshot, portable data manifest
  paper_figures/         Final five manuscript figures
  supplementary_figures/All-model confusion matrices and heatmaps
  tables/                Raw seeds, summaries, per-class, stain, ablation CSVs
  MIHIC_paper_tables.xlsx
scripts/                 Dataset validation, numerical audit, figure rebuilding
src/mihic_repro/         Reusable model, metric, and metadata components
tests/                   Architecture, metric, manifest, and result checks
run_experiment.py        Complete portable experiment runner
```

## Reproducibility boundaries

- The official release groups patches from one annotated region into one split, but patient-independent separation cannot be verified because patient identifiers/mappings are unavailable.
- Exact model metrics can vary slightly across software/hardware despite the recorded seeds and deterministic cuDNN settings. The archived values are the actual paper run; no claim of bitwise reproducibility across platforms is made.
- CPU/GPU latency, throughput, peak memory, and training time are hardware-specific. The paper environment is recorded in [`results/manifests/environment.json`](results/manifests/environment.json).
- The public Figure 5 source table contains the two-decimal values displayed in the manuscript. A full rerun writes the underlying per-seed count matrices and prediction files before reconstructing the aggregated matrix.

More detail is available in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation

Until the article receives its final bibliographic record, use the repository citation exposed by [`CITATION.cff`](CITATION.cff). When using the MIHIC images, also cite the dataset creators:

> Wang R, Qiu Y, Wang T, et al. MIHIC: a multiplex IHC histopathological image classification dataset for lung cancer immune microenvironment quantification. *Frontiers in Immunology*. 2024;15:1334348. https://doi.org/10.3389/fimmu.2024.1334348

## License and data ownership

Repository code is released under the [MIT License](LICENSE). MIHIC images are not part of this repository and remain governed by the terms stated in the [official Zenodo dataset record](https://doi.org/10.5281/zenodo.10065510). Cite the dataset when using it.
