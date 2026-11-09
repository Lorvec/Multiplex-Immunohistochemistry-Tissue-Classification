# Compact CNNs and Knowledge Distillation for Multiplex-IHC Tissue Classification

[![MIHIC dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10065510.svg)](https://doi.org/10.5281/zenodo.10065510)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-green.svg)](LICENSE)

Reproducibility code for:

**“Performance–Efficiency Trade-offs of Compact CNNs and Knowledge Distillation for Multiplex Immunohistochemistry Tissue Classification.”**

This repository is intentionally **code-first**. It does not redistribute MIHIC images, trained checkpoints, or archived result tables. Users download the public MIHIC dataset from its official Zenodo record and run the complete experiment locally.

## What the workflow reproduces

The pipeline evaluates:

- Nano, Tiny, and Small compact CNNs;
- single-scale and mixed-kernel depthwise processing;
- matched training with and without knowledge distillation;
- ConvNeXt-Tiny teacher and six additional comparator/baseline architectures;
- three fixed seeds: `42`, `123`, and `2026`;
- 19 configurations × 3 seeds = **57 seed-level training runs**;
- classification, AUROC, calibration, class-specific and stain-specific metrics;
- temperature scaling;
- CPU/GPU model-only efficiency measurements;
- final tables, figures, and machine-readable reports.

## Repository files

```text
README.md
LICENSE
CITATION.cff
requirements.txt
.gitignore

reproduce_mihic.py
mihic_pipeline.py
MIHIC_Reproduce_Full_Study.ipynb

docs/
    DATASET_SETUP.md
    REPRODUCIBILITY.md
```

## 1. Download MIHIC

Download the dataset from the official Zenodo record:

https://doi.org/10.5281/zenodo.10065510

Extract it locally. The directory supplied to the code must contain `train/`, `val/`, and `test/` folders. See [`docs/DATASET_SETUP.md`](docs/DATASET_SETUP.md).

The dataset itself is **not** included in this repository.

## 2. Create the environment

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the CUDA 12.8 software stack used for the study:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

## 3. Run the complete experiment

One command runs the full reproducibility workflow:

```bash
python reproduce_mihic.py \
  --data-root /path/to/MIHIC \
  --output-root MIHIC_reproduction_output
```

The workflow performs the dataset audit, architecture checks, all 57 training runs, hardware benchmarking, final test evaluation, calibration analyses, and report generation.

Completed model/seed runs are stored in the output directory, so interrupted training can be resumed without retraining already completed runs.

## 4. Notebook option

If preferred, open:

```text
MIHIC_Reproduce_Full_Study.ipynb
```

Set `DATA_ROOT` and `OUTPUT_ROOT` in the configuration cell and run the notebook from top to bottom. The notebook calls the same `reproduce_mihic.py` workflow; it is not a separate implementation.

## Fixed paper settings

```text
Seeds:              42, 123, 2026
Maximum epochs:     45
Early stopping:     9 epochs
Warm-up:            3 epochs
Train batch size:   64
Evaluation batch:   128
Optimizer:          AdamW
Scratch/KD LR:      1e-3
Pretrained LR:      3e-4
Weight decay:       1e-4
Label smoothing:    0.05
KD alpha:           0.55
KD temperature:     3.0
Input size:         128 × 128
```

Images are resized to 128×128 pixels, converted to tensors, and normalized using ImageNet statistics. The same deterministic preprocessing is applied to training, validation, and test data.

## Output

The output directory created by the user's run contains, among other files:

```text
validation_by_seed.csv
validation_summary.csv
test_by_seed.csv
efficiency.csv
paper_tables/
paper_figures/
runs/
test_results/
audit/
environment.json
protocol.json
```

The repository does not ship the authors' complete output directory; the goal is to allow each user to regenerate the analyses from the public dataset.

## Reproducibility boundaries

- MIHIC uses the released official partitions. Patient-level identifiers sufficient to independently verify patient-disjoint splitting are not available, so the study does not claim patient-independent generalization.
- Three seeds characterize optimization variability; they are not patient-level uncertainty estimates.
- Runtime measurements are hardware-specific and use synthetic FP32 model-only inputs, not end-to-end whole-slide processing.
- Exact numerical results can vary slightly across hardware/software environments despite fixed seeds and deterministic settings.
- Stain-specific analyses are descriptive because stain subsets have different class composition.

More detail is provided in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data availability

MIHIC is publicly available via Zenodo:

https://doi.org/10.5281/zenodo.10065510

## Citation

A repository citation is provided through [`CITATION.cff`](CITATION.cff). When using the MIHIC images, please also cite the dataset creators:

> Wang R, Qiu Y, Wang T, et al. MIHIC: a multiplex IHC histopathological image classification dataset for lung cancer immune microenvironment quantification. *Frontiers in Immunology*. 2024;15:1334348. https://doi.org/10.3389/fimmu.2024.1334348

## License

Source code in this repository is released under the MIT License unless otherwise stated. MIHIC image data are not redistributed here and remain subject to the terms of the original dataset record.
