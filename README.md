# Compact CNNs and Knowledge Distillation for MIHIC Tissue Classification

[![MIHIC dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10065510.svg)](https://doi.org/10.5281/zenodo.10065510)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-green.svg)](LICENSE)

Official reproducibility repository for the study:

**“Performance–Efficiency Trade-offs of Compact CNNs and Knowledge Distillation for Multiplex Immunohistochemistry Tissue Classification.”**

This study evaluates compact convolutional neural networks for tissue classification on the public MIHIC benchmark. The experiments systematically compare model capacity, knowledge distillation, single-scale versus mixed-kernel depthwise processing, probability quality, and measured computational efficiency.

The study includes **19 model configurations evaluated across three training seeds (42, 123, and 2026), corresponding to 57 seed-level experiments**.

---

## Study Overview

The compact-model analysis uses a full-factorial design across:

- three model sizes: **Nano, Tiny, and Small**;
- two depthwise-kernel configurations:
  - single-scale 3×3 depthwise convolution;
  - mixed-kernel 3×3/5×5 depthwise processing;
- two training objectives:
  - standard supervised training;
  - knowledge distillation from a ConvNeXt-Tiny teacher.

This results in **12 compact configurations**.

Seven additional reference and comparator configurations are evaluated:

- ConvNeXt-Tiny teacher;
- RepViT M0.9;
- MobileNetV4 Conv Small;
- SHViT-S2;
- EfficientNet-B0;
- MobileNetV3-Small;
- TinyCNN baseline.

All models use the official MIHIC training, validation, and test partitions.

---

## Dataset

The study uses the publicly available **MIHIC** dataset:

> Wang R, Qiu Y, Wang T, et al. MIHIC: a multiplex IHC histopathological image classification dataset for lung cancer immune microenvironment quantification. *Frontiers in Immunology*. 2024;15:1334348.

Dataset DOI:

https://doi.org/10.5281/zenodo.10065510

The dataset contains **309,698 non-overlapping 128×128 image patches** from 47 tissue-microarray sections representing 114 patients and 12 IHC stains.

The seven tissue categories are:

- Immune
- Necrosis
- Other
- Stroma
- Tumor
- Alveoli
- Background

Official split sizes:

| Partition | Patches |
|---|---:|
| Training | 195,001 |
| Validation | 49,260 |
| Test | 65,437 |
| **Total** | **309,698** |

The public release groups patches originating from the same annotated region into one partition. However, patient identifiers and a patient-to-split mapping are not available. Therefore, **patient-independent separation cannot be independently verified**, and the reported results should be interpreted as performance on the supplied MIHIC benchmark rather than evidence of patient-independent clinical generalization.

---

## Compact CNN Configurations

The three compact model sizes use the following trainable parameter counts:

| Model size | Single-scale | Mixed-kernel |
|---|---:|---:|
| Nano | 291,011 | 299,843 |
| Tiny | 804,451 | 819,427 |
| Small | 2,494,407 | 2,528,199 |

All compact models use:

- inverted-residual blocks;
- batch normalization;
- SiLU activation;
- depthwise convolutions;
- squeeze-and-excitation;
- global adaptive average pooling;
- a seven-class linear classifier.

The mixed-kernel configuration partitions channels between 3×3 and 5×5 depthwise convolutions following established **MixConv** principles.

---

## Knowledge Distillation

The teacher is an ImageNet-1K-pretrained **ConvNeXt-Tiny** model.

Student training combines weighted label-smoothed cross-entropy with temperature-scaled Kullback–Leibler divergence:

```text
L = (1 − α) L_CE + α T² L_KL

α = 0.55
T = 3.0
```

For every Nano, Tiny, and Small architecture, a matched control without knowledge distillation is evaluated using the same optimization settings.

Teacher logits are cached on normalized training images and associated with student samples using a verified sample-index mapping.

---

## Training Protocol

The main training configuration uses:

```text
Seeds:              42, 123, 2026
Maximum epochs:     45
Early stopping:     9 epochs
Warm-up:            3 epochs
Train batch size:   64
Evaluation batch:   128
Optimizer:          AdamW
Weight decay:       1e-4
Label smoothing:    0.05
Scratch/KD LR:      1e-3
Pretrained LR:      3e-4
Image size:         128 × 128
```

Images are resized to 128×128 pixels, converted to tensors, and normalized using ImageNet channel statistics. The same deterministic preprocessing pipeline is applied to the training, validation, and test partitions.

Checkpoint selection is based on validation Macro-F1.

---

## Main Results

The highest mean test Macro-F1 was obtained by **RepViT M0.9**.

| Model | Parameters (M) | Test Macro-F1 |
|---|---:|---:|
| RepViT M0.9 | 4.725 | **0.8122 ± 0.0069** |
| Small single-scale + KD | 2.494 | **0.8102 ± 0.0044** |
| Small mixed-kernel + KD | 2.528 | 0.8068 ± 0.0034 |
| ConvNeXt-Tiny teacher | 27.826 | 0.8008 ± 0.0116 |
| Nano mixed-kernel + KD | 0.300 | 0.7965 ± 0.0123 |
| Tiny single-scale + KD | 0.804 | 0.7940 ± 0.0074 |
| MobileNetV4 Conv Small | 2.502 | 0.7892 ± 0.0051 |

The purpose of the study is not to identify a universally dominant architecture, but to characterize **performance–efficiency trade-offs across model size, training strategy, and architecture configuration**.

---

## Knowledge-Distillation Ablation

Knowledge distillation increased mean test Macro-F1 in all six matched compact-model comparisons:

| Configuration | Δ Macro-F1 |
|---|---:|
| Nano single-scale | +0.0031 |
| Nano mixed-kernel | +0.0132 |
| Tiny single-scale | +0.0120 |
| Tiny mixed-kernel | +0.0092 |
| Small single-scale | +0.0143 |
| Small mixed-kernel | +0.0049 |

Mean AUROC also increased in all six comparisons, while mean negative log-likelihood decreased in all six.

Expected calibration error improved in only three of the six comparisons, indicating that the effect of knowledge distillation on calibration depends on the architecture and calibration metric.

---

## Single-Scale versus Mixed-Kernel Processing

Mixed 3×3/5×5 depthwise processing did **not** provide a consistent advantage over the simpler single-scale configuration.

With knowledge distillation, the mean test Macro-F1 difference for mixed-kernel versus single-scale processing was:

```text
Nano:  +0.0057
Tiny:  +0.0004
Small: −0.0034
```

Without knowledge distillation:

```text
Nano:  −0.0044
Tiny:  +0.0032
Small: +0.0059
```

The direction of the effect therefore varies with model size and training objective.

---

## Probability Calibration

Probability quality is assessed using:

- negative log-likelihood (NLL);
- expected calibration error (ECE);
- classwise calibration summaries;
- post-hoc temperature scaling.

Temperature scaling is fitted using validation predictions and evaluated on the corresponding test predictions.

Across the 19 configurations, temperature scaling reduced mean NLL in **13/19** configurations but reduced mean ECE in only **4/19**, showing that post-hoc scaling did not provide a uniform calibration improvement.

---

## Computational Efficiency

Efficiency reporting includes:

- trainable parameters;
- approximate convolution-plus-linear GMACs;
- checkpoint storage;
- CPU batch-1 latency;
- GPU batch-1 latency;
- GPU batch-64 throughput;
- peak GPU inference memory.

Runtime measurements use synthetic FP32 model-only inputs and should not be interpreted as end-to-end whole-slide or clinical pipeline latency.

For example:

| Model | Params (M) | GMACs | GPU b1 (ms) | GPU b64 (img/s) | CPU b1 (ms) |
|---|---:|---:|---:|---:|---:|
| Small single + KD | 2.494 | 0.108 | 2.60 | 16,494 | 5.04 |
| RepViT M0.9 | 4.725 | 0.267 | 4.58 | 12,027 | 11.94 |
| ConvNeXt-Tiny teacher | 27.826 | 1.455 | 2.21 | 8,813 | 14.44 |
| Nano mixed-kernel + KD | 0.300 | 0.028 | 2.30 | 25,277 | 3.01 |
| Tiny single + KD | 0.804 | 0.036 | 1.58 | 41,700 | 2.53 |

Lower parameter count or theoretical operation count does not necessarily imply lower latency in every hardware setting.

---

## Hardware and Software Environment

The archived experimental environment records:

```text
Python:       3.12.3
PyTorch:      2.9.1+cu128
torchvision:  0.24.1+cu128
timm:         1.0.24
CUDA runtime: 12.8
cuDNN:        9.10.2
GPU:          NVIDIA GeForce RTX 5090
```

CPU measurements used eight PyTorch threads under WSL2.

Detailed environment metadata are included with the reproducibility materials.

---

## Repository Contents

The repository contains the code and machine-readable outputs required to inspect and reproduce the reported analyses.

```text
code/
    MIHIC_NoAug_Full_Study.ipynb
    mihic_noaug_full_study.py

results/
    validation_by_seed.csv
    validation_summary.csv
    test_by_seed.csv
    efficiency.csv

figures/
    Figure_1_Study_Design.png
    Figure_2_KD_Effects.png
    Figure_3_Performance_Efficiency.png
    Figure_4_Class_Specific_F1.png
    Figure_5_Stain_Specific_F1.png

supplementary/
    architecture_specifications.csv
    full_model_metrics.csv
    kd_ablation_results.csv
    single_vs_multiscale.csv
    calibration_outputs/
    reporting_metadata.json

environment/
    environment.json
```

The exact directory names may evolve slightly as the public release is finalized, but all files reported in the manuscript will be included before publication.

The repository does **not** redistribute MIHIC image data.

---

## Reproduce the Study

Download the MIHIC dataset from:

https://doi.org/10.5281/zenodo.10065510

Create a Python environment and install the required packages.

Example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install timm==1.0.24
```

Place the extracted MIHIC dataset outside the repository and configure the dataset path in the notebook or command-line runner.

The expected dataset structure contains:

```text
train/
val/
test/
```

The study includes audit, architecture smoke tests, training, validation reporting, efficiency benchmarking, test evaluation, calibration analyses, and paper-output generation.

---

## Reproducibility Boundaries

Several limitations should be considered when interpreting or reproducing the results:

1. Patient-independent separation cannot be independently verified from the released MIHIC metadata.
2. The study uses three training seeds; these characterize optimization variability but do not provide patient-level uncertainty estimates.
3. Pretrained models, randomly initialized models, and distilled students do not receive identical sources of prior information.
4. Runtime measurements are hardware-specific.
5. GMAC estimates describe model operations rather than end-to-end image-processing cost.
6. Stain-specific subsets contain different class compositions and should be interpreted descriptively.
7. Small numerical variation may occur across different CUDA, cuDNN, PyTorch, or hardware environments despite fixed seeds.

---

## Data Availability

The MIHIC dataset is publicly available through Zenodo:

https://doi.org/10.5281/zenodo.10065510

This repository contains the code, experimental configuration, architecture specifications, seed-level results, calibration outputs, efficiency measurements, figures, and reporting metadata associated with the study.

---

## Citation

The manuscript is currently under preparation/submission.

Until a final bibliographic record is available, please cite the MIHIC dataset when using the underlying images:

> Wang R, Qiu Y, Wang T, et al. MIHIC: a multiplex IHC histopathological image classification dataset for lung cancer immune microenvironment quantification. *Frontiers in Immunology*. 2024;15:1334348. https://doi.org/10.3389/fimmu.2024.1334348

A repository citation will be provided through `CITATION.cff`.

---

## License

Source code in this repository is released under the MIT License unless otherwise stated.

The MIHIC images are **not distributed by this repository** and remain subject to the terms specified by the original dataset creators and Zenodo record.

---

## Contact

**Dimitris Kalatzis**

School of Science and Technology  
Hellenic Open University  
Patras, Greece

Correspondence: kalatzis.dimitrios@ac.eap.gr
