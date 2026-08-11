# Reproducibility protocol

## Two verification levels

### 1. Archived-result verification

This is fast and does not require MIHIC or PyTorch:

```bash
python -m pip install -r requirements/reporting.txt
python scripts/verify_reported_results.py
python scripts/make_paper_figures.py --output-dir reproduced_figures
```

The numerical audit begins with the 39 model–seed rows in `results/tables/raw_seed_results.csv`. It independently recomputes numeric means and sample standard deviations and verifies the manuscript’s main comparisons. It also checks the frozen source and dataset-manifest hashes.

### 2. Full computational reproduction

This retrains all models and recreates checkpoints, histories, per-sample probabilities, metrics, per-class reports, confusion matrices, subgroup tables, the Excel workbook, and generated figures:

```bash
python run_experiment.py \
  --config configs/final.json \
  --data-root /path/to/MIHIC \
  --output-root outputs
```

The full FINAL registry is:

1. ConvNeXt-Tiny teacher (ImageNet-1K pretrained)
2. Proposed Nano + KD
3. Proposed Tiny + KD
4. Proposed Small + KD
5. Frozen Tiny without KD
6. Tiny + tested attention module
7. Single-scale Tiny + KD
8. Clean TinyCNN control
9. EfficientNet-B0, scratch
10. MobileNetV3-Small, scratch
11. MobileNetV4 Conv Small, ImageNet-1K pretrained
12. RepViT M0.9, ImageNet-1K/distilled pretrained weights
13. SHViT-S2, ImageNet-1K pretrained

Each model is trained with seeds 42, 123, and 2026. The checkpoint with the highest validation Macro-F1 is evaluated on the official test split.

## Freeze protection

- PILOT and FREEZE_CHECK instantiate only `train/` and `val/`; they do not instantiate `test/`.
- FINAL requires explicit Boolean values for the frozen attention and multiscale choices.
- The reported choice is `attention=false`, `multi_scale=true`.
- The full FINAL ablation registry is reported without retrospectively replacing the frozen model with the later-better single-scale result.

## Offline knowledge distillation

For each seed, the trained ConvNeXt-Tiny teacher is selected by validation Macro-F1. Teacher logits are then cached on canonical, normalized training images. Students receive independently augmented views indexed to those cached targets. The loss is:

```text
0.45 × weighted label-smoothed cross-entropy + 0.55 × T² KL, T = 3.0
```

Class weights are square-root inverse-frequency weights normalized to mean one.

## Output contract

Every model/seed directory created by `run_experiment.py` contains:

- `architecture.txt`
- `model_spec.json`
- `parameter_inventory.csv`
- `best_model.pth`
- `training_history.csv`
- validation/test metrics JSON
- validation/test per-class CSV
- validation/test confusion-matrix CSV
- validation/test prediction and probability CSV
- `result.json`

The run root additionally contains aggregate CSV tables, figures, `MIHIC_paper_tables.xlsx`, environment information, `pip freeze`, the methods snapshot, dataset manifest/checksum, and a checksum of the exact executed public runner.

## Why archived results and reruns are both provided

Training results are not expected to be bitwise identical on arbitrary GPUs, drivers, or future libraries. The archived seed-level exports establish exactly what was used in the paper. The full runner establishes how those exports were produced. Hardware benchmarks are especially environment-specific and should be interpreted as measurements rather than portable model properties.

## Provenance

`provenance/executed_cell_source.py` is the unchanged source from the reported experiment. `run_experiment.py` preserves the training/evaluation logic but adds a command-line interface, external JSON configuration, portable source capture, and removal of private absolute paths from exported manifests.

The archived source hash is stored in `provenance/executed_cell_source_sha256.txt`.

## Figure 5 note

The archived paper and supplementary PNGs are the original figures. `results/tables/figure_5_confusion_matrices.csv` records the two-decimal row-normalized values displayed in the manuscript and is sufficient to rebuild the displayed panel. A full run retains the underlying seed-specific count matrices and per-sample probabilities.
