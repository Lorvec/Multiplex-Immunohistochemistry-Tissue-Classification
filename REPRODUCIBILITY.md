# Reproducibility Notes

The public workflow is intentionally simple: download MIHIC, configure the dataset path, and run the complete pipeline.

## What is fixed

The paper settings are encoded in `reproduce_mihic.py` and `mihic_pipeline.py`, including architecture definitions, seeds, optimization settings, knowledge-distillation parameters, preprocessing, checkpoint selection, calibration, and reporting.

## Resume behavior

Each completed model/seed run writes its checkpoint and summary into the output directory. Re-running the workflow reuses completed runs, allowing interrupted experiments to continue.

## Pretrained weights

ConvNeXt-Tiny, RepViT, MobileNetV4, and SHViT use pretrained weights supplied by torchvision/timm. These may be downloaded automatically on first use.

## Numerical variation

Fixed random seeds and deterministic cuDNN settings reduce run-to-run variation, but bitwise identity across different GPUs, CUDA/cuDNN versions, PyTorch builds, and operating systems is not guaranteed.

## Hardware measurements

Latency, throughput, and peak GPU memory are measured using synthetic FP32 model-only inputs. They are intended for within-environment comparison and should not be interpreted as end-to-end slide-processing latency.

## Dataset limitation

The official MIHIC release is region-grouped, but patient identifiers sufficient to independently verify patient-disjoint splitting are unavailable. Consequently, the workflow does not claim patient-independent or clinical generalization.
