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

## Stain-specific Macro-F1 in Figure 5

For each stain, Macro-F1 is the unweighted mean of the class-specific F1 scores for the classes present in that stain's ground-truth test labels. The same class set is used for every model and seed within a stain. Figure 5 shows the mean and standard deviation across seeds 42, 123, and 2026 for each model and stain.

See the [report regeneration instructions in the README](../README.md#regenerate-reports-from-a-completed-run) to rebuild Figure 5 and its tables from the saved test predictions.

## Post-training INT8 quantization

The [INT8 notebook](../MIHIC_INT8_ThreeSeeds.ipynb) exports the validation-selected Small single + KD checkpoints for seeds 42, 123, and 2026 to FP32 ONNX (opset 17) and checks agreement with PyTorch on eight training patches. Static quantization uses ONNX Runtime 1.31.1, QDQ format, signed 8-bit weights and activations, per-channel weight quantization, and MinMax calibration. The same 1,000 training patches are used for each checkpoint, with at least one patch from each of the 72 observed stain and tissue class combinations.

Each FP32 and INT8 pair is evaluated on the same 65,437 official test patches. NLL and ECE use the unscaled model probabilities. The notebook reports the mean, sample standard deviation, and paired INT8 minus FP32 differences across the three seeds.

The CPU benchmark uses the ONNX Runtime CPU execution provider with eight intra-operator threads, one inter-operator thread, and identical synthetic FP32 inputs for both models. The reported experiment ran on an Intel Core Ultra 7 265K under WSL2. Batch-1 latency is the median of 40 timed calls after 10 warm-up calls. Batch-64 throughput is calculated from the median latency of 20 timed calls after five warm-up calls. The notebook saves operator profiling and the hardware and software configuration alongside the results. Follow the [README instructions](../README.md#5-reproduce-the-int8-experiment) to reproduce Table 6 and Figure 6.

## Dataset limitation

The official MIHIC release is region-grouped, but patient identifiers sufficient to independently verify patient-disjoint splitting are unavailable. Consequently, the workflow does not claim patient-independent or clinical generalization.
