# MIHIC paper experiment: compact model family + offline teacher distillation.
# Public command-line runner derived from the frozen executed source archived in
# provenance/executed_cell_source.py.

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path


# =============================================================================
# 0. FROZEN CONFIGURATION AND COMMAND-LINE INTERFACE
# =============================================================================

REPOSITORY_ROOT = Path(__file__).resolve().parent


def parse_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen MIHIC PILOT, FREEZE_CHECK, or FINAL experiment. "
            "The FINAL configuration trains 13 models with seeds 42, 123, and 2026."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "final.json",
        help="JSON configuration (default: configs/final.json).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing the official train/, val/, and test/ folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for checkpoints, predictions, tables, and figures.",
    )
    parser.add_argument(
        "--mode",
        choices=("PILOT", "FREEZE_CHECK", "FINAL"),
        default=None,
        help="Optional mode override. PILOT and FREEZE_CHECK never load test/.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Replace completed runs for the resolved configuration.",
    )
    return parser.parse_args()


def load_json_config(path: Path):
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    parent_name = config.pop("extends", None)
    if parent_name:
        parent, _ = load_json_config(path.parent / parent_name)
        parent.update(config)
        config = parent
    return config, path


CLI = parse_cli()
PAPER_CONFIG, CONFIG_PATH = load_json_config(CLI.config)

RUN_MODE = (CLI.mode or PAPER_CONFIG["run_mode"]).upper()
DATA_ROOT = Path(
    CLI.data_root
    or os.environ.get("MIHIC_DATA_ROOT")
    or PAPER_CONFIG["data_root"]
).expanduser()
OUTPUT_ROOT = Path(CLI.output_root or PAPER_CONFIG["output_root"]).expanduser()
EXPERIMENT_NAME = str(PAPER_CONFIG["experiment_name"])

PILOT_SEEDS = list(PAPER_CONFIG["seeds"]["PILOT"])
FREEZE_CHECK_SEEDS = list(PAPER_CONFIG["seeds"]["FREEZE_CHECK"])
FINAL_SEEDS = list(PAPER_CONFIG["seeds"]["FINAL"])

IMAGE_SIZE = int(PAPER_CONFIG["image_size"])
NUM_WORKERS = int(
    CLI.num_workers
    if CLI.num_workers is not None
    else PAPER_CONFIG["num_workers"]
)
TRAIN_BATCH_SIZE = int(PAPER_CONFIG["train_batch_size"])
# One shared batch size is deliberate: all compared models receive the same
# number of optimizer updates per epoch.
KD_BATCH_SIZE = TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = int(PAPER_CONFIG["eval_batch_size"])

PILOT_EPOCHS = int(PAPER_CONFIG["epochs"]["PILOT"])
FREEZE_CHECK_EPOCHS = int(PAPER_CONFIG["epochs"]["FREEZE_CHECK"])
FINAL_EPOCHS = int(PAPER_CONFIG["epochs"]["FINAL"])
WARMUP_EPOCHS = int(PAPER_CONFIG["warmup_epochs"])
EARLY_STOPPING_PATIENCE = int(PAPER_CONFIG["early_stopping_patience"])

FREEZE_CANDIDATE_USE_ATTENTION = bool(
    PAPER_CONFIG["freeze_candidate_use_attention"]
)
FREEZE_CANDIDATE_MULTI_SCALE = bool(
    PAPER_CONFIG["freeze_candidate_multi_scale"]
)
FINAL_USE_ATTENTION = PAPER_CONFIG["final_use_attention"]
FINAL_MULTI_SCALE = PAPER_CONFIG["final_multi_scale"]
MIN_COMPONENT_MACRO_F1_GAIN = float(
    PAPER_CONFIG["min_component_macro_f1_gain"]
)

LR_SCRATCH = float(PAPER_CONFIG["lr_scratch"])
LR_PRETRAINED = float(PAPER_CONFIG["lr_pretrained"])
WEIGHT_DECAY = float(PAPER_CONFIG["weight_decay"])
LABEL_SMOOTHING = float(PAPER_CONFIG["label_smoothing"])
CLASS_WEIGHT_POWER = float(PAPER_CONFIG["class_weight_power"])
KD_ALPHA = float(PAPER_CONFIG["kd_alpha"])
KD_TEMPERATURE = float(PAPER_CONFIG["kd_temperature"])

USE_AMP = bool(PAPER_CONFIG["use_amp"])
DETERMINISTIC_CUDNN = bool(PAPER_CONFIG["deterministic_cudnn"])
CPU_BENCHMARK_THREADS = int(PAPER_CONFIG["cpu_benchmark_threads"])
AUTO_INSTALL_OPTIONAL_PACKAGES = bool(
    PAPER_CONFIG["auto_install_optional_packages"]
)
RUN_TIMM_BASELINES_IN_FINAL = bool(
    PAPER_CONFIG["run_timm_baselines_in_final"]
)
RUN_LEGACY_CONTROLS_IN_FINAL = bool(
    PAPER_CONFIG["run_legacy_controls_in_final"]
)

GPU_WARMUP_STEPS = int(PAPER_CONFIG["gpu_warmup_steps"])
GPU_BENCHMARK_STEPS = int(PAPER_CONFIG["gpu_benchmark_steps"])
CPU_WARMUP_STEPS = int(PAPER_CONFIG["cpu_warmup_steps"])
CPU_BENCHMARK_STEPS = int(PAPER_CONFIG["cpu_benchmark_steps"])

SECTION_REGEX = str(PAPER_CONFIG["section_regex"])
MIN_SUBGROUP_SUPPORT = int(PAPER_CONFIG["min_subgroup_support"])
FORCE_RERUN = bool(CLI.force_rerun)


# =============================================================================
# 1. DEPENDENCIES
# =============================================================================

def ensure_package(import_name: str, pip_name: str | None = None, required: bool = True):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        if not AUTO_INSTALL_OPTIONAL_PACKAGES:
            if required:
                raise
            return None
        package = pip_name or import_name
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        importlib.invalidate_caches()
        return importlib.import_module(import_name)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import datasets, models, transforms
    from torchvision.transforms import functional as TF
except ImportError as exc:
    raise RuntimeError(
        "PyTorch/torchvision are required. Install a CUDA-compatible PyTorch build "
        "before running this cell; the cell intentionally does not replace your "
        "existing CUDA installation."
    ) from exc

np = ensure_package("numpy")
pd = ensure_package("pandas")
ensure_package("sklearn", "scikit-learn")
ensure_package("tqdm")
ensure_package("matplotlib")
ensure_package("seaborn")
ensure_package("openpyxl")

from IPython.display import display
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm

TIMM = None
if RUN_MODE.upper() == "FINAL" and RUN_TIMM_BASELINES_IN_FINAL:
    TIMM = ensure_package("timm", "timm>=1.0.15")
else:
    try:
        TIMM = importlib.import_module("timm")
    except ImportError:
        TIMM = None

try:
    FVCORE = ensure_package("fvcore", "fvcore", required=False)
except Exception:
    FVCORE = None


# =============================================================================
# 2. CONFIG VALIDATION, OUTPUTS, AND REPRODUCIBILITY
# =============================================================================

RUN_MODE = RUN_MODE.upper().strip()
VALID_RUN_MODES = {"PILOT", "FREEZE_CHECK", "FINAL"}
if RUN_MODE not in VALID_RUN_MODES:
    raise ValueError(
        "RUN_MODE must be 'PILOT', 'FREEZE_CHECK', or 'FINAL'."
    )
if KD_BATCH_SIZE != TRAIN_BATCH_SIZE:
    raise ValueError(
        "KD_BATCH_SIZE must equal TRAIN_BATCH_SIZE so all models receive the "
        "same number of optimizer updates per epoch."
    )
if RUN_MODE == "FINAL":
    if not isinstance(FINAL_USE_ATTENTION, bool) or not isinstance(FINAL_MULTI_SCALE, bool):
        raise ValueError(
            "FINAL architecture is not frozen. Review FREEZE_DECISION.md, then "
            "set FINAL_USE_ATTENTION and FINAL_MULTI_SCALE explicitly to True/False."
        )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = bool(USE_AMP and DEVICE.type == "cuda")
SEEDS_BY_MODE = {
    "PILOT": PILOT_SEEDS,
    "FREEZE_CHECK": FREEZE_CHECK_SEEDS,
    "FINAL": FINAL_SEEDS,
}
EPOCHS_BY_MODE = {
    "PILOT": PILOT_EPOCHS,
    "FREEZE_CHECK": FREEZE_CHECK_EPOCHS,
    "FINAL": FINAL_EPOCHS,
}
SEEDS = SEEDS_BY_MODE[RUN_MODE]
MAX_EPOCHS = EPOCHS_BY_MODE[RUN_MODE]
EVALUATE_TEST = RUN_MODE == "FINAL"

CONFIG = {
    "version": "1.2-public",
    "run_mode": RUN_MODE,
    # Avoid writing a private local path into exported reproducibility files.
    # The short token still prevents accidental resume across different roots.
    "data_root_token": hashlib.sha256(
        str(DATA_ROOT.resolve()).encode("utf-8")
    ).hexdigest()[:12],
    "config_file": CONFIG_PATH.name,
    "experiment_name": EXPERIMENT_NAME,
    "seeds": SEEDS,
    "image_size": IMAGE_SIZE,
    "num_workers": NUM_WORKERS,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "kd_batch_size": KD_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "warmup_epochs": WARMUP_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "lr_scratch": LR_SCRATCH,
    "lr_pretrained": LR_PRETRAINED,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "class_weight_power": CLASS_WEIGHT_POWER,
    "kd_alpha": KD_ALPHA,
    "kd_temperature": KD_TEMPERATURE,
    "amp": AMP_ENABLED,
    "deterministic_cudnn": DETERMINISTIC_CUDNN,
    "cpu_benchmark_threads": CPU_BENCHMARK_THREADS,
    "timm_baselines": bool(RUN_TIMM_BASELINES_IN_FINAL),
    "legacy_controls": bool(RUN_LEGACY_CONTROLS_IN_FINAL),
    "freeze_candidate_use_attention": FREEZE_CANDIDATE_USE_ATTENTION,
    "freeze_candidate_multi_scale": FREEZE_CANDIDATE_MULTI_SCALE,
    "final_use_attention": FINAL_USE_ATTENTION,
    "final_multi_scale": FINAL_MULTI_SCALE,
    "min_component_macro_f1_gain": MIN_COMPONENT_MACRO_F1_GAIN,
}
CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()[:12]

RUN_ROOT = OUTPUT_ROOT / f"{RUN_MODE.lower()}_{CONFIG_FINGERPRINT}"
RUNS_DIR = RUN_ROOT / "runs"
TABLES_DIR = RUN_ROOT / "tables"
FIGURES_DIR = RUN_ROOT / "figures"
MANIFESTS_DIR = RUN_ROOT / "manifests"
for directory in [RUN_ROOT, RUNS_DIR, TABLES_DIR, FIGURES_DIR, MANIFESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

(RUN_ROOT / "experiment_config.json").write_text(
    json.dumps(CONFIG, indent=2, sort_keys=True), encoding="utf-8"
)

print("=" * 88)
print("MIHIC MODERN LIGHTWEIGHT FAMILY EXPERIMENT")
print("=" * 88)
print(f"Mode: {RUN_MODE}")
print(f"Device: {DEVICE}")
print(f"AMP: {AMP_ENABLED}")
print(f"Seeds: {SEEDS}")
print(f"Output: {RUN_ROOT.resolve()}")
if RUN_MODE == "PILOT":
    print("PILOT SAFETY: the official test set will NOT be loaded or evaluated.")
elif RUN_MODE == "FREEZE_CHECK":
    print("FREEZE CHECK: validation-only; the official test set will NOT be loaded.")
    print(
        "Candidate family: "
        f"attention={FREEZE_CANDIDATE_USE_ATTENTION}, "
        f"multi_scale={FREEZE_CANDIDATE_MULTI_SCALE}"
    )
else:
    print("FINAL MODE: architectures must already be frozen; test evaluation is enabled.")


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(DETERMINISTIC_CUDNN)
    torch.backends.cudnn.benchmark = not bool(DETERMINISTIC_CUDNN)


try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# =============================================================================
# 3. DATA, AUGMENTATION, AND DATASET MANIFEST
# =============================================================================

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"
required_split_dirs = [TRAIN_DIR, VAL_DIR] + ([TEST_DIR] if EVALUATE_TEST else [])
for split_dir in required_split_dirs:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {split_dir.resolve()}")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomRightAngleRotation:
    def __call__(self, image):
        return TF.rotate(image, random.choice((0, 90, 180, 270)))


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    RandomRightAngleRotation(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.01)
    ], p=0.35),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index


train_dataset = IndexedImageFolder(TRAIN_DIR, transform=train_transform)
train_eval_dataset = IndexedImageFolder(TRAIN_DIR, transform=eval_transform)
val_dataset = IndexedImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = (
    IndexedImageFolder(TEST_DIR, transform=eval_transform)
    if EVALUATE_TEST else None
)

CLASS_NAMES = list(train_dataset.classes)
NUM_CLASSES = len(CLASS_NAMES)
if val_dataset.classes != CLASS_NAMES:
    raise ValueError("Class ordering differs between train and validation directories.")
if test_dataset is not None and test_dataset.classes != CLASS_NAMES:
    raise ValueError("Class ordering differs across train/validation/test directories.")

print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
test_status = f"{len(test_dataset):,}" if test_dataset is not None else "NOT LOADED"
print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {test_status}")

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


def infer_stain(path_text: str):
    for _, canonical, pattern in STAIN_MATCHERS:
        if pattern.search(path_text):
            return canonical
    return None


# Guard specifically against the former CD3/CD34/CD38 substring collision.
for _example_path, _expected_stain in {
    "section/CD3_patch_001.png": "CD3",
    "section/CD34_patch_001.png": "CD34",
    "section/CD38_patch_001.png": "CD38",
}.items():
    assert infer_stain(_example_path) == _expected_stain


def infer_section(path_text: str):
    match = re.search(SECTION_REGEX, Path(path_text).name)
    return match.group(1) if match else None


def build_manifest(split_name: str, dataset: IndexedImageFolder):
    rows = []
    root = Path(dataset.root)
    for index, (path, target) in enumerate(dataset.samples):
        path_obj = Path(path)
        relative = path_obj.relative_to(root).as_posix()
        rows.append({
            "split": split_name,
            "sample_index": index,
            "relative_path": relative,
            "class_index": int(target),
            "class": CLASS_NAMES[int(target)],
            "file_size_bytes": path_obj.stat().st_size,
            "stain": infer_stain(relative),
            "section_id_heuristic": infer_section(relative),
        })
    return pd.DataFrame(rows)


manifest_frames = [
    build_manifest("train", train_eval_dataset),
    build_manifest("val", val_dataset),
]
if test_dataset is not None:
    manifest_frames.append(build_manifest("test", test_dataset))
manifest_df = pd.concat(manifest_frames, ignore_index=True)
manifest_path = MANIFESTS_DIR / "dataset_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)

manifest_digest_source = manifest_df[
    ["split", "relative_path", "class", "file_size_bytes"]
].to_csv(index=False)
manifest_sha256 = hashlib.sha256(manifest_digest_source.encode("utf-8")).hexdigest()
(MANIFESTS_DIR / "dataset_manifest_sha256.txt").write_text(manifest_sha256, encoding="utf-8")

class_distribution_df = (
    manifest_df.groupby(["split", "class"], observed=True)
    .size().rename("count").reset_index()
)
class_distribution_df.to_csv(TABLES_DIR / "table_dataset_distribution.csv", index=False)

train_counts = np.bincount(
    np.asarray(train_dataset.targets, dtype=int), minlength=NUM_CLASSES
).astype(float)
if CLASS_WEIGHT_POWER > 0:
    class_weights_np = (train_counts.max() / np.maximum(train_counts, 1.0)) ** CLASS_WEIGHT_POWER
    class_weights_np = class_weights_np / class_weights_np.mean()
else:
    class_weights_np = np.ones(NUM_CLASSES, dtype=float)
CLASS_WEIGHTS = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(seed: int, batch_size: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            **common,
        )
    return train_loader, val_loader, test_loader


# =============================================================================
# 4. MODERN LIGHTWEIGHT MODEL FAMILY
# =============================================================================

def make_divisible(value: float, divisor: int = 8):
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        groups=1,
        activation=True,
    ):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        gate = self.pool(x)
        gate = F.silu(self.reduce(gate), inplace=True)
        gate = torch.sigmoid(self.expand(gate))
        return x * gate


class MultiScaleUIB(nn.Module):
    """Hardware-friendly inverted bottleneck with 3x3/5x5 depthwise mixing."""

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=3.0, multi_scale=True):
        super().__init__()
        hidden = make_divisible(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.multi_scale = bool(multi_scale)
        self.expand = ConvBNAct(in_channels, hidden, kernel_size=1)

        if self.multi_scale:
            first = hidden // 2
            second = hidden - first
            self.first_channels = first
            self.dw3 = ConvBNAct(first, first, kernel_size=3, stride=stride, groups=first)
            self.dw5 = ConvBNAct(second, second, kernel_size=5, stride=stride, groups=second)
        else:
            self.dw = ConvBNAct(hidden, hidden, kernel_size=3, stride=stride, groups=hidden)

        self.se = SqueezeExcite(hidden, reduction=4)
        self.project = ConvBNAct(hidden, out_channels, kernel_size=1, activation=False)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        if self.multi_scale:
            x1, x2 = torch.split(
                x, [self.first_channels, x.shape[1] - self.first_channels], dim=1
            )
            x = torch.cat([self.dw3(x1), self.dw5(x2)], dim=1)
        else:
            x = self.dw(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class GlobalLocalSingleHeadMixer(nn.Module):
    """Partial single-head attention plus a parallel local depthwise pathway."""

    def __init__(self, channels, attention_ratio=0.25):
        super().__init__()
        attention_channels = min(
            channels, make_divisible(max(8, channels * attention_ratio))
        )
        self.attention_channels = attention_channels
        self.scale = attention_channels ** -0.5
        self.local = ConvBNAct(
            channels, channels, kernel_size=3, groups=channels, activation=False
        )
        self.qkv = nn.Conv2d(attention_channels, attention_channels * 3, 1, bias=False)
        self.project = ConvBNAct(channels, channels, kernel_size=1, activation=False)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, x):
        batch, _, height, width = x.shape
        local = self.local(x)
        attention_input = x[:, : self.attention_channels]
        q, k, v = self.qkv(attention_input).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)                 # B, N, C
        k = k.flatten(2)                                # B, C, N
        v = v.flatten(2).transpose(1, 2)                 # B, N, C
        weights = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(
            batch, self.attention_channels, height, width
        )
        mixed = torch.cat([
            local[:, : self.attention_channels] + attended,
            local[:, self.attention_channels :],
        ], dim=1)
        return x + self.gamma * self.project(mixed)


MODERN_FAMILY_CONFIGS = {
    "nano": {
        # Approximately 0.33M parameters for seven MIHIC classes.
        "widths": (24, 40, 80, 160),
        "depths": (1, 2, 3, 1),
        "expand_ratio": 3.0,
        "attention_stages": (3,),
        "dropout": 0.15,
    },
    "tiny": {
        "widths": (24, 48, 96, 192),
        "depths": (1, 2, 3, 2),
        "expand_ratio": 3.0,
        "attention_stages": (2, 3),
        "dropout": 0.20,
    },
    "small": {
        "widths": (32, 64, 128, 256),
        "depths": (2, 3, 4, 2),
        "expand_ratio": 4.0,
        "attention_stages": (2, 3),
        "dropout": 0.25,
    },
}


class ModernMIHICNet(nn.Module):
    def __init__(
        self,
        num_classes,
        variant="tiny",
        use_attention=True,
        multi_scale=True,
    ):
        super().__init__()
        if variant not in MODERN_FAMILY_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}")
        cfg = MODERN_FAMILY_CONFIGS[variant]
        widths = cfg["widths"]
        depths = cfg["depths"]
        expand_ratio = cfg["expand_ratio"]
        attention_stages = set(cfg["attention_stages"] if use_attention else ())

        self.variant = variant
        self.use_attention = bool(use_attention)
        self.multi_scale = bool(multi_scale)
        self.stem = ConvBNAct(3, widths[0], kernel_size=3, stride=2)

        stages = []
        mixers = []
        in_channels = widths[0]
        for stage_index, (out_channels, depth) in enumerate(zip(widths, depths)):
            blocks = []
            for block_index in range(depth):
                blocks.append(MultiScaleUIB(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2 if block_index == 0 else 1,
                    expand_ratio=expand_ratio,
                    multi_scale=multi_scale,
                ))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
            if stage_index in attention_stages:
                mixers.append(GlobalLocalSingleHeadMixer(out_channels))
            else:
                mixers.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.mixers = nn.ModuleList(mixers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(widths[-1], num_classes),
        )

    def forward_features(self, x):
        x = self.stem(x)
        for stage, mixer in zip(self.stages, self.mixers):
            x = mixer(stage(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class TinyCNNClean(nn.Module):
    """Clean, single-forward implementation of the earlier 0.391M control."""

    def __init__(self, num_classes):
        super().__init__()
        layers = []
        in_channels = 3
        for index, out_channels in enumerate((32, 64, 128, 256)):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            if index < 3:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


# =============================================================================
# 5. MODEL REGISTRY: TEACHER, PROPOSED FAMILY, ABLATION, AND BASELINES
# =============================================================================

TEACHER_ID = "teacher_convnext_tiny"


def modern_family_spec(
    model_id,
    variant,
    use_attention,
    multi_scale,
    role,
    distilled,
    ablation_axis=None,
):
    return {
        "model_id": model_id,
        "kind": "modern_family",
        "variant": variant,
        "use_attention": bool(use_attention),
        "multi_scale": bool(multi_scale),
        "role": role,
        "pretraining": "teacher-distilled" if distilled else "none",
        "distilled": bool(distilled),
        "batch_size": TRAIN_BATCH_SIZE,
        "is_ablation": ablation_axis is not None,
        "ablation_axis": ablation_axis,
    }


def model_specs_for_mode():
    if RUN_MODE == "PILOT":
        family_attention = True
        family_multi_scale = True
    elif RUN_MODE == "FREEZE_CHECK":
        family_attention = FREEZE_CANDIDATE_USE_ATTENTION
        family_multi_scale = FREEZE_CANDIDATE_MULTI_SCALE
    else:
        family_attention = FINAL_USE_ATTENTION
        family_multi_scale = FINAL_MULTI_SCALE

    specs = [
        {
            "model_id": TEACHER_ID,
            "kind": "torchvision_convnext_tiny",
            "role": "teacher / strong reference",
            "pretraining": "ImageNet-1K",
            "distilled": False,
            "batch_size": TRAIN_BATCH_SIZE,
            "is_ablation": False,
            "ablation_axis": None,
        },
        modern_family_spec(
            "proposed_nano_kd", "nano", family_attention, family_multi_scale,
            "proposed Nano", distilled=True,
        ),
        modern_family_spec(
            "proposed_tiny_kd", "tiny", family_attention, family_multi_scale,
            "proposed Tiny", distilled=True,
        ),
        modern_family_spec(
            "proposed_small_kd", "small", family_attention, family_multi_scale,
            "proposed Small", distilled=True,
        ),
        modern_family_spec(
            "ablation_tiny_no_kd", "tiny", family_attention, family_multi_scale,
            "ablation: no distillation", distilled=False, ablation_axis="distillation",
        ),
    ]

    toggled_attention = not family_attention
    specs.append(modern_family_spec(
        (
            "ablation_tiny_with_attention"
            if toggled_attention else "ablation_tiny_no_attention"
        ),
        "tiny",
        toggled_attention,
        family_multi_scale,
        (
            "ablation: add attention"
            if toggled_attention else "ablation: remove attention"
        ),
        distilled=True,
        ablation_axis="attention",
    ))

    toggled_multi_scale = not family_multi_scale
    specs.append(modern_family_spec(
        (
            "ablation_tiny_multiscale"
            if toggled_multi_scale else "ablation_tiny_single_scale"
        ),
        "tiny",
        family_attention,
        toggled_multi_scale,
        (
            "ablation: add multi-scale depthwise"
            if toggled_multi_scale else "ablation: single-scale depthwise"
        ),
        distilled=True,
        ablation_axis="multi_scale",
    ))

    # FREEZE_CHECK intentionally excludes old baselines: the PILOT already
    # cleared the go/no-go gate, and this run exists only to freeze components.
    if RUN_MODE in {"PILOT", "FINAL"}:
        specs.extend([
            {
                "model_id": "control_tinycnn_gap_clean",
                "kind": "tinycnn_clean",
                "role": "clean earlier lightweight control",
                "pretraining": "none",
                "distilled": False,
                "batch_size": TRAIN_BATCH_SIZE,
                "is_ablation": False,
                "ablation_axis": None,
            },
            {
                "model_id": "baseline_efficientnet_b0_scratch",
                "kind": "torchvision_efficientnet_b0",
                "role": "scratch baseline",
                "pretraining": "none",
                "distilled": False,
                "batch_size": TRAIN_BATCH_SIZE,
                "is_ablation": False,
                "ablation_axis": None,
            },
        ])

    if RUN_MODE == "FINAL" and RUN_LEGACY_CONTROLS_IN_FINAL:
        specs.extend([
            {
                "model_id": "baseline_mobilenet_v3_small_scratch",
                "kind": "torchvision_mobilenet_v3_small",
                "role": "scratch baseline",
                "pretraining": "none",
                "distilled": False,
                "batch_size": TRAIN_BATCH_SIZE,
                "is_ablation": False,
                "ablation_axis": None,
            },
        ])

    if RUN_MODE == "FINAL" and RUN_TIMM_BASELINES_IN_FINAL:
        specs.extend([
            {
                "model_id": "baseline_mobilenetv4_conv_small_pretrained",
                "kind": "timm",
                "timm_name": "mobilenetv4_conv_small.e1200_r224_in1k",
                "role": "modern pretrained baseline",
                "pretraining": "ImageNet-1K",
                "distilled": False,
                "batch_size": TRAIN_BATCH_SIZE,
                "is_ablation": False,
                "ablation_axis": None,
            },
            {
                "model_id": "baseline_repvit_m0_9_pretrained",
                "kind": "timm",
                "timm_name": "repvit_m0_9.dist_450e_in1k",
                "role": "modern pretrained baseline",
                "pretraining": "ImageNet-1K/distilled",
                "distilled": False,
                "batch_size": TRAIN_BATCH_SIZE,
                "is_ablation": False,
                "ablation_axis": None,
            },
            {
                "model_id": "baseline_shvit_s2_pretrained",
                "kind": "timm",
                "timm_name": "shvit_s2.in1k",
                "role": "modern pretrained baseline",
                "pretraining": "ImageNet-1K",
                "distilled": False,
                "batch_size": TRAIN_BATCH_SIZE,
                "is_ablation": False,
                "ablation_axis": None,
            },
        ])
    return specs


MODEL_SPECS = model_specs_for_mode()
MODEL_SPEC_BY_ID = {spec["model_id"]: spec for spec in MODEL_SPECS}


def build_model(spec, pretrained_override=None):
    kind = spec["kind"]
    if kind == "modern_family":
        return ModernMIHICNet(
            num_classes=NUM_CLASSES,
            variant=spec["variant"],
            use_attention=spec.get("use_attention", True),
            multi_scale=spec.get("multi_scale", True),
        )
    if kind == "tinycnn_clean":
        return TinyCNNClean(NUM_CLASSES)
    if kind == "torchvision_convnext_tiny":
        use_pretrained = True if pretrained_override is None else pretrained_override
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if use_pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
        return model
    if kind == "torchvision_efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model
    if kind == "torchvision_mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
        return model
    if kind == "timm":
        if TIMM is None:
            raise RuntimeError("timm is required for the configured modern baselines.")
        use_pretrained = True if pretrained_override is None else pretrained_override
        return TIMM.create_model(
            spec["timm_name"], pretrained=use_pretrained, num_classes=NUM_CLASSES
        )
    raise ValueError(f"Unknown model kind: {kind}")


def unwrap_logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        tensors = [item for item in output if torch.is_tensor(item)]
        if not tensors:
            raise TypeError("Model returned a tuple/list without tensors.")
        return torch.stack(tensors, dim=0).mean(dim=0)
    if isinstance(output, dict):
        for key in ("logits", "out", "prediction"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"Unsupported model output type: {type(output)}")


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def save_architecture_artifacts(model, spec, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "architecture.txt").write_text(str(model), encoding="utf-8")
    (destination / "model_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = []
    for name, parameter in model.named_parameters():
        rows.append({
            "parameter": name,
            "shape": str(tuple(parameter.shape)),
            "numel": parameter.numel(),
            "trainable": parameter.requires_grad,
        })
    pd.DataFrame(rows).to_csv(destination / "parameter_inventory.csv", index=False)


# =============================================================================
# 6. METRICS, CALIBRATION, AND EVALUATION
# =============================================================================

def expected_calibration_error(probabilities, labels, bins=15):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean())
    return float(ece)


def metric_bundle(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_precision": precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )[0],
        "macro_recall": precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )[1],
        "mcc": matthews_corrcoef(labels, predictions),
        "cohen_kappa": cohen_kappa_score(labels, predictions),
        "nll": log_loss(labels, probabilities, labels=list(range(NUM_CLASSES))),
        "ece_15bin": expected_calibration_error(probabilities, labels, bins=15),
    }
    try:
        metrics["macro_roc_auc_ovr"] = roc_auc_score(
            labels, probabilities, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = np.nan
    return {key: float(value) for key, value in metrics.items()}


@torch.inference_mode()
def evaluate_loader(model, loader, dataset, split_name, output_dir, model_id, seed):
    model.eval()
    all_indices, all_labels, all_probabilities = [], [], []
    for images, labels, indices in tqdm(loader, desc=f"{split_name}: {model_id}", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            logits = unwrap_logits(model(images))
        probabilities = torch.softmax(logits.float(), dim=1)
        all_indices.append(indices.numpy())
        all_labels.append(labels.numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    indices = np.concatenate(all_indices)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    order = np.argsort(indices)
    indices, labels, probabilities = indices[order], labels[order], probabilities[order]
    predictions = probabilities.argmax(axis=1)

    metrics = metric_bundle(labels, probabilities)
    metrics.update({"split": split_name, "model_id": model_id, "seed": seed})

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for class_name in CLASS_NAMES:
        row = report[class_name]
        per_class_rows.append({
            "model_id": model_id,
            "seed": seed,
            "split": split_name,
            "class": class_name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1-score"],
            "support": row["support"],
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(output_dir / f"{split_name}_per_class.csv", index=False)

    cm = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(output_dir / f"{split_name}_confusion_matrix.csv")

    split_manifest = manifest_df[manifest_df["split"] == split_name].set_index("sample_index")
    prediction_df = pd.DataFrame({
        "sample_index": indices,
        "true_index": labels,
        "pred_index": predictions,
        "true_class": [CLASS_NAMES[index] for index in labels],
        "pred_class": [CLASS_NAMES[index] for index in predictions],
    })
    for class_index, class_name in enumerate(CLASS_NAMES):
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", class_name).strip("_").lower()
        prediction_df[f"prob_{safe_name}"] = probabilities[:, class_index]
    prediction_df = prediction_df.join(
        split_manifest[["relative_path", "stain", "section_id_heuristic"]],
        on="sample_index",
    )
    prediction_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
    (output_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics, per_class_df, cm_df


# =============================================================================
# 7. TRAINING WITH CORRECT SINGLE FORWARD AND OPTIONAL DISTILLATION
# =============================================================================

def make_criterion():
    return nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS,
        label_smoothing=LABEL_SMOOTHING,
    )


def distillation_loss(student_logits, teacher_logits, labels, hard_criterion):
    hard = hard_criterion(student_logits, labels)
    temperature = KD_TEMPERATURE
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return (1.0 - KD_ALPHA) * hard + KD_ALPHA * soft, hard.detach(), soft.detach()


def epoch_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
    teacher_logits_lookup=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    labels_all, predictions_all = [], []

    for images, labels, indices in tqdm(loader, leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=AMP_ENABLED,
            ):
                # Exactly ONE student forward pass. Do not duplicate this line.
                student_logits = unwrap_logits(model(images))
                if teacher_logits_lookup is not None:
                    teacher_logits = torch.as_tensor(
                        np.asarray(teacher_logits_lookup[indices.numpy()], dtype=np.float32),
                        device=DEVICE,
                    )
                    loss, hard_loss, soft_loss = distillation_loss(
                        student_logits, teacher_logits, labels, criterion
                    )
                else:
                    loss = criterion(student_logits, labels)
                    hard_loss = loss.detach()
                    soft_loss = torch.zeros((), device=loss.device)

            if training:
                if scaler is not None and AMP_ENABLED:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_hard += float(hard_loss) * batch_size
        total_soft += float(soft_loss) * batch_size
        predictions_all.extend(student_logits.detach().argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    metrics = epoch_metrics(np.asarray(labels_all), np.asarray(predictions_all))
    metrics.update({
        "loss": total_loss / len(loader.dataset),
        "hard_loss": total_hard / len(loader.dataset),
        "soft_loss": total_soft / len(loader.dataset),
    })
    return metrics


def lr_multiplier(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / max(1, WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS - 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def run_directory(model_id, seed):
    return RUNS_DIR / model_id / f"seed_{seed}"


def checkpoint_path(model_id, seed):
    return run_directory(model_id, seed) / "best_model.pth"


def load_teacher_for_seed(seed):
    teacher_spec = MODEL_SPEC_BY_ID[TEACHER_ID]
    path = checkpoint_path(TEACHER_ID, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint missing for seed {seed}: {path}. "
            "The registry must run the teacher before distilled students."
        )
    teacher = build_model(teacher_spec, pretrained_override=False)
    payload = torch.load(path, map_location="cpu")
    teacher.load_state_dict(payload["model_state"])
    teacher.to(DEVICE).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def teacher_logits_path(seed):
    return run_directory(TEACHER_ID, seed) / "train_teacher_logits.npy"


@torch.inference_mode()
def load_or_create_teacher_logits(seed):
    """Cache teacher targets on canonical (non-augmented) training images.

    Students may still receive augmented views. Offline targets make distillation
    much faster and explicitly encourage invariance to the mild augmentations.
    """
    path = teacher_logits_path(seed)
    if path.exists():
        return np.load(path, mmap_mode="r")

    print(f"Caching canonical teacher logits for seed {seed} ...")
    teacher = load_teacher_for_seed(seed)
    common = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        worker_init_fn=seed_worker,
        persistent_workers=(NUM_WORKERS > 0),
    )
    loader = DataLoader(
        train_eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common,
    )
    logits_cache = np.empty((len(train_eval_dataset), NUM_CLASSES), dtype=np.float16)
    for images, _, indices in tqdm(loader, desc="Teacher-logit cache", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            logits = unwrap_logits(teacher(images))
        logits_cache[indices.numpy()] = logits.float().cpu().numpy().astype(np.float16)
    np.save(path, logits_cache)
    del teacher
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(path, mmap_mode="r")


# =============================================================================
# 8. COMPLEXITY AND HARDWARE BENCHMARKING
# =============================================================================

def percentile_latency_ms(samples):
    return {
        "median_ms": float(np.median(samples)),
        "p25_ms": float(np.percentile(samples, 25)),
        "p75_ms": float(np.percentile(samples, 75)),
    }


@torch.inference_mode()
def benchmark_device(model, device, batch_size, warmup_steps, benchmark_steps):
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    for _ in range(warmup_steps):
        unwrap_logits(model(x))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(benchmark_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        unwrap_logits(model(x))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    stats = percentile_latency_ms(times)
    stats["images_per_second"] = batch_size * 1000.0 / stats["median_ms"]
    stats["peak_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else np.nan
    )
    return stats


def estimate_flops(model):
    if FVCORE is None:
        return np.nan
    try:
        from fvcore.nn import FlopCountAnalysis
        model_copy = copy.deepcopy(model).cpu().eval()
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(FlopCountAnalysis(model_copy, dummy).total())
    except Exception as exc:
        print(f"FLOP analysis skipped: {type(exc).__name__}: {exc}")
        return np.nan


def benchmark_model(model):
    original_threads = torch.get_num_threads()
    result = {}
    if DEVICE.type == "cuda":
        gpu_model = copy.deepcopy(model).to(DEVICE)
        batch1 = benchmark_device(
            gpu_model, DEVICE, 1, GPU_WARMUP_STEPS, GPU_BENCHMARK_STEPS
        )
        batch64 = benchmark_device(
            gpu_model, DEVICE, 64, GPU_WARMUP_STEPS, GPU_BENCHMARK_STEPS
        )
        result.update({
            "gpu_batch1_latency_median_ms": batch1["median_ms"],
            "gpu_batch1_latency_iqr_low_ms": batch1["p25_ms"],
            "gpu_batch1_latency_iqr_high_ms": batch1["p75_ms"],
            "gpu_batch1_images_per_second": batch1["images_per_second"],
            "gpu_batch64_images_per_second": batch64["images_per_second"],
            "gpu_peak_memory_mb": max(batch1["peak_memory_mb"], batch64["peak_memory_mb"]),
        })
        del gpu_model
        torch.cuda.empty_cache()
    else:
        result.update({
            "gpu_batch1_latency_median_ms": np.nan,
            "gpu_batch1_latency_iqr_low_ms": np.nan,
            "gpu_batch1_latency_iqr_high_ms": np.nan,
            "gpu_batch1_images_per_second": np.nan,
            "gpu_batch64_images_per_second": np.nan,
            "gpu_peak_memory_mb": np.nan,
        })

    torch.set_num_threads(CPU_BENCHMARK_THREADS)
    cpu_model = copy.deepcopy(model).cpu()
    cpu_stats = benchmark_device(
        cpu_model,
        torch.device("cpu"),
        1,
        CPU_WARMUP_STEPS,
        CPU_BENCHMARK_STEPS,
    )
    result.update({
        "cpu_batch1_latency_median_ms": cpu_stats["median_ms"],
        "cpu_batch1_latency_iqr_low_ms": cpu_stats["p25_ms"],
        "cpu_batch1_latency_iqr_high_ms": cpu_stats["p75_ms"],
        "cpu_batch1_images_per_second": cpu_stats["images_per_second"],
    })
    torch.set_num_threads(original_threads)
    return result


# =============================================================================
# 9. ONE COMPLETE MODEL x SEED RUN WITH RESUME SUPPORT
# =============================================================================

def train_model_seed(spec, seed):
    model_id = spec["model_id"]
    destination = run_directory(model_id, seed)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"

    if result_path.exists() and checkpoint_path(model_id, seed).exists() and not FORCE_RERUN:
        print(f"SKIP completed: {model_id} | seed {seed}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 88)
    print(f"TRAINING {model_id} | seed {seed}")
    print("=" * 88)
    set_seed(seed)

    model = build_model(spec).to(DEVICE)
    parameter_count = count_parameters(model)
    save_architecture_artifacts(model, spec, destination)
    print(f"Trainable parameters: {parameter_count / 1e6:.4f} M")

    teacher_logits_lookup = (
        load_or_create_teacher_logits(seed) if spec.get("distilled", False) else None
    )
    batch_size = int(spec.get("batch_size", TRAIN_BATCH_SIZE))
    train_loader, val_loader, test_loader = make_loaders(seed, batch_size)

    criterion = make_criterion()
    is_pretrained = spec.get("pretraining") not in {"none", "teacher-distilled"}
    learning_rate = LR_PRETRAINED if is_pretrained else LR_SCRATCH
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    best_val_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    training_start = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            teacher_logits_lookup=teacher_logits_lookup,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            scaler=None,
            teacher_logits_lookup=None,
        )
        scheduler.step()

        history_row = {
            "model_id": model_id,
            "seed": seed,
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_min": (time.perf_counter() - epoch_start) / 60.0,
        }
        history_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        history_row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(history_row)
        pd.DataFrame(history).to_csv(destination / "training_history.csv", index=False)

        print(
            f"Epoch {epoch + 1:02d}/{MAX_EPOCHS} | "
            f"train F1={train_metrics['macro_f1']:.4f} | "
            f"val F1={val_metrics['macro_f1']:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": copy.deepcopy(model.state_dict()),
                    "model_id": model_id,
                    "seed": seed,
                    "class_names": CLASS_NAMES,
                    "model_spec": spec,
                    "config_fingerprint": CONFIG_FINGERPRINT,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                checkpoint_path(model_id, seed),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    training_time_min = (time.perf_counter() - training_start) / 60.0
    payload = torch.load(checkpoint_path(model_id, seed), map_location="cpu")
    model.load_state_dict(payload["model_state"])
    model.to(DEVICE).eval()

    val_metrics, _, _ = evaluate_loader(
        model, val_loader, val_dataset, "val", destination, model_id, seed
    )
    test_metrics = {}
    if EVALUATE_TEST:
        if test_loader is None or test_dataset is None:
            raise RuntimeError("FINAL mode requires a loaded official test set.")
        test_metrics, _, _ = evaluate_loader(
            model, test_loader, test_dataset, "test", destination, model_id, seed
        )

    flops = estimate_flops(model)
    hardware_metrics = benchmark_model(model)
    checkpoint_size_mb = checkpoint_path(model_id, seed).stat().st_size / (1024 ** 2)

    result = {
        "model_id": model_id,
        "role": spec["role"],
        "pretraining": spec["pretraining"],
        "distilled": bool(spec.get("distilled", False)),
        "seed": seed,
        "params": parameter_count,
        "params_m": parameter_count / 1e6,
        "flops": flops,
        "gmacs": flops / 1e9 if np.isfinite(flops) else np.nan,
        "checkpoint_size_mb": checkpoint_size_mb,
        "train_batch_size": batch_size,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "stopped_early": len(history) < MAX_EPOCHS,
        "training_time_min": training_time_min,
        "use_attention": spec.get("use_attention"),
        "multi_scale": spec.get("multi_scale"),
        "ablation_axis": spec.get("ablation_axis"),
        "config_fingerprint": CONFIG_FINGERPRINT,
        "dataset_manifest_sha256": manifest_sha256,
    }
    result.update({f"val_{key}": value for key, value in val_metrics.items()
                   if key not in {"split", "model_id", "seed"}})
    result.update({f"test_{key}": value for key, value in test_metrics.items()
                   if key not in {"split", "model_id", "seed"}})
    result.update(hardware_metrics)

    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    if teacher_logits_lookup is not None:
        del teacher_logits_lookup
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


# =============================================================================
# 10. EXECUTE THE COMPLETE REGISTRY
# =============================================================================

all_results = []
for spec in MODEL_SPECS:
    for seed in SEEDS:
        try:
            all_results.append(train_model_seed(spec, seed))
        except Exception as exc:
            error_dir = run_directory(spec["model_id"], seed)
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "ERROR.txt").write_text(
                f"{type(exc).__name__}: {exc}", encoding="utf-8"
            )
            raise


# =============================================================================
# 11. PAPER-READY TABLES, SUBGROUP ANALYSIS, FIGURES, AND WORKBOOK
# =============================================================================

raw_results_df = pd.DataFrame(all_results)
raw_results_df.to_csv(TABLES_DIR / "raw_seed_results.csv", index=False)

PRIMARY_SPLIT = "test" if EVALUATE_TEST else "val"
PRIMARY_METRIC = f"{PRIMARY_SPLIT}_macro_f1"

numeric_columns = [
    column for column in raw_results_df.select_dtypes(include=[np.number]).columns
    if column != "seed"
]
summary_mean = raw_results_df.groupby("model_id", observed=True)[numeric_columns].mean()
summary_std = raw_results_df.groupby("model_id", observed=True)[numeric_columns].std()
summary_df = summary_mean.copy()
for column in numeric_columns:
    summary_df[f"{column}_std"] = summary_std[column]
summary_df = summary_df.reset_index()

metadata_df = raw_results_df[["model_id", "role", "pretraining", "distilled"]].drop_duplicates("model_id")
summary_df = metadata_df.merge(summary_df, on="model_id", how="right")
summary_df = summary_df.sort_values(PRIMARY_METRIC, ascending=False)
summary_df.to_csv(TABLES_DIR / "summary_mean_std_numeric.csv", index=False)


def mean_std_text(row, column, digits=4):
    mean = row.get(column, np.nan)
    std = row.get(f"{column}_std", np.nan)
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


main_rows = []
for _, row in summary_df.iterrows():
    if str(row["model_id"]).startswith("ablation_"):
        continue
    main_rows.append({
        "Model": row["model_id"],
        "Role": row["role"],
        "Pretraining": row["pretraining"],
        "Params (M)": f"{row['params_m']:.3f}",
        "GMACs": f"{row['gmacs']:.4f}" if np.isfinite(row.get("gmacs", np.nan)) else "NA",
        "Accuracy": mean_std_text(row, f"{PRIMARY_SPLIT}_accuracy"),
        "Balanced Acc.": mean_std_text(row, f"{PRIMARY_SPLIT}_balanced_accuracy"),
        "Macro-F1": mean_std_text(row, f"{PRIMARY_SPLIT}_macro_f1"),
        "Weighted-F1": mean_std_text(row, f"{PRIMARY_SPLIT}_weighted_f1"),
        "Macro-AUROC": mean_std_text(row, f"{PRIMARY_SPLIT}_macro_roc_auc_ovr"),
        "ECE": mean_std_text(row, f"{PRIMARY_SPLIT}_ece_15bin"),
    })
table_main = pd.DataFrame(main_rows)
table_main.to_csv(TABLES_DIR / "table_1_main_performance.csv", index=False)

efficiency_columns = [
    "model_id", "role", "pretraining", "params_m", "gmacs", "checkpoint_size_mb",
    "gpu_batch1_latency_median_ms", "gpu_batch64_images_per_second",
    "gpu_peak_memory_mb", "cpu_batch1_latency_median_ms", "training_time_min",
]
table_efficiency = summary_df[
    ~summary_df["model_id"].astype(str).str.startswith("ablation_")
][[column for column in efficiency_columns if column in summary_df.columns]].copy()
table_efficiency.to_csv(TABLES_DIR / "table_3_efficiency.csv", index=False)

per_class_frames = []
for spec in MODEL_SPECS:
    for seed in SEEDS:
        path = run_directory(spec["model_id"], seed) / f"{PRIMARY_SPLIT}_per_class.csv"
        if path.exists():
            per_class_frames.append(pd.read_csv(path))
per_class_raw = pd.concat(per_class_frames, ignore_index=True)
per_class_raw.to_csv(TABLES_DIR / "per_class_raw_seed_results.csv", index=False)

per_class_summary = (
    per_class_raw.groupby(["model_id", "class"], observed=True)
    .agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        support_mean=("support", "mean"),
    )
    .reset_index()
)
per_class_summary.to_csv(TABLES_DIR / "table_2_per_class_full.csv", index=False)

per_class_pivot = per_class_summary.pivot(index="model_id", columns="class", values="f1_mean")
per_class_pivot = per_class_pivot.reindex(columns=CLASS_NAMES)
per_class_pivot.to_csv(TABLES_DIR / "table_2b_per_class_f1_matrix.csv")

ablation_ids = ["proposed_tiny_kd"] + [
    spec["model_id"] for spec in MODEL_SPECS if spec.get("is_ablation", False)
]
table_ablation = summary_df[summary_df["model_id"].isin(ablation_ids)].copy()
table_ablation.to_csv(TABLES_DIR / "table_4_ablation.csv", index=False)

freeze_analysis = None
table_freeze_components = pd.DataFrame()
if RUN_MODE == "FREEZE_CHECK":
    summary_by_id = summary_df.set_index("model_id")
    base_id = "proposed_tiny_kd"
    no_kd_id = next(
        spec["model_id"] for spec in MODEL_SPECS
        if spec.get("ablation_axis") == "distillation"
    )
    attention_toggle_id = next(
        spec["model_id"] for spec in MODEL_SPECS
        if spec.get("ablation_axis") == "attention"
    )
    scale_toggle_id = next(
        spec["model_id"] for spec in MODEL_SPECS
        if spec.get("ablation_axis") == "multi_scale"
    )

    base_f1 = float(summary_by_id.loc[base_id, PRIMARY_METRIC])
    no_kd_f1 = float(summary_by_id.loc[no_kd_id, PRIMARY_METRIC])
    toggle_attention_f1 = float(summary_by_id.loc[attention_toggle_id, PRIMARY_METRIC])
    toggle_scale_f1 = float(summary_by_id.loc[scale_toggle_id, PRIMARY_METRIC])

    if FREEZE_CANDIDATE_USE_ATTENTION:
        with_attention_f1, without_attention_f1 = base_f1, toggle_attention_f1
        with_attention_id, without_attention_id = base_id, attention_toggle_id
    else:
        with_attention_f1, without_attention_f1 = toggle_attention_f1, base_f1
        with_attention_id, without_attention_id = attention_toggle_id, base_id

    if FREEZE_CANDIDATE_MULTI_SCALE:
        multiscale_f1, single_scale_f1 = base_f1, toggle_scale_f1
        multiscale_id, single_scale_id = base_id, scale_toggle_id
    else:
        multiscale_f1, single_scale_f1 = toggle_scale_f1, base_f1
        multiscale_id, single_scale_id = scale_toggle_id, base_id

    attention_gain = with_attention_f1 - without_attention_f1
    multiscale_gain = multiscale_f1 - single_scale_f1
    distillation_gain = base_f1 - no_kd_f1
    recommend_attention = attention_gain >= MIN_COMPONENT_MACRO_F1_GAIN
    recommend_multi_scale = multiscale_gain >= MIN_COMPONENT_MACRO_F1_GAIN
    distillation_supported = distillation_gain >= MIN_COMPONENT_MACRO_F1_GAIN

    boundary_models = []
    for model_id, row in summary_by_id.iterrows():
        if (
            int(round(row["epochs_ran"])) >= MAX_EPOCHS
            and int(round(row["best_epoch"])) >= MAX_EPOCHS - 1
        ):
            boundary_models.append(model_id)

    observed_batch_sizes = sorted(
        int(value) for value in raw_results_df["train_batch_size"].dropna().unique()
    )
    equal_batch_sizes = observed_batch_sizes == [TRAIN_BATCH_SIZE]
    ready_for_final = (
        equal_batch_sizes
        and not boundary_models
        and distillation_supported
    )

    table_freeze_components = pd.DataFrame([
        {
            "component": "attention",
            "enabled_model": with_attention_id,
            "disabled_model": without_attention_id,
            "enabled_val_macro_f1": with_attention_f1,
            "disabled_val_macro_f1": without_attention_f1,
            "gain_when_enabled": attention_gain,
            "minimum_required_gain": MIN_COMPONENT_MACRO_F1_GAIN,
            "recommended_enabled": recommend_attention,
        },
        {
            "component": "multi_scale",
            "enabled_model": multiscale_id,
            "disabled_model": single_scale_id,
            "enabled_val_macro_f1": multiscale_f1,
            "disabled_val_macro_f1": single_scale_f1,
            "gain_when_enabled": multiscale_gain,
            "minimum_required_gain": MIN_COMPONENT_MACRO_F1_GAIN,
            "recommended_enabled": recommend_multi_scale,
        },
        {
            "component": "distillation",
            "enabled_model": base_id,
            "disabled_model": no_kd_id,
            "enabled_val_macro_f1": base_f1,
            "disabled_val_macro_f1": no_kd_f1,
            "gain_when_enabled": distillation_gain,
            "minimum_required_gain": MIN_COMPONENT_MACRO_F1_GAIN,
            "recommended_enabled": distillation_supported,
        },
    ])
    table_freeze_components.to_csv(
        TABLES_DIR / "table_freeze_component_decision.csv", index=False
    )
    freeze_analysis = {
        "primary_metric": PRIMARY_METRIC,
        "minimum_component_macro_f1_gain": MIN_COMPONENT_MACRO_F1_GAIN,
        "recommend_final_use_attention": bool(recommend_attention),
        "recommend_final_multi_scale": bool(recommend_multi_scale),
        "distillation_supported": bool(distillation_supported),
        "attention_gain": attention_gain,
        "multi_scale_gain": multiscale_gain,
        "distillation_gain": distillation_gain,
        "observed_batch_sizes": observed_batch_sizes,
        "equal_batch_sizes": bool(equal_batch_sizes),
        "boundary_models": boundary_models,
        "ready_for_final_review": bool(ready_for_final),
        "official_test_loaded": bool(test_dataset is not None),
    }

architecture_rows = []
for spec in MODEL_SPECS:
    matching = summary_df[summary_df["model_id"] == spec["model_id"]]
    if matching.empty:
        continue
    row = matching.iloc[0]
    architecture_rows.append({
        "model_id": spec["model_id"],
        "kind": spec["kind"],
        "variant": spec.get("variant"),
        "attention": spec.get("use_attention"),
        "multi_scale": spec.get("multi_scale"),
        "distilled": spec.get("distilled", False),
        "ablation_axis": spec.get("ablation_axis"),
        "pretraining": spec["pretraining"],
        "params_m": row["params_m"],
        "gmacs": row.get("gmacs", np.nan),
    })
table_architecture = pd.DataFrame(architecture_rows)
table_architecture.to_csv(TABLES_DIR / "table_5_architecture_inventory.csv", index=False)


def subgroup_table():
    rows = []
    subgroup_columns = ["stain", "section_id_heuristic"]
    for spec in MODEL_SPECS:
        model_id = spec["model_id"]
        seed_prediction_frames = []
        for seed in SEEDS:
            path = run_directory(model_id, seed) / f"{PRIMARY_SPLIT}_predictions.csv"
            if path.exists():
                frame = pd.read_csv(path)
                frame["seed"] = seed
                seed_prediction_frames.append(frame)
        if not seed_prediction_frames:
            continue
        predictions = pd.concat(seed_prediction_frames, ignore_index=True)
        probability_columns = [column for column in predictions if column.startswith("prob_")]
        for subgroup_column in subgroup_columns:
            if subgroup_column not in predictions or predictions[subgroup_column].notna().mean() < 0.80:
                continue
            for (seed, subgroup), group in predictions.dropna(subset=[subgroup_column]).groupby(
                ["seed", subgroup_column], observed=True
            ):
                if len(group) < MIN_SUBGROUP_SUPPORT or group["true_index"].nunique() < 2:
                    continue
                metrics = metric_bundle(
                    group["true_index"].to_numpy(dtype=int),
                    group[probability_columns].to_numpy(dtype=float),
                )
                rows.append({
                    "model_id": model_id,
                    "seed": seed,
                    "subgroup_type": subgroup_column,
                    "subgroup": subgroup,
                    "support": len(group),
                    **metrics,
                })
    return pd.DataFrame(rows)


table_subgroups = subgroup_table()
if not table_subgroups.empty:
    table_subgroups.to_csv(TABLES_DIR / "table_6_subgroup_performance.csv", index=False)

# Pareto figure.
plt.figure(figsize=(9, 6))
for _, row in summary_df.iterrows():
    proposed = str(row["model_id"]).startswith("proposed_")
    plt.scatter(
        row["params_m"], row[PRIMARY_METRIC],
        s=110 if proposed else 65,
        marker="*" if proposed else "o",
        label=row["model_id"],
    )
    plt.annotate(row["model_id"], (row["params_m"], row[PRIMARY_METRIC]),
                 xytext=(4, 4), textcoords="offset points", fontsize=7)
plt.xscale("log")
plt.xlabel("Trainable parameters (millions, log scale)")
plt.ylabel(f"{PRIMARY_SPLIT.capitalize()} Macro-F1")
plt.title("MIHIC performance-parameter Pareto view")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_pareto_parameters_macro_f1.png", dpi=300)
plt.close()

# Per-class F1 heatmap.
plt.figure(figsize=(10, max(5, 0.42 * len(per_class_pivot))))
sns.heatmap(per_class_pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
plt.title(f"{PRIMARY_SPLIT.capitalize()} per-class F1")
plt.xlabel("Class")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_per_class_f1_heatmap.png", dpi=300)
plt.close()

# Class distribution figure.
plt.figure(figsize=(10, 5))
sns.barplot(data=class_distribution_df, x="class", y="count", hue="split")
plt.xticks(rotation=35, ha="right")
plt.title("MIHIC class distribution")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure_class_distribution.png", dpi=300)
plt.close()

# Aggregate confusion matrices for every model.
for spec in MODEL_SPECS:
    matrices = []
    for seed in SEEDS:
        path = run_directory(spec["model_id"], seed) / f"{PRIMARY_SPLIT}_confusion_matrix.csv"
        if path.exists():
            matrices.append(pd.read_csv(path, index_col=0).to_numpy())
    if not matrices:
        continue
    matrix = np.sum(matrices, axis=0).astype(float)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0,
        vmax=1,
    )
    plt.title(f"{spec['model_id']} — normalized confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_{spec['model_id']}.png", dpi=300)
    plt.close()

# One workbook containing every paper-facing table.
workbook_path = RUN_ROOT / "MIHIC_paper_tables.xlsx"
with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
    table_main.to_excel(writer, sheet_name="Main performance", index=False)
    summary_df.to_excel(writer, sheet_name="Numeric summary", index=False)
    per_class_summary.to_excel(writer, sheet_name="Per-class full", index=False)
    per_class_pivot.to_excel(writer, sheet_name="Per-class F1 matrix")
    table_efficiency.to_excel(writer, sheet_name="Efficiency", index=False)
    table_ablation.to_excel(writer, sheet_name="Ablation", index=False)
    table_architecture.to_excel(writer, sheet_name="Architectures", index=False)
    class_distribution_df.to_excel(writer, sheet_name="Dataset", index=False)
    raw_results_df.to_excel(writer, sheet_name="Raw seeds", index=False)
    if not table_freeze_components.empty:
        table_freeze_components.to_excel(
            writer, sheet_name="Freeze decision", index=False
        )
    if not table_subgroups.empty:
        table_subgroups.to_excel(writer, sheet_name="Subgroups", index=False)


# =============================================================================
# 12. METHODS SNAPSHOT, ENVIRONMENT, REFERENCES, AND EXACT CELL SOURCE
# =============================================================================

environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": getattr(TIMM, "__version__", None),
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cpu_count": os.cpu_count(),
}
(MANIFESTS_DIR / "environment.json").write_text(
    json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
)
try:
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    (MANIFESTS_DIR / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
except Exception:
    pass

methods_snapshot = {
    "dataset": "MIHIC official directory splits",
    "loaded_splits": manifest_df["split"].drop_duplicates().tolist(),
    "official_test_loaded": bool(test_dataset is not None),
    "dataset_manifest_sha256": manifest_sha256,
    "class_names_in_order": CLASS_NAMES,
    "input_size": [3, IMAGE_SIZE, IMAGE_SIZE],
    "normalization_mean": IMAGENET_MEAN,
    "normalization_std": IMAGENET_STD,
    "training_augmentation": [
        "horizontal flip p=0.5",
        "vertical flip p=0.5",
        "random 0/90/180/270 rotation",
        "mild color jitter p=0.35",
    ],
    "optimizer": "AdamW",
    "scheduler": "epoch-wise warmup + cosine decay",
    "shared_training_batch_size": TRAIN_BATCH_SIZE,
    "selection_metric": "validation macro-F1",
    "test_selection_rule": "test never used for checkpoint or architecture selection",
    "family_configuration": {
        "attention": MODEL_SPEC_BY_ID["proposed_tiny_kd"].get("use_attention"),
        "multi_scale": MODEL_SPEC_BY_ID["proposed_tiny_kd"].get("multi_scale"),
    },
    "distillation": {
        "teacher": TEACHER_ID,
        "alpha": KD_ALPHA,
        "temperature": KD_TEMPERATURE,
        "loss": "weighted label-smoothed CE + temperature-scaled KL",
        "teacher_targets": (
            "offline logits cached on canonical normalized training images; "
            "students receive independently augmented views of the same indexed samples"
        ),
    },
    "seeds": SEEDS,
    "configuration_fingerprint": CONFIG_FINGERPRINT,
}
(MANIFESTS_DIR / "paper_methods_snapshot.json").write_text(
    json.dumps(methods_snapshot, indent=2, sort_keys=True), encoding="utf-8"
)

bibtex = r"""
@article{wang2024mihic,
  title={MIHIC: a multiplex IHC histopathological image classification dataset for lung cancer immune microenvironment quantification},
  author={Wang, Ranran and Qiu, Yusong and Wang, Tong and others},
  journal={Frontiers in Immunology},
  volume={15},
  year={2024},
  doi={10.3389/fimmu.2024.1334348}
}
@article{qin2024mobilenetv4,
  title={MobileNetV4 -- Universal Models for the Mobile Ecosystem},
  author={Qin, Danfeng and Leichner, Chas and Delakis, Manolis and others},
  journal={arXiv preprint arXiv:2404.10518},
  year={2024}
}
@inproceedings{wang2024repvit,
  title={RepViT: Revisiting Mobile CNN From ViT Perspective},
  author={Wang, Ao and Chen, Hui and Lin, Zijia and others},
  booktitle={CVPR},
  year={2024}
}
@inproceedings{yun2024shvit,
  title={SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  author={Yun, Seokju and Ro, Youngmin},
  booktitle={CVPR},
  year={2024}
}
""".strip()
(MANIFESTS_DIR / "model_references.bib").write_text(bibtex + "\n", encoding="utf-8")

source_path = Path(__file__).resolve()
archived_source = MANIFESTS_DIR / "executed_source.py"
shutil.copyfile(source_path, archived_source)
(MANIFESTS_DIR / "executed_source_sha256.txt").write_text(
    hashlib.sha256(source_path.read_bytes()).hexdigest() + "\n",
    encoding="utf-8",
)

readme = f"""# MIHIC experimental bundle

- Mode: {RUN_MODE}
- Configuration fingerprint: `{CONFIG_FINGERPRINT}`
- Primary reported split: `{PRIMARY_SPLIT}`
- Dataset manifest SHA-256: `{manifest_sha256}`

The `tables/` directory contains numeric CSV exports and the root workbook
`MIHIC_paper_tables.xlsx` contains all paper-facing tables. Each model/seed
directory contains the exact architecture, parameter inventory, checkpoint,
training history, probabilities, per-class report, confusion matrix, and metrics.

PILOT and FREEZE_CHECK do not load or evaluate the official test set. FINAL
loads it only after the family configuration has been frozen on validation.
"""
(RUN_ROOT / "README.md").write_text(readme, encoding="utf-8")

pilot_decision_text = None
freeze_decision_text = None
if RUN_MODE == "PILOT":
    proposed_summary = summary_df[
        summary_df["model_id"].astype(str).str.startswith("proposed_")
    ]
    reference_summary = summary_df[
        summary_df["model_id"].astype(str).str.startswith(("baseline_", "control_"))
    ]
    if not proposed_summary.empty and not reference_summary.empty:
        best_proposed = proposed_summary.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]
        best_reference = reference_summary.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]
        delta = float(best_proposed[PRIMARY_METRIC] - best_reference[PRIMARY_METRIC])
        parameter_ratio = float(best_proposed["params_m"] / best_reference["params_m"])
        proceed = (delta >= 0.0) or (delta >= -0.003 and parameter_ratio <= 0.5)
        pilot_decision_text = f"""# PILOT go/no-go summary

- Best proposed: `{best_proposed['model_id']}` ({PRIMARY_METRIC}={best_proposed[PRIMARY_METRIC]:.4f}, params={best_proposed['params_m']:.3f}M)
- Best clean reference: `{best_reference['model_id']}` ({PRIMARY_METRIC}={best_reference[PRIMARY_METRIC]:.4f}, params={best_reference['params_m']:.3f}M)
- Macro-F1 difference: {delta:+.4f}
- Parameter ratio: {parameter_ratio:.3f}
- Automated directional recommendation: **{'PROCEED TO FROZEN FINAL' if proceed else 'ONE BOUNDED REDESIGN BEFORE FINAL'}**

This is a one-seed directional gate, not a significance claim. Do not select
architectures using the official test set.
"""
        (RUN_ROOT / "PILOT_DECISION.md").write_text(
            pilot_decision_text, encoding="utf-8"
        )
elif RUN_MODE == "FREEZE_CHECK":
    if freeze_analysis is None:
        raise RuntimeError("FREEZE_CHECK analysis was not created.")
    if freeze_analysis["official_test_loaded"]:
        raise AssertionError("FREEZE_CHECK must never load the official test set.")

    (RUN_ROOT / "FREEZE_DECISION.json").write_text(
        json.dumps(freeze_analysis, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    boundary_text = (
        ", ".join(f"`{model_id}`" for model_id in freeze_analysis["boundary_models"])
        if freeze_analysis["boundary_models"] else "None"
    )
    batch_text = ", ".join(
        str(value) for value in freeze_analysis["observed_batch_sizes"]
    )
    status_text = (
        "READY FOR MANUAL FREEZE AND FINAL CONFIGURATION"
        if freeze_analysis["ready_for_final_review"]
        else "NOT YET READY FOR FINAL — REVIEW THE FLAGS BELOW"
    )

    convergence_rows = []
    for model_id in [spec["model_id"] for spec in MODEL_SPECS]:
        row = summary_by_id.loc[model_id]
        convergence_rows.append(
            f"| `{model_id}` | {row[PRIMARY_METRIC]:.4f} | "
            f"{int(round(row['best_epoch']))} | {int(round(row['epochs_ran']))} |"
        )
    convergence_table = "\n".join(convergence_rows)

    freeze_decision_text = f"""# FREEZE_CHECK decision

**Status: {status_text}**

- Official test loaded: **No**
- Shared training batch size observed: `{batch_text}`
- Models still improving at the 35-epoch boundary: {boundary_text}
- Minimum gain required to retain an added component: `{MIN_COMPONENT_MACRO_F1_GAIN:.4f}` validation Macro-F1

## Component decisions

| Component | Enabled F1 | Disabled F1 | Gain when enabled | Recommendation |
|---|---:|---:|---:|---|
| Attention | {table_freeze_components.iloc[0]['enabled_val_macro_f1']:.4f} | {table_freeze_components.iloc[0]['disabled_val_macro_f1']:.4f} | {freeze_analysis['attention_gain']:+.4f} | `FINAL_USE_ATTENTION = {freeze_analysis['recommend_final_use_attention']}` |
| Multi-scale depthwise | {table_freeze_components.iloc[1]['enabled_val_macro_f1']:.4f} | {table_freeze_components.iloc[1]['disabled_val_macro_f1']:.4f} | {freeze_analysis['multi_scale_gain']:+.4f} | `FINAL_MULTI_SCALE = {freeze_analysis['recommend_final_multi_scale']}` |
| Distillation | {table_freeze_components.iloc[2]['enabled_val_macro_f1']:.4f} | {table_freeze_components.iloc[2]['disabled_val_macro_f1']:.4f} | {freeze_analysis['distillation_gain']:+.4f} | {'Retain KD' if freeze_analysis['distillation_supported'] else 'KD not supported; review before FINAL'} |

The component threshold is a predeclared engineering guardrail, not a
statistical-significance claim. The official test set remains unopened.

## Convergence check

| Model | Validation Macro-F1 | Best epoch | Epochs run |
|---|---:|---:|---:|
{convergence_table}

Before FINAL, copy the two recommended Boolean values into the configuration
section and review any boundary-model warning. The code deliberately blocks
FINAL while either frozen value remains `None`.
"""
    (RUN_ROOT / "FREEZE_DECISION.md").write_text(
        freeze_decision_text, encoding="utf-8"
    )


# =============================================================================
# 13. FINAL DISPLAY
# =============================================================================

print("\n" + "=" * 88)
print(f"COMPLETE — {RUN_MODE}")
print("=" * 88)
print(f"All outputs: {RUN_ROOT.resolve()}")
print(f"Paper workbook: {workbook_path.resolve()}")
print("\nMain performance table:")
display(table_main)
print("\nEfficiency table:")
display(table_efficiency)
print("\nPer-class F1 matrix:")
display(per_class_pivot)
print("\nAblation table:")
display(table_ablation)
if not table_freeze_components.empty:
    print("\nFreeze component decision table:")
    display(table_freeze_components)
if not table_subgroups.empty:
    print("\nAutomatically extracted subgroup results:")
    display(table_subgroups.head(50))
if pilot_decision_text is not None:
    print("\n" + pilot_decision_text)
if freeze_decision_text is not None:
    print("\n" + freeze_decision_text)
