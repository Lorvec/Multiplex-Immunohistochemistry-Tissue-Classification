#!/usr/bin/env python3
"""Rebuild all five paper-facing figures from archived machine-readable tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPOSITORY_ROOT / "results" / "tables"

NAVY = "#173B63"
BLUE = "#216D9B"
GREEN = "#2A7B59"
ORANGE = "#C36B0A"
GRAY = "#69747D"
LIGHT_GRAY = "#A7AEB5"
PALE = "#EAF1F8"
BLACK = "#222222"

DISPLAY_LABELS = {
    "teacher_convnext_tiny": "Teacher",
    "proposed_nano_kd": "Nano",
    "proposed_tiny_kd": "Tiny",
    "proposed_small_kd": "Small",
    "control_tinycnn_gap_clean": "TinyCNN",
    "baseline_efficientnet_b0_scratch": "EfficientNet-B0",
    "baseline_mobilenet_v3_small_scratch": "MobileNetV3",
    "baseline_mobilenetv4_conv_small_pretrained": "MobileNetV4",
    "baseline_repvit_m0_9_pretrained": "RepViT",
    "baseline_shvit_s2_pretrained": "SHViT-S2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "reproduced_figures",
    )
    return parser.parse_args()


def rounded_box(
    ax,
    x,
    y,
    width,
    height,
    edge,
    title,
    lines=(),
    *,
    face="white",
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.67,
        title,
        ha="center",
        va="center",
        color=edge,
        fontsize=11.5,
        fontweight="bold",
    )
    if lines:
        ax.text(
            x + width / 2,
            y + height * 0.34,
            "\n".join(lines),
            ha="center",
            va="center",
            color="#2B2B2B",
            fontsize=9.3,
            linespacing=1.18,
        )
    return box


def arrow(ax, start, end, color=GRAY, lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def figure_1(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(
        0.5,
        0.965,
        "Study design and compact-model family",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#151515",
    )
    ax.text(
        0.025,
        0.88,
        "A  Validation-frozen study design",
        fontsize=13.2,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    rounded_box(
        ax,
        0.035,
        0.665,
        0.245,
        0.13,
        GRAY,
        "PILOT",
        ("Exploratory setup", "No final claims"),
    )
    rounded_box(
        ax,
        0.377,
        0.665,
        0.245,
        0.13,
        BLUE,
        "FREEZE_CHECK",
        ("Validation only", "Seed 42 · ≤35 epochs"),
        face="#F8FBFD",
    )
    rounded_box(
        ax,
        0.719,
        0.665,
        0.245,
        0.13,
        GREEN,
        "FINAL",
        ("Official test opened once", "Seeds 42, 123, 2026"),
        face="#F7FBF8",
    )
    arrow(ax, (0.285, 0.73), (0.372, 0.73))
    arrow(ax, (0.627, 0.73), (0.714, 0.73))
    ax.text(
        0.5,
        0.625,
        "Architecture and component decisions were frozen before final test evaluation.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#303030",
    )
    ax.text(
        0.025,
        0.535,
        "B  Offline knowledge distillation",
        fontsize=13.2,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    rounded_box(
        ax, 0.035, 0.335, 0.155, 0.105, BLACK, "ConvNeXt-Tiny", ("teacher",)
    )
    rounded_box(
        ax,
        0.285,
        0.335,
        0.155,
        0.105,
        ORANGE,
        "Cached soft targets",
        ("T = 3.0",),
        face="#FFFBF5",
    )
    rounded_box(
        ax,
        0.035,
        0.165,
        0.155,
        0.105,
        BLUE,
        "Augmented",
        ("128×128 patch",),
        face="#F8FBFD",
    )
    rounded_box(
        ax,
        0.285,
        0.165,
        0.155,
        0.105,
        GREEN,
        "Student family",
        ("Nano / Tiny / Small",),
        face="#F7FBF8",
    )
    arrow(ax, (0.195, 0.387), (0.28, 0.387))
    arrow(ax, (0.195, 0.217), (0.28, 0.217))
    arrow(ax, (0.362, 0.33), (0.362, 0.275), color=ORANGE)
    ax.text(0.372, 0.303, "KL", fontsize=8.8, color=ORANGE, va="center")
    ax.text(
        0.238,
        0.112,
        "Loss = 0.45 × weighted CE + 0.55 × T² KL",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#303030",
    )
    ax.text(
        0.535,
        0.535,
        "C  Scalable compact family",
        fontsize=13.2,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    arrow(ax, (0.585, 0.475), (0.935, 0.475), lw=1.5)
    ax.text(0.76, 0.495, "capacity", ha="center", va="bottom", fontsize=9.5, color=GRAY)
    cards = [
        (0.535, "Nano", "0.300M params", "0.0298 GMACs", "Macro-F1\n0.8006"),
        (0.685, "Tiny", "0.819M params", "0.0391 GMACs", "Macro-F1\n0.8016"),
        (0.835, "Small", "2.528M params", "0.1174 GMACs", "Macro-F1\n0.8104"),
    ]
    for x, name, params, gmacs, score in cards:
        card = FancyBboxPatch(
            (x, 0.17),
            0.13,
            0.245,
            boxstyle="round,pad=0.009,rounding_size=0.012",
            linewidth=2.0,
            edgecolor=BLUE,
            facecolor=PALE,
        )
        ax.add_patch(card)
        ax.text(x + 0.065, 0.365, name, ha="center", va="center", fontsize=12.7, fontweight="bold", color=NAVY)
        ax.text(x + 0.065, 0.305, params, ha="center", va="center", fontsize=9.2, color="#2B2B2B")
        ax.text(x + 0.065, 0.255, gmacs, ha="center", va="center", fontsize=9.0, color=GRAY)
        ax.text(x + 0.065, 0.205, score, ha="center", va="center", fontsize=8.8, color=GREEN, fontweight="bold", linespacing=1.05)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def figure_2(output: Path) -> None:
    summary = pd.read_csv(TABLES / "summary_mean_std_numeric.csv")
    summary = summary[summary["model_id"].isin(DISPLAY_LABELS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), dpi=300, sharey=True)
    for ax, x_column, x_label, panel in [
        (axes[0], "params_m", "Parameters (millions, log scale)", "A  Parameter–performance trade-off"),
        (axes[1], "gmacs", "Compute per image (GMACs, log scale)", "B  Compute–performance trade-off"),
    ]:
        for row in summary.itertuples(index=False):
            proposed = row.model_id.startswith("proposed_")
            teacher = row.model_id == "teacher_convnext_tiny"
            marker = "*" if teacher else ("o" if proposed else "s")
            color = BLACK if teacher else (BLUE if proposed else LIGHT_GRAY)
            size = 145 if teacher else 55
            ax.errorbar(
                getattr(row, x_column),
                row.test_macro_f1,
                yerr=row.test_macro_f1_std,
                fmt=marker,
                markersize=np.sqrt(size),
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                markeredgewidth=0.5,
            )
            if row.model_id in {
                "teacher_convnext_tiny",
                "proposed_nano_kd",
                "proposed_tiny_kd",
                "proposed_small_kd",
                "baseline_repvit_m0_9_pretrained",
                "baseline_mobilenetv4_conv_small_pretrained",
            }:
                ax.annotate(
                    DISPLAY_LABELS[row.model_id],
                    (getattr(row, x_column), row.test_macro_f1),
                    xytext=(6, 5 if row.model_id != "proposed_nano_kd" else -14),
                    textcoords="offset points",
                    fontsize=8.5,
                )
        ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_title(panel, loc="left", color=NAVY, fontweight="bold", fontsize=11.5)
        ax.grid(alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test Macro-F1 (mean ± SD; three seeds)")
    fig.suptitle(
        "Performance preservation across model size and computation",
        fontsize=16,
        fontweight="bold",
    )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, label="Proposed family"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=BLACK, markeredgecolor=BLACK, markersize=11, label="Teacher"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=LIGHT_GRAY, markeredgecolor=LIGHT_GRAY, label="Comparator"),
    ]
    fig.legend(handles=legend, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_3(output: Path) -> None:
    per_class = pd.read_csv(TABLES / "table_2_per_class_full.csv")
    model_order = [
        "baseline_repvit_m0_9_pretrained",
        "proposed_small_kd",
        "teacher_convnext_tiny",
        "baseline_mobilenetv4_conv_small_pretrained",
        "proposed_tiny_kd",
        "proposed_nano_kd",
    ]
    class_order = ["Immune cells", "Necrosis", "Other", "Stroma", "Tumor", "alveoli", "background"]
    matrix = per_class.pivot(index="model_id", columns="class", values="f1_mean").loc[model_order, class_order]
    matrix.index = [DISPLAY_LABELS[name] for name in matrix.index]
    matrix.columns = [name.capitalize() if name in {"alveoli", "background"} else name for name in matrix.columns]
    fig, ax = plt.subplots(figsize=(11.5, 4.45), dpi=300)
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Blues", vmin=0.55, vmax=1.0, linewidths=0.7, cbar_kws={"label": "Mean class F1"}, ax=ax)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Class-specific test performance", loc="left", fontsize=17, fontweight="bold", pad=24)
    small_row = list(matrix.index).index("Small")
    for column in range(matrix.shape[1]):
        ax.texts[small_row * matrix.shape[1] + column].set_fontweight("bold")
    fig.text(0.12, 0.015, "Stroma and Other remained the most difficult categories across models.", color=GRAY, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_4(output: Path) -> None:
    subgroups = pd.read_csv(TABLES / "table_6_subgroup_performance.csv")
    stains = subgroups[
        (subgroups["subgroup_type"] == "stain")
        & subgroups["model_id"].isin(["teacher_convnext_tiny", "proposed_small_kd"])
    ]
    means = stains.groupby(["subgroup", "model_id"])["macro_f1"].mean().unstack()
    support = stains.groupby("subgroup")["support"].max()
    order = ["CD20", "CD3", "CD34", "CD38", "CD68", "CDK4", "D2-40", "FAP", "Ki67", "P53", "SMA", "cyclin-D1"]
    y = np.arange(len(order))
    teacher = means.loc[order, "teacher_convnext_tiny"].to_numpy()
    small = means.loc[order, "proposed_small_kd"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.7, 7.1), dpi=300)
    for index, (left, right) in enumerate(zip(teacher, small)):
        ax.plot([left, right], [index, index], color="#C8CDD2", linewidth=2, zorder=1)
    ax.scatter(teacher, y, color=BLACK, s=54, label="ConvNeXt-Tiny teacher", zorder=2)
    ax.scatter(small, y, color=BLUE, s=54, label="Proposed Small", zorder=3)
    ax.set_yticks(y, [f"{stain}  (n={int(support[stain]):,})" for stain in order])
    ax.invert_yaxis()
    ax.set_xlabel("Stain-specific Macro-F1 (mean across three seeds)")
    ax.set_title("Stain-level comparison of the compact Small model and its teacher", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.28)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.19, 0.015, "Interpret descriptively: stain subsets differ in class presence and composition; n indicates test support.", color=GRAY, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_5(output: Path) -> None:
    values = pd.read_csv(TABLES / "figure_5_confusion_matrices.csv")
    classes = ["Immune cells", "Necrosis", "Other", "Stroma", "Tumor", "alveoli", "background"]
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.0), dpi=300)
    for ax, model_id in zip(axes, ["teacher_convnext_tiny", "proposed_small_kd"]):
        subset = values[values["model_id"] == model_id]
        matrix = subset.pivot(index="true_class", columns="predicted_class", values="row_normalized_value").loc[classes, classes]
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=classes, yticklabels=classes, cbar=True, ax=ax)
        ax.set_title(f"{model_id} — normalized confusion matrix", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builders = [figure_1, figure_2, figure_3, figure_4, figure_5]
    names = [
        "fig1_study_design.png",
        "fig2_efficiency_tradeoffs.png",
        "fig3_per_class_f1.png",
        "fig4_stain_specific_f1.png",
        "fig5_teacher_small_confusion.png",
    ]
    for builder, name in zip(builders, names):
        destination = args.output_dir / name
        builder(destination)
        print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
