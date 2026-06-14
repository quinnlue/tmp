"""Summarize the autoresearch per-example versus per-class MGD experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RUNS = (
    ("q1_primary.json", "Large objective\nvehicle skew"),
    ("q3_hard_animal_skew.json", "Large objective\nanimal skew"),
    ("q4_small_objective.json", "Small objective\nseed 0"),
    ("q5_small_objective_seed1.json", "Small objective\nseed 1"),
)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def selected_summary(result: dict) -> dict:
    selected = result["selected"]
    return {
        "step": selected["step"],
        "objective_ce": selected["objective_ce"],
        "validation_ce": selected["validation"]["target_ce"],
        "test_ce": selected["test"]["target_ce"],
        "test_acc": selected["test"]["target_acc"],
        "balanced_test_ce": selected["test"]["balanced_ce"],
        "balanced_test_acc": selected["test"]["balanced_acc"],
        "test_gap": selected["test_gap"],
        "overall_ess": selected["weights"]["overall_ess"],
        "class_masses": selected["weights"]["class_masses"],
        "class_ess": selected["weights"]["class_ess"],
    }


def control_summary(result: dict) -> dict:
    return {
        "objective_ce": result["objective_ce"],
        "validation_ce": result["validation"]["target_ce"],
        "test_ce": result["test"]["target_ce"],
        "test_acc": result["test"]["target_acc"],
        "balanced_test_ce": result["test"]["balanced_ce"],
        "balanced_test_acc": result["test"]["balanced_acc"],
        "test_gap": result["test_gap"],
    }


def compact_run(raw: dict) -> dict:
    methods = {
        name: selected_summary(result)
        for name, result in raw["methods"].items()
    }
    return {
        "config": raw["config"],
        "design": raw["design"],
        "uniform": control_summary(raw["uniform"]),
        "oracle": control_summary(raw["oracle"]),
        "methods": methods,
        "fairness_checks": raw["fairness_checks"],
    }


def improvement(uniform: float, value: float) -> str:
    absolute = uniform - value
    relative = 100.0 * absolute / uniform
    return f"{absolute:+.3f} ({relative:+.1f}%)"


def markdown_report(summary: dict) -> str:
    lines = [
        "# Autoresearch: Per-example vs Per-class Persistent-softmax MGD",
        "",
        "## Setup",
        "",
        "- Balanced CIFAR-10 training pool: 4,000 examples, exactly 400/class.",
        "- Objective and meta-validation are disjoint and share an explicit all-class skew.",
        "- The official CIFAR-10 test set is evaluated only after selecting the best meta-step by validation CE.",
        "- Both methods share data, model initialization, trajectory, optimizer, objective, and fixed-KL signed updates.",
        "- Only the weighting granularity differs: 10 class logits versus 4,000 example logits.",
        "- This compares persistent-softmax MGD parameterizations, not the paper's discrete count-MGD algorithm.",
        "",
        "## Selected Test Results",
        "",
        "| Run | Method | Test CE | Gain vs uniform | Target acc | Balanced CE | Obj-to-test gap | Selected step |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, run in summary["comparison_runs"].items():
        uniform = run["uniform"]
        lines.append(
            f"| {run_name} | uniform | {uniform['test_ce']:.3f} | - | "
            f"{uniform['test_acc']:.3f} | {uniform['balanced_test_ce']:.3f} | "
            f"{uniform['test_gap']:+.3f} | - |"
        )
        for method in ("per_class", "per_example"):
            result = run["methods"][method]
            lines.append(
                f"| {run_name} | {method} | {result['test_ce']:.3f} | "
                f"{improvement(uniform['test_ce'], result['test_ce'])} | "
                f"{result['test_acc']:.3f} | {result['balanced_test_ce']:.3f} | "
                f"{result['test_gap']:+.3f} | {result['step']} |"
            )
        oracle = run["oracle"]
        lines.append(
            f"| {run_name} | class-ratio oracle | {oracle['test_ce']:.3f} | "
            f"{improvement(uniform['test_ce'], oracle['test_ce'])} | "
            f"{oracle['test_acc']:.3f} | {oracle['balanced_test_ce']:.3f} | "
            f"{oracle['test_gap']:+.3f} | - |"
        )

    primary = summary["comparison_runs"]["Large objective / vehicle skew"]
    hard = summary["comparison_runs"]["Large objective / animal skew"]
    small0 = summary["comparison_runs"]["Small objective / seed 0"]
    small1 = summary["comparison_runs"]["Small objective / seed 1"]
    q2 = summary["per_class_small_step"]
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. **Per-example MGD dominates with a large objective.** "
            f"On the primary skew it lowers test CE from {primary['uniform']['test_ce']:.3f} "
            f"to {primary['methods']['per_example']['test_ce']:.3f}, while per-class reaches "
            f"{primary['methods']['per_class']['test_ce']:.3f}.",
            "2. **The result is not driven by easy target classes.** "
            f"On the hard-animal skew, per-example reaches {hard['methods']['per_example']['test_ce']:.3f}, "
            f"nearly matching the class-ratio oracle at {hard['oracle']['test_ce']:.3f}; "
            f"per-class reaches {hard['methods']['per_class']['test_ce']:.3f}.",
            "3. **Per-example gains come from within-class selection.** Its learned aggregate "
            "class masses stay close to uniform while within-class effective sample sizes fall, "
            "showing that it selects useful examples rather than merely recovering label prevalence.",
            "4. **Per-class MGD is update-sensitive and noisy.** A 10x smaller KL step improves its "
            f"primary-skew test CE to {q2['methods']['per_class']['test_ce']:.3f}, but it remains "
            "well behind per-example MGD.",
            "5. **A small objective narrows the gap and exposes per-example overfitting.** "
            f"At seed 0, test CE is {small0['methods']['per_class']['test_ce']:.3f} vs "
            f"{small0['methods']['per_example']['test_ce']:.3f}; at seed 1 it is "
            f"{small1['methods']['per_class']['test_ce']:.3f} vs "
            f"{small1['methods']['per_example']['test_ce']:.3f}. "
            "Per-example has a positive objective-to-test gap in both small-objective runs, "
            "but still wins target-test CE.",
            "6. **There is a target-versus-balanced tradeoff.** Per-example usually improves the "
            "skewed target more, while per-class often preserves or improves balanced-test performance.",
            "",
            "## Interpretation",
            "",
            "The class basis is too restrictive for this supervised CIFAR-10 setting: examples "
            "within the same label differ substantially in usefulness for the skewed target. "
            "Per-example MGD exploits that signal and can match a class-ratio oracle without moving "
            "aggregate class masses much. The expected statistical advantage of the low-dimensional "
            "class basis appears only as reduced overfitting and better balanced-task retention when "
            "the objective set is small; it does not translate into better target-test performance here.",
            "",
            "## Limitations",
            "",
            "- One seed was used for the large-objective conditions; the noisy small-objective condition was confirmed with two seeds.",
            "- The model is a smooth GroupNorm ResNet-9 and the inner horizon is 192 steps, not full convergence.",
            "- Fixed-KL signed updates are fair in induced-distribution movement, but other outer optimizers may change the ranking.",
            "- This is persistent-softmax MGD, not the paper's paper-faithful count-based dataset-selection algorithm.",
            "",
            "Paper grounding: [Optimizing ML Training with Metagradient Descent](https://arxiv.org/abs/2503.13751).",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_summary(raw_runs: dict, summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [label for _, label in RUNS]
    keys = [label.replace("\n", " / ") for label in labels]
    methods = ("uniform", "per_class", "per_example", "oracle")
    colors = {
        "uniform": "#777777",
        "per_class": "#e76f51",
        "per_example": "#2a9d8f",
        "oracle": "#264653",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    positions = np.arange(len(keys))
    width = 0.2
    for offset, method in enumerate(methods):
        ce_values = []
        accuracy_values = []
        for key in keys:
            run = summary["comparison_runs"][key]
            result = run[method] if method in ("uniform", "oracle") else run["methods"][method]
            ce_values.append(result["test_ce"])
            accuracy_values.append(result["test_acc"])
        axes[0, 0].bar(
            positions + (offset - 1.5) * width,
            ce_values,
            width,
            label=method,
            color=colors[method],
        )
        axes[0, 1].bar(
            positions + (offset - 1.5) * width,
            accuracy_values,
            width,
            label=method,
            color=colors[method],
        )
    for axis, title, ylabel in (
        (axes[0, 0], "Validation-selected target-test CE", "cross-entropy (lower is better)"),
        (axes[0, 1], "Validation-selected target-test accuracy", "accuracy (higher is better)"),
    ):
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=2)

    for axis, filename, title in (
        (axes[1, 0], "q1_primary.json", "Primary validation trajectory"),
        (axes[1, 1], "q4_small_objective.json", "Small-objective validation trajectory"),
    ):
        raw = raw_runs[filename]
        for method in ("per_class", "per_example"):
            history = raw["methods"][method]["history"]
            axis.plot(
                [record["step"] for record in history],
                [record["validation"]["target_ce"] for record in history],
                marker="o",
                label=method,
                color=colors[method],
            )
        axis.axhline(
            raw["uniform"]["validation"]["target_ce"],
            color=colors["uniform"],
            linestyle="--",
            label="uniform",
        )
        axis.axhline(
            raw["oracle"]["validation"]["target_ce"],
            color=colors["oracle"],
            linestyle=":",
            label="oracle",
        )
        axis.set_title(title)
        axis.set_xlabel("meta-step")
        axis.set_ylabel("skew-matched validation CE")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Persistent-softmax MGD granularity autoresearch", fontsize=15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)


def main(artifacts_dir: Path) -> None:
    raw_runs = {filename: load(artifacts_dir / filename) for filename, _ in RUNS}
    comparison_runs = {
        label.replace("\n", " / "): compact_run(raw_runs[filename])
        for filename, label in RUNS
    }
    summary = {
        "comparison_runs": comparison_runs,
        "per_class_small_step": compact_run(
            load(artifacts_dir / "q2_per_class_small_step.json")
        ),
    }
    with (artifacts_dir / "autoresearch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (artifacts_dir / "AUTORESEARCH_RESULTS.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    plot_summary(raw_runs, summary, artifacts_dir / "autoresearch_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    main(args.artifacts_dir)
