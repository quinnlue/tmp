from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


METRICS = (
    "objective_balanced_ce",
    "meta_validation_balanced_ce",
    "val_balanced_ce",
    "test_balanced_ce",
)


def load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def selected_method_summary(method: dict[str, Any]) -> dict[str, Any]:
    history = method["history"]
    selected_step = int(method["selected_step"])
    selected = history[selected_step]
    return {
        "selected_step": selected_step,
        "selection_rule": method["selection_rule"],
        "loss_deltas": {
            metric: selected["loss_deltas"].get(metric)
            for metric in METRICS
            if metric in selected["loss_deltas"]
        },
        "selected_metrics": {
            "objective_balanced_ce": selected["objective_metrics"]["balanced_ce"],
            "meta_validation_balanced_ce": selected["meta_validation_metrics"]["balanced_ce"],
            **(
                {"val_balanced_ce": selected["val_metrics"]["balanced_ce"]}
                if "val_metrics" in selected
                else {}
            ),
            **(
                {"test_balanced_ce": selected["test_metrics"]["balanced_ce"]}
                if "test_metrics" in selected
                else {}
            ),
        },
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "design": result["design"],
        "fairness_checks": result["fairness_checks"],
        "controls": {
            name: control["history"][0]
            for name, control in result.get("controls", {}).items()
        },
        "methods": {
            name: selected_method_summary(method)
            for name, method in result["methods"].items()
        },
    }


def markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "# ImageNet-LT MGD Loss-Delta Study",
        "",
        "| Technique | Selected step | Objective balanced CE delta | Meta-val balanced CE delta | Val balanced CE delta | Test balanced CE delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, method in summary["methods"].items():
        deltas = method["loss_deltas"]
        values = [
            "-" if deltas.get(metric) is None else f"{deltas[metric]:+.6f}"
            for metric in METRICS
        ]
        lines.append(
            f"| {name} | {method['selected_step']} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Positive deltas mean the selected MGD step reduced cross-entropy relative to that technique's uniform-logit step 0.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_trajectories(result: dict[str, Any], output: str | Path) -> None:
    import matplotlib.pyplot as plt

    available_metrics = [
        metric
        for metric in METRICS
        if any(metric in record["loss_deltas"] for method in result["methods"].values() for record in method["history"])
    ]
    figure, axes = plt.subplots(
        len(available_metrics),
        1,
        figsize=(9, max(3.5, 2.8 * len(available_metrics))),
        squeeze=False,
    )
    for axis, metric in zip(axes[:, 0], available_metrics, strict=True):
        for name, method in result["methods"].items():
            history = method["history"]
            axis.plot(
                [record["step"] for record in history],
                [record["loss_deltas"].get(metric, float("nan")) for record in history],
                marker="o",
                label=name,
            )
        axis.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("meta-step")
        axis.set_ylabel("CE reduction vs step 0")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def summarize(input_path: str | Path, output_prefix: str | Path) -> dict[str, Any]:
    result = load_result(input_path)
    summary = summarize_result(result)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prefix.with_suffix(".md").write_text(markdown_table(summary), encoding="utf-8")
    plot_trajectories(result, prefix.with_suffix(".png"))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ImageNet-LT MGD loss-delta results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = summarize(args.input, args.output_prefix)
    print(json.dumps(summary["methods"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
