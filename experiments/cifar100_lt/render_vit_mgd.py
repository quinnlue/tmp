"""Render the CIFAR100-LT / ViT-Tiny MGD granularity comparison as markdown.

Reads the artifacts produced by `experiments.cifar100_lt.vit_mgd`:
  baseline.json (uniform), reeval_<granularity>.json (full-recipe re-eval under
  learned weights), and search_<granularity>.json (the MGD search trajectory).

Usage: python -m experiments.cifar100_lt.render_vit_mgd [--artifact-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GRANULARITIES = ("per_class", "per_cluster", "per_example")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _row(name: str, result: dict[str, Any]) -> str:
    test = result["test"]
    val = result["val"]
    return (
        f"| {name} | {_pct(val['balanced_accuracy'])} | "
        f"**{_pct(test['balanced_accuracy'])}** | {_pct(test['many_accuracy'])} | "
        f"{_pct(test['medium_accuracy'])} | {_pct(test['few_accuracy'])} | "
        f"{result['best_epoch']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/cifar100_lt_vit_mgd")
    args = parser.parse_args()
    root = Path(args.artifact_dir)

    baseline = _load(root / "baseline.json")
    lines: list[str] = []
    lines.append("## CIFAR100-LT ViT-Tiny -- MGD curation granularity comparison")
    lines.append("")
    lines.append("Balanced final-test accuracy under the identical 75-epoch recipe, "
                 "differing only in the learned per-example data weights.")
    lines.append("")
    lines.append("| Method | Val bal | Test bal | Many | Medium | Few | Best epoch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    if baseline is not None:
        lines.append(_row("uniform (baseline)", baseline))
    for granularity in GRANULARITIES:
        result = _load(root / f"reeval_{granularity}.json")
        if result is not None:
            lines.append(_row(granularity, result))
    lines.append("")

    # Search trajectory summary
    lines.append("### MGD search summary (balanced meta-validation CE)")
    lines.append("")
    lines.append("| Granularity | Groups | Selected step | Meta-val CE (start -> selected) | Final entropy | Final ESS |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for granularity in GRANULARITIES:
        search = _load(root / f"search_{granularity}.json")
        if search is None:
            continue
        history = search["history"]
        selected = search["selected_step"]
        start_ce = history[0]["meta_validation_ce"]
        sel_ce = history[selected]["meta_validation_ce"]
        final = history[selected]
        lines.append(
            f"| {granularity} | {search['num_groups']} | {selected}/{len(history) - 1} | "
            f"{start_ce:.4f} -> {sel_ce:.4f} | {final['entropy']:.3f} | "
            f"{final['effective_sample_size']:.1f} |"
        )
    lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
