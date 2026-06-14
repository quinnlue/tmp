from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


@dataclass(frozen=True)
class ImageNetLTEntry:
    relative_path: str
    label: int
    absolute_path: Path


@dataclass(frozen=True)
class ImageNetLTSplit:
    train_indices: Tensor
    objective_indices: Tensor
    meta_validation_indices: Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("train_indices", self.train_indices),
            ("objective_indices", self.objective_indices),
            ("meta_validation_indices", self.meta_validation_indices),
        ):
            if value.ndim != 1 or value.dtype != torch.long:
                raise ValueError(f"{name} must be a one-dimensional torch.long tensor")
        all_indices = torch.cat(
            (self.train_indices, self.objective_indices, self.meta_validation_indices)
        )
        if all_indices.numel() != torch.unique(all_indices).numel():
            raise ValueError("split indices must be disjoint")


@dataclass(frozen=True)
class ClassFrequencySummary:
    class_counts: dict[int, int]
    many_classes: tuple[int, ...]
    medium_classes: tuple[int, ...]
    few_classes: tuple[int, ...]
    many_samples: int
    medium_samples: int
    few_samples: int


def parse_manifest_line(line: str, *, line_number: int | None = None) -> tuple[str, int]:
    stripped = line.strip()
    if not stripped:
        raise ValueError(_format_line_error("empty line", line_number))

    parts = stripped.rsplit(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            _format_line_error(
                "expected manifest lines in 'relative/path label' format", line_number
            )
        )
    relative_path, label_text = parts
    try:
        label = int(label_text)
    except ValueError as exc:
        raise ValueError(_format_line_error(f"invalid label: {label_text!r}", line_number)) from exc
    if label < 0:
        raise ValueError(_format_line_error("label must be non-negative", line_number))
    return relative_path, label


def resolve_manifest_relative_path(
    root: str | Path,
    relative_path: str,
    *,
    validate_exists: bool = True,
) -> Path:
    root_path = Path(root).expanduser().resolve()
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise ValueError(f"manifest path must be relative: {relative_path!r}")
    if raw_path.drive:
        raise ValueError(f"manifest path must not include a drive prefix: {relative_path!r}")

    candidate = (root_path / raw_path).resolve(strict=False)
    if candidate != root_path and root_path not in candidate.parents:
        raise ValueError(
            f"manifest path escapes dataset root {str(root_path)!r}: {relative_path!r}"
        )
    if validate_exists and not candidate.is_file():
        raise FileNotFoundError(f"missing manifest image file: {candidate}")
    return candidate


def load_manifest_entries(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
    validate_paths: bool = True,
) -> list[ImageNetLTEntry]:
    manifest = Path(manifest_path)
    dataset_root = manifest.parent if root is None else Path(root)
    entries: list[ImageNetLTEntry] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            relative_path, label = parse_manifest_line(line, line_number=line_number)
            absolute_path = resolve_manifest_relative_path(
                dataset_root, relative_path, validate_exists=validate_paths
            )
            entries.append(ImageNetLTEntry(relative_path, label, absolute_path))
    if not entries:
        raise ValueError(f"manifest is empty: {manifest}")
    return entries


def class_counts(entries: Sequence[ImageNetLTEntry] | Sequence[int] | Tensor) -> dict[int, int]:
    labels = _labels_from_entries(entries)
    counts = Counter(labels)
    return dict(sorted(counts.items()))


def summarize_class_frequency(
    entries: Sequence[ImageNetLTEntry] | Sequence[int] | Tensor,
    *,
    many_threshold: int = 100,
    few_threshold: int = 20,
) -> ClassFrequencySummary:
    if few_threshold <= 0:
        raise ValueError("few_threshold must be positive")
    if many_threshold < few_threshold:
        raise ValueError("many_threshold must be at least few_threshold")

    counts = class_counts(entries)
    many: list[int] = []
    medium: list[int] = []
    few: list[int] = []
    many_samples = 0
    medium_samples = 0
    few_samples = 0
    for class_id, count in counts.items():
        if count > many_threshold:
            many.append(class_id)
            many_samples += count
        elif count < few_threshold:
            few.append(class_id)
            few_samples += count
        else:
            medium.append(class_id)
            medium_samples += count
    return ClassFrequencySummary(
        class_counts=counts,
        many_classes=tuple(many),
        medium_classes=tuple(medium),
        few_classes=tuple(few),
        many_samples=many_samples,
        medium_samples=medium_samples,
        few_samples=few_samples,
    )


def split_train_objective_meta(
    entries: Sequence[ImageNetLTEntry] | Sequence[int] | Tensor,
    *,
    objective_per_class: int | Mapping[int, int],
    meta_validation_per_class: int | Mapping[int, int],
    seed: int,
    min_train_per_class: int = 1,
) -> ImageNetLTSplit:
    if min_train_per_class < 0:
        raise ValueError("min_train_per_class must be non-negative")

    labels = _labels_tensor(entries)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    train_parts: list[Tensor] = []
    objective_parts: list[Tensor] = []
    meta_parts: list[Tensor] = []

    for class_id in sorted(class_counts(labels)):
        class_index = torch.where(labels == class_id)[0]
        shuffled = class_index[torch.randperm(class_index.numel(), generator=generator)]
        objective_count = _count_for_class(objective_per_class, class_id)
        meta_count = _count_for_class(meta_validation_per_class, class_id)
        if objective_count < 0 or meta_count < 0:
            raise ValueError("per-class split counts must be non-negative")
        required = objective_count + meta_count + min_train_per_class
        if shuffled.numel() < required:
            raise ValueError(
                f"class {class_id} has {shuffled.numel()} examples but requires {required}"
            )
        objective_stop = objective_count
        meta_stop = objective_stop + meta_count
        objective_parts.append(shuffled[:objective_stop])
        meta_parts.append(shuffled[objective_stop:meta_stop])
        train_parts.append(shuffled[meta_stop:])

    train_indices = _shuffle_concat(train_parts, generator)
    objective_indices = _shuffle_concat(objective_parts, generator)
    meta_indices = _shuffle_concat(meta_parts, generator)
    return ImageNetLTSplit(train_indices, objective_indices, meta_indices)


def batch_index_schedule(
    indices: Sequence[int] | Tensor,
    *,
    batch_size: int,
    epochs: int,
    seed: int,
    drop_last: bool = False,
) -> tuple[Tensor, ...]:
    return tuple(
        iter_batch_index_schedule(
            indices,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            drop_last=drop_last,
        )
    )


def iter_batch_index_schedule(
    indices: Sequence[int] | Tensor,
    *,
    batch_size: int,
    epochs: int,
    seed: int,
    drop_last: bool = False,
) -> Iterator[Tensor]:
    cpu_indices = _indices_tensor(indices)
    if cpu_indices.numel() == 0:
        raise ValueError("indices must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(epochs):
        permuted = cpu_indices[torch.randperm(cpu_indices.numel(), generator=generator)]
        limit = permuted.numel() if not drop_last else permuted.numel() - (permuted.numel() % batch_size)
        for start in range(0, limit, batch_size):
            stop = start + batch_size
            batch = permuted[start:stop]
            if batch.numel() < batch_size and drop_last:
                continue
            yield batch.clone()


class ImageNetLTDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        root: str | Path | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
        target_transform: Callable[[int], int] | None = None,
        validate_paths: bool = True,
        entries: Sequence[ImageNetLTEntry] | None = None,
        read_mode: ImageReadMode = ImageReadMode.RGB,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent if root is None else Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.read_mode = read_mode
        self.entries = (
            list(entries)
            if entries is not None
            else load_manifest_entries(
                self.manifest_path, root=self.root, validate_paths=validate_paths
            )
        )
        self.labels = _labels_tensor(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        entry = self.entries[index]
        image = read_image(str(entry.absolute_path), mode=self.read_mode)
        label = entry.label
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(class_counts(self.labels))

    def subset_labels(self, indices: Sequence[int] | Tensor) -> Tensor:
        return self.labels.index_select(0, _indices_tensor(indices))


def _labels_from_entries(entries: Sequence[ImageNetLTEntry] | Sequence[int] | Tensor) -> list[int]:
    if isinstance(entries, Tensor):
        if entries.ndim != 1:
            raise ValueError("labels tensor must be one-dimensional")
        return [int(value) for value in entries.detach().cpu().tolist()]
    if not entries:
        return []
    first = entries[0]
    if isinstance(first, ImageNetLTEntry):
        return [entry.label for entry in entries]  # type: ignore[arg-type]
    return [int(value) for value in entries]  # type: ignore[arg-type]


def _labels_tensor(entries: Sequence[ImageNetLTEntry] | Sequence[int] | Tensor) -> Tensor:
    labels = _labels_from_entries(entries)
    if not labels:
        raise ValueError("labels must be non-empty")
    return torch.tensor(labels, dtype=torch.long)


def _indices_tensor(indices: Sequence[int] | Tensor) -> Tensor:
    if isinstance(indices, Tensor):
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices tensor must be one-dimensional torch.long")
        return indices.detach().cpu()
    if not indices:
        raise ValueError("indices must be non-empty")
    return torch.tensor(list(indices), dtype=torch.long)


def _count_for_class(value: int | Mapping[int, int], class_id: int) -> int:
    if isinstance(value, Mapping):
        return int(value.get(class_id, 0))
    return int(value)


def _shuffle_concat(parts: Iterable[Tensor], generator: torch.Generator) -> Tensor:
    realized = [part.detach().cpu().to(torch.long) for part in parts if part.numel()]
    if not realized:
        return torch.zeros(0, dtype=torch.long)
    merged = torch.cat(realized)
    return merged[torch.randperm(merged.numel(), generator=generator)]


def _format_line_error(message: str, line_number: int | None) -> str:
    if line_number is None:
        return message
    return f"line {line_number}: {message}"
