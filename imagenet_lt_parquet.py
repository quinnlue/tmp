from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image, read_image

_SHARD_NAME = re.compile(r"^(?P<split>[a-zA-Z0-9_]+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.parquet$")

_PYARROW_IMPORT_ERROR = (
    "pyarrow is required for Hugging Face parquet ImageNet-LT support. "
    "Install it with `pip install pyarrow`."
)


@dataclass(frozen=True)
class ParquetShardRowGroup:
    row_group_index: int
    num_rows: int
    start_index: int

    @property
    def stop_index(self) -> int:
        return self.start_index + self.num_rows


@dataclass(frozen=True)
class ParquetShardMetadata:
    path: Path
    split: str
    shard_index: int
    shard_count: int
    num_rows: int
    start_index: int
    row_groups: tuple[ParquetShardRowGroup, ...]

    @property
    def stop_index(self) -> int:
        return self.start_index + self.num_rows


@dataclass(frozen=True)
class ParquetSplitIndex:
    split: str
    shards: tuple[ParquetShardMetadata, ...]
    num_rows: int

    def locate(self, index: int) -> tuple[ParquetShardMetadata, ParquetShardRowGroup, int]:
        normalized = _normalize_index(index, self.num_rows)
        shard_starts = [shard.start_index for shard in self.shards]
        shard_position = bisect.bisect_right(shard_starts, normalized) - 1
        shard = self.shards[shard_position]
        local_index = normalized - shard.start_index
        row_group_starts = [row_group.start_index for row_group in shard.row_groups]
        row_group_position = bisect.bisect_right(row_group_starts, local_index) - 1
        row_group = shard.row_groups[row_group_position]
        row_index = local_index - row_group.start_index
        return shard, row_group, row_index


def discover_hf_parquet_splits(root: str | Path) -> dict[str, tuple[Path, ...]]:
    dataset_root = Path(root)
    search_root = dataset_root / "data" if (dataset_root / "data").is_dir() else dataset_root
    grouped: dict[str, list[tuple[int, int, Path]]] = {}
    for path in sorted(search_root.glob("*.parquet")):
        match = _SHARD_NAME.match(path.name)
        if match is None:
            continue
        split = match.group("split")
        grouped.setdefault(split, []).append(
            (int(match.group("index")), int(match.group("total")), path.resolve())
        )

    if not grouped:
        raise FileNotFoundError(f"no Hugging Face parquet shards found under {search_root}")

    discovered: dict[str, tuple[Path, ...]] = {}
    for split, shard_records in grouped.items():
        shard_records.sort()
        expected_total = shard_records[0][1]
        if any(total != expected_total for _, total, _ in shard_records):
            raise ValueError(f"split {split!r} mixes different shard-count suffixes")
        actual_indices = [index for index, _, _ in shard_records]
        expected_indices = list(range(expected_total))
        if actual_indices != expected_indices:
            raise ValueError(
                f"split {split!r} shards are incomplete or out of order: "
                f"expected {expected_indices}, got {actual_indices}"
            )
        discovered[split] = tuple(path for _, _, path in shard_records)
    return dict(sorted(discovered.items()))


def build_split_index(
    parquet_files: Sequence[str | Path],
    *,
    split: str | None = None,
    image_column: str = "image",
    label_column: str = "label",
) -> ParquetSplitIndex:
    pq = _pyarrow_parquet()
    if not parquet_files:
        raise ValueError("parquet_files must be non-empty")
    ordered_files = sorted((Path(path).resolve() for path in parquet_files), key=_parquet_sort_key)

    shard_entries: list[ParquetShardMetadata] = []
    running_start = 0
    inferred_split = split
    expected_shard_count: int | None = None

    for position, path in enumerate(ordered_files):
        match = _SHARD_NAME.match(path.name)
        if match is not None:
            file_split = match.group("split")
            shard_index = int(match.group("index"))
            shard_count = int(match.group("total"))
        else:
            file_split = inferred_split or "unknown"
            shard_index = position
            shard_count = len(parquet_files)
        if inferred_split is None:
            inferred_split = file_split
        elif file_split != inferred_split:
            raise ValueError(f"mixed split names are not supported: {file_split!r} != {inferred_split!r}")
        if expected_shard_count is None:
            expected_shard_count = shard_count
        elif shard_count != expected_shard_count:
            raise ValueError("all shards for a split must agree on the total shard count")

        parquet_file = pq.ParquetFile(path)
        schema_names = set(parquet_file.schema_arrow.names)
        missing = {image_column, label_column}.difference(schema_names)
        if missing:
            raise ValueError(f"parquet shard {path} is missing required columns: {sorted(missing)}")

        row_groups: list[ParquetShardRowGroup] = []
        row_group_start = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group_metadata = parquet_file.metadata.row_group(row_group_index)
            row_groups.append(
                ParquetShardRowGroup(
                    row_group_index=row_group_index,
                    num_rows=row_group_metadata.num_rows,
                    start_index=row_group_start,
                )
            )
            row_group_start += row_group_metadata.num_rows

        shard_entries.append(
            ParquetShardMetadata(
                path=path,
                split=inferred_split,
                shard_index=shard_index,
                shard_count=shard_count,
                num_rows=parquet_file.metadata.num_rows,
                start_index=running_start,
                row_groups=tuple(row_groups),
            )
        )
        running_start += parquet_file.metadata.num_rows

    return ParquetSplitIndex(
        split=inferred_split or "unknown",
        shards=tuple(sorted(shard_entries, key=lambda shard: shard.shard_index)),
        num_rows=running_start,
    )


def load_split_labels(
    parquet_files: Sequence[str | Path],
    *,
    label_column: str = "label",
) -> Tensor:
    pq = _pyarrow_parquet()
    labels: list[int] = []
    for file_name in parquet_files:
        parquet_file = pq.ParquetFile(Path(file_name).resolve())
        for batch in parquet_file.iter_batches(columns=[label_column], use_threads=False):
            column = batch.column(0)
            labels.extend(int(value.as_py()) for value in column)
    if not labels:
        raise ValueError("parquet label columns are empty")
    return torch.tensor(labels, dtype=torch.long)


def class_counts(labels: Sequence[int] | Tensor) -> dict[int, int]:
    values = _labels_to_list(labels)
    return dict(sorted(Counter(values).items()))


class ImageNetLTParquetDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        split: str = "train",
        parquet_files: Sequence[str | Path] | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
        target_transform: Callable[[int], int] | None = None,
        image_column: str = "image",
        label_column: str = "label",
        read_mode: ImageReadMode = ImageReadMode.RGB,
    ) -> None:
        if parquet_files is None:
            if root is None:
                raise ValueError("root is required when parquet_files is omitted")
            split_map = discover_hf_parquet_splits(root)
            if split not in split_map:
                raise KeyError(f"split {split!r} is unavailable; found {sorted(split_map)}")
            parquet_files = split_map[split]
        elif not parquet_files:
            raise ValueError("parquet_files must be non-empty")

        self.root = _infer_dataset_root(root, parquet_files)
        self.transform = transform
        self.target_transform = target_transform
        self.image_column = image_column
        self.label_column = label_column
        self.read_mode = read_mode
        self.parquet_files = tuple(Path(path).resolve() for path in parquet_files)
        self.index = build_split_index(
            self.parquet_files,
            split=split,
            image_column=image_column,
            label_column=label_column,
        )
        self.split = self.index.split
        self.shards = self.index.shards
        self.labels = load_split_labels(self.parquet_files, label_column=label_column)
        self._row_group_cache_key: tuple[Path, int] | None = None
        self._row_group_cache: list[tuple[Any, int]] | None = None

    def __len__(self) -> int:
        return self.index.num_rows

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        shard, row_group, row_index = self.index.locate(index)
        rows = self._load_row_group(shard, row_group)
        image_value, label = rows[row_index]
        image = _decode_hf_image(image_value, dataset_root=self.root, read_mode=self.read_mode)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(class_counts(self.labels))

    @property
    def shard_metadata(self) -> tuple[ParquetShardMetadata, ...]:
        return self.shards

    @property
    def label_counts(self) -> dict[int, int]:
        return class_counts(self.labels)

    def subset_labels(self, indices: Sequence[int] | Tensor) -> Tensor:
        return self.labels.index_select(0, _indices_tensor(indices))

    def global_index_to_location(self, index: int) -> tuple[ParquetShardMetadata, ParquetShardRowGroup, int]:
        return self.index.locate(index)

    def _load_row_group(
        self,
        shard: ParquetShardMetadata,
        row_group: ParquetShardRowGroup,
    ) -> list[tuple[Any, int]]:
        cache_key = (shard.path, row_group.row_group_index)
        if self._row_group_cache_key == cache_key and self._row_group_cache is not None:
            return self._row_group_cache

        pq = _pyarrow_parquet()
        batch = pq.ParquetFile(shard.path).read_row_group(
            row_group.row_group_index,
            columns=[self.image_column, self.label_column],
            use_threads=False,
        )
        image_column = batch.column(self.image_column)
        label_column = batch.column(self.label_column)
        rows = [
            (image_column[offset].as_py(), int(label_column[offset].as_py()))
            for offset in range(batch.num_rows)
        ]
        self._row_group_cache_key = cache_key
        self._row_group_cache = rows
        return rows


def _decode_hf_image(
    image_value: Any,
    *,
    dataset_root: Path,
    read_mode: ImageReadMode,
) -> Tensor:
    if not isinstance(image_value, Mapping):
        raise TypeError(f"expected Hugging Face image struct mapping, got {type(image_value)!r}")

    image_bytes = image_value.get("bytes")
    image_path = image_value.get("path")
    if image_bytes is not None:
        encoded = _bytes_to_uint8_tensor(image_bytes)
        return decode_image(encoded, mode=read_mode)
    if image_path:
        resolved = _resolve_image_path(dataset_root, image_path)
        return read_image(str(resolved), mode=read_mode)
    raise ValueError("image struct must include bytes or path")


def _resolve_image_path(dataset_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    candidate = (dataset_root / path).resolve(strict=False)
    if candidate != dataset_root and dataset_root not in candidate.parents:
        raise ValueError(f"image path escapes dataset root {str(dataset_root)!r}: {image_path!r}")
    return candidate


def _bytes_to_uint8_tensor(value: bytes | bytearray | memoryview) -> Tensor:
    raw = value.tobytes() if isinstance(value, memoryview) else value
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)


def _infer_dataset_root(root: str | Path | None, parquet_files: Sequence[str | Path]) -> Path:
    if root is not None:
        return Path(root).resolve()
    first_parent = Path(parquet_files[0]).resolve().parent
    if first_parent.name == "data":
        return first_parent.parent
    return first_parent


def _parquet_sort_key(path: Path) -> tuple[str, int, str]:
    match = _SHARD_NAME.match(path.name)
    if match is None:
        return ("unknown", 0, path.name)
    return (match.group("split"), int(match.group("index")), path.name)


def _labels_to_list(labels: Sequence[int] | Tensor) -> list[int]:
    if isinstance(labels, Tensor):
        if labels.ndim != 1:
            raise ValueError("labels tensor must be one-dimensional")
        return [int(value) for value in labels.detach().cpu().tolist()]
    return [int(value) for value in labels]


def _indices_tensor(indices: Sequence[int] | Tensor) -> Tensor:
    if isinstance(indices, Tensor):
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices tensor must be one-dimensional torch.long")
        return indices.detach().cpu()
    if not indices:
        raise ValueError("indices must be non-empty")
    return torch.tensor(list(indices), dtype=torch.long)


def _normalize_index(index: int, size: int) -> int:
    normalized = int(index)
    if normalized < 0:
        normalized += size
    if normalized < 0 or normalized >= size:
        raise IndexError(f"index {index} is out of range for dataset of size {size}")
    return normalized


def _pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(_PYARROW_IMPORT_ERROR) from exc
    return pq
