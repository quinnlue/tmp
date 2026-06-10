"""One-shot converter: rewrite class_weight_movement.ipynb for CE classification.

Replaces the cells that change structurally and applies targeted string fixes to
the rest, then writes the notebook back. Deleted after use.
"""
import json

NB = "class_weight_movement.ipynb"
data = json.load(open(NB, encoding="utf-8"))
cells = data["cells"]


def setsrc(i, src):
    cells[i]["source"] = src


def replace_in(i, old, new):
    s = "".join(cells[i]["source"])
    assert old in s, f"cell {i}: missing {old!r}"
    cells[i]["source"] = s.replace(old, new)


CELL2 = r'''from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ViTConfig, VisionTransformerClassifier
from functional_train import SmoothAdamWConfig, initialize_train_state
from metagrad import InnerBatch, ObjectiveBatch
from replay import ReplayCheckpointConfig, replay_metagradient


@dataclass(frozen=True)
class ExperimentConfig:
    # --- reproducibility / hardware -------------------------------------------------
    seed: int = 0
    device: str = "auto"         # "auto" (CUDA if available) | "cpu" | "cuda"
    replay_tf32: bool = False     # keep disabled: strict IEEE FP32 matmuls are required for stability

    # --- dataset + class subset (the clusters / D' basis) ---------------------------
    dataset_name: str = "benjamin-paine/imagenet-1k-128x128"
    offline: bool = False         # read only the local HF cache (no network)
    train_split: str = "train"          # training-pool images come from here
    objective_split: str = "validation"  # held-out c* objective images (disjoint -> no leakage)
    # Keep only labels [0, num_labels) from each split.
    num_labels: int = 1000
    target: object = 0           # c* must be one of the retained labels
    include_target_in_pool: bool = True  # required: the CE objective on c* needs an output node for c*
    images_per_class: int = 100   # metagrad pool: 1000 x 100 @ 64px ~ 4.6 GiB (lives in host RAM)
    num_objective_images: int = 50
    keep_metagrad_pool_on_device: bool = False  # host pool -> ~13 GiB VRAM for REPLAY; per-batch PCIe copies are negligible vs compute

    # --- model (ViT classifier; mean-pooled patch tokens -> num_groups logits) -------
    image_size: int = 64         # ImageNet resized to this; 64px/patch8 -> 64 patches
    patch_size: int = 8
    encoder_dim: int = 256
    encoder_depth: int = 12
    encoder_heads: int = 8
    mlp_ratio: float = 2.0

    # --- inner training trajectory (the differentiable A) ---------------------------
    inner_steps: int = 3_000     # T: ~7.7 epochs over the 100k pool; solid inner convergence within the L4 budget
    batch_size: int = 256        # L4-sized differentiable batch; lower if backward-over-backward OOMs

    # --- inner optimizer (differentiable smooth AdamW) ------------------------------
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-4
    weight_decay: float = 0.0

    # --- meta-optimization: theta <- theta - alpha_k * sign(grad) -------------------
    temperature: float = 0.5     # tau
    meta_steps: int = 20         # weights plateau by ~step 18 with alpha_decay=0.90 (within the L4 budget)
    alpha0: float = 0.25         # initial signed step
    alpha_decay: float = 0.90    # faster geometric anneal so sign-descent settles inside meta_steps
    branching_factor: int = 24   # measured ~6.2 GiB peak checkpoint states at T=3000; low recompute
    replay_checkpoint_backend: str = "memory"  # L4 24 GB VRAM can retain the lazy tree without disk I/O
    replay_checkpoint_directory: str | None = None
    replay_checkpoint_interval: int | None = None

    # --- deep training runs (strong standard recipe; NOT the differentiable loop) ----
    deep_steps: int = 10_000     # optimizer steps for EACH weighted/unweighted deep run
    deep_batch: int = 256         # conservative FP32 standard-training batch for the L4
    deep_lr: float = 1e-3        # peak LR (linear warmup -> cosine decay to eta_min)
    warmup_steps: int = 500
    eta_min: float = 1e-5
    deep_weight_decay: float = 0.05
    deep_betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 250        # held-out phi cadence
    image_every: int = 1000      # label-histogram / accuracy cadence
    num_workers: int = 8         # tune to the VM's vCPU count and image-decoding throughput
    sampler_candidate_batch: int = 8192  # bounded label reads for lazy rejection sampling

    # --- Weights & Biases logging ---------------------------------------------------
    wandb_project: str = "metagrad-cluster-curation"
    wandb_entity: object = None  # str | None (your team/user; None = default)
    wandb_mode: str = "online"   # "online" | "offline" | "disabled"
    wandb_group: object = None   # str | None (None -> timestamped group)


cfg = ExperimentConfig()
replay_checkpoint_config = ReplayCheckpointConfig(
    backend=cfg.replay_checkpoint_backend,
    directory=cfg.replay_checkpoint_directory,
    interval_steps=cfg.replay_checkpoint_interval,
)
cfg'''


CELL6 = r'''if cfg.offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import load_dataset
from torchvision.transforms import v2

ds = load_dataset(cfg.dataset_name, keep_in_memory=False)
class_names = ds[cfg.train_split].features["label"].names
short = lambda c: class_names[c].split(",")[0]


def resolve_spec(spec) -> int:
    """An int index, or a case-insensitive substring that uniquely matches one class name."""
    if isinstance(spec, (int, np.integer)):
        i = int(spec)
        if not 0 <= i < len(class_names):
            raise ValueError(f"class index {i} out of range 0..{len(class_names) - 1}")
        return i
    matches = [i for i, n in enumerate(class_names) if str(spec).lower() in n.lower()]
    if not matches:
        raise ValueError(f"no class name contains {spec!r}")
    if len(matches) > 1:
        opts = ", ".join(f"{i}:{short(i)}" for i in matches)
        raise ValueError(f"{spec!r} is ambiguous -> {opts}  (use an index or a longer substring)")
    return matches[0]


target_class = resolve_spec(cfg.target)
if not 0 <= target_class < cfg.num_labels:
    raise ValueError(f"target label {target_class} must be in [0, {cfg.num_labels})")

# The cross-entropy objective scores held-out c* images, so c* must be one of the
# classifier's output classes -- i.e. it must stay in the pool.
if not cfg.include_target_in_pool:
    raise ValueError("classification objective requires include_target_in_pool=True")

pool_classes = list(range(cfg.num_labels))
num_groups = len(pool_classes)
assert num_groups >= 2, "need at least two retained labels"
pool_index = {c: j for j, c in enumerate(pool_classes)}

# The classifier predicts one logit per cluster/class in the pool.
mae_config = ViTConfig(
    image_size=cfg.image_size, patch_size=cfg.patch_size,
    encoder_dim=cfg.encoder_dim, encoder_depth=cfg.encoder_depth, encoder_heads=cfg.encoder_heads,
    mlp_ratio=cfg.mlp_ratio, num_classes=num_groups,
)

image_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((cfg.image_size, cfg.image_size), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])


def find_first_indices(split: str, counts_by_label: dict[int, int]) -> dict[int, list[int]]:
    """Scan the label-only Arrow view, retaining only the bounded set of requested row indices."""
    found = {label: [] for label in counts_by_label}
    remaining = sum(counts_by_label.values())
    label_rows = ds[split].select_columns(["label"])
    for row_index, row in enumerate(label_rows):
        label = int(row["label"])
        if label in found and len(found[label]) < counts_by_label[label]:
            found[label].append(row_index)
            remaining -= 1
            if remaining == 0:
                break
    if remaining:
        missing = {label: counts_by_label[label] - len(rows) for label, rows in found.items()
                   if len(rows) < counts_by_label[label]}
        raise ValueError(f"split {split!r} is missing requested examples: {missing}")
    return found


def decode_indices(split: str, rows: list[int]) -> torch.Tensor:
    """Decode only explicitly selected rows; never materialize an entire split."""
    return torch.stack([image_transform(ds[split][row]["image"].convert("RGB")) for row in rows])


train_rows = find_first_indices(
    cfg.train_split, {class_id: cfg.images_per_class for class_id in pool_classes}
)
images_list, group_list = [], []
for c in pool_classes:
    images_list.append(decode_indices(cfg.train_split, train_rows[c]))
    group_list += [pool_index[c]] * cfg.images_per_class
# The current L4 profile keeps the ~2.5 GB metagrad pool on-device to avoid repeating the same
# host-to-device copies during REPLAY recomputation. Disable the knob for smaller GPUs.
pool_device = device if cfg.keep_metagrad_pool_on_device else torch.device("cpu")
training_images = torch.cat(images_list).to(device=pool_device, dtype=torch_dtype)
# Clusters coincide with class labels, so the cluster id is also the classification target.
group_ids = torch.tensor(group_list, dtype=torch.long, device=pool_device)
labels = group_ids
base_group_masses = torch.full((num_groups,), 1.0 / num_groups, dtype=torch_dtype, device=device)

objective_rows = find_first_indices(cfg.objective_split, {target_class: cfg.num_objective_images})
objective_cpu = decode_indices(cfg.objective_split, objective_rows[target_class])
objective_images = objective_cpu.to(device=device, dtype=torch_dtype)
# Every held-out objective image carries the target label c*; phi = mean CE over them.
objective_labels = torch.full(
    (cfg.num_objective_images,), pool_index[target_class], dtype=torch.long, device=device
)
objective_batch = ObjectiveBatch(objective_images, objective_labels)
assert training_images.dtype == objective_batch.images.dtype == torch.float32'''


CELL11 = r'''model = VisionTransformerClassifier(mae_config).to(device=device, dtype=torch_dtype)
initial_state = initialize_train_state(model)
assert all(value.dtype == torch.float32 for value in initial_state.parameters.values())
assert all(value.dtype == torch.float32 for value in initial_state.first_moments.values())
assert all(value.dtype == torch.float32 for value in initial_state.second_moments.values())

# Minibatch inner trajectory: one fixed, seeded shuffle of the whole pool, then one slice
# per inner step. The L4 profile keeps the pool on GPU; smaller-GPU profiles can keep it on CPU.
# Only the current minibatch enters the differentiable step, so the full T-step trajectory is
# never materialized. The shuffle is deterministic (seeded CPU generator) and `trajectory[t]`
# is identical on the forward pass and on every REPLAY recompute, so the determinism GATE
# still holds. With no MAE mask, the inner loop has no per-step randomness at all -- determinism
# now rests purely on the fixed data order.
N = training_images.shape[0]
shuffle = torch.Generator().manual_seed(cfg.seed)
perm = torch.randperm(N, generator=shuffle)          # global indices, fixed for the whole run


def make_batch(t: int) -> InnerBatch:
    start = (t * cfg.batch_size) % N
    idx = perm[start:start + cfg.batch_size]
    if idx.numel() < cfg.batch_size:                 # wrap the tail -> keep batch size fixed
        idx = torch.cat([idx, perm[:cfg.batch_size - idx.numel()]])
    image_idx = idx.to(training_images.device)
    group_idx = idx.to(group_ids.device)
    batch_groups = group_ids.index_select(0, group_idx).to(device)
    return InnerBatch(
        training_images.index_select(0, image_idx).to(device=device, dtype=torch_dtype),
        batch_groups,   # labels == cluster id (clusters are class labels)
        batch_groups,
    )



class DeterministicInnerTrajectory:
    """Rebuild fixed minibatches on demand instead of retaining all T batches in RAM/VRAM."""

    def __len__(self):
        return cfg.inner_steps

    def __getitem__(self, step):
        if not isinstance(step, int) or not 0 <= step < len(self):
            raise IndexError(step)
        return make_batch(step)


trajectory = DeterministicInnerTrajectory()
optimizer_config = SmoothAdamWConfig(
    learning_rate=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
    eps=cfg.eps, weight_decay=cfg.weight_decay,
)
tree_depth = math.ceil(math.log(cfg.inner_steps + 1, cfg.branching_factor))
views = cfg.inner_steps * cfg.batch_size
state_bytes = 3 * sum(p.numel() * p.element_size() for p in model.parameters())
checkpoint_bound = 1 + (cfg.branching_factor - 1) * tree_depth
print(f"model params = {sum(p.numel() for p in model.parameters()):,}  |  T = {cfg.inner_steps} inner steps")
print(f"batch_size = {cfg.batch_size}  ->  {views:,} example-views over the run "
      f"({views / N:.2f} epochs of the {N:,}-image pool)")
print(f"REPLAY branching factor = {cfg.branching_factor}  ->  tree depth {tree_depth}")
print(f"conservative checkpoint-state bound = {checkpoint_bound} states / "
      f"{checkpoint_bound * state_bytes / 2**30:.1f} GiB")'''


CELL23 = r'''import time
from torch.utils.data import Dataset, DataLoader, Sampler
from model import per_example_cross_entropy_loss, cross_entropy_loss
from weighting import weighted_example_loss


# ---- standard map-style dataset over disk-backed Arrow shards (decoded per batch) ----------------
class ImagenetImages(Dataset):
    def __init__(self, hf_split, transform):
        self.split, self.transform = hf_split, transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        ex = self.split[int(i)]
        return self.transform(ex["image"].convert("RGB")), int(ex["label"])


deep_dataset = ImagenetImages(ds[cfg.train_split], image_transform)
deep_labels = ds[cfg.train_split].select_columns(["label"])

# label -> cluster column (identity when every class is in the pool); -1 marks a dropped class.
label_to_group = torch.full((len(class_names),), -1, dtype=torch.long)
for _j, _c in enumerate(pool_classes):
    label_to_group[_c] = _j


class LazyClassSampler(Sampler):
    """Sample map-style rows lazily using bounded label reads and rejection sampling."""

    def __init__(self, labels, class_weights, num_samples, seed, candidate_batch):
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device="cpu")
        if weights.shape != (len(class_names),) or not torch.isfinite(weights).all():
            raise ValueError("class_weights must be one finite weight per dataset class")
        if (weights < 0).any() or not (weights > 0).any():
            raise ValueError("class_weights must be nonnegative with at least one positive entry")
        if int(num_samples) < 1 or int(candidate_batch) < 1:
            raise ValueError("num_samples and candidate_batch must be positive")
        self.labels = labels
        self.acceptance = weights / weights.max()
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.candidate_batch = int(candidate_batch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        yielded = 0
        while yielded < self.num_samples:
            candidates = torch.randint(len(self.labels), (self.candidate_batch,), generator=generator)
            labels = torch.as_tensor(self.labels[candidates.tolist()]["label"], dtype=torch.long)
            keep = torch.rand(self.candidate_batch, generator=generator) < self.acceptance[labels]
            for index in candidates[keep].tolist():
                yield index
                yielded += 1
                if yielded == self.num_samples:
                    return

DEEP = dict(steps=cfg.deep_steps, batch=cfg.deep_batch, lr=cfg.deep_lr, warmup=cfg.warmup_steps,
            weight_decay=cfg.deep_weight_decay, betas=tuple(cfg.deep_betas), grad_clip=cfg.grad_clip,
            eval_every=cfg.eval_every, image_every=cfg.image_every, num_workers=cfg.num_workers,
            sampler_candidate_batch=cfg.sampler_candidate_batch,
            eta_min=cfg.eta_min)


def _cosine_lr(step):
    if step < DEEP["warmup"]:
        return DEEP["lr"] * (step + 1) / DEEP["warmup"]
    prog = (step - DEEP["warmup"]) / max(1, DEEP["steps"] - DEEP["warmup"])
    return DEEP["eta_min"] + 0.5 * (DEEP["lr"] - DEEP["eta_min"]) * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def _eval_phi(net):
    net.eval()
    val = cross_entropy_loss(net(objective_batch.images), objective_batch.labels).item()
    net.train()
    return val


@torch.no_grad()
def _eval_target_accuracy(net):
    net.eval()
    pred = net(objective_batch.images).argmax(dim=1)
    acc = (pred == objective_batch.labels).float().mean().item()
    net.train()
    return acc


def train_strong(run_name, class_sampling_weights, loss_logits, seed, *, weighted_sampling):
    """Strong from-scratch ViT-classifier training through a finite, lazy, map-style DataLoader."""
    torch.manual_seed(seed)
    net = VisionTransformerClassifier(mae_config).to(device=device, dtype=torch_dtype)
    assert all(parameter.dtype == torch.float32 for parameter in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=DEEP["lr"], betas=DEEP["betas"],
                            weight_decay=DEEP["weight_decay"])
    sampler = LazyClassSampler(
        deep_labels, class_sampling_weights, DEEP["steps"] * DEEP["batch"], seed,
        DEEP["sampler_candidate_batch"],
    )
    loader = DataLoader(deep_dataset, batch_size=DEEP["batch"], sampler=sampler,
                        num_workers=DEEP["num_workers"], pin_memory=(device.type == "cuda"),
                        drop_last=True, persistent_workers=DEEP["num_workers"] > 0)
    run = start_run(run_name, {**DEEP, "run": run_name, "weighted_loss": loss_logits is not None,
                               "weighted_sampling": weighted_sampling,
                               "image_size": cfg.image_size, "patch_size": cfg.patch_size,
                               "num_groups": num_groups,
                               "target_name": short(target_class)}, job_type="deep-train")

    phi_curve, phi_steps = [], []
    seen, t0 = 0, time.time()
    net.train()
    for step, (images, labels) in enumerate(loader):
        images = images.to(device=device, dtype=torch_dtype, non_blocking=True)
        assert images.dtype == torch.float32
        gids = label_to_group[labels].to(device)
        lr = _cosine_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        logits = net(images)
        per_ex = per_example_cross_entropy_loss(logits, gids)
        if loss_logits is None:
            loss = per_ex.mean()
        else:
            loss = weighted_example_loss(per_ex, loss_logits, gids, base_group_masses, cfg.temperature)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), DEEP["grad_clip"])
        opt.step()
        seen += images.shape[0]

        log = {"train/loss": float(loss), "train/lr": lr, "train/grad_norm": float(gnorm),
               "train/images_per_sec": seen / max(1e-9, time.time() - t0)}
        if step % DEEP["eval_every"] == 0 or step == DEEP["steps"] - 1:
            val = _eval_phi(net); phi_curve.append(val); phi_steps.append(step)
            log["eval/phi"] = val
            log["eval/target_accuracy"] = _eval_target_accuracy(net)
        if _WANDB_ON and (step % DEEP["image_every"] == 0 or step == DEEP["steps"] - 1):
            log["sample/label_hist"] = wandb.Histogram(labels.float().numpy())
        run.log(log, step=step)
        if step % 500 == 0 or step == DEEP["steps"] - 1:
            print(f"[{run_name}] step {step:>5}/{DEEP['steps']}  loss {float(loss):.4f}  "
                  f"lr {lr:.2e}  phi {phi_curve[-1]:.5f}", flush=True)

    run.summary.update(dict(phi_initial=phi_curve[0], phi_final=phi_curve[-1],
                            phi_delta=phi_curve[-1] - phi_curve[0]))
    run.finish()
    return dict(phi_curve=phi_curve, phi_steps=phi_steps, net=net)'''


setsrc(2, CELL2)
setsrc(6, CELL6)
setsrc(11, CELL11)
setsrc(23, CELL23)

# Targeted wording fixes elsewhere.
replace_in(13, "recon MSE", "cross-entropy (CE)")
replace_in(17, 'ylabel="phi (held-out MSE)"', 'ylabel="phi (held-out CE)"')
replace_in(17, f'title=f"Objective: held-out {{short(target_class)}} reconstruction"',
           'title=f"Objective: held-out {short(target_class)} classification (CE)"')
replace_in(27, "recon MSE", "CE")

json.dump(data, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", NB)
