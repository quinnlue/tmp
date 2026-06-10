import json


def load(nb):
    return json.load(open(nb, encoding="utf-8"))


def dump(nb, data):
    json.dump(data, open(nb, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", nb)


def setsrc(cells, i, src):
    cells[i]["source"] = src


def replace_in(cells, i, old, new):
    s = "".join(cells[i]["source"])
    assert old in s, f"cell {i}: missing {old!r}"
    cells[i]["source"] = s.replace(old, new)


# ---------------------------------------------------------------- replay_demo.ipynb
data = load("replay_demo.ipynb")
cells = data["cells"]

setsrc(cells, 1, r'''from __future__ import annotations

import math

import torch

from model import ViTConfig, VisionTransformerClassifier
from functional_train import SmoothAdamWConfig, initialize_train_state
from metagrad import InnerBatch, ObjectiveBatch, unrolled_objective
from replay import replay_metagradient

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)  # CPU float64 oracle -> crisp REPLAY == backprop match
device = torch.device("cpu")
print("torch", torch.__version__, "| device", device, "| dtype", torch.get_default_dtype())''')

setsrc(cells, 3, r'''config = ViTConfig(
    image_size=16, patch_size=4,
    encoder_dim=32, encoder_depth=2, encoder_heads=4,
    mlp_ratio=2.0, num_classes=2,
)
model = VisionTransformerClassifier(config).to(device)
initial_state = initialize_train_state(model)

# One shared low-frequency template; cluster 0 (class 0) and the objective are template + small
# jitter, while cluster 1 (class 1) is unstructured noise.
template_gen = torch.Generator().manual_seed(1234)
template = torch.nn.functional.interpolate(
    torch.rand(1, config.channels, 2, 2, generator=template_gen),
    size=config.image_size, mode="bilinear", align_corners=False,
)


def template_images(n, seed, jitter=0.03):
    gen = torch.Generator().manual_seed(seed)
    jit = torch.randn(n, config.channels, config.image_size, config.image_size, generator=gen)
    return (template.expand(n, config.channels, config.image_size, config.image_size) + jitter * jit).clamp(0, 1)


def noise_images(n, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, config.channels, config.image_size, config.image_size, generator=gen)


per_cluster = 4
training_images = torch.cat([template_images(per_cluster, 1), noise_images(per_cluster, 2)]).to(device)
group_ids = torch.tensor([0] * per_cluster + [1] * per_cluster)   # 0 = template, 1 = noise
labels = group_ids                                                # clusters are the class labels
base_group_masses = torch.tensor([0.5, 0.5])                      # even base distribution D
image_ids = torch.arange(per_cluster * 2)                         # stable candidate ids (paper section)

objective_images = template_images(4, 99).to(device)             # held-out, same template family
objective_labels = torch.zeros(4, dtype=torch.long)             # objective scores class 0 (template)
objective_batch = ObjectiveBatch(objective_images, objective_labels)

T = 20
# No MAE mask: with augmentation off the inner step is a pure function of the (fixed) batch, so
# every step reuses the same images/labels/groups and determinism rests on the fixed data order.
trajectory = tuple(
    InnerBatch(training_images, labels, group_ids)
    for _ in range(T)
)

optimizer_config = SmoothAdamWConfig(learning_rate=1.5e-2, betas=(0.9, 0.99), eps=1e-4, weight_decay=0.0)
temperature = 0.5

print(f"T={T} inner steps | clusters: {per_cluster} template + {per_cluster} noise | "
      f"objective = {objective_images.shape[0]} held-out template images")
print(f"model parameters = {sum(p.numel() for p in model.parameters()):,}")''')

setsrc(cells, 9, r'''print(f"{'T':>4}  {'#segments':>10}  {'max|delta grad|':>16}")
for T_try in (8, 16, 32, 64):
    traj = tuple(
        InnerBatch(training_images, labels, group_ids)
        for _ in range(T_try)
    )
    te = theta_values.clone().requires_grad_(True)
    (ge,) = torch.autograd.grad(
        unrolled_objective(model, initial_state, traj, objective_batch,
                           te, base_group_masses, optimizer_config, temperature, True),
        te,
    )
    tr = theta_values.clone().requires_grad_(True)
    _, gr = replay_metagradient(model, initial_state, traj, objective_batch,
                                tr, base_group_masses, optimizer_config, temperature,
                                branching_factor=3)
    depth = math.ceil(math.log(T_try + 1, 3))
    print(f"{T_try:>4}  {depth:>10}  {(gr - ge).abs().max().item():>16.2e}")''')

setsrc(cells, 14, r'''from paper_mgd import (
    CandidatePool,
    PaperMGDConfig,
    build_count_trajectory,
    initialize_counts,
    make_probe_batch,
    paper_mgd_outer_step,
    paper_replay_metagradient,
    paper_unrolled_objective,
)

paper_T = 8
paper_batch_size = training_images.shape[0]
perturbation_step = paper_T - 2
candidate_pool = CandidatePool(training_images, image_ids.to(torch.long), labels.to(torch.long))
initial_counts = initialize_counts(candidate_pool.num_candidates)
all_candidates = torch.arange(candidate_pool.num_candidates)

count_trajectory = build_count_trajectory(
    candidate_pool,
    initial_counts,
    inner_steps=paper_T,
    batch_size=paper_batch_size,
    shuffle_seed=101,
)
probe_batch = make_probe_batch(candidate_pool, all_candidates)

# The paper differentiates at z=0. First verify recursive REPLAY changes only memory scheduling.
z_unrolled = torch.zeros(candidate_pool.num_candidates, requires_grad=True)
phi_paper_unrolled = paper_unrolled_objective(
    model,
    initial_state,
    count_trajectory,
    probe_batch,
    objective_batch,
    perturbation_step,
    z_unrolled,
    optimizer_config,
    probe_chunk_size=4,
    create_graph=True,
)
(grad_paper_unrolled,) = torch.autograd.grad(phi_paper_unrolled, z_unrolled)

z_replay = torch.zeros(candidate_pool.num_candidates, requires_grad=True)
phi_paper_replay, grad_paper_replay = paper_replay_metagradient(
    model,
    initial_state,
    count_trajectory,
    probe_batch,
    objective_batch,
    perturbation_step,
    z_replay,
    optimizer_config,
    probe_chunk_size=4,
    branching_factor=3,
)

torch.testing.assert_close(phi_paper_replay, phi_paper_unrolled.detach(), rtol=rtol, atol=atol)
torch.testing.assert_close(grad_paper_replay, grad_paper_unrolled, rtol=rtol, atol=atol)

print(f"paper trajectory: T={paper_T}, batch={paper_batch_size}, perturbation step k={perturbation_step}")
print(f"fixed inner sample budget = {paper_T * paper_batch_size}")
print(f"paper surrogate at z=0: phi={phi_paper_replay.item():.8f}")
print("mean metagradient: template={:.3e}, noise={:.3e}".format(
    grad_paper_replay[:per_cluster].mean().item(),
    grad_paper_replay[per_cluster:].mean().item(),
))
print("PASS: paper count-MGD REPLAY == paper count-MGD explicit unrolled backprop")''')

setsrc(cells, 15, r'''def run_paper_count_policy(policy, outer_steps=3):
    counts = initial_counts.clone()
    config_paper = PaperMGDConfig(
        inner_steps=paper_T,
        batch_size=paper_batch_size,
        perturbation_step=perturbation_step,
        update_policy=policy,
        coordinate_fraction=1.0,
        exchange_fraction=0.25,
        shuffle_seed=101,
        selection_seed=303,
        probe_chunk_size=4,
        branching_factor=3,
    )
    rows = []
    for outer_step in range(outer_steps + 1):
        result = paper_mgd_outer_step(
            model,
            initial_state,
            candidate_pool,
            counts,
            objective_batch,
            optimizer_config,
            config_paper,
            outer_step=outer_step,
        )
        rows.append({
            "step": outer_step,
            "phi": result.objective.item(),
            "template_count": int(counts[:per_cluster].sum()),
            "noise_count": int(counts[per_cluster:].sum()),
            "total_count": int(counts.sum()),
            "counts": counts.tolist(),
        })
        counts = result.updated_counts
    return rows


projected_history = run_paper_count_policy("projected_sign")
fixed_budget_history = run_paper_count_policy("fixed_budget_ranked")

for name, rows in (
    ("projected_sign (Algorithm 1)", projected_history),
    ("fixed_budget_ranked (Appendix C.2)", fixed_budget_history),
):
    print(f"\n{name}")
    print(f"{'step':>4} {'phi':>11} {'template':>10} {'noise':>7} {'total':>7}  counts")
    for row in rows:
        print(
            f"{row['step']:>4} {row['phi']:>11.6f} {row['template_count']:>10} "
            f"{row['noise_count']:>7} {row['total_count']:>7}  {row['counts']}"
        )''')

replace_in(cells, 11, "phi (held-out MSE)", "phi (held-out CE)")
replace_in(cells, 12, "phi (held-out MSE)", "phi (held-out CE)")
replace_in(cells, 16, "phi (held-out MSE)", "phi (held-out CE)")
dump("replay_demo.ipynb", data)


# ---------------------------------------------------- paper_mgd_comparison.ipynb
data = load("paper_mgd_comparison.ipynb")
cells = data["cells"]

setsrc(cells, 2, r'''from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from functional_train import SmoothAdamWConfig, initialize_train_state
from metagrad import InnerBatch, ObjectiveBatch
from paper_mgd import CandidatePool, PaperMGDConfig, initialize_counts, paper_mgd_outer_step
from replay import replay_metagradient, replay_objective
from weighting import group_masses

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")
print("torch", torch.__version__, "| device", device, "| dtype", torch.get_default_dtype())''')

setsrc(cells, 3, r'''class ConstantLogitModel(nn.Module):
    """An image-agnostic two-class model that makes curation easy to inspect.

    Logits are a free parameter shared across the batch, so each candidate's effect
    on the objective is set entirely by its class label.
    """

    def __init__(self) -> None:
        super().__init__()
        self.class_logits = nn.Parameter(torch.zeros(2))

    def forward(self, images: Tensor) -> Tensor:
        return self.class_logits.unsqueeze(0).expand(images.shape[0], 2)


model = ConstantLogitModel().to(device)
initial_state = initialize_train_state(model)

# Candidates 0-1 are the target class (useful); candidates 2-3 are distractors.
useful_images = torch.ones(2, 3, 2, 2, device=device)
distractor_images = torch.zeros(2, 3, 2, 2, device=device)
candidate_images = torch.cat((useful_images, distractor_images))
candidate_ids = torch.tensor([10, 11, 12, 13])
candidate_labels = torch.tensor([0, 0, 1, 1])
pool = CandidatePool(candidate_images, candidate_ids, candidate_labels)

objective_batch = ObjectiveBatch(
    torch.ones(2, 3, 2, 2, device=device),
    torch.zeros(2, dtype=torch.long, device=device),
)
optimizer_config = SmoothAdamWConfig(
    learning_rate=0.1, betas=(0.8, 0.9), eps=0.1, weight_decay=0.0
)
inner_steps = 3
meta_steps = 4

print("candidate order: [useful, useful, distractor, distractor]")
print("fixed sample budget per outer step:", inner_steps * candidate_images.shape[0])''')

setsrc(cells, 4, r'''def persistent_trajectory(group_ids: Tensor) -> tuple[InnerBatch, ...]:
    return tuple(
        InnerBatch(candidate_images, candidate_labels, group_ids)
        for _ in range(inner_steps)
    )


def run_persistent_softmax(
    name: str,
    group_ids: Tensor,
    base_masses: Tensor,
    useful_groups: tuple[int, ...],
) -> list[dict]:
    logits = torch.zeros(base_masses.numel(), requires_grad=True)
    trajectory = persistent_trajectory(group_ids)
    history = []
    for step in range(meta_steps + 1):
        phi, gradient = replay_metagradient(
            model,
            initial_state,
            trajectory,
            objective_batch,
            logits,
            base_masses,
            optimizer_config,
            temperature=0.5,
            branching_factor=3,
        )
        masses = group_masses(logits.detach(), temperature=0.5)
        history.append(
            {
                "method": name,
                "step": step,
                "objective": phi.item(),
                "useful_mass": masses[list(useful_groups)].sum().item(),
                "total_count": 4,
                "distribution": [round(value, 3) for value in masses.tolist()],
            }
        )
        if step < meta_steps:
            logits = (logits.detach() - 0.25 * gradient.sign()).requires_grad_(True)
    return history


def run_count_mgd(name: str, policy: str) -> list[dict]:
    counts = initialize_counts(pool.num_candidates)
    config = PaperMGDConfig(
        inner_steps=inner_steps,
        batch_size=4,
        perturbation_step=1,
        update_policy=policy,
        coordinate_fraction=1.0,
        exchange_fraction=0.5,
        shuffle_seed=7,
        selection_seed=8,
        branching_factor=3,
    )
    history = []
    for step in range(meta_steps + 1):
        result = paper_mgd_outer_step(
            model,
            initial_state,
            pool,
            counts,
            objective_batch,
            optimizer_config,
            config,
            outer_step=step,
        )
        history.append(
            {
                "method": name,
                "step": step,
                "objective": result.objective.item(),
                "useful_mass": (counts[:2].sum() / counts.sum()).item(),
                "total_count": int(counts.sum()),
                "distribution": counts.tolist(),
            }
        )
        counts = result.updated_counts
    return history''')

replace_in(cells, 8, '"Held-out reconstruction objective"', '"Held-out classification objective"')
replace_in(cells, 8, '"MSE (lower is better)"', '"CE (lower is better)"')
dump("paper_mgd_comparison.ipynb", data)
