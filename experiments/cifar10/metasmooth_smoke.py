"""Phase C de-risk: confirm the existing functional engine computes a metagradient
of held-out CE w.r.t. per-cluster data-weight logits z, using the smooth GroupNorm
ResNet-9.  Random data, CPU -- this only checks the autograd path is wired."""
import torch
import metasmooth as ms
from functional_train import SmoothAdamWConfig, initialize_train_state
from metagrad import InnerBatch, ObjectiveBatch, unrolled_objective

torch.manual_seed(0)
dev = "cpu"

for routine in (ms.SMOOTH_ROUTINE, ms.SMOOTH_GN_ROUTINE):
    model = ms.ResNet9(routine, num_classes=10).to(dev)
    model.train()
    nbuf = sum(b.numel() for b in model.buffers())
    print(f"{routine.name:10s} norm={routine.norm}  buffer_elems={nbuf}")

model = ms.ResNet9(ms.SMOOTH_GN_ROUTINE, num_classes=10).to(dev)
model.train()
state = initialize_train_state(model)

z = torch.zeros(10, requires_grad=True)          # per-cluster logits (R^10)
base = torch.full((10,), 0.1)                    # uniform base group masses
opt_cfg = SmoothAdamWConfig(learning_rate=1e-2)

B = 16
traj = []
for _ in range(3):
    imgs = torch.randn(B, 3, 32, 32)
    labels = torch.randint(0, 10, (B,))
    traj.append(InnerBatch(imgs, labels, labels.clone()))  # group = class
obj = ObjectiveBatch(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))

loss = unrolled_objective(model, state, traj, obj, z, base, opt_cfg,
                          temperature=1.0, create_graph=True)
(g,) = torch.autograd.grad(loss, z)
print(f"held-out CE objective = {float(loss):.4f}")
print(f"metagradient dphi/dz   = {g.tolist()}")
print(f"finite={torch.isfinite(g).all().item()}  nonzero={(g.abs().sum()>0).item()}")
