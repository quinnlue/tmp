import json

NB = "train_tiny_classifier.ipynb"
data = json.load(open(NB, encoding="utf-8"))
cells = data["cells"]


def setsrc(i, src):
    cells[i]["source"] = src


setsrc(0, """# Train the Tiny ViT classifier on ImageNet

Train the root-level `model.py` ViT classifier on a small streamed subset of `benjamin-paine/imagenet-1k-128x128`. This is a smoke experiment: success means held-out cross-entropy drops and top-1 accuracy on a fixed validation batch improves.""")

setsrc(1, r'''from __future__ import annotations

import math
import random
from itertools import islice

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from torchvision.transforms import v2

from model import (
    ViTConfig,
    VisionTransformerClassifier,
    cross_entropy_loss,
)

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device''')

setsrc(2, """## Experiment setup

- Stream ImageNet instead of downloading the full multi-gigabyte dataset.
- Stream the entire shuffled training split exactly once, materializing only the current batch.
- Resize images from `128x128` to `64x64` for a quick laptop-friendly run.
- Train a ViT classifier with cross-entropy over all 1000 labels.
- Compare validation cross-entropy and top-1 accuracy before and after training on one fixed batch.""")

setsrc(3, r'''config = ViTConfig(
    image_size=64,
    patch_size=8,
    encoder_dim=192,
    encoder_depth=4,
    encoder_heads=6,
    mlp_ratio=4.0,
    num_classes=1000,
)
num_eval_images = 128
batch_size = 128
learning_rate = 1e-3
shuffle_buffer_size = 10_000

image_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((config.image_size, config.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def transform_examples(examples) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack(
        [image_transform(example["image"].convert("RGB")) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return images, labels


ds = load_dataset("benjamin-paine/imagenet-1k-128x128", streaming=True)
train_split = ds["train"].shuffle(seed=SEED, buffer_size=shuffle_buffer_size)
eval_examples = list(ds["validation"].shuffle(seed=SEED + 1, buffer_size=2048).take(num_eval_images))
eval_images, eval_labels = transform_examples(eval_examples)
class_names = ds["train"].features["label"].names
num_train_images = ds["train"].info.splits["train"].num_examples
steps_per_epoch = math.ceil(num_train_images / batch_size)

model = VisionTransformerClassifier(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=steps_per_epoch, eta_min=1e-5
)

print(
    f"device={device}, streamed train images={num_train_images:,}, validation={tuple(eval_images.shape)}, "
    f"steps={steps_per_epoch:,}, parameters={sum(p.numel() for p in model.parameters()):,}"
)''')

setsrc(4, r'''@torch.no_grad()
def evaluate() -> tuple[float, float, torch.Tensor]:
    model.eval()
    batch = eval_images.to(device)
    labels = eval_labels.to(device)
    logits = model(batch)
    loss = cross_entropy_loss(logits, labels)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean()
    return loss.item(), accuracy.item(), predictions.cpu()


initial_loss, initial_accuracy, _ = evaluate()
initial_loss, initial_accuracy''')

setsrc(7, r'''loss_history = []
step = 0
seen = 0
train_iterator = iter(train_split)

model.train()
with tqdm(total=steps_per_epoch, desc="one ImageNet epoch", unit="batch") as progress:
    while True:
        examples = list(islice(train_iterator, batch_size))
        if not examples:
            break

        batch, labels = transform_examples(examples)
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        seen += len(examples)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = cross_entropy_loss(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        step += 1
        progress.update(1)
        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

print(f"trained on {seen:,} images in {step:,} optimizer steps")''')

setsrc(8, r'''final_loss, final_accuracy, final_predictions = evaluate()
result = {
    "initial_validation_loss": initial_loss,
    "final_validation_loss": final_loss,
    "initial_accuracy": initial_accuracy,
    "final_accuracy": final_accuracy,
    "training_steps": step,
}
assert final_loss < initial_loss, "training did not improve fixed-batch validation cross-entropy"
result''')

setsrc(9, r'''plt.figure(figsize=(7, 3))
plt.plot(loss_history)
plt.xlabel("training step")
plt.ylabel("cross-entropy")
plt.title("Tiny ViT classifier training loss")
plt.grid(alpha=0.25)
plt.show()''')

setsrc(10, """## Inspect validation predictions

Show a few held-out images with their predicted and true class labels (green = correct).""")

setsrc(11, r'''fig, axes = plt.subplots(2, 4, figsize=(11, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(eval_images[i].permute(1, 2, 0))
    ax.axis("off")
    pred = int(final_predictions[i])
    true = int(eval_labels[i])
    colour = "green" if pred == true else "red"
    ax.set_title(f"pred: {class_names[pred][:18]}\ntrue: {class_names[true][:18]}",
                 fontsize=8, color=colour)
plt.tight_layout()
plt.show()''')

setsrc(12, """## Next steps

- Select a balanced subset of target, neighboring, and distractor ImageNet classes.
- Increase image resolution and model size after this smoke run is stable.
- Replace ordinary `AdamW.step()` with an explicit functional optimizer only when beginning metagradient work.""")

json.dump(data, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", NB)
