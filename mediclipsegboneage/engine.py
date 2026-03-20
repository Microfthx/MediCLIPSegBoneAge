import math
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))


def compute_loss(predictions, targets):
    l1 = F.l1_loss(predictions, targets)
    mse = F.mse_loss(predictions, targets)
    return l1 + 0.1 * mse


def run_train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        genders = batch["gender"].to(device)
        targets = batch["target"].to(device)
        prompts = batch["prompt"]

        optimizer.zero_grad()
        predictions = model(images, prompts, genders)
        loss = compute_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_mae += compute_mae(predictions.detach(), targets).item() * batch_size
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "mae": running_mae / max(total, 1),
    }


@torch.no_grad()
def run_eval_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        genders = batch["gender"].to(device)
        targets = batch["target"].to(device)
        prompts = batch["prompt"]

        predictions = model(images, prompts, genders)
        loss = compute_loss(predictions, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_mae += compute_mae(predictions, targets).item() * batch_size
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "mae": running_mae / max(total, 1),
    }


def save_checkpoint(state, output_dir, filename):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path / filename)


def format_metrics(prefix, metrics):
    return f"{prefix} loss={metrics['loss']:.4f} mae={metrics['mae']:.4f}"


def is_better_metric(current, best):
    if best is None:
        return True
    if math.isnan(best):
        return True
    return current < best
