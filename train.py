import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from mediclipsegboneage.config import RSNAPaths
from mediclipsegboneage.dataset import RSNABoneAgePromptDataset
from mediclipsegboneage.engine import format_metrics
from mediclipsegboneage.engine import is_better_metric
from mediclipsegboneage.engine import run_eval_epoch
from mediclipsegboneage.engine import run_train_epoch
from mediclipsegboneage.engine import save_checkpoint
from mediclipsegboneage.model import MedCLIPBoneAgeRegressor
from mediclipsegboneage.utils import get_device
from mediclipsegboneage.utils import seed_everything


def build_parser():
    paths = RSNAPaths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=paths.image_dir)
    parser.add_argument("--train-csv", default=paths.train_csv)
    parser.add_argument("--val-csv", default=paths.val_csv)
    parser.add_argument("--output-dir", default="outputs/rsna_clip_baseline")
    parser.add_argument("--medclipseg-root", default=str(Path(__file__).resolve().parent.parent / "MedCLIPSeg"))
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = get_device(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "runs")) if SummaryWriter is not None else None

    train_dataset = RSNABoneAgePromptDataset(
        image_dir=args.image_dir,
        csv_path=args.train_csv,
        image_size=args.image_size,
        has_labels=True,
        train=True,
    )
    val_dataset = RSNABoneAgePromptDataset(
        image_dir=args.image_dir,
        csv_path=args.val_csv,
        image_size=args.image_size,
        has_labels=True,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MedCLIPBoneAgeRegressor(
        medclipseg_root=args.medclipseg_root,
        backbone=args.backbone,
        freeze_image_encoder=args.freeze_image_encoder,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history_path = output_dir / "history.csv"
    best_val_mae = None

    with history_path.open("w", encoding="utf-8") as fp:
        fp.write("epoch,train_loss,train_mae,val_loss,val_mae,lr\n")

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_train_epoch(model, train_loader, optimizer, device)
            val_metrics = run_eval_epoch(model, val_loader, device)
            scheduler.step(val_metrics["mae"])
            lr = optimizer.param_groups[0]["lr"]

            fp.write(
                f"{epoch},{train_metrics['loss']:.6f},{train_metrics['mae']:.6f},"
                f"{val_metrics['loss']:.6f},{val_metrics['mae']:.6f},{lr:.8f}\n"
            )
            fp.flush()

            if writer is not None:
                writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                writer.add_scalar("MAE/train", train_metrics["mae"], epoch)
                writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                writer.add_scalar("MAE/val", val_metrics["mae"], epoch)
                writer.add_scalar("LR", lr, epoch)
                writer.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"{format_metrics('train', train_metrics)} | "
                f"{format_metrics('val', val_metrics)} | lr={lr:.7f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "val_mae": val_metrics["mae"],
            }
            save_checkpoint(checkpoint, output_dir, "last.pt")

            if is_better_metric(val_metrics["mae"], best_val_mae):
                best_val_mae = val_metrics["mae"]
                save_checkpoint(checkpoint, output_dir, "best.pt")
                print(f"Saved new best checkpoint with val_mae={best_val_mae:.4f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
