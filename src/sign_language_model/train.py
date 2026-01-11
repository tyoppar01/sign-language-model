import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Assuming these imports exist in your local structure
from .dataset import WLASLDataset
from .evaluation import compute_metrics, plot_confusion_matrix
from .models import (
    BiLSTMKeypoints,
    I3DClassifier,
    LateFusionClassifier,
    RGBFlowClassifier,
    TransformerEncoderKeypoints,
)
from .utils import set_seed


# --- Helper: Robust Checkpoint Loading ---
def load_weights(model, ckpt_path, device):
    """Loads weights handling both raw state_dict and dict wrappers. Also freezezes model."""
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Check if checkpoint is a dictionary containing the state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        print("Attempting strict=False load...")
        model.load_state_dict(state_dict, strict=False)


# --- Forward Functions ---
def forward_kps(model, xb, device):
    return model(xb["kps"].to(device))


def forward_rgb(model, xb, device):
    return model(xb["rgb"].to(device))


def forward_rgb_flow(model, xb, device):
    inputs = torch.stack([xb["rgb"], xb["flow"]], dim=1).to(device)
    return model(inputs)


def forward_late_fusion(model, xb, device):
    kps = xb["kps"].to(device)
    rgb_flow = torch.stack([xb["rgb"], xb["flow"]], dim=1).to(device)

    return model(kps, rgb_flow)


# --- Training/Validation Loops ---
def train_one_epoch(
    model, loader, forward_fn, optimizer, criterion, device, grad_clip=1.0
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, leave=False, desc="Training")

    for xb, yb in pbar:
        optimizer.zero_grad(set_to_none=True)
        yb = yb.to(device)

        logits = forward_fn(model, xb, device)
        loss = criterion(logits, yb)

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        total_loss += loss.item() * yb.size(0)

        pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


def validate_one_epoch(model, loader, forward_fn, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, leave=False, desc="Validating")
        for xb, yb in pbar:
            yb = yb.to(device)

            logits = forward_fn(model, xb, device)
            loss = criterion(logits, yb)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

            correct += (preds == yb).sum().item()
            total += yb.size(0)
            total_loss += loss.item() * yb.size(0)

            pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return (
        total_loss / total,
        correct / total,
        torch.cat(all_probs),
        torch.cat(all_preds),
        torch.cat(all_labels),
    )


def main():
    parser = argparse.ArgumentParser("Train ASL classifier")

    # Data
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--gloss-map-path", type=str, required=True)
    parser.add_argument(
        "--modalities",
        choices=["kps", "rgb", "rgb+flow", "kps+rgb+flow"],
        required=True,
    )
    parser.add_argument(
        "--kp-model-type", choices=["transformer", "lstm"], default="transformer"
    )

    # Pretrained Models for Late Fusion
    parser.add_argument(
        "--pretrained-kps-ckpt",
        type=str,
        default=None,
        help="Path to pretrained Keypoint model (.pth)",
    )
    parser.add_argument(
        "--pretrained-rgb-flow-ckpt",
        type=str,
        default=None,
        help="Path to pretrained RGB+Flow model (.pth)",
    )

    # Augmentation
    parser.add_argument("--augment-kps", action="store_true", default=True)
    parser.add_argument("--augment-features", action="store_true", default=True)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience", type=int, default=20, help="ReduceLROnPlateau patience"
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    # Logging & Saving
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Only add the keypoint model type to the name if "kps" is involved
    kp_suffix = f"_{args.kp_model_type}" if "kps" in args.modalities else ""
    run_name = f"{args.modalities}{kp_suffix}_bs{args.batch_size}_lr{args.lr}_{int(time.time())}"

    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"Logging results to: {log_dir}")
    print(f"Saving checkpoints to: {ckpt_dir}")

    # Load datasets
    train_ds = WLASLDataset(
        args.train_npz,
        args.gloss_map_path,
        augment_kps=args.augment_kps,
        augment_features=args.augment_features,
    )
    test_ds = WLASLDataset(args.test_npz, args.gloss_map_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    num_classes = len(train_ds.index_to_label)

    # --- Model Initialization & Forward Selection ---
    forward_fn = None

    if args.modalities == "kps":
        forward_fn = forward_kps
        if args.kp_model_type == "transformer":
            model = TransformerEncoderKeypoints(num_classes=num_classes)
        elif args.kp_model_type == "lstm":
            model = BiLSTMKeypoints(num_classes=num_classes)

    elif args.modalities == "rgb":
        forward_fn = forward_rgb
        model = I3DClassifier(num_classes=num_classes)

    elif args.modalities == "rgb+flow":
        forward_fn = forward_rgb_flow
        model = RGBFlowClassifier(num_classes=num_classes)

    elif args.modalities == "kps+rgb+flow":
        forward_fn = forward_late_fusion

        print(f"\nInitializing Late Fusion with KPS model: {args.kp_model_type}")

        # 2. Init and Load Keypoint Model
        if args.kp_model_type == "transformer":
            kps_model = TransformerEncoderKeypoints(num_classes=num_classes)
        elif args.kp_model_type == "lstm":
            kps_model = BiLSTMKeypoints(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown KP model type: {args.kp_model_type}")

        if args.pretrained_kps_ckpt:
            load_weights(
                kps_model, args.pretrained_kps_ckpt, "cpu"
            )  # Load to CPU first before moving to device
            print(f"Loaded pretrained KPS model from {args.pretrained_kps_ckpt}")

        # 3. Init and Load RGB/Flow Model
        rgb_flow_model = RGBFlowClassifier(num_classes=num_classes)

        if args.pretrained_rgb_flow_ckpt:
            load_weights(rgb_flow_model, args.pretrained_rgb_flow_ckpt, "cpu")
            print(
                f"Loaded pretrained RGB+Flow model from {args.pretrained_rgb_flow_ckpt}"
            )

        # 4. Create Fusion Wrapper
        # Assuming LateFusionClassifier takes the instantiated sub-models
        model = LateFusionClassifier(
            num_classes=num_classes,
            kps_model=kps_model,
            rgb_flow_model=rgb_flow_model,
        )
    else:
        raise ValueError(f"Unknown modalities: {args.modalities}")

    model.to(device)

    # --- Log Model Graph ---
    try:
        writer.add_text("Model Architecture", str(model))
    except Exception as e:
        print(f"Could not log model graph: {e}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.modalities} | Params: {n_params:,} | Device: {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience // 2
    )

    # --- Training Loop ---
    best_test_acc = 0.0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, forward_fn, optimizer, criterion, device
        )

        test_loss, test_acc, _, _, _ = validate_one_epoch(
            model, test_loader, forward_fn, criterion, device
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Hyperparameters/LearningRate", current_lr, epoch)

        scheduler.step(test_loss)

        lr_info = ""
        if current_lr != prev_lr:
            lr_info = f" | LR: {prev_lr:.2e} -> {current_lr:.2e}"
            prev_lr = current_lr

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[Epoch {epoch:03d}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}{lr_info}"
            )

        # Save Best Model to ckpt_dir
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = ckpt_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_test_acc,
                },
                best_model_path,
            )
            # Optional: Log best acc update to console
            print(f"  New best accuracy! Saved to {best_model_path}")

    # --- Final Evaluation ---
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Load best model for final evaluation if it exists
    best_model_path = ckpt_dir / "best_model.pth"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path} for final metrics...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    _, final_acc, _, test_preds, test_labels = validate_one_epoch(
        model, test_loader, forward_fn, criterion, device
    )

    metrics = compute_metrics(test_labels, test_preds)

    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}")

    # Save outputs to Log Dir
    class_names = None
    if args.gloss_map_path:
        with open(args.gloss_map_path, "r") as f:
            gloss_map = json.load(f)
            idx2gloss = {v: k for k, v in gloss_map.items()}
            class_names = [idx2gloss.get(i, f"Class_{i}") for i in range(num_classes)]

    plot_confusion_matrix(
        test_labels,
        test_preds,
        class_names=class_names,
        title=f"CM - {args.modalities}",
        save_path=str(log_dir / "confusion_matrix.png"),
    )

    writer.add_hparams(
        {
            "lr": args.lr,
            "bs": args.batch_size,
            "modality": args.modalities,
            "kp_model": args.kp_model_type,
        },
        {"hparam/accuracy": final_acc, "hparam/f1_weighted": metrics["f1_weighted"]},
    )

    results = {
        "modalities": args.modalities,
        "kp_model_type": args.kp_model_type,
        "test_accuracy": float(final_acc),
        "best_test_accuracy": float(best_test_acc),
        "test_f1": float(metrics["f1_weighted"]),
        "train_losses": train_losses,
        "test_losses": test_losses,
    }

    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRun complete. Results: {log_dir} | Checkpoints: {ckpt_dir}")
    writer.close()


if __name__ == "__main__":
    main()
