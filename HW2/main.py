"""Train the CIFAR-10 CNN and save one experiment's metrics."""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from data import CIFAR10_MEAN, CIFAR10_STD, CLASSES, get_dataloaders
from model import CIFAR10CNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train for one epoch if optimizer is given; otherwise evaluate."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total_correct = total_images = 0

    with torch.enable_grad() if training else torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_images += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_images, total_correct / total_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    return parser.parse_args()


def save_random_predictions(model, dataset, device, output_path: Path) -> None:
    """Save predictions for ten randomly selected, normalized test images."""
    model.eval()
    indices = random.sample(range(len(dataset)), k=10)
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    with torch.no_grad():
        for axis, index in zip(axes.flat, indices):
            image, label = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)
            prediction = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, prediction].item()
            display_image = (image * std + mean).clamp(0, 1)
            axis.imshow(display_image.permute(1, 2, 0))
            axis.set_title(
                f"True: {CLASSES[label]}\n"
                f"Pred: {CLASSES[prediction]} ({confidence:.1%})"
            )
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size, num_workers=args.num_workers)
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best_test_accuracy = [], -1.0
    best_epoch, epochs_without_improvement = 0, 0
    training_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "test_loss": test_loss, "test_accuracy": test_accuracy})
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss: {train_loss:.4f}, "
              f"train acc: {train_accuracy:.2%} | test loss: {test_loss:.4f}, "
              f"test acc: {test_accuracy:.2%}")
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}: best test accuracy "
                      f"{best_test_accuracy:.2%} was at epoch {best_epoch}.")
                break

    training_time_seconds = time.perf_counter() - training_start_time
    summary = {
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "epochs_run": len(history),
        "early_stopping_patience": args.early_stopping_patience,
        "training_time_seconds": training_time_seconds,
    }

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", weights_only=True))
    save_random_predictions(
        model, test_loader.dataset, device, args.output_dir / "random_predictions.png"
    )
    print(f"Best test accuracy: {best_test_accuracy:.2%} at epoch {best_epoch}")
    print(f"Training time: {training_time_seconds:.1f}s")
    print(f"Saved results to: {args.output_dir.resolve()}")

    # Plot learning curve
    history = json.loads((args.output_dir / "history.json").read_text(encoding="utf-8"))
    epochs = [e["epoch"] for e in history]
    train_loss = [e["train_loss"] for e in history]
    test_loss = [e["test_loss"] for e in history]
    train_acc = [e["train_accuracy"] for e in history]
    test_acc = [e["test_accuracy"] for e in history]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(epochs, train_loss, label="train loss")
    axs[0].plot(epochs, test_loss, label="test loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].plot(epochs, train_acc, label="train accuracy")
    axs[1].plot(epochs, test_acc, label="test accuracy")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "learning_curve.png", dpi=150)
    plt.close(fig)
    print("Saved learning curve to:", args.output_dir / "learning_curve.png")


if __name__ == "__main__":
    main()
