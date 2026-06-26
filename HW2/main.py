"""Train the CIFAR-10 CNN and save one experiment's metrics."""

import argparse
import json
import random
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
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", weights_only=True))
    save_random_predictions(
        model, test_loader.dataset, device, args.output_dir / "random_predictions.png"
    )
    print(f"Best test accuracy: {best_test_accuracy:.2%}")
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
