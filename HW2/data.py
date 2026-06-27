"""CIFAR-10 dataset loading and preprocessing utilities."""

from pathlib import Path
import ssl
import sys
import urllib.request

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# CIFAR-10's RGB channel means and standard deviations, estimated from its training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _configure_https_certificates() -> None:
    """Use certifi on Windows to avoid broken local certificate-store entries."""
    if sys.platform != "win32":
        return

    try:
        import certifi
    except ImportError:
        return

    context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = (
        lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
    )
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    )


def get_dataloaders(
    batch_size: int,
    data_dir: str | Path = "data",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 if needed and return training and test DataLoaders."""
    _configure_https_certificates()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def _smoke_test_dataset() -> None:
    """Download CIFAR-10 and verify that labels/classes load correctly."""
    data_dir = Path(__file__).resolve().parent / "data"
    train_loader, test_loader = get_dataloaders(
        batch_size=4,
        data_dir=data_dir,
        num_workers=0,
    )

    train_set = train_loader.dataset
    test_set = test_loader.dataset
    images, labels = next(iter(train_loader))

    assert len(train_set) == 50_000, f"Expected 50000 training images, got {len(train_set)}"
    assert len(test_set) == 10_000, f"Expected 10000 test images, got {len(test_set)}"
    assert len(CLASSES) == 10, f"Expected 10 display classes, got {len(CLASSES)}"
    assert torch.all((0 <= labels) & (labels < len(CLASSES))), "Batch labels are out of range"

    print("CIFAR-10 dataset smoke test passed.")
    print(f"Dataset files: {data_dir.resolve()}")
    print(f"Train images: {len(train_set)}")
    print(f"Test images: {len(test_set)}")
    print(f"Display classes: {CLASSES}")
    print(f"Torchvision classes: {tuple(train_set.classes)}")
    print(f"Sample batch image shape: {tuple(images.shape)}")
    print(f"Sample batch labels: {[CLASSES[label] for label in labels.tolist()]}")


if __name__ == "__main__":
    _smoke_test_dataset()