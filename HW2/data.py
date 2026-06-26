"""CIFAR-10 dataset loading and preprocessing utilities."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# CIFAR-10's RGB channel means and standard deviations, estimated from its training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    batch_size: int,
    data_dir: str | Path = "data",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 if needed and return training and test DataLoaders."""
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
