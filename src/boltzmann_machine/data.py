from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_bm_mnist_loader(
    batch_size=128,
    image_size=28,
    max_samples=None,
    digit=None,
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    indices = list(range(len(dataset)))

    if digit is not None:
        indices = [
            i for i, target in enumerate(dataset.targets) if target.item() == digit
        ]

    if max_samples is not None:
        indices = indices[:max_samples]

    dataset = Subset(
        dataset,
        indices,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
