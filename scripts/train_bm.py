from pathlib import Path

import torch
from tqdm import tqdm

from boltzmann_machine.bm import BoltzmannMachine
from boltzmann_machine.data import get_bm_mnist_loader

IMAGE_SIZE = 28
NUM_UNITS = IMAGE_SIZE * IMAGE_SIZE
SEED = 42

BATCH_SIZE = 128
EPOCHS = 50

LEARNING_RATE = 0.05
GIBBS_SWEEPS = 1

NUM_SAMPLES = 5000


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    torch.manual_seed(SEED)
    device = get_device()

    print(f"device: {device}")

    loader = get_bm_mnist_loader(
        batch_size=BATCH_SIZE,
        max_samples=NUM_SAMPLES,
    )

    model = BoltzmannMachine(
        num_units=NUM_UNITS,
        device=device,
    )

    persistent_state = torch.bernoulli(
        torch.full(
            (BATCH_SIZE, NUM_UNITS),
            0.5,
            device=device,
        )
    )

    for epoch in range(EPOCHS):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{EPOCHS}")

        for images, _ in progress:
            positive_state = (images.to(device).flatten(start_dim=1) > 0.5).float()

            for _ in range(GIBBS_SWEEPS):
                persistent_state = model.gibbs_sweep(persistent_state)

            negative_state = persistent_state

            model.update_parameters(
                positive_state=positive_state,
                negative_state=negative_state,
                learning_rate=LEARNING_RATE,
            )

            mean_error = torch.mean(
                torch.abs(positive_state.mean(dim=0) - negative_state.mean(dim=0))
            )

            progress.set_postfix(mean=f"{mean_error.item():.4f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/bm_mnist.pt")

    print("saved: checkpoints/bm_mnist.pt")


if __name__ == "__main__":
    main()
