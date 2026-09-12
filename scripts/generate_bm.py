from pathlib import Path

import matplotlib.pyplot as plt
import torch

from boltzmann_machine.bm import BoltzmannMachine

IMAGE_SIZE = 28
NUM_UNITS = IMAGE_SIZE * IMAGE_SIZE
SEED = 42

NUM_SAMPLES = 16
GIBBS_SWEEPS = 1000


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    torch.manual_seed(SEED)
    device = get_device()
    model = BoltzmannMachine(
        num_units=NUM_UNITS,
        device=device,
    )

    model.load_state_dict(
        torch.load(
            "checkpoints/bm_mnist.pt",
            map_location=device,
            weights_only=True,
        )
    )

    initial_state = torch.bernoulli(
        torch.full((NUM_SAMPLES, NUM_UNITS), 0.5, device=device)
    )
    samples = model.sample(initial_state, sweeps=GIBBS_SWEEPS)
    samples = samples.reshape(NUM_SAMPLES, IMAGE_SIZE, IMAGE_SIZE).cpu()

    Path("outputs").mkdir(exist_ok=True)
    _, axes = plt.subplots(4, 4, figsize=(6, 6))

    for image, ax in zip(samples, axes.flat):
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/bm_generated.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
