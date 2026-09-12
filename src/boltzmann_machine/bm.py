import torch
from torch import nn


class BoltzmannMachine(nn.Module):
    """Binary fully connected Boltzmann machine."""

    def __init__(
        self,
        num_units: int,
        device: torch.device | str = "cpu",
    ):
        super().__init__()

        self.num_units = num_units
        self.device = torch.device(device)

        self.W = nn.Parameter(
            torch.zeros(num_units, num_units, device=self.device),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(num_units, device=self.device),
            requires_grad=False,
        )

    def energy(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        interaction = -0.5 * torch.sum((state @ self.W) * state, dim=1)
        bias_energy = -torch.sum(state * self.bias, dim=1)

        return interaction + bias_energy

    def conditional_probability(
        self,
        state: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        return torch.sigmoid(state @ self.W[:, index] + self.bias[index])

    @torch.no_grad()
    def gibbs_sweep(self, state: torch.Tensor) -> torch.Tensor:
        state = state.clone()

        for index in range(self.num_units):
            state[:, index] = torch.bernoulli(
                self.conditional_probability(state, index)
            )

        return state

    @torch.no_grad()
    def sample(
        self,
        initial_state: torch.Tensor,
        sweeps: int = 100,
    ) -> torch.Tensor:
        state = initial_state.clone()

        for _ in range(sweeps):
            state = self.gibbs_sweep(state)

        return state

    @torch.no_grad()
    def update_parameters(
        self,
        positive_state: torch.Tensor,
        negative_state: torch.Tensor,
        learning_rate: float,
    ) -> None:
        batch_size = positive_state.shape[0]
        positive_correlation = positive_state.T @ positive_state / batch_size
        negative_correlation = negative_state.T @ negative_state / batch_size

        self.W.add_(learning_rate * (positive_correlation - negative_correlation))
        self.bias.add_(
            learning_rate * (positive_state.mean(dim=0) - negative_state.mean(dim=0))
        )
        self.W.copy_(0.5 * (self.W + self.W.T))
        self.W.clamp_(-1.0, 1.0)
        self.W.fill_diagonal_(0)
