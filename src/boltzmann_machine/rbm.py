import torch
from torch import nn


class RestrictedBoltzmannMachine(nn.Module):
    """Binary restricted Boltzmann machine with visible and hidden layers."""

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        device: torch.device | str = "cpu",
    ):
        super().__init__()

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.device = torch.device(device)

        self.W = nn.Parameter(
            0.01 * torch.randn(num_visible, num_hidden, device=self.device),
            requires_grad=False,
        )
        self.visible_bias = nn.Parameter(
            torch.zeros(num_visible, device=self.device),
            requires_grad=False,
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(num_hidden, device=self.device),
            requires_grad=False,
        )

    def energy(
        self,
        visible: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        interaction = -torch.sum((visible @ self.W) * hidden, dim=1)
        visible_energy = -torch.sum(visible * self.visible_bias, dim=1)
        hidden_energy = -torch.sum(hidden * self.hidden_bias, dim=1)

        return interaction + visible_energy + hidden_energy

    def hidden_probability(self, visible: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(visible @ self.W + self.hidden_bias)

    def visible_probability(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(hidden @ self.W.T + self.visible_bias)

    @torch.no_grad()
    def sample_hidden(self, visible: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(self.hidden_probability(visible))

    @torch.no_grad()
    def sample_visible(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(self.visible_probability(hidden))

    @torch.no_grad()
    def gibbs_sweep(self, visible: torch.Tensor) -> torch.Tensor:
        return self.sample_visible(self.sample_hidden(visible))

    @torch.no_grad()
    def sample(
        self,
        initial_state: torch.Tensor,
        sweeps: int = 100,
    ) -> torch.Tensor:
        visible = initial_state.clone()

        for _ in range(sweeps):
            visible = self.gibbs_sweep(visible)

        return visible

    @torch.no_grad()
    def update_parameters(
        self,
        positive_state: torch.Tensor,
        negative_state: torch.Tensor,
        learning_rate: float,
    ) -> None:
        batch_size = positive_state.shape[0]
        positive_hidden = self.hidden_probability(positive_state)
        negative_hidden = self.hidden_probability(negative_state)

        positive_correlation = positive_state.T @ positive_hidden / batch_size
        negative_correlation = negative_state.T @ negative_hidden / batch_size

        self.W.add_(learning_rate * (positive_correlation - negative_correlation))
        self.W.clamp_(-1.0, 1.0)
        self.visible_bias.add_(
            learning_rate * (positive_state.mean(dim=0) - negative_state.mean(dim=0))
        )
        self.hidden_bias.add_(
            learning_rate * (positive_hidden.mean(dim=0) - negative_hidden.mean(dim=0))
        )
