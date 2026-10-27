"""How a ViewpointScorer learns from the rewards of the viewpoints its training episodes take: Q-regression
(Q-learning with discount 0) or policy gradient (REINFORCE). Training episodes take the objective's exploratory
choices; deployed, a policy takes the candidate of highest score under either objective."""

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class Learner(Protocol):
    """An objective's state during a training run."""

    def loss(self, scores: Tensor, taken: Tensor, advantage: Tensor) -> Tensor:
        """The loss to minimize, given the scores [N, num_candidates] at the decisions of a training step, the
        candidates taken [N] and their standardized rewards [N]."""
        ...

    def update(self, scores: Tensor) -> None:
        """Adapt to the scores [N, num_candidates] of a training step, for the next step's loss."""
        ...


def mean_entropy(log_probs: Tensor) -> Tensor:
    """The mean entropy, in nats, of the distributions log_probs [N, num_candidates]."""
    return -(log_probs.exp() * log_probs).sum(dim=1).mean()


@dataclass(frozen=True)
class QRegression:
    """Q-learning with discount 0: scores are action values, and the score of each viewpoint taken regresses on its
    standardized reward (mean squared error), with no credit from later glimpses. The scorer has a dueling head.
    Training episodes take a uniformly random candidate with probability epsilon, else the candidate of highest
    score."""

    epsilon: float = 0.5
    title: ClassVar[str] = "Q-regression"
    dueling: ClassVar[bool] = True

    def explore(self, scores: Tensor) -> Tensor:
        """[B, num_candidates] scores -> [B] candidates."""
        greedy = scores.argmax(dim=1)
        uniform = torch.randint(scores.shape[1], greedy.shape, device=scores.device)
        return torch.where(torch.rand(greedy.shape, device=scores.device) < self.epsilon, uniform, greedy)

    def learner(self) -> Learner:
        return self  # nothing adapts during training

    def loss(self, scores: Tensor, taken: Tensor, advantage: Tensor) -> Tensor:
        return F.mse_loss(scores.gather(1, taken.unsqueeze(1)).squeeze(1), advantage)

    def update(self, scores: Tensor) -> None:
        pass


@dataclass(frozen=True)
class PolicyGradient:
    """REINFORCE with discount 0 and an entropy bonus: scores are the logits of a softmax policy, which training
    episodes sample, and the loss is the score-function estimator with the standardized reward as advantage, minus
    the bonus. The bonus weight adapts by dual ascent, as Soft Actor-Critic tunes its temperature: it rises while the
    policy's mean entropy is below entropy_floor and falls otherwise, between min_entropy_weight and
    max_entropy_weight."""

    entropy_floor: float = 1.0
    """In nats."""
    min_entropy_weight: float = 0.01
    """The bonus weight's initial value and lower bound."""
    max_entropy_weight: float = 10.0
    entropy_weight_rate: float = 0.05
    """Step of the log weight per nat of entropy below the floor."""
    title: ClassVar[str] = "policy gradient"
    dueling: ClassVar[bool] = False

    def explore(self, scores: Tensor) -> Tensor:
        """[B, num_candidates] scores -> [B] candidates."""
        return torch.multinomial(scores.softmax(dim=1), 1).squeeze(1)

    def learner(self) -> Learner:
        return EntropyRegularizedPolicyGradient(self)


class EntropyRegularizedPolicyGradient:
    """PolicyGradient's learner: its state is the entropy bonus weight."""

    def __init__(self, objective: PolicyGradient) -> None:
        self.objective = objective
        # On the CPU until the first update moves it to the device of the scores.
        self.log_entropy_weight = torch.tensor(math.log(objective.min_entropy_weight))

    def loss(self, scores: Tensor, taken: Tensor, advantage: Tensor) -> Tensor:
        log_probs = scores.log_softmax(dim=1)
        policy_gradient = -(advantage * log_probs.gather(1, taken.unsqueeze(1)).squeeze(1)).mean()
        return policy_gradient - self.log_entropy_weight.exp() * mean_entropy(log_probs)

    @torch.no_grad()
    def update(self, scores: Tensor) -> None:
        objective = self.objective
        shortfall = objective.entropy_floor - mean_entropy(scores.log_softmax(dim=1))
        self.log_entropy_weight = (self.log_entropy_weight + objective.entropy_weight_rate * shortfall).clamp(
            math.log(objective.min_entropy_weight), math.log(objective.max_entropy_weight),
        )


Objective = QRegression | PolicyGradient
OBJECTIVES: dict[str, type[Objective]] = {objective.__name__: objective for objective in (QRegression, PolicyGradient)}
"""By the class name a training record holds."""
