from __future__ import annotations

import itertools
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

QUERY_COUNT = 4
LABEL_CONFIGS = torch.tensor(list(itertools.product([0.0, 1.0], repeat=QUERY_COUNT)))
CONFIG_SIZES = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])


def class_means(
    values: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = (labels * weights).unsqueeze(-1)
    negative = ((1 - labels) * weights).unsqueeze(-1)
    mean_positive = (values * positive).sum(1) / positive.sum(1)
    mean_negative = (values * negative).sum(1) / negative.sum(1)
    return mean_positive, mean_negative, positive, negative


class ParticipantModule(nn.Module):
    def __init__(
        self,
        feature_size: int = 312,
        raw_size: int = 312,
        hidden_size: int = 128,
        num_experts: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.raw_size = raw_size
        self.num_experts = num_experts
        self.register_buffer("feature_mean", torch.zeros(feature_size))
        self.register_buffer("feature_std", torch.ones(feature_size))
        self.register_buffer("count_prior", torch.full((QUERY_COUNT + 1,), 1.0 / (QUERY_COUNT + 1)))
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.experts = nn.Linear(hidden_size, num_experts)
        self.log_temperature = nn.Parameter(torch.tensor(1.0))
        self.proto_scale = nn.Parameter(torch.tensor(0.1))
        self.expert_scale = nn.Parameter(torch.tensor(1.0))
        self.raw_scale = nn.Parameter(torch.tensor(10.0))

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_std

    def forward(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
        support_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if support_features is None or support_labels is None:
            return query_features.new_zeros(query_features.shape[:-1])
        labels = support_labels.float()
        weights = torch.ones_like(labels) if support_weights is None else support_weights
        support = self.encoder(self.normalize(support_features))
        query = self.encoder(self.normalize(query_features))

        center_positive, center_negative, positive, negative = class_means(support, labels, weights)
        proto = ((query - center_negative[:, None]) ** 2).sum(-1) - (
            (query - center_positive[:, None]) ** 2
        ).sum(-1)
        proto = proto / support.shape[-1] ** 0.5

        support_scores = self.experts(support)
        query_scores = self.experts(query)
        mean_positive, mean_negative, _, _ = class_means(support_scores, labels, weights)
        spread = (
            (support_scores - mean_positive[:, None]) ** 2 * positive
            + (support_scores - mean_negative[:, None]) ** 2 * negative
        ).sum(1)
        std = (spread / (weights.sum(1, keepdim=True) - 2) + 1e-2).sqrt()
        separation = (mean_positive - mean_negative) / std
        routing = torch.softmax(
            self.log_temperature.exp() * torch.cat([separation, -separation], dim=-1), dim=-1
        )
        direction = routing[:, : self.num_experts] - routing[:, self.num_experts :]
        centered = (query_scores - ((mean_positive + mean_negative) / 2)[:, None]) / std[:, None]
        experts = (centered * direction[:, None]).sum(-1)
        return self.proto_scale * proto + self.expert_scale * experts

    def raw_logits(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        support_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = support_labels.float()
        weights = torch.ones_like(labels) if support_weights is None else support_weights
        support = F.normalize(self.normalize(support_features)[..., : self.raw_size], dim=-1)
        query = F.normalize(self.normalize(query_features)[..., : self.raw_size], dim=-1)
        center_positive, center_negative, _, _ = class_means(support, labels, weights)
        center_positive = F.normalize(center_positive, dim=-1)
        center_negative = F.normalize(center_negative, dim=-1)
        return self.raw_scale * (
            (query * center_positive[:, None]).sum(-1) - (query * center_negative[:, None]).sum(-1)
        )

    @staticmethod
    def leave_one_out_errors(
        scorer: Callable[..., torch.Tensor],
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        batch, size = support_labels.shape
        weights = (1 - torch.eye(size)).repeat(batch, 1)
        features = support_features.repeat_interleave(size, dim=0)
        labels = support_labels.repeat_interleave(size, dim=0)
        held_out = support_features.reshape(batch * size, 1, -1)
        logits = scorer(held_out, features, labels, weights).reshape(batch, size)
        return ((logits >= 0).float() != support_labels.float()).sum(1)

    @torch.no_grad()
    def predict_logits(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        head = self(query_features, support_features, support_labels)
        raw = self.raw_logits(query_features, support_features, support_labels)
        head_errors = self.leave_one_out_errors(self, support_features, support_labels)
        raw_errors = self.leave_one_out_errors(self.raw_logits, support_features, support_labels)
        use_head = (head_errors <= raw_errors + 1).unsqueeze(-1)
        return torch.where(use_head, head, raw)


def decode_labels(logits: torch.Tensor, count_prior: torch.Tensor) -> torch.Tensor:
    configs = LABEL_CONFIGS.to(logits)
    scores = F.logsigmoid(logits) @ configs.T + F.logsigmoid(-logits) @ (1 - configs).T
    counts = configs.sum(1).long()
    scores = scores + torch.log(count_prior[counts] / CONFIG_SIZES.to(logits)[counts])
    return configs[scores.argmax(1)]


def consistent_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    return torch.where(labels > 0, probabilities.clamp(min=0.5), probabilities.clamp(max=0.499))
