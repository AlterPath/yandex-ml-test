from __future__ import annotations

import os
from typing import Sequence

import torch

from starter.runtime.backbone import FrozenBackbone
from starter.solution.data import Episode

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_LENGTH = 58
FEATURE_SIZE = 3 * 312


def encode_texts(
    backbone: FrozenBackbone,
    texts: Sequence[str],
    batch_size: int = 64,
) -> torch.Tensor:
    tokenizer = backbone.tokenizer
    lengths = [
        len(ids)
        for ids in tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)["input_ids"]
    ]
    order = sorted(range(len(texts)), key=lengths.__getitem__)
    features = torch.zeros(len(texts), FEATURE_SIZE)
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        batch = tokenizer(
            [texts[i] for i in index],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            states = backbone(**batch, output_hidden_states=True, return_dict=True).hidden_states
        mask = batch["attention_mask"].unsqueeze(-1).float()
        cls = states[-1][:, 0]
        mean_last = (states[-1] * mask).sum(1) / mask.sum(1)
        mean_first = (states[0] * mask).sum(1) / mask.sum(1)
        features[index] = torch.cat([cls, mean_last, mean_first], dim=-1)
    return features


def encode_episodes(
    backbone: FrozenBackbone,
    episodes: Sequence[Episode],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = sorted(
        {example.text for episode in episodes for example in episode.support + episode.queries}
    )
    position = {text: i for i, text in enumerate(texts)}
    features = encode_texts(backbone, texts)
    support_index = torch.tensor(
        [[position[example.text] for example in episode.support] for episode in episodes]
    )
    query_index = torch.tensor(
        [[position[example.text] for example in episode.queries] for episode in episodes]
    )
    support_labels = torch.tensor(
        [[example.label for example in episode.support] for episode in episodes]
    )
    return features, support_index, query_index, support_labels
