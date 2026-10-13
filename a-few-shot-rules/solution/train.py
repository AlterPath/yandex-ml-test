from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from starter.runtime.backbone import FrozenBackbone
from starter.runtime.meta import validate_artifact, validate_trainable_parameter_count
from starter.solution.data import iter_episode_batches, read_episodes
from starter.solution.features import FEATURE_SIZE, encode_episodes
from starter.solution.model import QUERY_COUNT, ParticipantModule

MIN_EPOCHS = 40
MIN_BATCH_SIZE = 32


def resample_split(
    support_index: torch.Tensor,
    support_labels: torch.Tensor,
    query_index: torch.Tensor,
    query_labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    items = torch.cat([support_index, query_index], dim=1)
    labels = torch.cat([support_labels, query_labels], dim=1)
    keys = labels * 2 + torch.rand(labels.shape, generator=generator)
    positive = keys.argsort(dim=1, descending=True)[:, :4]
    negative = keys.argsort(dim=1)[:, :4]
    support = torch.cat([positive, negative], dim=1)
    chosen = torch.zeros_like(labels, dtype=torch.bool).scatter(1, support, True)
    rest = (~chosen).float().argsort(dim=1, descending=True, stable=True)[:, :QUERY_COUNT]
    rest = rest.gather(1, torch.rand(rest.shape, generator=generator).argsort(dim=1))
    return (
        items.gather(1, support),
        labels.gather(1, support),
        items.gather(1, rest),
        labels.gather(1, rest),
    )


def save_adapter(model: ParticipantModule, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(output),
    )
    validate_artifact(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-episodes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    epochs = max(args.epochs, MIN_EPOCHS)
    batch_size = max(args.batch_size, MIN_BATCH_SIZE)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    backbone = FrozenBackbone()
    episodes = read_episodes(args.train, require_query_labels=True)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    started = time.perf_counter()
    features, support_index, query_index, support_labels = encode_episodes(backbone, episodes)
    query_labels = torch.tensor(
        [[example.label for example in episode.queries] for episode in episodes]
    )
    encode_seconds = time.perf_counter() - started

    model = ParticipantModule(feature_size=FEATURE_SIZE)
    model.feature_mean.copy_(features.mean(0))
    model.feature_std.copy_(features.std(0, unbiased=False) + 1e-6)
    counts = torch.bincount(query_labels.sum(1), minlength=QUERY_COUNT + 1).float()
    model.count_prior.copy_((counts + 1) / (counts.sum() + QUERY_COUNT + 1))
    trainable = validate_trainable_parameter_count(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    steps_per_epoch = (len(episodes) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=epochs * steps_per_epoch,
        pct_start=0.1,
    )

    started = time.perf_counter()
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        s_index, s_labels, q_index, q_labels = resample_split(
            support_index, support_labels, query_index, query_labels, generator
        )
        total = 0.0
        for batch in iter_episode_batches(
            len(episodes), batch_size, shuffle=True, generator=generator
        ):
            support = features[s_index[batch]]
            query = features[q_index[batch]]
            target = q_labels[batch].float()
            loss = F.binary_cross_entropy_with_logits(
                model(query, support, s_labels[batch]), target
            ) + F.binary_cross_entropy_with_logits(
                model.raw_logits(query, support, s_labels[batch]), target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(episodes))
    train_seconds = time.perf_counter() - started

    model.eval()
    save_adapter(model, args.adapter)
    artifact = validate_artifact(args.adapter)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "episodes": len(episodes),
                "unique_texts": features.shape[0],
                "epochs": epochs,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "count_prior": model.count_prior.tolist(),
                "encode_seconds": encode_seconds,
                "train_seconds": train_seconds,
                "trainable_parameters": trainable,
                "adapter_scalars": artifact.scalar_count,
                "adapter_bytes": artifact.size_bytes,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
