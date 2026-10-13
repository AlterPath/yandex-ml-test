from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from starter.runtime.backbone import FrozenBackbone
from starter.runtime.meta import validate_artifact, verify_backbone_sha
from starter.solution.data import iter_episode_batches, prediction_rows, read_episodes, write_jsonl
from starter.solution.features import FEATURE_SIZE, encode_episodes
from starter.solution.model import ParticipantModule, consistent_probabilities, decode_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    torch.set_num_threads(2)
    verify_backbone_sha()
    validate_artifact(args.adapter)

    backbone = FrozenBackbone()
    episodes = read_episodes(args.data)
    started = time.perf_counter()
    features, support_index, query_index, support_labels = encode_episodes(backbone, episodes)
    encode_seconds = time.perf_counter() - started

    model = ParticipantModule(feature_size=FEATURE_SIZE)
    model.load_state_dict(load_file(str(args.adapter), device="cpu"), strict=True)
    model.eval()

    started = time.perf_counter()
    probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in iter_episode_batches(len(episodes), args.batch_size, shuffle=False):
            support = features[support_index[batch]]
            query = features[query_index[batch]]
            logits = model.predict_logits(query, support, support_labels[batch])
            labels = decode_labels(logits, model.count_prior)
            probabilities.append(consistent_probabilities(logits, labels))
    inference_seconds = time.perf_counter() - started

    write_jsonl(prediction_rows(episodes, torch.cat(probabilities)), args.output)
    print(
        json.dumps(
            {
                "episodes": len(episodes),
                "queries": len(episodes) * 4,
                "unique_texts": features.shape[0],
                "encode_seconds": encode_seconds,
                "inference_seconds": inference_seconds,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
