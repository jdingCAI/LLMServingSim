#!/usr/bin/env python3
"""
Generate synthetic multi-turn conversation dataset for LLMServingSim.

Creates JSONL with integer token IDs structured so that:
- All conversations share a common prefix (for cross-conversation prefix caching)
- Each conversation has unique per-turn tokens
- Consecutive turns share a growing prefix (for within-conversation prefix caching)

Usage:
    python generate_synthetic_multi_turn.py \
        --num-conversations 2000 --num-turns 10 \
        --tokens-per-turn 5120 --common-prefix 512 \
        --output synthetic_2000conv_10turns.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn dataset")
    parser.add_argument("--num-conversations", type=int, default=2000)
    parser.add_argument("--num-turns", type=int, default=10)
    parser.add_argument("--tokens-per-turn", type=int, default=5120,
                        help="New user tokens added per turn")
    parser.add_argument("--common-prefix", type=int, default=512,
                        help="Shared prefix tokens across all conversations")
    parser.add_argument("--output", type=str,
                        default="synthetic_2000conv_10turns.jsonl")
    parser.add_argument("--turn-latency-ns", type=int, default=2_000_000_000,
                        help="Arrival time stagger between turns (ns)")
    args = parser.parse_args()

    output_path = Path(__file__).parent / args.output

    # Common prefix shared by all conversations
    common_prefix = list(range(1, args.common_prefix + 1))

    # Token ID offset to ensure uniqueness across conversations
    # Conv C uses tokens starting at C * conv_token_space
    conv_token_space = (args.num_turns + 1) * (args.tokens_per_turn + 10)

    total_requests = args.num_conversations * args.num_turns
    print(f"Generating dataset: {args.num_conversations} conversations × {args.num_turns} turns = {total_requests} requests")
    print(f"Common prefix: {args.common_prefix} tokens")
    print(f"New tokens per turn: {args.tokens_per_turn}")
    print(f"Output: {output_path}")

    records = []
    for conv_idx in range(args.num_conversations):
        # Start each conversation with the common prefix
        tokens = list(common_prefix)

        # Base offset for this conversation's unique tokens
        conv_base = args.common_prefix + 1 + conv_idx * conv_token_space

        for turn_idx in range(args.num_turns):
            # Add new user tokens for this turn
            turn_base = conv_base + turn_idx * (args.tokens_per_turn + 2)
            new_tokens = list(range(turn_base, turn_base + args.tokens_per_turn))
            tokens.extend(new_tokens)

            record = {
                "input_toks": len(tokens),
                "output_toks": 2,
                "arrival_time_ns": turn_idx * args.turn_latency_ns,
                "input_tok_ids": list(tokens),  # copy
                "output_tok_ids": [0, 0],
            }
            records.append(record)

            # Add 2 assistant response tokens for next turn's history
            assistant_base = turn_base + args.tokens_per_turn
            tokens.extend([assistant_base, assistant_base + 1])

        if (conv_idx + 1) % 200 == 0:
            print(f"  Generated {conv_idx + 1}/{args.num_conversations} conversations...")

    # Sort by arrival time
    records.sort(key=lambda r: r["arrival_time_ns"])

    print(f"Writing {len(records)} requests to {output_path}...")
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Summary
    input_lens = [r["input_toks"] for r in records]
    total_kv_gb = sum(input_lens) * 128 / 1024 / 1024  # 128 KB per token, in GB
    print(f"\nDataset summary:")
    print(f"  Total requests: {len(records)}")
    print(f"  Input tokens: min={min(input_lens)}, max={max(input_lens)}, mean={sum(input_lens)/len(input_lens):.0f}")
    print(f"  Total KV cache (all requests): {total_kv_gb:.0f} GB ({total_kv_gb/1024:.1f} TB)")
    print(f"  Arrival time range: 0 - {max(r['arrival_time_ns'] for r in records) / 1e9:.0f}s")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
