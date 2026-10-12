"""
Preprocess the SlimPajama dataset into LITPKDS packed binary format.

Streams from HuggingFace, tokenizes with TinyLlama's sentencepiece tokenizer,
and writes packed chunks using lit_gpt.packed_dataset.PackedDatasetBuilder.

Usage:
    python scripts/prepare_slimpajama.py
    python scripts/prepare_slimpajama.py --split train --num_workers 64
"""

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Process, cpu_count

import numpy as np
from datasets import load_dataset

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "packed_dataset",
    str(Path(__file__).parent.parent / "lit_gpt" / "packed_dataset.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PackedDatasetBuilder = _mod.PackedDatasetBuilder

CHUNK_SIZE = 2049 * 1024  # 2,098,176 tokens per chunk: 1024 blocks of block_size + 1
VOCAB_SIZE = 32000
BOS_ID = 1  # TinyLlama sentencepiece bos_id, used as document separator
EOS_ID = 2  # TinyLlama sentencepiece eos_id, appended to each document
# TinyLlama drops the GitHub/code subset from every split
SKIP_SET_NAME = "RedPajamaGithub"


def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )
    return tokenizer


def process_shard(shard_idx, num_shards, split, destination_path):
    tokenizer = get_tokenizer()

    builder = PackedDatasetBuilder(
        outdir=str(destination_path),
        prefix=f"train_slim_{shard_idx:04d}" if split == "train" else f"validation_{shard_idx:04d}",
        chunk_size=CHUNK_SIZE,
        sep_token=BOS_ID,
        dtype="auto",
        vocab_size=VOCAB_SIZE,
    )

    ds = load_dataset(
        "gmongaras/SlimPajama-627B_Reupload",
        split=split,
        streaming=True,
    )
    ds = ds.shard(num_shards=num_shards, index=shard_idx)

    doc_count = 0
    token_count = 0
    t0 = time.time()

    for example in ds:
        if example["meta"]["redpajama_set_name"] == SKIP_SET_NAME:
            continue

        text = example["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False) + [EOS_ID]
        arr = np.array(token_ids, dtype=builder.dtype)
        builder.add_array(arr)

        doc_count += 1
        token_count += len(token_ids)

        if doc_count % 10000 == 0:
            elapsed = time.time() - t0
            print(
                f"[Shard {shard_idx}/{num_shards}] "
                f"{doc_count} docs, {token_count/1e6:.1f}M tokens, "
                f"{elapsed:.0f}s elapsed, "
                f"{token_count/elapsed/1e3:.1f}K tok/s"
            )

    # Following TinyLlama, discard the final partial chunk to avoid a
    # bos-padded "meaningless corpus" tail (see jzhang38/TinyLlama#83).

    elapsed = time.time() - t0
    print(
        f"[Shard {shard_idx}/{num_shards}] DONE. "
        f"{doc_count} docs, {token_count/1e6:.1f}M tokens, "
        f"{elapsed:.0f}s total. "
        f"Wrote {len(builder.filenames)} chunk files."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=str,
        default="data/slim_star_combined",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--num_workers", type=int, default=64)
    args = parser.parse_args()

    destination_path = Path(args.destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing SlimPajama ({args.split} split)")
    print(f"  Destination: {destination_path}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Chunk size: {CHUNK_SIZE} tokens")

    processes = []
    for shard_idx in range(args.num_workers):
        p = Process(
            target=process_shard,
            args=(shard_idx, args.num_workers, args.split, destination_path),
        )
        p.start()
        processes.append(p)
        # Stagger launches to avoid HuggingFace API rate limits
        time.sleep(1)

    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"ERROR: {len(failed)} workers failed: {failed}")
        sys.exit(1)

    print(f"All {args.num_workers} workers finished for split={args.split}.")


if __name__ == "__main__":
    main()
