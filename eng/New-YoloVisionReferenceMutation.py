#!/usr/bin/env python3
"""Create a deterministic single-value mutation for a structured tensor reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--delta", type=float, default=10000.0)
    args = parser.parse_args()
    if args.index < 0:
        parser.error("--index must be non-negative")
    if not math.isfinite(args.delta) or args.delta == 0.0:
        parser.error("--delta must be finite and nonzero")
    return args


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    with input_path.open("r", encoding="utf-8-sig") as stream:
        reference: dict[str, Any] = json.load(stream)

    values = reference.get("values")
    if reference.get("schemaVersion") != 1 or not isinstance(values, list):
        raise ValueError("Input must be a schemaVersion 1 structured tensor reference with values.")
    if args.index >= len(values):
        raise ValueError(f"Mutation index {args.index} is outside the {len(values)} reference values.")

    original_value = float(values[args.index])
    mutated_value = original_value + args.delta
    if not math.isfinite(original_value) or not math.isfinite(mutated_value):
        raise ValueError("The original and mutated values must be finite.")

    original_sha256 = sha256_file(input_path)
    values[args.index] = mutated_value
    reference["sourceClassification"] = "controlled-single-value-mutation"
    reference["controlledMutation"] = {
        "kind": "single-reference-value-mutation",
        "sourcePath": str(input_path),
        "sourceSha256": original_sha256,
        "index": args.index,
        "originalValue": original_value,
        "delta": args.delta,
        "mutatedValue": mutated_value,
        "expectedRuntimeOutcome": "nonzero-exit-and-one-mismatch-at-the-mutated-index",
        "proofBoundary": "controlled-negative-only; never a valid model-output reference",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(reference, stream, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        stream.write("\n")

    print(f"ControlledReferenceMutation={output_path}")
    print(f"SourceSha256={original_sha256}")
    print(f"MutatedSha256={sha256_file(output_path)}")
    print(f"Index={args.index} Original={original_value!r} Mutated={mutated_value!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
