#!/usr/bin/env python3
"""Export the pinned torchvision ResNet18 weights to a fixed ONNX contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import onnx
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    model = resnet18(weights=None)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample,
            args.onnx,
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    graph = onnx.load(str(args.onnx))
    onnx.checker.check_model(graph)

    categories = ResNet18_Weights.DEFAULT.meta["categories"]
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.labels.write_text("\n".join(categories) + "\n", encoding="utf-8", newline="\n")

    write_json(
        args.report,
        {
            "schemaVersion": 1,
            "recordKind": "classification-resnet18-onnx-export",
            "torchVersion": torch.__version__,
            "torchvisionVersion": torchvision.__version__,
            "weights": {
                "path": str(args.weights.resolve()),
                "length": args.weights.stat().st_size,
                "sha256": sha256(args.weights),
            },
            "onnx": {
                "path": str(args.onnx.resolve()),
                "length": args.onnx.stat().st_size,
                "sha256": sha256(args.onnx),
                "opset": 17,
                "input": {"name": "images", "shape": [1, 3, 224, 224]},
                "output": {"name": "logits", "shape": [1, 1000]},
                "checkerPassed": True,
            },
            "labels": {
                "path": str(args.labels.resolve()),
                "count": len(categories),
                "sha256": sha256(args.labels),
            },
            "preprocess": {
                "resize": "shorter-side-256",
                "crop": "center-224x224",
                "colorOrder": "RGB",
                "scale": 1.0 / 255.0,
                "mean": [0.485, 0.456, 0.406],
                "standardDeviation": [0.229, 0.224, 0.225],
            },
            "policy": {
                "storedOutsideGitRepository": True,
                "uploadsAssets": False,
                "performsPublish": False,
            },
        },
    )

    print(f"OnnxPath={args.onnx.resolve()}")
    print(f"OnnxLength={args.onnx.stat().st_size}")
    print(f"OnnxSha256={sha256(args.onnx)}")
    print(f"LabelsSha256={sha256(args.labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
