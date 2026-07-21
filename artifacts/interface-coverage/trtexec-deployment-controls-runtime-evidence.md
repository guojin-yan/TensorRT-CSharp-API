# TensorRtExec Deployment Controls Runtime Evidence

- Evidence kind: `local-project-reference-synthetic-deployment-controls-smoke`
- Source state: working tree after `81954a5c23f792ffbadbff53130392d747f2b3e0`
- Host: RTX 3060 Laptop / driver 576.02 / TensorRT 10.11 / CUDA 12.9

## TRT10 GPU Policy Run

- Exit code: `0`
- State: `identity-roundtrip`
- Output match: `true`
- Device: requested/readback `0/0`, dedicated execution thread
- Applied/read back: GPU fallback, tactic sources, DirectIO, SparseWeights
- Strongly typed network creation: `true`, using the TensorRT 10 raw `1u << 1` bit
- Report SHA256: `101DE1A2599802FACB8F62F5DE373B24509602C8ABA9D62F8FBA2AFB5C1F9023`
- Engine SHA256: `64EC4A51BAF2B22918EED397C3781BD2680D4E466508A92441EBE884E641E23A`

## Guard Runs

- DLA: host reported 0 cores; core 0 request failed with nonzero exit and did not silently fall back.
- TRT8 strongly typed: device selection applied, strongly typed remained parse-only behind the TRT8 guard; build stayed dependency-probe-only because this bridge lacked ONNX parser support.
- TRT11 strongly typed: the implementation does not reuse the TRT10 raw bit and relies on the TensorRT 11 always-strongly-typed contract, but this host hit the known vendor structured exception `3228369022` during runtime creation, before network creation. The run therefore remains dependency-probe-only and `--stronglyTyped` remains parse-only.
- Sparsity force: build succeeded, `--sparsity` remained parse-only, and the builder snapshot did not contain `SparseWeights`.

## Boundary

This evidence proves local option routing, builder readback, TRT10 strongly typed network creation, identity engine round-trip, and synthetic enqueue only. It is not DLA model execution proof, sparse tactic selection proof, real-model proof, repository-external package-consumer proof, publish approval, or release-close approval. `canPublishPublicly=false`.
