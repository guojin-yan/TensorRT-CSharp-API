# TensorRT 8 Native Modules

This directory contains include-units extracted from `native/src/tensorrt/v8/api.cpp`.

Current rule:
- Keep TensorRT 8 exported C ABI names unchanged.
- Keep version-specific TensorRT behavior inside the v8 adapter line.
- Split stable deployment areas first: `layers` and `context` are the first extracted groups.
- Continue splitting `runtime`, `builder`, `network`, `engine`, `parser`, `inspector`, and metadata groups in later batches.

