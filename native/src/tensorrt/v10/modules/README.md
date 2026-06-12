# TensorRT 10 Native Modules

This directory contains include-units extracted from `native/src/tensorrt/v10/api.cpp`.

Current rule:
- Keep TensorRT 10 exported C ABI names unchanged.
- Keep version-specific TensorRT behavior inside the v10 adapter line.
- Split stable deployment areas first: `layers` and `context` are the first extracted groups.
- Continue splitting `runtime`, `builder`, `network`, `engine`, `parser`, `inspector`, and metadata groups in later batches.

