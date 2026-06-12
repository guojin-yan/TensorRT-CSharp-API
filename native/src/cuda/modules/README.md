# CUDA Native Modules

This directory contains include-units extracted from `native/src/cuda/api.cpp`.

Current rule:
- Keep public C ABI entrypoint names unchanged.
- Keep module files included from `api.cpp` while they still depend on anonymous-namespace helpers and shared object validation.
- Split by deployment area before adding more APIs: `device`, `stream`, `event`, `memory`, `pitched_memory`, `pinned_memory`, `managed_memory`, and `error`.
- Move a module to a standalone `.cpp` only after its helper dependencies have been promoted to reusable headers.

