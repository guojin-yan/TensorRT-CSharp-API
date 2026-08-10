# Trtexec Engine Packaging Runtime Evidence

- Evidence kind: `local-project-reference-engine-packaging-and-weight-streaming-policy`
- TRT8: `dependency-probe-only`; version/exclude applied, refit conflict and strip/streaming guarded
- TRT10 version-compatible/refit: inference `True`; output match `True`
- TRT10 strip mode: `RefitIdentical`; readback `True`
- TRT10 weighted model streamable bytes: `35829504`
- TRT10 50% budget/readback/scratch: `17914752` / `17914752` / `5901824`
- TRT10 load-engine automatic budget: `35829504`; readback `True`
- TRT11 refit/reload: `external-onnx-refit-reload-reference-validated-runtime`; parser load, engine commit, persistence, reload, enqueue, and zero-mismatch reference comparison passed
- TRT11 historical probe: `dependency-probe-only`; structured exception `3228369022` retained as immutable 2026-07-22 environment evidence

This is local builder/engine policy, weighted-model enqueue, and TRT11 stripped-plan refit lifecycle evidence. It is not model accuracy, cross-version lean-runtime, package-consumer, post-publish, or public release proof.
