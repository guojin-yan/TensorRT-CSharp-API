# Parser preflight copied readback proof

- Scope: `IParser::getError`, `IParser::getErrorCount`, `IParser::isSubgraphSupported`
- Managed surface: `OnnxEngineParserPreflightSnapshot`
- Evidence kind: `copied-parser-preflight`
- Pointer-free: `True`
- Deferred history retained: `True`
- TRT8: diagnostics copied; subgraph support remains controlled unavailable where the vendor API is not exposed.
- TRT10/TRT11: existing caller-buffer diagnostic and subgraph support bridge is surfaced through the build report.
- Verification: Tools build passed, TensorRtExec build passed, report schema tests `5/5` passed.

This artifact is build/preflight evidence only. It is not real-model-runtime, package-consumer-runtime, post-publish, or release-owner proof.
