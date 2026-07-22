# TensorRtExec Refitted Plan Persistence Evidence

- Schema: `trtexec-refitted-plan-persistence-evidence.v1`
- Option: `--saveRefittedEngine`
- TensorRT 10: persisted, original owner disposed, independent reload and two-process enqueue completed
- Serialization flags: `3 -> 2`; `ExcludeWeights` was cleared and refittable weights were included
- Persisted plan: `408876` bytes, SHA256 `5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb`
- Reload metadata: I/O `2`, layers `5`, profiles `1`, context gate `true`
- Reload refittable: `false`; a TensorRT 10 full-weight plan need not retain stripped-plan refittable state
- Same-process output: `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041`
- Second-process output: `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041`
- Full-weight baseline: `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041`
- TensorRT 8: dry-run parse-only; non-dry rejected before native execution
- TensorRT 11: dependency-probe-only; the option was not reported as applied

This is local source-tree persistence and enqueue evidence. It is not package-consumer runtime,
model accuracy, public package, publish, or release-close proof.
