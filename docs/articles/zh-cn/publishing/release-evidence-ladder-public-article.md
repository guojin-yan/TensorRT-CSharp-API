# Release Evidence Ladder：哪些证据能推动发布

发布证据需要分层看待。README、模板、build report、readonly diagnostics 都能帮助定位问题，但只有 owner 可复核的真实 runtime proof 才能推动 release close。

## 适合

- 准备执行发布候选审计的人。
- 需要理解 `package-consumer-runtime proof` 与 `real-model-runtime proof` 区别的人。
- 想把 `applications/TensorRtExec`、sample runner 和 owner proof 产物串起来的维护者。

## 关键路径

- Owner input：`artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json`。
- Schema：`artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json`。
- Forbidden scan：`artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json`。
- Record validator：`eng/Test-PackageConsumerRuntimeProofRecord.ps1`。
- Release close gate：`eng/Test-ReleaseIssueCloseRecord.ps1`。

## proof 边界

证据梯子建议这样理解：

| 层级 | 示例 | 能否晋级 proof |
|---|---|---|
| 文档/模板 | owner input template、runbook | 否 |
| 预检 | dependency probe、preflight | 否 |
| build-only | OnnxToEngine build report、TensorRtExec build report | 否 |
| readonly diagnostics | load-engine metadata、engine inspector text | 否 |
| real runtime smoke | clean consumer run + log hash + validator | 可能 |
| release close | strict validators + owner approval | 可能 |

ProjectReference、local feed、direct `.nupkg`、GUI screenshot、dry-run 和 build-only 都是 forbidden substitutes。

## 配图建议

- 一张 evidence ladder，底层是 template/preflight/build-only，上层是 clean consumer runtime proof。
- 每一层用红色标出不能替代 package-consumer-runtime proof 的证据。

## 下一步

先让 owner input schema、forbidden substitute scan 和 import chain 稳定，再由 owner 在兼容 CUDA/TensorRT 主机上回填真实 clean consumer 运行结果。
