# Release Candidate 文章矩阵总结

本文汇总 TensorRtSharp4.0 当前中文文章矩阵，用于发布候选收尾、公众号/博客排期和 owner handoff。它不是 runtime proof，也不能替代 `package-consumer-runtime`、`real-model-runtime`、Linux runner proof 或 `post-publish verification`。

当前矩阵已经从 API 文档扩展到项目定位、使用教程、样例实战、TensorRtExec、YoloVision、runtime package、release proof 和 callback / allocator / debug listener 边界。文章矩阵的价值是让项目“可理解、可复现、可交接”，而不是把真实 blocker 写成已经完成。

## 1. 项目定位与架构

这一组文章负责回答“项目是什么、为什么不是 plain P/Invoke、为什么 deferred 仍然存在”：

- `project-overview.md`
- `project-release-story-and-boundaries.md`
- `release-public-story-pack.md`
- `why-not-plain-pinvoke.md`
- `interface-zero-to-deferred-boundary.md`
- `trt-cross-version-strategy.md`
- `tensorrt-object-model.md`

核心边界：

- `manifest/source 100%` 不等于所有 runtime 场景已完成。
- deferred row 是 ABI、ownership、callback lifetime 或 borrowed pointer 风险的显式记录。
- 高层 C# wrapper、smoke、package consumer 和真实模型 evidence 才能支撑用户可用性。

## 2. 安装、环境和 runtime package

这一组文章负责帮助用户选择正确的运行时组合：

- `getting-started.md`
- `windows-local-dev-environment.md`
- `runtime-package-selection.md`
- `runtime-package-matrix-reading-guide.md`
- `runtime-packages.md`
- `runtime-distribution-strategy.md`
- `package-readiness-current-state.md`
- `cuda-error-35-troubleshooting.md`

核心边界：

- managed package 与 runtime package 分工不同。
- runtime package key 必须匹配 RID、TensorRT、CUDA 和 cuDNN。
- `blocked-by-cuda-driver` 是兼容主机问题，不是 smoke 通过。
- restore/build/native-copy 可以 ready，但不能替代 runtime smoke。

## 3. 低资产依赖样例

这一组文章适合用户立即复现：

- `cuda-stream-event-multistream-tutorial.md`
- `dynamic-shape-optimization-profile-tutorial.md`
- `inference-bindings-tutorial.md`
- `onnx-parser-to-serialized-engine-tutorial.md`
- `sample-runners.md`
- `sample-evidence-ladder.md`

对应样例：

- `samples/MultiStream`
- `samples/DynamicShape`
- `samples/InferenceBindings`
- `samples/OnnxToEngine`

这些文章证明样例路径和 wrapper 使用方式清楚，但不自动变成 release proof record。sample-level evidence 和 `package-consumer-runtime` 必须分开。

## 4. TensorRtExec 与 OnnxToEngine

这一组文章负责解释官方 trtexec-like 转换能力和项目内工具边界：

- `tensorrtexec-tool-getting-started.md`
- `tensorrtexec-gui-user-guide.md`
- `tensorrtexec-external-onnx-build-report.md`
- `tensorrtexec-option-layering-deep-dive.md`
- `onnx-to-engine-trtexec-conversion-guide.md`
- `onnxtoengine-and-tensorrtexec-boundary.md`
- `tool-report-to-release-proof-record.md`

核心边界：

- `applications/TensorRtExec` 可以生成 build/precheck report、normalized command、sidecar 和 WinForms 入口。
- `build-only` 是构建证据，不是 inference proof。
- `parse-only` 是参数解析和报告，不是 native TensorRT 行为已完整执行。
- `sidecar-only` 是 metadata / handoff，不是 runtime proof。

## 5. Classification 与 YoloVision

这一组文章负责外部真实模型接入：

- `classification-real-asset-walkthrough.md`
- `classification-model-assets.md`
- `classification-asset-candidates.md`
- `yolovision-all-task-overview.md`
- `yolovision-detection-tutorial.md`
- `yolovision-segmentation-tutorial.md`
- `yolovision-pose-tutorial.md`
- `yolovision-obb-tutorial.md`
- `yolovision-classification-semantic-tutorial.md`
- `yolovision-multi-output-metadata-guide.md`
- `yolo-family-profile-and-postprocess-guide.md`
- `real-model-evidence-backfill-playbook.md`

`samples/YoloVision` 是统一 YOLO-family 样例，覆盖 `YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom` 与 `det、cls、seg、obb、pose、sem`。support matrix、asset template 和 sidecar 不等于 `real-model-runtime`；真实晋级需要模型、labels、input、license、SHA256、runner log 和 validator。

## 6. Release proof 与 owner action

这一组文章负责 release close 的真实证据链：

- `release-final-audit-map.md`
- `release-frontpage-and-proof-boundary-final-audit.md`
- `release-candidate-final-cross-check.md`
- `release-owner-handoff.md`
- `release-owner-proof-backlog.md`
- `release-final-owner-action-sequence.md`
- `release-proof-non-substitutes.md`
- `package-consumer-runtime-proof-playbook.md`
- `post-publish-verification-proof-playbook.md`
- `release-close-gap-dashboard.md`
- `compatible-host-proof-execution-pack.md`
- `release-candidate-final-evidence-freeze.md`

当前仍未消失的 blocker：

1. owner authorization
2. `package-consumer-runtime`
3. Linux runner proof
4. `real-model-runtime`
5. `post-publish verification`

这些 blocker 必须由真实 proof record、真实日志、真实 hash、真实 host metadata 和 validator 共同消除。

## 7. Callback / allocator / debug listener 边界

这一组文章负责解释高风险 deferred API：

- `callback-allocator-safety-bridge-roadmap.md`
- `real-callback-runtime-evidence-schema.md`
- `real-callback-trampoline-gate.md`
- `output-allocator-runtime-proof-precheck.md`
- `debug-listener-runtime-proof-precheck.md`
- `debug-listener-real-callback-runtime-proof.md`
- `allocator-owner-ledger-safety-gate.md`

核心边界：

- callback trampoline 需要 nothrow callback、exception status mapping、in-flight accounting 和真实 invocation。
- allocator / debug listener / borrowed tensor 不能暴露无语义裸指针。
- 没有真实 callback runtime proof 前，deferred boundary 应继续保留。

## 8. 矩阵使用建议

对外发布建议按以下顺序组织：

1. 项目定位和架构。
2. 安装、runtime package 和环境排障。
3. 低资产依赖样例。
4. TensorRtExec / OnnxToEngine。
5. Classification / YoloVision。
6. Release proof 和 owner action。
7. Callback / allocator / debug listener 边界。

每篇文章都应链接到 final audit 或 final cross-check，避免读者只看到能力介绍而错过 proof boundary。文章矩阵可以证明项目文档体系已成形，但不能替代真实外部 proof。ProjectReference 只适合源码开发或内部验证，不能替代 package consumer proof。

## 9. 最终验证命令

文章矩阵收尾后至少执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
dotnet build .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateFinalEvidenceFreeze|FullyQualifiedName~CompatibleHostProofExecutionPack|FullyQualifiedName~ReleaseCloseGapDashboard"
```

如果上述验证通过，在没有真实外部条件时，合理结论是：中文文章矩阵、README 前台入口和 release proof boundary 已完成收尾；真实 release close 仍等待 owner proof。
