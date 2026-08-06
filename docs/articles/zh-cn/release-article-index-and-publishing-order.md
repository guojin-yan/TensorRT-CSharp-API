# 发布文章索引与推荐发布顺序

TensorRtSharp4.0 的中文材料已经从“接口补全记录”扩展为完整的项目说明、样例教程、工具教程、runtime package 文档、release proof runbook 和 deferred 边界说明。本文用于把这些文章按读者路径重新组织，方便 README、公众号、博客专栏和 release issue 统一引用。

这份索引不是 proof。它只说明文章阅读顺序和发布顺序，不能替代 `package-consumer-runtime`、`real-model-runtime`、Linux runner proof 或 post-publish verification。

## 1. 项目定位与边界

建议先发布这些文章，让读者理解项目为什么不是简单 P/Invoke：

| 顺序 | 文章 | 读者收获 |
| --- | --- | --- |
| 1 | `project-overview.md` | 项目范围、托管包、native bridge、样例入口 |
| 2 | `project-release-story-and-boundaries.md` | manifest/source 匹配、deferred boundary、发布边界 |
| 3 | `release-public-story-pack.md` | 对外介绍素材、可宣传能力和不能越级宣传的内容 |
| 4 | `why-not-plain-pinvoke.md` | 为什么要有 native bridge 和 C# wrapper |
| 5 | `interface-zero-to-deferred-boundary.md` | 从 missing 清零到 deferred 边界提升的叙事 |

这一组文章要反复强调：`manifest/source 100%` 不等于所有 runtime 场景都已经可用。真实完成度还要看高层 wrapper、smoke、package consumer、真实模型和 owner proof。

## 2. 入门、安装与 runtime package

第二组文章帮助用户把包和本机环境对齐：

| 顺序 | 文章 | 读者收获 |
| --- | --- | --- |
| 6 | `getting-started.md` | 快速构建和基础使用 |
| 7 | `runtime-package-selection.md` | 如何选择 runtime package key |
| 8 | `runtime-package-matrix-reading-guide.md` | CUDA / TensorRT / cuDNN 矩阵读法 |
| 9 | `cuda-error-35-troubleshooting.md` | `blocked-by-cuda-driver` 与 CUDA error 35 排查 |
| 10 | `package-readiness-current-state.md` | package layout ready 与 runtime smoke blocker 的区别 |

这里的重点是：restore/build/native-copy 可以 ready，但当前主机 runtime smoke 可能仍被 `blocked-by-cuda-driver` 阻塞。这个状态不能写成通过。

## 3. 低资产依赖样例

第三组文章适合用户立即动手：

| 顺序 | 文章 | 样例 |
| --- | --- | --- |
| 11 | `cuda-stream-event-multistream-tutorial.md` | `samples/Performance/01.MultiStream` |
| 12 | `dynamic-shape-optimization-profile-tutorial.md` | `samples/Inference/02.DynamicShapes` |
| 13 | `inference-bindings-tutorial.md` | `samples/Inference/01.Bindings` |
| 14 | `onnx-parser-to-serialized-engine-tutorial.md` | `applications/OnnxToEngine` |
| 15 | `sample-evidence-ladder.md` | `samples/README.md` |

这些文章可以证明 wrapper 和样例路径可用，但不自动变成 `package-consumer-runtime`。样例输出属于 sample-level evidence，不属于 release proof record。

## 4. TensorRtExec 与 OnnxToEngine

第四组文章围绕工具化转换：

| 顺序 | 文章 | 重点 |
| --- | --- | --- |
| 16 | `tensorrtexec-tool-getting-started.md` | CLI / WinForms 入口 |
| 17 | `onnx-to-engine-trtexec-conversion-guide.md` | ONNX 到 engine 的转换路径 |
| 18 | `tensorrtexec-external-onnx-build-report.md` | 外部 ONNX build report |
| 19 | `tensorrtexec-option-layering-deep-dive.md` | implemented / parse-report-only / build-only 边界 |
| 20 | `onnxtoengine-and-tensorrtexec-boundary.md` | sample 与应用的分工 |

这一组要把 `build-only` 和 `parse-only` 讲清楚：它们适合做构建报告、命令归一化和交接诊断，但不证明任意外部模型的推理输出正确。

## 5. Classification 与 YoloVision

第五组文章面向真实模型资产：

| 顺序 | 文章 | 重点 |
| --- | --- | --- |
| 21 | `classification-real-asset-walkthrough.md` | 分类模型、labels、input、hash、license |
| 22 | `yolovision-all-task-overview.md` | YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 总览 |
| 23 | `yolo-family-profile-and-postprocess-guide.md` | family/task/profile/postprocess |
| 24 | `yolovision-multi-output-metadata-guide.md` | seg/pose/obb 多输出 metadata |
| 25 | `real-model-evidence-backfill-playbook.md` | real-model-runtime 回填流程 |

YoloVision 当前统一承载 det、cls、seg、obb、pose、sem，不再使用旧检测样例名作为当前入口。support matrix、asset template 和 sidecar-only 只是准备材料，不能替代真实 sample-run-evidence。

## 6. Release proof 与 owner action

第六组文章给 release owner：

| 顺序 | 文章 | 重点 |
| --- | --- | --- |
| 26 | `release-final-audit-map.md` | 最终审计总入口 |
| 27 | `release-owner-handoff.md` | owner 交接 |
| 28 | `release-owner-proof-backlog.md` | 剩余真实 proof backlog |
| 29 | `release-final-owner-action-sequence.md` | 最后一公里执行顺序 |
| 30 | `release-proof-non-substitutes.md` | 不可替代 proof 清单 |
| 31 | `package-consumer-runtime-proof-playbook.md` | clean consumer runtime proof |
| 32 | `post-publish-verification-proof-playbook.md` | 真实发布后的验证 |
| 33 | `release-final-owner-action-sequence.md` 中的 Release Issue Close Record 小节 | owner final close decision、evidence bundle SHA256、rollback plan 与 `Test-ReleaseIssueCloseRecord.ps1` |

这一组的共同边界是：helper、template、draft、runbook、collection package、input package、local feed、ProjectReference、build-only、parse-only、sidecar-only、release close preflight、release issue close template 和 `blocked-by-cuda-driver` 都只能辅助 owner 执行，不能替代 proof。`release-issue-close-record-validation.json` 当前为 `blocked-template-only`，只有真实 owner final close decision、post-publish proof、stale audit、preflight 和 evidence bundle SHA256 全部验证后，才可能进入 issue close review。

## 7. Callback / allocator / debug listener 边界

最后发布安全边界系列：

| 顺序 | 文章 | 重点 |
| --- | --- | --- |
| 33 | `callback-allocator-safety-bridge-roadmap.md` | owner ledger、borrowed pointer、callback proof |
| 34 | `real-callback-runtime-evidence-schema.md` | real callback runtime evidence schema |
| 35 | `output-allocator-runtime-proof-precheck.md` | allocator runtime proof 前置条件 |
| 36 | `debug-listener-runtime-proof-precheck.md` | debug listener runtime proof 前置条件 |
| 37 | `real-callback-trampoline-gate.md` | callback trampoline gate |

这组文章要避免读者误解 deferred：callback / allocator / debug listener 的 deferred row 是安全边界，不是隐藏完成度。没有真实 callback invocation、borrowed metadata copy 和 nothrow bridge 前，public API 不应暴露危险指针。

## 发布顺序建议

对外发布时，建议采用四周节奏：

1. 第一周：项目定位、安装、runtime package、CUDA error 35。
2. 第二周：低资产样例、OnnxToEngine、TensorRtExec。
3. 第三周：Classification、YoloVision 和真实模型 evidence。
4. 第四周：release proof、owner action、callback / allocator 边界。

每篇文章结尾都应链接到最终审计地图或 owner handoff，并保留 `blocked-real-proof-required` 的当前边界，直到真实 proof record 和 validator 共同消除 blocker。
