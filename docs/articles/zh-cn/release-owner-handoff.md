# Release Owner Handoff

这篇文章是发布候选交接入口，面向最终 release owner。它把自动化已经能生成的材料、仍缺的真实 proof、允许接受的证据文件和不能替代 proof 的材料放在同一张清单里，方便 owner 在兼容主机、真实包渠道和真实模型资产准备好之后继续推进。

本文不会授权发布，也不会执行上传命令。当前仓库内的 runbook、template、draft、collection package、sidecar、local feed、ProjectReference、build-only、parse-only 和 `blocked-by-cuda-driver` 记录都只能作为 owner guidance 或诊断材料，不能作为 release proof record。

本文固定使用 `package-consumer-runtime proof` 和 `real-model-runtime proof` 两个精确边界词，避免把 clean consumer 运行证据、真实模型运行证据、build-only 报告或 sidecar 记录混在一起。

## 交接入口

建议 owner 按以下顺序阅读和执行：

1. 先阅读 `docs/articles/zh-cn/owner-release-execution-package.md` 与 `artifacts/final-release/owner-release-execution-package.md`，以其中的 `oneScreenReleaseHoldChecklist` 作为最短 owner action 面；它镜像 5 个 blocker，但仍是 guidance，不是 proof。
2. 阅读 `docs/articles/zh-cn/release-close-preflight.md`，确认 close gate 的输入、输出和不可替代 proof 清单。
3. 阅读 `docs/articles/zh-cn/release-evidence-bundle.md`，确认当前 evidence bundle 中哪些是指导材料，哪些是真实 proof。
4. 阅读 `docs/articles/zh-cn/external-runtime-proof-record.md`，准备真实 `external-runtime-proof-record.json`。
5. 阅读 `docs/articles/zh-cn/post-publish-verification-record.md`，准备真实发布后的 clean consumer verification record。
6. 阅读 `docs/articles/zh-cn/stale-claim-prepublish-audit.md`，避免在 README、博客、release note 或 issue 中写出越级宣传。
7. 使用 `artifacts/final-release/owner-action-required.md` 作为执行清单，逐项补齐缺口。

`owner-release-execution-package` 的一屏 Release Hold 清单只收敛 owner 操作入口，不能替代 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 或 `post-publish verification` 的真实记录；在真实 proof 缺失时，`canCloseReleaseIssue=false` 必须保持不变。

## 当前可接受的真实 Proof

| Proof 类型 | 最低要求 | 验证入口 |
| --- | --- | --- |
| External runtime proof | 兼容 CUDA/TensorRT 主机、真实 consumer project、目标 runtime package key、managed/runtime nupkg SHA256、真实 smoke 命令、stdout/stderr 摘要、真实 log SHA256 | `eng/Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` |
| Package consumer runtime proof | 不使用 ProjectReference 的干净 consumer，restore/build/native asset copy/runtime smoke 全部来自包渠道或被验证的包输入 | `artifacts/final-release/external-runtime-proof-validation.json` |
| Post publish verification proof | 真实包渠道、package id/version、package URL、下载后 nupkg SHA256、clean consumer restore/build/smoke、stdout/stderr 摘要、真实日志 hash | `eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` |
| Real model runtime proof | 模型来源和 license、model/labels/input hash、TensorRtExec build sidecar、sample runner log、`Passed=True` 期望行和 owner reviewed evidence | `eng/Test-SampleAssetManifest.ps1` 与 `eng/Test-SampleRunEvidenceRecord.ps1` |
| Linux runner proof | 真实 Linux runner 输出、目标 runtime key、runner 环境、日志和 validator 结果 | `artifacts/linux-dry-run/.../linux-runner-evidence-validation.json` |

## Owner 必须补齐的缺口

| 缺口 | 当前状态 | Owner action | 不可替代材料 |
| --- | --- | --- | --- |
| 发布授权 | pending owner input | 填写真实 owner decision/input record，确认渠道、签名、NVIDIA redistributable、包体积和回滚策略 | 自动生成的 approval template、dry run、command plan |
| External runtime proof | 仍需要兼容主机真实 smoke | 在兼容 CUDA driver/runtime + TensorRT 主机运行 package consumer smoke，并回填真实日志 hash | dependency probe、blocked-by-cuda-driver、build-only report |
| Post publish verification | 只能在真实渠道发布后执行 | 从真实渠道下载包，在干净 consumer 中 restore/build/smoke，并记录 stdout/stderr 摘要 | draft、template、local feed、ProjectReference |
| YoloVision / Classification 真实模型 | 需要 owner 提供或授权下载资产 | 提供模型、labels、input image、license、hash、runner log 和 evidence record | asset template、sidecar-only、support matrix |
| TensorRtExec 高级参数 | `TrtexecAlignmentStatus=parse-only` | 只有在 native TensorRT 行为和模型级 smoke 同时证明后才能提升 | parser/report/GUI 覆盖、normalized command |
| Linux runner | 需要真实 Linux runner 记录 | 在目标 Linux runner 生成 evidence record 并通过 validator | Windows handoff、template-only、dry-run summary |

## 不能写成完成的句式

发布候选材料、博客和 issue 中不要把以下内容写成已经完成：

- 把 `ready-needs-manual-approval` 写成正式发布完成。
- 把 `blocked-by-cuda-driver` 写成 runtime smoke 通过。
- 把 build-only report、parse-only option coverage 或 evidence sidecar 写成 release proof record。
- 把 `YoloVision` support matrix 写成真实模型已经运行验证。
- 不要把 local feed 或 ProjectReference consumer 写成 clean package consumer proof。
- 把 post-publish template 或 backfill plan 写成真实渠道验证结果。

推荐写法：

- 当前 release close preflight 仍要求真实 owner action。
- 当前 TensorRtExec 高级 trtexec-like 参数仍处于 parse/report 覆盖边界。
- 当前 YoloVision 支持多 family/task 配置，但真实模型 proof 需要 owner 提供模型、license、hash 和运行日志。
- 当前 package consumer runtime proof 需要兼容 CUDA 主机上的干净 consumer 记录。

## 最小交接命令

Owner 可从这些命令开始刷新状态：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
```

如果有真实 external runtime proof：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 `
  -InputPath .\artifacts\final-release\external-runtime-proof-record.json `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RequireExistingLog `
  -FailOnNotProof
```

如果真实渠道发布动作已经由 owner 手工完成，并且需要验证 post publish record：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 `
  -InputPath .\artifacts\final-release\post-publish-verification-record.json `
  -RequireExistingLog `
  -FailOnNotProof
```

## 交接完成定义

本阶段交接完成，不等于项目正式发布完成。交接完成只表示：

- owner 能看懂当前所有 blocker。
- 每个 blocker 都有明确可接受 proof 和不可替代示例。
- 自动化不会执行发布命令。
- release close preflight 在 proof 缺失时继续保持关闭受阻。
- stale claim audit 能阻止越级宣传进入文档或 artifact。

真正的 release close 仍需要真实 external runtime proof、真实 owner authorization、真实 post publish verification、干净 package consumer runtime proof、真实模型 evidence 和 owner review 一起成立。
