# Release Owner Proof Backlog

这份 backlog 是给 release owner 的剩余真实工作清单。它不是 runbook 的替代，也不是 proof；它把当前不能由本仓库自动完成的事项拆成可执行任务，方便 owner 在兼容主机、真实模型资产和真实发布渠道准备好后一次性推进。

## Backlog 总览

| Backlog | 当前状态 | 需要的真实输入 | 完成判定 |
| --- | --- | --- | --- |
| owner authorization | pending owner action | 发布渠道、版本、许可、签名、回滚策略 | owner decision/input record 通过 |
| package-consumer-runtime | blocked-real-proof-required | 兼容 CUDA/TensorRT 主机、clean consumer、真实包源、smoke log | `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` |
| linux-runner-proof | pending real Linux runner | Linux x64 runner、目标 runtime key、日志 | `Test-LinuxRunnerEvidenceRecord.ps1` |
| real-model-runtime | pending real assets | Classification/YoloVision 模型、labels、input、license、hash、runner log | `Test-SampleAssetManifest.ps1` + `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog` |
| post-publish verification | blocked until real publish | 真实渠道 package、download hash、clean consumer smoke | `Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof` |

## 1. Owner Authorization

Owner authorization 必须回答：

- 是否允许公开发布。
- 使用哪个 package source 或 channel。
- NVIDIA TensorRT/CUDA/cuDNN redistributable disposition 是否复核。
- 是否使用签名。
- package id/version 是否最终确认。
- 如果发布后验证失败，如何 rollback。

自动生成的 owner approval template、dry run、command plan 和 execution package 只能降低漏项风险，不能替代 owner 输入。

## 2. Package Consumer Runtime

真实 `package-consumer-runtime` 的核心是干净 consumer：

- consumer 在仓库外。
- 不使用 ProjectReference。
- restore 来源是真实 package source 或 owner 指定的待发布 package input。
- native assets 来自 runtime package。
- smoke command 包含目标 `--runtime-package-key`。
- stdout/stderr summary 与日志 SHA256 可复核。

不接受：

- local bin output
- ProjectReference
- dependency-probe-only
- build-only
- parse-only
- sidecar-only
- `blocked-by-cuda-driver`

## 3. Linux Runner Proof

Linux runner proof 需要真实 Linux x64 runner，而不是 Windows handoff 或 template-only。记录中至少应包含：

- OS / arch
- CUDA driver/runtime
- TensorRT line/version
- cuDNN version
- package identity
- runtime package key
- restore/build/probe/smoke log
- validator output

Windows 上生成的 Linux handoff 可以帮助 owner 准备命令，但不能写成 Linux runner proof。

## 4. Real Model Runtime

`real-model-runtime` 面向样例，不等于 release package consumer proof。Classification 和 YoloVision 都需要：

- 模型来源 URL 或 owner 内部来源说明。
- license 或 redistribution note。
- model SHA256。
- labels SHA256。
- input image / input data SHA256。
- TensorRtExec build-only report 或 evidence sidecar。
- sample runner log。
- sample-run-evidence record。

YoloVision 范围继续固定为 `YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom` 与 `det、cls、seg、obb、pose、sem`。support matrix、asset template、metadata sidecar 都不能单独晋级为 `real-model-runtime`。

## 5. Post Publish Verification

post-publish verification 只能在真实渠道发布后发生。它必须验证“用户从渠道下载包后能否在 clean consumer 中运行”：

- package id/version/channel URL
- downloaded nupkg SHA256
- clean consumer project path
- no ProjectReference
- restore/build/native asset listing
- dependency probe
- runtime smoke
- stdout/stderr summary
- smoke log SHA256

input draft、template、collection package 和 clean consumer scan 都不是 proof。它们只帮助 owner 采集字段。

## 执行优先级

如果 owner 只能先完成一部分，建议顺序如下：

1. 先完成 owner authorization，避免后续 proof 采集后仍不能决定渠道。
2. 再完成 package-consumer-runtime，因为它验证包消费路径。
3. 同步准备 real-model-runtime 的 Classification / YoloVision 资产。
4. 安排 Linux runner proof，补齐跨平台证据。
5. 真实发布后立即执行 post-publish verification。

任何阶段失败都应保留 blocker，不要删除 stale claim 规则或修改 validator 来制造通过。
