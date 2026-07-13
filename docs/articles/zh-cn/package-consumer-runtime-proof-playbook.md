# Package Consumer Runtime Proof Playbook

这篇文章面向 release owner 和验证执行人，目标是把 `package-consumer-runtime proof` 从“知道需要补齐”推进到“可以按步骤采集”。它不执行发布，不替代 owner authorization，也不会把 local feed、ProjectReference、dependency-probe-only、build-only、parse-only、sidecar-only、managed-readiness-only、precheck-only、dry-run-only、schema-only 或 `blocked-by-cuda-driver` 当作 runtime proof。

## Proof 定义

`package-consumer-runtime proof` 只来自一个干净 consumer 工程：它从包源 restore，复制 native assets，使用目标 runtime package key 执行 runtime smoke，并记录真实日志、host metadata、package hash 和 stdout/stderr 摘要。

最小要求：

| 项目 | 要求 | 不接受 |
| --- | --- | --- |
| Consumer 工程 | 不在仓库内部，不使用 ProjectReference | 当前仓库测试项目、local ProjectReference consumer |
| 包来源 | 明确 package source、managed/runtime package id、version、nupkg SHA256 | 未记录 hash 的本地 bin 输出 |
| Runtime key | 与目标 release runtime package key 一致 | smoke 命令缺少 `--runtime-package-key` |
| Host metadata | CUDA driver/runtime、TensorRT、cuDNN、OS、arch | dependency probe 输出但没有 smoke |
| Smoke log | 真实文件路径、SHA256、stdout/stderr summary | template、draft、blocked-by-cuda-driver |
| Validator | `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` | 只生成 collection package 或 runbook |

## Owner 执行顺序

1. 在兼容 CUDA/TensorRT 主机上创建干净目录，例如 `C:\trtsharp-consumer-proof\consumer`。
2. 用真实 package source 配置 NuGet，不要引用当前仓库项目。
3. restore/build consumer，确认 native assets 来自 package。
4. 运行 runtime smoke，命令中显式传入目标 runtime package key。
5. 保存 restore/build/native asset listing/dependency probe/smoke log。
6. 计算 managed/runtime nupkg SHA256 和 smoke log SHA256。
7. 回填 `artifacts/final-release/external-runtime-proof-record.json`。
8. 运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 `
  -InputPath .\artifacts\final-release\external-runtime-proof-record.json `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RequireExistingLog `
  -FailOnNotProof
```

9. 刷新：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1
```

## 记录字段清单

| 字段组 | 必填内容 |
| --- | --- |
| package source | managed package id/version/path/hash、runtime package id/version/path/hash、runtime package key |
| consumer identity | project path、target framework、RID、no ProjectReference、clean restore source |
| host metadata | OS、arch、CUDA driver supported runtime、CUDA runtime、TensorRT line/version、cuDNN |
| command | restore/build/smoke command、`--runtime-package-key`、log paths |
| results | smoke status、stdout summary、stderr summary、native asset listing summary |
| hashes | restore log、native listing、dependency probe、smoke log SHA256 |

## 边界

- `blocked-by-cuda-driver` 是兼容主机缺口，不是 smoke 通过。
- dependency probe 是诊断，不是 runtime execution proof。
- ProjectReference consumer 即使能运行，也不是 package consumer proof。
- build-only 或 parse-only 输出不证明 runtime smoke。
- `TensorRtCallbackAllocatorReadinessSnapshot`、`CallbackAllocatorReadinessSnapshot=` 和 `RuntimeEvidenceKind=managed-readiness` 只能说明 managed wrapper readiness，不证明包消费者路径真实执行。
- `managed-readiness-only`、`precheck-only`、`dry-run-only` 或 `schema-only` 记录不能替代 compatible host smoke、exitCode=0、native assets copied、真实日志 SHA256 和 validator 通过。
- `real-model-runtime` 属于样例和真实模型证据，不等于 package consumer proof。

只有 validator 产出真实 runtime proof，且 release evidence bundle 中 external runtime proof 可以推广，才可以把该链路交给 release close preflight 继续判断。否则必须保持 owner action，不得把 checklist、runbook、collection package 或 sidecar 写成 proof。

## 第三批正文门禁

### 适用读者

本文适合 release owner、外部验证执行人和需要证明公开包可被干净项目消费的维护者。

### 解决问题

本文解决 package consumer runtime proof 的采集顺序：包来源、clean consumer、native assets、runtime smoke、日志 hash 和 validator 缺一不可。

### 背景与场景

本仓库可以生成大量 preflight、runbook 和 draft artifact，但最终 release close 需要真实外部 clean consumer 证据。它必须来自仓库外项目和真实包来源，而不是本地源码引用。

### 代码与文件入口

- `artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.md`
- `artifacts/final-release/clean-consumer-proof-owner-execution-pack.md`
- `eng/Test-PackageConsumer.ps1`
- `tests/JYPPX.ProjectQuality.Tests/CleanConsumerRuntimeProofExecutionChecklistTests.cs`

### 操作路径

先准备公开包或 owner 指定包源，再创建仓库外 clean consumer，运行 restore/build/runtime smoke，保存日志、host metadata、package hash 和 stdout/stderr summary，最后运行 strict validator。

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics、readonly summary、bridge-only 和 dependency probe 都不是 runtime proof。playbook 是执行指南，不是 proof 本身；`package-consumer-runtime-proof-preflight-matrix.json` 也是预检合同，只有 strict validator 绑定真实日志、hash 和 host metadata 后才可能晋级。

### 下一步

下一步等待 owner 回填真实执行结果；如果没有输入，就继续完善外部 proof collection bundle 和文章教程。
