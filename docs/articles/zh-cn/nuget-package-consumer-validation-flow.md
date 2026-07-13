# NuGet 消费端验证全流程

包消费端验证用于回答一个很实际的问题：离开源码目录以后，普通 .NET 项目能否从本地或远端包源 restore、build、复制 native assets，并在可选 smoke 中启动 packaged runtime。

TensorRtSharp4.0 使用 `eng/Test-PackageConsumer.ps1` 做这件事。它不是简单的项目构建脚本，而是 release readiness 的核心证据来源之一。

## 验证内容

消费端验证至少覆盖：

| 步骤 | 目的 |
| --- | --- |
| managed package restore | 验证 `JYPPX.TensorRT.CSharp.API` 可被普通项目引用。 |
| runtime package restore | 验证匹配 runtime package 能被还原。 |
| build | 验证消费端项目能编译。 |
| native asset copy | 验证 bridge、CUDA、cuDNN、TensorRT DLL 复制到输出目录。 |
| optional smoke | 验证 packaged runtime 是否能在当前机器启动。 |
| report | 写出 JSON/Markdown，用于 readiness 汇总。 |

默认报告位于：

```text
artifacts/package-consumer/package-consumer-validation-summary.md
artifacts/package-consumer/package-consumer-validation-summary.json
```

## 基本命令

典型命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -RunSmoke `
  -SmokeRuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22
```

如果只想验证 restore/build/native-copy，可以不传 `-RunSmoke`。如果要排查生成的临时消费端项目，使用：

```powershell
-KeepConsumerOutput
```

如果 Windows Defender Application Control 阻止新生成的消费端输出，使用：

```powershell
-SignConsumerOutput
```

需要允许 smoke 失败但仍生成报告时，使用：

```powershell
-AllowSmokeFailure
```

这对于记录 `blocked-by-cuda-driver` 这类环境阻塞特别重要。

## 当前 Windows 证据

当前已记录的重点 Windows 证据包括：

| Runtime key | 结果 |
| --- | --- |
| `win-x64-trt10.11-cuda11.8-cudnn8.9` | `16/16` native assets，consumer smoke passed。 |
| `win-x64-trt10.11-cuda12.9-cudnn9.22` | `19/19` native asset patterns，consumer smoke passed。 |
| `win-x64-trt11.0-cuda12.9-cudnn9.22` | `19/19` native asset patterns，consumer smoke passed。 |
| `win-x64-trt11.0-cuda13.2-cudnn9.22` | `19/19` native asset patterns，restore/build/native-copy passed；runtime smoke 为 `blocked-by-cuda-driver`。 |

`win-x64-trt11.0-cuda13.2-cudnn9.22` 的 blocked 状态说明 packaged runtime 已启动到 TensorRT/bridge 探测和 CUDA runtime 边界，但当前机器驱动不兼容 CUDA 13.2 runtime。它不是 package restore/build/native-copy 失败。

## RealCallbackRuntimeEvidence

报告中会包含 `RealCallbackRuntimeEvidence`。它用于防止把普通 smoke 或 compile-only evidence 误读成真实 callback proof。

只有同时满足下面条件时，才可以把 callback proof 晋级：

- `Status=ready`
- `EvidenceKind=real-callback-runtime`
- `RealCallbackRuntime=True`
- required markers 齐全
- `IsRealCallbackRuntimeProof=True`
- 来源是 full package consumer runtime evidence

当前 CUDA 13.2 full package consumer report 是：

```text
RealCallbackRuntimeEvidence.Status=blocked-by-cuda-driver
EvidenceKind=not-present
IsRealCallbackRuntimeProof=False
```

因此真实 callback runtime proof 仍然是 `false`。

## Evidence Classification 字段

新的 package consumer summary 同时输出一组机器可读字段，方便 readiness、release note 和文章引用：

| 字段 | 含义 | 不能误读 |
| --- | --- | --- |
| `EvidenceKind` | 当前报告的证据类别，例如 `package-consumer-native-copy`、`full-runtime-package-consumer-smoke-driver-blocked`。 | 不是越长越“更 ready”，必须结合 smoke classification。 |
| `RuntimeSmokeClassification` | runtime smoke 分类，例如 `not-requested`、`runtime-smoke-passed`、`runtime-smoke-driver-blocked`。 | `runtime-smoke-driver-blocked` 不是 smoke passed。 |
| `IsRuntimeExecutionEvidence` | 只有 smoke 真的执行并成功退出时才为 `true`。 | 即使为 `true`，也不自动代表 callback proof。 |
| `IsDependencyProbeOnly` | 表示当前报告只能作为 packaging/native-copy/dependency evidence。 | 为 `true` 时不能写成 runtime execution proof。 |
| `IsRealCallbackRuntimeProof` | 是否满足真实 callback runtime proof。 | 当前仍为 `false`，不能由 design gate 或 precheck 替代。 |

对当前 CUDA 13.2 Windows 包线，预期读法是：

```text
EvidenceKind=full-runtime-package-consumer-smoke-driver-blocked
RuntimeSmokeClassification=runtime-smoke-driver-blocked
IsRuntimeExecutionEvidence=False
IsDependencyProbeOnly=True
IsRealCallbackRuntimeProof=False
```

这组字段的价值在于把“包能被消费”和“runtime 真正跑通”拆开，避免发布材料把环境阻塞写成 API 可用性缺陷，也避免把 dependency probe 写成 runtime proof。

## 怎么读 native asset 数量

native asset 数量反映 expected runtime assets 是否复制到消费端输出目录。例如 `19/19` 表示当前 manifest 期望的 native asset patterns 都已复制。

它可以证明包布局和 copy targets 工作正常，但不能证明：

- 当前机器能运行对应 CUDA runtime。
- TensorRT builder/runtime 一定能创建对象。
- callback trampoline 已被 TensorRT runtime 调用。
- allocator/debug listener deferred rows 可以解除。

## 失败归因

| 状态 | 含义 | 下一步 |
| --- | --- | --- |
| restore failed | 包源、版本或依赖解析失败 | 检查本地 nupkg、包 ID、version、source。 |
| build failed | 消费端项目无法编译 | 检查 managed package、target framework、API surface。 |
| missing native assets | runtime package copy 不完整 | 检查 runtime manifest 和 `.targets`。 |
| smoke failed | runtime 启动失败 | 看 exit code 和 diagnostic。 |
| `blocked-by-cuda-driver` | driver/runtime 不兼容 | 换 CUDA-capable driver 或兼容机器复测。 |
| `blocked-by-application-control` | 系统策略阻止执行 | 尝试 `-SignConsumerOutput` 并记录证据。 |

## 与 readiness 的关系

`eng\Test-RuntimePackageReadiness.ps1` 会消费 package consumer 报告，并汇总成：

```text
artifacts/package-readiness/runtime-package-readiness-summary.md
```

readiness clean 不等于 100% runtime 可用。当前 `readiness blockers: 0` 表示 package/readiness 层没有发现阻塞项；CUDA 13.2 runtime smoke 仍然是 `blocked-by-cuda-driver`，真实 callback proof 仍为 `false`。

## 下一步阅读

- [Package Readiness Summary 怎么读](readiness-summary-guide.md)
- [CUDA error 35 与驱动兼容排查](cuda-error-35-troubleshooting.md)
- [发布候选质量门禁](release-candidate-gate.md)
