# CUDA error 35 与驱动兼容排查

`cudaRuntimeGetVersion failed with CUDA error 35` 通常表示当前 NVIDIA driver 版本不足以支持正在加载的 CUDA runtime。TensorRtSharp4.0 的 package consumer 会把这种情况记录为 `blocked-by-cuda-driver`。

这不是 C# wrapper 缺失，也不是 TensorRT API manifest 错误。它表示程序已经走到 packaged CUDA runtime 的执行边界，但当前机器驱动/runtime 组合不兼容。

## 当前项目中的证据

当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` full package consumer smoke 已实际请求，并输出：

```text
TensorRtAssemblyBridge=jyppxtrtbridge
CudaAssemblyBridge=jyppxtrtbridge
Bridge=jyppxtrtbridge TRT=11.0.0 CUDA=13.2
```

随后在 `CudaEnvironmentProbe.GetCurrent()` 中调用 `cudaRuntimeGetVersion` 时失败：

```text
cudaRuntimeGetVersion failed with CUDA error 35
```

package consumer report 因此写入：

```text
SmokeResult=blocked-by-cuda-driver
RealCallbackRuntimeEvidence.Status=blocked-by-cuda-driver
RealCallbackRuntimeEvidence.IsRealCallbackRuntimeProof=False
```

readiness summary 也写入：

```text
runtime execution smoke: blocked-by-cuda-driver
real callback runtime evidence: blocked-by-cuda-driver; evidence-kind=not-present; proof=False
```

## 为什么不是 API 缺口

如果 native DLL 缺失，通常会表现为 loader failure、missing native assets 或 bridge dependency diagnostic。当前 readiness 中：

- native assets 已复制：`19/19`。
- vendor blockers：none。
- TensorRT/CUDA/cuDNN root 均存在。
- bridge consumer native dependency：ready。

因此问题不是包里少文件，而是当前机器驱动不能执行该 CUDA runtime。

## 如何复现

请求 full package consumer smoke：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RunSmoke `
  -AllowSmokeFailure
```

刷新 readiness：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

抽查：

```powershell
rg -n "blocked-by-cuda-driver|runtime execution smoke|real callback runtime evidence|proof=False" `
  .\artifacts\package-readiness\runtime-package-readiness-summary.md
```

## 排查步骤

1. 确认目标 runtime key。

```text
win-x64-trt11.0-cuda13.2-cudnn9.22
```

2. 确认本机安装的 NVIDIA driver 是否支持对应 CUDA runtime。

可以使用：

```powershell
nvidia-smi
```

查看 Driver Version 和 CUDA Version。注意 `nvidia-smi` 显示的是驱动支持的最高 CUDA runtime 能力，不等于本机 Toolkit 安装版本。

3. 确认 Toolkit root。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

4. 确认 runtime assets。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

如果 vendor blockers 为 none，但 smoke 仍是 `blocked-by-cuda-driver`，优先升级驱动或换到 CUDA 13.2 兼容机器。

## 外部兼容主机回填要求

在兼容机器上复测时，不要只截图 `nvidia-smi`。需要把以下字段回填到 external runtime proof record：

- CUDA driver version 与 driver supported CUDA runtime。
- 实际加载的 CUDA runtime version。
- TensorRT line 与 TensorRT runtime version。
- 目标 runtime package key。
- managed/runtime nupkg SHA256。
- restore/build/smoke 命令、exit code、stdout/stderr 摘要。
- smoke log path 与 64 位十六进制 SHA256。

只有 `proofClassification=package-consumer-runtime`、`isRuntimeExecutionEvidence` 明确为通过态、runtime proof promotion 字段由 validator 判定为通过、`smokeStatus=passed` 且 `Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof` 通过时，才能把这次外部执行作为 runtime proof。

## 不要做什么

不要为了解决 CUDA error 35 做这些事：

- 删除 deferred rows。
- 修改 manifest 来绕过 runtime smoke。
- 把 `blocked-by-cuda-driver` 写成 `passed`。
- 把 bridge-only probe 当成 full runtime proof。
- 把普通 smoke 通过当成 callback proof。
- 降级调用不匹配的 CUDA DLL 来制造“可运行”结果。

正确处理方式是保留 blocked evidence，并在兼容驱动环境重新运行 smoke。

## 与真实 callback proof 的关系

CUDA error 35 阻塞的是当前 full package consumer runtime smoke。它还没有进入真实 TensorRT callback 证明阶段。

即使未来 CUDA driver 兼容、普通 smoke 通过，也仍然需要 full package consumer 输出完整 `real-callback-runtime` evidence，才能把 callback proof 提升为 true。 required markers 至少包括：

- `EvidenceKind=real-callback-runtime`
- `RealCallbackRuntime=True`
- `CallbackKind`
- `TensorRtLine`
- `CudaLine`
- `RuntimePackageKey`
- `OwnerId`
- `InvocationCount`
- `AllocationCount`
- `ReleaseCount`
- `FailureCount`
- `InFlightCallbackCount`
- `LastStatus`
- `LastDiagnostic`
- `FullPackageConsumerReport`

缺少这些 markers 时，readiness 必须继续保持 `proof=False`。

## 第三批正文门禁

### 适用读者

本文适合遇到 CUDA error 35、driver/runtime mismatch 或 compatible host blocker 的用户，也适合负责外部 runtime proof 回填的维护者。

### 解决问题

本文解决的是环境阻塞诊断：如何区分 API 缺口、驱动不兼容、runtime package 不匹配和真实 proof 缺失。

### 背景与场景

CUDA error 35 常见于 CUDA runtime 版本高于当前驱动可支持范围。它会阻断 runtime smoke，但不代表 C# wrapper、manifest 或 native bridge 本身缺失。

### 代码与文件入口

- `eng/Test-PackageConsumer.ps1`
- `artifacts/final-release/runtime-proof-compatible-host-kit.json`
- `docs/articles/zh-cn/runtime-package-installation-deep-dive.md`
- `docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md`

### 操作路径

先记录 driver、CUDA runtime、TensorRT、cuDNN、RID 和 package key，再在兼容主机复测 dependency probe 与 runtime smoke，最后把日志、hash 和 host metadata 交给 strict validator。

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。error 35 的 blocked 记录不能被写成 passed proof。

### 下一步

下一步把 error 35 排查与 runtime package installation deep dive 串起来，形成安装、排障、proof 回填一条完整文章链。
