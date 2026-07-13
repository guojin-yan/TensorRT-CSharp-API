# Linux Runtime Handoff

本文说明当前 Linux runtime 包线的交接状态。它面向接手 Linux x64 runner 的发布或验证负责人，重点是区分 handoff/dry-run 证据和真实 Linux runner proof。

## 当前状态

代表性 runtime key：

```text
linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
```

当前工作区已生成 handoff 文件：

- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-handoff-index.md`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runtime-dry-run.json`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-execution-status.json`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-package-consumer-plan.md`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-validation.md`

`linux-runner-execution-status.json` 当前记录的是 Windows 主机生成的 dry-run/handoff：`isLinux=false`、`status=blocked`。这不是 Linux runner proof。

## 不能误读

- Linux handoff present 不等于 Linux runtime package 已经真实通过。
- Windows 主机生成的 `dry-run-only` 不等于 Linux x64 runner build proof。
- `blocked-by-cuda-driver` 是 Windows 当前环境的 runtime smoke 阻塞，不是 Linux proof，也不是 API proof。
- `ready-needs-manual-approval` 说明自动证据链无 blocker，但仍需要 release owner 处理 Linux、签名和 runtime smoke 决策。

## Linux Runner 必跑顺序

在 Linux x64 self-hosted runner 上执行：

```bash
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
cmake --preset linux-x64-trt11-cuda13-release
cmake --build --preset linux-x64-trt11-cuda13-release --parallel
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
dotnet pack ./pack/runtime/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda13.2.cudnn9.22.csproj -c Release -o ./artifacts/runtime-nupkg
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
```

有可用 NVIDIA driver、CUDA runtime 兼容性和 GPU 访问时，再追加：

```bash
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22 -RunSmoke
```

## 晋级条件

Linux runtime package 从 handoff 晋级为真实 Linux proof，至少需要：

- 当前 runner 是 Linux x64。
- TensorRT、CUDA、cuDNN root 路径解析成功。
- CMake configure/build 成功。
- `Collect-RuntimeAssets.ps1` 收集所有声明的 `.so` pattern。
- Linux runtime `.nupkg` 生成。
- package consumer restore/build/native-copy 通过。
- `Test-LinuxRunnerEvidenceRecord.ps1` 输出 `validationState=real-linux-runner-proof`。
- 如声明 runtime smoke 通过，则必须有兼容 driver 和真实 smoke 输出。

真实 callback runtime proof 仍然独立判断：只有 full package consumer 输出 `InvocationCount>0` 且 `IsRealCallbackRuntimeProof=True`，才能晋级。
