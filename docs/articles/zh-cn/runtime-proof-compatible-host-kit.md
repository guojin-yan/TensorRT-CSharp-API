# 兼容主机 Runtime Proof 采集包

`runtime-proof-compatible-host-kit` 把兼容 GPU/CUDA/TensorRT 主机上的 runtime proof 采集任务集中为可执行 lane，但不把本地 precheck、driver-blocked、DependencyProbe-only 或 dry-run 记录晋级为真实 runtime proof。

## 当前状态

- 状态：`blocked-runtime-proof-compatible-host-kit-owner-proof-required`
- 默认结果：`Passed=false`
- 边界：`not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push`

## Owner 必填字段

- `hostId`
- `runtimePackageKey`
- `cudaVersion`
- `tensorRtVersion`
- `driverVersion`
- `cudaSmokeExitCode`
- `tensorRtSmokeExitCode`
- `packageConsumerExitCode`
- `logSha256`
- `versionGuardSummary`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RuntimeProofCompatibleHostKit.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimeProofCompatibleHostKit.ps1 -Strict
```
