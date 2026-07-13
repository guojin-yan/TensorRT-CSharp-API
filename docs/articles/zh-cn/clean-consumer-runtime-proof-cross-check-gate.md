# Clean Consumer 与 Runtime Proof 交叉核对 Gate

`clean-consumer-runtime-proof-cross-check-gate` 用于交叉核对仓库外 clean consumer proof 与兼容主机 runtime proof 的 package id、version、source 和 runtime package key 一致性。

它默认 `Passed=false`，缺少真实执行日志 hash、出现 ProjectReference、direct `.nupkg`、local feed、DependencyProbe-only、driver-blocked 或 build-only 结果时，不能晋级为 runtime proof 或 post-publish proof。

## 边界

- 不是 runtime proof。
- 不是 post-publish proof。
- 不是 publish approval。
- 不是 release close approval。
- 不是 package push。

## Owner 输入

- `cleanConsumerPackageId`
- `cleanConsumerPackageVersion`
- `cleanConsumerPackageSource`
- `runtimeHostPackageId`
- `runtimeHostPackageVersion`
- `runtimePackageKey`
- `projectReferenceCount`
- `directNupkgReferenceCount`
- `localFeedReferenceCount`
- `dependencyProbeOnly`
- `driverBlocked`
- `buildOnly`

## 验证

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CleanConsumerRuntimeProofCrossCheckGate.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CleanConsumerRuntimeProofCrossCheckGate.ps1 -Strict
```
