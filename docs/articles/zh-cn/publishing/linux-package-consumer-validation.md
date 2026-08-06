# Linux Package Consumer 验证规则

## 目标

Linux runtime 包不能只依赖 manifest / dry-run / workflow 结构验证。真正晋级到 `local-validated` 前，必须在 Linux x64 self-hosted runner 上验证消费端项目可以真实安装托管主包和 Linux runtime 包，并能复制 `.so` 原生资产。

## 必须执行的顺序

1. `pwsh -File ./eng/Validate-RuntimeManifest.ps1`
2. `pwsh -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
3. `pwsh -File ./eng/Invoke-LinuxRuntimeDryRun.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
4. `pwsh -File ./eng/Validate-LinuxDryRunArtifacts.ps1 -RuntimePackageKey <linux key>`
5. `pwsh -File ./eng/Test-LinuxRuntimeWorkflowContract.ps1 -RuntimePackageKey <linux key> -ConfigurePreset <preset> -BuildPreset <preset>`
6. `pwsh -File ./eng/Export-LinuxPackageConsumerPlan.ps1 -RuntimePackageKey <linux key>`
7. `cmake --preset <preset>`
8. `cmake --build --preset <preset>`
9. `pwsh -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
10. `dotnet pack ./pack/JYPPX.TensorRT.CSharp.API/JYPPX.TensorRT.CSharp.API.csproj -c Release -o ./artifacts/managed`
11. `dotnet pack ./pack/runtime/<linux key>/<packageId>.csproj -c Release -o ./artifacts/runtime-nupkg`
12. `pwsh -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey <linux key>`

## Smoke 规则

`-RunSmoke` 不是 Linux 晋级的第一门槛。只有 runner 具备以下条件时才应启用：

- NVIDIA 驱动可用
- GPU 权限可用
- CUDA / TensorRT 运行时与 runtime key 匹配
- 原生桥接库能加载对应 `.so`

## 晋级条件

Linux runtime key 从 `dry-run-only` 改为 `local-validated` 前，至少需要满足：

- Linux native bridge 构建成功
- runtime 资产收集成功
- runtime nupkg 生成成功
- `Test-PackageConsumer.ps1` 非 smoke 验证通过
- `artifacts/package-consumer/package-consumer-validation-summary.md` 留存为证据
- 如果有 GPU，则额外留存 smoke 结果

## 当前发布策略

- Linux 组合在真实 runner 验证前继续保持 `dry-run-only`
- Linux 组合不进入公开 NuGet.org 发布候选
- `TRT8` Windows 组合优先作为公开验证样板
- `TRT10` Windows 组合优先作为私有源或拆分交付候选
- NVIDIA TensorRT / CUDA 再分发许可复核仍是公开发布前阻塞项
