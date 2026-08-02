# TensorRtSharp 4.0.0 RC 发布检查清单

## 发布集合

- `JYPPX.TensorRT.CSharp.API 4.0.0`
- `JYPPX.TensorRT.CSharp.API.YoloVision 4.0.0`
- 目标平台与依赖组合对应的 `.Bridge 4.0.0` 包
- Git 跟踪源码与文档

CUDA、cuDNN、TensorRT、NVRTC 由用户自行安装。full-runtime、CUDA/cuDNN、TensorRT component 和 collection/meta 包不在发布集合中。

## 核心代码门禁

```powershell
dotnet restore .\TensorRtSharp.sln
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-RuntimeManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 -StaticOnly
dotnet build .\TensorRtSharp.sln -c Release --no-restore /p:UseSharedCompilation=false /m:1 /nr:false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Release --no-build `
  --filter "FullyQualifiedName~RuntimeManifestTests|FullyQualifiedName~ManagedPackageTests|FullyQualifiedName~ExternalVendorRuntimePackagePolicyTests|FullyQualifiedName~BridgePackageConsumerTests|FullyQualifiedName~YoloVisionManagedPackagePublicationTests|FullyQualifiedName~ReleaseAutomationTests|FullyQualifiedName~ReleaseQualityGateWorkflowTests"
dotnet docfx .\docs\docfx.json
```

第一版不要求执行当前 480 类完整 ProjectQuality 矩阵；核心定向门禁失败仍必须修复，不能用“后续迭代”绕过。

## Managed 包门禁

```powershell
dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj `
  -c Release -o .\artifacts\managed -p:JYPPXPackageVersion=4.0.0

dotnet pack .\samples\YoloVision\YoloVision.csproj `
  -c Release -o .\artifacts\managed -p:JYPPXPackageVersion=4.0.0

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ManagedPackageContent.ps1 `
  -PackagePath .\artifacts\managed\JYPPX.TensorRT.CSharp.API.4.0.0.nupkg

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 `
  -PackagePath .\artifacts\managed `
  -ExpectedPackageId 'JYPPX.TensorRT.CSharp.API,JYPPX.TensorRT.CSharp.API.YoloVision' `
  -ExpectedPackageVersion 4.0.0 `
  -RequireExactPackageSet

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionPackageSurface.ps1 `
  -PackagePath .\artifacts\managed\JYPPX.TensorRT.CSharp.API.YoloVision.4.0.0.nupkg `
  -PackageVersion 4.0.0

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionManagedPackageDryRun.ps1 `
  -PackageDirectory .\artifacts\managed `
  -PackageVersion 4.0.0 `
  -Configuration Release
```

`artifacts/managed` 必须是干净 staging 目录，并且精确包含上述两个 `4.0.0` 包。

## Bridge 包门禁

对第一版选定的主要 Windows TRT8/TRT10/TRT11 key 逐一执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey <runtime-key> `
  -Version 4.0.0 `
  -SplitPackageRole bridge `
  -Configuration Release

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 `
  -PackagePath .\artifacts\runtime-split-nupkg\<runtime-key>

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageConsumer.ps1 `
  -SourceRuntimeKey <runtime-key> `
  -BridgePackageDirectory .\artifacts\runtime-split-nupkg\<runtime-key>
```

bridge 包必须只包含一个项目自有 native bridge 文件。禁止运行或恢复退休的 full-runtime 打包命令。

## 发布前确认

- 两个 managed 包和每个 bridge 包均通过 external vendor runtime policy，未包含 NVIDIA 厂商二进制。
- 每个 nupkg 声明非占位的 license expression 或包含非空 license file；源码归档根目录包含 Owner 确认的许可证文件。
- managed/YoloVision clean consumer 不包含 `ProjectReference`，包版本与 source commit 对齐。
- 第一版主要演示能构建；需要 GPU 的运行结果只按实际兼容主机证据描述。
- 文档不得把 local build、dry run、local feed 或模板记录写成公开发布和 post-publish proof。
- `IDebugListener::processDebugTensor` 等 deferred callback 不得宣称已有真实 runtime proof，除非存在可复核的兼容主机调用记录。
- grape-yan 仅用于日常 Action 编译检查；两项验证工作流通过后，源码提交同步到 guojin-yan 正式仓库。
- 所有上传、GitHub Release 和 NuGet push 开关保持关闭，直到 Owner 明确批准目标版本、包清单、SHA256 和发布渠道。
- `Test-PublicationLicenseReadiness.ps1` 必须在任何 Release 创建、资产上传或 package push 之前通过；当前许可证未决时应 fail closed。
