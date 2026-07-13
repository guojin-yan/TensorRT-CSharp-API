# TensorRtSharp 4.0.0 RC 发布检查清单

## 必跑门禁

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet build .\TensorRtSharp.sln -c Debug --no-restore /p:UseSharedCompilation=false
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build /p:UseSharedCompilation=false
dotnet docfx .\docs\docfx.json
```

## 包消费门禁

```powershell
dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj -c Debug -o .\artifacts\managed -p:JYPPXPackageVersion=4.0.0 /p:UseSharedCompilation=false
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageConsumer.ps1 -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22 -BridgePackageDirectory .\artifacts\runtime-split-nupkg\win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RuntimePackageDirectory .\artifacts\runtime-nupkg -RunSmoke -AllowSmokeFailure
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LocalNuGetFeedConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowSmokeFailure
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicApiBilingualDocumentation.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicApiBilingualDocumentationBacklog.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowRuntimeSmokeBlocked
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -AllowRuntimeSmokeBlocked -WarnOnly
```

## 必须确认

- release candidate readiness 没有 blocker。
- `artifacts/release-candidate/release-candidate-checklist.md` 中的 pending 项被逐项解释；pending 不是通过失败，但发布前必须有负责人确认。
- `artifacts/api-doc-audit/public-api-bilingual-documentation-backlog.md` 只代表双语文档 finding 已分批整理，不代表双语文档 gate 已通过。
- `artifacts/final-release/final-release-dry-run-summary.md` 中的 manual approval 项被逐项确认，不能把 dry run 误写为正式发布完成。
- warnings 中明确列出 CUDA error 35、unsigned/local trust 和 real callback runtime proof=false。
- local feed consumer 不包含 `ProjectReference`。
- runtime package matrix 包含 Windows TRT8/TRT10/TRT11 与 Linux Ubuntu 20.04/22.04/24.04 x64 目标。
- 文档不得宣称 `IDebugListener::processDebugTensor` 已有真实 runtime proof，除非 full package consumer 输出 `InvocationCount>0`。
