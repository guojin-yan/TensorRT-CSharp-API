# Managed + Bridge 安装与排障深入

TensorRtSharp4.0 把 managed API、项目自有 native bridge 和用户安装的 NVIDIA runtime 分层。本文从用户安装角度解释如何选择 `.Bridge` package、配置本机依赖，以及如何判断错误来自包、loader、驱动还是 proof 缺失。

## 适用读者

适合准备安装 NuGet 包的用户，也适合负责 Windows/Linux bridge 发布矩阵和 clean consumer proof 的维护者。

## 解决问题

TensorRT、CUDA、cuDNN 版本组合很多。restore 成功、bridge copied、dependency probe passed 和 runtime proof 经常被混为一谈。本文固定当前策略：包内只有 managed + bridge，NVIDIA dependencies 永远来自 host installation。

## 背景与场景

Windows x64、Ubuntu 20.04/22.04/24.04、CUDA 11/12/13、TensorRT 8/10/11 可能对应不同 runtime key。选择 key 既决定 bridge 的 ABI，也决定用户必须安装的 vendor 版本和 driver 要求。

历史 vendor split、collection 和 full package 已退休。manifest 中的旧 identity 只用于清理审计，不能重新 pack、push 或上传。

## 操作路径

1. 从 `pack/runtime/runtime-packages.manifest.json` 选择目标 runtime key。
2. 按 NVIDIA 官方方式安装匹配 TensorRT、CUDA、cuDNN 和可选 NVRTC。
3. 从批准的公开来源 restore `JYPPX.TensorRT.CSharp.API` 与一个 `.Bridge` package。
4. 运行 package content gate，确认 nupkg 没有 vendor binary。
5. 运行 dependency probe，记录系统安装依赖的 resolved path/version。
6. 在兼容 GPU host 执行 runtime smoke，并保存 host metadata、runtime JSON、stdout/stderr 和 SHA256。

本地发布前可运行 `eng/Test-BridgePackageRuntimeConsumer.ps1`。它在仓库外生成仅含 managed + `.Bridge` 两个
`PackageReference` 的 consumer；TRT10/TRT11 还必须真实触发 DebugListener callback，并验证 invocation>0、
failure/in-flight=0、copied metadata、无 borrowed pointer 暴露和 detach>0。报告只会把该结果标记为
`local-package`，不会把它提升成 `public-package` 或 `post-publish`。

```powershell
dotnet add package JYPPX.TensorRT.CSharp.API --version <version> --source <approved-source>
dotnet add package JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge --version <version> --source <approved-source>
dotnet restore --force-evaluate
dotnet build -c Release
```

## 代码与文件入口

- `pack/external-vendor-runtime-policy.json`
- `pack/runtime/runtime-packages.manifest.json`
- `pack/runtime-split/split-runtime-packages.manifest.json`
- `eng/Test-ExternalVendorRuntimePackagePolicy.ps1`
- `eng/Invoke-PublicReleaseBridgePackageConsumer.ps1`
- `eng/Test-PublicReleaseBridgePackageConsumer.ps1`
- `eng/Test-BridgePackageRuntimeConsumer.ps1`
- `docs/articles/zh-cn/runtime-packages.md`

## 主机依赖检查

Windows 使用 `where.exe`、`dumpbin /dependents` 和项目 dependency diagnostics 区分 bridge 缺失与 vendor dependency 缺失；Linux 使用 `ldd`、`ldconfig` 和实际 loader path。输出必须说明 `VendorDependenciesSource=host-installed`，不能暗示 NVIDIA 文件来自 nupkg。

NVRTC 是可选能力。没有使用 runtime compilation 的 consumer 不应仅因为未安装 NVRTC 而无法加载基础 bridge；使用 RTC 时则必须同时验证 NVRTC 与 matching builtins。

## 两条公开来源

GitHub Release verified staging 校验下载资产的 immutable URL、digest、SHA256、nuspec identity 和 repository commit。NuGet-compatible source 按 package id/version restore。managed 与 bridge source commit 必须一致；`-AllowCrossCommitPair` 只生成 diagnostic-only 结果。

## 常见排障

- `DllNotFoundException`：先区分 bridge 自身缺失还是主机 NVIDIA 依赖缺失。
- `BadImageFormatException`：检查 x64、RID、TensorRT ABI line 和 OS。
- CUDA error 35：记录 `blocked-by-cuda-driver`，换兼容主机，不要删除 blocker。
- PATH 污染：在干净 shell 只保留目标版本路径后复测，并记录 resolved DLL/`.so`。
- restore/build 成功但 runtime 失败：继续检查 dependency probe、engine deserialize/enqueue 和输出校验。

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。安装成功不等于 public package proof，也不等于 post-publish verification。

bridge nupkg hash、依赖探针和主机资产清单仍不能替代真实 runtime execution。只有仓库外 clean consumer 使用公开来源并通过 strict validator，才可能形成 package-consumer-runtime proof。

## 下一步

用 `Invoke-PublicReleaseBridgePackageConsumer.ps1` 执行 GitHub Release 路线，再对 NuGet-compatible source 执行独立 consumer。两条路线都要绑定同提交 provenance、真实日志 hash 与 owner review。
