# YoloVision Managed Extension 发布 Dry-Run

`JYPPX.TensorRT.CSharp.API.YoloVision` 与 `JYPPX.TensorRT.CSharp.API.Classification` 都是纯 C# managed extension。它们与基础包 `JYPPX.TensorRT.CSharp.API` 一起由 `.github/workflows/package-managed.yml` 打包；CUDA、cuDNN、TensorRT、NVRTC、模型和 engine 都不属于这些 artifact。

## 日常 Actions 验证

正式仓库和 `grape-yan` 都可以手动运行零发布 dry-run：

```powershell
gh workflow run release-quality-gate.yml `
  --repo grape-yan/TensorRT-CSharp-API `
  --ref TensorRtSharp4.0 `
  -f run_package_managed_dry_run=true `
  -f run_release_artifact_audit=false `
  -f run_split_package_build=false
```

reusable workflow 的有效输入固定为：

```text
owner_publish_approved=false
publish_to_nuget=false
publish_to_github_packages=false
attach_to_github_release=false
```

`grape-yan` 的 pack job 只有 `contents: read`。它会上传三个 nupkg 和独立 validation reports artifact，但不会运行
NuGet push、GitHub Packages publish、Release create/upload 或 docs deploy。

`release-quality-gate.yml` 的 reusable caller 必须声明被调用 workflow 中所有条件 job 可能需要的权限上限，否则 GitHub 会在
job 启动前拒绝 workflow。`package-managed.yml` 会把实际 pack job 降为 `contents: read`；三个发布 job 还同时受发布输入、
`owner_publish_approved` 和正式仓库 owner 条件保护，因此在 `grape-yan` 必定 skipped。

## managed 包集合验证内容

pack job 必须依次通过：

1. 基础 managed package content 验证。
2. 精确包集合验证：只能有主包、YoloVision 与 Classification 三个固定 ID，版本完全一致。
3. 三个 nuspec 的 repository URL 与 commit 必须绑定当前 Actions SHA。
4. YoloVision 和 Classification 对基础 managed 包的 dependency version 必须一致。
5. 三个包的 native entry 和 NVIDIA vendor runtime entry 都必须为 0。
6. YoloVision package DLL/XML 必须与本次 build output 一致，公开 surface 不得泄漏 pointer、handle 或 sample-internal 类型。
7. YoloVision 的纯 managed 仓库外 consumer 仍只引用主包与 YoloVision 两个包；ProjectReference、直接 DLL、restore graph project library 都必须为 0。Classification 的真实三包运行由独立 ResNet18 消费脚本验证。

consumer 只执行 capability matrix、layout inference 和默认 options 等纯 managed 路径，固定输出：

```text
YoloVisionManagedPackageConsumer Passed=True PackageReferenceOnly=True NativeRuntimeLoaded=False
```

它不加载 bridge，也不要求 GitHub-hosted runner 安装 NVIDIA SDK。

## 本地 dry-run

在仓库根目录使用同一版本和 source commit 打包：

```powershell
$version = "4.0.0"
$commit = (git rev-parse HEAD).Trim()
$output = ".\artifacts\yolovision-managed-dry-run\packages"

dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj `
  -c Release -o $output `
  -p:JYPPXPackageVersion=$version `
  -p:RepositoryCommit=$commit `
  -p:ContinuousIntegrationBuild=true

dotnet pack .\samples\YoloVision\YoloVision.csproj `
  -c Release -o $output `
  -p:JYPPXPackageVersion=$version `
  -p:RepositoryCommit=$commit `
  -p:ContinuousIntegrationBuild=true

dotnet pack .\samples\Classification\Classification.csproj `
  -c Release -o $output `
  -p:JYPPXPackageVersion=$version `
  -p:RepositoryCommit=$commit `
  -p:ContinuousIntegrationBuild=true

pwsh -NoProfile -File .\eng\Test-YoloVisionManagedPackageDryRun.ps1 `
  -PackageDirectory $output `
  -PackageVersion $version `
  -ExpectedSourceCommit $commit
```

报告位于 `artifacts/yolovision/managed-package-dry-run`。它是本地 pack/content/surface/consumer dry-run，不是公开 feed
下载、TensorRT runtime、post-publish 或 release proof。

## 三包 Owner Handoff

Bridge 在匹配 SDK 的 runner 或本机单独构建。三个候选包就绪后，运行：

```powershell
pwsh -NoProfile -File .\eng\Export-YoloVisionPackagePublicationHandoff.ps1 `
  -ManagedPackagePath <managed-nupkg> `
  -YoloVisionPackagePath <yolovision-nupkg> `
  -BridgePackagePath <bridge-nupkg> `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -PackageVersion 4.0.0 `
  -ExpectedSourceCommit <40-char-commit>
```

handoff 只有在三个 ID、版本、source commit、SHA256、YoloVision dependency 与 Bridge native entry 全部一致时才进入
`ready-local-three-package-handoff`。它仍要求 Owner review、公开 channel 选择、发布后 URL/download hash 和公开源 clean consumer。

## 正式发布门

所有发布默认关闭。只有正式账号明确批准某一版本时，才在 `guojin-yan` 仓库传入：

```text
owner_publish_approved=true
publish_managed_to_nuget=true|false
publish_managed_to_github_packages=true|false
attach_to_github_release=true|false
```

发布前置条件通过不等于发布后证明。真正 publish 后仍必须从公开源重新下载基础 managed、YoloVision 与 Bridge，复核 URL、
版本、SHA256、source commit，并在仓库外完成真实 TensorRT clean consumer，才能进入 post-publish 审核。
