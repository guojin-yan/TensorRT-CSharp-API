# Linux Runner 准备说明

## 目标

当前 Linux runtime 打包已经具备结构和工作流骨架，但默认假设会在 self-hosted Linux x64 runner 上执行。

## Runner 基本要求

- Linux x64 主机
- self-hosted runner 标签：`self-hosted`、`linux`、`x64`
- .NET 10 SDK
- `pwsh`
- CMake
- 从 NVIDIA 官方渠道下载并安装/解压的对应版本 CUDA Toolkit
- 从 NVIDIA 官方渠道下载并解压的对应版本 TensorRT Linux 包
- 从 NVIDIA 官方渠道下载并安装/解压的对应版本 cuDNN Linux 包

不要把 CUDA / cuDNN / TensorRT 二进制提交到 Git。当前 workflow 不在 CI 中自动登录 NVIDIA 或自动下载安装包，而是读取 self-hosted runner 上已经准备好的官方库目录。

## `runtime-linux.yml` 需要的输入

- `version`
- `runtime_keys`：逗号分隔的 Linux runtime key
- `run_smoke`：只有 runner 有可用 NVIDIA GPU、驱动和匹配 runtime 时才打开
- `publish_to_github_packages`：默认关闭，Linux runtime 大包通常保留为 GitHub Release assets
- `release_tag`
- `attach_to_github_release`

## 根目录示例

- TensorRT：`/opt/tensorrt/trt10-cuda12.9`
- CUDA：`/usr/local/cuda-12.9`
- cuDNN：`/opt/cudnn/cuda12`

root 解析顺序：

1. `pack/runtime/runtime-packages.local.json`
2. `JYPPX_RUNTIME_PACKAGE_ROOTS_FILE` 指向的 JSON 文件
3. `~/.jyppx/runtime-packages.local.json`
4. `pack/runtime/runtime-packages.manifest.json` 中的默认 Linux root

建议从 `pack/runtime/runtime-packages.local.example.json` 复制后修改。该示例已经包含 Windows 和 Linux 全矩阵。`Resolve-RuntimeRoots.ps1` 会展开 `<repo-root>`、系统环境变量和 `~`。

## 建议的 dry-run 路径

在真正执行完整 pack 前，建议先按下面步骤做静态或半静态验证：

1. `pwsh -File ./eng/Validate-RuntimeManifest.ps1`
2. `pwsh -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
3. `pwsh -File ./eng/Invoke-LinuxRuntimeDryRun.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
4. `pwsh -File ./eng/Validate-LinuxDryRunArtifacts.ps1 -RuntimePackageKey <linux key>`
5. 检查 `artifacts/linux-dry-run/<key>/linux-runtime-dry-run.json`
6. 检查 `artifacts/linux-dry-run/<key>/linux-runner-checklist.md`
7. `pwsh -File ./eng/Export-LinuxPreflightSummary.ps1 -RuntimePackageKey <linux key>`
8. `pwsh -File ./eng/Export-LinuxHandoffIndex.ps1 -RuntimePackageKey <linux key>`
9. 然后再执行：
   - `cmake --preset <linux preset>`
   - `cmake --build --preset <linux preset>`
   - `pwsh -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
   - `dotnet pack ./pack/runtime/<key>/<packageId>.csproj -c Release -o ./artifacts/runtime-nupkg`

## 当前 dry-run 会产出的文件

- `artifacts/linux-dry-run/<key>/linux-runtime-dry-run.json`
- `artifacts/linux-dry-run/<key>/README.md`
- `artifacts/linux-dry-run/<key>/linux-runner-checklist.md`
- `artifacts/linux-dry-run/<key>/linux-preflight-summary.md`
- `artifacts/linux-dry-run/<key>/linux-workflow-contract.md`
- `artifacts/linux-dry-run/<key>/linux-handoff-index.md`

配套校验脚本：

- `eng/Validate-LinuxDryRunArtifacts.ps1`
- `eng/Export-LinuxPreflightSummary.ps1`
- `eng/Test-LinuxRuntimeWorkflowContract.ps1`
- `eng/Export-LinuxHandoffIndex.ps1`

当前 workflow 侧消费顺序：

1. `Validate-RuntimeManifest`
2. `Validate-LinuxRuntimeInputs`
3. `Invoke-LinuxRuntimeDryRun`
4. `Validate-LinuxDryRunArtifacts`
5. `Test-LinuxRuntimeWorkflowContract`
6. `Export-LinuxPreflightSummary`
7. `Collect-RuntimeAssets`
8. `dotnet pack`
9. `Test-RuntimePublishReadiness`

这三个文件分别用于：

- JSON 摘要：给脚本和后续 workflow 读取
- README：给执行者快速浏览当前组合、根目录、命令和预期产物
- checklist：给 Linux runner 维护者逐项确认 preflight / build / pack / failure scenario

## 预期产物规则

- native build 产物：`build-out/<preset>/bin/Release/<bridgeFile>`
- native import library / 辅助产物：`build-out/<preset>/lib/Release/`
- runtime 资产收集目录：`artifacts/runtime/<key>/runtimes/<rid>/native/`
- runtime 资产清单：`artifacts/runtime/<key>/artifact-manifest.json`
- runtime nupkg 输出：`artifacts/runtime-nupkg/`

## 当前状态

仓库中已经补齐 Linux package manifest 与 Linux pack workflow，但在这台 Windows 工作站上还没有完成真实 Linux 打包验证。

## 优先检查的失败场景

- `TensorRT root` 与请求的 `runtime_key` 版本线不一致
- `CUDA root` 与请求的 `runtime_key` 版本线不一致
- 预期 `.so` 通配规则没有匹配到文件
- self-hosted runner 上缺少 `pwsh`
- self-hosted runner 上缺少 `dotnet` 或 `cmake`
- native build 成功，但 `Collect-RuntimeAssets.ps1` 仍失败，通常说明传入的根目录布局不匹配 manifest 约定
- artifact 上传成功，但 native 资产不完整，通常是因为传入的根目录不正确
