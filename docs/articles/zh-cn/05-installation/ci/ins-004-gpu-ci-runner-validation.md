# TensorRT CSharp API v4.0 GPU CI Runner：标签、环境与发布证据

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：INS-004；适用版本：4.0.0；当前状态：review。

## 1. 前言
<!-- public-article-project-preface:start -->
TensorRT CSharp API v4.0 是一个面向 C#/.NET 开发者的 TensorRT 与 CUDA 工程化接口项目。它把 NVIDIA 原生运行时、生成式绑定、C++ Bridge、托管对象模型和可验证的示例程序组织成一条完整链路，使使用者可以在熟悉的 .NET 项目中完成 Engine 构建、反序列化、ExecutionContext 管理、CUDA 内存操作、异步流同步和结果校验。项目的目标不是隐藏 TensorRT 的概念，而是把这些概念转换为有明确生命周期、所有权和错误边界的 C# API。

4.0.0 是一次完整重构后的正式版本。核心接口、Bridge 边界、Runtime 包命名、样例目录和验证方式都以 4.x 设计为准，不能把 3.x 的类型名、旧包名或旧 DLL 目录直接复制到新项目。托管包只提供项目接口和自有 Bridge；TensorRT、CUDA、cuDNN、显卡驱动以及对应许可证仍由使用者按目标平台安装和管理。

单篇文章也应能够独立阅读：读者可以先从项目入口确认源码和包，再根据本文的程序路径准备依赖，最后用输出中的状态、计数、Shape、哈希或结果图片判断流程是否真的完成。对于尚未具备兼容 GPU 的环境，本文会把静态检查、期望输出和真实运行结果分开标记，不把帮助命令或 build-only 结果包装成推理成功。

项目、包和源码入口（以下地址保留明文，便于复制到不完整支持 Markdown 链接的平台）：

项目主页：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0
```

核心 NuGet：

```text
https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0
```

Runtime Bridge 包列表：

```text
https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance
```

运行库清单：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json
```

### 1.1 程序出处与输出说明

本文涉及的程序、脚本或命令均以仓库中的实现为准；对应源码入口：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0
```

运行示例必须同时给出程序输出和判定标准。终端中的 `status`、`Ready`、`Bound`、`enqueueCount`、`OutputMatch`、进程退出码、报告文件或结果图片分别说明不同层次的事实；只有明确写出这些结果，读者才能区分程序启动、Engine 构建、GPU enqueue 和业务结果语义。
<!-- public-article-project-preface:end -->

本文说明如何为 TensorRT CSharp API v4.0 配置 GPU CI runner，并把 runner 可用性、构建结果、运行库 smoke 和发布证据分开管理。自托管 runner 是执行环境，不是自动生成可信证明的工具。

## 2. Runner 标签与隔离

仓库 runner 应使用可组合且足够窄的标签，例如 `self-hosted`、`linux`、`x64`、`gpu` 和明确的 Ubuntu 版本。项目脚本可以先检查 GitHub API 返回的在线 runner 是否满足标签集合：

```powershell
gh auth status
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-GitHubRunnerAvailability.ps1 `
  -Repository <owner>/<repository> `
  -RequiredLabelSet "self-hosted,linux,x64,gpu" `
  -WarnOnly
```

脚本输出 `artifacts/runner-availability/github-runner-availability.json` 和 Markdown 报告。`-WarnOnly` 适合预检；发布作业应在标签不满足时失败，而不是退回到不兼容的 hosted runner。

自托管 runner 应使用专用账号、最小权限和隔离工作目录。来自不可信 Pull Request 的代码不能接触发布凭据、长期缓存或生产 GPU；使用容器或一次性 runner 时仍需限制设备和网络权限。

## 3. 先验证环境，再验证项目

runner 作业开头保存以下信息：

```bash
uname -a
cat /etc/os-release
uname -m
nvidia-smi
dotnet --info
cmake --version
ninja --version
```

项目 Linux runner 目前按 Ubuntu 20.04、22.04、24.04 和 x64 组合组织。Ubuntu 20.04 使用 hosted-container 模式时，宿主机标签、容器镜像和容器内用户态库必须一起记录。ARM64、Jetson/L4T 和非 Ubuntu 发行版不能借用 x64 proof。

## 4. 运行仓库的 Linux 证据链

先校验清单和输入，再配置、构建和 dry-run：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-RuntimeManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-LinuxRuntimeInputs.ps1 `
  -RuntimePackageKey <linux-runtime-key> `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LinuxRuntimeDryRun.ps1 `
  -RuntimePackageKey <linux-runtime-key> `
  -TensorRtRoot <TensorRT-root> `
  -CudaRoot <CUDA-root>
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-LinuxDryRunArtifacts.ps1 `
  -RuntimePackageKey <linux-runtime-key>
```

选择对应的 CMake preset 后再构建：

```bash
cmake --preset linux-x64-trt11-cuda13-release
cmake --build --preset linux-x64-trt11-cuda13-release --parallel
```

真实 GPU 验证必须在目标 runner 上执行，并保存进程退出码、TensorRT/CUDA/cuDNN 版本、Bridge 包、Engine、输入、输出和日志。只在 Windows、WSL 或本地容器中通过的结果不能升级为 Linux runner proof。

仓库提供专用工作流 `.github/workflows/runtime-linux-gpu-smoke.yml`。它不会回退到 hosted runner，作业固定路由到 `self-hosted`、`linux`、`x64`、`ubuntu-24.04`、`gpu` 五个标签，并依次校验 Ubuntu 版本、`nvidia-smi`、NVIDIA roots、Native Bridge、managed/Bridge 包、仓库外最小 consumer、真实 enqueue、同步和 identity 输出：

```powershell
gh workflow run runtime-linux-gpu-smoke.yml `
  --ref TensorRtSharp4.0 `
  -f runtime_key=linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22 `
  -f version=4.0.0
```

触发前应先以不带 `-WarnOnly` 的 runner 可用性检查作为门禁。没有匹配的在线 runner 时不要触发后长期等待，也不要删减标签以便让其他机器接单：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-GitHubRunnerAvailability.ps1 `
  -Repository guojin-yan/TensorRT-CSharp-API `
  -RequiredLabelSet "self-hosted,linux,x64,ubuntu-24.04,gpu"
```

## 5. Runner 状态与 proof pack

可以用项目脚本生成当前执行状态：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-LinuxRunnerExecutionStatus.ps1 `
  -RuntimePackageKey <linux-runtime-key>
```

该报告会明确当前主机是否为 Linux、x64、GitHub Actions session，以及还缺少哪些证据。最终 proof pack 由 Owner 填写 runner 日志、GPU、驱动、CUDA、TensorRT、包 SHA256 和验证命令，再由：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerProofPack.ps1 `
  -PackPath .\artifacts\final-release\linux-runner-proof-execution-pack.json
```

校验。校验通过也只表示 proof candidate 满足字段和哈希格式；脚本不会发布包、关闭 Release issue 或把 WSL/dry-run 记录变成真实 Linux proof。

专用工作流成功后会上传 `runtime-linux-gpu-smoke-<runtime-key>` artifact，其中包括 runner 环境日志、Linux dry-run、managed/Bridge nupkg、包消费者报告、最小 GPU consumer 的 JSON/Markdown/原始日志和 `sha256-inventory.json`。INS-004 的晋级复核必须确认工作流路径确为 `runtime-linux-gpu-smoke.yml`、作业实际落在五标签 runner 上，并且最小 consumer 报告的 `runtimeExecutionProof=true`、`canPromoteGpuCiRuntimeProof=true`；仅有 artifact 名称或成功的 build step 不够。

## 6. CI 缓存与失败复现

缓存键必须包含 OS、架构、TensorRT line、CUDA line、cuDNN major、.NET SDK 和项目提交。清理旧的 `build-out/<preset>`、NuGet 本地缓存和模型缓存时只删除当前作业拥有的目录。失败报告应包含 runner label、实际 runner name、镜像 digest（如有）、环境变量中与库搜索路径相关的非机密部分，以及完整 loader 错误。

不要因为 runner 暂时离线而把作业改成 `WarnOnly` 后继续发布；预检和发布门禁应使用不同策略。发布作业还要确认生成包的 RID、内容 allowlist、SHA256、消费者 restore 和 post-publish 可见性。

## 7. 证据边界矩阵

| 证据 | 结论 | 不足之处 |
| --- | --- | --- |
| API runner 查询 | 标签和在线状态 | 没有项目构建结果 |
| 环境信息 | OS、架构、GPU 和 SDK | 没有库加载结果 |
| CMake build | Native Bridge 可构建 | 没有 Runtime 推理 |
| dry-run | 输入、清单和目录规则正确 | 没有真实 GPU 执行 |
| runner smoke | 目标组合执行成功 | 只覆盖该 runner/组合 |
| proof pack 校验 | Owner 字段和哈希完整 | 不执行发布动作 |

## 8. 小结

2026-08-14 通过 GitHub API 再次查询仓库 runner，结果仍为 `total_count=0`，当前没有可执行本文 GPU CI smoke 的 self-hosted runner。使用 `Test-GitHubRunnerAvailability.ps1 -Repository guojin-yan/TensorRT-CSharp-API -RequiredLabelSet 'self-hosted,linux,x64,ubuntu-24.04,gpu' -WarnOnly` 的结构化结果为 `querySucceeded=true`、`runnerCount=0`、`matchingRunnerCount=0`、`onlineMatchingRunnerCount=0`；报告位于 `artifacts/runner-availability/github-runner-availability.json`。仓库现已补齐固定五标签的 `runtime-linux-gpu-smoke.yml` 和可复用的最小 PackageReference-only GPU consumer，但工作流尚未在目标 runner 上执行，不能据此晋级。历史 GitHub Actions 运行 `31412970912` 与 `31412959494` 只证明 Ubuntu hosted container 的 restore、绑定生成、Native build、dry-run 与 pack 曾成功；同日完成的本地 Docker GPU 推理也不在 GitHub Actions session 中。审计记录和报告哈希见 `docs/articles/zh-cn/05-installation/installation-runtime-evidence-20260814.json`。

GPU CI 的可信链路是标签匹配、环境快照、清单校验、Native 构建、真实 smoke、包消费者验证和 proof pack。任何一层缺失，都应保留为 review 或 blocked，而不是通过修改 runner 标签或切换环境来掩盖缺口。本文继续保持 `review`，下一步需要上线具备明确标签的 Linux GPU runner 并固化完整 proof pack。

<!-- public-article-declaration:start -->
## 9. 文章声明

### 9.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 9.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 9.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 9.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 9.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
