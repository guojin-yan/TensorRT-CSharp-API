# TensorRT CSharp API v4.0 WSL 安装：GPU 透传、运行库与边界验证

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：INS-002；适用版本：4.0.0；当前状态：review。

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

本文说明在 Windows 主机上使用 WSL2 验证 TensorRT CSharp API v4.0 的方法。WSL 的 GPU 透传、Linux 用户态 CUDA/TensorRT 运行库、.NET 项目和项目自有 Bridge 必须分别验证；Windows 主机能看到显卡，不等于 WSL 内的 TensorRT 推理已经成立。

## 2. WSL 主机与发行版

在管理员 PowerShell 中确认 WSL 状态，并选择项目支持的 Ubuntu 发行版：

```powershell
wsl --status
wsl --update
wsl -l -v
```

发行版必须显示版本 `2`。WSL 相关平台规则以以下官方文档为准：

```text
https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
https://docs.nvidia.com/cuda/wsl-user-guide/index.html
```

WSL GPU 方案依赖 Windows 侧 NVIDIA 驱动提供透传能力。不要把 Linux 驱动安装命令当成 WSL 的必需步骤，也不要把 Windows `nvidia-smi` 输出复制成 Linux 运行证据。

## 3. WSL 内部 GPU 与工具链

进入发行版后，保存系统、架构、驱动和 .NET 信息：

```bash
uname -a
cat /etc/os-release
uname -m
nvidia-smi
dotnet --info
```

`nvidia-smi` 成功只证明设备可以被 WSL 看到。继续检查 WSL 注入的库目录和动态链接器：

```bash
ls -l /usr/lib/wsl/lib
ldconfig -p | grep -E 'libcuda|libnvidia-ml'
```

如果发行版没有 .NET SDK，按目标框架安装后再执行 `dotnet --info`。不要在 WSL 中把 Windows 的 `dotnet`、`PATH` 或 `CUDA_PATH` 当作 Linux 依赖使用。

## 4. 安装项目依赖

在 Linux 工作区创建最小项目并还原核心包：

```bash
dotnet new console -n TensorRtWslSmoke
cd TensorRtWslSmoke
dotnet add package JYPPX.TensorRT.CSharp.API --version 4.0.0
dotnet restore
dotnet build -c Release --no-restore
```

Runtime Bridge 的完整包 ID、RID、TensorRT、CUDA 和 cuDNN 组合必须从仓库清单选择：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json
```

包还原完成不代表 NVIDIA 运行库已安装。按清单中的 Linux 条目准备匹配的 TensorRT、CUDA 和 cuDNN，并检查动态链接器：

```bash
ldconfig -p | grep -E 'nvinfer|nvonnxparser|cudart|cudnn'
find ./bin -type f \( -name '*.so' -o -name '*.so.*' \)
```

临时诊断可以使用 `LD_LIBRARY_PATH`，生产部署应固定系统库目录或明确的进程启动环境：

```bash
export LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```

## 5. 分层运行验证

建议按以下顺序记录结果：

| 层次 | 证据 | 结论边界 |
| --- | --- | --- |
| WSL | `wsl -l -v`、`uname -m` | 发行版与架构可用 |
| GPU 透传 | WSL 内 `nvidia-smi` | 设备对 Linux 可见 |
| .NET | `dotnet --info`、restore、build | 托管项目可构建 |
| 动态库 | `ldconfig`、`ldd` | 依赖库可解析 |
| TensorRT | 创建 Runtime、反序列化 Engine | TensorRT 用户态运行库可用 |
| 推理 | 真实输入、enqueue、输出校验 | 只能由真实 GPU smoke 证明 |

仓库已有 Linux dry-run 流程，可在 WSL 中用于检查输入和目录，但脚本会把 WSL 视为非 GitHub Actions Linux runner。不要用 dry-run 或 `--help` 输出替代 runner proof：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-RuntimeManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey <linux-runtime-key> -TensorRtRoot <TensorRT-root> -CudaRoot <CUDA-root>
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LinuxRuntimeDryRun.ps1 -RuntimePackageKey <linux-runtime-key> -TensorRtRoot <TensorRT-root> -CudaRoot <CUDA-root>
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-LinuxDryRunArtifacts.ps1 -RuntimePackageKey <linux-runtime-key>
```

### 5.1 WSL 专用环境审计

从 Windows PowerShell 运行 WSL 专用导出器。它从 WSL 注册信息中选择独立 Ubuntu 发行版，拒绝把 `docker-desktop` 当成 Ubuntu 证明，并在发行版内检查架构、`/dev/dxg`、`nvidia-smi`、.NET、PowerShell 和动态库：

```powershell
$runtimeKey = "linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22"
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-WslRuntimeEvidence.ps1 `
  -RuntimePackageKey $runtimeKey `
  -Distribution "Ubuntu-24.04"
```

报告输出到 `artifacts/wsl-runtime/<runtime-key>/wsl-runtime-evidence.json`。`status=wsl-environment-ready-runtime-proof-required` 只表示环境预检通过；导出器本身不执行 TensorRT，不会把环境预检晋级为推理证明。

### 5.2 最小 PackageReference-only GPU consumer

在 Ubuntu WSL 的 Linux 文件系统内进入仓库，先构建本地 managed 与 Bridge 包，再运行固定的最小 identity consumer：

```powershell
$runtimeKey = "linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22"
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey $runtimeKey `
  -Version 4.0.0 `
  -SplitPackageRole bridge `
  -Configuration Release
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-MinimalLinuxBridgePackageRuntimeConsumer.ps1 `
  -SourceRuntimeKey $runtimeKey `
  -ManagedPackageDirectory ./artifacts/managed `
  -BridgePackageDirectory "./artifacts/runtime-split-nupkg/$runtimeKey"
```

该脚本在仓库外创建临时 consumer，只生成两个 `PackageReference`，不使用 `ProjectReference` 或直接程序集引用。只有报告同时包含 `ReadyForEnqueue=True`、`EnqueueCompleted=True`、`StreamSynchronized=True`、`IdentityOutputMatch=True` 和退出码 `0`，才是基础安装链路的真实 GPU 运行候选。报告默认位于 `artifacts/minimal-linux-bridge-runtime/<runtime-key>/minimal-linux-bridge-package-runtime-consumer.json`。

最后从 Windows 侧再次运行导出器，并用 `-RuntimeConsumerEvidencePath` 指向这份报告。只有环境检查和 consumer 合同同时通过时，WSL 报告才会成为 `wsl-gpu-runtime-proof-candidate`；容器或 GitHub Actions 的相邻报告不会被接受。

## 6. WSL 特有的失败边界

`nvidia-smi` 在 Windows 成功、WSL 失败时，先检查 WSL 内核、发行版版本、Windows 驱动和 `/usr/lib/wsl/lib`，不要先修改项目代码。动态库加载失败时，记录缺失库的完整名称、`ldd` 输出和实际 `LD_LIBRARY_PATH`。Engine 反序列化失败时，重新核对 Engine 构建环境与 WSL 中的 TensorRT/GPU 组合。

WSL 文件系统还会影响 I/O 和路径：源码、构建目录和模型缓存应优先放在发行版的 Linux 文件系统中；跨 `/mnt/c` 访问时，报告中记录实际路径和权限。Windows 路径不能直接作为 `LD_LIBRARY_PATH`，Linux 路径也不能直接传给 Windows native loader。

## 7. 证据记录模板

```text
Host OS and NVIDIA driver: recorded
WSL distro/version: recorded
WSL kernel and architecture: recorded
WSL nvidia-smi: exit 0; output recorded
.NET SDK: recorded
Managed package: JYPPX.TensorRT.CSharp.API 4.0.0
Runtime key and Bridge package: recorded
TensorRT/CUDA/cuDNN user-space versions: recorded
ldd unresolved libraries: none or explicitly listed
real GPU inference: output checksum or semantic result recorded
```

WSL 适合作为开发与复现环境，也可以帮助定位 Linux 用户态问题；没有 GitHub Actions Linux runner 的真实执行日志、包哈希和 proof pack 时，本文结果仍只能标记为本地验证。

## 8. 小结

2026-08-14 再次执行专用 `Export-WslRuntimeEvidence.ps1`，报告显示 `status=blocked`、`registeredUbuntuDistributionCount=0`、`dockerDesktopDistributionPresent=true`、`environmentReady=false`，唯一阻塞项是没有注册独立 Ubuntu WSL 发行版。报告位于 `artifacts/wsl-runtime/linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22/wsl-runtime-evidence.json`，SHA256 为 `b08b815e8433358e9f979e937909f9b574a237eac1feb978a699030769c9bb35`。因此没有执行 WSL Ubuntu 内的 TensorRT 安装、Bridge 加载、GPU enqueue 与输出校验，文章继续保持 `review`。同日完成的 Ubuntu 24.04 Docker GPU 与项目包消费者证明属于容器边界，不能替代 WSL Ubuntu 证明。环境审计和报告哈希见 `docs/articles/zh-cn/05-installation/installation-runtime-evidence-20260814.json`。

可靠的 WSL 结论必须同时包含 WSL2 发行版、GPU 透传、Linux 用户态依赖、Bridge 加载和真实推理证据。Windows 主机可见、NuGet 还原成功或 dry-run 通过，都不能单独证明 TensorRT CSharp API 已经完成 WSL GPU 推理。

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
