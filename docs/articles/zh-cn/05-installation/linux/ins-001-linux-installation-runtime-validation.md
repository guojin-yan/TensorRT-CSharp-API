# TensorRT CSharp API v4.0 Linux 安装与运行环境验证

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：INS-001；适用版本：4.0.0；当前状态：review。

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

本文给出 Ubuntu 环境安装 TensorRT CSharp API v4.0 的分层验证方法。重点不是罗列一条安装命令，而是区分 .NET SDK、托管 NuGet 包、TensorRT/CUDA 原生库、动态链接器和真实 GPU 推理五个层次。

## 2. 先确认兼容矩阵

Linux 安装前至少需要确认：Ubuntu 版本、CPU 架构、NVIDIA 驱动、CUDA、TensorRT、cuDNN、.NET Target Framework 和项目选择的 Runtime 包。仓库运行时清单覆盖 Ubuntu 20.04、Ubuntu 22.04、Ubuntu 24.04 的 x64 运行时条目以及多个 TensorRT/CUDA 组合，但部分组合只完成干运行或清单校验。

因此，选择包时应以 `pack/runtime/runtime-packages.manifest.json` 中的实际记录为准，并阅读对应条目的验证状态。`dry-run-only` 只能证明打包输入与命名规则可解析，不能证明该组合已经完成 Linux GPU 推理。

## 3. 安装 .NET 与创建项目

先安装仓库目标框架支持的 .NET SDK，再创建一个干净控制台项目。

```bash
dotnet --info
dotnet new console -n TensorRtLinuxSmoke
cd TensorRtLinuxSmoke
dotnet add package JYPPX.TensorRT.CSharp.API --version 4.0.0
```

随后根据目标系统选择匹配的 Runtime 包。包名和可用版本应从 NuGet 页面和仓库运行时清单核对，不要根据 Windows 包名推断 Linux 包名。

```text
NuGet：
https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0

Runtime 包清单：
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json
```

## 4. 原生库与动态链接器

托管包还原成功，只证明 NuGet 依赖图成立。运行时仍需找到 TensorRT、CUDA、cuDNN 以及项目附带的本机绑定库。

建议依次检查：

```bash
nvidia-smi
ldconfig -p | grep -E 'nvinfer|cudart|cudnn'
find ./bin -type f \( -name '*.so' -o -name '*.so.*' \)
ldd ./bin/Debug/net*/runtimes/linux-*/native/*.so
```

如果依赖库安装在非系统目录，可在当前 shell 中设置 `LD_LIBRARY_PATH` 进行诊断；生产部署应使用明确的安装目录、运行时包布局或动态链接器配置，避免依赖用户登录脚本中的隐式环境变量。

```bash
export LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
```

## 5. 分层验证流程

| 层次 | 命令或动作 | 能证明什么 |
|---|---|---|
| SDK | `dotnet --info` | .NET SDK 与运行时可用 |
| 还原 | `dotnet restore` | NuGet 依赖可解析 |
| 构建 | `dotnet build -c Release` | 托管代码和资产选择可编译 |
| 加载 | 运行只创建 Logger、Runtime 的烟雾程序 | 本机绑定和基础原生库可加载 |
| 反序列化 | 加载匹配环境构建的 Engine | TensorRT Runtime 可消费 Engine |
| 推理 | 绑定真实输入并校验输出 | GPU 推理链路完整 |

帮助输出或 `--help` 只能验证命令行入口；创建 Runtime 只能验证部分原生加载；只有真实输入、成功 Enqueue 和输出校验才能作为推理成功证据。

## 6. 常见错误定位

出现 `DllNotFoundException` 时，先检查报错的具体库名、RID 资产目录和 `ldd` 输出。出现 TensorRT 反序列化错误时，检查 Engine 的 TensorRT 版本、GPU 架构、插件和构建环境。出现 CUDA 初始化错误时，检查驱动、容器设备映射、用户权限和设备可见性。

在容器中还应记录基础镜像、NVIDIA Container Toolkit、挂载的设备、容器内驱动可见性和 `dotnet --info`。不要只保存宿主机的 `nvidia-smi` 输出。

## 7. 最小证据模板

```text
OS: Ubuntu 22.04 or 24.04
Architecture: x64; arm64 需要单独构建与验证记录
NVIDIA driver: recorded
CUDA / TensorRT / cuDNN: recorded
.NET SDK: recorded
Managed package: recorded
Runtime package: recorded
dotnet restore: exit 0
dotnet build: exit 0
native dependency scan: no unresolved required library
real inference: output checksum or semantic result recorded
```

本文已完成包清单、命令和证据边界的静态复核，状态保持 `review`。正式发布前应在至少一个 Ubuntu 22.04 或 24.04 目标环境完成真实 GPU 烟雾测试，并把 Runtime 包版本与输出一并固化。

## 8. 小结

Linux 安装不是“NuGet 还原成功”这一项检查。可靠的结论必须沿着 SDK、包还原、托管构建、本机库加载、Engine 反序列化和真实推理逐层推进，并明确记录每一层能够证明和不能证明的内容。

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
