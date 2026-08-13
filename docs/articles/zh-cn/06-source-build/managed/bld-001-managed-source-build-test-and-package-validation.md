# TensorRT CSharp API v4.0 托管层源码编译、测试与本地包验证

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：BLD-001；适用版本：4.0.0；当前状态：ready。

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

本文说明如何从仓库源码完成托管层还原、编译、测试、打包和干净消费者验证。源码构建成功、项目内测试通过和 NuGet 包可被外部项目消费是三个不同结论，需要分别保留证据。

## 2. 构建前检查

在仓库 `TensorRtSharp4.0` 目录执行：

```powershell
dotnet --info
dotnet restore .\TensorRtSharp4.0.sln
dotnet build .\TensorRtSharp4.0.sln -c Release --no-restore
```

构建前应记录当前提交、工作树状态、.NET SDK 版本、目标框架和运行系统。若工作树包含本地修改，报告中应说明构建的是哪个源码状态，避免把未提交修改的结果归到公开提交。

## 3. 测试分层

仓库质量测试覆盖公共 API 形状、包布局、文档、样例和发布资产等静态或可重复检查。GPU、TensorRT Runtime 和真实模型相关测试则依赖目标机器环境。

```powershell
dotnet test `
  .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj `
  -c Release `
  --no-build
```

测试报告至少应记录总数、通过数、失败数、跳过数和退出码。跳过的 GPU 测试不能被统计为真实运行通过，环境缺失也不应通过修改断言来隐藏。

## 4. 生成本地 NuGet 包

先把输出集中到独立目录，便于检查和清理。

```powershell
New-Item -ItemType Directory -Force .\artifacts\managed | Out-Null

dotnet pack `
  .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj `
  -c Release `
  -o .\artifacts\managed `
  -p:JYPPXPackageVersion=4.0.0
```

如需同时验证 CUDA 托管层，应对 `JYPPX.CudaSharp` 执行同样流程，并确认 TensorRT 包引用的版本与本地输出一致。不要让验证项目意外从 nuget.org 解析到同名旧版本。

## 5. 检查包内容

`.nupkg` 是 ZIP 容器。应检查 Nuspec、目标框架目录、XML 文档、依赖声明和必要的构建资产，而不是只判断文件是否生成。

```powershell
Get-ChildItem .\artifacts\managed\*.nupkg

dotnet nuget locals all --list
```

包内容需要满足：版本与预期一致、程序集位于正确的 `lib/<tfm>` 目录、依赖版本没有漂移、不包含源码树中的临时文件，并且公开文档所使用的包 ID 与实际 Nuspec 一致。

## 6. 干净消费者验证

最重要的一步是在仓库外或临时目录创建全新项目，只从本地源还原刚生成的包。

```powershell
$consumer = Join-Path $env:TEMP ("TensorRtSharpConsumer-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $consumer | Out-Null

dotnet new console -n Smoke -o $consumer
dotnet add $consumer\Smoke.csproj package JYPPX.TensorRT.CSharp.API `
  --version 4.0.0 `
  --source .\artifacts\managed
dotnet build $consumer\Smoke.csproj -c Release
```

验证时应检查生成的 `project.assets.json`，确认包来自本地输出目录。若项目还需要 Runtime 包，则应加入与当前系统匹配的本地 Runtime 包，并继续执行本机库加载和真实推理验证。

## 7. 源码入口

```text
解决方案：
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/TensorRtSharp4.0.sln

项目质量测试：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/tests/JYPPX.ProjectQuality.Tests

托管项目源码：
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/src
```

## 8. 结果判定与边界

一次完整的托管层验证应满足：Restore、Build、Test、Pack 和干净消费者 Build 均以退出码 `0` 结束；包内容检查通过；消费者解析到本地版本；文档中的包 ID、版本和目标框架与产物一致。

2026-08-13 的复核以提交 `3c2cbb7fb4bd66c3f99e7168e1bd5f5b0a325598` 为基线，并明确包含当时工作树中的源码修改。该源码状态完成 locked restore 和 Release solution build，结果为 `0 warnings / 0 errors`；随后生成 `JYPPX.TensorRT.CSharp.API.4.0.0.nupkg`，包大小为 `15597221` 字节，SHA256 为 `b0840e4cb120b47ee18976d8bff540088f45b19b63e2bfed092f626c26f22f53`。独立 `net8.0` 消费者只使用该本地源还原 `JYPPX.TensorRT.CSharp.API/4.0.0`，restore 与 Release build 均成功。

同一轮也在干净基线提交上执行了对照：生成器检查通过，但 solution build 因工作树中的配套源码尚未进入该提交而失败。因此上述成功结果严格归属于证据记录中的源码状态，不能归为干净提交构建证明。完整命令、产物哈希和边界见 `docs/articles/zh-cn/06-source-build/source-build-evidence-20260813.json`。

这些结果仍不能替代真实 TensorRT/CUDA 推理、公开 NuGet 消费或 post-publish 验证。托管包可以成功编译和加载，但目标机器上的原生库、GPU、Engine 或插件仍可能不兼容。本文晋级 `ready` 只表示文章命令、包入口和本地消费者链路已经复核。

## 9. 小结

源码编译是起点，不是发布结论。只有把测试、包内容、干净消费者和运行时环境分别验证，才能证明仓库源码能够转化为可被外部项目稳定消费的交付物。

<!-- public-article-declaration:start -->
## 10. 文章声明

### 10.1 开源协议声明
作者所有开源项目代码均遵循 Apache License 2.0 开源协议。
特别说明：本项目集成了若干第三方库。若任何第三方库的许可协议与 Apache 2.0 协议存在冲突或不一致，均以该第三方库的原始许可协议为准。本项目不包含也不代表这些第三方库的授权声明，使用前请务必阅读并遵守第三方库的相关许可。

### 10.2 代码开发与质量说明
AI 辅助开发：本代码在开发过程中使用了人工智能（AI）辅助生成与优化，并非完全由人工逐行编写。
安全性承诺：作者郑重声明，本代码中绝无任何有意设置的后门、病毒、木马或旨在破坏用户设备、窃取数据的恶意代码。
技术局限性：受限于作者个人的技术水平与能力，代码中可能存在因逻辑不严谨、优化不足或经验欠缺导致的低级问题（例如但不限于内存泄漏、偶发崩溃、资源未释放等）。这些问题纯属能力不足所致，并非主观故意。
测试范围：由于作者精力有限，未对本软件进行全方位、覆盖所有边缘场景的完整测试。

### 10.3 免责声明（重要）
请在将本代码应用于任何实际项目（特别是商业、工业或关键任务环境）之前，务必进行详尽、严格的自行测试与验证。 鉴于上述可能存在的代码缺陷及测试覆盖不足，因使用本代码而导致的任何直接或间接损失（包括但不限于设备故障、数据丢失、系统瘫痪或利润损失等），本作者概不负责。 一旦您开始使用本代码，即表示您已知晓上述风险并同意自行承担一切后果，相关问题与本作者无关。

### 10.4 代码开源范围
本项目承诺核心逻辑代码完全开源，但上述提到的“第三方库”的二进制文件、源代码或相关资源不在本项目的开源义务范围内，请根据其各自的指引获取。

### 10.5 社区与反馈
尽管存在上述不足，我们仍欢迎大家下载使用、提交 Issue 或参与测试，共同完善项目。如果您在使用过程中发现 Bug、内存溢出或有改进建议，欢迎通过项目主页提供的联系方式与作者取得联系，我们将尽力在有限的时间内提供协助。

<img src="../../../../images/personal-contact-banner-v6-zh.png" alt="作者联系方式" width="640" style="display:block;max-width:100%;height:auto;margin:16px auto;" />
<!-- public-article-declaration:end -->
