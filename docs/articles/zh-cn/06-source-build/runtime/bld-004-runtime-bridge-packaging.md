# TensorRT CSharp API v4.0 Runtime Bridge 打包：Split Package、Manifest 与消费者验证

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->


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

本文说明如何构建和审计项目自有 Runtime Bridge 包。当前交付策略是 split delivery：Bridge 包只携带项目自有 native 二进制，CUDA、cuDNN、TensorRT、NVRTC、parser、plugin 和 builder-resource 库由消费者按目标环境安装。

## 2. 先读两份清单

完整运行时清单定义 source runtime key、RID、平台、TensorRT/CUDA/cuDNN 组合和验证状态；split 清单定义 Bridge 包的角色、包 ID 和资产分配：

```text
pack/runtime/runtime-packages.manifest.json
pack/runtime-split/split-runtime-packages.manifest.json
```

Bridge 包在 `runtimes/<rid>/native` 下应只有项目自有库：Windows 为 `jyppxtrtbridge.dll`，Linux 为 `libjyppxtrtbridge.so`。任何 NVIDIA 厂商库进入 Bridge nupkg 都是包内容违规。

## 3. 构建一个 Bridge 包

从清单选择真实的 source runtime key，使用 split 入口构建本地包：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole bridge
```

输出位于 `artifacts/runtime-split-nupkg/<source-runtime-key>`。脚本还会记录本地 split validation summary；除非显式启用对应消费者或 smoke 参数，否则不会把 pack 过程当成运行时证明。

不要使用已经退役的完整 Runtime 资产收集入口：项目不再把 NVIDIA 二进制复制进项目包。若需要构建多个组合，应逐个使用清单 key，避免把不同 CUDA/TensorRT 线混在同一个包目录。

## 4. 包内容审计

构建后先验证 split 清单和外部厂商运行库策略：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Validate-SplitRuntimePackages.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 `
  -PackagePath .\artifacts\runtime-split-nupkg\win-x64-trt11.0-cuda12.9-cudnn9.22
```

审计项目至少包括：包 ID 与 RID 一致、Bridge 文件数量正确、没有 `nvinfer`/`cudart`/`cudnn` 等厂商资产、依赖版本没有被意外写成项目包内文件、nupkg 内容可由消费者复现。

## 5. 外部 PackageReference 消费者

包内容通过后，使用独立目录或专门的 package consumer 还原 Bridge 包，不要直接引用源码项目：

```powershell
dotnet new console -n BridgePackageConsumer
cd BridgePackageConsumer
dotnet add package JYPPX.TensorRT.CSharp.API --version 4.0.0
dotnet add package <matching-Bridge-package-id> --version 4.0.0
dotnet restore
dotnet build -c Release --no-restore
```

消费者运行时仍必须能从系统 loader 找到匹配的 CUDA、cuDNN 和 TensorRT。建议同时执行 `ldd` 或 Windows `dumpbin`，保存 restore、build、Bridge 路径和实际加载库的证据。

仓库中的 Bridge consumer 验证还可以覆盖 restore、Bridge 布局、托管包装器、依赖诊断以及 TensorRT 10/11 的 DebugListener 生命周期。local package consumer、public package consumer 和 post-publish consumer 必须在报告中分开。

## 6. 版本、哈希与发布边界

每个候选包应记录：

```text
source runtime key
package ID and version
RID and target architecture
Bridge file list and SHA256
nupkg SHA256
managed package version
external CUDA/cuDNN/TensorRT versions
consumer restore/build/smoke result
publication feed and visibility result
```

本地 nupkg 通过不等于公开 feed 可见，公开 feed 可见也不等于 post-publish 机器能运行。发布前还要通过许可证、包大小、allowlist、签名策略和消费者验证。

## 7. 常见错误

如果脚本提示 source key 不存在，回到 `runtime-packages.manifest.json` 选择完整 key；如果 split 校验提示资产重复或遗漏，修复 split manifest 而不是手工删除 nupkg 文件；如果消费者找不到 `libjyppxtrtbridge.so` 或 DLL，检查 RID、输出目录和包内路径；如果 Bridge 加载后 TensorRT 初始化失败，检查外部 NVIDIA 库而不是把它们塞回 Bridge 包。

## 8. 小结

Runtime Bridge 打包的核心是“项目资产内置、厂商资产外置、清单驱动、消费者独立验证”。只有 split 清单、包内容策略、本地消费者、真实运行库和发布后验证都分层记录，候选包才具备进入发布评审的依据。

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
