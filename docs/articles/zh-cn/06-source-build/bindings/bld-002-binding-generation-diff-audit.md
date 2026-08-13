# TensorRT CSharp API v4.0 绑定生成：Manifest、Generated 代码与差异审计

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：BLD-002；适用版本：4.0.0；当前状态：ready。

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

本文说明从 native manifest 生成 C# 与 C++ 绑定，并审计生成结果的方法。绑定生成是可重复构建步骤，生成文件不应手工修改后再作为稳定 ABI 依据。

## 2. Manifest 是输入合同

绑定输入位于 `native/manifests`，每个 manifest 至少描述 `module`、`versionLine` 和 `apis`。API 项需要稳定 ID、entry point、返回类型、所有权、manual override、参数名称、参数类型和方向。先检查重复 ID、重复 entry point 和缺失字段：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
```

该入口会读取全部 manifest，并验证生成文件存在、非空且可重复生成，同时导出 native methods comparison、wrapper lift candidates 和 generated API coverage 报告。

## 3. 运行生成器

需要单独观察生成器输出时，可以直接执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
```

生成器项目为 `tools/JYPPX.BindingGenerator`，通过 `--repo-root` 读取仓库根目录。典型输出包含：

```text
native/generated/bridge_api_catalog.g.h
native/generated/bridge_entrypoints.g.h
src/JYPPX.Shared/Generated/*.g.cs
src/JYPPX.TensorRtSharp/Internal/Interop/Generated/*.g.cs
src/JYPPX.CudaSharp/Internal/Interop/Generated/*.g.cs
```

不要从终端输出推断生成成功；应确认脚本退出码为 0、所有预期文件存在，并查看 `artifacts/interop-comparison` 下的报告。

## 4. 差异审计流程

建议在独立工作树或干净提交上完成审计：

```powershell
git status --short
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
git diff -- native/generated src/JYPPX.Shared/Generated src/JYPPX.TensorRtSharp/Internal/Interop/Generated src/JYPPX.CudaSharp/Internal/Interop/Generated
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
```

需要区分三类差异：

| 差异 | 处理方式 |
| --- | --- |
| 生成时间或行尾差异 | 先确认生成器是否写入时间；不能直接忽略业务差异 |
| Manifest 新增 API | 检查 entry point、版本 guard、所有权和 Native 实现 |
| Generated 代码变化 | 检查包装器签名、模块分流、错误边界和 ABI 报告 |

生成器重复运行后，所有预期文件的 SHA256 应一致。若同一输入产生不同哈希，先修复生成器的排序、时间戳或模板问题，再继续评审。

## 5. 从输入到 ABI 的复核点

Manifest 通过后，继续检查以下链路：

1. C++ 声明和导出 entry point 与 Bridge 实现一致；
2. C# P/Invoke 的 calling convention、指针、数组和 UTF-8 语义一致；
3. ownership、释放责任和 nullable 约束没有因生成而丢失；
4. TensorRT 8/10/11 或 CUDA line 的 guard 没有被错误复用；
5. `eng/Export-NativeMethodsComparison.ps1` 报告中的新增、缺失和手工 override 都有审计结论。

## 6. 生成结果与源码修改边界

生成文件应由 generator 和 manifest 产生。若必须修改生成器模板，应同时提交：

```text
manifest 输入变化
generator 或模板变化
生成文件差异
Test-BindingGeneratorOutputs.ps1 输出
Native ABI comparison 报告
托管和 Native 测试结果
```

不得只提交一个 `.g.cs` 文件来修复 ABI。`git diff --check`、编译和测试能发现部分问题，但不能替代对 entry point、参数方向和释放责任的人工审计。

## 7. 失败定位

生成器报告 manifest 缺字段时，修复输入合同；报告重复 entry point 时，先确认是不是跨模块 ABI 冲突；生成的 C# 代码编译失败时，对照对应的 Native 声明和模板，而不是手工把类型改到能编译为止。运行时 `EntryPointNotFoundException`、访问冲突或输出乱码，通常需要回到 manifest、calling convention、字符编码和库实际版本复核。

## 8. 小结

2026-08-13 在独立工作树中两次执行仓库验证入口，共读取 `214` 份 manifest、核对 `4046` 条 API 记录；两次生成结果一致，native methods comparison、wrapper lift candidates 和 generated API coverage 报告均成功导出。报告 SHA256 和源码状态记录在 `docs/articles/zh-cn/06-source-build/source-build-evidence-20260813.json`。

绑定生成的可复现性来自 manifest、generator、模板、生成文件和差异报告的完整闭环。只有重复生成哈希稳定、ABI 对比有结论、托管和 Native 测试通过，绑定变化才具备进入评审或发布候选的基础。本文晋级 `ready` 表示生成与差异审计流程已复核，不表示每个生成 entry point 都完成了真实 GPU 调用。

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
