# TensorRT CSharp API v4.0 CMake Preset：Native Bridge 构建与调试

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：BLD-003；适用版本：4.0.0；当前状态：ready。

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

本文说明使用仓库 `CMakePresets.json` 构建和调试 Native Bridge 的方法。Preset 负责固定平台、TensorRT line、CUDA line、cuDNN major、编译器和输出目录；不要把一次手工 CMake 命令当成矩阵构建证明。

## 2. 工具链与依赖

当前 CMake 最低版本为 3.27。Windows 使用 Visual Studio 17 2022，Linux 使用 Ninja。TensorRT、CUDA、cuDNN 和编译器版本必须与所选 preset 及运行时清单一致：

```powershell
cmake --version
dotnet --info
cmake --list-presets
```

根目录 `CMakeLists.txt` 支持 `JYPPX_TENSORRT_ROOT`、`JYPPX_CUDA_ROOT`、`JYPPX_CUDNN_ROOT` 和对应环境变量。显式传根目录比依赖机器默认搜索路径更容易复现。

## 3. 选择和配置 Preset

开发时可以先用不启用厂商绑定的 preset 检查配置器和编译器入口：

```powershell
cmake --preset win-x64-dev
cmake --build --preset win-x64-dev-debug --parallel
```

这一 preset 不是 TensorRT lane 的替代品。本次源码状态下 `win-x64-dev` configure 成功，但 build 在关闭厂商绑定的 v8 兼容实现处失败，因此不能把它列为本轮通过项；问题应按生成 guard 或 no-binding 实现单独修复。

目标 TensorRT/CUDA lane 则使用清单中存在的 preset，例如：

```powershell
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

若本机安装的是 TensorRT 10 与 CUDA 12，可选择对应 lane，并通过环境变量或 `-D` 参数传入 SDK 根目录：

```powershell
cmake --preset win-x64-trt10-cuda12-release `
  -DJYPPX_TENSORRT_ROOT="$env:JYPPX_TENSORRT_ROOT" `
  -DJYPPX_CUDA_ROOT="$env:JYPPX_CUDA_ROOT" `
  -DJYPPX_CUDNN_ROOT="$env:JYPPX_CUDNN_ROOT"
cmake --build --preset win-x64-trt10-cuda12-release --parallel
```

Linux 使用同名 Linux preset：

```bash
cmake --preset linux-x64-trt11-cuda13-release
cmake --build --preset linux-x64-trt11-cuda13-release --parallel
```

配置输出位于 `build-out/<preset>`。不要在同一个 binary directory 中切换 TensorRT 根目录或 CUDA line；更换 SDK 后删除对应 preset 的构建目录并重新 configure。

## 4. 查看 CMake 发现结果

配置日志应明确记录：

```text
JYPPX_TARGET_ARCH
JYPPX_TENSORRT_LINE
JYPPX_CUDA_LINE
JYPPX_CUDA_VERSION
JYPPX_CUDNN_MAJOR
TensorRT_ROOT
CUDAToolkit_ROOT
JYPPX_CUDNN_ROOT
```

CMake 找不到 TensorRT 或 CUDA 时，可能仍能配置一个关闭厂商绑定的 Bridge。这个结果只能说明编译器和基础目标可用，不能宣称 TensorRT API 已启用。构建日志中应区分 `JYPPX_ENABLE_TENSORRT_BINDINGS`、`JYPPX_ENABLE_CUDA_BINDINGS` 的实际值。

## 5. Native 调试工具

Windows 可用 `dumpbin` 检查导出和依赖，Linux 可用 `ldd`、`readelf` 和 `nm`：

```powershell
dumpbin /DEPENDENTS .\build-out\win-x64-trt11-cuda13-release\<configuration>\jyppxtrtbridge.dll
dumpbin /EXPORTS .\build-out\win-x64-trt11-cuda13-release\<configuration>\jyppxtrtbridge.dll
```

```bash
ldd ./build-out/linux-x64-trt11-cuda13-release/libjyppxtrtbridge.so
readelf -d ./build-out/linux-x64-trt11-cuda13-release/libjyppxtrtbridge.so
nm -D ./build-out/linux-x64-trt11-cuda13-release/libjyppxtrtbridge.so | head
```

输出应保存目标架构、库版本、构建提交和实际文件 SHA256。`dumpbin` 或 `ldd` 无缺失只证明 loader 依赖关系，不证明每个 entry point 的参数语义正确。

## 6. ABI 与运行时定位

出现 `DllNotFoundException`、`EntryPointNotFoundException`、访问冲突或 TensorRT 初始化失败时，按以下顺序定位：

1. 确认进程架构与 Bridge 架构一致；
2. 确认加载的是本次构建目录中的 Bridge，而不是 PATH 或旧包中的文件；
3. 对比 `dumpbin`/`nm` 导出与生成的 entry point；
4. 检查 TensorRT、CUDA、cuDNN 的实际库文件和版本；
5. 清理 preset binary directory 后重新配置，排除 CMake cache 污染。

不要通过复制另一台机器的 DLL 或修改名称掩盖 ABI 不匹配。厂商运行库不属于项目 Bridge 的打包内容，必须由目标环境按清单提供。

## 7. 构建证据模板

```text
CMake version and generator: recorded
Preset: recorded
Compiler and SDK: recorded
TensorRT/CUDA/cuDNN: recorded
Configure exit code: 0
Build exit code: 0
Bridge file and SHA256: recorded
dumpbin or ldd output: recorded
Managed consumer result: recorded separately
GPU inference result: recorded separately
```

Native 构建可以作为源码编译证据，但不能自动升级为 NuGet 发布证明或目标 GPU 推理证明。

## 8. 小结

2026-08-13 的 Windows 实测使用 `win-x64-trt10-cuda12-release`，CMake 发现 TensorRT `10.11`、CUDA Toolkit `12.9.41`，并启用 TensorRT、CUDA、ONNX parser 与 ONNX config。Release 构建成功，`jyppxtrtbridge.dll` 为 `1197056` 字节，SHA256 为 `2e417858af6b3be929f2564da5a1f7f19e25512cd7e3fe1a75701a3d8800d4ca`。这证明该 Windows lane 的 Native Bridge 可构建，不证明其它 TensorRT/CUDA lane、Linux 构建、GPU 推理或公开包消费。

Preset、独立 binary directory、依赖发现日志和 ABI 工具输出共同构成可复现的 Native 构建记录。调试时优先修复根目录、缓存和实际加载路径，保持源码、生成绑定、Bridge 和消费者的版本一致。完整正负结果见 `docs/articles/zh-cn/06-source-build/source-build-evidence-20260813.json`。

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
