# TensorRT CSharp API v4.0 容器部署：NVIDIA Container Toolkit、运行库与 Smoke

<!-- public-article-layout:start -->
<style>
.content article { min-width: 0; overflow-wrap: anywhere; }
.content article a, .content article :not(pre) > code { overflow-wrap: anywhere; word-break: break-word; }
.content article pre:not(.mermaid), .content article table { display: block; max-width: 100%; overflow-x: auto; }
.content pre.mermaid { max-width: 640px; margin: 16px auto; overflow-x: auto; }
.content pre.mermaid svg { display: block; width: 100%; max-width: 640px; height: auto; }
</style>
<!-- public-article-layout:end -->

> 文章编号：INS-003；适用版本：4.0.0；当前状态：ready。

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

本文给出容器中部署 TensorRT CSharp API v4.0 的验证路径。容器解决文件系统和工具链隔离，但不会自动提供 NVIDIA 驱动、TensorRT、CUDA、cuDNN 或项目 Bridge；镜像构建成功也不等于 GPU 推理成功。

## 2. 宿主机与容器前提

宿主机需要兼容的 NVIDIA 驱动、Docker 或兼容容器运行时，以及 NVIDIA Container Toolkit。安装规则以官方文档为准：

```text
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

先在宿主机确认 Docker 和 GPU 工具：

```bash
nvidia-smi
docker info
docker version
```

再使用与项目 CUDA 线匹配的基础镜像进行设备可见性测试。镜像标签应由当前 NVIDIA CUDA 镜像清单确认，不要把示例标签当作长期固定版本：

```bash
docker run --rm --gpus all <cuda-base-image> nvidia-smi
```

这个命令只证明容器能看到 GPU 和驱动接口，不能证明 TensorRT 或 .NET 项目可运行。

## 3. 镜像层次与依赖边界

建议把镜像拆成三层：

1. 基础 OS、.NET SDK/runtime 和诊断工具；
2. 与运行时清单匹配的 CUDA、cuDNN、TensorRT 用户态库；
3. `JYPPX.TensorRT.CSharp.API`、Bridge 包、应用和模型。

项目的 Runtime Bridge 包只包含项目自有的 `jyppxtrtbridge.dll` 或 `libjyppxtrtbridge.so`，不包含 NVIDIA 厂商库。包和运行库组合以清单为准：

```text
https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json
https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0/pack/runtime-split
```

容器内应使用 Linux RID 的包，不要从宿主机复制 Windows DLL。若使用宿主机挂载的模型或缓存，记录挂载模式、只读属性和容器用户。

## 4. 构建和启动最小消费者

在仓库根目录构建镜像时，先保留可复现的 SDK、包源和提交信息：

```dockerfile
FROM <dotnet-sdk-image> AS build
WORKDIR /src
COPY . .
RUN dotnet restore ./TensorRtSharp4.0.sln
RUN dotnet build ./TensorRtSharp4.0.sln -c Release --no-restore

FROM <dotnet-runtime-image>
WORKDIR /app
COPY --from=build /src/<consumer-output>/ ./
ENTRYPOINT ["dotnet", "<consumer>.dll"]
```

镜像中至少执行以下静态和加载检查：

```bash
dotnet --info
dotnet --list-runtimes
nvidia-smi
ldconfig -p | grep -E 'nvinfer|nvonnxparser|cudart|cudnn'
ldd ./runtimes/linux-x64/native/libjyppxtrtbridge.so
```

启动时显式传递 GPU：

```bash
docker run --rm --gpus all \
  --read-only \
  --tmpfs /tmp \
  <image>:<tag> \
  --help
```

`--help` 只验证进程入口。真实 smoke 必须使用匹配的 Engine、输入 Shape、CUDA stream 和输出校验，并把容器日志导出到证据目录。

## 5. 容器内运行库验证

容器内的动态链接器路径与宿主机不同。将 `LD_LIBRARY_PATH`、`ldconfig` 配置、库文件 SHA256 和镜像 digest 一并记录：

```bash
echo "$LD_LIBRARY_PATH"
find /usr/local /opt -type f \( -name 'libnvinfer*.so*' -o -name 'libcudart*.so*' -o -name 'libcudnn*.so*' \) 2>/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

如果 Bridge 能加载但 TensorRT Runtime 初始化失败，优先检查容器内的厂商库版本、架构和 loader 路径。`nvidia-smi` 成功时仍可能缺少 `libnvinfer` 或 parser/plugin 库。

## 6. 发布与安全边界

镜像 tag 不足以作为发布证据，应保存 immutable digest、Dockerfile、基础镜像版本、Git 提交、包锁定版本、运行库清单、GPU 型号和完整日志。不要把 NVIDIA 驱动设备节点、宿主机 `/usr` 或整个工作区以读写方式挂入生产容器。

Pull Request 的容器 smoke 还要考虑不可信代码：GPU runner 应隔离、限制凭据和缓存，发布凭据只进入受保护环境。容器通过不代表宿主机或其它 GPU 型号都通过。

## 7. 结果分级

| 结果 | 可以宣称 | 不能宣称 |
| --- | --- | --- |
| 镜像 build 通过 | Dockerfile 和托管层可构建 | GPU 可用 |
| `--gpus all nvidia-smi` 通过 | 容器设备可见 | TensorRT 可加载 |
| `ldd` 无缺失 | loader 依赖可解析 | Engine 推理正确 |
| Runtime/Engine smoke 通过 | 该镜像、GPU 和运行库组合可运行 | 所有宿主机都兼容 |
| digest、日志、哈希齐全 | 可复核该次执行 | 已完成公开发布授权 |

## 8. 小结

2026-08-14 在 Docker Desktop `4.60.1`、Docker client/server `29.2.0` 上，以固定 digest `sha256:c3108f6ea3d012d79d376293ef9e16879a5a98f661153d148fb516630ca0bc69` 运行 NVIDIA TensorRT 官方镜像 `nvcr.io/nvidia/tensorrt:25.06-py3`。容器为 Ubuntu 24.04.2 x64，使用 TensorRT `10.11.0.33`、CUDA `12.9`、cuDNN `9.22.0.52-1`、.NET SDK `8.0.424`，可见 NVIDIA GeForce RTX 3060 Laptop GPU 和驱动 `576.02`。

本次不再停留于 GPU 探针：官方 `trtexec` MNIST smoke 返回 `PASSED`，吞吐量为 `4219.11 qps`，GPU compute mean 为 `0.0810061 ms`；仓库 Linux Bridge 构建及 `ldd` 检查通过，managed/Bridge 本地包的外部两 PackageReference 消费者完成 Engine 序列化、反序列化、enqueue、stream 同步与 identity 输出比对。因此该固定镜像、运行库、包和 GPU 组合的容器部署主路径达到 `ready`。

结论仍受明确边界约束：本次不是裸机 Linux、WSL Ubuntu、GPU CI runner、公网 NuGet 或 post-publish proof。回调增强版消费者在 `populate_callback_state_snapshot` 内触发 `SIGSEGV`（退出码 `139`），没有形成 callback-state snapshot 或 DebugListener runtime proof；主路径成功不能替代这项失败。机器可读证据与原始日志哈希见 `docs/articles/zh-cn/05-installation/installation-runtime-evidence-20260814.json`。

容器部署的最小闭环是宿主机驱动、Container Toolkit、`--gpus all`、Linux 用户态 NVIDIA 运行库、项目 Bridge 和真实输出校验。每一层都要保存自己的证据，不能用镜像构建结果或 `nvidia-smi` 单项结果替代 TensorRT 推理证明。

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
