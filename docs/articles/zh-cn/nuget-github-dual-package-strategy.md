# NuGet 与 GitHub 双包发布策略

TensorRtSharp4.0 采用两条发布路线，目的是兼顾易用性、包大小限制和 CUDA/TensorRT/cuDNN 的再分发边界。

## 两条路线

| 路线 | 内容 | 适合用户 | 边界 |
| --- | --- | --- | --- |
| GitHub full runtime 包 | C# API、C++ bridge、CUDA/TensorRT/cuDNN 运行时依赖、runtime assets | 想开箱即用、能接受大包下载的用户 | 需要 GitHub Release asset 真实证据 |
| NuGet small bridge/core 包 | C# core API 与中间 C++ bridge 小包 | 已在机器上安装 CUDA/TensorRT/cuDNN 的用户 | 需要用户自装 NVIDIA runtime |

这两条路线可以同时存在。GitHub full runtime 包解决大依赖分发问题；NuGet 小包降低安装门槛，便于项目宣传和普通 .NET 用户引用。

## NuGet 小包使用模型

用户安装 NuGet 包后，需要在本机安装 CUDA、TensorRT、cuDNN，并通过标准路径或环境变量让 probing 找到 native runtime：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
$env:JYPPX_TENSORRT_ROOT = "C:\nvidia\TensorRT-10.x"
$env:JYPPX_CUDA_ROOT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
```

这种模式下，NuGet 包不应该声明已经包含所有 NVIDIA runtime。文档必须明确用户负责安装和版本匹配。

## GitHub Full Runtime 使用模型

GitHub release 可以承载大体积 runtime 包，适合包含：

- Windows x64 native bridge。
- CUDA runtime assets。
- TensorRT runtime assets。
- cuDNN runtime assets。
- 版本矩阵说明。
- SHA256 和 asset metadata。

但 GitHub asset 上传必须由 Owner 执行或明确授权，本项目脚本不能自动上传。生成 release bundle、package review、candidate record 都不是 GitHub 发布 proof，也不能替代真实发布 proof。只有 Owner 提供的公开渠道、下载元数据、hash、外部消费者日志和最终授权能进入发布闭环。

## 证明边界

| 材料 | 是否可作为公开发布 proof |
| --- | --- |
| local feed consumer | 否 |
| ProjectReference consumer | 否 |
| direct `.nupkg` install | 否 |
| build-only 或 dependency-probe-only | 否 |
| pre-publish smoke | 否 |
| failedBlockerCount=0 | 否 |
| 公开包 URL + SHA256 + downloaded metadata + CleanConsumer restore/build/smoke | 是，仍需 validator |
| Owner push transcript 或 GitHub-only lane reason | 是，仍需与包证据交叉校验 |

## 发布前必须回填

- 公开包 URL。
- 包 SHA256。
- 下载来源和下载元数据。
- CleanConsumer restore/build/smoke 日志与 SHA256。
- PostPublish restore/build/smoke 日志与 SHA256。
- host runtime metadata。
- package identity 和 dependency graph。
- rollback review。
- final close decision。
- final release close approval。

## 文档口径

对外文章可以说“项目提供两条包路线”，但不能说“已经公开发布完成”，除非最终 release gates 已被真实 Owner evidence 关闭。宣传文章要把安装路径讲清楚，也要把 proof 边界讲清楚。
