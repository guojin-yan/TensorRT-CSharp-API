# CUDA / TensorRT / cuDNN 版本矩阵：TRT8、TRT10、TRT11 的边界与选择

## 适用读者

这篇文章适合需要部署 TensorRtSharp 4.0 的工程师、CI 维护者、Windows/Linux 运维人员，以及正在选择 CUDA/TensorRT/cuDNN 组合的团队。它不是 NVIDIA 安装手册的替代品，而是解释本项目如何看待版本矩阵、runtime package 和 proof 边界。

## 解决问题

TensorRT 的使用难点通常不在单个 API，而在版本组合：CUDA driver 支持的 runtime、TensorRT major line、cuDNN ABI、Windows/Linux RID、native assets copy 方式都会影响最终运行。TensorRtSharp 4.0 用 `runtime key` 和版本 guard 来描述这些组合，避免把某个本机可运行结果错误推广到所有平台。

## 如何阅读版本矩阵

项目中的版本证据分散在几个位置：

- `artifacts/interface-coverage/tensorrt-interface-comparison.csv`：接口层面的跨版本比较。
- `artifacts/final-release/final-release-pre-publish-audit-matrix.json`：发布前 proof 与非 proof 状态。
- `artifacts/final-release/release-evidence-bundle.json`：release evidence 聚合。
- `docs/articles/zh-cn/runtime-package-matrix-reading-guide.md`：runtime package 选择说明。
- `docs/articles/zh-cn/runtime-package-selection.md`：选择具体包时的用户说明。

阅读时不要只看“某个接口存在”。一个 runtime key 至少要同时满足：managed package 版本、native runtime package 版本、TensorRT line、CUDA runtime、cuDNN line、RID、host driver 能力和 smoke/proof 记录。

## TRT8、TRT10、TRT11 的维护策略

TRT8 更像兼容线：它适合已有部署和保守升级路径，但不能假设 TRT10/TRT11 的新 API 都可用。TRT10 是较稳定的现代主线，适合较多部署路径。TRT11 承载更新的 layer、engine inspector 和现代 TensorRT 能力，但更需要严格 version guard，尤其是 CUDA 13、driver 和 runtime package 匹配。

跨版本实现时，manifest、native source、managed route、tests 和 artifact 都要一起变化。只改 manifest 不改 native/source，不算完成；只改 native 不补 wrapper，不算用户可用；只跑 build 不跑 smoke/package-consumer proof，也不能进入 release close。

## 安装选择建议

普通用户优先选择项目 README 和 runtime package 文档推荐的稳定 runtime key；需要 TRT11/CUDA13 的用户，应先确认本机 driver 支持对应 CUDA runtime，再运行 smoke。部署团队则建议把 runtime key、driver、CUDA、TensorRT、cuDNN、GPU 名称和 OS/RID 全部记录到 proof input。

```powershell
dotnet --info
nvidia-smi
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleasePrePublishAuditMatrix.ps1
```

这些命令帮助收集环境与矩阵信息，但并不自动证明 package-consumer-runtime 这条发布证据线已经完成。

## 边界说明

版本矩阵是部署导航，不是 runtime proof。`build-only`、`dry-run`、`template`、local feed、ProjectReference、direct `.nupkg`、GUI screenshot、TensorRtExec report、YoloVision matrix、OnnxToEngine report、sidecar、readonly diagnostics 都不能替代真实 runtime proof。不同 CUDA/TensorRT 组合必须由对应 host 上的日志、hash 和 validator 证明。

## 下一步

下一步阅读 `tensorrtsharp-nuget-runtime-package-guide.md`，了解 managed package 与 native runtime package 如何分工，以及为什么 clean consumer proof 必须从 public package source 开始。
