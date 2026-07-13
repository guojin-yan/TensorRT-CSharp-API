# Runtime Package Windows/Linux 安装 FAQ

## 适用读者

本文面向准备安装 TensorRtSharp runtime packages 的 Windows 与 Linux 用户，尤其是需要区分 CUDA、TensorRT、cuDNN、RID、native DLL/SO 和 NuGet 包关系的开发者。

## 解决问题

Runtime package 安装失败通常不是 C# API 调用问题，而是本机 CUDA driver、runtime package、TensorRT native library 和 RID 不匹配。本文以 FAQ 形式说明常见问题，并强调 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不能替代 runtime proof。

## 背景与场景

项目支持多个 CUDA/TensorRT 组合，Windows 依赖 DLL 搜索路径，Linux 依赖 SO 搜索路径与运行时 linker。用户需要先选对 runtime package，再验证 native library 是否能被实际 consumer 加载。

## 操作路径

1. 确认 GPU driver 支持目标 CUDA runtime，避免 CUDA error 35 等驱动不兼容问题。
2. 按项目 runtime matrix 选择 Windows 或 Linux RID 对应 package。
3. 在 clean consumer 中安装主包和 runtime package，不使用 ProjectReference 或 local feed 作为最终 proof。
4. 运行最小 native load smoke，确认 TensorRT/CUDA/cuDNN library 可以被解析。
5. 再运行 engine deserialize 或样例推理，保存 stdout、stderr、package version、hash 和 host metadata。

## 代码与文件入口

- `pack/runtime`：runtime package 定义与本地覆盖文件。
- `docs/articles/zh-cn/runtime-package-matrix.md`：runtime package 矩阵。
- `docs/articles/zh-cn/runtime-package-selection.md`：选择指南。
- `docs/articles/zh-cn/runtime-package-installation-deep-dive.md`：安装深潜。
- `docs/articles/zh-cn/cuda-error-35-troubleshooting.md`：驱动兼容排查。

## 图示建议

建议配一张双列图：Windows 展示 DLL search path，Linux 展示 SO search path。中间用 runtime package ID 连接 CUDA/TensorRT/cuDNN 版本。

## 边界说明

安装 FAQ 是采用材料，不是 proof。direct `.nupkg`、ProjectReference、local feed、template、dry-run、build-only、TensorRtExec report、OnnxToEngine report、YoloVision matrix 和 readonly diagnostics 都不能替代真实 clean consumer runtime proof。

## 下一步

下一轮应补充 Windows 与 Linux 的最小 smoke 命令模板，并把常见错误映射到修复建议和 evidence 字段。
