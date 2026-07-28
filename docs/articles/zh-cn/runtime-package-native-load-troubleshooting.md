# Runtime Package Native Load 排查指南

## 适用读者

本文面向安装 runtime package 后遇到 DLL/SO 加载失败、CUDA driver 不兼容或 TensorRT native dependency 缺失的用户。

## 解决问题

Native load 失败通常表现为 `DllNotFoundException`、CUDA error 35、找不到 TensorRT library 或 Linux linker 无法解析 SO。本文给出排查顺序，并说明 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不能替代 runtime proof。

## 背景与场景

Windows 和 Linux 的 native library resolution 规则不同。Windows 依赖 DLL 搜索路径和 RID asset，Linux 依赖 rpath、LD_LIBRARY_PATH 和系统 linker。TensorRtSharp runtime package 需要与本机 driver、CUDA、TensorRT 和 cuDNN 版本共同匹配。

## 操作路径

1. 先确认 GPU driver 支持目标 CUDA runtime，避免 driver 太旧导致 CUDA error 35。
2. 检查 NuGet runtime package 是否与 RID 和 TensorRT/CUDA 版本匹配。
3. 使用最小 native load smoke 打印 library path、load result 和 native version。
4. 若 TensorRtExec build 成功但 consumer 加载失败，优先检查 package source 和 consumer 运行目录。
5. 将失败记录为 blocker，保留 stdout/stderr、host metadata 和 dependency probe 输出。

## 代码与文件入口

- `docs/articles/zh-cn/cuda-error-35-troubleshooting.md`：CUDA driver 兼容排查。
- `docs/articles/zh-cn/runtime-package-minimal-smoke-commands.md`：最小 smoke 命令。
- `docs/articles/zh-cn/runtime-package-selection.md`：runtime package 选择。
- `docs/articles/zh-cn/runtime-package-matrix.md`：版本矩阵。
- `src/JYPPX.TensorRtSharp/Diagnostics/TensorRtEnvironmentProbe.cs`：环境探测入口。

## 图示建议

建议用决策树展示：driver -> package RID -> native library path -> TensorRT version -> smoke run。每个节点列出失败时应采集的字段。

## 边界说明

Native load troubleshooting 是诊断材料，不是 runtime proof。readonly diagnostics、TensorRtExec report、OnnxToEngine report、YoloVision matrix、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 都必须保留非 proof 标签。

## 下一步

下一轮应把常见错误码和修复建议整理成机器可读表，供 CLI/GUI report 和文档共同引用。
