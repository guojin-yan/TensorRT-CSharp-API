# 安装布局说明

## 推荐的本地 Windows 布局

CUDA 根目录：

- `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA`

本仓库已验证过的 CUDA 目录示例：

- `v11.6`
- `v11.8`
- `v12.1`
- `v12.9`
- `v13.2`

TensorRT / cuDNN 本地依赖：

- 由用户安装在仓库外，并分别通过 `JYPPX_TENSORRT_ROOT`、`JYPPX_CUDNN_ROOT` 选择

不得把 CUDA、cuDNN、TensorRT 或 NVRTC 复制到仓库目录，也不得放入发布包。

本仓库已验证过的重点组合：

- `TensorRT-8.6.1.6-cuda 11.8` + CUDA `11.8` + cuDNN `8.9`
- `TensorRT-8.6.1.6-cuda 12.0/12.1` + CUDA `12.1` + cuDNN `8.9`
- `TensorRT-10.11.0.33-cuda 11.8` + CUDA `11.8` + cuDNN `8.9`
- `TensorRT-10.11.0.33-cuda 12.9` + CUDA `12.9` + cuDNN `9.22`
- `TensorRT-11.0.0.114-cuda 12.9` + CUDA `12.9` + cuDNN `9.22`
- `TensorRT-11.0.0.114-cuda 13.2` + CUDA `13.2` + cuDNN `9.22`

## 版本配对规则

构建和 runtime 包必须显式选择 TensorRT / CUDA / cuDNN 组合，不能只依赖自动探测。当前 CUDA `12.9` 已经安装，目标为 `cuda12.9` 的组合必须使用 CUDA `12.9`，之前临时使用 CUDA `12.3` 的 fallback 已废弃，不能再作为本地验证依据。

`TensorRT 11 + CUDA 13.2` 当前可完成原生编译，但运行 smoke 仍受本机驱动 CUDA 能力限制，暂不标记为完整本地验证。

## 本机专属路径

本机专属路径应写入：

- `pack/runtime/runtime-packages.local.json`

该文件不纳入 Git。建议从 `pack/runtime/runtime-packages.local.example.json` 复制后修改。公开的 `runtime-packages.manifest.json` 不应包含机器专属盘符路径。

## 已验证结论

已完成 Windows 本地真实链路和包验证的重点组合：

- `trt8.6-cuda11.8-cudnn8.9`
- `trt8.6-cuda12.1-cudnn8.9`
- `trt10.11-cuda11.8-cudnn8.9`
- `trt10.11-cuda12.9-cudnn9.22`
- `trt11.0-cuda12.9-cudnn9.22`

Linux 结构继续保留，但当前仍以 Windows API 完整化为优先。Linux 真机打包和验证等待后续 GitHub/self-hosted runner 环境。

