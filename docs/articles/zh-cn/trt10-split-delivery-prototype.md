# TRT10 Split Delivery 原型说明

## 目标

TensorRT 10 Windows runtime 包体积较大，不适合默认作为公开 NuGet.org 大包发布。项目保留 split-delivery 原型，用于探索把核心部署资产和构建 / 插件 / 解析资产拆分交付。

## 当前原型结构

- manifest：`pack/runtime-split/split-runtime-packages.manifest.json`
- package project：`pack/runtime-split/<split-key>/<packageId>.csproj`
- 校验脚本：`eng/Validate-SplitDeliveryPrototype.ps1`
- 资产收集脚本：`eng/Collect-SplitRuntimeAssets.ps1`
- 当前发布链已并入 `.github/workflows/runtime-windows.yml` 与 `.github/workflows/release-bundle.yml`

## 包命名规则

当前 split 原型使用包含 TensorRT / CUDA / cuDNN 主次版本的 package id：

- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda11.8.cudnn8.9.Extensions`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Core`
- `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Extensions`

## Core 包职责

`Core` 原型包用于轻量部署路径，当前包含：

- `jyppxtrtbridge.dll`
- CUDA runtime，例如 `cudart64_110.dll` 或 `cudart64_12.dll`
- `nvinfer_10.dll`
- `nvinfer_dispatch_10.dll`
- `nvinfer_lean_10.dll`

## Extensions 包职责

`Extensions` 原型包用于模型构建、插件和解析路径，当前包含：

- `nvinfer_builder_resource_10.dll`
- `nvinfer_plugin_10.dll`
- `nvinfer_vc_plugin_10.dll`
- `nvonnxparser_10.dll`

## 当前验证状态

- `win-x64-trt10.11-cuda11.8-cudnn8.9-core` 可以从完整 runtime 资产中收集 split 资产。
- `win-x64-trt10.11-cuda11.8-cudnn8.9-core` 可以本地打出原型 nupkg。
- `Extensions` 包保持设计原型状态，避免当前阶段复制更大的 builder/plugin/parser 资产。

## 发布边界

这些 split 包目前仍是 `design-only`：

- 不允许直接公开发布。
- 不替代当前已验证的完整 runtime 包。
- 必须完成 split package consumer validation 后，才能作为 release candidate 讨论。
- NVIDIA TensorRT / CUDA / cuDNN 再分发许可复核仍是公开发布前阻塞项。
