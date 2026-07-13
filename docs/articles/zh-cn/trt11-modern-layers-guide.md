# TensorRT 11 Modern Layers Guide

TensorRT 11 引入或强化了一批 modern layer 和 metadata 能力。TensorRtSharp4.0 对 TRT11 专属能力必须保留 version guard，不能让 TRT8 或 TRT10 路径误调用。

## 推荐阅读对象

- `smoke/NetworkTrt11ModernLayersSmokeRunner`
- TRT11 manifest
- native `v11` source
- managed wrapper 中的 version guard

## 实现原则

- TRT11 专属 API 只在 TRT11 wrapper 或 guard 后暴露。
- 公共 API 文档需要说明最低 TensorRT line。
- smoke 输出应包含 TensorRT line 和 skipped/blocked 原因。
- 新增 layer wrapper 时同步更新 manifest、native source、interop、C# wrapper 和质量测试。

## 证据要求

每个 modern layer 至少记录：

- layer 创建是否成功。
- 输入输出 tensor dtype 和 shape。
- TRT line。
- unsupported line 的 guard 行为。
- 是否执行到 engine build 或只覆盖 network build。

## 边界

TRT11 文章不能把版本专属能力写成所有 TensorRT 版本通用能力。跨版本一致封装的目标是行为清晰，不是抹平 ABI 差异。
