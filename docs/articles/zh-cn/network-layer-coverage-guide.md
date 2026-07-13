# Network Layer Coverage Guide

Network layer coverage 用来解释 TensorRtSharp4.0 当前哪些 network builder 路径有 smoke 或文档证据。它与 manifest/source 匹配不同，重点看 wrapper、native 实现和 runner 输出。

## 覆盖对象

常见 layer 包括：

- convolution、pooling、activation、elementwise。
- resize、slice、shuffle、concat。
- topk、reduce、matrix multiply。
- quantize、dequantize、scale。
- TRT11 专属 modern layers。

## 评估维度

每个 layer 建议记录：

- API 是否有非 deferred native 实现。
- C# wrapper 是否隐藏 borrowed pointer。
- 是否有 version guard。
- 是否有 network build smoke。
- 是否有 engine build 或 runtime enqueue 证据。

## 推荐输出

后续可以生成 `artifacts/network-layer-coverage/network-layer-coverage.json`，字段包括 layer name、TensorRT line、wrapper status、smoke runner、evidence file 和 remaining boundary。

## 边界

layer coverage 不应被写成模型精度证明。它只证明对应 network 构建路径或 smoke 路径可达。

## 第二批正文门禁

### 适用读者

本文适合需要判断 TensorRtSharp4.0 network builder API 覆盖度的维护者，也适合准备撰写“接口完成度”和“真实可用度”文章的发布负责人。

### 解决问题

manifest/source 匹配只能说明入口存在，不能说明用户能安全调用。network layer coverage 要解决的是：某个 layer 是否有非 deferred native 实现，C# wrapper 是否隐藏 borrowed pointer，TRT8/TRT10/TRT11 是否有一致 guard，是否有 build 或 enqueue 级别证据。

### 核心思路

核心思路是把每个 layer 分成 `manifest`、`native`、`managed wrapper`、`smoke`、`runtime proof` 五层。build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 可以作为线索，但不能单独证明 runtime proof。

### 操作路径

从 interface coverage matrix 筛选 network/layer API，对照 native manifest、common/v8/v10/v11 source 和 C# wrapper，标记 wrapper 是否暴露裸 pointer，关联 smoke runner、sample 或 article evidence，并输出 layer coverage JSON/Markdown。

### 边界说明

layer coverage 不应被写成模型精度证明，也不等于 public package proof、post-publish proof 或 release close approval。只有真实 engine enqueue/readback 或模型 proof 通过后，才能把某个 layer 场景标记为 runtime proof。

### 下一步

下一步建议生成 `artifacts/network-layer-coverage/network-layer-coverage.json`，并新增测试校验高频 layer 的 wrapper 状态、version guard 和 evidence link。
