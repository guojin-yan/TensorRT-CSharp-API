# Refit Weights Guide

Refit weights 适合在 engine 结构固定时更新部分权重。它不等同于重新构建 network，也不适合改变 layer 拓扑或 tensor shape。

## 适用场景

- 同一模型结构下替换卷积或全连接层权重。
- A/B 测试小范围权重更新。
- 在不重新 parser/build 的情况下验证 refitter 路径。

## 推荐证据

优先参考：

- `smoke/RefitWeightsSmokeRunner`
- refitter error recorder snapshot
- serialized engine round-trip 输出

完整记录应包含 engine 来源、可 refit layer 名称、weight role、更新前后校验和、refit 返回状态。

## 常见限制

- 不能改变 network 拓扑。
- 不能改变 tensor rank、dtype 或 shape 语义。
- 需要 TensorRT 对目标 engine 和权重 role 支持 refit。
- 如果 refit 失败，应复制 error recorder snapshot，不要只暴露 TensorRT 内部指针。

## 边界

Refit 文档只说明权重更新路径，不表示所有 layer 的 runtime accuracy 都已验证。每个模型仍需要自己的 package consumer 或样例证据。
