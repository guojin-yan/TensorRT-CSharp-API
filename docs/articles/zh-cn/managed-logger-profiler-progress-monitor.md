# Managed Logger Profiler Progress Monitor

Managed logger、profiler 和 progress monitor 是托管回调体验的入口，但它们必须遵守 no-throw、owner 生命周期和 callback proof 边界。

## Logger

Logger 用于接收 TensorRT 日志。托管侧实现必须捕获异常并转换为安全状态，不能让异常跨 ABI 传播。

## Profiler

Profiler 用于接收 layer profile 信息。公共 API 应提供复制后的名称、时间和 layer 记录，不暴露 TensorRT 内部字符串指针。

## Progress Monitor

Progress monitor 用于构建或长任务进度。它需要明确 owner attach/detach 顺序，并在释放前断开 native 回调入口。

## 证据要求

每条 managed callback 路径都应记录：

- owner 是否稳定。
- attach/detach 是否成对。
- callback 是否 no-throw。
- 异常是否被捕获并映射。
- 是否有真实 runtime invocation。

## 边界

Managed callback scaffold 或 design gate 不能写成真实 callback runtime proof。真实 proof 必须来自 runtime consumer，并看到 `InvocationCount>0`。
