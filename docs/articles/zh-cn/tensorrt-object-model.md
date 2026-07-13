# TensorRT Builder Runtime Engine Object Model

TensorRtSharp4.0 的 TensorRT 封装按对象生命周期组织，而不是把所有 C ABI 入口直接暴露给应用层。核心路径是 builder 创建 network，config 控制构建，runtime 反序列化 engine，execution context 执行推理。

## 常见对象

- Logger：接收 TensorRT 日志，托管侧必须避免跨 ABI 抛异常。
- Builder：创建 network、builder config 和 optimization profile。
- Network：描述 tensor、layer、shape 和输出标记。
- BuilderConfig：设置 workspace、profiling、flag、profile 和 tactic 策略。
- HostMemory：承载 serialized engine bytes。
- Runtime：从 serialized engine 创建 engine。
- Engine：保存可执行网络结构和 tensor 元数据。
- ExecutionContext：绑定输入输出地址，设置 shape，执行 enqueue。

## 推荐路径

最小端到端路径通常是：

1. 创建 logger。
2. 创建 builder/network/config。
3. 添加输入、layer、输出。
4. 如果是 dynamic shape，添加 optimization profile。
5. 构建 serialized network。
6. 创建 runtime 并反序列化 engine。
7. 创建 execution context。
8. 绑定 tensor 地址并 enqueue。

`smoke/TensorRtSmokeRunner`、`smoke/NetworkBuilderSmokeRunner` 和 `samples/InferenceBindings` 是优先参考对象。

## 边界

对象模型文章不能替代接口完成度证明。manifest/source 匹配说明扫描接口已覆盖，但真实可用性仍取决于 native 实现、C# wrapper、smoke 和 package consumer evidence。
