# TRT11 ONNX Parser Builder-Config Candidate Review

审查日期：2026-07-18

## 结论

本批选择 `IParser::setBuilderConfig`，并同步补齐 TRT11 的 `REPORT_CAPABILITY_DLA`、`ENABLE_PLUGIN_OVERRIDE`、`ADJUST_FOR_DLA` parser flags。

该入口是 ONNX/DLA 部署关键路径。公开 API 只接收现有 `TensorRtBuilderConfig` owner，native ABI 只接收受类型和版本线验证的 opaque handle，不向调用方暴露 vendor pointer。TensorRT parser 借用 config，因此 managed parser 使用 `SafeTensorRtObjectHandleLease` 保持 config native owner，直到另一个 config 被成功接受或 parser 释放。

## 生命周期判定

- 调用前为 config SafeHandle 建立长期 lease。
- vendor 返回 `true` 后才替换旧 lease；返回 `false` 时保留旧关联。
- parser 释放时先销毁 native parser，再释放 config lease。
- parser/config 版本线不一致时 fail closed。
- TRT8/TRT10 明确 `NotSupported`，不让 TRT11 ABI 泄漏到旧版本线。
- C++ exception 和 Windows SEH 均在 native 边界内转换为 bridge status。

## 未选择候选

| 候选 | 判定 | 原因 |
| --- | --- | --- |
| `IDimensionExpr` / `IExprBuilder` | 继续 deferred | 对象依赖 shape callback/build callback 生命周期，尚无稳定 owner。 |
| `IAlgorithm*` | 继续 deferred | algorithm selector callback owner 尚未闭环。 |
| calibrator / output allocator | 继续 deferred | 需要 callback trampoline、vtable 和跨线程释放协议。 |
| `Global::initLibNvInferPlugins` | 继续 deferred | 进程级 plugin register 副作用，并引入额外 vendor library 依赖。 |
| `Global::setInternalLibraryPath` | 继续 deferred | 进程全局 mutation，隔离和恢复语义不足。 |
| plugin create/register/deregister/load | 继续 deferred | plugin object 和 library 生命周期不明确。 |
| allocator/resource acquire/release | 继续 deferred | 资源 ownership 和异步使用窗口未完成设计。 |

## 验收信号

- manifest、header、guarded native source、generated P/Invoke 和高层 wrapper 全部存在。
- coverage 行必须为 `implemented-with-deferred-history`，旧 deferred manifest 保留。
- OnnxToEngine smoke 与 trtexec-like build service 使用真实 attachment；DLA 路径在 TRT11 设置 config 后启用 capability/adjust flags。
- package consumer 从本地包编译该公开面，不依赖 ProjectReference。
