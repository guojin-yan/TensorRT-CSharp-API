# LoggerFinder Metadata 安全设计门

## 目标

本设计门覆盖 TensorRT 11 的 `ILoggerFinder::getInterfaceInfo`。`ILoggerFinder` 是应用侧 logger callback provider，不是可安全借用的 TensorRT-owned query object。

## 安全边界

- 只允许 copied interface metadata。
- 不暴露 finder handle 或 logger callback pointer。
- 不启用 logger callback lookup/invocation。
- finder owner lifetime 和 logger callback ownership 继续 deferred。
- 本设计门不是 runtime proof。

## 当前证据

- evaluator：`src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtLoggerFinderMetadataDesignGate.cs`
- result model：`src/JYPPX.TensorRtSharp/Callbacks/Monitoring/TensorRtLoggerFinderMetadataDesignGateResult.cs`
- 测试：`tests/JYPPX.ProjectQuality.Tests/LoggerFinderMetadataDesignGateTests.cs`
- 机器清单：`artifacts/interface-coverage/deferred-readonly-candidate-list.json`

## 下一步

下一阶段如果继续推进 logger finder，应先建模 owner lifetime、callback ownership、跨 ABI 异常映射和 no-throw callback 边界。不能因为 interface-info 是只读方法就公开裸指针。
