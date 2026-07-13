# Engine Inspector：如何读取 engine 信息而不伪造 runtime proof

Engine inspector 适合在 engine build 或 load-engine readonly diagnostics 阶段查看层信息、profiling verbosity 和 inspector text。它帮助定位模型结构问题，但不创建 execution context、不绑定输入输出、不 enqueue，也不验证输出。

## 适合

- 想调试 serialized engine 层信息和 profiling verbosity 的使用者。
- 需要理解 `TensorRtEngineInspector` 与 `applications/TensorRtExec` load-engine readonly diagnostics 边界的维护者。
- 正在区分 build/read-only diagnostics 与真实 package-consumer-runtime proof 的发布负责人。

## 关键路径

- 高层入口：`src/JYPPX.TensorRtSharp/TensorRtEngineInspector.Trt11Diagnostics.cs`。
- 工具投影：`src/JYPPX.TensorRtSharp.Tools/OnnxEngineBuildDiagnostics.cs`。
- TensorRtExec 报告：`applications/TensorRtExec/Core/TensorRtExecReport.cs`。
- CLI/GUI 输出：`applications/TensorRtExec/Console/TensorRtExecCommand.cs` 和 `applications/TensorRtExec/WinForms/MainForm.cs`。

## proof 边界

Engine inspector 输出属于 readonly diagnostics。即使 `--loadEngine` 能反序列化 engine 并读取 IO tensor count、layer count、profile count、device memory、auxiliary streams 和 inspector text，也不能晋级为 runtime proof。

真实 runtime proof 至少需要：

- 外部 clean consumer 使用 public package source restore。
- 真实 TensorRT runtime smoke 运行成功。
- 非 ProjectReference、非 local feed、非 direct `.nupkg`。
- log path 与 SHA256 对齐。
- owner input validator 与 record validator 通过。

## 配图建议

- 一张流程图：load engine -> read metadata -> report，只到 readonly diagnostics。
- 在图上明确标注未发生的步骤：create execution bindings、enqueue、output validation。

## 下一步

继续扩展 inspector 的安全 readback 字段；不要把 inspector text、build-only report 或 GUI screenshot 写成 package-consumer-runtime proof。
