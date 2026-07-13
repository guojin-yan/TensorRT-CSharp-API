# 下一批只读安全门汇总

## 本批范围

本批把 5 组高风险但高用户价值的只读/metadata 候选纳入机器清单：

- Stream IO interface info。
- INT8 calibrator interface info。
- LoggerFinder metadata。
- DebugListener interface info 与 borrowed debug tensor safety。
- Allocator interface info。

## 共同原则

- 只允许 copied metadata、presence diagnostics 或 owner-scoped design gate。
- 不暴露 `IntPtr` / `nint` / borrowed native handle。
- 不启用 callback trampoline。
- 不删除 deferred history。
- 不把 readonly diagnostics 或 design gate 晋级为 runtime proof。

## 真实完成条件

这些候选后续要晋级为真实 API，必须逐项补齐 owner lifetime、callback lifetime、跨 ABI no-throw 边界、native/source/managed wrapper、smoke 和 package-consumer-runtime proof。
