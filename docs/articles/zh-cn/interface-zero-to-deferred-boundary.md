# 从接口清零到 deferred 边界提升

TensorRtSharp4.0 已经完成本地扫描 TensorRT/CUDA 头文件范围内的 manifest/source 匹配清零，但这不是“所有 API 都已经可安全使用”。接口清零解决的是“官方接口有没有被识别、登记、纳入追踪”；deferred 边界提升解决的是“这个接口是否已经有真实 native 实现、高层 C# wrapper、生命周期规则和 smoke/package-consumer 证据”。

这两个阶段必须分开理解。前者让项目不再遗漏官方接口，后者才决定普通 C# 用户能否稳定调用。

## 什么是接口清零

接口清零来自 `eng/Export-InterfaceCoverageMatrix.ps1` 生成的覆盖矩阵。它会扫描本地 TensorRT/CUDA 头文件，并和仓库中的 manifest、native source 记录进行匹配。

当前覆盖报告位于：

- `artifacts/interface-coverage/interface-coverage-summary.md`
- `artifacts/interface-coverage/tensorrt-interface-coverage.csv`
- `artifacts/interface-coverage/cuda-runtime-interface-coverage.csv`
- `artifacts/interface-coverage/tensorrt-interface-comparison.csv`
- `artifacts/interface-coverage/cuda-runtime-interface-comparison.csv`

截至当前复审，TensorRT 8/10/11 以及 CUDA 11.6 到 13.2 的扫描接口都已经能匹配到 manifest/source 记录。换句话说，本地扫描到的接口没有再停留在“仓库完全不知道它存在”的状态。

## 什么是 deferred

`deferred` 表示接口已经被识别和登记，但当前仍保留在安全边界上。常见原因包括：

- C++ ABI 或版本差异尚未稳定封装。
- 返回 borrowed pointer，生命周期不适合直接暴露给 C#。
- 需要 callback trampoline、托管对象 pinning、native owner ledger 或 no-throw vtable。
- 需要 caller buffer、count/copy、snapshot 等安全数据复制模式。
- 需要真实 smoke 或 package consumer 证据证明 runtime 路径可执行。

deferred row 不能删除来制造完成度。正确做法是把一小批安全 API 从 no-arg deferred stub 提升为真实参数、返回结构、native 实现、C# interop 和高层 wrapper，然后用 smoke 或质量测试验证。

## 为什么 100% manifest/source 匹配不是 100% 可用

manifest/source 匹配说明 API 已进入项目账本，但不自动说明：

- native 实现已经不是 deferred stub。
- 托管层有面向用户的 public wrapper。
- 字符串、数组、metadata 不会返回悬空指针。
- ABI 失败路径不会跨边界抛异常。
- TRT8/TRT10/TRT11 version guard 行为一致。
- NuGet 消费端离开源码目录后仍能 restore/build/native-copy/smoke。
- callback 真的由 TensorRT runtime 触发。

因此项目现在的真实主线是 deferred 边界提升，而不是继续追逐 missing rows。

## 当前重点边界

当前规模较大、风险较高的 deferred 组包括：

| 领域 | 当前判断 |
| --- | --- |
| TensorRT plugins | plugin inventory 只读 API 已开始提升；创建、注册、资源 acquire/release、V2/V3 callback 仍需独立安全桥接。 |
| TensorRT callback | logger/profiler/progress monitor 有安全控制和诊断； allocator、output allocator、debug listener 的真实 callback runtime proof 仍未完成。 |
| TensorRT allocator | `IGpuAllocator::*`、`IGpuAsyncAllocator::*` 仍必须保留 deferred，等待 native owner ledger 和真实 runtime proof。 |
| TensorRT output/debug | `IOutputAllocator::notifyShape/reallocateOutput`、`IDebugListener::processDebugTensor` 仍必须保留 deferred。 |
| CUDA graph 高级 API | 常用路径已有 wrapper，但大量 node parameter、user object、memory node 仍需要对象模型。 |
| CUDA 13 新增 runtime API | library、kernel attribute、execution context、device resource 等接口需要逐批提升。 |

## 正确提升一个 deferred 批次

推荐每批只选择 5 到 15 个安全只读 API，按下面顺序推进：

1. 用 `rg` 和 coverage CSV 找出候选 deferred rows。
2. 检查对应 manifest、native source、generated interop、现有 wrapper。
3. 确认 ABI 输入输出不会暴露 borrowed pointer。
4. 使用 caller buffer、count/copy 或 snapshot 模式返回字符串和数组。
5. 更新 native 实现，保持 no-throw C ABI。
6. 重新运行 `eng/Generate-Bindings.ps1` 和 binding generator 输出测试。
7. 更新 C# interop、高层 wrapper 和 XML 注释。
8. 增加 smoke 或 project quality 测试。
9. 重新导出 interface coverage matrix。
10. 在 plan/diary 中记录完成项、验证结果和仍保留的 deferred rows。

## 证据分级

下面这些证据不能混为一谈：

| 证据 | 可以证明 | 不能证明 |
| --- | --- | --- |
| manifest/source matched | 接口已进入追踪账本 | API 可安全 public 调用 |
| non-deferred native export | native C ABI 有真实实现 | 高层 C# wrapper 易用且完整 |
| wrapper surface compiled | 托管调用面能编译 | runtime 行为一定正确 |
| package restore/build/native-copy | NuGet 包布局可消费 | TensorRT runtime callback 已触发 |
| `SmokeResult=passed` | 对应 smoke 路径通过 | callback proof 自动完成 |
| `real-callback-runtime` markers 齐全 | callback runtime proof 可晋级 | 其它 deferred 组自动完成 |

当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` 的 package readiness 为 `Overall=ready`，readiness blockers 为 `0`，但 full package consumer runtime smoke 在本机被 CUDA driver/runtime compatibility 阻塞为 `blocked-by-cuda-driver`。真实 callback runtime proof 仍为 `false`。

## 下一步阅读

- [Package Readiness 当前状态](package-readiness-current-state.md)
- [Package Readiness Summary 怎么读](readiness-summary-guide.md)
- [真实 Callback Runtime Evidence Schema](real-callback-runtime-evidence-schema.md)
