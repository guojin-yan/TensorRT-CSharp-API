# Algorithm Snapshot 安全设计门

## 目标

`IAlgorithm`、`IAlgorithmContext`、`IAlgorithmIOInfo`、`IAlgorithmVariant` 来自 TensorRT algorithm selector callback，生命周期短、owner 不清晰。当前阶段不把这些对象提升为 public handle，而是形成 callback-scoped copied snapshot 的设计门。

本设计门不是 runtime proof，也不删除 deferred history。

## 已覆盖接口

本阶段固定 14 个候选：

- `IAlgorithm::getTimingMSec`
- `IAlgorithm::getWorkspaceSize`
- `IAlgorithm::getAlgorithmVariant`
- `IAlgorithm::getAlgorithmIOInfoByIndex`
- `IAlgorithmContext::getName`
- `IAlgorithmContext::getNbInputs`
- `IAlgorithmContext::getNbOutputs`
- `IAlgorithmContext::getDimensions`
- `IAlgorithmIOInfo::getDataType`
- `IAlgorithmIOInfo::getStrides`
- `IAlgorithmIOInfo::getVectorizedDim`
- `IAlgorithmVariant::getImplementation`
- `IAlgorithmVariant::getTactic`
- `IAlgorithmSelector::getInterfaceInfo`

## 安全输出模式

输出模式固定为：

`callback-scoped copied algorithm timing, workspace, context, IO, and variant snapshot`

这意味着未来可实现方向只能是在 callback 返回前复制：

- timing / workspace scalar。
- algorithm context name、input/output count、dimensions。
- IO info data type、strides、vectorized dim。
- variant implementation、tactic。

任何 `IAlgorithm*`、`IAlgorithmContext*`、`IAlgorithmIOInfo*`、`IAlgorithmVariant*` 都不能逃逸到 public API。

## 当前阻塞点

- selector callback owner lifetime 未建模。
- algorithm result lifetime 未建模。
- callback trampoline 的异常边界、线程边界、返回数组 lifetime 未形成闭环。
- 没有 package-consumer-runtime proof。

## 当前证据链

- TRT8 deferred source：`native/src/tensorrt/v8/modules/deferred/cross_version_tenth_batch_other_deferred.inc`
- TRT10 deferred source：`native/src/tensorrt/v10/modules/deferred/cross_version_other_deferred.inc`
- comparison matrix：`artifacts/interface-coverage/tensorrt-interface-comparison.csv`
- design gate：`src/JYPPX.TensorRtSharp/Callbacks/Core/TensorRtAlgorithmSnapshotDesignGate.cs`
- result model：`src/JYPPX.TensorRtSharp/Callbacks/Core/TensorRtAlgorithmSnapshotDesignGateResult.cs`
- quality：`tests/JYPPX.ProjectQuality.Tests/AlgorithmSnapshotDesignGateTests.cs`

源码职责保持清晰：design gate 文件只拥有两组 evaluation 入口，Result 文件拥有构造、公开属性、候选集合、
阻塞项、诊断和 `ToString`。该归类不改变 public surface、pointer-free 边界或 non-proof 分类。

## 不可晋级项

下列内容仍禁止公开或晋级为 proof：

- `selectAlgorithms` / `reportAlgorithms` callback trampoline。
- public API 暴露 algorithm/context/ioinfo/variant native handle。
- 通过 build-only、dry-run、readonly diagnostics 或 sample report 替代 runtime proof。
- 删除 deferred manifest/source 行制造完成度。

## 下一步

下一阶段若继续推进 algorithm selector，应先设计 managed owner 与 native callback bridge，保证 callback 内复制 metadata，跨 ABI 不抛异常，并明确返回数组和 selector result 的生命周期。只有完成 owner、callback、snapshot、smoke 和 package consumer proof 后，才可考虑从 design gate 晋级为真实 wrapper。
