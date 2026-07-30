# TensorRT 8 RNNv2 Borrowed State 设计门

本批次处理 `IRNNv2Layer` 的 12 条 C-tier triage 行。目标不是机械删除 deferred，而是把安全 scalar、owner-bound borrowed tensor 和 copied weights 三类边界分开：

源码按顶层职责分为 `TensorRtRnnV2BorrowedStateDesignGate.cs` 与
`TensorRtRnnV2BorrowedStateDesignGateResult.cs`，不改变 triage 计数或 proof 分类。

- 2 条 `getDataLength` 记录已通过真实 manifest、native entrypoint、C# interop 和 `TensorRtLayer.GetRnnV2DataLength()` 提升。
- 10 条 borrowed tensor/weights 记录已通过 owner-bound wrapper 或 caller-buffer copy-out 提升。
- 旧 deferred manifest 记录继续保留为审计历史，不用于制造完成度。
- 全部 12 条 triage 行已离开 C-tier；这不代表真实 RNNv2 runtime 已执行。
- 本文描述的是 design gate，not runtime proof，也不是公开发布证明。

## 已提升的 Scalar

`IRNNv2Layer::getDataLength` 返回 `int32_t`，复用现有 `TensorRtLayer` owner handle，并通过 `BridgeStatusCode + out int32_t` 返回 copied scalar。该路径不会返回 `ITensor*`、`Weights.values` 或其他 TensorRT-owned 地址。

高层 API：

```csharp
int dataLength = layer.GetRnnV2DataLength();
```

该 API 仅适用于 TensorRT 8 RNNv2 layer；TRT10/TRT11 调用会得到明确的 `NotSupported`。

## Owner-Bound Tensor Reference

三个 TensorRT-owned `ITensor` getter 返回 bridge tensor wrapper，但 wrapper 本身不会销毁 TensorRT 的 `ITensor`。`TensorRtNetworkDefinition.GetLayer()` 创建 network owner lease，layer 和后续 borrowed tensor wrapper 共享该 lease；最后一个 wrapper 释放后才对 network SafeHandle 执行 `DangerousRelease`。

```csharp
TensorRtTensor? cellState = layer.GetRnnV2CellState();
TensorRtTensor? hiddenState = layer.GetRnnV2HiddenState();
TensorRtTensor? sequenceLengths = layer.GetRnnV2SequenceLengths();
bool ownerBound = cellState?.IsOwnerLifetimeBound ?? false;
```

如果 layer 不是通过 network owner 路径取得，调用这些 borrowed getter 会在托管侧抛出明确异常，不会跨 C ABI 抛异常。

## Copied Gate Weights Snapshot

`getBiasForGate` 和 `getWeightsForGate` 使用两阶段 caller-buffer 模式。第一次查询 metadata 与 required byte count，第二次复制字节；public API 不暴露 `Weights.values`。

```csharp
TensorRtRnnV2GateWeightsSnapshot weights =
    layer.GetRnnV2WeightsForGate(0, TensorRtRnnGateType.Input, isInputWeights: true);

TensorRtRnnV2GateWeightsSnapshot bias =
    layer.GetRnnV2BiasForGate(0, TensorRtRnnGateType.Input, isInputWeights: true);

byte[] copiedBytes = weights.ToArray();
```

snapshot 包含：

| 字段 | 含义 |
| --- | --- |
| `LayerIndex` | RNN 物理层索引。 |
| `Gate` | input/output/forget/update/reset/cell/hidden gate。 |
| `IsInputWeights` | 输入侧 W/Wb 或 recurrent 侧 R/Rb。 |
| `IsBias` | bias 或 weight matrix。 |
| `DataType` | TensorRT weight data type。 |
| `ElementCount` | 元素数量。 |
| `ByteCount` | 复制后的字节数。 |
| `ToArray()` | 每次返回新的托管字节副本。 |

native 层验证 RNNv2 layer 类型、layer index、gate enum、bridge boolean、负 count、空 buffer、byte overflow 和 buffer size。

## 设计门状态

- `EvidenceKind=rnnv2-borrowed-state-design-gate`
- `SelectedTriageRowCount=12`
- `PromotedScalarTriageRowCount=12`
- `RemainingDeferredTriageRowCount=0`
- `BorrowedTensorPointerExposed=False`
- `BorrowedWeightsPointerExposed=False`
- `BorrowedStateEscapesCall=False`
- `BorrowedSnapshotPromotionReady=True`
- `DeferredBorrowedRowsStillRequired=False`
- `CanPromoteRuntimeProof=False`

## 仍未完成的证明

1. 在 compatible host 上构造真实 TensorRT 8 RNNv2 network/layer。
2. 执行 cell/hidden/sequence borrowed tensor 路径，验证 layer/network/tensor dispose 顺序。
3. 执行至少一个 gate weight 和 bias copy，核对 type、element count、byte count 与内容。
4. 保存 package、host、stdout/stderr、runtime log 和 SHA256，并通过严格 runtime proof validator。

在上述真实执行完成前，`IsRuntimeExecutionProof` 和 `CanPromoteRuntimeProof` 必须保持 `false`。
