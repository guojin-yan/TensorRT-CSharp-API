# Dimension Expression Snapshot Design Gate

`dimension-expression-snapshot-design-gate` 用来收口 `IDimensionExpr::isConstant/getConstantValue/isSizeTensor` 与 `IExprBuilder::constant/operation/declareSizeTensor` 这组中风险 deferred 候选。它是 pointer-free 设计门，不是 native ABI 实现，也不是 runtime execution proof；换句话说，它是 design gate，not proof。

## 已完成

- 新增 public `TensorRtDimensionExpressionSnapshotDesignGate` 与 `TensorRtDimensionExpressionSnapshotDesignGateResult`。
- dependency-probe smoke 输出 `DimensionExpressionSnapshotDesignGate=dimension-expression-snapshot-design-gate;...`。
- readiness / release evidence 能识别该 marker，并保持 `RuntimeEvidenceKind=design-gate`。
- public surface 固定为无裸指针：`ExpressionPointerExposed=False`、`ExpressionPointerProduced=False`、`BorrowedExpressionPointerEscaped=False`。
- `IExprBuilder` 创建路径继续关闭：`ExprBuilderCreationEnabled=False`。

## 关键边界

该 gate 只说明项目已经把 DimensionExpr 的下一步设计边界写清楚：

- `SnapshotTypeReady=True`
- `RequiredOutputMode=copied bool/int64 scalar snapshot tied to a proven owner object`
- `CandidateMethods=IDimensionExpr::isConstant, IDimensionExpr::isSizeTensor, IDimensionExpr::getConstantValue, IExprBuilder::constant, IExprBuilder::operation, IExprBuilder::declareSizeTensor`
- `OwnerLifetimeKnown=False`
- `ExpressionPointerExposed=False`
- `BorrowedExpressionPointerEscaped=False`
- `ExprBuilderCreationEnabled=False`
- `DirectDimensionExpressionRowsDeferred=True`
- `DirectExpressionBuilderRowsDeferred=True`
- `CanPromoteWithoutRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `DeferredRowsStillRequired=True`

这些字段意味着：即使 manifest/source coverage 已有对应 entry，也不能把 `IDimensionExpr*` 或 `IExprBuilder*` 当作 C# public 对象暴露。只有当 owner object lifetime、plugin shape callback lifetime 和 package-consumer runtime proof 都可验证后，才能进入真实 native ABI 提升。

## Deferred 行必须保留

以下 direct rows 仍必须保留在 coverage / manifest 中：

| Interface | Method | 原因 |
| --- | --- | --- |
| `IDimensionExpr` | `getConstantValue` | borrowed expression lifetime 未绑定到安全 owner。 |
| `IDimensionExpr` | `isConstant` | borrowed expression lifetime 未绑定到安全 owner。 |
| `IDimensionExpr` | `isSizeTensor` | TRT10/TRT11 才有，且仍需 owner lifetime。 |
| `IExprBuilder` | `constant` | expression node ownership 未建模。 |
| `IExprBuilder` | `operation` | expression node ownership 未建模。 |
| `IExprBuilder` | `declareSizeTensor` | TRT10/TRT11 才有，且 node lifetime 未建模。 |

## 下一步

真正提升前，需要先确定：

1. 哪个 high-level owner 能安全复制 `IDimensionExpr` metadata。
2. copied snapshot 是否能覆盖 constant value、constant flag、size tensor flag。
3. plugin shape callback / expression builder 生命周期是否能在 native bridge 内部闭环。
4. runtime proof 是否来自 full package consumer，而不是 dependency probe、design gate 或 driver-blocked smoke。

下一轮如果没有证明 owner object lifetime，不要新增 public `IDimensionExpr` wrapper；只允许继续扩展 design gate、候选清单和质量门禁。
