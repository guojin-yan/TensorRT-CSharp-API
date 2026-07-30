# Calibrator Metadata Design Gate

`calibrator-metadata-design-gate` 用来收口 `IInt8Calibrator` / `IInt8EntropyCalibrator` / `IInt8EntropyCalibrator2` / `IInt8LegacyCalibrator` / `IInt8MinMaxCalibrator` 这组中风险 deferred 候选。它只允许 presence 与 copied metadata 规划，不会调用 calibration callback，不会读取或写入 calibration cache，也不是 runtime execution proof。

源码按顶层职责分为 `TensorRtCalibratorMetadataDesignGate.cs` 与
`TensorRtCalibratorMetadataDesignGateResult.cs`；readiness 同时读取 evaluator/result source-set。

## 当前结论

- `RuntimeEvidenceKind=design-gate`。
- `PresenceProbeAvailable=True`，TRT8/TRT10 已通过 `TensorRtBuilderConfig.HasInt8CalibratorCompatibility` 暴露 presence-only 查询。
- `CopiedMetadataShapeReady=True`，后续可以设计 algorithm/interface-info copied metadata，但不能直接持有 calibrator 对象。
- `PointerFreeSurfaceReady=True`，public API 不暴露、返回或保存 calibrator pointer。
- `CalibratorPointerExposed=False`，`BorrowedCalibratorPointerEscaped=False`。
- `CallbackInvocationEnabled=False`，不能从 public API 调用 `getBatch`、`readCalibrationCache` 或 `writeCalibrationCache`。
- `BatchBufferAccessEnabled=False`，`CacheBufferAccessEnabled=False`。
- `DirectCalibratorCallbackRowsDeferred=True`，`DirectCalibratorCacheRowsDeferred=True`。
- `CanPromoteWithoutRuntimeProof=False`，`RuntimeProofBlocked=True`，`DeferredRowsStillRequired=True`，该门禁是 not proof。

## 已允许的 public 边界

当前只允许 builder config 上的 presence query：

- `TensorRtBuilderConfig.HasInt8CalibratorCompatibility`

该属性只返回是否附加了 INT8 calibrator，不转移 ownership，不暴露 borrowed calibrator pointer，也不能用来调用 calibrator callback。

## 继续 deferred 的内容

以下 direct rows 仍必须保留在 coverage / manifest 中：

| Interface | Method | 原因 |
| --- | --- | --- |
| `IInt8Calibrator` | `getAlgorithm` | 只能后续做 copied metadata；不能暴露 calibrator object。 |
| `IInt8Calibrator` | `getBatch` | callback 会进入用户 batch buffer，ownership 未建模。 |
| `IInt8Calibrator` | `getBatchSize` | callback surface 未建模前不能作为 public runtime proof。 |
| `IInt8Calibrator` | `readCalibrationCache` | cache buffer lifetime 与 ownership 未建模。 |
| `IInt8Calibrator` | `writeCalibrationCache` | cache buffer lifetime 与 ownership 未建模。 |
| `IInt8EntropyCalibrator` | `getAlgorithm` / `getInterfaceInfo` | 只允许 copied metadata 规划；direct callback 继续 deferred。 |
| `IInt8EntropyCalibrator2` | `getAlgorithm` / `getInterfaceInfo` | 只允许 copied metadata 规划；direct callback 继续 deferred。 |
| `IInt8LegacyCalibrator` | `getAlgorithm` / `getQuantile` / `getRegressionCutoff` | 只允许 copied metadata 规划；histogram cache ownership 未建模。 |
| `IInt8MinMaxCalibrator` | `getAlgorithm` / `getInterfaceInfo` | 只允许 copied metadata 规划；direct callback 继续 deferred。 |

## 下一步

真正提升前，需要先完成：

1. calibrator owner lifetime 模型。
2. no-throw callback boundary。
3. batch buffer 与 calibration cache buffer ownership 设计。
4. package-consumer runtime proof，而不是 dependency probe、design gate 或 dry-run。
5. 只在 metadata 能被复制且不暴露 borrowed pointer 时，才考虑 algorithm/interface-info copied snapshot。
