# Deferred 下一批只读候选清单

## 适用读者

本文面向继续推进 deferred 边界提升的维护者，用于在下一轮直接选择 5 到 15 个安全只读 API，而不是重新从全仓库开始筛选。

## 解决问题

Deferred API 升级容易在 callback、allocator、borrowed pointer 和 plugin instance 生命周期上陷入高风险区域。本文列出下一批优先候选：engine/layer metadata、builder config readback、plugin field metadata、error recorder snapshot 和 readonly diagnostics，并明确 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不是 runtime proof。

## 背景与场景

项目已经完成接口追平，但真实完成度要看非 deferred 实现、高层 wrapper、smoke 和 package consumer proof。下一批应选择“读取复制快照”的 API，避免创建资源、注册资源、接管指针或进入跨 ABI callback。

## 操作路径

1. 用 `rg "deferred|plugin|field|engine|layer|builder config|error recorder"` 定位候选。
2. 对每个候选检查是否只读、是否可复制输出、是否无 ownership 接管。
3. 每批只挑 5 到 15 个 API，补 manifest、native source、C# interop、高层 wrapper、XML 注释和 smoke。
4. 对字符串和数组使用 caller buffer 或 count/copy 模式。
5. 对所有 public API 保持 pointer-free，不暴露裸 `IntPtr` 或 `nint`。

## 已固定的 B-tier 推进队列

后续阶段不要再从全量 deferred 文件重新摸底。机器工作包 `artifacts/interface-coverage/deferred-btier-implementation-work-package.json` 和闭环台账 `artifacts/interface-coverage/deferred-btier-work-item-proof-closure-ledger.json` 已把 `btier-001` 到 `btier-051` 固定为 `source-quality-proof-closed`，并由 `tests/JYPPX.ProjectQuality.Tests/DeferredBTierWorkItemProofClosureLedgerTests.cs`、`DeferredBTier41To45ProofClosureTests.cs` 与 `DeferredBTier46To50ProofClosureTests.cs` 持续回归。

| 批次 | 工作项 | 主题 | 当前处理方式 |
|---|---|---|---|
| 第一批 | `btier-001` 到 `btier-012` | execution context name、profiler interface info、engine profile shape、layer input、TRT10 builder/config scalar readback | 已进入 safe alternative / wrapper / docs / quality proof 收口，不删除 deferred history。 |
| 第二批 | `btier-013` 到 `btier-024` | TRT10 builder config scalar getter、TRT11 parser/refitter copied diagnostics、TRT8 builder compatibility getter | 已进入同一质量门禁，重点证明 public wrapper 和 native/source/manifest 证据链。 |
| 第三批 | `btier-025` 到 `btier-040` | TRT8 builder/config/engine/context/parser legacy safe alternative | 已进入质量门禁，证明 legacy copied value/string/shape wrapper，不把 smoke 或 readonly diagnostics 晋级为 runtime proof。 |
| 第四批 | `btier-041` 到 `btier-046` | TRT8 Caffe binaryproto copied snapshot、UFF required version copied snapshot | 已由 `DeferredBTier41To45ProofClosureTests` 和 `deferred-btier-41-45-proof-closure.md` 固化为 proof closure；保留 deferred history，不晋级 runtime/release proof。 |
| 第五批 | `btier-047` 到 `btier-051` | TRT10 tiling level、legacy implicit-batch bool、execution context NVTX verbosity、ONNX parser/refitter copied diagnostics | 已由 `DeferredBTier46To50ProofClosureTests` 和 `deferred-btier-46-50-proof-closure.md` 固化为 proof closure；保留 deferred history，不晋级 runtime/release proof。 |

## 下一批建议

下一轮应直接从新 candidate audit 或独立 runtime/model gap 中挑选，而不是重复处理 `btier-001` 到 `btier-051`：

1. **ONNX parser copied diagnostics / layer-output presence**：优先补强已存在的 `LayerOutputTensorExists`、subgraph count/copy、parser error copied diagnostics 和 sample/smoke 证据；继续避免返回 parser-owned `ITensor*`。
2. **IAlgorithm 结果快照设计**：仅做设计门禁和 copied snapshot 草图，暂不把 `IAlgorithm*`、`IAlgorithmContext*`、`IAlgorithmIOInfo*` 作为 public handle 暴露。
3. **IErrorRecorder owner snapshot**：本批已把 `TensorRtErrorRecorderSnapshot` 扩展到 Builder、NetworkDefinition、EngineInspector，并同步 TRT8/TRT10/TRT11 native/manifest/interop；继续禁止 ref-count ownership API 晋级。
4. **IPluginCreatorV3One / IVersionedInterface metadata**：只允许 copied interface info、name/version/namespace 快照；不进入 plugin create/clone/enqueue/resource。
5. **Legacy IStreamReader / IStreamWriter readonly diagnostics**：`IStreamReaderV2` 已完成 immutable native owner、borrower ledger、no-throw read/seek、pointer-free snapshot 和 TensorRT 10.11 本地双包实机证明；下一轮只评估 legacy reader、writer 与 TRT11 实机缺口，不重复实现 v2。
6. **CUDA graph / memory range copied query**：只选择 count/copy、scalar attribute、caller-owned output struct 的查询路径；不处理 user object、external memory、IPC、callback destructor 或跨进程 ownership。

已明确降级为 design gate 的候选不要硬推：

- `IDimensionExpr::isConstant/getConstantValue/isSizeTensor` 需要已知 owner object 和 shape callback lifetime；没有 owner lifetime 证明前不得新增 public `IDimensionExpr` wrapper。
- legacy `IStreamReader/IStreamWriter::getInterfaceInfo` 来自应用侧 callback object；仍不得暴露 native reader/writer pointer。已完成的 `TensorRtStreamReader` 只返回 copied snapshot，也不能作为 writer 或 legacy reader 的完成证明。
- plugin instance create/clone/enqueue、registry register/deregister/load library、resource acquire/release 仍保持 deferred。

## 每轮固定验收

- 每轮至少处理 8 到 15 个明确工作项，除非涉及真实 native ABI 变更需要拆小批。
- 每个工作项必须给出 manifest、native source、generated/manual interop、高层 wrapper、quality test 或 smoke 中至少三类证据。
- 每个工作项必须明确 `canPromoteReleaseProof=false`、`canDeleteDeferredRecord=false`。
- 每个工作项必须说明 deferred history 保留位置，不能删除 deferred manifest 来制造完成度。
- 每轮结束必须写 `plan/YYYY-MM-DD-HHmm-*.md`、`diary/YYYY-MM-DD-HHmm-*-开发日记.md`、`diary/YYYY-MM-DD-HHmm-下一阶段提示词-*.md`。

## 代码与文件入口

- `artifacts/interface-coverage/tensorrt-interface-comparison.csv`：deferred 行筛选。
- `native/manifests/tensorrt/v10` 与 `native/manifests/tensorrt/v11`：manifest。
- `native/src/tensorrt/common`：跨版本 adapter。
- `src/JYPPX.TensorRtSharp`：托管 wrapper。
- `tests/JYPPX.ProjectQuality.Tests`：质量门禁和 smoke。

## 图示建议

建议用优先级表：候选组、风险、输出模式、需要补的层、建议 smoke。将 callback/allocator/plugin instance 标红为暂缓。

## 边界说明

Readonly candidate list 是开发计划，不是 proof。即使某些 API 升级完成，TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、template、local feed、ProjectReference 和 direct `.nupkg` 仍不能替代 runtime proof、package-consumer-runtime proof 或 post-publish verification。

## 下一步

下一轮建议优先做 ONNX parser copied diagnostics / layer-output presence、CUDA graph/memory copied query 和 ErrorRecorder copied diagnostics 中的 8 到 15 个低风险工作项。若发现需要 borrowed pointer、callback owner、external resource 或跨 ABI ownership，立即降级为 design gate，不进入 public API。
