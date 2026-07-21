# trtexec-like Option Coverage

生成时间：2026-07-03

## 当前结论

本记录用于说明 `samples/OnnxToEngine`、`applications/TensorRtExec` CLI 与 WinForms 在本阶段的参数覆盖边界。它不是 runtime proof，也不是 package-consumer-runtime 证明；真实发布证据仍必须由 release proof record 和外部 package consumer smoke 补齐。

报告现在输出 `OptionImplementationStatus`，将参数分为 `ParsedOptions`、`AppliedOptions` 和 `ParseOnlyOptions`。该字段用于防止把 parser/report 覆盖误读成真实 TensorRT 行为覆盖：高级 timing、precision constraints、engine packaging/refit、weight-streaming、safety/consistency、builder cache、timing cache export 与 `--infStreams` 仍必须留在 parse-only 边界内。

Runtime/output artifact 现在输出结构化 proof boundary：`ArtifactProofBoundary`、`RuntimeProofClass`、`HasTensorOutputProof`、`HasRawBindingProof`、`IsBuildOnlyEvidence`、`IsDependencyProbeOnly`、`IsSyntheticRuntime`、`ModelSource`、`EnginePath` 和 `PreflightMetadata`。当 runtime artifact 路径存在时，还会生成相邻 `*.engine-readback.json`，单独记录 `LoadedEngineDiagnostics`、`ReadbackFingerprint`、`ReadbackSha256`、`ReadbackAvailable` 和 skipped reason。这些字段只说明 artifact 自身的证据等级，不替代 release proof record；`synthetic-input-runtime is not real-model-runtime`，readback hash / artifact 也不能替代 `package-consumer-runtime`。

TrtexecAlignmentStatus=parse-only 是高级 trtexec-like 参数的默认对齐状态，表示 parser/report/GUI 已接住参数，但真实 TensorRT 行为仍需要 native 实现、模型级 smoke 和运行证据才能提升。

## 已对齐参数

| 分组 | 参数 | 覆盖入口 | 当前语义 |
| --- | --- | --- | --- |
| Engine 输入输出 | `--onnx`、`--saveEngine`、`--save-engine`、`--engine`、`--loadEngine`、`--load-engine` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | build 或 load-engine readonly diagnostics；`--onnx` 与 `--loadEngine` 互斥；`PreflightMetadata` 输出 exists、length bytes、SHA256 和 proof 边界；`*.engine-readback.json` 只记录只读 metadata |
| Shape profile | `--shapes`、`--inputShapes`、`--minShapes`、`--optShapes`、`--maxShapes` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | 生成 optimization profile |
| Precision | `--fp16`、`--bf16`、`--int8`、`--noTF32` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | FP16/BF16/TF32 进入 builder config；INT8 仍是 calibrator 边界诊断 |
| Workspace / pools | `--workspace`、`--memPoolSize` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | 无后缀默认 MiB；支持 GiB/GB/MiB/MB/KiB/KB/B；memPoolSize 要求整 MiB |
| Timing cache / plugin | `--timingCacheFile`、`--timingCache`、`--plugins` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | 记录诊断，不加载 plugin library，不导入/导出 timing cache |
| Profiling / layer info | `--profilingVerbosity`、`--verbose`、`--dumpLayerInfo`、`--exportLayerInfo` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | profiling verbosity 归一化为 `none` / `layer_names_only` / `detailed` |
| Runtime timing | `--iterations`、`--warmUp`、`--duration`、`--streams`、`--infStreams`、`--useCudaGraph` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | bounded runtime 实际执行；CUDA graph 只有 capture/instantiate/launch 全部成功才 applied，否则写 fallback reason 并保留 parse-only |
| Advanced timing | `--avgTiming`、`--minTiming` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | `--avgTiming` 在 TRT8/10/11 真实构建中 setter/readback；TRT8 `--minTiming` 使用 legacy setter，TRT10/11 保持 parse/report-only；只覆盖 builder evidence，不声明 tactic quality 或 runtime proof |
| Builder scalar controls | `--maxNbTactics`、`--tilingOptimizationLevel`、`--l2LimitForTiling`、`--quantizationFlags` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | TRT10/11 compatible-host build-only 已验证 max tactics/tiling/L2 `3MiB` readback；L2 `256MiB` 受控拒绝并保留实际 3MiB；TRT8 实际 quantization readback、现代 scalar unsupported；TRT11 quantization removed-by-vendor；TRT8 parser-enabled snapshot 仍缺 cuDNN8 |
| Deployment controls | `--device`、`--useDLACore`、`--allowGPUFallback`、`--tacticSources`、`--directIO`、`--sparsity`、`--stronglyTyped` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | TRT10.11/CUDA12.9 identity smoke 已验证 device thread、typed builder set/readback、strongly typed network、engine round-trip 与 enqueue；TRT8 strongly typed、sparsity force 保持 parse-only，DLA 真执行仍需 DLA 主机和真实模型 |
| Precision constraints | `--precisionConstraints`、`--layerPrecisions`、`--layerOutputTypes` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | parse/report-only；等待模型专属 layer precision routing |
| Engine packaging / refit | `--versionCompatible`、`--excludeLeanRuntime`、`--stripWeights`、`--refit`、`--weightStreamingBudget` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | parse/report-only；不伪造成 lean runtime、weight stripping、refit 或 weight streaming 已真实执行 |
| Safety / consistency | `--safe`、`--consistency` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | parse/report-only；记录安全 runtime / consistency check 意图，不声明已完成安全 runtime 行为 |
| Builder cache policy | `--builderCache`、`--noBuilderCache` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | parse/report-only；两个选项互斥，当前不声明 builder cache 生命周期已提升 |
| Timing cache export | `--exportTimingCache` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | parse/report-only；timing cache export lifecycle 未在本阶段提升 |
| Runtime/output | `--noDataTransfers`、`--useSpinWait`、`--threads`、`--avgRuns`、`--percentile`、`--sleepTime`、`--idleTime`、`--loadInputs`、`--dumpOutput`、`--dumpRawBindingsToFile`、`--exportOutput`、`--exportTimes`、`--exportProfile`、`--saveProfile` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | threads/spin/no-transfer/avg/percentile/idle 在 bounded run 后可 applied；no-transfer 抑制 H2D/D2H、output match 与 output/raw export，`--sleepTime` 仍 parse-only；load-engine readback artifact 不晋级模型或包 proof |
| Report / evidence | `--exportReport`、`--report`、`--evidenceSidecar`、`--dryRun`、`--previewOnly`、`--buildOnly`、`--skipInference` | OnnxToEngine / TensorRtExec CLI / TensorRtExec WinForms | `--report` 是 `--exportReport` 的兼容别名并归一化回 canonical 参数；形成 build/report evidence，不声明 package-consumer-runtime |
| Option implementation status | `OptionImplementationStatus.ParsedOptions`、`AppliedOptions`、`ParseOnlyOptions` | OnnxToEngine / TensorRtExec report / TensorRtExec report | 明确区分 parsed、applied、parse-only；parse-only/build-only evidence 不能提升 real-model-runtime 或 package-consumer-runtime |

## 未提升边界

- plugin register / deregister / load library / deregister library 尚未开放。
- timing cache 与 builder cache 导入、导出和生命周期 ownership 尚未开放。
- INT8 calibrator callback 尚未完整提升。
- DLA builder config 已有 range check 与 readback，但 DLA layer placement、IO format binding、custom plugin runtime execution 仍需要模型级 smoke。
- load-engine readonly diagnostics 最多记录文件 metadata 和只读反序列化 readback metadata；`*.engine-readback.json` 不推断 binding、不执行 inference、不晋级 runtime proof。
- build-only artifact 必须保持 `HasTensorOutputProof=false` 与 `HasRawBindingProof=false`；dependency-probe-only artifact 只能把 `PreflightMetadata` 作为文件/依赖 metadata 证据。
- raw bindings 只有 embedded synthetic runtime 真实执行且输出匹配时才写二进制；其他路径写 skipped JSON，不能被当作 tensor 输出证明。
- evidence sidecar 只能记录模型/hash/license/log 摘要，不能把 build report 提升为 `package-consumer-runtime`；sample run evidence record 声明 `package-consumer-runtime` 必须被 validator 拒绝。
- `blocked-by-cuda-driver` 只能记录为环境阻塞，不是 smoke passed。

## 验证要求

- `OnnxToEngineTrtexecLikeTests` 覆盖 alias、memory unit、profiling verbosity、runtime/output artifact、engine-readback artifact、safety/consistency 和 builder cache parse-only 边界。
- `TensorRtExecApplicationTests` 覆盖 CLI usage、WinForms 字段、advanced timing / precision / packaging / refit / weight-streaming / safety / builder-cache surface 与共享 `TensorRtExecOptions` 参数线。
- 高级 trtexec-like 参数必须在 report diagnostics 与 `OptionImplementationStatus.ParseOnlyOptions` 中保留 parse-only，直到 native TensorRT 行为和模型级 smoke 能证明它们已真实执行。
- `ReleaseCandidateReadinessTests` 覆盖 sidecar 记录 `package-consumer-runtime` 但不晋级，以及 sample run evidence record 禁止声明 `package-consumer-runtime`。
- release readiness 在缺少真实外部 runtime proof 前仍应失败，不能用本文件替代发布证据。
