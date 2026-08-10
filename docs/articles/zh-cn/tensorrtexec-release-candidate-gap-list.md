# TensorRtExec 发布候选缺口清单

TensorRtExec 的方向是复刻官方 `trtexec` 的模型转换体验，同时提供 C# CLI 和 WinForms 页面。当前它已经覆盖了不少 build/report 能力，但离“可以作为发布候选功能集合”还差一层工作清单：哪些能力只是参数解析，哪些能生成报告，哪些仍然 deferred，哪些完全不能作为 runtime proof。

仓库中的机器可读清单位于：

```text
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.json
```

配套 Markdown 位于：

```text
applications/TensorRtExec/tensor-rt-exec-release-candidate-gap-list.md
```

## 怎么读这个清单

每一项都包含：

- 官方 `trtexec` 对应参数。
- 当前状态：`implemented`、`partial`、`implemented-report`、`diagnostic`、`deferred`、`owner-action-required`。
- 是否支持 CLI。
- 是否支持 WinForms。
- 是否能作为 runtime proof。
- 是否能作为 package-consumer-runtime proof。
- 下一步实现路径。

## 当前最重要的缺口

第一类是 bounded runtime 与模型正确性的边界。`--loadEngine` 已具备 read-only deserialize、binding metadata，以及 compatible float engine 的 typed enqueue/readback；但没有 expected output、模型来源、输入资产和 owner hash 时，仍不能宣传成模型正确性或 package runtime proof。

第二类是 builder config readback。`--workspace`、已知 `--memPoolSize` pool 和 `--avgTiming` 现在会在真实 build 中调用 typed setter 并用 getter read back；TRT8 的 `--minTiming` 使用 legacy compatibility setter，TRT10/11 保持 parse-only；dynamic profile 等仍需要报告和真实模型证据。readback 只说明 TensorRT 接收了 builder 配置，不是 runtime 输出或 package-consumer proof。

第三类是 benchmark scheduler 完整度。`--iterations`、`--warmUp`、`--duration`、effective `--streams/--infStreams`、`--sleepTime`、`--idleTime`、`--avgRuns`、`--percentile`、布尔 `--threads`、`--useSpinWait`、`--useCudaGraph` 和 `--noDataTransfers` 已按官方语义接入 bounded runtime，并完成 TRT10/CUDA12.9 smoke。sleepTime 使用 bridge-owned `cudaLaunchHostFunc` state，记录 event 后一次性扇出到全部推理 stream；CUDA graph 捕获失败的单次 run 仍保持 parse-only 并记录 fallback；no-transfer run 不读回输出、不声明模型正确性。

第四类是 deployment policy 的执行证明。`--device`、DLA/GPU fallback、tactic sources、DirectIO、sparsity enable/disable 和 strongly typed 已接入 typed set/readback 或 version-aware network creation：TRT10 使用 raw bit，TRT11 依赖 always-strongly-typed 契约。TRT10.11 identity smoke 只证明主机配置和 synthetic runtime；TRT8 strongly typed、sparsity force 保持 parse-only，DLA layer 真执行还需要 DLA 主机和真实模型。

第五类是 I/O 与 layer precision policy。TRT8/10 已完成官方 grammar、IO broadcast/count、exact-before-wildcard、later-rule override、单 output type broadcast 和 typed readback；TRT10.11 外部 YOLOv8n-cls 进一步验证 `fp32:chw` 输入/输出、`obey` 和首个卷积层 FP32 precision/output type 的 requested/applied/readback，并以 1000 个独立参考值零 mismatch 收口输出。详细 inspector 现在导出自描述的合法 JSON 并由顶层 `LayerInfoArtifact` 记录长度/SHA256。本次 TRT10.11 产物含 87 层，目标卷积层显示 Float 输入/输出、Float 权重/偏置和选中的 `sm80_xmma_fprop_implicit_gemm...` tactic；层数和 tactic 只绑定该次 engine build。TRT11.0 也完成 88 层导出、engine round-trip 与 1000 值零 mismatch，但只对 inferred type 已匹配的 FP32 I/O format 报告 applied，移除的 precision constraint/layer setters 保持 parse-only。TRT8 Windows parser 已通过原生 SEH/status 转换隔离风险，并在独立子进程完成 MNIST parse、engine build/round-trip、enqueue 与 10 值零 mismatch；本次没有 TRT8 detailed layer artifact，仍不能声明 layer policy 或外部模型语义。TensorRT 没有提供独立的内部计算/累加精度字段，因此这部分继续保持未观测边界，也不替代 YoloVision 模型语义证明。

第六类是 WinForms parity。当前 GUI 已覆盖 84 个 normalized options，命令预览继续由 `TensorRtExecOptions.ToArgumentLine()` 生成；CLI 与 WinForms 现在还共用 `TensorRtExecReportFormatter`，因此结构化 binding 摘要、refit 状态、最终状态和错误分类来自同一实现。8 月 4 日的真实 MNIST GUI build-only 截图与 assembly 证据保持原样，8 月 9 日的 formatter 源码更新被单独标记为未重采运行证据，不能借此晋级 runtime proof。

`binding-metadata` 也已从字符串化 `IOTensorSummaries` 收口为顶层 `BindingMetadata`：build、load-engine 和 bounded runtime 都会复制 engine-order input/output mode、dtype、engine/profile shape、location、format、vectorization、byte-size fallback 和 diagnostics，并固定 `PointerFreeCopiedSnapshot=true`、`CanPromoteRuntimeProof=false`、`CanPromoteReleaseProof=false`。外部静态 YOLOv8n-cls 现在使用隐式 profile 0，输入 `1x3x224x224`、输出 `1x1000` 均在 enqueue 前完成读回；证据位于 `samples/assets/tensorrtexec-yolov8n-cls-precision-policy-runtime-evidence.json`。YoloVision 仍负责模型特定的 semantic role 和 real-model-runtime 晋级。

第七类是 proof 边界。TensorRtExec 可以辅助生成 build report 和 sidecar，但不能替代 YoloVision real-model-runtime proof。本次 current Release 托管包与当前 bridge 包已通过仓外、仅两个本地 feed、PackageReference-only 的 refitted-plan consumer，严格校验为 `53/53`；它仍明确是 local package-consumer engineering evidence，不能替代 public-feed、post-publish 的 package-consumer-runtime proof。

## 配图建议

- 一张 CLI 参数到 WinForms 控件的映射图。
- 一张 gap list 表格截图。
- 一张 evidence ladder 图，说明 build report、sample run、real-model-runtime、package-consumer-runtime 的区别。

## 下一步

下一阶段不再重复补 TRT10.11 YOLOv8n-cls 的 I/O/layer requested-applied-readback、expected-output、引擎层 I/O datatype/format 与 tactic 本地证据；剩余工作是内部计算/累加精度的独立可观测性、其他受支持 TensorRT 线的 compatible-host owner record，以及公开 package consumer/post-publish 证明。`binding-metadata` 与 `winforms-command-surface` 已完成本地代码、schema、validator、CLI/GUI 输出和合同测试收口。
