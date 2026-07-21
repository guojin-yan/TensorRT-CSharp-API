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

第三类是 benchmark scheduler 完整度。`--iterations`、`--warmUp`、`--duration`、effective `--streams/--infStreams`、`--idleTime`、`--avgRuns`、`--percentile`、布尔 `--threads`、`--useSpinWait`、`--useCudaGraph` 和 `--noDataTransfers` 已按官方语义接入 bounded runtime，并完成 TRT10/CUDA12.9 smoke。CUDA graph 捕获失败的单次 run 仍保持 parse-only 并记录 fallback；no-transfer run 不读回输出、不声明模型正确性。`--sleepTime` 仍必须保持 parse-only，直到存在忠实的 device-side launch-to-compute gap 实现。

第四类是 deployment policy 的执行证明。`--device`、DLA/GPU fallback、tactic sources、DirectIO、sparsity enable/disable 和 strongly typed 已接入 typed set/readback 或 version-aware network creation：TRT10 使用 raw bit，TRT11 依赖 always-strongly-typed 契约。TRT10.11 identity smoke 只证明主机配置和 synthetic runtime；TRT8 strongly typed、sparsity force 保持 parse-only，DLA layer 真执行还需要 DLA 主机和真实模型。

第五类是 I/O 与 layer precision policy。TRT8/10 已完成官方 grammar、IO broadcast/count、exact-before-wildcard、later-rule override、单 output type broadcast 和 typed readback；TRT11 只能在请求 type 与 inferred type 相等时设置 allowed formats，移除的 precision constraint/layer setters 保持 parse-only。identity smoke 证明路由和 readback，但不证明真实模型的 caller layout、tactic 或数值正确性。

第六类是 WinForms parity。GUI 不应该只是“能打开页面”，而是要能覆盖 CLI 的主要参数、生成可复制命令、展示 report 摘要和错误诊断。

第七类是 proof 边界。TensorRtExec 可以辅助生成 build report 和 sidecar，但不能替代 YoloVision real-model-runtime proof，更不能替代 clean external consumer 的 package-consumer-runtime proof。

## 配图建议

- 一张 CLI 参数到 WinForms 控件的映射图。
- 一张 gap list 表格截图。
- 一张 evidence ladder 图，说明 build report、sample run、real-model-runtime、package-consumer-runtime 的区别。

## 下一步

下一阶段优先补 I/O format 与 layer policy 的真实外部模型 layout/expected-output 证据、剩余 runtime mechanics、`binding-metadata` 和 `winforms-command-surface`；deployment policy、I/O/layer policy、`workspace-memory-pool` 与 `timing-iterations` 继续进入 compatible-host owner build record。所有实现都需要同步更新 CLI、WinForms、文档、测试和 proof 边界说明。
