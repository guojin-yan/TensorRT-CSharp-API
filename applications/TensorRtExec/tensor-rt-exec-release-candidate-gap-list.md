# TensorRtExec 发布候选 Gap List

本文件把 `tensor-rt-exec-trtexec-parity-matrix.json` 转成发布候选工作清单。它不是 runtime proof，也不是发布批准记录；它用于把 CLI、WinForms、OnnxToEngine、YoloVision 和 package-consumer proof 的缺口集中管理。

## 边界

- TensorRtExec build-only report 不是 runtime proof。
- TensorRtExec GUI 截图不是 runtime proof。
- local feed、ProjectReference、direct `.nupkg` 不是 package-consumer-runtime proof。
- package-consumer-runtime 只能来自干净外部 consumer 使用公开包的 strict validator。

## 优先级

| ID | 状态 | CLI | WinForms | 下一步 |
|---|---|---:|---:|---|
| onnx-input | implemented | 是 | 是 | 关联真实模型 candidate hash/log |
| save-engine | implemented | 是 | 是 | 增加 engine SHA256 owner overlay |
| load-engine | bounded-runtime-output | 是 | 是 | compatible float engine 可 bounded enqueue/readback；模型正确性仍由 runtime proof records 管理 |
| dynamic-shape | implemented-report | 是 | 是 | 绑定 profile metadata 与真实运行日志 |
| fp16 | wrapper-ready | 是 | 是 | 记录 owner host/model FP16 evidence |
| int8 | parse-report-only-calibration-boundary | 是 | 是 | CLI/WinForms 已暴露 INT8 与校准缓存意图，仍等待 calibration/cache ownership 设计 |
| workspace-memory-pool | implemented-readback-report | 是 | 是 | 继续把真实模型/运行证明交给 proof records |
| timing-iterations | implemented-builder-config-readback | 是 | 是 | `--avgTiming` 跨 TRT8/10/11 设置并 read back；TRT8 `--minTiming` 使用 legacy setter，TRT10/11 保持 parse-only |
| io-layer-precision-policies | implemented-build-readback-with-version-guards | 是 | 是 | TRT8/10 已完成 I/O 与 layer policy typed set/readback；TRT11 仅应用 type 已匹配的 allowed formats，移除的 precision setters 保持 guard |
| bounded-benchmark-scheduler | implemented-bounded-runtime | 是 | 是 | 独立 context/stream、预热、次数+时长双下限、idle、平均窗口和 percentile 已执行；其余 runtime mechanics 保持 parse-only |
| timing-cache | implemented-build-cache-lifecycle | 是 | 是 | 成功构建会导入/导出 cache 并记录 `TimingCacheArtifact` 大小与 SHA256；仍需 owner 将 cache 文件与真实模型 build 记录一起归档 |
| plugin-library-boundary | diagnostic-gui-cli | 是 | 是 | GUI/CLI 已共享 plugin path 字段，保持 register/load-library deferred |
| profiling | implemented-report | 是 | 是 | 真实 enqueue log 后才能晋级 |
| layer-dump | implemented-inspector-readback | 是 | 是 | 真实 build/load-engine 会复制 inspector layer text 并可导出文件；仍是 diagnostic metadata |
| verbose-logging | implemented-report | 是 | 是 | hash owner stdout/stderr/log |
| binding-metadata | implemented-pointer-free-binding-report | 是 | 是 | YoloVision 输出 JSON/控制台复制 `TensorRtEngineBindingReport` 的 role、dtype、shape、format、profile 和 diagnostics；仍需真实模型 owner 证据 |
| winforms-command-surface | checklist-backed-command-preview | 否 | 是 | 继续用 GUI/CLI 字段映射和命令预览锁定非 proof 边界 |
| package-consumer-runtime-proof-boundary | owner-action-required | 否 | 否 | 等待公开包外部 consumer proof |

## 下一步

`load-engine` 已推进到 compatible-float bounded runtime：可反序列化 engine、复制 metadata、创建 typed bindings 并 enqueue/readback；bounded benchmark scheduler 同时执行独立 execution context/stream、预热、次数+时长双下限、idle gap、平均窗口和 percentile。YoloVision 的 task/output role 仍以 `samples/YoloVision/yolovision-task-output-contract.json` 为机器契约。它们不替代模型 expected output、owner hash 或 package-consumer proof。下一步继续补 `--sleepTime` 等尚未实现的 runtime mechanics、WinForms parity checklist、owner proof schema 和真实外部 consumer proof；所有 proof 晋级仍必须由外部 proof validator 决定。
