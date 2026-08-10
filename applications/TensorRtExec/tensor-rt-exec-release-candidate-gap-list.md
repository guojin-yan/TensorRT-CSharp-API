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
| engine-packaging-refit-weight-streaming | implemented-build-refit-persist-reload-with-version-guards | 是 | 是 | 官方参数保持 versioned readback；managed extensions 已完成 TRT10/11 parser load、engine commit、persisted reload、enqueue 与输出比较，TRT11 另有 loadEngine-only 第二进程和 `69/69` strict 校验；本地包 consumer runtime 仍仅覆盖 TRT10 |
| io-layer-precision-policies | implemented-build-readback-with-version-guards | 是 | 是 | TRT10.11 外部 YOLOv8n-cls 已完成 5 项 requested/applied/readback、1000 元素参考校验，以及 detailed inspector 的实际引擎层 Float I/O 与选中 tactic 记录；内部计算/累加精度仍不冒充已观测，TRT11 移除的 setters 保持 guard |
| bounded-benchmark-scheduler | implemented-bounded-runtime | 是 | 是 | 独立 context/stream、预热、次数+时长双下限、stream-ordered sleepTime event fan-out、idle、平均窗口和 percentile 已执行；模型与 package proof 仍独立 |
| timing-cache | implemented-build-cache-lifecycle | 是 | 是 | 成功构建会导入/导出 cache 并记录 `TimingCacheArtifact` 大小与 SHA256；仍需 owner 将 cache 文件与真实模型 build 记录一起归档 |
| plugin-library-boundary | diagnostic-gui-cli | 是 | 是 | GUI/CLI 已共享 plugin path 字段，保持 register/load-library deferred |
| profiling | implemented-report | 是 | 是 | 真实 enqueue log 后才能晋级 |
| layer-dump | implemented-inspector-readback | 是 | 是 | 真实 build/load-engine 会复制 inspector layer data 并可导出；`detailed` verbosity 使用自描述 JSON 和顶层 `LayerInfoArtifact` 文件/hash 校验，其余模式保持 one-line text；仍是 diagnostic metadata |
| verbose-logging | implemented-report | 是 | 是 | hash owner stdout/stderr/log |
| binding-metadata | implemented-structured-pointer-free-binding-report-and-reference-validation | 是 | 是 | 外部静态 YOLOv8n-cls 已用隐式 profile 0 完成 build/load/runtime `BindingMetadata` 与结构化 reference comparison；generic 记录不替代 YoloVision 模型语义或公开 consumer 证据 |
| winforms-command-surface | shared-command-report-error-formatting-checklist-and-real-gui-build-backed | 是 | 是 | CLI/GUI 共用 normalized command 与 report/error formatter；保留 84 参数字段映射和真实 MNIST ONNX GUI build-only 记录 |
| package-consumer-runtime-proof-boundary | local-refitted-plan-package-consumer-runtime-public-proof-owner-action-required | 否 | 否 | 2026-08-09 当前 Release 本地包的 file-feed PackageReference consumer 与严格校验 `53/53` 已完成；公开 feed、post-publish 与 release proof 仍等待 Owner |

## 下一步

`load-engine` 已推进到 compatible-float bounded runtime：可反序列化 engine、复制结构化 `BindingMetadata`、创建 typed bindings 并 enqueue/readback；CLI 与 WinForms 继续共用 `TensorRtExecReportFormatter`。外部静态 YOLOv8n-cls 已在 TRT10.11 与 TRT11.0 上验证隐式 profile 0、FP32 I/O、1000 元素零 mismatch 参考输出，以及带长度/SHA256 的合法 detailed inspector JSON；目标卷积层记录了 Float 输入/输出、Float 权重/偏置和选中 tactic，但 TensorRT 没有提供独立的内部计算/累加精度字段。TRT8 Windows parser 已把 `createParser` vendor SEH/C++ exception 收敛为 native status，失败时保持 null pointer 且不创建 owner handle；独立 TensorRtExec 子进程已完成 MNIST parse、engine build/round-trip、enqueue 和 10 值零 mismatch，compact evidence 严格检查为 `28/28`。TRT8 detailed layer-info 与外部模型语义仍未采集，该结果只分类为 `synthetic-input-runtime`。跨版本记录位于 `samples/assets/tensorrtexec-yolov8n-cls-precision-policy-runtime-evidence.json`、`samples/assets/tensorrtexec-yolov8n-cls-cross-version-runtime-evidence.json` 与 `artifacts/interface-coverage/trt8-windows-onnx-parser-seh-boundary-evidence.json`。模型特定 real-model-runtime 仍由链接的 YoloVision 证据及 `applications/YoloVision/yolovision-task-output-contract.json` 负责。`tests/fixtures/package-consumers/RefittedPlan.PackageConsumer` 已从两个声明的本地 feed restore 本次 current Release managed/bridge 包，在仓库外完成 enqueue、raw output SHA 对照、owner cleanup 和 `53/53` strict validation；该结果只分类为 local package-consumer engineering evidence。WinForms 的 84 参数字段映射与 2026-08-04 MNIST build-only 截图继续保留，8 月 9 日 current-source Release build 被单独记录，没有伪装成截图或运行值重采。剩余公开 feed 外部 consumer proof 必须等实际发布后采集。
