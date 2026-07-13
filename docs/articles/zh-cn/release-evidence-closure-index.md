# 发布证据闭环索引

TensorRtSharp4.0 现在已经不能只按 “接口有没有登记” 来判断完成度。真正能支撑发布的，是样例、工具、文章、owner 输入、package consumer 验证和 post publish 验证串起来之后的证据链。`artifacts/final-release/release-evidence-closure-index.json` 就是这条证据链的索引。

这份索引不是 runtime proof，也不会执行发布。它的作用是把当前所有容易混淆的材料放到同一张表里：哪些只是矩阵，哪些只是 build report，哪些是 owner-action-required，哪些必须等真实 clean consumer 运行后才能晋级。

## 这份索引解决什么问题

项目里已经有很多材料：`samples/YoloVision/yolo-model-matrix.json`、`samples/OnnxToEngine/trtexec-parity-matrix.json`、`applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json`、30+ 文章路线、真实模型 evidence pack、package-consumer proof 模板、post publish 验证模板。单独看每个文件都合理，但发布时最怕把它们混成一种证明。

闭环索引把这些材料拆成六条 lane：

| Lane | 当前状态 | 能不能当 proof |
| --- | --- | --- |
| YoloVision real model runtime | owner-action-required | 不能，det/cls/seg/obb/pose/sem 都缺真实模型、日志、hash 和 owner review |
| OnnxToEngine conversion | build-report-only | 不能，build report 不证明输出正确 |
| TensorRtExec CLI/WinForms | tooling-ready-with-report-only-boundaries | 不能，report、GUI 截图和 dry-run 都不是 runtime proof |
| 中文文章体系 | documentation-ready-not-proof | 不能，文章只能解释证据链 |
| Package consumer runtime | template-only | 不能，必须由外部 clean consumer 跑通 |
| Post publish verification | blocked-real-publication-required | 不能，必须等公开发布后验证 |

## 为什么 YoloVision 矩阵不是 runtime proof

`YoloVision` 已经从早期检测样例扩展成统一 YOLO vision sample，覆盖 YOLOv5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det、cls、seg、obb、pose、sem。这个矩阵能说明托管后处理和 metadata 设计覆盖了哪些类型，但它不能证明某个真实外部模型已经跑通。

真实模型 proof 至少需要：

- owner 批准的 ONNX 模型、labels、输入图片或预处理 tensor。
- 模型来源、许可证、导出命令、SHA256。
- TensorRtExec 或 OnnxToEngine build report 和 engine SHA256。
- Owner 真实 YoloVision run log 中可出现 `YoloVision Passed=True`，但该日志项本身不是 release proof，仍需 asset/hash/host metadata/validator 一致。
- 输出 JSON、stdout/stderr 摘要、日志 hash。
- owner review 身份和时间。

缺少这些字段时，`YoloVision` 仍然只能算 support matrix、template 或 pipeline evidence。

## 为什么 TensorRtExec report 不是 package-consumer proof

`TensorRtExec` 的方向是复刻官方 `trtexec` 的体验：CLI、WinForms、shape profile、precision、timing cache、layer/profile dump 和 report schema 都很重要。但它生成的是 build/report evidence，不是 clean consumer runtime evidence。

package-consumer-runtime proof 必须满足更严格的条件：

- consumer 项目在仓库外部。
- 从 public package source restore。
- 没有 `ProjectReference`。
- 没有 local feed。
- 没有 direct `.nupkg` 引用。
- smoke exit code 为 0。
- smoke status 为 passed。
- native assets copied 为真。
- managed/runtime package hash、日志 hash、host metadata 全部真实匹配。

因此，TensorRtExec report 可以帮助 owner 构建 engine、记录参数和排查问题，但不能替代 `package-consumer-runtime-proof-record.json`。

## 禁止替代项

闭环索引集中列出了 forbidden substitutes。下面这些材料都不能被写成 runtime proof、package-consumer proof、post-publish proof、发布批准或 issue close ready：

- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- build-only
- dry-run
- preflight-only
- sidecar-only
- template-only
- handoff-only
- screenshot-only
- `Skipped=True`
- blocked-by-cuda-driver
- dependency-probe-only
- local feed
- ProjectReference
- direct `.nupkg`
- article
- roadmap

这条规则很重要：项目越接近发布，越不能用“看起来完整”的材料替代真实运行证据。

## 发布关闭还差什么

当前 release close 仍然被真实 owner 输入阻塞。最短闭环是：

1. 用真实模型填充 real-case evidence record，并通过 `eng/Test-RealCaseEvidenceRecord.ps1`。
2. 在兼容 CUDA/TensorRT 主机上跑外部 clean consumer，填充 `package-consumer-runtime-proof-record.json`，并通过 `eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof`。
3. 公开发布后，从 public package source 做 post publish clean consumer 验证，并通过 `eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`。
4. 最后运行 `eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。

闭环索引本身不会让这些 gate 通过。它只是把路线收敛到一个地方，让下一轮开发和 owner 回填都少走弯路。
