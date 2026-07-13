# TensorRtExec 报告为什么不能替代真实模型运行 Proof

`applications/TensorRtExec` 是面向用户的 ONNX-to-engine CLI / WinForms 工具。它能把外部 ONNX、shape profile、precision、workspace、timing cache、report schema 和 GUI/CLI parity 组织成可审计的 build/report 工作流，但 `TensorRtExec report` 不能替代真实模型运行 proof。

## 适用读者

- 使用 `applications/TensorRtExec` 转换外部 ONNX 的用户。
- 需要审核 release proof 的 owner。
- 希望把 CLI/WinForms 报告写入文章或交接材料的维护者。

## 解决问题

TensorRtExec 的价值是把 trtexec-like 参数收拢到 .NET 工具中，帮助用户完成：

- ONNX 路径、engine 输出路径和参数归一化。
- `applications/TensorRtExec/tensor-rt-exec-trtexec-parity-matrix.json` 中的 parity 覆盖。
- CLI 与 WinForms 共用 `TensorRtExecOptions`、`TensorRtExecService` 和 report model。
- build-only report、load-engine preflight、dry-run preview 和 normalized command hash。

这些能力可以证明“工具表面”和“构建参数”可追溯，但不能证明模型语义正确，也不能证明公开包 clean consumer 或 post-publish verification 已经完成。

## 为什么 report 不是 runtime proof

真实模型运行 proof 至少需要真实模型、输入资产、输出 JSON、stdout/stderr、hash、validator 和 owner review。TensorRtExec report 可能只说明：

- 参数成功解析。
- ONNX 路径和 engine 路径被记录。
- build-only 或 preview 生成了报告。
- load-engine preflight 读取了元数据。
- GUI 截图展示了用户输入。

它没有自然包含任务级 postprocess、golden output、业务语义或 `YoloVision Passed=True` 这类样例运行证据。因此它必须保持 non-proof。

## 边界说明

以下内容不能替代 runtime proof：

- build-only
- dry-run
- template
- local feed
- ProjectReference
- direct `.nupkg`
- TensorRtExec report
- YoloVision matrix
- OnnxToEngine report
- readonly diagnostics
- design gate
- blocked-by-cuda-driver

`sample-run-evidence` 不能替代 `package-consumer-runtime`，`package-consumer-runtime` 不能替代 `post-publish verification`。TensorRtExec report 只能作为工具采用路径和 owner 回填辅助材料，不能写成“可公开发布”或“可关闭 release issue”的通过状态。

## 可复制命令

下面命令适合生成 build/report 辅助材料，不是 runtime proof：

```powershell
dotnet run --project applications\TensorRtExec -- `
  --onnx <owner-model.onnx> `
  --saveEngine <owner-model.engine> `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --fp16 `
  --buildOnly `
  --exportReport <owner-build-report.json>
```

真实 proof 需要由 `samples/YoloVision`、`samples/Classification` 或用户自己的推理程序补齐输入、输出、日志和 validator。

## 截图与图示建议

- CLI 与 WinForms 字段对照图。
- `tensor-rt-exec-trtexec-parity-matrix.json` 参数覆盖表。
- “TensorRtExec report -> sample-run-evidence -> package-consumer-runtime -> post-publish verification” 分层图。

## 下一步

1. 将 TensorRtExec report 继续定位为 build/report evidence。
2. 将真实模型语义 proof 交给任务级 sample runner 和 owner evidence 模板。
3. 将 public package clean consumer 和 post-publish verification 保持为独立 release proof ladder。
4. 在文档中持续声明：TensorRtExec report 不是 runtime proof、不是 public package proof、不是 post-publish proof、不是 package push、不是 release close approval。
