# TensorRtExec 参数分层深挖

TensorRtExec 的目标不是简单复制一个命令行外壳，而是把官方 `trtexec` 的常用转换、诊断和部署参数分层接入到 .NET 工作流中。本文适合正在把 ONNX 模型转成 TensorRT engine 的用户，也适合 release owner 审核 build report、sidecar 和 runtime proof 边界。

## 一句话结论

TensorRtExec 当前可以作为 CLI/WinForms 双入口接收 trtexec-like 参数，生成 normalized command 和 build report；但高级参数必须继续区分 implemented、parse/report-only、build-only 和 release proof。`TrtexecAlignmentStatus=parse-only` 是保守边界，不是功能失败。

## 参数分层

| 层级 | 含义 | 例子 | 证据 |
| --- | --- | --- | --- |
| implemented | 参数真实进入当前 build/report 工作流 | ONNX、engine、workspace、shape profile、precision flags 中已落地部分 | build report、diagnostics |
| parse/report-only | 参数被 CLI/GUI/parser/report 接收，但 native TensorRT 行为未完整提升 | TRT10/11 `--minTiming`、未执行的 `--infStreams`、TRT11 已移除的 layer precision setters、weight streaming | `OptionImplementationStatus.ParseOnlyOptions` |
| implemented-builder-config-readback | 真实 build 调用 typed builder-config setter 并回读请求值；只覆盖 builder evidence | `--avgTiming`（TRT8/10/11）、TRT8 legacy `--minTiming` | `TrtexecTiming` log 与 `OptionImplementationStatus.AppliedOptions` |
| build-only | 能证明 engine 构建或报告生成，不证明推理输出正确 | TensorRtExec build report、OnnxToEngine report | build-only evidence |
| release proof | 只能来自 clean consumer 或真实发布后验证 | package-consumer-runtime、post publish verification | release proof record |

## OptionImplementationStatus 怎么读

报告中的 `OptionImplementationStatus` 用来把参数拆成三类：`ParsedOptions`、`AppliedOptions`、`ParseOnlyOptions`。这不是换一种方式包装完成度，而是防止把 parser/report 覆盖误写成真实 TensorRT 行为。

如果某个参数在 `ParseOnlyOptions` 中，应该写成：

- 当前 parser/report/GUI 已接收该参数。
- 当前仍需 native TensorRT 行为提升和模型级 smoke。
- 当前不能作为 package-consumer-runtime proof。

不要写成：

- 该官方行为已经完整复刻。
- 该参数已经通过真实 runtime proof。
- 该 sidecar 能替代 release close proof。

上面三种写法都属于过度声明。更准确的写法是：sidecar 只能帮助复核参数、模型资产和 build report，它不能关闭 release issue，也不能替代真实 package-consumer-runtime、real-model-runtime 或 post-publish verification proof。

## 推荐命令路径

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\model.onnx `
  --saveEngine .\artifacts\models\model.engine `
  --workspace 4096 `
  --fp16 `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:1x3x640x640 `
  --buildOnly `
  --exportProfile .\artifacts\models\build-report.json
```

这条命令可以形成 build-only/report 证据。它不证明真实图片推理正确，也不证明 package-consumer-runtime。

## GUI 工作流

WinForms 页面适合团队交付时复核参数：左侧填 ONNX、engine、precision、shape、workspace；中间查看 command preview；右侧查看 diagnostics 和 report 输出。GUI 的价值在于减少命令拼写错误，不在于提升 proof 等级。

## 证据边界

必须保留这些边界词：

- `TrtexecAlignmentStatus=parse-only`
- `build-only`
- `parse-only`
- `sidecar-only`
- `blocked-by-cuda-driver`
- `package-consumer-runtime`
- `real-model-runtime`
- `owner action`

TensorRtExec 的 report、sidecar 和 normalized command 可以帮助 owner 补齐真实证据，但不能替代 `external-runtime-proof-record.json`、post-publish verification record 或 sample-run-evidence record。

## 常见误区

| 误区 | 正确说法 |
| --- | --- |
| 有 engine 就等于推理通过 | engine 构建是 build-only，推理输出还需要 sample runner 或 consumer smoke |
| parse-only 参数等于已经实现官方行为 | parse-only 只说明入口和报告已接住 |
| sidecar 等于 runtime proof | sidecar 只是连接模型资产和 build report |
| 本机 blocked-by-cuda-driver 是 API 缺失 | 这是环境兼容阻塞，需要 compatible host owner action |

## 下一步

要把某个 parse/report-only 参数提升为真实能力，需要 native 行为、C# wrapper、模型级 smoke 和质量测试一起补齐。要把某次模型转换提升为 release proof，需要进入 package-consumer-runtime 或 post publish verification 链路。
