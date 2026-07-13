# Real Model Evidence Backfill Playbook

这篇文章把 Classification 和 YoloVision 的真实模型 evidence 回填步骤放在同一处。它面向 owner 或样例维护者，强调模型、labels、input、license、hash 和 runner log 都必须真实存在；缺素材时只能记录 owner action，不能伪造 `real-model-runtime proof`。

## Proof 分层

| 层级 | 说明 | 不能替代 |
| --- | --- | --- |
| support matrix | YoloVision 支持 family/task/profile/postprocess 配置 | 真实模型运行 |
| build-only | TensorRtExec/OnnxToEngine 能生成 engine 或 report | 推理输出正确 |
| sidecar-only | 连接 build report 与模型资产 metadata | runtime proof |
| sample-run-evidence | 真实 runner log、hash、期望输出和 validator | package-consumer-runtime proof |
| real-model-runtime | 真实模型、真实输入、真实日志、license/hash/evidence 全部成立 | release package proof |

## Classification 回填步骤

1. 选择可使用的分类模型，记录下载来源和 license。
2. 准备 labels 和 input image，记录来源和 license。
3. 导出或获取 ONNX，记录 opset、input shape、preprocess。
4. 计算 model/labels/input SHA256。
5. 使用 TensorRtExec 或 OnnxToEngine 生成 build-only report 和 sidecar。
6. 运行 Classification sample，保存 runner log、stdout/stderr summary、Top-K 输出摘要。
7. 从 `sample-run-evidence-record.classification.template.json` 复制真实记录。
8. 运行 `Test-SampleAssetManifest.ps1` 与 `Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog`。

## YoloVision 回填步骤

1. 明确 family：v5、v6、v7、v8、v9、v10、v11、v26 或 custom。
2. 明确 task：det、cls、seg、obb、pose、sem。
3. 记录模型来源、license、export 命令、ONNX opset、input shape。
4. 记录 labels、input image、preprocess、layout、objectness、NMS、mask/keypoint/angle metadata。
5. 计算 model/labels/input SHA256。
6. 使用 TensorRtExec 生成 build-only report；注意 `TrtexecAlignmentStatus=parse-only` 的高级参数不能写成真实 TensorRT 行为 proof。
7. 运行 YoloVision sample，保存 runner log、stdout/stderr summary 和输出摘要。
8. 从 `sample-run-evidence-record.yolovision.template.json` 或任务专属模板复制真实记录。
9. 运行 manifest 和 sample evidence validators。

## 分任务注意事项

| 任务 | 必填 metadata | 常见风险 |
| --- | --- | --- |
| det | box layout、class count、confidence threshold、NMS mode | `[1,84,8400]` 与 `[1,8400,84]` 混淆 |
| cls | label count、Top-K、softmax/logit 语义 | labels 与输出维度不一致 |
| seg | box output、mask prototype、mask coefficients | 多输出 tensor role 未记录 |
| obb | angle unit、box layout、class count | degrees/radians 混淆 |
| pose | keypoint count、坐标 layout、visibility score | keypoint metadata 缺失 |
| sem | class count、spatial output layout、argmax 规则 | output resize/preprocess 不一致 |

## Validator 约束

sample run evidence record 只能晋级 `real-model-runtime`。它不能声明 `package-consumer-runtime`，也不能替代 external runtime proof、post publish verification proof、ProjectReference-free package proof 或 release close owner authorization。`blocked-by-cuda-driver` 仍是环境阻塞，不是真实模型 proof。

推荐验证：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 `
  -InputPath .\models\yolovision-sample-run-evidence.json `
  -RequireExistingLog
```

## Owner action 清单

- 提供或授权下载模型、labels、input image。
- 确认 license 是否允许项目文章和样例引用。
- 提供真实 SHA256，不使用占位 hash。
- 提供真实 runner log，不使用文档示例日志。
- 明确失败时是环境问题、模型问题、layout 问题还是 API 问题。

在这些材料补齐前，文档只能写 owner action required，不能写成真实模型 proof 已完成。
