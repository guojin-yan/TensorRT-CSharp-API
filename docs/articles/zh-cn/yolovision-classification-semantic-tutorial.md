# YoloVision Classification 与 Semantic Segmentation 教程

YoloVision 不只服务检测类模型，也可以承接分类和语义分割任务。本文把 cls 与 sem 放在一篇中说明：它们都需要明确 labels、output layout 和真实 evidence，但不需要复用 detection 的 NMS 思路。

## Classification

分类任务关注 label score 排序：

- label count 必须和输出维度一致。
- 需要说明输出是 logit、probability 还是 softmax 后结果。
- Top-K 输出应写入 runner summary。
- input preprocess 必须记录 resize、crop、normalize。

推荐 evidence：

- model/labels/input SHA256。
- build-only report。
- runner log。
- Top-K summary。
- sample-run-evidence。

## Semantic segmentation

语义分割关注 dense class map：

- output spatial layout。
- class count。
- argmax 规则。
- 输出 resize 到原图的策略。
- palette 或 label mapping。

sem 不应该走 detection NMS。若文章中出现框、NMS、objectness，需要说明它们不属于 sem 的核心路径。

## 命令路径

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\vision-cls-or-sem.onnx `
  --saveEngine .\models\vision-cls-or-sem.engine `
  --buildOnly `
  --exportProfile .\models\vision-cls-or-sem-build-report.json
```

然后用 YoloVision 选择 `--task cls` 或 `--task sem`，并在 manifest 中写清 output metadata。

## 边界

- build-only 不证明分类 Top-K 或语义分割 mask 正确。
- parse-only 不证明高级 TensorRT 行为完整执行。
- sidecar-only 不是 runtime proof。
- sample-run-evidence 最多晋级 `real-model-runtime`。
- `package-consumer-runtime` 仍属于 release proof record。
- `blocked-by-cuda-driver` 是 owner action，不是通过。

## 下一步

当真实模型资产到位后，建议分别写两篇实战文章：分类 Top-K 验证和语义分割输出可视化。当前缺少模型、license、hash 或日志时，文案必须保持 owner action required。
