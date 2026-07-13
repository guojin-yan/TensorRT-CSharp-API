# YoloVision Pose 教程

Pose 模型在检测框之外还输出 keypoints。本文说明如何记录 keypoint metadata、如何生成 build-only report，以及如何把真实 pose runner log 回填到 sample-run-evidence。

## 关键 metadata

| 字段 | 说明 |
| --- | --- |
| keypoint count | 每个目标的关键点数量 |
| coordinate layout | x/y 或 x/y/score 排列 |
| visibility score | 是否存在可见性或置信度 |
| class count | 是否只有 person 或多类别 |
| image transform | resize、letterbox、坐标还原 |

Pose 的最大风险是把 detection layout 和 keypoint layout 混在一起。metadata 必须明确每个输出 tensor 的 role。

## 命令示例

```powershell
dotnet run --project .\applications\TensorRtExec -- `
  --onnx .\models\yolo-pose.onnx `
  --saveEngine .\models\yolo-pose.engine `
  --fp16 `
  --buildOnly `
  --exportProfile .\models\yolo-pose-build-report.json
```

随后运行 YoloVision 时使用 `--task pose`，并在资产 manifest 中写清 keypoint metadata。

## 证据回填

真实 pose evidence 至少包含：

- model/labels/input SHA256。
- TensorRtExec build report。
- runner log。
- 检测框数量和 keypoint summary。
- sample-run-evidence record。

sample-run-evidence 只能用于 `real-model-runtime`，不能用于 `package-consumer-runtime`。如果当前主机出现 `blocked-by-cuda-driver`，应记录为 owner action，不要写成运行通过。

## 常见误区

- build-only report 不是 pose 输出正确 proof。
- keypoint score 与 class score 不能混用。
- sidecar-only 不是 runtime proof。
- parse-only 参数不能证明官方 trtexec 行为已完整执行。
- ProjectReference-free package proof 属于 release 链路，不属于 pose 样例本身。

## 下一步

当 pose 样例证据稳定后，可以写真实案例文章：模型来源、ONNX export、输入图、输出可视化、关键点数量、日志摘要和 validator 结果。缺少模型授权时，保持 owner action required。
