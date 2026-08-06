# YoloVision Classification 真实模型教程

本文说明如何把 YoloVision 从 detection 扩展到 classification 场景。classification 没有检测框和 NMS，但它同样需要模型来源、labels、输入预处理、输出 top-k 和真实 runtime proof 记录。

## 适用读者

适合想用 YOLO classification 或轻量分类 ONNX 模型验证 TensorRtSharp 推理链路的用户，也适合准备撰写分类模型案例文章的维护者。

## 解决问题

classification 案例常被误认为“输出一个 tensor 就算完成”。本文要求明确 labels 顺序、resize/crop/normalize 参数、top-k 规则和输出 dtype，避免把 build-only engine 或 readonly diagnostics 写成分类结果正确。

## 背景与场景

分类模型适合作为真实模型 proof 的第一批候选，因为后处理相对简单：输入图片经过固定预处理，输出 logits 或 probabilities，再映射到 labels。它可以帮助验证 package consumer、runtime assets、binding、enqueue 和 readback 的完整链路。

## 操作路径

1. 选择许可证清晰的分类模型，并记录来源、版本和 SHA256。
2. 导出 ONNX，记录 input name、shape、dtype 和 normalization 参数。
3. 用 OnnxToEngine report 做 build-only 预检。
4. 用 YoloVision classification profile 执行真实图片推理。
5. 保存 top-k 输出、labels hash、模型 hash、图片 hash、host metadata 和 validator 结果。

## 代码与文件入口

- `applications/YoloVision/Program.cs`
- `samples/ComputerVision/01.Classification/Program.cs`
- `samples/assets/yolovision-assets.template.json`
- `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json`
- `docs/articles/zh-cn/classification-real-asset-walkthrough.md`

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。classification proof 还必须证明 labels、preprocess 和 top-k 输出与真实运行日志一致。

## 下一步

下一步选择一个可公开再分发或用户自行下载的分类模型，把下载命令、ONNX 导出命令和 YoloVision classification runner 输出补成完整博客案例。
