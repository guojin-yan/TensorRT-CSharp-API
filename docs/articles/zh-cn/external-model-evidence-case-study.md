# 外部模型 Evidence 回填案例总览

本文把外部模型 evidence 的三条线放在一起：TensorRtExec build-only、YoloVision/Classification real-model-runtime、package-consumer-runtime。它适合作为 owner 准备真实模型和发布证据前的总览。

## 三条证据线

| 证据线 | 证明什么 | 不证明什么 |
| --- | --- | --- |
| build-only | ONNX 可被转换或生成 report | 推理输出正确 |
| real-model-runtime | 真实模型在样例 runner 中跑出预期输出 | NuGet 包 clean consumer |
| package-consumer-runtime | 包在外部 consumer 中 restore/build/smoke | 具体模型精度或业务效果 |

Post publish verification 是第四条线：它只能在真实渠道发布后，从下载后的包和 clean consumer 记录中产生。

## 推荐案例包

```text
models/
  model.onnx
  labels.txt
  input.jpg
  build-report.json
  build-sidecar.json
  sample-run.log
  sample-run-evidence.json
```

每个文件都需要 SHA256，模型和图片还需要 license 说明。

## 执行顺序

1. 选定模型和 license。
2. 生成 TensorRtExec build-only report。
3. 写 sidecar，连接 model hash、report hash、runtime package key。
4. 运行样例 runner。
5. 回填 sample-run-evidence。
6. 运行 sample validators。
7. 如果要做 release proof，另行执行 clean package consumer。
8. 如果已经真实发布，再执行 post publish verification。

## 不能混淆

- sidecar-only 不是 runtime proof。
- sample-run-evidence 不能声明 `package-consumer-runtime`。
- build-only 不是 `real-model-runtime`。
- local feed 不是 post publish proof。
- ProjectReference 不是 clean consumer。
- `blocked-by-cuda-driver` 不是通过。
- `TrtexecAlignmentStatus=parse-only` 不代表高级参数已经完整执行。

## Owner action

owner 需要决定：模型是否可再分发、图片是否可用于文章、是否允许公开 hash、是否有兼容 CUDA 主机、是否已完成真实渠道发布。缺任一项，就应写 owner action required，而不是把文章示例写成 proof。
