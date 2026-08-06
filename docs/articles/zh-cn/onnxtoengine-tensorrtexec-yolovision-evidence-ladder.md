# OnnxToEngine、TensorRtExec 与 YoloVision 证据梯度

本项目现在有三类容易混淆的用户入口：`applications/OnnxToEngine`、`applications/TensorRtExec` 和 `applications/YoloVision`。它们都能帮助用户走向 TensorRT 部署，但证据等级不同，不能互相替代。

## 梯度总览

| 层级 | 入口 | 证明什么 | 不证明什么 |
|---|---|---|---|
| 教程层 | `applications/OnnxToEngine` | 最小 ONNX parser、profile、engine serialization、identity round-trip 教程 | 不证明用户真实模型质量，不证明公开包可消费 |
| 应用层 | `applications/TensorRtExec` | CLI/WinForms 参数、build report、parity matrix、sidecar、preflight | 不证明 real-model-runtime，不证明 package-consumer-runtime |
| 样例层 | `applications/YoloVision` | YOLO-family 后处理、真实模型运行候选、sample-run evidence | 不证明 NuGet package consumer proof |
| 模型 proof | `real-model-runtime` record | 某个真实模型、真实输入、真实 log 和 hash 已通过 | 不证明公开包发布后 clean consumer 可用 |
| 发布 proof | `package-consumer-runtime` record | 公开包源 + clean external consumer + runtime package key + smoke passed | 不证明所有模型质量，只证明发布包消费路径 |

## OnnxToEngine

`OnnxToEngine` 适合作为最小教程：

- 创建或读取 ONNX。
- 设置 shape profile。
- 构建 serialized engine。
- 做最小 round-trip 或 build report。

它的输出可以作为学习材料和 build-only evidence，但不能作为 public package proof。

## TensorRtExec

`TensorRtExec` 是应用程序：

- 支持 CLI。
- 支持 WinForms。
- 接近官方 `trtexec` 参数分层。
- 能导出 report、layer info、profile path、sidecar。

它适合在真实模型 proof 之前完成 build/report 和参数归一化。但 `dry-run`、`parse-only`、`build-only`、`dependency-probe-only`、GUI 截图和 sidecar 都不能晋级 release proof。

## YoloVision

`YoloVision` 是真实模型样例入口：

- 支持 YOLOv5/v6/v7/v8/v9/v10/v11/v26。
- 支持 det/cls/seg/obb/pose/sem。
- 支持输入 tensor、labels、output role map 和后处理 metadata。

只有当 owner 提供模型、labels、图片、预处理 tensor、运行日志、SHA256、stdout/stderr summary 和 license evidence 后，它才能形成 `real-model-runtime` 候选。

## 不能替代 proof 的材料

以下材料不能替代真实 proof：

- template
- draft
- README
- 技术文章
- dry-run
- parse-only
- build-only
- dependency-probe-only
- local feed
- ProjectReference
- direct `.nupkg`
- GUI 截图
- TensorRtExec build report
- sidecar-only
- blocked-by-cuda-driver

## 正确路线

1. 使用 `OnnxToEngine` 学会最小 build。
2. 使用 `TensorRtExec` 生成真实模型的 build report 和 sidecar。
3. 使用 `YoloVision` 或 `Classification` 运行真实模型。
4. 回填 sample-run evidence，形成 `real-model-runtime` 候选。
5. 对发布包，另行运行 clean external package consumer proof，形成 `package-consumer-runtime` 候选。

这条梯度能让文章、样例、工具和发布门禁互相支撑，又不互相越界。
