# 项目对外介绍与发布边界素材包

这篇文章面向博客、公众号、README 宣发和 release issue 摘要。它把 TensorRtSharp4.0 可以对外讲清楚的能力、仍需 owner action 的边界和不能越级宣传的证据类型放在一起，避免“项目已经很完整”被误写成“所有 runtime proof 已完成”。

## 一句话介绍

TensorRtSharp4.0 是一个面向 .NET 的 TensorRT / CUDA 互操作项目，提供 native bridge、C# wrapper、跨 TensorRT 8/10/11 的版本防护、样例、smoke runner、TensorRtExec 工具、runtime package 策略和 release evidence 自动化。

更准确地说，它不是简单的 P/Invoke 集合，而是一套围绕 ABI 边界、ownership、deferred safety、package consumer proof 和真实模型 evidence 建立的工程化封装。

## 可以对外强调的内容

| 能力 | 可以怎么说 | 不能怎么说 |
| --- | --- | --- |
| 接口覆盖 | manifest/source 匹配和 deferred 边界已经可审计 | 100% manifest/source 匹配等于 100% runtime 可用 |
| C# wrapper | 高层 wrapper 覆盖核心 builder/runtime/engine/parser 场景 | 所有 TensorRT callback 都已真实触发验证 |
| Plugin Inventory | plugin registry / creator metadata 只读 API 已有安全封装 | 可以暴露裸 `IntPtr` plugin creator 或创建 plugin instance |
| TensorRtExec | CLI/WinForms 能生成 ONNX build/precheck/report/sidecar | build-only 报告等于 release proof |
| YoloVision | 覆盖 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 与 det/cls/seg/obb/pose/sem 的配置底座 | support matrix 等于真实模型 proof 已通过 |
| runtime package | managed/runtime/split package layout 已建模并可审计 | 本机 `blocked-by-cuda-driver` 等于 smoke passed |

## 推荐宣传结构

### 1. 为什么不是 plain P/Invoke

TensorRT 的 C++ API 和 ABI 边界并不适合直接散落成大量裸 P/Invoke。项目通过 native bridge 把异常、字符串、数组、borrowed pointer、version guard 和 ownership 统一收口，再由 C# wrapper 提供更稳定的使用面。

### 2. 为什么保留 deferred

deferred 不是偷懒，也不是未登记接口；它是对高风险边界的显式刹车。典型包括 callback trampoline、allocator、debug listener、borrowed pointer、plugin instance create/clone/enqueue。没有真实 callback runtime proof 或 ownership 设计前，继续 deferred 比暴露危险 API 更接近可发布质量。

### 3. 用户今天能试什么

用户可以从这些入口开始：

- `applications/OnnxToEngine`：最小 ONNX round-trip。
- `applications/TensorRtExec`：外部 ONNX build-only、dry-run、report 和 WinForms 入口。
- `samples/ComputerVision/01.Classification`：自备分类模型、labels 和输入图像。
- `applications/YoloVision`：自备 YOLO-family ONNX、metadata、labels 和输入图像。
- `smoke/*Runner`：本机或兼容主机上的 runtime smoke。

这些路径必须保留证据边界：build-only 是构建证据，sample-run-evidence 是样例证据，`package-consumer-runtime` belongs to release proof records。

## 当前发布边界

当前可以说：

- release candidate final evidence freeze 已生成。
- compatible host proof execution pack 已生成。
- release close gap dashboard 已聚合剩余 blocker。
- stale claim audit 已用于防止越级宣传。
- 真实 close blocker 已明确。

当前不能说：

- 当前不能说已经进入公开发布就绪状态。
- post-publish verification 已完成。
- `package-consumer-runtime` 已经通过。
- `real-model-runtime` 已经通过。
- local feed 或 ProjectReference consumer 可以替代 clean external consumer。
- `blocked-by-cuda-driver` 是通过。

## 面向用户的安装说法

安装和 runtime package 选择应该强调版本矩阵：

- .NET managed package 只负责托管 API。
- runtime package 负责具体 RID、CUDA、TensorRT、cuDNN native assets。
- `win-x64-trt11.0-cuda13.2-cudnn9.22` 这类 key 必须和主机 driver/runtime 能力匹配。
- CUDA error 35 是 driver/runtime compatibility blocker，需要换兼容主机或升级驱动，不应写成 API 缺失。

## 面向 release owner 的说法

owner 需要补齐五类真实证据：

1. owner authorization
2. package-consumer-runtime
3. linux-runner-proof
4. real-model-runtime
5. post-publish verification

每一类都必须有真实日志、真实 hash、真实主机 metadata 和 validator 输出。helper、template、draft、runbook、collection package、input package、sidecar-only 和 parse-only 都只能辅助执行。

## 推荐摘要

可以在博客或 README 中使用以下摘要：

> TensorRtSharp4.0 已进入发布候选最终审计阶段。项目提供 TensorRT/CUDA native bridge、跨版本 C# wrapper、TensorRtExec ONNX 转换工具、YoloVision 样例体系和 release evidence 自动化。当前自动化材料已经能清楚区分 build-only、sample-run-evidence、real-model-runtime、package-consumer-runtime 和 post-publish verification；剩余发布 close blocker 需要 release owner 在兼容 CUDA/TensorRT 主机、真实模型资产、Linux runner 和真实包渠道上补齐。

这段摘要保留了项目价值，也没有把未完成的 proof 写成完成。
