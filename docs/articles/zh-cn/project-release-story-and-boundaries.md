# TensorRtSharp4.0 项目能力与发布边界

TensorRtSharp4.0 的价值在于把 TensorRT/CUDA 能力带到 .NET 生态：底层有 native bridge 和 manifests，上层有 C# wrapper、samples、applications、runtime packages 和 release evidence 工具链。本文面向博客、公众号和项目介绍，但保持一个原则：能宣传能力，也要诚实说明 proof 边界。

## 项目已经具备的能力

- TensorRT/CUDA 接口 inventory 和 deferred boundary 管理。
- C# 高层 wrapper 与 interop 分层。
- samples：OnnxToEngine、Classification、YoloVision 等。
- applications：TensorRtExec CLI + WinForms。
- runtime package matrix。
- release evidence bundle、close preflight、stale claim audit。
- owner-release-execution-package 与 `oneScreenReleaseHoldChecklist` 一屏 Release Hold 清单。
- package consumer、post publish verification、real model evidence 的 playbook。

## 为什么不写“全部完成”

manifest/source 匹配不等于 100% 可用。真实完成度要看：

- 非 deferred 实现。
- 高层 C# wrapper。
- smoke / package-consumer-runtime。
- real-model-runtime。
- post publish verification。
- owner authorization。

这也是项目持续保留 `blocked-by-cuda-driver`、`build-only`、`parse-only`、`sidecar-only` 等边界词的原因。

## TensorRtExec 的定位

TensorRtExec 让用户用 CLI/WinForms 完成 ONNX 到 engine 的 build/report 工作流。它支持 trtexec-like 参数接入，但高级参数仍需要看 `TrtexecAlignmentStatus=parse-only` 和 `OptionImplementationStatus`。这不是弱点，而是防止把 parser/report 覆盖误写成真实 TensorRT 行为。

## YoloVision 的定位

YoloVision 是 YOLO-family 多任务样例：v5/v6/v7/v8/v9/v10/v11/v26、custom、det/cls/seg/obb/pose/sem。它提供 support matrix 和 managed postprocess 底座。真实模型 proof 仍需要模型、labels、input、license、hash、runner log 和 sample-run-evidence。

## 发布边界

- package-consumer-runtime 只能来自干净 consumer。
- post publish verification 只能来自真实渠道发布后的验证。
- real-model-runtime 只能来自真实模型样例证据。
- owner action required 不是 proof。
- ProjectReference、local feed、template、draft、runbook、sidecar-only 不能关闭 release issue。
- `owner-release-execution-package` 和一屏 Release Hold 清单是 owner guidance，不是 proof；真实记录缺失时 `canCloseReleaseIssue=false` 必须保持不变。

## 面向用户的诚实承诺

这个项目不是靠删 deferred 记录制造完成度，而是逐批提升高价值、低 ownership 风险的接口，并把每一批提升落实到 wrapper、文档、smoke 和质量门禁。对于 callback、allocator、borrowed pointer 等高风险区域，项目宁愿慢一点，也不把不清楚 ownership 的裸指针交给用户。

## 下一步

下一步重点是 owner 在兼容主机和真实包渠道中补齐 proof，同时继续扩写案例教程。等真实 evidence 完整后，再进入最终发布总检，而不是提前写成正式发布完成。
