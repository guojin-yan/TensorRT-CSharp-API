# OnnxToEngine 与 trtexec-like 转换：能力、报告和 Proof 边界

`applications/OnnxToEngine` 是最小 ONNX 到 TensorRT engine 样例，`applications/TensorRtExec` 是面向用户的 trtexec-like 工具。两者共同帮助用户理解转换链路，但 `OnnxToEngine report` 和 shape profile matrix 不能替代 runtime proof。

## 适用读者

- 从 ONNX 迁移到 TensorRT engine 的模型部署工程师。
- 熟悉官方 `trtexec`，希望在 .NET 项目中复用类似参数的人。
- 需要区分 build/report evidence 与 release proof 的项目维护者。

## 解决问题

OnnxToEngine 适合验证最小链路：

- `applications/OnnxToEngine/Program.cs`
- `applications/OnnxToEngine/trtexec-parity-matrix.json`
- `applications/OnnxToEngine/trtexec-parity-matrix.md`
- `tests/JYPPX.ProjectQuality.Tests/OnnxToEngineTrtexecLikeTests.cs`

它可以覆盖 ONNX input、engine output、min/opt/max shape profile、FP16/INT8 boundary、workspace / memory pool intent、timing cache intent、verbose diagnostics 和 report output。

## report 与 proof 的分层

OnnxToEngine 可以证明“转换路径可以被描述和验证”，但不能自动证明：

- 外部模型输出语义正确。
- YOLO det/cls/seg/obb/pose/sem 后处理正确。
- clean consumer 从公开包恢复并运行成功。
- post-publish verification 已经从发布渠道完成。

所以 `OnnxToEngine report`、`build-only` engine artifact、`dry-run` command、shape profile matrix 都必须保持 non-proof。

## 边界说明

以下内容必须保持非 proof：

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

即使 OnnxToEngine 完成最小 round-trip，它也不能替代 `package-consumer-runtime` 或 `post-publish verification`。这些 release proof 必须由 owner 在真实环境中回填。

## 可复制命令

最小样例命令：

```powershell
dotnet run --project applications\OnnxToEngine -- --tensor-rt-line 10 --batch 2
```

trtexec-like 外部模型预检应转到 TensorRtExec：

```powershell
dotnet run --project applications\TensorRtExec -- `
  --onnx <owner-model.onnx> `
  --saveEngine <owner-model.engine> `
  --minShapes input:1x3x640x640 `
  --optShapes input:1x3x640x640 `
  --maxShapes input:4x3x640x640 `
  --previewOnly `
  --exportReport <owner-precheck-report.json>
```

这些命令可以生成采用路径和交接材料，但仍不是 runtime proof。

## 截图与图示建议

- OnnxToEngine 最小 round-trip 流程图。
- trtexec-like 参数到 TensorRtExec options 的映射表。
- report evidence 与 real proof ladder 的分层图。

## 下一步

1. 保持 OnnxToEngine 作为最小转换样例。
2. 将外部模型 build/report 工作流交给 TensorRtExec。
3. 将模型语义 proof 交给 YoloVision、Classification 或用户自己的 sample runner。
4. 用 `artifacts/final-release/release-proof-owner-input-dashboard.json` 追踪 owner proof，而不是把 report 晋级为 proof。
