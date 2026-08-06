# YoloVision Owner Asset Evidence 示例

`yolovision-owner-asset-evidence.example.json` 是 owner 填写真实资产记录前的示例骨架。它展示字段位置和证据边界，但所有值都保持 `owner-to-fill`，因此它是 `example-not-proof`。

机器可读示例：

`artifacts/final-release/yolovision-owner-asset-evidence.example.json`

## 适用读者

- 准备第一次回填 YoloVision 真实资产 evidence 的 owner。
- 需要给文章补充模型、图片、labels、hash 和输出校验的作者。
- 审核 sample-run-evidence 是否具备 proof 条件的维护者。

## 解决问题

真实 YoloVision evidence 容易漏字段：模型来源、许可证、labels、输入图片、engine、命令、stdout/stderr、output JSON、SHA256 和 owner review 都必须齐全。示例文件用 YOLOv8 det 作为占位，但不包含真实模型、不包含真实 hash、不包含真实运行日志。

## 边界说明

以下内容不是 runtime proof：

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

示例文件不能替代 `package-consumer-runtime` 或 `post-publish verification`。只有 owner 回填真实文件、真实日志、真实 SHA256，并通过 strict validator 后，才可以进入 proof review。

## 可复制命令

真实运行时 owner 需要提供类似命令，但示例不会执行：

```powershell
dotnet run --project applications\TensorRtExec -- --onnx <owner-model.onnx> --saveEngine <owner-model.engine> --buildOnly --report <owner-build-report.json>

dotnet run --project applications\YoloVision -- --model <owner-model.onnx> --input-data <owner-preprocessed-fp32.bin> --input-shape 1x3x640x640 --family v8 --task det --labels <owner-labels> --output <owner-output.json>
```

`--report` 是 `--exportReport` 的兼容别名，工具会在 normalized command 和 report 中归一化为 `--exportReport`。该报告仍是 build-only evidence，不是 runtime proof。

输出必须包含 `YoloVision Passed=True` 和 `OutputJson=<owner-output.json>`，并提交 output JSON 与日志 SHA256。若 owner 另行使用 `TensorRtExec` 生成 engine/build report，该 report 仍是 build-only evidence，不能替代 `YoloVision` run log 与 output JSON。

## 下一步

1. owner 将 `owner-to-fill` 全部替换为真实来源、路径、hash 和 review。
2. 保存 stdout/stderr、engine、output JSON 和 SHA256。
3. 用 sample-run-evidence validator 验证记录。
4. 仍需独立完成 package-consumer-runtime 和 post-publish verification。
