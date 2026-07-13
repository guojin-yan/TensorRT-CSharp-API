# YoloVision Owner 真实资产 Evidence 指南

`yolovision-owner-asset-evidence.template.json` 是 `samples/YoloVision` 的真实资产输入模板。它帮助 owner 收集模型、图片、labels、命令、日志、输出 JSON 和 SHA256，但模板本身不是 runtime proof。

机器可读模板：

`artifacts/final-release/yolovision-owner-asset-evidence.template.json`

## 适用读者

- 准备用 YoloVision 做真实模型样例的维护者。
- 需要为 YOLO family 文章补充真实资产 provenance 的作者。
- release owner 和 proof reviewer。

## 解决问题

YoloVision 已经统一覆盖 YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLOv11、YOLOv26 和 custom 模型族，以及 det、cls、seg、obb、pose、sem 任务。但矩阵、README、candidate pack 只能说明样例表面和计划，不能证明真实模型已经在目标环境中运行。

这个模板要求 owner 明确填写：

- 模型来源、许可证和 SHA256。
- labels 来源和 SHA256。
- 输入图片来源、许可证和 SHA256。
- TensorRT engine 路径和 SHA256。
- `TensorRtExec` build command。
- `YoloVision` run command。
- stdout/stderr log path。
- output JSON path 和 SHA256。
- owner review status。

## Evidence 字段

每条真实记录至少要能证明：

- 使用的是 owner 选择的真实模型，不是示例占位。
- 输入图片和 labels 有来源、许可证和 hash。
- engine 由明确命令生成，并记录 hash。
- Owner 真实 YoloVision run log 中可出现 `YoloVision Passed=True`；它只是待校验日志信号，不是 release proof，仍需 asset/hash/host metadata/validator 一致。
- 输出 JSON 可以通过 schema 或任务级 validator。
- owner review 明确为 approved。

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

YoloVision owner asset evidence 也不能替代 `package-consumer-runtime` 或 `post-publish verification`。即使真实模型跑通，发布关闭仍需要公开包 clean consumer proof 和 post-publish clean consumer proof。

## 可复制命令

模板中的命令是 owner 填写占位，不应直接作为 proof：

```powershell
dotnet run --project applications\TensorRtExec -- --onnx <owner-model.onnx> --saveEngine <owner-model.engine> --buildOnly --report <owner-build-report.json>

dotnet run --project samples\YoloVision -- --model <owner-model.onnx> --input-data <owner-preprocessed-fp32.bin> --input-shape <owner-input-shape> --family <custom|v5|v6|v7|v8|v9|v10|v11|v26> --task <det|cls|seg|obb|pose|sem> --labels <owner-labels> --output <owner-output.json>
```

`--report` 是 `--exportReport` 的兼容别名，工具会在 normalized command 和 report 中归一化为 `--exportReport`。该报告仍是 build-only evidence，不是 runtime proof。

真实 proof 需要同时提交命令输出、日志文件、输出 JSON 和 SHA256。当前 `samples/YoloVision` 使用 ONNX 模型和预处理后的 tensor 输入；`TensorRtExec` 生成的 engine/build report 应作为单独的 build-only 证据记录，不能替代样例运行输出。

## 下一步

1. owner 选定一个低风险模型族和任务，例如 YOLOv8 det 或 cls。
2. 回填 template 中所有 `owner-required` 字段。
3. 运行样例并保存 stdout/stderr、output JSON 和 hash。
4. 用 sample-run-evidence validator 验证记录。
5. 将通过验证的结果再纳入 release owner close 审核，不要直接把模板晋级为 proof。
