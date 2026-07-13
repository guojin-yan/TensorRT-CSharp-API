# YoloVision 真实资产记录模板指南

## 适用读者

本文面向需要为 `samples/YoloVision` 回填真实模型、标签、输入图片和运行日志的维护者，尤其是准备把样例输出整理为 runtime proof candidate 的用户。

## 解决问题

真实资产记录不能只写“模型已下载”。它必须能说明资产来源、许可证、hash、导出命令、engine 构建命令、输入数据和输出结果。本文给出记录模板字段，并强调 build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 不能单独成为 runtime proof。

## 背景与场景

YoloVision 支持多个 YOLO 家族和任务类型。每个真实资产都可能有不同下载地址、许可证、导出工具、opset、输入 shape 和后处理约定。模板的目标是让每个模型资产都可审计、可复现、可拒绝，而不是把矩阵中的一行当作真实运行证据。

## 操作路径

1. 在 `asset` 区记录 model、labels、input image 的 sourceUrl、downloadUrl、license、SHA256 和 owner note。
2. 在 `export` 区记录导出工具版本、命令、opset、dynamic axes 和输出 ONNX hash。
3. 在 `engineBuild` 区记录 TensorRtExec 或 OnnxToEngine 命令、TensorRT/CUDA 版本、engine path 和 engine hash。
4. 在 `runtimeRun` 区记录 YoloVision 命令、stdout/stderr、输出 JSON、host metadata 和 run timestamp。
5. 在 `boundary` 区显式写出是否 validator-passed，以及哪些材料仍是非 proof。

## 代码与文件入口

- `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json`：样例运行记录模板。
- `artifacts/user-acceptance/real-model-owner-handoff.json`：owner 资产交接字段。
- `samples/YoloVision/yolo-model-matrix.json`：资产矩阵入口。
- `docs/articles/zh-cn/yolovision-real-asset-walkthrough.md`：真实资产 walkthrough。
- `docs/articles/zh-cn/release-evidence-non-substitute-guide.md`：非替代项边界。

## 图示建议

建议用表格展示资产记录的四段：download、export、build、run。每段列出必填字段、候选产物、validator 和是否可晋级。

## 边界说明

模板本身永远不是 proof。只有真实填写、hash 可核对、日志可复查且 validator 通过的记录才可能成为 runtime proof candidate。TensorRtExec report、OnnxToEngine report、YoloVision matrix、readonly diagnostics、build-only、dry-run、local feed、ProjectReference 和 direct `.nupkg` 仍不能替代真实运行记录。

## 下一步

下一轮应把模板字段与输出 JSON schema 对齐，并为每个任务类型补一份最小示例记录，明确哪些字段可以由工具自动生成，哪些字段必须 owner 人工确认。
