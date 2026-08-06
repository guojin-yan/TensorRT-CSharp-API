# YoloVision 样例总览

`applications/YoloVision` 是 TensorRtSharp4.0 面向 YOLO-family 模型的统一视觉样例入口。它替代旧的单一检测命名思路，把 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 与 `det`、`cls`、`seg`、`obb`、`pose`、`sem` 放到同一个 family/task/profile/postprocess 框架里。

本文是用户进入 YoloVision 的第一篇文章。它说明样例能做什么、资产怎么准备、哪些命令可以离线运行，以及哪些输出只能算 sample evidence，不能升级成 public package proof 或 post-publish proof。

## 目标读者

- 已经有 YOLO-family ONNX 模型，准备在 .NET 中完成 TensorRT engine 构建与样例推理的用户。
- 需要同时覆盖 detection、classification、segmentation、OBB、pose、semantic segmentation 的视觉模型部署工程师。
- 需要把模型资产、build report、sample-run evidence 和 release proof record 分开管理的发布负责人。
- 正在维护 `applications/YoloVision`，需要避免旧检测-only 命名和 proof 越级声明的贡献者。

## 适用范围

- 你已有 YOLO-family ONNX，希望用 TensorRtSharp4.0 生成 engine 并运行托管后处理。
- 你正在准备模型、labels、输入图片、预处理 tensor、license 和 hash。
- 你需要理解 `--family`、`--task`、`--layout`、`--has-objectness`、`--nms-mode` 等参数。
- 你需要把 sample-run-evidence 与 release proof record 分开管理。

## 快速入口

先查看离线能力矩阵：

```powershell
dotnet run --project .\applications\YoloVision -- --list-capabilities
```

该命令不需要 CUDA、TensorRT、ONNX、labels 或图片。它只说明当前样例层支持哪些 family/task/postprocess 组合，不证明真实模型已经运行。

准备 owner 资产时，可以先运行离线 preflight：

```powershell
dotnet run --project .\applications\YoloVision -- `
  --preflight `
  --family v8 `
  --task seg `
  --model .\models\yolovision\yolov8n-seg.onnx `
  --labels .\models\yolovision\coco.names `
  --input-data .\models\yolovision\yolov8n-seg-fp32.bin `
  --input-shape 1x3x640x640 `
  --output-role-map boxes:det,proto:mask-prototypes `
  --mask-coefficient-count 32 `
  --preflight-report .\artifacts\yolovision\yolov8n-seg-preflight.json
```

该报告是 `yolovision-preflight.v1` 的 `precheck` 证据：会记录 profile、资产存在性和 SHA256、输出 metadata 与规范化命令 hash，但不会调用 TensorRT、ONNX parser、engine build 或 inference。`owner-action-required` 不是失败通过；只有真实模型、真实输入、运行日志、hash、输出 JSON 和 owner review 都回填后，才可以进入 `real-model-runtime` 候选。

## 可复制命令

真实模型的典型命令如下：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"

dotnet run --project .\applications\YoloVision -- `
  --model .\models\yolovision\model.onnx `
  --labels .\models\yolovision\labels.txt `
  --input-data .\models\yolovision\input-fp32.bin `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 11 `
  --family v8 `
  --task det `
  --layout auto `
  --has-objectness auto `
  --nms-mode class-aware `
  --confidence 0.25 `
  --iou-threshold 0.45
```

如果要先生成 build-only 报告，使用 `applications/TensorRtExec` 或 `applications/OnnxToEngine`。build-only 只能证明 ONNX 构建路径，不证明检测框、分类标签、mask、pose keypoint 或 OBB angle 正确。

## 目录关系

| 路径 | 用途 | 证据边界 |
| --- | --- | --- |
| `applications/YoloVision` | YOLO-family 托管样例和后处理 | sample-level evidence |
| `samples/assets/yolovision-assets.template.json` | 资产清单模板 | template-only |
| `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json` | 真实运行回填模板 | owner input required |
| `applications/TensorRtExec` | ONNX build/report 工具 | build-only/precheck |
| `docs/articles/zh-cn/yolo-family-profile-and-postprocess-guide.md` | family/task/postprocess 深入说明 | documentation-only |

## 支持任务

| Task | 代表输出 | 关键 metadata |
| --- | --- | --- |
| `det` | box、score、class、NMS | box layout、class count、objectness、NMS mode |
| `cls` | Top-K label score | label count、score mode |
| `seg` | detection + mask coefficient + prototype | prototype shape、mask threshold |
| `obb` | rotated box + angle | angle unit、angle channel |
| `pose` | detection + keypoints | keypoint count、visibility layout |
| `sem` | dense class map | logits/argmax、spatial layout |

## 最小资产清单

真实模型运行至少需要：

- ONNX 模型路径与 model SHA256；
- labels 文件路径与 labels SHA256；
- 输入图片或预处理 tensor 与 SHA256；
- 输入 shape；
- family/task/profile；
- 输出 tensor 名称、layout、class count；
- license 说明；
- TensorRT/CUDA/runtime 环境记录；
- sample stdout/stderr 摘要和日志 SHA256。

## 边界说明

- `--list-capabilities` 是 capability matrix，不是 runtime proof。
- synthetic input 只能证明管线可执行，不证明真实模型质量。
- build-only report 不证明推理输出正确。
- `sample-run-evidence` 只能晋级 sample-level `real-model-runtime`，不能替代 public package proof。
- `package-consumer-runtime` 属于 release proof record，不属于 YoloVision 样例自身。
- `blocked-by-cuda-driver` 是环境阻塞，不是失败通过。
- Boundary keywords: not public package proof, not post-publish proof, not package push, not release close approval.

YoloVision 文章、模板、asset candidates 和 `--list-capabilities` 仍然是 not runtime proof，直到 owner 回填真实模型、labels、输入资产、run log、SHA256、stdout/stderr summary，并通过 sample-run evidence validator。

## 截图与图示建议

- family/task 矩阵图：YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 对 det/cls/seg/obb/pose/sem。
- 一张真实模型运行日志截图，至少包含 `YoloVision Passed=True`。
- 资产证据流图：model/labels/input/preprocessed tensor -> sample-run evidence -> real-model-runtime。

## 下一步

- 先填写 `samples/assets/yolovision-assets.template.json`。
- 再填写 `artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json`。
- 如果需要 build-only 报告，先用 `applications/TensorRtExec` 生成 report，再把 report path 写入 evidence sidecar。
- 维护者新增 family/task 时，必须同步更新 `applications/YoloVision/yolo-model-matrix.json`、README、托管测试和 sample-run evidence requirement。

## 第二批正文门禁

### 适用读者

本文适合准备用 `applications/YoloVision` 承载 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 的用户，也适合准备写公众号或博客案例教程的维护者。

### 解决问题

单一检测样例无法承载 YOLO 系列越来越多的任务形态。YoloVision 要解决的是统一样例入口、统一资产记录、统一输出 evidence sidecar，并让每个 family/task 都能说明模型来源、许可证、输入输出 metadata、后处理边界和 proof 状态。

### 核心思路

核心思路是把“支持范围”和“真实 proof”拆开。支持范围用 family/task matrix 描述，真实 proof 用每个模型资产的 hash、labels、输入图片、命令、日志和 validator 描述。

### 操作路径

先运行 capability matrix，再为目标模型准备 ONNX、labels、输入图片和许可证记录；用 `applications/OnnxToEngine` 或 `applications/TensorRtExec` 生成 build-only report；最后用 YoloVision runner 完成真实推理、后处理和输出摘要。

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。没有真实模型、run log、SHA256、stdout/stderr summary 和 validator 时，YoloVision 只能标记为 planned/documented。

### 下一步

下一步按 family/task 拆出多篇真实模型教程，优先补模型来源、许可证、labels、输入图片和 validator。
