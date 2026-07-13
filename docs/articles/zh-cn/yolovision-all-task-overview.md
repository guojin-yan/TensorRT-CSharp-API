# YoloVision 全任务系列总览

YoloVision 是项目面向 YOLO 系列的统一样例入口。它不再只面向检测，而是把 YOLO v5/v6/v7/v8/v9/v10/v11/v26 以及 custom family 的 det、cls、seg、obb、pose、sem 放到同一个 family/task/profile/postprocess 框架里。

## 适用场景

- 想把自己的 YOLO-family ONNX 接入 TensorRtSharp4.0。
- 想确认输出 layout、objectness、NMS、mask、keypoint、angle metadata 该如何描述。
- 想为文章、样例或 release evidence 准备真实模型资产。

## 支持矩阵怎么读

YoloVision 的 capability matrix 是 support matrix，不是 `real-model-runtime` proof。它说明样例层已经有 family/task/profile 的配置底座；真实模型是否通过，还需要 owner 提供 ONNX、labels、input image、license、hash、TensorRtExec build-only report 和 sample-run-evidence。

| Family | 典型任务 | 重点关注 |
| --- | --- | --- |
| v5/v6/v7 | det、seg、pose | output layout、objectness、NMS |
| v8/v9/v10/v11 | det、cls、seg、obb、pose | 多输出 metadata、class count、angle/keypoint |
| v26/custom | det、cls、seg、obb、pose、sem | 明确 profile，不猜 layout |

## 通用接入流程

1. 选择 family 和 task。
2. 准备 ONNX、labels、input image，并记录 license。
3. 计算 model/labels/input SHA256。
4. 用 TensorRtExec 生成 build-only report。
5. 在 asset manifest 中写明 input/output metadata。
6. 运行 YoloVision sample。
7. 用 sample-run-evidence record 回填真实日志。
8. 用 validator 检查是否可晋级 `real-model-runtime`。

## 任务差异

| Task | 输出解释 | 必填 metadata |
| --- | --- | --- |
| det | box + class score + NMS | box layout、class count、objectness、NMS mode |
| cls | label score 排序 | label count、Top-K、logit/softmax |
| seg | box + mask coefficient + prototype | tensor role、mask shape、resize policy |
| obb | oriented box + angle | angle unit、box layout、class count |
| pose | detection + keypoints | keypoint count、visibility、coordinate layout |
| sem | dense class map | class count、spatial layout、argmax rule |

## 证据边界

所有文章和 README 都必须保留：

- `build-only` 不证明推理输出正确。
- `parse-only` 和 `TrtexecAlignmentStatus=parse-only` 不证明高级 TensorRT 行为完整执行。
- `sidecar-only` 只是资产/报告桥接。
- `package-consumer-runtime` 属于 release proof record，不属于样例 runner。
- `real-model-runtime` 只属于真实模型、真实输入、真实日志和 sample evidence。
- `blocked-by-cuda-driver` 是环境阻塞，需要 owner action。

## 推荐文章拆分

YoloVision 系列适合拆成多篇：检测、分割、姿态、OBB、分类/语义分割、外部模型 evidence 回填。每篇都应该带命令、metadata 表、proof 边界和 owner action，而不是只展示 API 名称。
