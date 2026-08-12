# TensorRT CSharp API v4.0 系列案例

本模块把 `samples/` 中小而专注的可运行程序整理成连续学习路线。文章必须明确帮助命令、真实运行所需的 GPU/模型/厂商运行库，以及“能显示帮助”和“完成真实推理”之间的边界。

## 1. 当前文章

| ID | 标题 | 源码版本 | 状态 |
| --- | --- | --- | --- |
| `SMP-001` | [TensorRT CSharp API v4.0 系列案例总览与学习路线](smp-001-sample-series-overview.md) | `4.0.0` | `ready` |
| `SMP-002` | [推理输入、显存绑定与 GPU 输出读回](smp-002-inference-bindings.md) | `4.0.0` | `ready` |
| `SMP-003` | [Dynamic Shape 与动态 Batch 推理](smp-003-dynamic-shapes.md) | `4.0.0` | `ready` |
| `SMP-004` | [从 ONNX 到 TensorRT Engine](smp-004-onnx-build-and-run.md) | `4.0.0` | `ready` |
| `SMP-005` | [Refitted Plan 权重替换、持久化与重新加载](smp-005-refitted-plan.md) | `4.0.0` | `ready` |
| `SMP-006` | [在 C# 中动态编译并运行 CUDA Kernel](smp-006-cuda-runtime-compilation.md) | `4.0.0` | `ready` |
| `SMP-007` | [CUDA 多流与 Event 同步](smp-007-cuda-multistream.md) | `4.0.0` | `ready` |
| `SMP-008` | [TensorRT Callback 生命周期](smp-008-callback-lifecycle.md) | `4.0.0` | `ready` |
| `SMP-009` | [C#、TensorRT 与 ResNet18 图像分类](smp-009-resnet18-classification.md) | `4.0.0` | `ready` |

历史教程仍保留在 `docs/articles/zh-cn` 根目录，供旧链接兼容。总览与 8 个案例专题现在都已迁移为独立 canonical 长文；新内容以本模块和 [`article-index.json`](../article-index.json) 为公开入口。

## 2. 新增规则

1. 使用 `SMP-###` 作为稳定 ID；总览先建立学习顺序，后续文章一篇只讲一个主要案例。
2. 每篇标题后必须有“前言”，先介绍 TensorRT CSharp API v4.0、项目 GitHub、核心 NuGet、Runtime Bridge，并链接本文对应的 GitHub 案例目录与入口代码。
3. 所有命令从仓库根目录执行，并使用相对路径或明确占位符。
4. 模型案例必须记录上游 URL、固定版本、许可证、转换命令、输入输出契约和 SHA256。
5. 图像任务必须展示真实程序页面或终端结果，以及叠加任务输出的结果图；非视觉案例按对应运行结果要求配图。
6. 不把 `--help`、预检查、构建成功或合成输入结果描述为真实模型推理成功。
