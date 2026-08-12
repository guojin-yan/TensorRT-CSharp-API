# TensorRT CSharp API v4.0 项目背景与其他主题

本模块收录不属于单一案例、完整应用、API 使用、安装或源码编译的公开文章，包括项目总览、架构背景和模型资产治理。安装与源码编译已经拆分为独立模块，后续可以分别扩展平台、构建链路和排错文章；内部发布工程记录不进入本模块。

项目主页与源码：`https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0`

核心 NuGet：`https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0`

## 1. 当前文章

| ID | 系列 | 标题 | 状态 |
| --- | --- | --- | --- |
| `MSC-001` | 项目总览 | [项目是什么](overview/msc-001-what-is-tensorrtsharp4.md) | `ready` |
| `MSC-002` | 架构背景 | [为什么不是简单 P/Invoke](architecture/msc-002-beyond-pinvoke.md) | `ready` |
| `MSC-009` | 模型资产 | [模型获取、ONNX 转换与 SHA256 管理](models/msc-009-model-acquisition-and-onnx-governance.md) | `ready` |

## 2. 新增规则

1. 使用 `MSC-###` 作为稳定 ID，并按主题建立子目录；安装和源码编译不再混入本模块。
2. 项目总览仍需给出项目主页、包入口、源码和本文程序的明文 URL。
3. 模型文章必须记录来源、许可证、版本和 SHA256，不能把临时下载地址当作长期资产。
4. 已发布文章遵守 `article-index.json` 的冻结与 `supersedes` 规则。
