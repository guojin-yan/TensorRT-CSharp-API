# eng 工程脚本目录

`eng` 不是面向最终用户的命令集合。它同时承载构建编排、资产获取、CI 验证、证据导出、Owner 回填模板和发布前只读门禁，因此文件数量很大。不能因为脚本存在，就认为它是日常支持入口，也不能直接批量删除或移动，否则会破坏 workflow、测试、文章和脚本之间的调用关系。

2026-08-03 的静态盘点结果：

| 类型 | 数量 | 定位 |
| --- | ---: | --- |
| `Test-*` | 351 | CI、合同、证据和 fail-closed 验证器 |
| `Export-*` | 367 | 生成机器可读报告、候选包、Owner 输入模板；多数不执行发布 |
| `Acquire-*` | 10 | 固定来源和 SHA 的模型/资产获取入口 |
| `Sync-*` | 2 | 本地资产同步和校验入口 |
| `Invoke-*` | 19 | 组合编排或本机 smoke 入口 |
| `Import-*` | 25 | 导入 Owner 或外部运行证据 |
| `New-*` / `Collect-*` | 5 | 脚手架、受控变异和收集器 |
| 其他 | 23 | 公共函数、归档工具和人工入口等 |

802 个 PowerShell 脚本中，779 个能在仓库源码、workflow、测试或文档中找到脚本名引用。其余 23 个包含公共函数文件、人工入口和待审计候选，不能仅凭“没有字面引用”判定无用。目录另有 9 个 Python 辅助脚本。

## 支持入口

最终用户或维护者应优先从以下入口开始，而不是在目录中随机选择脚本：

| 目标 | 入口 | 说明 |
| --- | --- | --- |
| 本地 release 质量编排 | `Invoke-LocalReleaseBundle.ps1` | 构建、测试、DocFX 和候选包检查；不等于授权发布 |
| runtime 包就绪检查 | `Test-RuntimePackageReadiness.ps1` | 验证 managed + bridge-only 边界和 runtime matrix |
| 演示 ONNX 暂存同步 | `Sync-DemoOnnxModels.ps1` | 把固定 ONNX 同步到外层 `models`；不上传模型 |
| YOLOv8n Detection 资产 | `Acquire-YoloV8DetectionOfficialAssets.ps1` | 下载并校验固定权重、labels、许可证与图片 |
| YOLOv8n Detection 本地三包验证 | `Test-YoloVisionDetectionLocalPackageConsumer.ps1` | 隔离三个本地包，执行真实 TensorRT 正例、独立对照和负例 |
| 通用 YoloVision 三包验证器 | `Test-YoloVisionLocalPackageConsumer.ps1` | 被各任务专用入口调用；不建议手工拼接参数 |
| 文章完整性门禁 | `Test-TechnicalArticleCompleteness.ps1` | 检查发布目录中的真实结果、配图、模型获取/转换和边界 |

每个支持入口必须同时具备：明确文档、失败即非零退出、ProjectQuality 测试或 CI 调用、禁止隐式发布、重资产不进入 Git。没有满足这些条件的脚本一律按内部工程脚本处理。

## 内部脚本

- `Test-*` 通常是可执行验证器，但很多只服务特定证据 schema，不是用户功能。
- `Export-*` 通常只生成 JSON/Markdown/模板。名字含 `Export` 不表示导出模型，也不表示上传或发布。
- `Owner*`、`*Proof*`、`*Candidate*`、`*Readiness*` 多数是发布治理或真实证据回填工具，缺少 Owner 输入时会保持 blocked/non-proof。
- `*.Common.ps1` 是 dot-source 公共函数，不能独立运行，也不能按“未引用”轻率删除。

## 后续整理规则

第一版发布前不做 800 个脚本的全量路径迁移。先以本 README 收窄公开入口，再逐批完成引用图审计：

1. 明确无人引用、没有人工入口且无历史兼容责任的脚本，先标记 deprecated。
2. 连续两个 release 周期无人使用且测试确认无依赖后再删除。
3. 可复用公共函数迁入模块时，同时修改 workflow、测试和文档，禁止只移动文件。
4. 新增脚本必须进入支持入口、内部工具或临时候选三类之一，并附测试和删除条件。

本目录中的脚本不会因为“质量门通过”而自动获得 tag、Release、NuGet push、GitHub Packages push、模型上传或文章发布权限。
