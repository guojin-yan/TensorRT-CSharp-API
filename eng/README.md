# eng 工程脚本目录

`eng` 不是面向最终用户的命令集合。它同时承载构建编排、资产获取、CI 验证、证据导出、Owner 回填模板和发布前只读门禁，因此文件数量很大。不能因为脚本存在，就认为它是日常支持入口，也不能直接批量删除或移动，否则会破坏 workflow、测试、文章和脚本之间的调用关系。

2026-08-05 四轮引用图审计与 callback owner 独立包验证补充后，目录保留 802 个 PowerShell 脚本、9 个 Python 辅助脚本和 1 个 Shell 脚本。已删除 15 个确认重复、失效或与当前交付边界冲突的入口；同时移除了 39 个退役 full-runtime/vendor 包项目。这里记录的是保留下来的工程资产，不是对外命令数量。

| PowerShell 类型 | 数量 | 定位 |
| --- | ---: | --- |
| `Test-*` | 358 | CI、合同、证据和 fail-closed 验证器 |
| `Export-*` | 360 | 生成机器可读报告、候选包和内部审计材料；多数不执行发布 |
| `Acquire-*` | 10 | 固定来源和 SHA 的模型/资产获取入口 |
| `Sync-*` | 2 | 本地资产同步和校验入口 |
| `Invoke-*` | 13 | 组合编排或本机 smoke 入口 |
| `Import-*` | 25 | 导入 Owner 或外部运行证据 |
| `New-*` / `Collect-*` | 5 | 脚手架、源码归档和收集器 |
| 其他 | 29 | 公共函数、验证、签名、归档工具和人工入口等 |

剩余脚本大多能在源码、workflow、测试或文档中找到调用关系。少数没有字面引用的是本机 CUDA/TensorRT smoke、原生 ABI 诊断和 Windows 开发证书入口，属于明确保留的人工工具；公共函数也可能通过 dot-source 间接加载。因此不能仅凭“没有字面引用”判定无用。9 个 Python 辅助脚本用于模型转换、独立 reference 和受控变异，不执行模型上传。

## 支持入口

最终用户或维护者应优先从以下入口开始，而不是在目录中随机选择脚本：

| 目标 | 入口 | 说明 |
| --- | --- | --- |
| 本地 release 质量编排 | `Invoke-LocalReleaseBundle.ps1` | 构建、测试、DocFX 和候选包检查；不等于授权发布 |
| Windows bridge 包矩阵 | `Invoke-WindowsBridgePackageMatrix.ps1` | 构建并隔离验证六组 bridge-only 本地候选；不执行上传、tag 或 Release |
| runtime 包就绪检查 | `Test-RuntimePackageReadiness.ps1` | 验证 managed + bridge-only 边界和 runtime matrix |
| 演示 ONNX 暂存同步 | `Sync-DemoOnnxModels.ps1` | 把固定 ONNX 同步到外层 `models`；不上传模型 |
| YOLOv8n Detection 资产 | `Acquire-YoloV8DetectionOfficialAssets.ps1` | 下载并校验固定权重、labels、许可证与图片 |
| YOLOv8n Detection 本地三包验证 | `Test-YoloVisionDetectionLocalPackageConsumer.ps1` | 隔离三个本地包，执行真实 TensorRT 正例、独立对照和负例 |
| GPU allocator 本地两包验证 | `Test-GpuAllocatorLocalPackageConsumer.ps1` | 隔离 managed 与 bridge-only 包，执行真实回调、零泄漏、拒绝和异常负例；不发布包 |
| OutputAllocator 本地两包验证 | `Test-OutputAllocatorLocalPackageConsumer.ps1` | 隔离 managed 与 bridge-only 包，执行动态输出分配、释放、detach 和拒绝负例；不发布包 |
| DebugListener 本地两包验证 | `Test-DebugListenerLocalPackageConsumer.ps1` | 隔离 managed 与 bridge-only 包，执行真实调试张量回调、复制元数据、detach 和拒绝负例；不发布包 |
| ProgressMonitor 本地两包验证 | `Test-ProgressMonitorLocalPackageConsumer.ps1` | 隔离 managed 与 bridge-only 包，执行真实构建进度、线程安全状态、主动取消和 detach 负例；不发布包 |
| Profiler 本地两包验证 | `Test-ProfilerLocalPackageConsumer.ps1` | 隔离 managed 与 bridge-only 包，执行即时/延迟 layer timing、异常记录和 detach；不发布包 |
| Logger 本地两包验证 | `Test-LoggerLocalPackageConsumer.ps1` | 隔离 managed 与 bridge-only 包，执行真实 TensorRT 日志、延迟释放、异常隔离和 detach；不发布包 |
| IStreamReaderV2 本地两包验证 | `Test-StreamReaderLocalPackageConsumer.ps1` | 隔离 managed 与 bridge-only 包，执行真实 read/seek、顺序复用、延迟释放和截断 plan 负例；不发布包 |
| 通用 YoloVision 三包验证器 | `Test-YoloVisionLocalPackageConsumer.ps1` | 被各任务专用入口调用；不建议手工拼接参数 |
| 文章完整性门禁 | `Test-TechnicalArticleCompleteness.ps1` | 检查发布目录中的真实结果、配图、模型获取/转换和边界 |

每个支持入口必须同时具备：明确文档、失败即非零退出、ProjectQuality 测试或 CI 调用、禁止隐式发布、重资产不进入 Git。没有满足这些条件的脚本一律按内部工程脚本处理。

## 内部脚本

- `Test-*` 通常是可执行验证器，但很多只服务特定证据 schema，不是用户功能。
- `Export-*` 通常只生成 JSON/Markdown/模板。名字含 `Export` 不表示导出模型，也不表示上传或发布。
- `Owner*`、`*Proof*`、`*Candidate*`、`*Readiness*` 多数是发布治理或真实证据回填工具，缺少 Owner 输入时会保持 blocked/non-proof。
- `*.Common.ps1` 是 dot-source 公共函数，不能独立运行，也不能按“未引用”轻率删除。

明确保留的无自动调用人工入口只有：`Invoke-CudaPowerShellSmoke.ps1`、`Invoke-CudaRtcLocalSmoke.ps1`、`Invoke-TensorRtPowerShellSmoke.ps1`、`Test-CudaDriverNativeAbiSurface.ps1`、`Test-CudaKernelLaunchNativeAbiSurface.ps1`、`Test-CudaRtcNativeAbiSurface.ps1` 和 `Trust-WindowsLocalDevCertificate.ps1`。前三项用于本机 smoke，三项 `*NativeAbiSurface` 用于导出桥接 ABI，最后一项只服务本机开发签名；它们都不下载、打包或发布 NVIDIA 运行库。

## 后续整理规则

第一版发布前不做大范围路径迁移，但会持续删除已经确认失效的内容。每批清理均先做引用图和质量门验证：

1. 明确无人引用、没有人工入口、无历史兼容责任且不能生成当前有效证据的脚本，直接删除。
2. 重复包装器、已失效发布快照和仅生成指标卡片而非真实截图的工具，确认替代入口后删除。
3. 可复用公共函数迁入模块时，同时修改 workflow、测试和文档，禁止只移动文件。
4. 发布候选、GitHub Packages、Release 或 tag 相关入口在开发完成并获得 Owner 明确授权前不得执行。
5. 新增脚本必须进入支持入口、内部工具或临时候选三类之一，并附测试和删除条件。

本目录中的脚本不会因为“质量门通过”而自动获得 tag、Release、NuGet push、GitHub Packages push、模型上传或文章发布权限。
