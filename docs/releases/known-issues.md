# TensorRtSharp4.0 已知问题与后续修复清单

本清单记录开发过程中发现、但不适合在当前已发布 4 系列包上直接改变的事项。修复后应在对应版本说明中引用并关闭条目。

| ID | 当前状态 | 影响 | 后续处理 |
| --- | --- | --- | --- |
| KI-001 | Closed | 首个公开托管包未包含 Tools 层，曾导致 `OnnxToEngine` 与 `TensorRtExec` 直接引用核心源码。 | 已增加不可打包的 `applications/_shared/JYPPX.TensorRtSharp.ApplicationTools`：链接 Tools 实现但使用公共 4 系列包编译；两个应用已移除对核心 CUDA/TensorRT 源码项目的引用。 |
| KI-002 | Open | OpenCV 图片解码已在 Windows x64 通过 `JYPPX.OpenCV.CSharp.API` 与 `JYPPX.OpenCV.runtime.win-x64` 验证；其他平台的 JPEG/PNG 原生运行时组合尚未完成本项目实测。 | OpenCV 对应 runtime 包可用后补 Linux/macOS 干净消费者验证；BMP/PPM 托管回退继续保留。 |
| KI-003 | Controlled | 用户安装命令不写死版本；仓库项目通过单一共享规则跟随当前 4 系列预览线。由于 NuGet 源中存在 API 不兼容的历史 4.x 包，不能使用会选中旧包的宽泛 `4.*`。 | 维护的 4 系列版本线前进时只更新 `build/JYPPX.PublicSamplePackages.props`，并运行全部案例 restore graph 检查。 |
| KI-004 | Open | 部分历史证据文件和旧文章文件名仍含 `local-package-consumer`，它们记录的是发布前本地 feed 证明，不代表当前公共包消费。 | 保留历史证据不可篡改；逐篇文章改为当前公共包流程，并在发布目录中隐藏未完成更新的旧稿。 |
| KI-005 | Closed | 部分历史 `ProjectQuality` 测试曾直接启动 `pwsh`，导致仅安装 Windows PowerShell 5.1 的机器无法运行。 | 已增加 `PowerShellHost` 统一解析器：优先使用 `JYPPX_POWERSHELL_EXECUTABLE`、PowerShell 7 默认安装目录和 `pwsh`，Windows 最后回退 `powershell.exe`；相关测试已全部替换并在 PowerShell 7.6.4 下完成回归。需要 PowerShell 7 语法的脚本仍明确要求 `pwsh`。 |
| KI-006 | Closed | `TechnicalArticleRoadmapTests.ProjectReleaseStoryMatchesCurrentCoverageMatricesAndFinalBlockers` 曾固定旧接口覆盖数 `3976`，与当前覆盖产物漂移。 | 路线图测试现在从当前 manifest 和生成矩阵动态读取数量；定向测试已通过，不再维护手写覆盖快照。 |

## 记录规则

- 不修改已经发布包的内容或覆盖同版本包。
- 不把源码构建、local feed、precheck 或 build-only 结果写成公共包运行证明。
- 问题修复必须包含本地构建、定向测试和干净消费者验证；远程 Action 只在本地通过且确有必要时手工触发。
- CUDA、cuDNN、TensorRT、NVRTC 和模型文件继续由用户提供，不作为问题修复的一部分重新打包。
