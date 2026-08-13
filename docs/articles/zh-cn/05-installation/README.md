# TensorRT CSharp API v4.0 安装与运行环境

本模块专门处理 TensorRT CSharp API v4.0 的安装、运行库选择、平台差异和安装排错。安装不是单条 `dotnet add package` 命令：托管核心包、项目自有 Runtime Bridge、NVIDIA TensorRT/CUDA/cuDNN 运行库、显卡驱动、进程架构和 DLL 搜索路径必须组成同一套兼容矩阵。把安装单独成模块，后续可以分别增加 Windows、Linux、WSL、容器和 CI runner 文章，而不把平台假设混进 API 或案例文章。

项目和包入口保持明文，方便复制到不完整支持 Markdown 链接的平台：

```text
项目源码：https://github.com/guojin-yan/TensorRT-CSharp-API/tree/TensorRtSharp4.0
核心 NuGet：https://www.nuget.org/packages/JYPPX.TensorRT.CSharp.API/4.0.0
Runtime Bridge 包列表：https://www.nuget.org/packages?q=+JYPPX.TensorRT.CSharp&includeComputedFrameworks=true&prerel=true&sortby=relevance
运行库清单：https://github.com/guojin-yan/TensorRT-CSharp-API/blob/TensorRtSharp4.0/pack/runtime/runtime-packages.manifest.json
```

## 1. 当前文章

| ID | 平台或主题 | 状态 |
| --- | --- | --- |
| `MSC-003` | Windows 安装、验证与常见问题 | `ready` |
| `MSC-004` | 托管包与 Bridge 包选择 | `ready` |
| `MSC-005` | Windows/Linux Runtime Bridge 矩阵 | `ready` |
| `MSC-008` | CUDA error 35、DLL 加载与版本不匹配 | `ready` |
| [`INS-001`](linux/ins-001-linux-installation-runtime-validation.md) | Linux 安装与运行环境验证 | `ready` |
| [`INS-002`](wsl/ins-002-wsl-gpu-passthrough-runtime-validation.md) | WSL 安装、GPU 透传与运行库边界 | `review` |
| [`INS-003`](container/ins-003-container-deployment-runtime-boundary.md) | 容器部署、NVIDIA Container Toolkit 与 Smoke | `ready` |
| [`INS-004`](ci/ins-004-gpu-ci-runner-validation.md) | GPU CI runner、标签与发布证据 | `review` |

2026-08-14 已在 Ubuntu 24.04.2 TensorRT 官方容器中完成固定版本厂商依赖、Linux Bridge 构建、`ldd`、本地 managed/Bridge 包以及外部 PackageReference 消费者的真实 GPU enqueue 和输出比对，`INS-001`、`INS-003` 因而晋级 `ready`。本机仍没有独立 WSL Ubuntu 发行版，仓库 self-hosted runner 数量仍为 0，因此 `INS-002`、`INS-004` 保持 `review`。回调增强 smoke 的 `SIGSEGV` 与所有证明边界保留在 [`installation-runtime-evidence-20260814.json`](installation-runtime-evidence-20260814.json)，不以相邻环境或精简主路径成功掩盖缺口。

## 2. 后续安装选题

| 主题 | 计划覆盖内容 |
| --- | --- |
| 多版本并存 | TensorRT 8/10/11、CUDA 11/12/13 的隔离策略 |
| Linux 部署深化 | systemd 服务、容器外部署和多用户动态库隔离 |

计划条目不是已发布文章，不能在文章结尾伪造下一篇链接。每个新平台条目在发布前必须补齐对应安装命令、包 ID、运行库版本、真实输出和排错边界。

## 3. 安装模块限制

1. 安装命令固定使用正式 `4.0.0`，不使用预览通配符。
2. 表格同时列出 RID、TensorRT、CUDA、cuDNN 和 Bridge 包完整 ID。
3. 项目包不替代用户安装的 NVIDIA 厂商运行库，文章必须明确这条边界。
4. 失败案例要记录完整命令、异常、加载阶段和实际 DLL 来源；“重装试试”不能作为唯一方案。
