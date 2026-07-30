# Runtime Package Matrix 读法指南

Runtime package matrix 现在是 bridge 编译目标与主机 NVIDIA 兼容组合的证据表。它列出 package ID、RID、validation state、build preset、consumer evidence 和 Linux handoff，但不再表示 TensorRT/CUDA/cuDNN vendor package 的发布计划。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -AllowRuntimeSmokeBlocked
```

输出包括：

- `artifacts/release-candidate/runtime-package-matrix.json`
- `artifacts/release-candidate/runtime-package-matrix.md`

## 状态分类

| 状态 | 当前含义 |
| --- | --- |
| `materialized-current` | 当前 key 有 bridge package/readiness/consumer 证据 |
| `materialized-historical` | 历史本机验证记录，仅供解释旧证据 |
| `planned-dry-run` | 目标已建模，但当前机器没有真实 runtime proof |
| `pending-local-validation` | bridge 目标存在，仍需兼容环境验证 |

Windows TRT11/CUDA13.2 已有 native bridge build 与历史 consumer 诊断；当前机器的 runtime smoke 曾被 CUDA driver 阻塞为 `blocked-by-cuda-driver`。这个状态不能写成 package-consumer-runtime proof，也不能用旧 vendor bundle evidence 替代。

`Overall` 描述当前 managed + bridge package/readiness 链；`Runtime proof` 描述是否已有真实 host runtime execution。`Overall=ready` 与 `Runtime proof=blocked-by-cuda-driver` 可以同时出现，前者不能覆盖后者。

## Vendor 字段的读法

`tensorRtFiles`、`cudaFiles`、`cudnnFiles` 和精确版本仍留在兼容 manifest 中，用于 build input、host dependency probe 和历史资产清理。它们不是 nupkg expected assets。当前可发布 role 只有 `bridge`。

## Linux 行的正确读法

Linux 行必须绑定发行版、architecture、runner、build preset 和系统安装依赖。`planned-dry-run` 只说明 key 已建模，不说明：

- Linux runner 已经真实构建 bridge。
- `.Bridge` nupkg 已经在目标 runner 生成。
- clean consumer 已从公开来源 restore/build。
- 系统 TensorRT/CUDA/cuDNN 已成功加载。
- GPU smoke 已通过。

这些证据必须由对应 Linux x64 runner 回填，详见 `linux-runtime-handoff.md` 与 `linux-runner-evidence-template.md`。

## 与发布审批的关系

matrix clean 是 release owner 的输入，不是发布按钮。公开发布仍要满足：managed/bridge candidate inventory、同提交 provenance、公开下载 hash、仓库外 clean consumer、Linux runner proof、real-model-runtime、owner authorization 和 post-publish verification。

旧 vendor package、collection 或 full consumer 字段如果仍出现在历史 schema 中，只能作为 historical diagnostic evidence，不能晋级当前发布路线。

## 与 callback proof 的关系

Matrix 不证明 callback runtime。`IDebugListener::processDebugTensor`、`IOutputAllocator::*` 和 allocator callback 的真实 proof 仍必须来自 `InvocationCount>0` 且 `IsRealCallbackRuntimeProof=True` 的兼容主机执行证据。

## 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、dependency probe 和历史 full consumer record 都不能替代 runtime proof。矩阵只能陈述现有证据，不能把缺失项推断为通过。

## 下一步

在 GitHub Release managed + bridge assets 和 NuGet-compatible source 上分别完成 clean consumer，再把 strict validator 接受的结果回填到 matrix。没有真实公开来源和 runtime execution 时，保持 blocker。
