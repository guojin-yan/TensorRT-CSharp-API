# Runtime Package Matrix 读法指南

Runtime package matrix 是发布候选跨平台、跨 TensorRT/CUDA/cuDNN 组合的证据表。它的价值在于把 package ID、RID、validation state、build preset、consumer evidence 和 Linux handoff 放到一张表中，而不是让发布负责人只看某个 `.nupkg` 是否存在。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -AllowRuntimeSmokeBlocked
```

输出：

- `artifacts/release-candidate/runtime-package-matrix.json`
- `artifacts/release-candidate/runtime-package-matrix.md`

## 状态分类

| 状态 | 含义 |
| --- | --- |
| `materialized-current` | 当前 runtime key 在本机已有 package/readiness/consumer 证据。 |
| `materialized-historical` | manifest 标记为历史本地验证组合。 |
| `planned-dry-run` | Linux 或未来包线已建模，但当前机器没有真实 runtime proof。 |
| `pending-local-validation` | 包线存在，但仍需兼容环境完成验证。 |

当前 Windows TRT11/CUDA13.2 包线已经有 native build、runtime package、bridge consumer、full package consumer native-copy 和 local feed consumer 证据；但 full runtime smoke 仍被当前机器驱动阻塞为 `blocked-by-cuda-driver`。

`runtime-package-readiness-summary.md` 的 `Overall` 列描述 package/readiness 链条是否完整；`Runtime proof` 列描述是否已有真实 runtime execution proof。`Overall=ready` 且 `Runtime proof=blocked-by-cuda-driver` 是合法但未完成发布运行证明的状态，不能写成 smoke passed。

`artifacts/release-candidate/runtime-package-matrix.md` 也包含 `Runtime proof` 列；`runtime-package-matrix.json` 对应字段为 `runtimeProofStatus`。当前 key 的值来自 runtime readiness 源字段，历史或 dry-run 行只说明先前验证或建模状态，不能替代当前机器的真实 runtime proof。

## Linux 行的正确读法

Linux Ubuntu 22.04 TRT11/CUDA13.2 行目前是 handoff/dry-run evidence。它说明 package identity、runner 目标、NVIDIA dependency mode、build preset 和 expected `.so` pattern 已建模。

它不说明：

- Linux runner 已经真实执行 CMake build。
- Linux runtime `.nupkg` 已经在目标 runner 生成。
- Linux package consumer 已经 restore/build/native-copy。
- GPU smoke 已经通过。

这些证据必须由 Linux x64 runner 回填，详见 `linux-runtime-handoff.md` 和 `linux-runner-evidence-template.md`。

## 与发布审批的关系

matrix clean 是 release owner 决策输入，不是发布按钮。公开发布前仍要看：

- `artifacts/final-release/final-release-dry-run-summary.md`
- `artifacts/final-release/release-owner-decision-template.md`
- `artifacts/final-release/stale-release-claims-audit.md`
- Linux runner proof 是否存在。
- signing/trust 是否审批。
- NVIDIA redistribution 是否审批。

## 与 callback proof 的关系

Matrix 不证明 callback runtime。`IDebugListener::processDebugTensor`、`IOutputAllocator::*` 和 allocator callback 的真实 proof 必须来自 full package consumer 中 `InvocationCount>0` 且 `IsRealCallbackRuntimeProof=True` 的 evidence。
