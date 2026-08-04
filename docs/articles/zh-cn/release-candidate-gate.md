# 发布候选质量门禁

发布候选门禁用于确认当前包不是“能编译就算完成”，而是具备可复现的消费端证据链。

当前 release owner 最短执行面以 `owner-release-execution-package` 的 `oneScreenReleaseHoldChecklist` 为准。它只把 owner authorization、`package-consumer-runtime`、Linux runner proof、`real-model-runtime` 和 `post-publish verification` 聚合到一屏 Release Hold 清单，不会把 guidance、local feed、ProjectReference、build-only、sidecar-only 或 `blocked-by-cuda-driver` 晋级为 proof；真实记录缺失时 `canCloseReleaseIssue=false` 必须保持不变。

## 当前基线

截至 2026-06-12：

- TensorRT interface coverage：`0` missing rows。
- CUDA runtime interface coverage：`0` missing rows。
- Manifest inventory：`3271` 条 API records，`102` 份 manifests。
- Deferred API 是明确的 manifest/native 边界，不能当作安全 public wrapper 宣传。

当前 package-consumer 证据（2026-06-12）：

- `win-x64-trt10.11-cuda11.8-cudnn8.9`：restore/build/native-copy/smoke 通过，native assets 为 `16/16`，探针输出 TensorRT `10.11.0`、CUDA `11.8`。
- `win-x64-trt10.11-cuda12.9-cudnn9.22`：restore/build/native-copy/smoke 通过，native asset patterns 为 `19/19`，探针输出 TensorRT `10.11.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda12.9-cudnn9.22`：restore/build/native-copy/smoke 通过，native asset patterns 为 `19/19`，探针输出 TensorRT `11.0.0`、CUDA `12.9`。
- `win-x64-trt11.0-cuda13.2-cudnn9.22`：2026-06-25 已完成完整 split/full 包本地打包，restore/build/native-copy 通过，native asset patterns 为 `19/19`；full package consumer smoke 已请求并启动 packaged runtime，但当前机器被 CUDA driver/runtime compatibility 阻塞为 `blocked-by-cuda-driver`，真实 callback runtime proof 仍为 `false`。

当前 `win-x64-trt11.0-cuda13.2-cudnn9.22` package consumer evidence schema 必须按字段读取。默认严格门禁仍把缺少 runtime proof 视为 blocker；如果 Owner 明确使用 `-AllowRuntimeSmokeBlocked`，则还必须通过 `pack/runtime-validation-disclosure-policy.json` 和 `runtime-package-matrix.md` 的固定披露字段，门禁才可将它降为 `ready-with-warnings`。这只是发布限制说明，不是 runtime proof：

| 字段 | 当前值 | 门禁含义 |
| --- | --- | --- |
| `packageConsumerEvidenceKind` | `full-runtime-package-consumer-smoke-driver-blocked` | full package consumer 已进入 smoke 路径，但被 driver/runtime compatibility 阻塞。 |
| `runtimeSmokeClassification` | `runtime-smoke-driver-blocked` | runtime smoke 是环境阻塞分类，不是 passed。 |
| `runtimeProofStatus` | `blocked-by-cuda-driver` | 发布运行证明仍未完成；不要被 package/readiness `overall=ready` 误导。 |
| `runtimeProofRequiredForRelease` | `true` | 仍未形成 runtime validated 证据；只有显式披露后才允许作为未验证环境 warning 继续候选流程。 |
| `isRuntimeExecutionEvidence` | `false` | 不能作为 runtime execution proof。 |
| `isDependencyProbeOnly` | `true` | 当前只能证明 dependency probe/native load 和阻塞诊断。 |
| `isRealCallbackRuntimeProof` | `false` | 不能晋级 callback runtime proof。 |

`runtime-package-readiness-summary.md`、`release-candidate-readiness-summary.md` 和 `runtime-package-matrix.md` 现在都会输出 runtime proof 状态。`Overall` 或 `overallStatus` 只表示 package/readiness 链条完整，`Runtime proof` / `runtimeProofStatus` 才表示 runtime execution proof；因此 `Overall=ready` 仍可以和 `Runtime proof=blocked-by-cuda-driver` 同时出现，不能把它解读为 CUDA 13.2 smoke passed。

门禁中凡引用 `blocked-by-cuda-driver`，都必须同时保留上述字段语义；不能把 `isDependencyProbeOnly=true` 写成 runtime smoke passed，也不能把 `isRealCallbackRuntimeProof=false` 写成 callback 已完成。

发布说明还必须直接提示使用者：TensorRT、CUDA、cuDNN 和 NVRTC 由用户自行安装；TensorRT 11.0 + CUDA 13.2 当前只完成构建、包结构、native-copy 和依赖探针验证，真实运行需要匹配的 CUDA-capable driver/runtime 环境。没有该说明时，`-AllowRuntimeSmokeBlocked` 不得打开 warning 路径。

## 本地质量门

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Export-InterfaceCoverageMatrix.ps1
dotnet restore .\TensorRtSharp.sln
dotnet build .\TensorRtSharp.sln -c Debug --no-restore
powershell -ExecutionPolicy Bypass -File .\eng\Test-PublicApiBilingualDocumentation.ps1
dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build
dotnet docfx .\docs\docfx.json
```

如果修改了 native、manifest 或 generated 文件，还需要运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
powershell -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
```

期望结果：

- TensorRT missing rows 保持 `0`。
- CUDA runtime missing rows 保持 `0`。
- `dotnet build` 为 `0` warning、`0` error。
- 公开 API XML 注释同时包含英文和中文。
- `JYPPX.ProjectQuality.Tests` 通过。
- DocFX 为 `0` warning、`0` error。

## 推荐 smoke 顺序

1. `CudaSmokeRunner`
2. `MultiStream`
3. `TensorRtSmokeRunner`
4. `LifecycleSmokeRunner`
5. `OnnxToEngineSmokeRunner`
6. `DynamicShape`
7. `InferenceBindings`
8. `NetworkBuilderSmokeRunner`
9. 各类 layer-specific network runners

以下 asset-dependent sample project 是可执行项目，但需要用户提供模型与 metadata：

- `Classification`
- `YoloVision`

这些项目只要 README 明确说明所需外部资产和可运行替代路径，就不单独作为 release blocker。CUDA custom-kernel preprocessing 先保留为文档路线图，等待安全 public `CudaModule` / `CudaKernel` wrapper 后再加入可运行 sample。

## Runtime package 门禁

发布候选涉及 runtime package 时，必须提供：

- 精确 TensorRT、CUDA、cuDNN root 校验。
- 匹配 CMake preset 输出的 runtime asset collection。
- 基础 managed、YoloVision 纯 managed 扩展与 matching bridge package 的 pack/hash/source-commit 对齐证据。
- package consumer restore/build/native asset copy 验证。
- 兼容机器上的 package consumer smoke 证据；如果被 WDAC、driver/runtime 不兼容阻塞，应记录为环境限制，而不是 package layout failure。只有 Owner 明确 opt-in 且发布说明完成时，才可将该行保留为 warning。
- private-feed 或 split-delivery readiness 必须要求 `local-validated`；`pending-local-validation` 不能视为 ready。

CUDA `12.9` 和 TensorRT 11 Windows 组合必须使用精确 CUDA/TensorRT/cuDNN 依赖链证据。`trt11.0-cuda13.2-cudnn9.22` 可以构建并完成 package consumer native-copy；当前机器 runtime smoke 在 CUDA error 35 处被阻塞，仍需要 CUDA 13-capable driver/runtime 环境完成普通 runtime smoke。普通 smoke 通过也不能自动提升 callback proof。

Linux Ubuntu 20.04/22.04/24.04 x64 都通过 GitHub-hosted runner 和匹配的 Ubuntu job container 发布，远程 workflow 必须完成 build、asset collection、pack、consumer validation 和 release/package 上传后才算发布证据。Ubuntu 20.04 x64 使用 `runner_mode=hosted-container`；Ubuntu 24.04 x64 只覆盖现代组合；ARM/Jetson 目标需要单独建包线。

## 远端发布前置条件

在启用远端发布链前，请先确认：

- `package-managed.yml` 固定打包并验证 `JYPPX.TensorRT.CSharp.API` 与 `JYPPX.TensorRT.CSharp.API.YoloVision` 两个包。发布必须同时满足 `owner_publish_approved=true`、正式仓库 owner、精确 ID/版本/source-commit allowlist；`publish_to_nuget=true` 时还要求仓库 secret `NUGET_API_KEY` 是纯文本 ASCII 的 nuget.org API key，并对两个 package ID 或其所属账号/组织拥有 push 权限。nuget.org `403` 是不可重试的权限错误，必须先替换失效、过期或 scope 不足的 key。
- `runtime-windows.yml` 要求 Windows self-hosted runner 在线，并带有 `self-hosted`、`windows`、`x64` 标签。
- `runtime-linux.yml` 可以通过 GitHub-hosted runner 和匹配的 Ubuntu job container 发布 Ubuntu 20.04、Ubuntu 22.04、Ubuntu 24.04 x64。Ubuntu 20.04 x64 使用 `runner_mode=hosted-container`，Ubuntu 24.04 x64 只覆盖现代组合，ARM/Jetson 目标需要单独建包线后才能发布。

## 发布候选冻结门禁

发布前最后一层冻结证据由以下脚本生成：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeChecklist.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFreezeSummary.ps1
```

输出位于：

- `artifacts/release/release-candidate-freeze-summary.json`
- `artifacts/release/release-candidate-freeze-checklist.json`
- `artifacts/release/release-candidate-freeze-validation.json`

当前缺真实 compatible-host proof 和 post-publish proof 时，freeze validator 的健康结果应是：

- `ValidationState=blocked-freeze-owner-action-required`
- `FailedValidationItemCount=0`
- `CanCloseReleaseIssue=False`

这表示 freeze 门禁能正确表达阻断状态；不是发布完成。`blocked-by-cuda-driver`、template、draft、runbook、collection bundle、dependency-probe-only 都不能晋级为真实 runtime proof。
