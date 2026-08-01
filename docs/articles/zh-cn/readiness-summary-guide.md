# Package Readiness Summary 怎么读

`artifacts/package-readiness/runtime-package-readiness-summary.md` 汇总 managed、bridge、consumer、主机依赖和 runtime smoke。2026-07-30 之后，历史 full/vendor package 字段只用于解释旧证据，不再是可打包或可发布条件。

## 当前优先字段

阅读每个 runtime key 时，按以下顺序判断：

1. managed package 是否存在且不含 `runtimes/*/native`。
2. `.Bridge` package 是否存在且只含一个项目自有 bridge binary。
3. managed 与 bridge 是否来自同一 repository commit。
4. consumer root 是否在仓库外，是否只用 `PackageReference`。
5. 包来源是否为公开 URL/source，下载 hash 是否与渠道 digest 一致。
6. TensorRT/CUDA/cuDNN/NVRTC 是否明确标记为机器安装依赖。
7. restore/build/enqueue/readback 是否实际执行且 exit code 为 0。
8. runtime JSON、stdout、stderr 和 nupkg SHA256 是否可复算。
9. 是否存在 post-publish clean consumer 与 Owner 审核。

## 三层状态

### Package completeness

只说明 managed 与 bridge 包的 identity、内容、版本和 hash 完整。它不证明主机依赖可加载，也不证明 TensorRT enqueue 成功。

### Consumer build

说明仓库外项目可以 restore/build，并把 bridge 复制到输出目录。它仍可能只是 build-only 或 dependency-probe-only evidence。

### Runtime execution

需要真实 TensorRT/CUDA 初始化、engine build/deserialize、enqueue、output readback 和结果验证。driver/runtime incompatibility、异常或 skipped output 都不能写成通过。

## Callback-State 诊断

`bridgeRuntimeConsumer.callbackStateSnapshot` 从每条 runtime key 对应的
`bridge-package-runtime-consumer-proof.json` 汇入正式 readiness JSON/Markdown。它把 aggregate snapshot 分为：

- `complete`：`complete` 与 `snapshotIsComplete` 都为 true，且 `lastStatus=Ok`；
- `partial`：两个 complete 标志都为 false，`lastStatus` 非 `Ok`，并保留非空 `lastOperation`；
- `incoherent`：报告自称 coherent，但 complete/status/phase 组合不满足上述任一契约；
- `missing`：旧报告尚未包含 callback-state 字段。

TRT10/TRT11 的 vendor 默认 debug listener metadata 可能得到
`snapshot-debug-listener-interface-info-partial`。这种 coherent partial 是 pointer-free 诊断，不是 callback invocation、
public-package 或 post-publish proof。readiness 还保留 negative-control scenario/requested/passed，负例即使按预期失败也不会被
升级为 runtime success。

## Release Evidence 中的 Callback 汇总

`eng/Export-BridgeCallbackRuntimeEvidenceSummary.ps1` 只读 TRT10.11/CUDA12.9 与 TRT11.0/CUDA12.9 的
`bridge-package-runtime-consumer-proof.json`，生成
`artifacts/final-release/bridge-callback-runtime-evidence-summary.json` / `.md`。每条线必须同时满足：

- success report 的 runtime execution 与 local-package callback proof 为 true；
- callback-state 已观察、coherent、pointer-free，complete 与 partial 都允许，但状态组合必须由源报告负责；
- callback 已 attach、安装 native vtable、真实 invocation 大于 0；
- failure 与 in-flight 都为 0，clear/detach 成功；
- `source-tree`、`public-package`、`post-publish` scope 不得被本地包报告误置为 true；
- `canPromoteRuntimeProof`、`canPublishPublicly`、`canCloseReleaseIssue` 保持 false。

摘要中的 `sourceReportsContainRuntimeExecutionProof=true` 和
`sourceReportsContainLocalPackageCallbackRuntimeProof=true` 表示源报告已经证明对应本地包执行；摘要本身没有执行 callback，
所以 `isRuntimeExecutionProof=false`。`Export-ReleaseEvidenceBundle.ps1` 只读消费该摘要，不会把本地 feed 提升为公开包、
post-publish、发布授权或 release close proof。

## GitHub Release 路线

GitHub Release 不是 NuGet feed。下载 managed 与 bridge `.nupkg` 后，先验证：

- immutable asset URL；
- GitHub `sha256:` digest；
- 实际文件 size/SHA256；
- package id/version；
- nuspec repository URL/commit；
- bridge-only native entries。

验证后的资产可以进入隔离 restore staging，但这不是 locally built package feed，也不能改成 direct `.nupkg` 或 DLL 引用。

managed 与 bridge commit 不一致时，严格模式必须拒绝。`-AllowCrossCommitPair` 只产生 diagnostic-only 记录，即使发生了部分 runtime 调用也不能晋级 proof。

## 主机依赖

主机 dependency report 应记录：

```text
GPU / driver
TensorRT root and runtime version
CUDA root and runtime/driver version
cuDNN root and version
NVRTC path/version when used
```

这些文件可以参与主机诊断和 hash inventory，但永远不能出现在 nupkg asset listing 中。

## 历史字段

旧 summary 可能包含 split collection、full consumer、full vendor inputs 或 vendor blockers。这些字段只说明旧版打包链当时观察到什么，不能驱动当前 pack/push，也不能替代 bridge-only policy gate。看到旧字段为 `ready` 时，不得推导当前公开包已发布或 runtime proof 已完成。

## 常见误读

以下结论都不成立：

- `overall=ready` 等于所有 runtime 行都可发布；
- package restore/build 等于 TensorRT runtime proof；
- dependency probe 找到 DLL 等于 enqueue/readback；
- 本地 feed 等于公开包消费；
- 跨提交 managed/bridge pair 等于同一发布候选；
- 一条 Windows 记录覆盖 Linux 或另一 CUDA/TRT 行；
- 绿色 dashboard、文章或截图可以替代原始日志/hash。

当前发布闭环以 `eng/Test-ExternalVendorRuntimePackagePolicy.ps1`、公开资产独立验证、clean consumer、post-publish validator 和 Owner 决策为准。
