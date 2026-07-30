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
