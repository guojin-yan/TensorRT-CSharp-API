# 本地 NuGet Feed 消费验证

本地 NuGet feed consumer 用来模拟用户从包源安装项目，而不是从源码项目引用 DLL。它验证的是发布候选最容易被忽略的一段路径：包已经生成，但普通用户能不能通过 NuGet restore/build 拿到 managed wrapper 和 runtime native assets。

## 验证目标

`eng/Test-LocalNuGetFeedConsumer.ps1` 会创建临时本地 feed，把 `artifacts/managed` 与 `artifacts/runtime-nupkg` 中的 `.nupkg` 放入同一个本地源，然后生成独立 consumer 项目：

- 只使用 `PackageReference`。
- 不允许 `ProjectReference`。
- 使用目标 runtime package key 的 RID。
- restore/build 后检查 native assets 是否复制到输出目录。
- 默认运行 dependency probe，不调用高风险 TensorRT callback proof。
- 可选运行 runtime probe；如果当前机器 CUDA driver 不兼容 CUDA 13，会分类为 `blocked-by-cuda-driver`。
- 同一 package id 下优先选择最新写入时间的 `.nupkg`，避免旧的 `4.0.0-local` 包遮蔽刚生成的发布候选包。
- 在生成的 consumer `Program.cs` 中编译检查关键高层 wrapper surface，包括 `TensorRtPluginRegistryInventory.FindCreator`、`TryFindCreator`、`TensorRtEnvironmentProbe.IsGlobalPluginRegistryAvailable`、`TryIsGlobalPluginRegistryAvailable`、`TensorRtOnnxParser.IsSubgraphSupported`、`TensorRtBuilderConfig.GetDlaCore`、`GetL2LimitForTiling`、`GetMaxTactics`、`GetQuantizationFlag`、`GetQuantizationFlags`、`GetAverageTimingIterations`、`TensorRtBuilder.MaxBatchSizeCompatibility` 和 `MaxDlaBatchSize`。

## 推荐命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LocalNuGetFeedConsumer.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -AllowSmokeFailure
```

生成报告：

- `artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json`
- `artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.md`

## 结果解读

`RunStatus=dependency-probe-passed` 表示 consumer 从本地 feed restore/build 成功，native bridge 与依赖探针可用。它不代表 TensorRT 推理、builder、callback runtime proof 都已经完成。

`HighLevelWrapperSurface` 行表示本地 feed consumer 已经从 `.nupkg` 侧编译引用关键高层 C# API。2026-07-05 的验证中，该行已覆盖 plugin registry inventory、ONNX parser subgraph 查询、TRT10/TRT11 builder config 只读 getter、TRT8 builder compatibility getter，以及 `no-public-borrowed-plugin-creator-pointer` 和 `local-feed-is-not-post-publish-proof` 边界 marker。这个结果证明“本地包消费端能编译并复制 native assets”，但仍不是正式 channel 的 post-publish proof。

`RunStatus=blocked-by-cuda-driver` 表示包已经到达 packaged runtime，但本机驱动/runtime 兼容性阻止继续执行。例如 CUDA 13 runtime 在不兼容驱动上可能触发 CUDA error 35。这是环境阻塞，不是 API proof，也不是删除 deferred 行的理由。

发布候选门禁会读取这份报告，并把它与 package consumer、bridge consumer、runtime readiness、docs、matrix、signing/trust 状态一起汇总。

## 第二批正文门禁

### 适用读者

本文适合负责 NuGet 包验收、runtime package 组合和 clean consumer proof 的维护者，也适合想先在本机模拟包消费路径的用户。

### 解决问题

本地 ProjectReference 很容易掩盖真实包依赖问题，direct `.nupkg` 又可能绕过公开 feed 行为。本地 NuGet feed consumer 的价值是把 package restore、runtime asset copy、native dependency probe 和 summary 输出集中验证。

### 核心思路

核心思路是把“包能被消费”和“真实 runtime proof”分开。local feed 可以发现 nuspec、runtime identifier、native asset、dependency probing 的问题，但它仍然不是 nuget.org 或 GitHub Packages 的 post-publish proof。

### 操作路径

生成 managed 包和 runtime split 包，把包放进临时 local feed，创建 clean consumer 项目且不使用 ProjectReference，然后 restore/build/run dependency probe，保存 summary JSON/Markdown。

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。真实 proof 必须来自公开包来源或 owner 指定的真实 clean consumer 环境，并包含 host metadata、hash、stdout/stderr summary 和 strict validator 结果。

### 下一步

下一步是在兼容主机上执行 clean consumer runtime proof，使用真实包来源、真实运行日志和 strict validator 回填 release close 记录。
