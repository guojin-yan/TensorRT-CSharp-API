# 当前 Package Readiness 状态说明

本文记录 2026-07-30 切换到 bridge-only 发布策略后的状态。历史 full/vendor package readiness 不再是当前发布条件。

## 当前结论

- 允许发布的 package kind：`managed`、`bridge`。
- 允许发布的其他资产：tracked-files-only 源码归档。
- `.Bridge` 包只允许一个 `jyppxtrtbridge.dll` 或 `libjyppxtrtbridge.so`。
- CUDA、cuDNN、TensorRT、NVRTC 与 builtins：用户机器依赖，禁止打包。
- full-runtime、`CudaCudnn`、`TensorRt`、`CudaRtc`、collection 与 meta 项目：不可 pack。
- 正式仓库中的 65 个 vendor package versions 与 65 个对应 Release assets 已经 Owner 指纹确认后删除。
- 删除后远端盘点的待删项为 0；managed 与 bridge 资产保留。

`eng/Test-ExternalVendorRuntimePackagePolicy.ps1 -StaticOnly` 是当前包边界门禁。`eng/Invoke-LocalRuntimePackage.ps1` 已 fail closed，`eng/Invoke-LocalSplitRuntimePackage.ps1` 只接受 `bridge` role。

## 公开资产消费状态

公开 GitHub Release managed `4.0.6170` 与 bridge `4.0.6156` 已完成 URL、GitHub digest、下载 SHA256、包身份、nuspec repository commit 和 bridge-only 内容验证，但它们来自不同源码提交：

```text
managed: 1d7b3f18c6f0d636d298c22ccc7bf134ae09972a
bridge:  8c417b39379a2137f08c0540d66502788f9397a6
```

严格模式会拒绝该组合。显式 `-AllowCrossCommitPair` 只能生成 `cross-commit-public-assets-diagnostic-only` 记录，必须保持：

- `isPublicReleaseAssetConsumerEvidence=false`；
- `isPackageConsumerRuntimeProof=false`；
- `isPostPublishProof=false`；
- `canPromoteCurrentHeadPackageConsumerProof=false`。

当前 TRT10/CUDA12.9 主机诊断已完成 engine serialize 和 TensorRT/CUDA 初始化，但 output readback 抛出 `ArgumentOutOfRangeException`，因此 `runtimeSmokePassed=false`。这条记录只能定位历史公开资产行为，不能进入发布闭环。

## 当前缺口

要形成可晋级的公开包消费证据，仍需：

1. 从同一源码提交发布 managed 与匹配 `.Bridge` 包。
2. 通过独立 validator 复算 nupkg、runtime JSON、stdout 与 stderr SHA256。
3. 在仓库外 clean consumer 上完成 restore/build/enqueue/readback。
4. 记录机器安装的 driver、GPU、TensorRT、CUDA、cuDNN 与可选 NVRTC metadata。
5. 完成 post-publish clean consumer 与 Owner 审核。

TRT11/CUDA13.2 行仍必须在 CUDA 13-capable driver/runtime 主机上单独验证；CUDA12.9 主机不能替代。

## 证据边界

可以写：

> bridge-only 包策略和远端 vendor 资产清理已经验证；现有跨提交公开 managed/bridge 组合仅为 diagnostic-only，尚不是 package-consumer-runtime 或 post-publish proof。

不能写：

> 所有公开包和 TRT8/10/11、CUDA11.8/12.1/12.9/13.2 runtime 行都已经完成发布验证。
