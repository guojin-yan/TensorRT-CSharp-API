# 包消费端验证

本文是面向维护者的 package-consumer 验证说明，不是发布公告。它说明如何从声明的 package source 还原托管包和 bridge-only 包，并在仓库外的 clean consumer 项目中执行构建与可选 runtime smoke。

## 目标与边界

local feed、ProjectReference、direct `.nupkg`、build-only、dependency-probe、dashboard、runbook、candidate 和 draft 都是 non-proof 替代物。只有 clean consumer restore/build/native-copy/runtime smoke、包和日志 SHA256、host metadata 以及 strict validator 全部通过，才允许形成 package-consumer runtime proof。

本地 smoke 失败或 CUDA 驱动不兼容时保持 `blocked-by-cuda-driver`；它不是 runtime proof，也不是 post-publish proof。`owner-action-required` 表示需要 Owner 在匹配的 CUDA/TensorRT 主机上补充真实输入。

## Windows 示例

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 -RunSmoke
```

命令中的相对路径以仓库根目录为工作目录。脚本会记录 restore source、native asset 数量、运行日志和 SHA256，并在完成后清理临时消费端输出。

## 结果解释

- `IsPackageConsumerRuntimeProof=True` 只能来自仓库外 clean consumer 和完整 runtime smoke。
- local feed、ProjectReference、bridge-only 日志、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics、dry-run、template 和 `Skipped=True` 必须保持 forbidden substitute。
- `FailOnNotProof` 会拒绝把这些材料晋级为发布证明、post-publish proof 或 release close。

## 下一步

正式发布前由 Owner 提供实际主机、包哈希、运行日志和严格校验结果。该文章本身只是执行指南，不执行上传、tag、Release 或 `dotnet nuget push`。
*** End Patch
