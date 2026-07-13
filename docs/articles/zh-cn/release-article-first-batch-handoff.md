# 发布文章首批案例交接稿

本文把下一轮可以继续扩写的 5 篇文章案例固定下来。它们面向项目发布前的公开材料准备，但本文件本身不是 runtime proof、不是 package-consumer-runtime proof，也不允许关闭 release issue。build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report 和 readonly diagnostics 都只能作为上下文或候选输入。

## 1. YoloVision 六任务真实资产证据链

目标读者：想用 `samples/YoloVision` 跑 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom 的用户。

应引用的代码与 artifact：

- `samples/YoloVision`
- `samples/YoloVision/yolovision-task-output-contract.json`
- `samples/YoloVision/examples/yolovision-output-det.example.json`
- `samples/YoloVision/examples/yolovision-output-cls.example.json`
- `samples/YoloVision/examples/yolovision-output-seg.example.json`
- `samples/YoloVision/examples/yolovision-output-obb.example.json`
- `samples/YoloVision/examples/yolovision-output-pose.example.json`
- `samples/YoloVision/examples/yolovision-output-sem.example.json`
- `eng\Test-YoloVisionOutputReport.ps1`
- `eng\Test-YoloVisionRealAssetOwnerProofInput.ps1`

文章骨架：

1. 六任务范围：det、cls、seg、obb、pose、sem。
2. 资产字段：model、labels、input image、preprocess、postprocess、license、SHA256。
3. 输出字段：boxes、scores、classes、masks、keypoints、angles、semantic map。
4. validator：`Test-YoloVisionOutputReport.ps1` 只验证 report shape，不代表 runtime proof。
5. owner proof：必须有真实 `YoloVision Passed=True` 日志、stdout/stderr summary、输入/输出 hash 和 owner review。

示例命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionOutputReport.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict
```

边界：YoloVision article case、matrix、example JSON 和 validator 都不是 package-consumer-runtime proof。

## 2. OnnxToEngine 与 TensorRtExec 转换边界

目标读者：希望从 ONNX 转 engine，并理解 `trtexec` parity 边界的用户。

应引用的代码与 artifact：

- `samples/OnnxToEngine`
- `applications/TensorRtExec`
- `docs/articles/zh-cn/onnx-to-engine-trtexec-conversion-guide.md`
- `docs/articles/zh-cn/tensorrtexec-report-proof-boundary.md`
- `eng\Test-TensorRtExecReport.ps1`
- `eng\Test-TensorRtExecGuiCliParityChecklist.ps1`

文章骨架：

1. `samples/OnnxToEngine` 适合演示最小 parser、builder、serialized engine round-trip。
2. `applications/TensorRtExec` 适合做 CLI/GUI 参数映射、build-only report、sidecar 输出。
3. report schema 能证明构建诊断字段稳定，但不能证明真实模型 runtime correctness。
4. `trtexec` parity 只说明选项可追溯，不等于 NVIDIA 官方 `trtexec` 完全替代。
5. 真实 proof 需要模型资产、runtime run log、输出 validator 和 owner review。

示例命令：

```powershell
dotnet run --project .\samples\OnnxToEngine\OnnxToEngine.csproj -c Debug
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-TensorRtExecReport.ps1 -InputPath <report.json>
```

边界：OnnxToEngine report 和 TensorRtExec report 是 build/report evidence，不是 runtime proof。

## 3. Package Consumer Proof 分层

目标读者：release owner、包维护者、想验证 NuGet 包的用户。

应引用的代码与 artifact：

- `eng\Test-BridgePackageConsumer.ps1`
- `eng\Test-PackageConsumer.ps1`
- `eng\Test-RuntimePackageReadiness.ps1`
- `eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1`
- `eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1`
- `eng\Test-ExternalRuntimeProofRecord.ps1`
- `artifacts/package-consumer/package-consumer-validation-summary.md`
- `artifacts/package-readiness/runtime-package-readiness-summary.md`
- `docs/articles/zh-cn/package-consumer-runtime-proof-owner-input-field-guide.md`

文章骨架：

1. bridge consumer：证明 managed package + bridge package compile surface，不需要完整 vendor runtime。
2. full package consumer：证明 restore/build/native assets copy。
3. `Smoke=not-requested`：只能说明未请求 runtime smoke。
4. `blocked-by-cuda-driver`：说明执行到 CUDA runtime 边界但主机不兼容。
5. clean external proof：必须走仓库外 consumer、公开或 owner-approved package source、真实 smoke log、hash、host metadata 和 strict validator。

示例命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageConsumer.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -SmokeRuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RunSmoke -KeepConsumerOutput
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts\final-release\external-runtime-proof-record.json -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 -RequireExistingLog -FailOnNotProof
```

边界：local feed、direct `.nupkg`、ProjectReference、dependency-probe-only 和 build-only 都不能晋级 package-consumer-runtime proof。

## 4. Source Build、CMake 与 Runtime Package Readiness

目标读者：需要从源码构建 native bridge 或维护 runtime split package 的用户。

应引用的代码与 artifact：

- `eng\Generate-Bindings.ps1`
- `eng\Test-BindingGeneratorOutputs.ps1`
- `eng\Collect-SplitRuntimeAssets.ps1`
- `eng\Invoke-LocalSplitRuntimePackage.ps1`
- `eng\Test-RuntimePackageReadiness.ps1`
- `pack/runtime-split/README.md`
- `docs/articles/zh-cn/source-build-cmake-windows-guide.md`
- `docs/articles/zh-cn/runtime-package-readiness-current-state.md`

文章骨架：

1. 绑定生成与 native ABI 的关系。
2. CMake preset、vendor root、runtime key 和 package asset layout。
3. readiness summary 的含义：native assets、wrapper surface、package layout。
4. readiness clean 不等于 runtime proof。
5. 修改 native 后必须补 CMake build 和 binding generator 验证。

示例命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Generate-Bindings.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BindingGeneratorOutputs.ps1
cmake --preset win-x64-trt11-cuda13-release
cmake --build --preset win-x64-trt11-cuda13-release --parallel
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
```

边界：source build 和 readiness 是发布准备证据，不是 clean consumer runtime proof。

## 5. CUDA / TensorRT / cuDNN 版本矩阵与 Runtime Package Key

目标读者：安装包、选择 runtime key 或排查 driver/runtime mismatch 的用户。

应引用的代码与 artifact：

- `eng\Resolve-RuntimeRoots.ps1`
- `eng\Test-ReleaseAssetCompleteness.ps1`
- `docs/articles/zh-cn/cuda-tensorrt-cudnn-version-matrix-guide.md`
- `docs/articles/zh-cn/runtime-packages.md`
- `artifacts/package-readiness/runtime-package-readiness-summary.md`

文章骨架：

1. runtime package key 的组成：RID、TRT line、CUDA、cuDNN。
2. Windows/Linux runtime asset 差异。
3. driver 支持的 CUDA runtime 与 packaged CUDA runtime 的关系。
4. CUDA error 35 的判断路径。
5. 选择 key 后如何跑 package consumer 和 external proof validator。

示例命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseAssetCompleteness.ps1
```

边界：版本矩阵帮助用户选包和排查环境，不是 runtime proof。只有同一主机、同一 runtime key、同一 package source 的真实 smoke log 和 strict validator 通过，才能进入 proof。

## 下一步

下一轮应优先把第 3 篇 package consumer proof 分层写成正式正文，并把第 1 篇 YoloVision 六任务证据链扩展为 owner asset input 示例。继续禁止把文章、matrix、validator、report、readiness 或 readonly summary 写成 proof。
