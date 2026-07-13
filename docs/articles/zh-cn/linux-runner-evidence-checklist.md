# Linux Runner Evidence Checklist

本文给 Linux x64 runner owner 使用，目标是把现有 handoff 文件变成真实 runner evidence。当前仓库中的 `linux-runner-evidence-template` 和 `linux-runner-issue-template` 仍是交接材料，不是已执行证明。

## 入口文件

代表 runtime key：

```text
linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
```

交接目录：

```text
artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
```

优先阅读：

1. `linux-handoff-index.md`
2. `linux-runner-evidence-template.md`
3. `linux-runner-issue-template.md`
4. `linux-package-consumer-plan.md`
5. `linux-runner-execution-status.json`
6. `linux-runner-evidence-record-template.md`
7. `linux-runner-evidence-validation.md`

## Runner 必填信息

Linux owner 需要补齐：

- runner OS、kernel、CPU architecture 和 runner labels。
- TensorRT root、CUDA root、cuDNN root。
- `Validate-LinuxRuntimeInputs.ps1` 输出。
- CMake configure/build 日志。
- `Collect-RuntimeAssets.ps1` 输出和 `.so` 清单。
- runtime `.nupkg` 路径和 SHA256。
- package consumer restore/build/native-copy 摘要。
- 如声明 runtime smoke 通过，还必须附 driver、CUDA runtime、GPU 访问和 smoke 输出。

## 推荐命令顺序

```bash
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
cmake --preset linux-x64-trt11-cuda13-release
cmake --build --preset linux-x64-trt11-cuda13-release --parallel
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
dotnet pack ./pack/runtime/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/JYPPX.TensorRT.CSharp.API.Runtime.linux-x64.ubuntu22.04.trt11.0.cuda13.2.cudnn9.22.csproj -c Release -o ./artifacts/runtime-nupkg
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
```

可用 GPU 环境再追加：

```bash
pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22 -RunSmoke
```

## 晋级规则

只有 Linux x64 runner 执行并附上真实输出后，才能把 Linux runtime line 从 handoff 推进到 runner proof。Windows 生成的 handoff、`template-only`、`dry-run-only` 都必须继续按交接状态记录。

`Test-LinuxRunnerEvidenceRecord.ps1` 是结构化校验入口。模板阶段的正确结果是 `validationState=template-only`、`isRealLinuxRunnerProof=false`、`canPromoteLinuxPackage=false`；只有真实 Linux runner 回填命令状态、日志路径、runtime nupkg、SHA256、native asset count 和 package consumer 状态后，validator 才能把记录提升。

真实 callback runtime proof 仍然单独判断，必须看到 `InvocationCount>0` 和 `IsRealCallbackRuntimeProof=True`。

## 第二批正文门禁

### 适用读者

本文适合在 Ubuntu runner、容器或远程 Linux 主机上收集 TensorRtSharp runtime proof 的维护者，也适合需要判断 Linux 证据能否关闭 release blocker 的发布负责人。

### 解决问题

Linux 证据容易混淆三类结果：workflow build 通过、runtime dependency probe 通过、真实 TensorRT/CUDA enqueue 通过。本文解决材料清单、晋级规则和 forbidden substitute 问题。

### 核心思路

核心思路是把 runner metadata、package source、命令输出、hash 和 validator 结果一起收集。单独的 TensorRtExec report、YoloVision matrix、OnnxToEngine report 或 local feed summary 都只能说明路径的一部分，不能证明完整 runtime proof。

### 操作路径

记录 OS、kernel、driver、CUDA、TensorRT、cuDNN、RID 和 package version；记录 clean consumer 项目来源和包来源；运行 dependency probe、minimal TensorRT smoke、可选样例 runner；保存 stdout/stderr summary、hash、exit code 和 skipped reason。

### 边界说明

build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof。Linux proof 必须来自真实兼容主机的执行记录；如果 driver/runtime 不兼容，应保持 blocked。

### 下一步

下一步应把 owner 提供的 Linux runner 结果导入 release proof validator。如果没有真实日志和 hash，就继续输出 repair pack 或 collection bundle。
