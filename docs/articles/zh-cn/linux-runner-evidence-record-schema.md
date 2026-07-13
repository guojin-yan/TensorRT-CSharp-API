# Linux Runner Evidence Record Schema

本文说明 `linux-runner-evidence-record-template` 的字段和晋级规则。它是 Linux runner owner 回填真实证据的结构化模板，不是 Windows 主机生成的 proof。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-LinuxRunnerEvidenceRecordTemplate.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22
```

生成文件：

- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-record-template.json`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-record-template.md`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-validation.json`
- `artifacts/linux-dry-run/linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22/linux-runner-evidence-validation.md`

## 核心字段

- `recordKind=linux-runner-evidence-record-template`
- `templateOnly=true`
- `recordState=template-only`
- `executionState=pending-linux-runner-execution`
- `isRealLinuxRunnerProof=false`
- `canPromoteLinuxPackage=false`

模板生成后立即运行 validator，预期结果必须仍是：

- `validationState=template-only`
- `isRealLinuxRunnerProof=false`
- `canPromoteLinuxPackage=false`

这说明 validator 没有把 Windows 生成的交接材料误判成真实 Linux runner proof。

Linux runner owner 需要填写：

- 将真实记录改为 `recordKind=linux-runner-evidence-record`。
- 将 `templateOnly=false`。
- `runnerOwner`。
- `evidenceRationale`。
- runner OS、kernel、architecture、runner labels。
- TensorRT root、CUDA root、cuDNN root。
- 每条命令的 start/finish time、exit code、log path。
- runtime `.nupkg` path 和 SHA256。
- native asset copied/missing count。
- package consumer status。
- optional smoke status。

## 晋级条件

只有满足以下条件，才能把 `isRealLinuxRunnerProof` 改为 `true`：

1. runner 是 Linux x64。
2. `recordKind=linux-runner-evidence-record` 且 `templateOnly=false`。
3. `runnerOwner` 和 `evidenceRationale` 已填写。
4. `Validate-LinuxRuntimeInputs.ps1` 成功解析依赖路径。
5. CMake configure/build 成功。
6. runtime assets 收集成功，missing count 为 0。
7. runtime `.nupkg` 生成并记录 SHA256。
8. package consumer restore/build/native-copy 成功且无 `ProjectReference`。
9. 所有日志路径可追溯。

runtime smoke passed 还需要兼容 NVIDIA driver、CUDA runtime 和 GPU 访问。真实 callback proof 继续独立判断，必须看到 `InvocationCount>0` 和 `IsRealCallbackRuntimeProof=True`。

## Validator 晋级规则

`Test-LinuxRunnerEvidenceRecord.ps1` 会检查：

- runtime key 必须匹配。
- runner 必须声明 Linux x64。
- 必需命令必须存在、非 pending、exit code 为 0。
- 命令必须带 log path 或等价证据引用。
- runtime `.nupkg` path 和 SHA256 必须存在。
- native asset missing count 必须为 0。
- package consumer restore/build/native-copy 状态必须 ready。

只有 validator 输出 `validationState=real-linux-runner-proof`、`isRealLinuxRunnerProof=true`、`canPromoteLinuxPackage=true` 时，才能把 Linux package line 从 handoff 提升为真实 runner proof。
