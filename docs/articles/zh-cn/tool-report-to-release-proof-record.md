# 从工具报告到 release proof record

TensorRtSharp4.0 的发布证据链里有很多文件：build report、evidence sidecar、asset manifest、sample-run-evidence、runbook、collection bundle、release checklist、external runtime proof record。它们看起来都像“证明材料”，但权限不同。

本文只回答一个问题：一份工具报告如何进入发布证据链，以及为什么它不能直接替代 `external-runtime-proof-record.json`。

## 一条完整证据链

典型外部模型路径如下：

1. `TensorRtExec` 生成 build report。
2. `OnnxEngineBuildEvidenceSidecar` 补模型来源、hash、许可证和输入资产摘要。
3. `Classification` 或 `YoloVision` 运行真实模型。
4. `sample-run-evidence-record` 记录 runner log、log SHA256、stdout/stderr 摘要和 expected evidence lines。
5. `Test-SampleAssetManifest.ps1` 校验 manifest 与 evidence record 是否一致。
6. `Export-UserAcceptanceSampleCatalog.ps1` 汇总样例验收状态。
7. `Export-ReleaseEvidenceBundle.ps1` 汇总发布证据。
8. `external-runtime-proof-record.json` 在兼容 host 上证明 package consumer runtime。

前七步都很有用，但只有最后一步能满足 package-consumer-runtime release gate。

## build report 的权限

TensorRtExec 的 report 记录模型转换和构建边界，例如：

```json
{
  "ProofClassification": "build-only",
  "BuildEvidenceOnly": true,
  "InferenceRan": false,
  "OutputMatch": false,
  "IsRuntimeExecutionProof": false,
  "IsRealModelRuntimeProof": false,
  "IsPackageConsumerRuntimeProof": false
}
```

这份报告适合定位 parser、shape profile、precision、workspace、plugin diagnostics 和 timing/cache 参数问题。它不能证明真实输入输出，也不能证明用户从 NuGet 包源消费项目后能够运行。

如果是 `ProofClassification=precheck`，权限更低：只证明参数归一化和报告生成，不读取 ONNX，不构建 engine。

## sidecar 的权限

sidecar 连接 build report 和模型资产。它可以写：

- `modelSource`
- `modelSha256`
- `modelLicense`
- `inputAssetName`
- `inputAssetSha256`
- `stdoutSummary`
- `stderrSummary`

sidecar 的价值是让 build report 更可审计。它不能把 `build-only` report 晋级成 `real-model-runtime`，更不能声明 `package-consumer-runtime`。如果 sidecar 中出现越级分类，工具和测试应把它当作诊断或拒绝项，而不是 proof。

## asset manifest 与 sample-run-evidence 的权限

Classification 和 YoloVision 的 asset manifest 记录真实模型所需资产。sample-run-evidence 记录真实 runner 日志。

它们可以共同证明 sample-level `real-model-runtime`：

```text
modelSha256 matches manifest
labelsSha256 matches manifest
inputAssetSha256 matches manifest
sampleRunLogSha256 matches actual log
expected evidence lines include Passed=True
canPromoteRealModelRuntime=true
```

这仍然不是 package consumer runtime proof。真实样例证明的是“这个样例用这组资产跑通了”，不证明“外部消费者安装包后在 clean 项目里跑通了”。

## runbook 与 collection bundle 的权限

`compatible-host-runtime-proof-runbook` 和 `compatible-host-runtime-proof-collection-bundle` 是 owner action 指南。它们说明在兼容主机上应该运行哪些命令、收集哪些日志、校验哪些字段。

它们不是运行结果。即使 collection bundle 包含正确命令，也不能把 `blocked-by-cuda-driver` 改成 smoke passed。当前主机如果因为 CUDA 13 runtime 与驱动不匹配而阻塞，正确状态就是继续阻塞，直到兼容 host 生成真实记录。

## external runtime proof record 的权限

`external-runtime-proof-record.json` 是 release proof 的关键输入。它应来自干净的 package consumer 项目，并至少证明：

- 使用真实包源，而不是 project reference。
- managed package 和 runtime package SHA256 对齐。
- native assets copy 成功。
- package consumer smoke 真正运行。
- `results.smokeStatus=passed`。
- `proofClassification=package-consumer-runtime`。
- smoke log 存在且 SHA256 匹配。
- validator 使用严格模式通过，例如 `-RequireExistingLog -FailOnNotProof`。

只有这类记录才能满足 package-consumer-runtime release gate。

## blocked-by-cuda-driver 的处理

`blocked-by-cuda-driver` is not smoke passed。它说明流程到达 CUDA runtime 边界，但当前 host 驱动不满足目标 CUDA runtime。它不是 API 未完成，也不是 proof 通过。

发布材料中可以写：

```text
当前主机 package-consumer runtime proof 被 blocked-by-cuda-driver 阻塞，需在兼容 CUDA host 上执行 external runtime proof。
```

不应写：

```text
runtime proof passed
package-consumer-runtime 已完成
```

除非真实 external runtime proof record 已经生成并通过 validator。

## 发布前命令

推荐检查：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked -WarnOnly
```

`Test-ReleaseCandidateReadiness.ps1` 在缺少真实 package consumer runtime proof 时返回 blocker 是正确行为。不要删除 blocker，也不要把 collection bundle、draft 或 template 改成 proof。

## 小结

工具报告是证据链的起点，不是终点。sidecar 让报告更可审计，sample-run-evidence 让真实模型样例更可信，runbook 和 collection bundle 指导 owner 在兼容主机上收集材料。最终 release gate 仍然只接受真实、可校验的 `external-runtime-proof-record.json`。
