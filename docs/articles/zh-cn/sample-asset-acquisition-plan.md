# Sample Asset Acquisition Plan

本文说明 `Export-SampleAssetAcquisitionPlan.ps1` 的用途。它读取 `samples/assets/*.template.json`，生成 Classification/YoloVision 的资产获取计划，但不会下载模型、不会运行样例，也不会把候选资产提升为 sample smoke passed。

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleAssetAcquisitionPlan.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1
```

生成：

- `artifacts/user-acceptance/sample-asset-acquisition-plan.json`
- `artifacts/user-acceptance/sample-asset-acquisition-plan.md`
- `artifacts/user-acceptance/real-model-owner-handoff.json`
- `artifacts/user-acceptance/real-model-owner-handoff.md`

默认状态：

- `planState=owner-action-required`
- `performsDownload=false`
- `performsSampleRun=false`
- `canPromoteSamples=false`
- manifest `proofClassification=template-only`

## 计划内容

每个样例会生成以下步骤：

1. license and redistribution review。
2. model acquisition。
3. ONNX export or verification。
4. labels acquisition。
5. test image acquisition。
6. SHA256 verification。
7. build-only evidence classification。
8. evidence sidecar backfill。
9. sample execution。
10. sample run evidence record backfill。

每一步都包含 owner action、需要补齐的 evidence 和不能越界的 boundary。

获取计划会把 manifest 的 `proofClassification` 带入输出。`template-only`、`build-only`、`dependency-probe-only`、`synthetic-input-runtime` 都不能被写成真实模型 proof；`package-consumer-runtime` 也不会由 sample asset plan 产生，它属于 release proof record。

`evidence.evidenceSidecar` 用来把模型来源、模型 SHA256、许可证、输入资产、stdout/stderr summary 回填到 `TensorRtExec` / `OnnxToEngine` build report。它是证据连接件，不是发布证明；sidecar 不能把 build-only report 提升为 `package-consumer-runtime`。

真实运行后，再用 `Export-SampleRunEvidenceRecordTemplate.ps1` 生成 sample run evidence record 模板，并用 `Test-SampleRunEvidenceRecord.ps1` 校验。这个记录保存真实 runner 命令、运行日志、日志 SHA256、预期 evidence lines、stdout/stderr 摘要和 `canPromoteRealModelRuntime`。它只能把 Classification/YoloVision 的样例证据推进到 `real-model-runtime`，不能声明 `package-consumer-runtime`。

## 与 Catalog 的关系

`Export-UserAcceptanceSampleCatalog.ps1` 会读取 acquisition plan，并把 `sampleAssetAcquisitionPlanState` 写入 user acceptance catalog。即使 acquisition plan 已生成，Classification/YoloVision 仍保持 `asset-required`，直到真实模型、labels、图片、hash 和命令输出齐全。

catalog 还会读取 `sample-run-evidence-record-validation.json`，并把 `sampleRunEvidenceValidationState`、`sampleRunEvidenceCanPromoteRealModelRuntime` 和 runner proof classification 写入 Classification/YoloVision 的 `assetRequirement`。默认模板会显示 `owner-action-required` 和 `canPromoteRealModelRuntime=false`，这是一条明确边界：真实 runner 证据未回填时，样例不能被当作 smoke passed。

`Export-RealModelOwnerHandoff.ps1` 是 acquisition plan 和 sample run evidence record 之间的交接层。它不会下载模型，也不会运行样例；它把 manifest、sidecar、runner record、validator 和 release evidence bundle 串成一份 owner handoff，方便真实模型维护者逐项回填。

## 不允许的写法

- 不要把 `owner-action-required` 写成资产已获取。
- 不要把 acquisition plan 写成样例已运行。
- 不要把 build-only report 写成真实模型 runtime。
- 不要把 evidence sidecar 写成 package-consumer runtime proof。
- 不要把 sample run evidence record 写成 package-consumer runtime proof。
- 不要把 sample asset manifest 写成 package-consumer runtime proof。
- 不要把 real model owner handoff 写成样例已运行。
- 不要把 synthetic input 结果写成模型精度或检测质量。
