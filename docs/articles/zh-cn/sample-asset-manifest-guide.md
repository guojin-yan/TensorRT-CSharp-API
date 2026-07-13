# Sample Asset Manifest Guide

本文说明 `samples/assets/*.template.json` 的用途。它们把 Classification 和 YoloVision 所需的模型、labels、图片、shape、layout、预处理、后处理和证据行记录成可审计结构，避免文章或 release checklist 把候选资产写成已跑通。

## 当前模板

- `samples/assets/classification-assets.template.json`
- `samples/assets/yolovision-assets.template.json`

默认状态：

- `status=candidate-not-downloaded`
- `proofClassification=template-only`
- `isSmokePassed=false`
- `isRedistributableInRepository=false`

这表示资产还没有下载、校验、授权复核或真实执行。它不是 sample smoke passed。

## Proof Classification

sample asset manifest 使用和 release proof 相同的证据词汇，但晋级边界不同：

| 分类 | 在 sample asset manifest 中的含义 | 是否可写成 sample smoke passed |
| --- | --- | --- |
| `template-only` | 只有模板和 owner 待填字段。 | 否 |
| `build-only` | 只有 TensorRtExec / OnnxToEngine build report。 | 否 |
| `dependency-probe-only` | 只有依赖探测或加载诊断。 | 否 |
| `synthetic-input-runtime` | 使用 synthetic input 跑通样例管线。 | 否 |
| `real-model-runtime` | 使用真实模型、labels、图片和日志跑通样例。 | 是，仍需 `status=smoke-passed` 和 `isSmokePassed=true` |
| `package-consumer-runtime` | 干净 NuGet/runtime package consumer proof。 | 不允许出现在 sample asset manifest |

`package-consumer-runtime` 属于 release proof record，不属于 Classification/YoloVision 的资产 manifest。真实模型样例最多晋级到 `real-model-runtime`，并且必须补齐模型、labels、图片和运行日志 SHA256，以及 stdout/stderr 摘要。

## 审计命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1
```

生成：

- `artifacts/user-acceptance/sample-asset-manifest-audit.json`
- `artifacts/user-acceptance/sample-asset-manifest-audit.md`

审计内容包括：

- schema version。
- status 枚举。
- SHA256 格式。
- model source URL 和 license。
- input shape、layout、dtype。
- run command。
- proof classification。
- sampleName 到 `samples/<sampleName>/<sampleName>.csproj` 的项目身份映射。
- evidence sidecar。
- sample run evidence record。
- sample run evidence validation。
- build-only command。
- stdout/stderr summary。
- smoke 状态是否与 `isSmokePassed` 一致。

## 与 User Acceptance Catalog 的关系

`Export-UserAcceptanceSampleCatalog.ps1` 会读取 asset manifest audit，并把 Classification/YoloVision 的 asset 状态写入 catalog。即使 asset manifest audit 通过，Classification/YoloVision 在真实模型、labels、图片和命令输出齐全前仍保持 `asset-required`。

`Export-SampleAssetAcquisitionPlan.ps1` 会把 manifest 转成 owner 可执行的获取计划。计划默认 `owner-action-required`，不会下载模型，也不会运行样例。

`Test-OnnxEngineBuildEvidenceSidecar.ps1` 用来单独审计 `evidence.evidenceSidecar`。默认扫描 `samples/assets/*.template.json` 和 `samples/assets/*example.json`：模板中还没有创建真实 sidecar 时，结果是 `owner-action-required`，不是 error。真实回填后，它会检查 `proofClassification`、模型/input SHA256、许可证、stdout/stderr summary，并确保 sidecar 不能把 build report 晋级为 `package-consumer-runtime`。

`Export-SampleRunEvidenceRecordTemplate.ps1` 与 `Test-SampleRunEvidenceRecord.ps1` 则用于真实 sample runner 证据。sidecar 负责连接 build report 和模型资产；sample run evidence record 负责连接真实 `Classification` / `YoloVision` 命令、运行日志、日志 SHA256、stdout/stderr 摘要和 `Passed=True` 证据行。模板状态同样是 `owner-action-required`，不会把未实跑样例写成通过。

`Test-SampleAssetManifest.ps1` 现在也会读取 manifest 中的 `evidence.sampleRunEvidenceRecord` 和 `evidence.sampleRunEvidenceValidation`。如果 record 文件不存在，模板保持 `owner-action-required`，不报 error；如果 record 文件存在，会 cross-check `sampleName`、`modelSha256`、`labelsSha256`、`inputAssetSha256`，并拒绝 `package-consumer-runtime`。这一步的目标是防止真实 runner 日志、manifest 和 sidecar 各说各话。

manifest 的 `sampleName` 还会被当作样例项目身份校验：`Classification` 必须对应 `samples/Classification/Classification.csproj`，`YoloVision` 必须对应 `samples/YoloVision/YoloVision.csproj`。这条规则用于防止样例改名后旧项目名、旧构建产物或错误 manifest 回流到发布证据链。

## 晋级规则

只有满足以下条件，才能把 manifest 从候选推进：

1. 模型、labels、图片来源和许可证已记录。
2. SHA256 已记录并匹配。
3. 输入输出 tensor 名、shape、layout、dtype 已确认。
4. 预处理和后处理规则已确认。
5. 真实命令输出已保存。
6. `Classification Passed=True` 或 `YoloVision Passed=True` 来自真实资产路径，而不是 synthetic input。
7. `proofClassification=real-model-runtime`。
8. stdout/stderr 摘要和运行日志 SHA256 已记录。
9. `evidence.evidenceSidecar` 已记录，并且 sidecar 中的 model hash、license、input hash、stdout/stderr summary 与 manifest 一致。
10. sample run evidence record 已从 template-only 改为真实记录，并且 `canPromoteRealModelRuntime=true` 只在完整证据齐全后设置。
11. manifest 的 `sampleName` 与样例目录、`.csproj` 文件名一致。

## 不允许的写法

- 不要把 `candidate-not-downloaded` 写成 sample ready。
- 不要把 `isSmokePassed=false` 写成已通过。
- 不要把 `build-only` 写成真实模型 runtime。
- 不要在 sample asset manifest 中使用 `package-consumer-runtime`。
- 不要让 evidence sidecar 把 TensorRtExec / OnnxToEngine build report 提升为 `package-consumer-runtime`。
- 不要让 sample run evidence record 声明 `package-consumer-runtime`；它最多只能证明真实模型样例 runtime。
- 不要把 synthetic input pipeline evidence 写成模型精度或检测质量证明。
