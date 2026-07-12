[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-InputChecklistItem {
  param(
    [string]$Id,
    [string]$Title,
    [string]$ProofClass,
    [string]$InputArtifact,
    [string]$OutputArtifact,
    [string]$Validator,
    [string[]]$RequiredInputs,
    [string]$CurrentState,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    proofClass = $ProofClass
    inputArtifact = $InputArtifact
    outputArtifact = $OutputArtifact
    validator = $Validator
    requiredInputs = $RequiredInputs
    currentState = $CurrentState
    ownerAction = "owner-action-required"
    performsPublish = $false
    canPromoteProof = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

$compatibleHostPackage = Read-JsonOrNull "artifacts\final-release\compatible-host-proof-backfill-package.json"
$externalRuntimeInputTemplate = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record.input-template.json"
$postPublishInputDraft = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record.input-draft.json"
$sampleRunTemplate = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record.template.json"
$yoloVisionAssetsTemplate = Read-JsonOrNull "samples\assets\yolovision-assets.template.json"
$classificationAssetsTemplate = Read-JsonOrNull "samples\assets\classification-assets.template.json"
$ownerReleasePackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"

$compatibleHostPackageState = [string](Get-PropertyOrDefault -Object $compatibleHostPackage -Name "packageState" -DefaultValue "missing-compatible-host-proof-backfill-package")
$externalTemplateKind = [string](Get-PropertyOrDefault -Object $externalRuntimeInputTemplate -Name "recordKind" -DefaultValue "missing-external-runtime-proof-record-input-template")
$postPublishDraftKind = [string](Get-PropertyOrDefault -Object $postPublishInputDraft -Name "recordKind" -DefaultValue "missing-post-publish-verification-record-input-draft")
$sampleRunTemplateKind = [string](Get-PropertyOrDefault -Object $sampleRunTemplate -Name "recordKind" -DefaultValue "missing-sample-run-evidence-record-template")
$ownerPackageState = [string](Get-PropertyOrDefault -Object $ownerReleasePackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$preflightState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$preflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)
$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")

$packageConsumerRuntimeInputs = @(
  "managed nupkg SHA256 and runtime nupkg SHA256 from the exact release candidate artifacts",
  "runtimePackageKey=$RuntimePackageKey",
  "clean consumer identity outside the repository",
  "host owner/machine/os/gpu/driver/CUDA/TensorRT/cuDNN metadata",
  "restore/build/dependency probe/runtime smoke commands",
  "stdoutSummary and stderrSummary reviewed from the real smoke log",
  "smokeLogPath and smokeLogSha256 from an existing log",
  "no ProjectReference and no local feed substitution",
  "validator command: Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof"
)

$realModelRuntimeInputs = @(
  "Classification model path, labels path, input asset path, license, and SHA256 values",
  "YoloVision model path, labels path, input asset path, license, and SHA256 values",
  "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom task metadata",
  "det/cls/seg/obb/pose/sem and det、cls、seg、obb、pose、sem task coverage notes",
  "TensorRtExec sidecar and build report for each real model",
  "sample runner command and sample runner log",
  "stdoutSummary and stderrSummary reviewed from the sample run",
  "sample-run-evidence record with proofClassification=real-model-runtime",
  "sample proof only promotes to real-model-runtime, never package-consumer-runtime"
)

$postPublishInputs = @(
  "real channel package URL and selected release channel",
  "downloaded managed/runtime nupkg SHA256 values",
  "clean consumer root outside the repository",
  "no ProjectReference",
  "native assets listing from the downloaded packages",
  "dependency probe log and runtime smoke log",
  "stdoutSummary and stderrSummary reviewed from real post-publish logs",
  "validator command: Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
)

$inputChecklists = @(
  New-InputChecklistItem `
    -Id "package-consumer-runtime-input" `
    -Title "Package consumer runtime proof input" `
    -ProofClass "package-consumer-runtime" `
    -InputArtifact "artifacts/final-release/external-runtime-proof-record.input-template.json" `
    -OutputArtifact "artifacts/final-release/external-runtime-proof-record.json" `
    -Validator "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredInputs $packageConsumerRuntimeInputs `
    -CurrentState "templateKind=$externalTemplateKind; compatibleHostPackageState=$compatibleHostPackageState; preflightState=$preflightState; failedItemCount=$preflightFailedItemCount" `
    -Boundary "Only a clean package consumer runtime smoke on a compatible host can produce package-consumer-runtime proof."

  New-InputChecklistItem `
    -Id "real-model-runtime-input" `
    -Title "Classification and YoloVision real model runtime proof input" `
    -ProofClass "real-model-runtime" `
    -InputArtifact "samples/assets/classification-assets.template.json; samples/assets/yolovision-assets.template.json; artifacts/user-acceptance/sample-run-evidence-record.template.json" `
    -OutputArtifact "artifacts/user-acceptance/sample-run-evidence-record.json" `
    -Validator "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -RequiredInputs $realModelRuntimeInputs `
    -CurrentState "sampleRunTemplateKind=$sampleRunTemplateKind; classificationTemplatePresent=$($null -ne $classificationAssetsTemplate); yoloVisionTemplatePresent=$($null -ne $yoloVisionAssetsTemplate)" `
    -Boundary "Classification/YoloVision sample evidence can promote only to real-model-runtime and cannot substitute package-consumer-runtime."

  New-InputChecklistItem `
    -Id "post-publish-verification-input" `
    -Title "Post-publish verification proof input" `
    -ProofClass "post-publish-package-consumer-runtime" `
    -InputArtifact "artifacts/final-release/post-publish-verification-record.input-draft.json" `
    -OutputArtifact "artifacts/final-release/post-publish-verification-record.json" `
    -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredInputs $postPublishInputs `
    -CurrentState "draftKind=$postPublishDraftKind; ownerPackageState=$ownerPackageState; releaseEvidenceState=$releaseEvidenceState" `
    -Boundary "Post-publish verification requires a real channel package and cannot be replaced by local feed, ProjectReference, helper scan, or draft."
)

$nonSubstitutes = @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "build-only",
  "parse-only",
  "sidecar-only",
  "DependencyProbe",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "Skipped=True",
  "bridge-only package consumer log",
  "bridge-only wrapper surface",
  "WrapperSurfaceEvidenceKind=compile-surface-proof",
  "IsRuntimeExecutionProof=False",
  "mismatched log SHA256",
  "Parser/ParserRefitter diagnostic snapshots",
  "copied managed diagnostic snapshot",
  "Windows handoff for Linux proof",
  "owner guidance without real logs",
  "owner-action-required without validator pass"
)

$copyableExecutionOrder = @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostProofBackfillPackage.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1",
  "Fill artifacts/final-release/external-runtime-proof-record.json with real package-consumer-runtime logs and hashes.",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
  "Fill Classification/YoloVision real model asset manifests with model/labels/input/license/SHA256.",
  "Run TensorRtExec build report and sample runner on real model assets.",
  "Fill artifacts/user-acceptance/sample-run-evidence-record.json with real-model-runtime logs.",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
  "After owner publishes to a real channel, fill artifacts/final-release/post-publish-verification-record.json.",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "real-model-and-package-proof-input-package"
  packageState = "owner-action-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  requiresHumanOwner = $true
  requiresCompatibleHost = $true
  compatibleHostProofBackfillPackageState = $compatibleHostPackageState
  ownerReleaseExecutionPackageState = $ownerPackageState
  releaseEvidenceBundleState = $releaseEvidenceState
  preflightState = $preflightState
  preflightFailedItemCount = $preflightFailedItemCount
  packageConsumerRuntimeInputChecklist = $inputChecklists[0]
  realModelRuntimeInputChecklist = $inputChecklists[1]
  postPublishVerificationInputChecklist = $inputChecklists[2]
  inputChecklists = $inputChecklists
  nonSubstituteProofKinds = $nonSubstitutes
  copyableExecutionOrder = $copyableExecutionOrder
  sourceArtifacts = @(
    "artifacts/final-release/compatible-host-proof-backfill-package.json",
    "artifacts/final-release/external-runtime-proof-record.input-template.json",
    "artifacts/final-release/post-publish-verification-record.input-draft.json",
    "artifacts/user-acceptance/sample-run-evidence-record.template.json",
    "samples/assets/yolovision-assets.template.json",
    "samples/assets/classification-assets.template.json",
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  safetyNotes = @(
    "This package is owner guidance and input material only.",
    "performsPublish=false, canPublishPublicly=false, and canCloseReleaseIssue=false are fixed for this script.",
    "package-consumer-runtime requires a clean package consumer runtime smoke and Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof.",
    "real-model-runtime requires real Classification/YoloVision assets, logs, hashes, licenses, and sample-run-evidence validation.",
    "post-publish verification requires a real channel package and Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof.",
    "blocked-by-cuda-driver remains an environment blocker and must not be written as smoke passed.",
    "template, draft, runbook, collection package, input package, local feed, ProjectReference, build-only, parse-only, sidecar-only, bridge-only, Skipped=True, mismatched log SHA256, copied diagnostic snapshots, and Windows handoff for Linux proof cannot substitute release proof."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "real-model-and-package-proof-input-package.json"
$markdownPath = Join-Path $artifactRoot "real-model-and-package-proof-input-package.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$checklistRows = $inputChecklists | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.proofClass)`` | ``$($_.validator)`` | $($_.currentState) | $($_.boundary) |"
}

$packageConsumerLines = $packageConsumerRuntimeInputs | ForEach-Object { "- ``$_``" }
$realModelLines = $realModelRuntimeInputs | ForEach-Object { "- ``$_``" }
$postPublishLines = $postPublishInputs | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $nonSubstitutes | ForEach-Object { "- ``$_``" }
$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }
$executionLines = for ($i = 0; $i -lt $copyableExecutionOrder.Count; $i++) {
  "$($i + 1). ``$($copyableExecutionOrder[$i])``"
}
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }

$markdown = @"
# Real Model And Package Proof Input Package

生成时间：$($record.generatedAtUtc)

## 总结

该输入包聚合真实模型资产、package-consumer-runtime、Linux compatible host 与 post-publish verification 的 owner 填写材料。它只生成 copyable input checklist，不执行发布、不上传包、不伪造 proof。``recordKind=real-model-and-package-proof-input-package``，``packageState=owner-action-required``，``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前状态

| 项目 | 当前值 |
|---|---|
| packageState | ``owner-action-required`` |
| compatibleHostProofBackfillPackageState | ``$compatibleHostPackageState`` |
| ownerReleaseExecutionPackageState | ``$ownerPackageState`` |
| releaseEvidenceBundleState | ``$releaseEvidenceState`` |
| preflightState | ``$preflightState`` |
| preflightFailedItemCount | ``$preflightFailedItemCount`` |
| runtimePackageKey | ``$RuntimePackageKey`` |
| linuxRuntimePackageKey | ``$LinuxRuntimePackageKey`` |

## 输入清单总表

| ID | Proof class | Validator | 当前状态 | 边界 |
|---|---|---|---|---|
$($checklistRows -join "`r`n")

## Package Consumer Runtime 输入

$($packageConsumerLines -join "`r`n")

该路径只能由 clean consumer runtime smoke 产生 package-consumer-runtime proof。``ProjectReference``、local feed、helper scan、DependencyProbe、build-only、parse-only、sidecar-only 和 ``blocked-by-cuda-driver`` 都不能替代。

## Real Model Runtime 输入

$($realModelLines -join "`r`n")

YoloVision 范围固定为 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det/cls/seg/obb/pose/sem（det、cls、seg、obb、pose、sem）。Classification/YoloVision 样例 proof 只能晋级 real-model-runtime，不能晋级 package-consumer-runtime。

## Post-Publish Verification 输入

$($postPublishLines -join "`r`n")

Post-publish verification 必须来自真实发布渠道、下载后的 nupkg SHA256、clean consumer restore/build/probe/smoke 日志和 ``Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof``。draft、template、runbook、collection package 不是 proof。

## Copyable Execution Order

$($executionLines -join "`r`n")

## 不可替代材料

$($nonSubstituteLines -join "`r`n")

## 来源材料

$($sourceLines -join "`r`n")

## Safety Notes

$($safetyLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Real model and package proof input package written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackageState=owner-action-required"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
