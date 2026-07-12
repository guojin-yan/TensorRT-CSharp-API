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

function New-ExecutionItem {
  param(
    [string]$GapId,
    [string]$ProofClass,
    [string]$Title,
    [string]$CurrentState,
    [string[]]$RequiredInputs,
    [string[]]$CopyCommands,
    [string[]]$RunCommands,
    [string[]]$ValidateCommands,
    [string[]]$ExpectedArtifacts,
    [string]$PromotionBoundary,
    [string[]]$RelatedGuidance
  )

  [pscustomobject]@{
    gapId = $GapId
    proofClass = $ProofClass
    title = $Title
    currentState = $CurrentState
    requiredInputs = $RequiredInputs
    copyCommands = $CopyCommands
    runCommands = $RunCommands
    validateCommands = $ValidateCommands
    expectedArtifacts = $ExpectedArtifacts
    promotionBoundary = $PromotionBoundary
    relatedGuidance = $RelatedGuidance
    nonSubstitutes = $script:NonSubstituteProofKinds
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    ownerAction = "owner-action-required"
  }
}

$releaseCloseGapDashboard = Read-JsonOrNull "artifacts\final-release\release-close-gap-dashboard.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$releasePromotionIssue = Read-JsonOrNull "artifacts\final-release\release-promotion-issue-record.json"
$realModelAndPackageProofInputPackage = Read-JsonOrNull "artifacts\final-release\real-model-and-package-proof-input-package.json"
$compatibleHostBackfillPackage = Read-JsonOrNull "artifacts\final-release\compatible-host-proof-backfill-package.json"
$compatibleHostCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$externalRuntimeCollectionPackage = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-collection-package.json"
$postPublishCollectionPackage = Read-JsonOrNull "artifacts\final-release\post-publish-verification-collection-package.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$linuxRunnerValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$sampleAssetAudit = Read-JsonOrNull "artifacts\user-acceptance\sample-asset-manifest-audit.json"

$dashboardState = [string](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "dashboardState" -DefaultValue "missing-release-close-gap-dashboard")
$dashboardGapCount = [int](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "gapCount" -DefaultValue -1)
$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$promotionState = [string](Get-PropertyOrDefault -Object $releasePromotionIssue -Name "promotionState" -DefaultValue "missing-release-promotion-issue-record")
$inputPackageState = [string](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "packageState" -DefaultValue "missing-real-model-and-package-proof-input-package")
$compatibleHostBackfillState = [string](Get-PropertyOrDefault -Object $compatibleHostBackfillPackage -Name "packageState" -DefaultValue "missing-compatible-host-proof-backfill-package")
$compatibleHostCollectionState = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "collectionState" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")
$externalRuntimeCollectionState = [string](Get-PropertyOrDefault -Object $externalRuntimeCollectionPackage -Name "packageState" -DefaultValue "missing-external-runtime-proof-collection-package")
$postPublishCollectionState = [string](Get-PropertyOrDefault -Object $postPublishCollectionPackage -Name "packageState" -DefaultValue "missing-post-publish-verification-collection-package")
$ownerReleaseExecutionState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$oneScreenReleaseHoldChecklist = @(Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "oneScreenReleaseHoldChecklist" -DefaultValue @())
$oneScreenReleaseHoldChecklistCount = if ($oneScreenReleaseHoldChecklist.Count -gt 0) {
  $oneScreenReleaseHoldChecklist.Count
}
else {
  [int](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "oneScreenReleaseHoldChecklistCount" -DefaultValue 0)
}
$linuxValidationState = [string](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxProof = [bool](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleValidationState = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleCanPromote = [bool](Get-PropertyOrDefault -Object $sampleRunValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$sampleAssetErrorCount = [int](Get-PropertyOrDefault -Object $sampleAssetAudit -Name "errorCount" -DefaultValue -1)
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofState" -DefaultValue "missing-external-runtime-proof-validation")
$externalCanPromote = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPromoteRuntimeProof" -DefaultValue $false)
$postPublishState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "isPostPublishVerificationProof" -DefaultValue $false)
$ownerApprovalState = [string](Get-PropertyOrDefault -Object $releasePromotionIssue -Name "ownerApprovalInputValidationStatus" -DefaultValue "missing-owner-authorization")

$script:NonSubstituteProofKinds = @(
  "helper",
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "DependencyProbe",
  "dependency-probe-only",
  "Skipped=True",
  "blocked-by-cuda-driver",
  "bridge-only package consumer log",
  "bridge-only wrapper surface",
  "WrapperSurfaceEvidenceKind=compile-surface-proof",
  "IsRuntimeExecutionProof=False",
  "mismatched log SHA256",
  "Parser/ParserRefitter diagnostic snapshots",
  "copied managed diagnostic snapshot",
  "build-only",
  "parse-only",
  "sidecar-only",
  "Windows handoff for Linux proof",
  "owner-action-required without validator pass"
)

$executionItems = @(
  New-ExecutionItem `
    -GapId "owner-authorization" `
    -ProofClass "owner-authorization" `
    -Title "Owner authorization and manual command materialization" `
    -CurrentState "ownerApprovalInputValidationStatus=$ownerApprovalState; ownerReleaseExecutionPackageState=$ownerReleaseExecutionState; promotionState=$promotionState" `
    -RequiredInputs @(
      "explicit non-template owner approval record",
      "release channel choice",
      "NVIDIA redistribution disposition",
      "owner-reviewed package identities and SHA256 hashes",
      "manual publish command materialization outside automation"
    ) `
    -CopyCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerReleaseExecutionPackage.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1"
    ) `
    -RunCommands @(
      "owner manually reviews artifacts/final-release/owner-release-execution-package.md",
      "owner fills release-owner-approval-input.json outside automation"
    ) `
    -ValidateCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/release-owner-approval-input-validation.json",
      "artifacts/final-release/owner-authorized-publish-command-plan-validation.json"
    ) `
    -PromotionBoundary "Owner execution materials do not publish packages and do not authorize publication by themselves." `
    -RelatedGuidance @(
      "artifacts/final-release/owner-release-execution-package.md",
      "docs/articles/zh-cn/owner-release-execution-package.md"
    )

  New-ExecutionItem `
    -GapId "package-consumer-runtime" `
    -ProofClass "package-consumer-runtime" `
    -Title "Clean package consumer runtime proof on compatible host" `
    -CurrentState "externalRuntimeProofState=$externalRuntimeProofState; canPromoteRuntimeProof=$externalCanPromote; compatibleHostCollectionState=$compatibleHostCollectionState; externalRuntimeCollectionState=$externalRuntimeCollectionState" `
    -RequiredInputs @(
      "managed/runtime nupkg SHA256 from exact release candidate artifacts",
      "runtimePackageKey=$RuntimePackageKey",
      "clean consumer project outside repository",
      "host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
      "restore/build/dependency probe/runtime smoke logs",
      "stdoutSummary and stderrSummary from reviewed real log",
      "smokeLogPath and smokeLogSha256 from an existing log",
      "no ProjectReference"
    ) `
    -CopyCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofCollectionBundle.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofCollectionPackage.ps1"
    ) `
    -RunCommands @(
      "run clean package consumer restore/build/smoke on a compatible CUDA/TensorRT host",
      "fill artifacts/final-release/external-runtime-proof-record.json from the real host outputs"
    ) `
    -ValidateCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/external-runtime-proof-record.json",
      "artifacts/final-release/external-runtime-proof-validation.json"
    ) `
    -PromotionBoundary "Only a validator-passing real external-runtime-proof-record.json can promote package-consumer-runtime; bridge-only logs, Skipped=True, dependency-probe-only, WrapperSurfaceEvidenceKind=compile-surface-proof, IsRuntimeExecutionProof=False, and copied Parser/ParserRefitter diagnostic snapshots are non-proof." `
    -RelatedGuidance @(
      "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md",
      "artifacts/final-release/external-runtime-proof-collection-package.md",
      "docs/articles/zh-cn/package-consumer-runtime-proof-playbook.md"
    )

  New-ExecutionItem `
    -GapId "linux-runner-proof" `
    -ProofClass "linux-runner-proof" `
    -Title "Linux runner proof on real Linux x64 host" `
    -CurrentState "linuxRuntimePackageKey=$LinuxRuntimePackageKey; validationState=$linuxValidationState; isRealLinuxRunnerProof=$linuxProof" `
    -RequiredInputs @(
      "real Linux x64 runner host metadata",
      "runtime package key $LinuxRuntimePackageKey",
      "CUDA/TensorRT/cuDNN versions from Linux runner host",
      "command log and validator output",
      "Linux runner evidence record captured from the Linux host"
    ) `
    -CopyCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-LinuxRunnerEvidenceRecordTemplate.ps1 -RuntimePackageKey $LinuxRuntimePackageKey"
    ) `
    -RunCommands @(
      "run the Linux runner on a real Linux x64 CUDA/TensorRT host",
      "copy the filled Linux runner evidence record back into artifacts/linux-dry-run/$LinuxRuntimePackageKey"
    ) `
    -ValidateCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey"
    ) `
    -ExpectedArtifacts @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json",
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
    ) `
    -PromotionBoundary "Windows handoff, template-only records, and dry-run guidance are not Linux runner proof." `
    -RelatedGuidance @(
      "docs/articles/zh-cn/linux-runner-evidence-record-schema.md",
      "docs/articles/zh-cn/linux-runner-evidence-checklist.md"
    )

  New-ExecutionItem `
    -GapId "real-model-runtime" `
    -ProofClass "real-model-runtime" `
    -Title "Classification and YoloVision real model runtime proof" `
    -CurrentState "sampleRunValidationState=$sampleValidationState; canPromoteRealModelRuntime=$sampleCanPromote; sampleAssetErrorCount=$sampleAssetErrorCount; inputPackageState=$inputPackageState" `
    -RequiredInputs @(
      "Classification model/labels/input/license/SHA256",
      "YoloVision model/labels/input/license/SHA256",
      "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom metadata",
      "det/cls/seg/obb/pose/sem and det、cls、seg、obb、pose、sem task notes",
      "TensorRtExec sidecar and build report",
      "sample runner command and real sample runner log",
      "stdoutSummary and stderrSummary from reviewed real sample log",
      "sample-run-evidence record with proofClassification=real-model-runtime"
    ) `
    -CopyCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealModelOwnerHandoff.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleRunEvidenceRecordTemplate.ps1"
    ) `
    -RunCommands @(
      "run Classification sample with real owner-provided assets",
      "run YoloVision sample with real YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom assets for det/cls/seg/obb/pose/sem coverage where applicable"
    ) `
    -ValidateCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    ) `
    -ExpectedArtifacts @(
      "artifacts/user-acceptance/sample-asset-manifest-audit.json",
      "artifacts/user-acceptance/sample-run-evidence-record.json",
      "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
    ) `
    -PromotionBoundary "Classification/YoloVision evidence can promote only real-model-runtime and never package-consumer-runtime." `
    -RelatedGuidance @(
      "artifacts/final-release/real-model-and-package-proof-input-package.md",
      "docs/articles/zh-cn/real-model-evidence-backfill-playbook.md",
      "docs/articles/zh-cn/yolovision-all-task-overview.md"
    )

  New-ExecutionItem `
    -GapId "post-publish-verification" `
    -ProofClass "post-publish verification" `
    -Title "Post-publish clean consumer verification after real channel publication" `
    -CurrentState "postPublishVerificationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof; postPublishCollectionState=$postPublishCollectionState" `
    -RequiredInputs @(
      "real release channel package URL and selected channel",
      "downloaded managed/runtime nupkg SHA256",
      "clean consumer root outside repository",
      "no ProjectReference",
      "native assets listing from downloaded packages",
      "dependency probe log",
      "runtime smoke log",
      "stdoutSummary and stderrSummary from reviewed real post-publish logs"
    ) `
    -CopyCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationCollectionPackage.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordInputDraft.ps1"
    ) `
    -RunCommands @(
      "after owner-approved real publication, restore/build/probe/smoke a clean external consumer from the real channel package",
      "fill post-publish-verification-record.json with downloaded hashes and real log SHA256 values"
    ) `
    -ValidateCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/post-publish-verification-record.json",
      "artifacts/final-release/post-publish-verification-validation.json"
    ) `
    -PromotionBoundary "Post-publish proof requires real channel packages; local feed, draft, helper scan, bridge-only package consumer logs, dependency-probe-only logs, and collection package cannot close the release issue." `
    -RelatedGuidance @(
      "artifacts/final-release/post-publish-verification-collection-package.md",
      "docs/articles/zh-cn/post-publish-verification-proof-playbook.md"
    )
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "compatible-host-proof-execution-pack"
  packageState = "owner-action-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  requiresCompatibleHost = $true
  requiresHumanOwner = $true
  releaseCloseGapDashboardState = $dashboardState
  releaseCloseGapDashboardGapCount = $dashboardGapCount
  releaseEvidenceBundleState = $releaseEvidenceState
  releasePromotionIssueState = $promotionState
  compatibleHostProofBackfillPackageState = $compatibleHostBackfillState
  compatibleHostRuntimeProofCollectionBundleState = $compatibleHostCollectionState
  externalRuntimeProofCollectionPackageState = $externalRuntimeCollectionState
  postPublishVerificationCollectionPackageState = $postPublishCollectionState
  realModelAndPackageProofInputPackageState = $inputPackageState
  ownerReleaseExecutionPackageState = $ownerReleaseExecutionState
  oneScreenReleaseHoldChecklist = @($oneScreenReleaseHoldChecklist)
  oneScreenReleaseHoldChecklistCount = $oneScreenReleaseHoldChecklistCount
  blockerCount = @($executionItems).Count
  executionItems = $executionItems
  nonSubstituteProofKinds = $script:NonSubstituteProofKinds
  sourceArtifacts = @(
    "artifacts/final-release/release-close-gap-dashboard.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-promotion-issue-record.json",
    "artifacts/final-release/real-model-and-package-proof-input-package.json",
    "artifacts/final-release/compatible-host-proof-backfill-package.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/external-runtime-proof-collection-package.json",
    "artifacts/final-release/post-publish-verification-collection-package.json",
    "artifacts/final-release/owner-release-execution-package.json"
  )
  safetyNotes = @(
    "This execution pack is owner guidance only and does not publish packages.",
    "performsPublish=false, canPublishPublicly=false, and canCloseReleaseIssue=false are fixed for this pack.",
    "Only validator-passing real proof records with existing logs and matching SHA256 values can promote proof classes.",
    "Package-consumer-runtime requires Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof.",
    "Post-publish verification requires Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof after real publication.",
    "Bridge-only package consumer logs, bridge-only wrapper surface, Skipped=True, dependency-probe-only, WrapperSurfaceEvidenceKind=compile-surface-proof, IsRuntimeExecutionProof=False, mismatched log SHA256, and copied Parser/ParserRefitter diagnostic snapshots cannot promote runtime proof.",
    "Real-model-runtime requires Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog and real Classification/YoloVision assets.",
    "Linux runner proof requires Test-LinuxRunnerEvidenceRecord.ps1 against a real Linux x64 runner record.",
    "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom and det、cls、seg、obb、pose、sem remain real asset requirements, not built-in proof."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "compatible-host-proof-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "compatible-host-proof-execution-pack.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$executionRows = $executionItems | ForEach-Object {
  $validators = ($_.validateCommands -join "<br/>").Replace("|", "\|")
  "| ``$($_.gapId)`` | ``$($_.proofClass)`` | $($_.currentState.Replace("|", "\|")) | $validators | $($_.promotionBoundary.Replace("|", "\|")) |"
}
$oneScreenReleaseHoldRows = $oneScreenReleaseHoldChecklist | ForEach-Object {
  $item = $_
  $id = ([string](Get-PropertyOrDefault -Object $item -Name "id" -DefaultValue "")).Replace("|", "\|")
  $blocker = ([string](Get-PropertyOrDefault -Object $item -Name "ownerVisibleBlocker" -DefaultValue "")).Replace("|", "\|")
  $currentState = ([string](Get-PropertyOrDefault -Object $item -Name "currentState" -DefaultValue "")).Replace("|", "\|")
  $nextAction = ([string](Get-PropertyOrDefault -Object $item -Name "ownerNextAction" -DefaultValue "")).Replace("|", "\|")
  $validator = ([string](Get-PropertyOrDefault -Object $item -Name "validatorCommand" -DefaultValue "")).Replace("|", "\|")
  "| ``$id`` | $blocker | $currentState | $nextAction | ``$validator`` |"
}
$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $script:NonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }

$markdown = @"
# Compatible Host Proof Execution Pack

生成时间：$($record.generatedAtUtc)

## 总结

该 execution pack 把 release close dashboard 中的 5 个 blocker 聚合成 owner 可执行路径。它只做命令、输入、输出和 validator 的集中导航，不执行真实发布、不上传包、不伪造 proof。``recordKind=compatible-host-proof-execution-pack``，``packageState=owner-action-required``，``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前状态

| 项目 | 当前值 |
|---|---|
| releaseCloseGapDashboardState | ``$dashboardState`` |
| releaseCloseGapDashboardGapCount | ``$dashboardGapCount`` |
| releaseEvidenceBundleState | ``$releaseEvidenceState`` |
| releasePromotionIssueState | ``$promotionState`` |
| compatibleHostProofBackfillPackageState | ``$compatibleHostBackfillState`` |
| realModelAndPackageProofInputPackageState | ``$inputPackageState`` |
| blockerCount | ``$($executionItems.Count)`` |
| ownerReleaseExecutionPackageState | ``$ownerReleaseExecutionState`` |
| oneScreenReleaseHoldChecklistCount | ``$oneScreenReleaseHoldChecklistCount`` |

## One-Screen Release Hold Checklist

This table is inherited from ``owner-release-execution-package``. It is owner guidance only and does not promote runtime proof, authorize publication, or close the release issue.

| ID | Owner visible blocker | Current state | Owner next action | Validator |
|---|---|---|---|---|
$($oneScreenReleaseHoldRows -join "`r`n")

## Execution Items

| Gap | Proof class | Current state | Validator commands | Promotion boundary |
|---|---|---|---|---|
$($executionRows -join "`r`n")

## Owner 执行顺序

1. 先用 ``owner-authorization`` 明确 owner 人工授权和手动命令边界；本包不会执行 ``dotnet nuget push``。
2. 在 compatible CUDA/TensorRT host 上执行 ``package-consumer-runtime``，并用 ``Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof`` 验证真实 external runtime proof。
   Bridge-only package consumer log、bridge-only wrapper surface、``Skipped=True``、``dependency-probe-only``、``WrapperSurfaceEvidenceKind=compile-surface-proof``、``IsRuntimeExecutionProof=False``、ONNX Parser diagnostic snapshot、ONNX ParserRefitter diagnostic snapshot 和 copied managed diagnostic snapshot 只能作为诊断或 compile-surface 证据，不能替代 clean package consumer runtime proof。
3. 在真实 Linux x64 runner 上执行 ``linux-runner-proof``，并用 ``Test-LinuxRunnerEvidenceRecord.ps1`` 验证。
4. 准备 Classification/YoloVision 真实资产，执行 ``real-model-runtime``，并用 ``Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog`` 验证真实样例日志。
5. 真实发布后才执行 ``post-publish verification``，并用 ``Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof`` 验证。
   Local feed、draft、helper scan、bridge-only package consumer log、dependency-probe-only log、collection package 和 input package 都不能关闭 release issue。

## YoloVision 范围

YoloVision proof 范围固定覆盖 ``YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom``，以及 ``det/cls/seg/obb/pose/sem``（``det、cls、seg、obb、pose、sem``）。这些真实样例 proof 只能晋级 ``real-model-runtime``，不能替代 ``package-consumer-runtime``。

## Source Artifacts

$($sourceLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Safety Notes

$($safetyLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Compatible host proof execution pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackageState=$($record.packageState)"
Write-Output "BlockerCount=$($executionItems.Count)"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
