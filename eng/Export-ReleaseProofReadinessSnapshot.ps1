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

function New-ReadinessItem {
  param(
    [string]$Id,
    [string]$ProofClass,
    [bool]$Ready,
    [string]$CurrentState,
    [string]$RequiredValidator,
    [string]$RequiredRealInputs,
    [string]$OwnerAction,
    [string]$CannotUse
  )

  [pscustomobject]@{
    id = $Id
    proofClass = $ProofClass
    ready = $Ready
    state = if ($Ready) { "ready" } else { "blocked" }
    currentState = $CurrentState
    requiredValidator = $RequiredValidator
    requiredRealInputs = $RequiredRealInputs
    ownerAction = $OwnerAction
    cannotUse = $CannotUse
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$ownerPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$freezeSummary = Read-JsonOrNull "artifacts\release\release-candidate-freeze-summary.json"
$runtimeMatrix = Read-JsonOrNull "artifacts\final-release\release-runtime-proof-execution-matrix.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$preflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$ownerApproval = Read-JsonOrNull "artifacts\final-release\release-owner-approval-input-validation.json"
$ownerDecision = Read-JsonOrNull "artifacts\final-release\release-owner-decision-record.json"
$authorizedPlan = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan-validation.json"
$externalValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$linuxValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$ownerPackageState = [string](Get-PropertyOrDefault -Object $ownerPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$ownerChecklistCount = [int](Get-PropertyOrDefault -Object $ownerPackage -Name "oneScreenReleaseHoldChecklistCount" -DefaultValue 0)
$freezeState = [string](Get-PropertyOrDefault -Object $freezeSummary -Name "freezeState" -DefaultValue "missing-release-candidate-freeze-summary")
$matrixState = [string](Get-PropertyOrDefault -Object $runtimeMatrix -Name "matrixState" -DefaultValue "missing-release-runtime-proof-execution-matrix")
$matrixBlockedCount = [int](Get-PropertyOrDefault -Object $runtimeMatrix -Name "blockedProofItemCount" -DefaultValue -1)
$evidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")

$ownerApprovalReady = [bool](Get-PropertyOrDefault -Object $ownerApproval -Name "canPublishPublicly" -DefaultValue $false)
$ownerDecisionReady = [bool](Get-PropertyOrDefault -Object $ownerDecision -Name "canPublishPublicly" -DefaultValue $false)
$authorizedPlanReady = [bool](Get-PropertyOrDefault -Object $authorizedPlan -Name "canMaterializeExecutableCommands" -DefaultValue $false)
$ownerAuthorizationReady = $ownerApprovalReady -and $ownerDecisionReady -and $authorizedPlanReady

$externalState = [string](Get-PropertyOrDefault -Object $externalValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalClassification = [string](Get-PropertyOrDefault -Object $externalValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$externalCanPromote = [bool](Get-PropertyOrDefault -Object $externalValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalRuntimeEvidence = [bool](Get-PropertyOrDefault -Object $externalValidation -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$externalFailedCount = [int](Get-PropertyOrDefault -Object $externalValidation -Name "failedProofItemCount" -DefaultValue -1)
$packageConsumerReady = $externalCanPromote -and $externalRuntimeEvidence -and [string]::Equals($externalClassification, "package-consumer-runtime", [System.StringComparison]::OrdinalIgnoreCase) -and $externalFailedCount -eq 0

$linuxState = [string](Get-PropertyOrDefault -Object $linuxValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxReady = [bool](Get-PropertyOrDefault -Object $linuxValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)

$sampleState = [string](Get-PropertyOrDefault -Object $sampleValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-record-validation")
$sampleClassification = [string](Get-PropertyOrDefault -Object $sampleValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$sampleReady = [bool](Get-PropertyOrDefault -Object $sampleValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)

$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishClassification = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")
$postPublishReady = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishFailedCount = [int](Get-PropertyOrDefault -Object $postPublishValidation -Name "failedProofItemCount" -DefaultValue -1)
$postPublishProofReady = $postPublishReady -and $postPublishCanClose -and $postPublishFailedCount -eq 0

$items = @(
  New-ReadinessItem `
    -Id "owner-authorization" `
    -ProofClass "owner-authorization" `
    -Ready $ownerAuthorizationReady `
    -CurrentState "ownerApprovalCanPublish=$ownerApprovalReady; ownerDecisionCanPublish=$ownerDecisionReady; authorizedPlanCanMaterialize=$authorizedPlanReady" `
    -RequiredValidator "Test-ReleaseOwnerApprovalInput.ps1 + Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -RequiredRealInputs "ownerName, ownerDecisionId, approvalTimestampUtc, targetChannel, package identity, rollback plan, credential handling, NVIDIA redistribution disposition, approved command/proof bundle SHA256" `
    -OwnerAction "Provide real owner approval and authorized publish command plan. Keep commands manual until all proof validators pass." `
    -CannotUse "template, draft, checklist, dashboard, collection package, input package"

  New-ReadinessItem `
    -Id "package-consumer-runtime" `
    -ProofClass "package-consumer-runtime" `
    -Ready $packageConsumerReady `
    -CurrentState "validationState=$externalState; proofClassification=$externalClassification; canPromoteRuntimeProof=$externalCanPromote; isRuntimeExecutionEvidence=$externalRuntimeEvidence; failedProofItemCount=$externalFailedCount" `
    -RequiredValidator "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredRealInputs "clean external consumer, no ProjectReference, real package source, compatible CUDA/TensorRT host, runtime smoke log, stdout/stderr summary, nupkg SHA256 and log SHA256" `
    -OwnerAction "Run clean package-consumer runtime smoke on a compatible host for $RuntimePackageKey and validate the real record." `
    -CannotUse "local feed, ProjectReference, dependency-probe-only, blocked-by-cuda-driver, build-only, sidecar-only"

  New-ReadinessItem `
    -Id "linux-runner-proof" `
    -ProofClass "linux-runner-proof" `
    -Ready $linuxReady `
    -CurrentState "validationState=$linuxState; isRealLinuxRunnerProof=$linuxReady; linuxRuntimePackageKey=$LinuxRuntimePackageKey" `
    -RequiredValidator "Test-LinuxRunnerEvidenceRecord.ps1" `
    -RequiredRealInputs "real Linux x64 runner, runtime package key, native build/copy evidence, host metadata, Linux runner log and SHA256" `
    -OwnerAction "Run the Linux proof flow on a real Linux CUDA/TensorRT runner and copy back the validated record." `
    -CannotUse "Windows handoff, template-only, dry run summary, runbook"

  New-ReadinessItem `
    -Id "real-model-runtime" `
    -ProofClass "real-model-runtime" `
    -Ready $sampleReady `
    -CurrentState "validationState=$sampleState; proofClassification=$sampleClassification; canPromoteRealModelRuntime=$sampleReady" `
    -RequiredValidator "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -RequiredRealInputs "Classification/YoloVision real model, labels, input, license, model/input/log SHA256, TensorRtExec sidecar, sample runner log" `
    -OwnerAction "Provide real model assets and collect Classification/YoloVision sample runtime evidence." `
    -CannotUse "support matrix, model candidate list, build-only, parse-only, sidecar-only, missing model/input/license hashes"

  New-ReadinessItem `
    -Id "post-publish-verification" `
    -ProofClass "post-publish-verification" `
    -Ready $postPublishProofReady `
    -CurrentState "validationState=$postPublishState; proofClassification=$postPublishClassification; isPostPublishVerificationProof=$postPublishReady; canCloseReleaseIssue=$postPublishCanClose; failedProofItemCount=$postPublishFailedCount" `
    -RequiredValidator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredRealInputs "real public package URL, downloaded nupkg SHA256, clean external consumer, restore/build/probe/smoke logs, stdout/stderr summary, host metadata" `
    -OwnerAction "After owner-approved public publish, validate the real channel package from a clean consumer." `
    -CannotUse "post-publish template, backfill plan, local feed, ProjectReference, draft, collection package"
)

$readyCount = @($items | Where-Object { $_.ready }).Count
$blockedCount = @($items | Where-Object { -not $_.ready }).Count
$allReady = $readyCount -eq 5
$canPublishPublicly = $allReady -and $ownerAuthorizationReady -and $packageConsumerReady -and $linuxReady -and $sampleReady
$canCloseReleaseIssue = $canPublishPublicly -and $postPublishProofReady

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-proof-readiness-snapshot"
  readinessState = if ($allReady) { "ready-for-owner-final-review" } else { "blocked-real-proof-required" }
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $canPublishPublicly
  canCloseReleaseIssue = $canCloseReleaseIssue
  isReleaseProofComplete = $allReady
  ownerReleaseExecutionPackageState = $ownerPackageState
  oneScreenReleaseHoldChecklistCount = $ownerChecklistCount
  releaseCandidateFreezeState = $freezeState
  releaseRuntimeProofExecutionMatrixState = $matrixState
  releaseRuntimeProofExecutionMatrixBlockedProofItemCount = $matrixBlockedCount
  releaseEvidenceBundleState = $evidenceState
  releaseClosePreflightState = $preflightState
  readinessItemCount = $items.Count
  readyProofItemCount = $readyCount
  blockedProofItemCount = $blockedCount
  readinessItems = $items
  sourceArtifacts = @(
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/release/release-candidate-freeze-summary.json",
    "artifacts/final-release/release-runtime-proof-execution-matrix.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
    "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json"
  )
  nonSubstituteProofKinds = @(
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
    "Skipped=True",
    "blocked-by-cuda-driver",
    "managed-readiness",
    "managed-readiness-only",
    "callback-allocator-readiness-snapshot",
    "CallbackAllocatorReadinessSnapshot",
    "TensorRtCallbackAllocatorReadinessSnapshot",
    "precheck-only",
    "dry-run-only",
    "schema-only",
    "Windows handoff for Linux proof",
    "support matrix"
  )
  boundary = "This snapshot is a release proof status view only. It does not publish packages, does not upload assets, does not close the release issue, and cannot promote checklist, dashboard, runbook, collection package, input package, local feed, ProjectReference, sidecar, blocked-by-cuda-driver, managed-readiness, CallbackAllocatorReadinessSnapshot, precheck-only, dry-run-only, or schema-only evidence into proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-proof-readiness-snapshot.json"
$markdownPath = Join-Path $artifactRoot "release-proof-readiness-snapshot.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $items | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.proofClass)`` | ``$($_.ready)`` | $($_.currentState.Replace("|", "\|")) | ``$($_.requiredValidator)`` | $($_.ownerAction.Replace("|", "\|")) | $($_.cannotUse.Replace("|", "\|")) |"
}
$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $record.nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release Proof Readiness Snapshot

生成时间：$($record.generatedAtUtc)

## 总结

``recordKind=release-proof-readiness-snapshot``，``readinessState=$($record.readinessState)``。这是一份 5-blocker 发布 proof readiness 快照，只做判断，不执行发布、不上传包、不关闭 release issue。

| 项目 | 值 |
|---|---|
| performsPublish | ``False`` |
| canPublishPublicly | ``$canPublishPublicly`` |
| canCloseReleaseIssue | ``$canCloseReleaseIssue`` |
| isReleaseProofComplete | ``$allReady`` |
| oneScreenReleaseHoldChecklistCount | ``$ownerChecklistCount`` |
| readiness item count | ``$($record.readinessItemCount)`` |
| ready proof item count | ``$readyCount`` |
| blocked proof item count | ``$blockedCount`` |
| owner package state | ``$ownerPackageState`` |
| freeze state | ``$freezeState`` |
| runtime proof matrix state | ``$matrixState`` |

## Readiness Items

| ID | Proof class | Ready | Current state | Required validator | Owner action | Cannot use |
|---|---|---|---|---|---|---|
$($rows -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release proof readiness snapshot written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ReadinessState=$($record.readinessState)"
Write-Output "ReadyProofItemCount=$readyCount"
Write-Output "BlockedProofItemCount=$blockedCount"
Write-Output "CanPublishPublicly=$canPublishPublicly"
Write-Output "CanCloseReleaseIssue=$canCloseReleaseIssue"
