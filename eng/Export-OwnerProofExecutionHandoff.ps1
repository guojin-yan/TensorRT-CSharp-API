[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

function Test-ArtifactCandidate {
  param([string]$RelativePath)

  if ([string]::IsNullOrWhiteSpace($RelativePath)) {
    return $false
  }

  $normalized = $RelativePath -replace "/", "\"
  $path = Join-Path $RepositoryRoot $normalized
  return Test-Path -LiteralPath $path -PathType Leaf
}

function New-HandoffLine {
  param(
    [object]$BackfillItem,
    [bool]$CanPromoteProof,
    [string]$OwnerNextAction
  )

  $expectedArtifacts = @(Get-PropertyOrDefault -Object $BackfillItem -Name "expectedArtifacts" -DefaultValue @())
  $sourceArtifacts = @(Get-PropertyOrDefault -Object $BackfillItem -Name "sourceArtifacts" -DefaultValue @())
  $requiredRealInputs = @(Get-PropertyOrDefault -Object $BackfillItem -Name "requiredRealInputs" -DefaultValue @())
  $cannotUse = @(Get-PropertyOrDefault -Object $BackfillItem -Name "cannotUse" -DefaultValue $script:NonSubstituteProofKinds)
  $candidateArtifacts = @($expectedArtifacts | Where-Object { Test-ArtifactCandidate $_ })
  $missingExpectedArtifacts = @($expectedArtifacts | Where-Object { -not (Test-ArtifactCandidate $_) })
  $missingRealInputs = @($requiredRealInputs)

  [pscustomobject]@{
    id = [string](Get-PropertyOrDefault -Object $BackfillItem -Name "id" -DefaultValue "")
    proofClass = [string](Get-PropertyOrDefault -Object $BackfillItem -Name "proofClass" -DefaultValue "")
    title = [string](Get-PropertyOrDefault -Object $BackfillItem -Name "title" -DefaultValue "")
    currentState = [string](Get-PropertyOrDefault -Object $BackfillItem -Name "currentState" -DefaultValue "missing-state")
    ownerNextAction = $OwnerNextAction
    firstCommand = [string](Get-PropertyOrDefault -Object $BackfillItem -Name "firstCommand" -DefaultValue "")
    validatorCommand = [string](Get-PropertyOrDefault -Object $BackfillItem -Name "validatorCommand" -DefaultValue "")
    requiredRealInputs = $requiredRealInputs
    requiredRealInputCount = $requiredRealInputs.Count
    missingRealInputs = $missingRealInputs
    missingRealInputCount = $missingRealInputs.Count
    expectedArtifacts = $expectedArtifacts
    existingCandidateArtifacts = $candidateArtifacts
    existingCandidateArtifactCount = $candidateArtifacts.Count
    missingExpectedArtifacts = $missingExpectedArtifacts
    sourceArtifacts = $sourceArtifacts
    cannotUse = $cannotUse
    nonSubstituteProofKinds = $cannotUse
    blockerReason = [string](Get-PropertyOrDefault -Object $BackfillItem -Name "blockerReason" -DefaultValue "blocked-real-proof-required")
    canPromoteProof = $CanPromoteProof
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$backfillPack = Read-JsonOrNull "artifacts\final-release\owner-proof-backfill-execution-pack.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseIssueCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$releaseProofReadinessSnapshot = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$ownerProofInputValidation = Read-JsonOrNull "artifacts\final-release\release-owner-proof-input-record-validation.json"
$externalRuntimeValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$linuxRunnerValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$script:NonSubstituteProofKinds = @(Get-PropertyOrDefault -Object $backfillPack -Name "nonSubstituteProofKinds" -DefaultValue @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "local feed",
  "ProjectReference",
  "direct .nupkg reference",
  "helper scan",
  "build-only",
  "parse-only",
  "sidecar-only",
  "dependency-probe-only",
  "readiness snapshot",
  "template-only release issue close record",
  "preflight-only release issue close record",
  "schema-only",
  "release-issue-close-record-template.json",
  "Windows handoff for Linux proof"
))

$backfillItems = @(Get-PropertyOrDefault -Object $backfillPack -Name "backfillItems" -DefaultValue @())

if ($backfillItems.Count -eq 0) {
  throw "owner-proof-backfill-execution-pack.json is missing or has no backfillItems. Run Export-OwnerProofBackfillExecutionPack.ps1 first."
}

$ownerProofInputCanPromote = [bool](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "canPromoteOwnerProofInput" -DefaultValue $false)
$externalRuntimeCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$linuxRunnerProof = [bool](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleRunProof = [bool](Get-PropertyOrDefault -Object $sampleRunValidation -Name "isRealSampleRunProof" -DefaultValue $false)
$postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$closeRecordCanPromote = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "canPromoteReleaseIssueCloseRecord" -DefaultValue $false)
$closeRecordCanClose = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "canCloseReleaseIssue" -DefaultValue $false)

$promotionMap = @{
  "owner-authorization" = $ownerProofInputCanPromote
  "package-consumer-runtime" = $externalRuntimeCanPromote
  "linux-runner-proof" = $linuxRunnerProof
  "real-model-runtime" = $sampleRunProof
  "post-publish-verification" = ($postPublishProof -and $postPublishCanClose)
  "release-issue-close-record" = ($closeRecordCanPromote -and $closeRecordCanClose)
}

$nextActionMap = @{
  "owner-authorization" = "Fill release-owner-proof-input-record.json with real owner authorization, selected channel, package hashes, runtime log hash, host metadata, and non-placeholder command approval."
  "package-consumer-runtime" = "Collect compatible-host package consumer runtime smoke from a clean external consumer, then validate external-runtime-proof-record.json with -FailOnNotProof."
  "linux-runner-proof" = "Run the Linux x64 runtime proof on a real Linux runner and validate linux-runner-evidence-record.json."
  "real-model-runtime" = "Backfill real Classification/YoloVision assets, model/input/label hashes, TensorRtExec sidecar, runner log, and sample-run evidence validation."
  "post-publish-verification" = "After owner performs real selected-channel publish outside automation, restore/build/smoke a clean consumer and validate post-publish-verification-record.json."
  "release-issue-close-record" = "After every real proof passes, fill release-issue-close-record.json with bundle SHA256, rollback plan, owner final close decision, and run -FailOnNotCloseReady."
}

$handoffLines = @($backfillItems | ForEach-Object {
  $id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
  New-HandoffLine -BackfillItem $_ -CanPromoteProof ([bool]$promotionMap[$id]) -OwnerNextAction ([string]$nextActionMap[$id])
})

$readyLineCount = @($handoffLines | Where-Object { $_.canPromoteProof }).Count
$blockedLineCount = $handoffLines.Count - $readyLineCount
$handoffState = if ($blockedLineCount -eq 0) { "ready-for-owner-close-review" } else { "blocked-real-proof-required" }

$releaseClosePreflightState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$releaseClosePreflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)
$releaseClosePreflightCanClose = [bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)
$releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$releaseEvidenceBundleCanPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPublishPublicly" -DefaultValue $false)
$releaseEvidenceBundleCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue $false)
$ownerReleasePackageState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$readinessState = [string](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessState" -DefaultValue "missing-release-proof-readiness-snapshot")

$sourceArtifacts = @(
  "artifacts/final-release/owner-proof-backfill-execution-pack.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-close-preflight.json",
  "artifacts/final-release/release-issue-close-record-validation.json",
  "artifacts/final-release/owner-release-execution-package.json",
  "artifacts/final-release/release-proof-readiness-snapshot.json"
)

$record = [pscustomobject]@{
  recordKind = "owner-proof-execution-handoff"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  handoffState = $handoffState
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  backfillSourcePackageState = [string](Get-PropertyOrDefault -Object $backfillPack -Name "packageState" -DefaultValue "missing-owner-proof-backfill-execution-pack")
  ownerReleaseExecutionPackageState = $ownerReleasePackageState
  releaseProofReadinessSnapshotState = $readinessState
  releaseClosePreflightState = $releaseClosePreflightState
  releaseClosePreflightFailedItemCount = $releaseClosePreflightFailedItemCount
  releaseClosePreflightCanCloseReleaseIssue = $releaseClosePreflightCanClose
  releaseEvidenceBundleState = $releaseEvidenceBundleState
  releaseEvidenceBundleCanPublishPublicly = $releaseEvidenceBundleCanPublish
  releaseEvidenceBundleCanCloseReleaseIssue = $releaseEvidenceBundleCanClose
  releaseIssueCloseRecordValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
  releaseIssueCloseRecordFailedValidationItemCount = [int](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "failedValidationItemCount" -DefaultValue -1)
  releaseIssueCloseRecordCanPromote = $closeRecordCanPromote
  releaseIssueCloseRecordCanCloseReleaseIssue = $closeRecordCanClose
  handoffLineCount = $handoffLines.Count
  readyHandoffLineCount = $readyLineCount
  blockedHandoffLineCount = $blockedLineCount
  handoffLines = $handoffLines
  nonSubstituteProofKinds = $script:NonSubstituteProofKinds
  sourceArtifacts = $sourceArtifacts
  safetyBoundary = "owner-proof-execution-handoff is guidance only; it never publishes packages, never approves public publication, and never closes the release issue."
  ownerExecutionOrder = @(
    "refresh stale release claims and release close preflight",
    "fill owner authorization and selected-channel fields",
    "collect package-consumer-runtime proof on compatible host",
    "collect Linux runner proof on a real Linux x64 runner",
    "collect Classification/YoloVision real-model-runtime proof",
    "owner manually executes publish commands outside automation only after authorization",
    "collect post-publish clean consumer proof from selected channel",
    "refresh release evidence bundle",
    "fill and validate release issue close record with -FailOnNotCloseReady"
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-proof-execution-handoff.json"
$markdownPath = Join-Path $artifactRoot "owner-proof-execution-handoff.md"

Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 14)
$lineRows = $handoffLines | ForEach-Object {
  $state = ([string]$_.currentState).Replace("|", "\|")
  $action = ([string]$_.ownerNextAction).Replace("|", "\|")
  $validator = ([string]$_.validatorCommand).Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.proofClass)`` | ``$($_.canPromoteProof)`` | ``$($_.missingRealInputCount)`` | ``$($_.existingCandidateArtifactCount)`` | $state | $action | ``$validator`` |"
}
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $script:NonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$orderLines = $record.ownerExecutionOrder | ForEach-Object { "- $_" }

$markdown = @"
# Owner Proof Execution Handoff

生成时间：$($record.generatedAtUtc)

## 总结

``owner-proof-execution-handoff`` 是 release owner 的真实外部 proof 采集/发布执行交接包。它从 ``owner-proof-backfill-execution-pack`` 继承 6 条 proof line，并补充候选产物、缺失输入、owner next action 和 promotion 状态。它只做 guidance：``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前 Gate

| 项目 | 当前值 |
|---|---|
| handoffState | ``$handoffState`` |
| handoffLineCount | ``$($handoffLines.Count)`` |
| readyHandoffLineCount | ``$readyLineCount`` |
| blockedHandoffLineCount | ``$blockedLineCount`` |
| releaseClosePreflightState | ``$releaseClosePreflightState`` |
| releaseClosePreflightFailedItemCount | ``$releaseClosePreflightFailedItemCount`` |
| releaseEvidenceBundleState | ``$releaseEvidenceBundleState`` |
| releaseIssueCloseRecordValidationState | ``$($record.releaseIssueCloseRecordValidationState)`` |
| releaseIssueCloseRecordFailedValidationItemCount | ``$($record.releaseIssueCloseRecordFailedValidationItemCount)`` |

## Handoff Lines

| ID | Proof class | Can promote proof | Missing real inputs | Existing candidate artifacts | Current state | Owner next action | Validator |
|---|---|---|---|---|---|---|---|
$($lineRows -join "`r`n")

## Owner Execution Order

$($orderLines -join "`r`n")

## Release Boundary

该 handoff 不会执行 ``dotnet nuget push``，不会上传 GitHub Packages，不会创建 GitHub Release，不会关闭 release issue。``release-issue-close-record-template.json``、schema-only、preflight-only、readiness snapshot、handoff、runbook、collection package、local feed、ProjectReference 和 dependency-probe-only 都不能替代真实 proof。

最终关闭仍必须等待真实 ``release-issue-close-record.json`` 通过：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady
```

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner proof execution handoff written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "HandoffState=$handoffState"
Write-Output "HandoffLineCount=$($handoffLines.Count)"
Write-Output "ReadyHandoffLineCount=$readyLineCount"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
