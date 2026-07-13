[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-OwnerInputLane {
  param(
    [string]$Id,
    [string]$State,
    [string[]]$SourceArtifacts,
    [string[]]$RequiredFields,
    [string[]]$FailFastOrder,
    [string[]]$OwnerCommands,
    [string[]]$ForbiddenSubstitutes,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    state = $State
    blocked = $true
    sourceArtifacts = @($SourceArtifacts)
    requiredFields = @($RequiredFields)
    requiredFieldCount = @($RequiredFields).Count
    failFastOrder = @($FailFastOrder)
    ownerCommands = @($OwnerCommands)
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

$yoloRepairPack = Read-JsonOrNull "artifacts\user-acceptance\yolovision-owner-proof-field-delta-repair-pack.json"
$packageValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$dashboard = Read-JsonOrNull "artifacts\final-release\release-proof-dashboard.json"
$finalGate = Read-JsonOrNull "artifacts\final-release\final-publish-proof-gate-report.json"

$commonForbidden = @(
  "build-only",
  "dry-run",
  "template",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "TensorRtExec report",
  "YoloVision matrix",
  "OnnxToEngine report",
  "readonly diagnostics",
  "screenshot",
  "sidecar-only report",
  "skipped run",
  "blocked-by-cuda-driver"
)

$realModelFields = @(
  "requiredGlobalEvidence.hostOs",
  "requiredGlobalEvidence.gpuName",
  "requiredGlobalEvidence.driverVersion",
  "requiredGlobalEvidence.cudaVersion",
  "requiredGlobalEvidence.tensorRtVersion",
  "requiredGlobalEvidence.runtimePackageVersion",
  "requiredGlobalEvidence.ownerReviewer",
  "requiredGlobalEvidence.ownerReviewedAtUtc",
  "cases[].model.source.url",
  "cases[].model.sha256",
  "cases[].labels.sha256",
  "cases[].input.sha256",
  "cases[].engine.sha256",
  "cases[].tensorRtExec.reportSha256",
  "cases[].yoloVision.runLogPath",
  "cases[].yoloVision.runLogSha256",
  "cases[].yoloVision.outputJsonPath",
  "cases[].yoloVision.outputJsonSha256",
  "cases[].ownerReview.accepted",
  "cases[].ownerReview.reviewedBy",
  "cases[].ownerReview.reviewedAtUtc"
)

$packageFields = @(
  "publicPackageSource",
  "managedPackageId",
  "managedPackageVersion",
  "runtimePackageVersion",
  "cleanConsumerRoot",
  "cleanConsumerRootOutsideRepository",
  "consumerProjectPath",
  "hostOs",
  "gpuName",
  "cudaDriverVersion",
  "cudaRuntimeVersion",
  "tensorRtVersion",
  "runtimeSmokeExitCode",
  "runtimeSmokePassed",
  "stdoutSummary",
  "stderrSummary",
  "runtimeSmokeLogPath",
  "runtimeSmokeLogSha256",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

$postPublishFields = @(
  "selectedChannel",
  "channelSourceUri",
  "publishedPackageUrl",
  "managedPackageUrl",
  "runtimePackageUrl",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "managedPackageSha256Source",
  "runtimePackageSha256Source",
  "cleanConsumerRootOutsideRepository",
  "consumerProjectPath",
  "noProjectReference",
  "noLocalPackageSource",
  "noLocalNupkgPackageReference",
  "restoreLogPath",
  "restoreLogSha256",
  "nativeAssetListingPath",
  "nativeAssetListingSha256",
  "dependencyProbeLogPath",
  "dependencyProbeLogSha256",
  "runtimeSmokeLogPath",
  "runtimeSmokeLogSha256",
  "runtimeSmokePassed",
  "runtimeSmokeExitCode",
  "stdoutSummary",
  "stderrSummary",
  "hostMetadata",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

$lanes = @(
  New-OwnerInputLane `
    -Id "real-model-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $yoloRepairPack -Name "repairPackState" -DefaultValue "blocked-owner-action-required")) `
    -SourceArtifacts @(
      "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
      "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
      "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json",
      "artifacts/user-acceptance/yolovision-real-asset-owner-proof-import-report.json"
    ) `
    -RequiredFields $realModelFields `
    -FailFastOrder @("01-host-metadata", "02-real-asset-paths", "03-sha256-hashes", "04-real-run-logs", "05-owner-review") `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    ) `
    -ForbiddenSubstitutes $commonForbidden `
    -Boundary "real-model-runtime requires owner-provided real model assets, logs, hashes, host metadata, and owner review; it cannot replace package-consumer-runtime proof."
  New-OwnerInputLane `
    -Id "package-consumer-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $packageValidation -Name "validationState" -DefaultValue "blocked-owner-input-required")) `
    -SourceArtifacts @(
      "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
      "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
      "artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json",
      "artifacts/final-release/package-consumer-runtime-proof-record.json"
    ) `
    -RequiredFields $packageFields `
    -FailFastOrder @("01-public-package-source", "02-clean-external-consumer", "03-no-local-substitutes", "04-host-runtime-metadata", "05-runtime-smoke-log", "06-owner-review") `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("sample-run-evidence", "real-model-runtime", "local package source")) `
    -Boundary "package-consumer-runtime requires a clean external consumer using a real public package source; local feed, ProjectReference, direct .nupkg, sample-run-evidence, and build-only evidence are forbidden substitutes."
  New-OwnerInputLane `
    -Id "post-publish-verification" `
    -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "template-only")) `
    -SourceArtifacts @(
      "artifacts/final-release/post-publish-verification-owner-input.template.json",
      "artifacts/final-release/post-publish-verification-record.json",
      "artifacts/final-release/post-publish-verification-validation.json"
    ) `
    -RequiredFields $postPublishFields `
    -FailFastOrder @("01-real-public-publish", "02-channel-package-urls", "03-downloaded-nupkg-hashes", "04-clean-post-publish-consumer", "05-runtime-key-smoke", "06-existing-log-hashes", "07-owner-review") `
    -OwnerCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationOwnerInputTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordFromOwnerInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
    ) `
    -ForbiddenSubstitutes (@($commonForbidden) + @("package-consumer-runtime", "pre-publish validation", "dependency-probe-only")) `
    -Boundary "post-publish verification requires real public channel publish followed by clean install/run evidence; package-consumer-runtime proof cannot replace it."
)

$ownerHandoffCommands = @(
  [pscustomobject]@{
    lane = "real-model-runtime"
    commands = @($lanes[0].ownerCommands)
  },
  [pscustomobject]@{
    lane = "package-consumer-runtime"
    commands = @($lanes[1].ownerCommands)
  },
  [pscustomobject]@{
    lane = "post-publish-verification"
    commands = @($lanes[2].ownerCommands)
  },
  [pscustomobject]@{
    lane = "release-close-owner-confirmation"
    commands = @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseProofLaneWorklist.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseProofDashboard.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict"
    )
  }
)

$bundle = [ordered]@{
  recordKind = "owner-input-preflight-bundle"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  bundleState = "blocked-owner-input-required"
  sourceArtifacts = @(
    "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/release-proof-dashboard.json",
    "artifacts/final-release/final-publish-proof-gate-report.json"
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  dashboardState = [string](Get-PropertyOrDefault -Object $dashboard -Name "dashboardState" -DefaultValue "blocked-owner-proof-required")
  finalGateState = [string](Get-PropertyOrDefault -Object $finalGate -Name "validationState" -DefaultValue "blocked-final-publish-real-proof-required")
  failedBlockerCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedBlockerCount" -DefaultValue 0)
  actionRequiredCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedActionRequiredCount" -DefaultValue 3)
  missingRealModelFieldCount = [int](Get-PropertyOrDefault -Object $yoloRepairPack -Name "missingFieldCount" -DefaultValue 0)
  laneCount = $lanes.Count
  blockedLaneCount = @($lanes | Where-Object { $_.blocked }).Count
  ownerInputLanes = @($lanes)
  ownerHandoffCommands = @($ownerHandoffCommands)
  boundary = "Owner input preflight bundle is a non-publishing handoff artifact. It does not promote runtime proof, does not close release issues, and does not replace real owner/public/post-publish evidence."
}

$jsonPath = Join-Path $OutputRoot "owner-input-preflight-bundle.json"
$markdownPath = Join-Path $OutputRoot "owner-input-preflight-bundle.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($bundle | ConvertTo-Json -Depth 16)
$laneRows = foreach ($lane in $lanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | ``$(ConvertTo-MarkdownCell $lane.state)`` | ``$($lane.requiredFieldCount)`` | ``$($lane.blocked)`` | $(ConvertTo-MarkdownCell (($lane.failFastOrder) -join " -> ")) |"
}

$handoffLines = foreach ($group in $ownerHandoffCommands) {
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("### $($group.lane)")
  foreach ($command in $group.commands) {
    $lines.Add("- ``$command``")
  }

  $lines -join "`r`n"
}

$markdown = @"
# Owner Input Preflight Bundle

Generated at: ``$($bundle.generatedAtUtc)``

## Summary

- recordKind: ``$($bundle.recordKind)``
- bundleState: ``$($bundle.bundleState)``
- dashboardState: ``$($bundle.dashboardState)``
- finalGateState: ``$($bundle.finalGateState)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``
- laneCount: ``$($bundle.laneCount)``
- blockedLaneCount: ``$($bundle.blockedLaneCount)``

## Owner Input Lanes

| Lane | State | Required Fields | Blocked | Fail-Fast Order |
| --- | --- | --- | --- | --- |
$($laneRows -join "`r`n")

## Owner Handoff Commands

$($handoffLines -join "`r`n`r`n")

## Boundary

$($bundle.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner input preflight bundle written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "BundleState=$($bundle.bundleState) LaneCount=$($bundle.laneCount) BlockedLaneCount=$($bundle.blockedLaneCount)"
