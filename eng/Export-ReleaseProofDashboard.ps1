[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

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

function New-DashboardLane {
  param(
    [string]$Id,
    [string]$ProofKind,
    [string]$State,
    [bool]$Blocked,
    [string[]]$BlockedReasons,
    [string[]]$ValidatorCommands,
    [string[]]$NextOwnerCommands,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    proofKind = $ProofKind
    state = $State
    blocked = $Blocked
    blockedReasons = @($BlockedReasons)
    validatorCommands = @($ValidatorCommands)
    nextOwnerCommands = @($NextOwnerCommands)
    canPromoteProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

$laneWorklist = Read-JsonOrNull "artifacts/final-release/release-close-proof-lane-worklist.json"
$finalGate = Read-JsonOrNull "artifacts/final-release/final-publish-proof-gate-report.json"
$yoloExecutionPack = Read-JsonOrNull "artifacts/user-acceptance/yolovision-real-asset-owner-proof-execution-pack.json"
$yoloRepairPack = Read-JsonOrNull "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json"
$packageValidation = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts/final-release/post-publish-verification-validation.json"
$advancedChecklist = Read-JsonOrNull "artifacts/final-release/tensorrtexec-advanced-proof-readiness-checklist.json"

$gateItems = @()
if ($null -ne $finalGate) {
  $gateItems = @(Get-PropertyOrDefault -Object $finalGate -Name "validationItems" -DefaultValue @())
}

$actionRequiredItems = @($gateItems | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "action-required" })

$laneMap = @{}
if ($null -ne $laneWorklist) {
  foreach ($lane in @(Get-PropertyOrDefault -Object $laneWorklist -Name "lanes" -DefaultValue @())) {
    $laneMap[[string](Get-PropertyOrDefault -Object $lane -Name "id" -DefaultValue "")] = $lane
  }
}

$realLane = $laneMap["real-model-runtime"]
$packageLane = $laneMap["package-consumer-runtime"]
$postPublishLane = $laneMap["post-publish-verification"]
$ownerLane = $laneMap["public-owner-confirmation"]

$dashboardLanes = @(
  New-DashboardLane `
    -Id "real-model-runtime" `
    -ProofKind "real-model-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $realLane -Name "laneState" -DefaultValue "owner-action-required")) `
    -Blocked $true `
    -BlockedReasons @(
      "YoloVision owner proof input is missing real model assets, run logs, output JSON, SHA256 values, host metadata, and owner review.",
      "Owner field delta repair pack missingFieldCount=$([int](Get-PropertyOrDefault -Object $yoloRepairPack -Name 'missingFieldCount' -DefaultValue 0)).",
      "Execution pack ownerRequiredFieldCount=$([int](Get-PropertyOrDefault -Object $yoloExecutionPack -Name 'ownerRequiredFieldCount' -DefaultValue 0))."
    ) `
    -ValidatorCommands @((Get-PropertyOrDefault -Object $realLane -Name "validatorCommands" -DefaultValue @())) `
    -NextOwnerCommands @(
      "Fill yolovision-real-asset-owner-proof-input.template.json.",
      "Run Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1 after each owner input pass.",
      "Run Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict and Import-YoloVisionRealAssetOwnerProofInput.ps1.",
      "Run Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog only after real logs exist."
    ) `
    -Boundary "real-model-runtime cannot be replaced by TensorRtExec report, build-only, sidecar, screenshot, or sample matrix."
  New-DashboardLane `
    -Id "package-consumer-runtime" `
    -ProofKind "package-consumer-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $packageLane -Name "laneState" -DefaultValue "blocked-owner-input-required")) `
    -Blocked $true `
    -BlockedReasons @(
      "Clean external consumer proof is missing.",
      "publicPackageSource, managedPackageId, managedPackageVersion, runtimePackageVersion, hostOs, gpuName, cudaDriverVersion, cudaRuntimeVersion, tensorRtVersion, exitCode=0, stdoutSummary, stderrSummary, and smokeLogSha256 remain owner inputs.",
      "local feed, ProjectReference, direct .nupkg, build-only, dry-run, template, and skipped run remain forbidden substitutes."
    ) `
    -ValidatorCommands @((Get-PropertyOrDefault -Object $packageLane -Name "validatorCommands" -DefaultValue @())) `
    -NextOwnerCommands @(
      "Fill package-consumer-runtime-proof-owner-input.template.json with real public package evidence.",
      "Run Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict.",
      "Run Import-PackageConsumerRuntimeProofOwnerInput.ps1 and Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof."
    ) `
    -Boundary "package-consumer-runtime cannot be replaced by sample-run-evidence, local feed, ProjectReference, or direct .nupkg."
  New-DashboardLane `
    -Id "post-publish-verification" `
    -ProofKind "post-publish-verification" `
    -State ([string](Get-PropertyOrDefault -Object $postPublishLane -Name "laneState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "template-only")))) `
    -Blocked $true `
    -BlockedReasons @(
      "Public channel publish has not been performed by owner.",
      "Public package source, package URLs, install logs, run logs, hashes, host metadata, and owner review are missing.",
      "package-consumer-runtime pre-publish proof cannot replace post-publish verification."
    ) `
    -ValidatorCommands @((Get-PropertyOrDefault -Object $postPublishLane -Name "validatorCommands" -DefaultValue @())) `
    -NextOwnerCommands @(
      "After real public publish, fill post-publish verification owner input.",
      "Run Test-PostPublishVerificationRecord.ps1 -FailOnNotProof."
    ) `
    -Boundary "post-publish verification requires real public channel publish and clean install/run evidence."
  New-DashboardLane `
    -Id "public-owner-confirmation" `
    -ProofKind "public-owner-confirmation" `
    -State ([string](Get-PropertyOrDefault -Object $ownerLane -Name "laneState" -DefaultValue "blocked-release-close-real-proof-required")) `
    -Blocked $true `
    -BlockedReasons @(
      "Owner cannot confirm final close until real-model-runtime, package-consumer-runtime, and post-publish verification are complete.",
      "Rollback owner, rollback trigger, rollback plan, final owner decision, and release issue metadata must be reviewed."
    ) `
    -ValidatorCommands @((Get-PropertyOrDefault -Object $ownerLane -Name "validatorCommands" -DefaultValue @())) `
    -NextOwnerCommands @(
      "Review release-close-proof-lane-worklist.json.",
      "Fill final owner release close input only after real proof lanes pass.",
      "Run final close validators."
    ) `
    -Boundary "public owner confirmation is not proof by itself."
)

$supportingEvidence = @(
  [pscustomobject]@{ id = "tensorrtexec-report"; state = "supporting-evidence-only"; boundary = "TensorRtExec report is not runtime proof." }
  [pscustomobject]@{ id = "yolovision-matrix"; state = "supporting-evidence-only"; boundary = "YoloVision matrix is not package-consumer-runtime proof." }
  [pscustomobject]@{ id = "onnxtoengine-report"; state = "supporting-evidence-only"; boundary = "OnnxToEngine report is build/report evidence only and is not runtime proof." }
  [pscustomobject]@{ id = "tensorrtexec-advanced-checklist"; state = [string](Get-PropertyOrDefault -Object $advancedChecklist -Name "checklistState" -DefaultValue "template-owner-input-required"); boundary = "Timing cache and INT8 checklists do not replace package-consumer-runtime proof." }
)

$dashboard = [ordered]@{
  recordKind = "release-proof-dashboard"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  dashboardState = "blocked-owner-proof-required"
  sourceArtifacts = @(
    "artifacts/final-release/release-close-proof-lane-worklist.json",
    "artifacts/final-release/final-publish-proof-gate-report.json",
    "artifacts/user-acceptance/yolovision-real-asset-owner-proof-execution-pack.json",
    "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/tensorrtexec-advanced-proof-readiness-checklist.json"
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  failedBlockerCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedBlockerCount" -DefaultValue 0)
  actionRequiredCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedActionRequiredCount" -DefaultValue $actionRequiredItems.Count)
  actionRequiredSources = @($actionRequiredItems | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
  laneCount = $dashboardLanes.Count
  blockedLaneCount = @($dashboardLanes | Where-Object { $_.blocked }).Count
  ownerProofActionRequiredLaneCount = @($dashboardLanes | Where-Object { $_.blocked }).Count
  lanes = @($dashboardLanes)
  supportingEvidence = @($supportingEvidence)
  nextOwnerCommands = @(
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionOwnerProofFieldDeltaRepairPack.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseProofLaneWorklist.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseProofDashboard.ps1"
  )
  boundary = "Release proof dashboard is an owner navigation artifact. It does not publish, close release issues, promote runtime proof, or replace real owner/public/post-publish evidence."
}

$jsonPath = Join-Path $OutputRoot "release-proof-dashboard.json"
$markdownPath = Join-Path $OutputRoot "release-proof-dashboard.md"
$dashboard | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = foreach ($lane in $dashboardLanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | ``$(ConvertTo-MarkdownCell $lane.proofKind)`` | ``$(ConvertTo-MarkdownCell $lane.state)`` | ``$($lane.blocked)`` | $(ConvertTo-MarkdownCell (($lane.blockedReasons | Select-Object -First 1) -join ' ')) |"
}

$supportRows = foreach ($item in $supportingEvidence) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$(ConvertTo-MarkdownCell $item.state)`` | $(ConvertTo-MarkdownCell $item.boundary) |"
}

$commandLines = $dashboard.nextOwnerCommands | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release Proof Dashboard

Generated at: ``$($dashboard.generatedAtUtc)``

## Summary

- recordKind: ``$($dashboard.recordKind)``
- dashboardState: ``$($dashboard.dashboardState)``
- actionRequiredCount: ``$($dashboard.actionRequiredCount)``
- failedBlockerCount: ``$($dashboard.failedBlockerCount)``
- laneCount: ``$($dashboard.laneCount)``
- blockedLaneCount: ``$($dashboard.blockedLaneCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``

## Proof Lanes

| Lane | Proof Kind | State | Blocked | First Reason |
| --- | --- | --- | --- | --- |
$($laneRows -join "`r`n")

## Supporting Evidence Only

| Artifact | State | Boundary |
| --- | --- | --- |
$($supportRows -join "`r`n")

## Next Owner Commands

$($commandLines -join "`r`n")

## Boundary

$($dashboard.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release proof dashboard written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "DashboardState=$($dashboard.dashboardState) ActionRequired=$($dashboard.actionRequiredCount) BlockedLaneCount=$($dashboard.blockedLaneCount)"
