[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-external-proof-execution-bundle.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner external proof execution bundle not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$bundleState = [string](Get-PropertyOrDefault -Object $record -Name "bundleState" -DefaultValue "")
$bundleItems = @(Get-PropertyOrDefault -Object $record -Name "executionBundleItems" -DefaultValue @())
$blockedItems = @($bundleItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "bundleItemState" -DefaultValue "") -eq "blocked-owner-external-proof-execution-required" })
$runtimeProofPreflightMatrixFound = [bool](Get-PropertyOrDefault -Object $record -Name "runtimeProofPreflightMatrixFound" -DefaultValue $false)
$runtimeProofPreflightEntryCount = [int](Get-PropertyOrDefault -Object $record -Name "runtimeProofPreflightEntryCount" -DefaultValue 0)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-external-proof-execution-bundle") -Severity "blocker" -Detail "recordKind must be owner-external-proof-execution-bundle.")) | Out-Null
$items.Add((New-ValidationItem -Id "bundle-state" -Passed ($bundleState -eq "blocked-owner-external-proof-execution-required") -Severity "blocker" -Detail "Bundle must remain blocked until real external proof is executed.")) | Out-Null
$items.Add((New-ValidationItem -Id "bundle-item-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "executionBundleItemCount" -DefaultValue 0) -eq 6 -and $bundleItems.Count -eq 6) -Severity "blocker" -Detail "Bundle must cover 6 proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-item-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedExecutionBundleItemCount" -DefaultValue -1) -eq $blockedItems.Count -and $blockedItems.Count -eq 6) -Severity "blocker" -Detail "All default bundle items must remain blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-item-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyExecutionBundleItemCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default bundle must not claim proof readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "runtime-proof-preflight-matrix" -Passed ($runtimeProofPreflightMatrixFound -and $runtimeProofPreflightEntryCount -ge 6) -Severity "blocker" -Detail "Owner execution bundle must include package-consumer-runtime-proof-preflight-matrix.json as the owner action contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Execution bundle must not promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-or-close" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -Severity "blocker" -Detail "Execution bundle must not publish or close release issue.")) | Out-Null

foreach ($bundleItem in $bundleItems) {
  $itemId = [string](Get-PropertyOrDefault -Object $bundleItem -Name "executionBundleItemId" -DefaultValue "unknown-bundle-item")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $bundleItem -Name "resultInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $bundleItem -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $bundleItem -Name "runtimePackageKey" -DefaultValue "")) -and
    @((Get-PropertyOrDefault -Object $bundleItem -Name "commandSequence" -DefaultValue @())).Count -ge 5 -and
    @((Get-PropertyOrDefault -Object $bundleItem -Name "requiredSha256Commands" -DefaultValue @())).Count -ge 5 -and
    @((Get-PropertyOrDefault -Object $bundleItem -Name "hostMetadataChecklist" -DefaultValue @())).Count -ge 7 -and
    @((Get-PropertyOrDefault -Object $bundleItem -Name "runtimeProofPreflightRequiredFields" -DefaultValue @())).Count -ge 8 -and
    @((Get-PropertyOrDefault -Object $bundleItem -Name "forbiddenProofSubstitutes" -DefaultValue @())).Count -ge 8 -and
    @((Get-PropertyOrDefault -Object $bundleItem -Name "nonSubstituteChecklist" -DefaultValue @())).Count -ge 8 -and
    @((Get-PropertyOrDefault -Object $bundleItem -Name "importTargetFieldMap" -DefaultValue @())).Count -ge 20
  $items.Add((New-ValidationItem -Id "$itemId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each bundle item must include commands, hash commands, host metadata, non-substitute checklist, and import target field map.")) | Out-Null
  $preflight = Get-PropertyOrDefault -Object $bundleItem -Name "runtimeProofPreflight" -DefaultValue $null
  $requiresRuntimePackagePreflightSelection = [bool](Get-PropertyOrDefault -Object $preflight -Name "requiresRuntimePackagePreflightSelection" -DefaultValue $false)
  $preflightOptionsReady = [bool](Get-PropertyOrDefault -Object $preflight -Name "matrixFound" -DefaultValue $false) -and
    [int](Get-PropertyOrDefault -Object $preflight -Name "availableRuntimePackageOptionCount" -DefaultValue 0) -ge 6 -and
    -not [bool](Get-PropertyOrDefault -Object $preflight -Name "canPromotePackageConsumerRuntimeProof" -DefaultValue $true)
  $preflightSelectionReady = (-not $requiresRuntimePackagePreflightSelection) -or (
    [bool](Get-PropertyOrDefault -Object $preflight -Name "entryFound" -DefaultValue $false) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $preflight -Name "selectedRuntimePackageId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $preflight -Name "selectedRestoreSourceMode" -DefaultValue "")) -and
    [int](Get-PropertyOrDefault -Object $preflight -Name "selectedNativeAssetCopyExpected" -DefaultValue 0) -gt 0
  )
  $items.Add((New-ValidationItem -Id "$itemId-runtime-proof-preflight-contract" -Passed ($preflightOptionsReady -and $preflightSelectionReady) -Severity "blocker" -Detail "Each bundle item must carry RuntimeProofPreflight options; package-consumer-runtime lanes must select package id, restore source mode, native asset count, validator command, and non-promotable boundary.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$itemId-real-execution-required" -Passed $false -Severity "action-required" -Detail "Owner must execute this lane outside the template flow and backfill real logs, hashes, host metadata, package identity, validator output, and review fields.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-external-proof-execution-bundle" } else { "blocked-owner-external-proof-execution-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-external-proof-execution-bundle-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  executionBundleItemCount = $bundleItems.Count
  blockedExecutionBundleItemCount = $blockedItems.Count
  readyExecutionBundleItemCount = 0
  runtimeProofPreflightMatrixFound = $runtimeProofPreflightMatrixFound
  runtimeProofPreflightEntryCount = $runtimeProofPreflightEntryCount
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validation checks execution bundle shape only. It is not runtime proof, publication approval, post-publish proof, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-external-proof-execution-bundle-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-external-proof-execution-bundle-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner External Proof Execution Bundle Validation")
$lines.Add("")
$lines.Add("| 项目 | 当前值 |")
$lines.Add("|---|---|")
$lines.Add("| validationState | ``$(ConvertTo-MarkdownCell $validation.validationState)`` |")
$lines.Add("| executionBundleItemCount | ``$($validation.executionBundleItemCount)`` |")
$lines.Add("| runtimeProofPreflightMatrixFound | ``$($validation.runtimeProofPreflightMatrixFound)`` |")
$lines.Add("| runtimeProofPreflightEntryCount | ``$($validation.runtimeProofPreflightEntryCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.safetyBoundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external proof execution bundle validation written to $jsonPath"
Write-Host "Owner external proof execution bundle validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner external proof execution bundle validation failed with $($failedBlockers.Count) blocker(s)."
}
