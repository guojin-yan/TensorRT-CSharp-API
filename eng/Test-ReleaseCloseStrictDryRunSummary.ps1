[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-strict-dry-run-summary.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release close strict dry-run summary not found: $resolvedInputPath"
}

$summary = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$closeDryRunItems = @(Get-PropertyOrDefault -Object $summary -Name "closeDryRunItems" -DefaultValue @())
$items = New-Object System.Collections.Generic.List[object]
$blockedItems = @($closeDryRunItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "closeDryRunItemState" -DefaultValue "") -eq "blocked-release-close-real-proof-required" })
$readyItems = @($closeDryRunItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForReleaseClose" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $summary -Name "recordKind" -DefaultValue "") -eq "release-close-strict-dry-run-summary") -Severity "blocker" -Detail "recordKind must be release-close-strict-dry-run-summary.")) | Out-Null
$items.Add((New-ValidationItem -Id "close-dry-run-state" -Passed ([string](Get-PropertyOrDefault -Object $summary -Name "closeDryRunState" -DefaultValue "") -eq "blocked-release-close-real-proof-required") -Severity "blocker" -Detail "Strict close dry-run must remain blocked until real proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "item-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "closeDryRunItemCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Strict close dry-run must cover 6 proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-item-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "blockedCloseDryRunItemCount" -DefaultValue -1) -eq $blockedItems.Count -and $blockedItems.Count -ge 6) -Severity "blocker" -Detail "Default strict close dry-run must keep all lanes blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-item-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "readyCloseDryRunItemCount" -DefaultValue -1) -eq $readyItems.Count -and $readyItems.Count -eq 0) -Severity "blocker" -Detail "Default strict close dry-run must not claim close-ready lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "remaining-gap-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "remainingGapCount" -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "Strict close dry-run must surface remaining gaps.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $summary -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Strict close dry-run must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $summary -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Strict close dry-run must not publish or approve public publication.")) | Out-Null

foreach ($dryRunItem in $closeDryRunItems) {
  $itemId = [string](Get-PropertyOrDefault -Object $dryRunItem -Name "closeDryRunItemId" -DefaultValue "unknown-close-dry-run-item")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $dryRunItem -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $dryRunItem -Name "strictCloseCommand" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $dryRunItem -Name "requiredNextCommand" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $dryRunItem -Name "closeImpact" -DefaultValue "")) -and
    [int](Get-PropertyOrDefault -Object $dryRunItem -Name "remainingGapCount" -DefaultValue 0) -gt 0
  $items.Add((New-ValidationItem -Id "$itemId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each strict close dry-run item must include lane, remaining gaps, commands, and close impact.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$itemId-release-close-required" -Passed ([bool](Get-PropertyOrDefault -Object $dryRunItem -Name "readyForReleaseClose" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must resolve real proof, post-publish, rollback, final decision, and strict close validation before this lane is close-ready.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-release-close-strict-dry-run-summary" } else { "blocked-release-close-real-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "release-close-strict-dry-run-summary-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  closeDryRunItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "closeDryRunItemCount" -DefaultValue 0)
  blockedCloseDryRunItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "blockedCloseDryRunItemCount" -DefaultValue 0)
  readyCloseDryRunItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "readyCloseDryRunItemCount" -DefaultValue 0)
  remainingGapCount = [int](Get-PropertyOrDefault -Object $summary -Name "remainingGapCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates strict close dry-run shape only. It is not runtime proof, package publish, post-publish verification, rollback approval, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-dry-run-summary-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-strict-dry-run-summary-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Close Strict Dry-Run Summary Validation")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$(ConvertTo-MarkdownCell $validation.validationState)`` |")
$lines.Add("| closeDryRunItemCount | ``$($validation.closeDryRunItemCount)`` |")
$lines.Add("| blockedCloseDryRunItemCount | ``$($validation.blockedCloseDryRunItemCount)`` |")
$lines.Add("| readyCloseDryRunItemCount | ``$($validation.readyCloseDryRunItemCount)`` |")
$lines.Add("| remainingGapCount | ``$($validation.remainingGapCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close strict dry-run summary validation written to $jsonPath"
Write-Host "Release close strict dry-run summary validation markdown written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Release close strict dry-run summary validation has blocker failures: $($failedBlockers.Count)"
}
