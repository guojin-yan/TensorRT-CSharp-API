[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\runtime-proof-lane-dry-run-summary.json",
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
  throw "Runtime proof lane dry-run summary not found: $resolvedInputPath"
}

$summary = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$laneItems = @(Get-PropertyOrDefault -Object $summary -Name "laneItems" -DefaultValue @())
$items = New-Object System.Collections.Generic.List[object]
$blockedLaneItems = @($laneItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneState" -DefaultValue "") -eq "blocked-runtime-proof-lane-real-evidence-required" })
$readyLaneItems = @($laneItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForRuntimeProof" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $summary -Name "recordKind" -DefaultValue "") -eq "runtime-proof-lane-dry-run-summary") -Severity "blocker" -Detail "recordKind must be runtime-proof-lane-dry-run-summary.")) | Out-Null
$items.Add((New-ValidationItem -Id "dry-run-state" -Passed ([string](Get-PropertyOrDefault -Object $summary -Name "dryRunState" -DefaultValue "") -eq "blocked-runtime-proof-real-evidence-required") -Severity "blocker" -Detail "Lane dry-run summary must remain blocked until real proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "laneItemCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Lane dry-run summary must cover 6 proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-lane-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "blockedLaneItemCount" -DefaultValue -1) -eq $blockedLaneItems.Count -and $blockedLaneItems.Count -ge 6) -Severity "blocker" -Detail "Default lane dry-run must keep all lanes blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-lane-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "readyLaneItemCount" -DefaultValue -1) -eq $readyLaneItems.Count -and $readyLaneItems.Count -eq 0) -Severity "blocker" -Detail "Default lane dry-run must not claim ready runtime proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "missing-evidence-count" -Passed ([int](Get-PropertyOrDefault -Object $summary -Name "missingRealEvidenceFieldCount" -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "Lane dry-run must surface missing real evidence fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $summary -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Lane dry-run must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $summary -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Lane dry-run must not publish or approve public publication.")) | Out-Null

foreach ($laneItem in $laneItems) {
  $laneItemId = [string](Get-PropertyOrDefault -Object $laneItem -Name "laneItemId" -DefaultValue "unknown-lane-item")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $laneItem -Name "executionInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $laneItem -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $laneItem -Name "requiredNextCommand" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $laneItem -Name "validatorCommand" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $laneItem -Name "closeImpact" -DefaultValue "")) -and
    [int](Get-PropertyOrDefault -Object $laneItem -Name "missingRealEvidenceFieldCount" -DefaultValue 0) -gt 0
  $items.Add((New-ValidationItem -Id "$laneItemId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each lane dry-run item must include identity, missing fields, next command, validator command, and close impact.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$laneItemId-real-proof-required" -Passed ([bool](Get-PropertyOrDefault -Object $laneItem -Name "readyForRuntimeProof" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must provide real external proof evidence before this lane can become runtime proof.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-runtime-proof-lane-dry-run-summary" } else { "blocked-runtime-proof-real-evidence-required" }

$validation = [pscustomobject]@{
  recordKind = "runtime-proof-lane-dry-run-summary-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  laneItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "laneItemCount" -DefaultValue 0)
  blockedLaneItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "blockedLaneItemCount" -DefaultValue 0)
  readyLaneItemCount = [int](Get-PropertyOrDefault -Object $summary -Name "readyLaneItemCount" -DefaultValue 0)
  missingRealEvidenceFieldCount = [int](Get-PropertyOrDefault -Object $summary -Name "missingRealEvidenceFieldCount" -DefaultValue 0)
  substituteBlockerCount = [int](Get-PropertyOrDefault -Object $summary -Name "substituteBlockerCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates lane dry-run shape only. It is not runtime proof, package publish, post-publish verification, rollback approval, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "runtime-proof-lane-dry-run-summary-validation.json"
$markdownPath = Join-Path $OutputRoot "runtime-proof-lane-dry-run-summary-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Runtime Proof Lane Dry-Run Summary Validation")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$(ConvertTo-MarkdownCell $validation.validationState)`` |")
$lines.Add("| laneItemCount | ``$($validation.laneItemCount)`` |")
$lines.Add("| blockedLaneItemCount | ``$($validation.blockedLaneItemCount)`` |")
$lines.Add("| readyLaneItemCount | ``$($validation.readyLaneItemCount)`` |")
$lines.Add("| missingRealEvidenceFieldCount | ``$($validation.missingRealEvidenceFieldCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime proof lane dry-run summary validation written to $jsonPath"
Write-Host "Runtime proof lane dry-run summary validation markdown written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Runtime proof lane dry-run summary validation has blocker failures: $($failedBlockers.Count)"
}
