[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-owner-input-bridge.json",
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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release close owner input bridge not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$bridgeState = [string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "")
$gates = @(Get-PropertyOrDefault -Object $record -Name "bridgeGates" -DefaultValue @())
$blockedGates = @($gates | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$readyGates = @($gates | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$strictCloseCommand = [string](Get-PropertyOrDefault -Object $record -Name "strictCloseCommand" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "release-close-owner-input-bridge") -Severity "blocker" -Detail "recordKind must be release-close-owner-input-bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "bridge-state" -Passed ($bridgeState -eq "blocked-release-close-owner-input-required" -or $bridgeState -eq "release-close-owner-input-ready") -Severity "blocker" -Detail "Bridge state must be blocked or ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "gate-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "bridgeGateCount" -DefaultValue 0) -ge 7 -and $gates.Count -ge 7) -Severity "blocker" -Detail "Bridge must include release close owner gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-gate-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedBridgeGateCount" -DefaultValue -1) -eq $blockedGates.Count) -Severity "blocker" -Detail "Blocked gate count must match gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-gate-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyBridgeGateCount" -DefaultValue -1) -eq $readyGates.Count) -Severity "blocker" -Detail "Ready gate count must match gates.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-command" -Passed ($strictCloseCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictCloseCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "Final close gate must remain Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-or-close" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not publish or close release issue.")) | Out-Null

foreach ($gate in $gates) {
  $gateId = [string](Get-PropertyOrDefault -Object $gate -Name "gateId" -DefaultValue "unknown-gate")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $gate -Name "state" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $gate -Name "requiredState" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $gate -Name "requiredAction" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $gate -Name "sourceArtifact" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$gateId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each bridge gate must include state, required state, action, and source artifact.")) | Out-Null
  if (-not [bool](Get-PropertyOrDefault -Object $gate -Name "ready" -DefaultValue $false)) {
    $items.Add((New-ValidationItem -Id "$gateId-owner-input-required" -Passed $false -Severity "action-required" -Detail "Release close owner input remains blocked until this gate reaches its required state.")) | Out-Null
  }
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-release-close-owner-input-bridge"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-release-close-owner-input-required"
}
else {
  "release-close-owner-input-ready"
}

$validation = [pscustomobject]@{
  recordKind = "release-close-owner-input-bridge-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  bridgeGateCount = $gates.Count
  blockedBridgeGateCount = $blockedGates.Count
  readyBridgeGateCount = $readyGates.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates release close owner input bridge shape only. It cannot publish packages, promote runtime proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "release-close-owner-input-bridge-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-owner-input-bridge-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Release Close Owner Input Bridge Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| bridgeGateCount | ``$($validation.bridgeGateCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close owner input bridge validation written to $jsonPath"
Write-Host "Release close owner input bridge validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Release close owner input bridge validation failed with $($failedBlockers.Count) blocker(s)."
}
