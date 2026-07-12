[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-release-close-hash-consistency-gate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

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
  throw "Final release close hash consistency gate not found: $resolvedInputPath"
}
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "hashLanes" -DefaultValue @()))
$mismatched = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "sha256Matches" -DefaultValue $false) })
$blocked = @($lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "") -match "^(blocked|missing|invalid)" })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-release-close-hash-consistency-gate") -Severity "blocker" -Detail "recordKind must be final-release-close-hash-consistency-gate.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "") -eq "blocked-final-release-close-hash-consistency-owner-proof-required") -Severity "blocker" -Detail "Hash gate must remain blocked until owner proof and close records exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "hash-lanes-present" -Passed ($lanes.Count -ge 8) -Severity "blocker" -Detail "Hash gate must cover final close source artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "hashes-match-current-files" -Passed ($mismatched.Count -eq 0) -Severity "blocker" -Detail "All listed local artifact hashes must match current files.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-proof-still-required" -Passed ($blocked.Count -eq 0) -Severity "action-required" -Detail "Blocked/missing lanes require Owner proof or strict validator convergence.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseRecordProof" -DefaultValue $true)) -Severity "blocker" -Detail "Hash gate must not publish, approve, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-text" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must preserve non-proof language.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-release-close-hash-consistency-gate" } else { "blocked-final-release-close-hash-consistency-owner-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "final-release-close-hash-consistency-gate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  gateState = [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "")
  hashLaneCount = $lanes.Count
  mismatchedHashCount = $mismatched.Count
  blockedHashLaneCount = $blocked.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  validationItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Validation checks final release close hash consistency gate shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-release-close-hash-consistency-gate-validation.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-hash-consistency-gate-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Final Release Close Hash Consistency Gate Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| hashLaneCount | ``$($validation.hashLaneCount)`` |
| mismatchedHashCount | ``$($validation.mismatchedHashCount)`` |
| blockedHashLaneCount | ``$($validation.blockedHashLaneCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
| --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release close hash consistency gate validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) HashLanes=$($validation.hashLaneCount) Mismatched=$($validation.mismatchedHashCount) Blocked=$($validation.blockedHashLaneCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final release close hash consistency gate validation failed with $($failedBlockers.Count) blocker(s)."
}
