[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-candidate-non-substitute-final-scan.json",
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

function Resolve-RepositoryPath { param([string]$Path) if ([System.IO.Path]::IsPathRooted($Path)) { return $Path } return Join-Path $RepositoryRoot $Path }
function Get-PropertyOrDefault { param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue) if ($null -eq $Object) { return $DefaultValue } if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value } return $DefaultValue }
function New-ValidationItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { throw "Release candidate non-substitute final scan not found: $resolvedInputPath" }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$checks = @((Get-PropertyOrDefault -Object $record -Name "substituteChecks" -DefaultValue @()))
$blockedChecks = @($checks | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) })
$promotedChecks = @($checks | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "promotedAsProof" -DefaultValue $false) })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-candidate-non-substitute-final-scan") -Severity "blocker" -Detail "recordKind must be release-candidate-non-substitute-final-scan.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "scanState" -DefaultValue "") -eq "blocked-release-candidate-non-substitute-final-scan-owner-proof-required") -Severity "blocker" -Detail "Non-substitute scan must stay blocked until real proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "checks-present" -Passed ($checks.Count -ge 14) -Severity "blocker" -Detail "Scan must cover all forbidden substitute kinds.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-promoted-substitutes" -Passed ($promotedChecks.Count -eq 0) -Severity "blocker" -Detail "No substitute kind may be promoted as proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-action-required" -Passed ($blockedChecks.Count -eq 0) -Severity "action-required" -Detail "Owner must replace substitute placeholders with real proof records.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "approvesPublicRelease" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseRecordProof" -DefaultValue $true)) -Severity "blocker" -Detail "Scan must not publish, approve, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-text" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must preserve non-proof language.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-release-candidate-non-substitute-final-scan" } else { "blocked-release-candidate-non-substitute-final-scan-owner-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "release-candidate-non-substitute-final-scan-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  scanState = [string](Get-PropertyOrDefault -Object $record -Name "scanState" -DefaultValue "")
  substituteCheckCount = $checks.Count
  blockedSubstituteCheckCount = $blockedChecks.Count
  promotedSubstituteCount = $promotedChecks.Count
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
  boundary = "Validation checks non-substitute final scan shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-non-substitute-final-scan-validation.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-non-substitute-final-scan-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Release Candidate Non-Substitute Final Scan Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| substituteCheckCount | ``$($validation.substituteCheckCount)`` |
| blockedSubstituteCheckCount | ``$($validation.blockedSubstituteCheckCount)`` |
| promotedSubstituteCount | ``$($validation.promotedSubstituteCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
| --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate non-substitute final scan validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Checks=$($validation.substituteCheckCount) Blocked=$($validation.blockedSubstituteCheckCount) Promoted=$($validation.promotedSubstituteCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Release candidate non-substitute final scan validation failed with $($failedBlockers.Count) blocker(s)." }
