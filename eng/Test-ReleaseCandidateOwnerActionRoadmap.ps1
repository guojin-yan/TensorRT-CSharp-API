[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-candidate-owner-action-roadmap.json",
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
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { throw "Release candidate owner action roadmap not found: $resolvedInputPath" }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$actions = @((Get-PropertyOrDefault -Object $record -Name "ownerActions" -DefaultValue @()))
$blockedActions = @($actions | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$requiredIds = @("owner-confirms-public-channel", "public-publish-real-result", "public-package-download-hash", "clean-consumer-smoke", "compatible-host-runtime-proof", "linux-runner-proof", "real-model-runtime-proof", "post-publish-verification", "forbidden-substitute-final-check", "rollback-review", "final-close-record", "release-issue-close-decision")
$actionIds = @($actions | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-candidate-owner-action-roadmap") -Severity "blocker" -Detail "recordKind must be release-candidate-owner-action-roadmap.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "roadmapState" -DefaultValue "") -eq "blocked-release-candidate-owner-action-roadmap-owner-proof-required") -Severity "blocker" -Detail "Roadmap must stay blocked until real owner proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "actions-present" -Passed (@($requiredIds | Where-Object { $actionIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Roadmap must expose all final owner actions.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-action-required" -Passed ($blockedActions.Count -eq 0) -Severity "action-required" -Detail "Owner must complete all roadmap actions.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "approvesPublicRelease" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseRecordProof" -DefaultValue $true)) -Severity "blocker" -Detail "Roadmap must not publish, approve, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-text" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must preserve non-proof language.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-release-candidate-owner-action-roadmap" } else { "blocked-release-candidate-owner-action-roadmap-owner-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "release-candidate-owner-action-roadmap-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  roadmapState = [string](Get-PropertyOrDefault -Object $record -Name "roadmapState" -DefaultValue "")
  ownerActionCount = $actions.Count
  blockedOwnerActionCount = $blockedActions.Count
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
  boundary = "Validation checks owner action roadmap shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-owner-action-roadmap-validation.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-owner-action-roadmap-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Release Candidate Owner Action Roadmap Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| ownerActionCount | ``$($validation.ownerActionCount)`` |
| blockedOwnerActionCount | ``$($validation.blockedOwnerActionCount)`` |
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

Write-Host "Release candidate owner action roadmap validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Actions=$($validation.ownerActionCount) Blocked=$($validation.blockedOwnerActionCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Release candidate owner action roadmap validation failed with $($failedBlockers.Count) blocker(s)." }
