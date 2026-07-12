[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-issue-close-owner-decision-input.template.json",
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

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-FileHashMatches {
  param([AllowNull()][object]$Path, [AllowNull()][object]$Sha256)
  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) { return $false }
  $resolved = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $false }
  $actual = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release issue close owner decision input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-issue-close-owner-decision-input") -Severity "blocker" -Detail "recordKind must be release-issue-close-owner-decision-input.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "ownerDecisionInputState" -DefaultValue "") -eq "blocked-release-issue-close-owner-decision-input-required") -Severity "blocker" -Detail "Template must remain blocked until owner fills final close decision.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Owner decision input must not publish, approve, promote proof, or close release issue.")) | Out-Null

foreach ($field in @("ownerName", "ownerEmail", "ownerDecisionTimestampUtc", "releaseIssueUrl", "selectedChannel", "approvedPublicPackageProofHash", "approvedPostPublishProofHash", "rollbackPlan", "rollbackOwner", "rollbackTrigger", "knownLimitationsAcknowledgement", "finalCloseDecision")) {
  $items.Add((New-ValidationItem -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be filled by release owner from real close-ready evidence.")) | Out-Null
}

foreach ($pair in @(
  @("release-evidence-bundle-hash", "releaseEvidenceBundlePath", "releaseEvidenceBundleSha256", "releaseEvidenceBundleSha256 must match releaseEvidenceBundlePath."),
  @("post-publish-validation-hash", "postPublishValidationPath", "postPublishValidationSha256", "postPublishValidationSha256 must match postPublishValidationPath.")
)) {
  $items.Add((New-ValidationItem -Id $pair[0] -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name $pair[1] -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name $pair[2] -DefaultValue "")) -Severity "action-required" -Detail $pair[3])) | Out-Null
}

$strictCommand = [string](Get-PropertyOrDefault -Object $record -Name "strictCloseValidatorCommand" -DefaultValue "")
$items.Add((New-ValidationItem -Id "strict-close-validator-command" -Passed ($strictCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "strictCloseValidatorCommand must use Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-release-issue-close-owner-decision-input" } else { "blocked-release-issue-close-owner-decision-input-required" }

$validation = [pscustomobject]@{
  recordKind = "release-issue-close-owner-decision-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Validation checks owner close decision input shape only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-issue-close-owner-decision-input-validation.json"
$markdownPath = Join-Path $OutputRoot "release-issue-close-owner-decision-input-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Release Issue Close Owner Decision Input Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Release issue close owner decision input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Release issue close owner decision input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
