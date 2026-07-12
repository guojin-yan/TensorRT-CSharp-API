[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/owner-only-publish-execution-candidate.json",
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

. (Join-Path $PSScriptRoot "OwnerRealEvidenceInput.Common.ps1")

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$inputFullPath = Resolve-OwnerRepositoryPath -RepositoryRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Missing owner-only publish execution candidate: $InputPath"
}

$candidate = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = $candidate | ConvertTo-Json -Depth 14
$commands = @($candidate.manualCommands)

$items = @(
  New-OwnerValidationItem -Id "record-kind" -Passed ([string]$candidate.recordKind -eq "owner-only-publish-execution-candidate") -Severity "blocker" -Detail "Candidate must use owner-only-publish-execution-candidate recordKind."
  New-OwnerValidationItem -Id "owner-manual-only" -Passed ([bool]$candidate.ownerManualOnly -and @($commands | Where-Object { [string]$_.mode -ne "owner-manual-only" }).Count -eq 0) -Severity "blocker" -Detail "Every command must be owner-manual-only."
  New-OwnerValidationItem -Id "non-executing-flags" -Passed (-not [bool]$candidate.performsPublish -and -not [bool]$candidate.canPublishPublicly -and -not [bool]$candidate.canCloseReleaseIssue -and -not [bool]$candidate.executesDotnetNugetPush -and -not [bool]$candidate.uploadsGitHubReleaseAssets -and -not [bool]$candidate.closesReleaseIssue) -Severity "blocker" -Detail "Candidate must not execute publish, upload assets, or close release issue."
  New-OwnerValidationItem -Id "blocked-without-real-input" -Passed (([int]$candidate.acceptedLaneCount -eq 6 -and [int]$candidate.blockedLaneCount -eq 0) -or ([string]$candidate.candidateState -eq "blocked-owner-real-evidence-required")) -Severity "blocker" -Detail "Candidate must remain blocked unless all six real evidence lanes pass."
  New-OwnerValidationItem -Id "manual-commands-present" -Passed (@($commands).Count -ge 4 -and $raw.Contains("dotnet nuget push") -and $raw.Contains("GitHub Release assets") -and $raw.Contains("public-channel install")) -Severity "blocker" -Detail "Candidate must include owner-manual publication and post-publish checklist commands."
  New-OwnerValidationItem -Id "forbidden-substitutes-visible" -Passed ((@($script:OwnerRealEvidenceForbiddenSubstitutes | Where-Object { $raw -notmatch [regex]::Escape($_) })).Count -eq 0) -Severity "blocker" -Detail "Candidate must keep forbidden substitutes visible."
)

$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$failedActionRequiredCount = if ($failedBlockerCount -eq 0 -and [string]$candidate.candidateState -eq "blocked-owner-real-evidence-required") { 6 } else { 0 }
$validationState = if ($failedBlockerCount -eq 0) {
  if ([string]$candidate.candidateState -eq "ready-for-owner-manual-publish-review") { "owner-only-publish-execution-candidate-ready-for-owner-review" } else { "blocked-owner-only-publish-execution-candidate-valid" }
}
else {
  "failed-owner-only-publish-execution-candidate"
}

$report = [pscustomobject]@{
  recordKind = "owner-only-publish-execution-candidate-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  sourceCandidate = $InputPath
  candidateState = [string]$candidate.candidateState
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  validationItems = @($items)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  executesDotnetNugetPush = $false
  uploadsGitHubReleaseAssets = $false
}

$jsonPath = Join-Path $OutputRoot "owner-only-publish-execution-candidate-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-only-publish-execution-candidate-validation.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-OwnerMarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-OwnerMarkdownCell $item.severity)`` | $(ConvertTo-OwnerMarkdownCell $item.detail) |"
}

$markdown = @"
# Owner-Only Publish Execution Candidate Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- candidateState: ``$($report.candidateState)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- failedActionRequiredCount: ``$($report.failedActionRequiredCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- executesDotnetNugetPush: ``False``
- uploadsGitHubReleaseAssets: ``False``

## Validation Items

| Item | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Owner-only publish execution candidate validation failed with $failedBlockerCount blocker(s)."
}

Write-Output "Owner-only publish execution candidate validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState FailedBlockerCount=$failedBlockerCount FailedActionRequiredCount=$failedActionRequiredCount"
