[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/public-publish-authorization-preflight.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
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

$inputFullPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Missing public publish authorization preflight: $InputPath"
}

$preflight = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$preflightText = $preflight | ConvertTo-Json -Depth 18
$requirements = @($preflight.requirements)

$items = @(
  New-ValidationItem -Id "record-kind" -Passed ([string]$preflight.recordKind -eq "public-publish-authorization-preflight") -Severity "blocker" -Detail "Preflight must use public-publish-authorization-preflight recordKind."
  New-ValidationItem -Id "requirements-present" -Passed (@($requirements).Count -ge 6) -Severity "blocker" -Detail "Preflight must include public source, package hashes, clean consumer, PostPublish, owner review, and release close requirements."
  New-ValidationItem -Id "required-authorization-categories-present" -Passed ((@("public-package-channel", "package-hash-chain", "package-consumer-runtime", "post-publish-verification", "owner-release-decision", "release-close-authorization") | Where-Object { @($requirements.category) -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "All public publish authorization categories must be present."
  New-ValidationItem -Id "non-executing-command-policy" -Passed (-not [bool]$preflight.commandPolicy.executesDotnetNugetPush -and -not [bool]$preflight.commandPolicy.uploadsGitHubReleaseAssets -and -not [bool]$preflight.commandPolicy.storesTokens -and -not [bool]$preflight.commandPolicy.closesReleaseIssue) -Severity "blocker" -Detail "Preflight must not execute publish commands, upload assets, store tokens, or close issues."
  New-ValidationItem -Id "flags-remain-false" -Passed (-not [bool]$preflight.performsPublish -and -not [bool]$preflight.canPublishPublicly -and -not [bool]$preflight.canCloseReleaseIssue -and -not [bool]$preflight.canPromotePackageConsumerRuntime -and -not [bool]$preflight.canPromoteRuntimeProof) -Severity "blocker" -Detail "Preflight must not claim public publish, release close, or proof promotion."
  New-ValidationItem -Id "owner-import-linked" -Passed ($preflightText.Contains("owner-real-evidence-import-packet.json") -and [int]$preflight.ownerImportLaneCount -eq 6) -Severity "blocker" -Detail "Preflight must link the Owner real evidence import packet and its six lanes."
  New-ValidationItem -Id "forbidden-substitutes-listed" -Passed ($preflightText.Contains("template-only record") -and $preflightText.Contains("dashboard-only record") -and $preflightText.Contains("local feed") -and $preflightText.Contains("ProjectReference") -and $preflightText.Contains("direct nupkg")) -Severity "blocker" -Detail "Preflight must explicitly block forbidden substitutes."
)

$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$failedActionRequiredCount = if ($failedBlockerCount -eq 0) { [int]@($requirements).Count } else { 0 }
$validationState = if ($failedBlockerCount -eq 0) { "blocked-owner-public-publish-authorization-required-preflight-valid" } else { "failed-public-publish-authorization-preflight" }

$report = [pscustomobject]@{
  recordKind = "public-publish-authorization-preflight-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  sourcePreflight = $InputPath
  requirementCount = @($requirements).Count
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  validationItems = @($items)
  performsPublish = $false
  executesDotnetNugetPush = $false
  uploadsGitHubReleaseAssets = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  boundary = "This validation confirms authorization-preflight structure only. It does not authorize publishing, package upload, release close, or proof promotion."
}

$jsonPath = Join-Path $OutputRoot "public-publish-authorization-preflight-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-authorization-preflight-validation.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Public Publish Authorization Preflight Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- requirementCount: ``$($report.requirementCount)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- failedActionRequiredCount: ``$($report.failedActionRequiredCount)``
- performsPublish: ``False``
- executesDotnetNugetPush: ``False``
- uploadsGitHubReleaseAssets: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Items

| Item | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Public publish authorization preflight validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) FailedBlockerCount=$($report.failedBlockerCount) ActionRequired=$($report.failedActionRequiredCount)"

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Public publish authorization preflight validation failed with $failedBlockerCount blocker(s)."
}
