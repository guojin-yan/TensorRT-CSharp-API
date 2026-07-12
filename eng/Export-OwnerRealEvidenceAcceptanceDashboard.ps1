[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts/final-release/owner-real-evidence-input-import.json",
  [string]$ValidationPath = "artifacts/final-release/owner-real-evidence-input-validation.json",
  [string]$FinalActionMapPath = "artifacts/final-release/final-publish-action-required-evidence-map.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

$import = Read-OwnerJsonOrNull -RepositoryRoot $RepositoryRoot -Path $ImportPath
$validation = Read-OwnerJsonOrNull -RepositoryRoot $RepositoryRoot -Path $ValidationPath
$finalMap = Read-OwnerJsonOrNull -RepositoryRoot $RepositoryRoot -Path $FinalActionMapPath

$laneResults = if ($null -ne $import) { @($import.lanes) } else { @() }
$finalActionCount = if ($null -ne $finalMap) { [int]$finalMap.actionRequiredCount } else { 6 }
$acceptedLaneCount = if ($null -ne $import) { [int]$import.acceptedLaneCount } else { 0 }
$blockedLaneCount = if ($null -ne $import) { [int]$import.blockedLaneCount } else { 6 }
$failedBlockerCount = if ($null -ne $validation) { [int]$validation.failedBlockerCount } else { 1 }
$importState = if ($null -ne $import) { [string]$import.importState } else { "blocked-owner-input-file-required" }
$validationState = if ($null -ne $validation) { [string]$validation.validationState } else { "missing-owner-real-evidence-input-validation" }
$allAccepted = $acceptedLaneCount -eq 6 -and $blockedLaneCount -eq 0 -and $failedBlockerCount -eq 0 -and $importState -eq "accepted-real-owner-evidence-input"

$dashboard = [pscustomobject]@{
  recordKind = "owner-real-evidence-acceptance-dashboard"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  dashboardState = if ($allAccepted) { "accepted-real-owner-evidence-ready-for-owner-publish-authorization-review" } else { "blocked-owner-real-evidence-required" }
  sourceImport = $ImportPath
  sourceValidation = $ValidationPath
  sourceFinalActionMap = $FinalActionMapPath
  finalActionRequiredCount = [int]$finalActionCount
  importState = $importState
  validationState = $validationState
  acceptedLaneCount = [int]$acceptedLaneCount
  blockedLaneCount = [int]$blockedLaneCount
  failedBlockerCount = [int]$failedBlockerCount
  canPrepareOwnerOnlyPublishCandidate = [bool]$allAccepted
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  executesDotnetNugetPush = $false
  uploadsGitHubReleaseAssets = $false
  forbiddenSubstitutes = @($script:OwnerRealEvidenceForbiddenSubstitutes)
  lanes = @($laneResults)
  boundary = "Dashboard summarizes owner real evidence acceptance only. It cannot substitute proof and cannot publish, upload assets, or close release issues."
}

$jsonPath = Join-Path $OutputRoot "owner-real-evidence-acceptance-dashboard.json"
$markdownPath = Join-Path $OutputRoot "owner-real-evidence-acceptance-dashboard.md"
$dashboard | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($lane in $laneResults) {
  "| ``$(ConvertTo-OwnerMarkdownCell $lane.laneId)`` | ``$(ConvertTo-OwnerMarkdownCell $lane.laneState)`` | ``$($lane.artifactSha256Matches)`` | ``$($lane.logSha256Matches)`` | ``$($lane.recordSha256Matches)`` |"
}
if ($rows.Count -eq 0) {
  $rows = @("| owner-input-file | ``blocked-owner-input-file-required`` | ``False`` | ``False`` | ``False`` |")
}

$markdown = @"
# Owner Real Evidence Acceptance Dashboard

Generated at: ``$($dashboard.generatedAtUtc)``

## Summary

- dashboardState: ``$($dashboard.dashboardState)``
- importState: ``$($dashboard.importState)``
- validationState: ``$($dashboard.validationState)``
- finalActionRequiredCount: ``$($dashboard.finalActionRequiredCount)``
- acceptedLaneCount: ``$($dashboard.acceptedLaneCount)``
- blockedLaneCount: ``$($dashboard.blockedLaneCount)``
- failedBlockerCount: ``$($dashboard.failedBlockerCount)``
- canPrepareOwnerOnlyPublishCandidate: ``$($dashboard.canPrepareOwnerOnlyPublishCandidate)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Lanes

| Lane | State | Artifact SHA | Log SHA | Record SHA |
| --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($dashboard.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner real evidence acceptance dashboard written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "DashboardState=$($dashboard.dashboardState) AcceptedLaneCount=$acceptedLaneCount BlockedLaneCount=$blockedLaneCount"
