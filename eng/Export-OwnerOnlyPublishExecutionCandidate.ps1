[CmdletBinding()]
param(
  [string]$AcceptanceDashboardPath = "artifacts/final-release/owner-real-evidence-acceptance-dashboard.json",
  [string]$PublicPublishAuthorizationPreflightPath = "artifacts/final-release/public-publish-authorization-preflight.json",
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

$dashboard = Read-OwnerJsonOrNull -RepositoryRoot $RepositoryRoot -Path $AcceptanceDashboardPath
$preflight = Read-OwnerJsonOrNull -RepositoryRoot $RepositoryRoot -Path $PublicPublishAuthorizationPreflightPath
$finalMap = Read-OwnerJsonOrNull -RepositoryRoot $RepositoryRoot -Path $FinalActionMapPath

$dashboardState = if ($null -ne $dashboard) { [string]$dashboard.dashboardState } else { "missing-owner-real-evidence-acceptance-dashboard" }
$acceptedLaneCount = if ($null -ne $dashboard) { [int]$dashboard.acceptedLaneCount } else { 0 }
$blockedLaneCount = if ($null -ne $dashboard) { [int]$dashboard.blockedLaneCount } else { 6 }
$requirementCount = if ($null -ne $preflight) { [int]$preflight.requirementCount } else { 0 }
$finalActionCount = if ($null -ne $finalMap) { [int]$finalMap.actionRequiredCount } else { 6 }
$candidateUnblocked = $dashboardState -eq "accepted-real-owner-evidence-ready-for-owner-publish-authorization-review" -and $acceptedLaneCount -eq 6 -and $blockedLaneCount -eq 0

$manualCommands = @(
  [pscustomobject]@{
    id = "owner-review-public-package-channel"
    mode = "owner-manual-only"
    command = "Review public package identity, source URL, version, and package hashes from owner-real-evidence-input-import.json."
    executesDotnetNugetPush = $false
    uploadsGitHubReleaseAssets = $false
  },
  [pscustomobject]@{
    id = "owner-manual-nuget-publish"
    mode = "owner-manual-only"
    command = "Owner may run dotnet nuget push only outside automation after all real evidence gates are accepted."
    executesDotnetNugetPush = $false
    uploadsGitHubReleaseAssets = $false
  },
  [pscustomobject]@{
    id = "owner-manual-github-release-assets"
    mode = "owner-manual-only"
    command = "Owner may upload GitHub Release assets only outside automation after hashes and rollback plan are accepted."
    executesDotnetNugetPush = $false
    uploadsGitHubReleaseAssets = $false
  },
  [pscustomobject]@{
    id = "owner-post-publish-verification"
    mode = "owner-manual-only"
    command = "After manual public publication, owner must run public-channel install and smoke verification and import the result."
    executesDotnetNugetPush = $false
    uploadsGitHubReleaseAssets = $false
  }
)

$candidate = [pscustomobject]@{
  recordKind = "owner-only-publish-execution-candidate"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  candidateState = if ($candidateUnblocked) { "ready-for-owner-manual-publish-review" } else { "blocked-owner-real-evidence-required" }
  sourceAcceptanceDashboard = $AcceptanceDashboardPath
  sourcePublicPublishAuthorizationPreflight = $PublicPublishAuthorizationPreflightPath
  sourceFinalActionMap = $FinalActionMapPath
  dashboardState = $dashboardState
  acceptedLaneCount = [int]$acceptedLaneCount
  blockedLaneCount = [int]$blockedLaneCount
  finalActionRequiredCount = [int]$finalActionCount
  publicPublishAuthorizationRequirementCount = [int]$requirementCount
  ownerManualOnly = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  executesDotnetNugetPush = $false
  uploadsGitHubReleaseAssets = $false
  closesReleaseIssue = $false
  manualCommands = @($manualCommands)
  forbiddenSubstitutes = @($script:OwnerRealEvidenceForbiddenSubstitutes)
  boundary = "This candidate is an owner-manual-only checklist. Automation must not run dotnet nuget push, upload GitHub Release assets, or close release issues."
}

$jsonPath = Join-Path $OutputRoot "owner-only-publish-execution-candidate.json"
$markdownPath = Join-Path $OutputRoot "owner-only-publish-execution-candidate.md"
$candidate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($command in $manualCommands) {
  "| ``$(ConvertTo-OwnerMarkdownCell $command.id)`` | ``$(ConvertTo-OwnerMarkdownCell $command.mode)`` | $(ConvertTo-OwnerMarkdownCell $command.command) | ``False`` | ``False`` |"
}

$markdown = @"
# Owner-Only Publish Execution Candidate

Generated at: ``$($candidate.generatedAtUtc)``

## Summary

- candidateState: ``$($candidate.candidateState)``
- dashboardState: ``$($candidate.dashboardState)``
- acceptedLaneCount: ``$($candidate.acceptedLaneCount)``
- blockedLaneCount: ``$($candidate.blockedLaneCount)``
- ownerManualOnly: ``True``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- executesDotnetNugetPush: ``False``
- uploadsGitHubReleaseAssets: ``False``

## Manual Commands

| ID | Mode | Command | Executes Push | Uploads Assets |
| --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($candidate.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner-only publish execution candidate written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "CandidateState=$($candidate.candidateState) AcceptedLaneCount=$acceptedLaneCount BlockedLaneCount=$blockedLaneCount"
