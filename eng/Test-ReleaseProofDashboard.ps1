[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-proof-dashboard.json",
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

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

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
  throw "Release proof dashboard not found: $resolvedInputPath"
}

$dashboard = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8

$items.Add((New-ValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $dashboard -Name "recordKind" -DefaultValue "") -eq "release-proof-dashboard") "blocker" "Dashboard must use recordKind=release-proof-dashboard.")) | Out-Null
$items.Add((New-ValidationItem "dashboard-state-blocked" ([string](Get-PropertyOrDefault -Object $dashboard -Name "dashboardState" -DefaultValue "") -eq "blocked-owner-proof-required") "blocker" "Dashboard must remain blocked until owner proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem "does-not-publish" (-not [bool](Get-PropertyOrDefault -Object $dashboard -Name "performsPublish" -DefaultValue $true)) "blocker" "Dashboard validator must not publish packages.")) | Out-Null
$items.Add((New-ValidationItem "does-not-promote-public-release" ((-not [bool](Get-PropertyOrDefault -Object $dashboard -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $dashboard -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $dashboard -Name "canPromoteRuntimeProof" -DefaultValue $true))) "blocker" "Dashboard must not publish, close the release issue, or promote runtime proof.")) | Out-Null
$items.Add((New-ValidationItem "owner-proof-action-required-lane-count" ([int](Get-PropertyOrDefault -Object $dashboard -Name "ownerProofActionRequiredLaneCount" -DefaultValue -1) -eq 4) "blocker" "Dashboard must retain four owner action-required proof lanes.")) | Out-Null

$lanes = @(Get-PropertyOrDefault -Object $dashboard -Name "lanes" -DefaultValue @())
$requiredLaneIds = @("real-model-runtime", "package-consumer-runtime", "post-publish-verification", "public-owner-confirmation")
foreach ($requiredLaneId in $requiredLaneIds) {
  $lane = @($lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq $requiredLaneId } | Select-Object -First 1)
  $exists = $lane.Count -eq 1
  $blocked = $exists -and [bool](Get-PropertyOrDefault -Object $lane[0] -Name "blocked" -DefaultValue $false)
  $notPromoted = $exists -and (-not [bool](Get-PropertyOrDefault -Object $lane[0] -Name "canPromoteProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $lane[0] -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $lane[0] -Name "canCloseReleaseIssue" -DefaultValue $true))
  $items.Add((New-ValidationItem "lane-$requiredLaneId-present-blocked" ($exists -and $blocked -and $notPromoted) "blocker" "Lane $requiredLaneId must exist, remain blocked, and not promote proof.")) | Out-Null
}

$supportingEvidence = @(Get-PropertyOrDefault -Object $dashboard -Name "supportingEvidence" -DefaultValue @())
$supportingEvidenceOnly = $supportingEvidence.Count -ge 1 -and @(
  $supportingEvidence | Where-Object {
    $state = [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "")
    $evidenceBoundary = [string](Get-PropertyOrDefault -Object $_ -Name "boundary" -DefaultValue "")
    $promotes = [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteProof" -DefaultValue $false) -or
      [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $false) -or
      [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $false)

    $isNonProofState = $state -in @("supporting-evidence-only", "template-owner-input-required", "template-only", "owner-action-required")
    -not $isNonProofState -or $promotes -or -not ($evidenceBoundary -match "(?i)not|cannot|do not|does not")
  }
).Count -eq 0
$items.Add((New-ValidationItem "supporting-evidence-only-not-proof" $supportingEvidenceOnly "blocker" "Supporting evidence entries must remain non-proof, non-promoting diagnostics or owner-input checklists.")) | Out-Null

foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "build-only", "dry-run", "template", "skipped run")) {
  $items.Add((New-ValidationItem "forbidden-substitute-$($marker.Replace(' ', '-').Replace('.', 'dot'))" ($raw.Contains($marker, [StringComparison]::OrdinalIgnoreCase)) "blocker" "Forbidden substitute '$marker' must remain visible as a non-proof boundary.")) | Out-Null
}

$boundary = [string](Get-PropertyOrDefault -Object $dashboard -Name "boundary" -DefaultValue "")
$items.Add((New-ValidationItem "dashboard-boundary-non-proof" ($boundary.Contains("does not publish", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("does not", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("replace real", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Dashboard boundary must state that it does not publish or replace real proof.")) | Out-Null

$failedBlockerCount = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockerCount -eq 0) { "blocked-owner-proof-required" } else { "failed-release-proof-dashboard-validation" }

$report = [pscustomobject]@{
  recordKind = "release-proof-dashboard-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  inputPath = $resolvedInputPath
  validationState = $validationState
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  failedBlockerCount = [int]$failedBlockerCount
  laneCount = [int]$lanes.Count
  blockedLaneCount = @($lanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "blocked" -DefaultValue $false) }).Count
  validationItems = [object[]]@($items.ToArray())
  supportingEvidence = [object[]]@($supportingEvidence)
  forbiddenSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "build-only", "dry-run", "template", "skipped run")
  boundary = "Release dashboard validation is a strict gate over dashboard structure only. It does not publish, does not promote proof, and cannot replace real owner/public/post-publish evidence."
}

$jsonPath = Join-Path $OutputRoot "release-proof-dashboard-validation.json"
$markdownPath = Join-Path $OutputRoot "release-proof-dashboard-validation.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Release Proof Dashboard Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- recordKind: ``$($report.recordKind)``
- validationState: ``$($report.validationState)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- laneCount: ``$($report.laneCount)``
- blockedLaneCount: ``$($report.blockedLaneCount)``

## Validation Items

| Id | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release proof dashboard validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) FailedBlockers=$($report.failedBlockerCount) LaneCount=$($report.laneCount) BlockedLaneCount=$($report.blockedLaneCount)"

if ($Strict.IsPresent -and $failedBlockerCount -gt 0) {
  exit 1
}
