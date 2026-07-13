[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-SubstituteCheck {
  param([string]$Id, [string]$ForbiddenKind, [string]$SearchHint, [string]$RequiredOutcome)
  [pscustomobject]@{
    id = $Id
    forbiddenKind = $ForbiddenKind
    searchHint = $SearchHint
    requiredOutcome = $RequiredOutcome
    promotedAsProof = $false
    blocked = $true
    boundary = "Release candidate non-substitute final scan check only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$bundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$publishabilityAudit = Read-JsonOrNull "artifacts\final-release\release-candidate-final-publishability-audit.json"
$ownerRoadmap = Read-JsonOrNull "artifacts\final-release\release-candidate-owner-action-roadmap.json"

$evidenceItems = @((Get-PropertyOrDefault -Object $bundle -Name "evidenceItems" -DefaultValue @()))
$sourceArtifacts = @((Get-PropertyOrDefault -Object $bundle -Name "sourceArtifacts" -DefaultValue @()))
$classificationFindingCount = [int](Get-PropertyOrDefault -Object $classificationAudit -Name "findingCount" -DefaultValue 1)

$checks = @(
  New-SubstituteCheck -Id "local-nupkg" -ForbiddenKind "local .nupkg" -SearchHint ".nupkg" -RequiredOutcome "Local packages cannot satisfy public package proof."
  New-SubstituteCheck -Id "local-feed" -ForbiddenKind "local feed" -SearchHint "local feed" -RequiredOutcome "Local feeds cannot satisfy public package proof."
  New-SubstituteCheck -Id "project-reference" -ForbiddenKind "ProjectReference" -SearchHint "ProjectReference" -RequiredOutcome "ProjectReference cannot satisfy clean consumer proof."
  New-SubstituteCheck -Id "direct-nupkg" -ForbiddenKind "direct nupkg" -SearchHint "direct nupkg" -RequiredOutcome "Direct package file references cannot satisfy public channel proof."
  New-SubstituteCheck -Id "template" -ForbiddenKind "template" -SearchHint "template" -RequiredOutcome "Templates cannot be promoted as proof."
  New-SubstituteCheck -Id "draft" -ForbiddenKind "draft" -SearchHint "draft" -RequiredOutcome "Draft records cannot be promoted as proof."
  New-SubstituteCheck -Id "dry-run" -ForbiddenKind "dry-run" -SearchHint "dry-run" -RequiredOutcome "Dry-runs cannot be promoted as runtime or publish proof."
  New-SubstituteCheck -Id "runbook" -ForbiddenKind "runbook" -SearchHint "runbook" -RequiredOutcome "Runbooks remain guidance only."
  New-SubstituteCheck -Id "dashboard" -ForbiddenKind "dashboard" -SearchHint "dashboard" -RequiredOutcome "Dashboards remain status views only."
  New-SubstituteCheck -Id "audit-pack" -ForbiddenKind "audit pack" -SearchHint "audit pack" -RequiredOutcome "Audit packs do not replace real execution proof."
  New-SubstituteCheck -Id "hash-only-lane" -ForbiddenKind "hash slot" -SearchHint "hash" -RequiredOutcome "Hash consistency alone cannot close release or publish."
  New-SubstituteCheck -Id "candidate" -ForbiddenKind "candidate" -SearchHint "candidate" -RequiredOutcome "Candidates require owner-filled real proof before promotion."
  New-SubstituteCheck -Id "local-only-scan" -ForbiddenKind "local-only scan" -SearchHint "scan" -RequiredOutcome "Local scans cannot replace external runtime or post-publish proof."
  New-SubstituteCheck -Id "manual-handoff" -ForbiddenKind "manual handoff" -SearchHint "handoff" -RequiredOutcome "Manual handoff is not proof without executed records."
)

$blockedChecks = @($checks | Where-Object { [bool]$_.blocked })

$record = [pscustomobject]@{
  recordKind = "release-candidate-non-substitute-final-scan"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  scanState = "blocked-release-candidate-non-substitute-final-scan-owner-proof-required"
  substituteCheckCount = $checks.Count
  blockedSubstituteCheckCount = $blockedChecks.Count
  promotedSubstituteCount = @($checks | Where-Object { [bool]$_.promotedAsProof }).Count
  evidenceItemCount = $evidenceItems.Count
  sourceArtifactCount = $sourceArtifacts.Count
  classificationFindingCount = $classificationFindingCount
  substituteChecks = @($checks)
  sourceArtifacts = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-evidence-classification-audit.json",
    "artifacts/final-release/release-candidate-final-publishability-audit.json",
    "artifacts/final-release/release-candidate-owner-action-roadmap.json"
  )
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
  boundary = "Release candidate non-substitute final scan is a blocked substitute audit only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-non-substitute-final-scan.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-non-substitute-final-scan.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $checks | ForEach-Object {
  "| ``$($_.id)`` | $($_.forbiddenKind) | ``$($_.blocked)`` | ``$($_.promotedAsProof)`` | $($_.requiredOutcome.Replace("|", "\|")) |"
}

$markdown = @"
# Release Candidate Non-Substitute Final Scan

| Field | Value |
| --- | --- |
| scanState | ``$($record.scanState)`` |
| substituteCheckCount | ``$($record.substituteCheckCount)`` |
| blockedSubstituteCheckCount | ``$($record.blockedSubstituteCheckCount)`` |
| promotedSubstituteCount | ``$($record.promotedSubstituteCount)`` |
| classificationFindingCount | ``$($record.classificationFindingCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Substitute Checks

| ID | Forbidden Kind | Blocked | Promoted As Proof | Required Outcome |
| --- | --- | ---: | ---: | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate non-substitute final scan written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ScanState=$($record.scanState) Checks=$($record.substituteCheckCount) Blocked=$($record.blockedSubstituteCheckCount) Promoted=$($record.promotedSubstituteCount)"
