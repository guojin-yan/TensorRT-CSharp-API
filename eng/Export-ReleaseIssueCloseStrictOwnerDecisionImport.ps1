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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function New-DecisionLane {
  param(
    [string]$Id,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$DefaultState,
    [string]$OwnerAction
  )

  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
  [pscustomobject]@{
    id = $Id
    state = $state
    ready = $false
    ownerAction = $OwnerAction
    notExecutedByAutomation = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    boundary = "Strict owner decision import lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$publicPublishRealResult = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-owner-input-contract-validation.json"
$cleanConsumerProof = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-contract-validation.json"
$releaseIssueCloseOwnerDecision = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input-validation.json"
$strictCloseReady = Read-JsonOrNull "artifacts\final-release\strict-close-ready-convergence-dashboard-validation.json"
$releaseIssueCloseRecord = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$decisionLanes = @(
  New-DecisionLane -Id "public-publish-real-result" -Record $publicPublishRealResult -StateProperty "validationState" -DefaultState "missing-public-publish-real-result-owner-input-contract-validation" -OwnerAction "Fill real public package source, URL, SHA256, publish timestamp, transcript, owner account, reviewer, and rollback review."
  New-DecisionLane -Id "post-publish-clean-consumer-proof" -Record $cleanConsumerProof -StateProperty "validationState" -DefaultState "missing-post-publish-clean-consumer-proof-record-contract-validation" -OwnerAction "Fill repository-external clean consumer restore/build/smoke evidence and no-substitute scan result."
  New-DecisionLane -Id "release-issue-close-owner-decision" -Record $releaseIssueCloseOwnerDecision -StateProperty "validationState" -DefaultState "missing-release-issue-close-owner-decision-input-validation" -OwnerAction "Fill final owner close decision after real publish, clean consumer proof, and rollback review."
  New-DecisionLane -Id "strict-close-ready-dashboard" -Record $strictCloseReady -StateProperty "validationState" -DefaultState "missing-strict-close-ready-convergence-dashboard-validation" -OwnerAction "Re-run strict close readiness dashboard after real owner inputs are present."
  New-DecisionLane -Id "release-issue-close-record-strict-validation" -Record $releaseIssueCloseRecord -StateProperty "validationState" -DefaultState "missing-release-issue-close-record-validation" -OwnerAction "Run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady after all real proof inputs pass."
)

$blockedLanes = @($decisionLanes | Where-Object { -not [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "release-issue-close-strict-owner-decision-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = "blocked-release-issue-close-strict-owner-decision-required"
  laneCount = $decisionLanes.Count
  blockedLaneCount = $blockedLanes.Count
  readyLaneCount = 0
  decisionLanes = @($decisionLanes)
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
    "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
    "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
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
  boundary = "This import maps strict owner close decision inputs only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-issue-close-strict-owner-decision-import.json"
$markdownPath = Join-Path $OutputRoot "release-issue-close-strict-owner-decision-import.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Release Issue Close Strict Owner Decision Import",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| importState | ``$($record.importState)`` |",
  "| laneCount | ``$($record.laneCount)`` |",
  "| blockedLaneCount | ``$($record.blockedLaneCount)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Decision Lanes",
  "",
  "| Lane | State | Owner Action |",
  "| --- | --- | --- |"
)

foreach ($lane in $decisionLanes) {
  $markdown += "| $($lane.id) | ``$($lane.state)`` | $($lane.ownerAction) |"
}

$markdown += @("", "## Boundary", "", $record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close strict owner decision import written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ImportState=$($record.importState) Lanes=$($record.laneCount) Blocked=$($record.blockedLaneCount)"
