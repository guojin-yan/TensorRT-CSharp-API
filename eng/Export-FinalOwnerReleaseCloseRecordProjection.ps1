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

function New-ProjectionLane {
  param([string]$Id, [string]$SourceArtifact, [string]$TargetField, [string]$CurrentState, [string]$OwnerAction)
  [pscustomobject]@{
    id = $Id
    sourceArtifact = $SourceArtifact
    targetField = $TargetField
    currentState = $CurrentState
    ownerAction = $OwnerAction
    ready = $false
    boundary = "Final owner release close record projection only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$realValidator = Read-JsonOrNull "artifacts\final-release\final-release-close-record-real-validator-validation.json"
$releaseIssueCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$strictRecordCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate-validation.json"
$finalOwnerCheckpointValidation = Read-JsonOrNull "artifacts\final-release\final-owner-close-readiness-checkpoint-validation.json"

$realValidatorState = [string](Get-PropertyOrDefault -Object $realValidator -Name "validationState" -DefaultValue "missing-final-release-close-record-real-validator-validation")
$releaseIssueCloseRecordValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
$strictRecordCandidateValidationState = [string](Get-PropertyOrDefault -Object $strictRecordCandidateValidation -Name "validationState" -DefaultValue "missing-release-close-strict-record-candidate-validation")
$finalOwnerCheckpointValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerCheckpointValidation -Name "validationState" -DefaultValue "missing-final-owner-close-readiness-checkpoint-validation")

$lanes = @(
  New-ProjectionLane -Id "real-validator-contracts" -SourceArtifact "artifacts/final-release/final-release-close-record-real-validator-validation.json" -TargetField "releaseCloseRecord.realFieldContracts" -CurrentState $realValidatorState -OwnerAction "Complete all blocked real close record field contracts."
  New-ProjectionLane -Id "strict-record-candidate" -SourceArtifact "artifacts/final-release/release-close-strict-record-candidate-validation.json" -TargetField "releaseCloseRecord.strictCandidateHashSet" -CurrentState $strictRecordCandidateValidationState -OwnerAction "Refresh strict close record candidate after real inputs are filled."
  New-ProjectionLane -Id "release-issue-close-record-validation" -SourceArtifact "artifacts/final-release/release-issue-close-record-validation.json" -TargetField "releaseCloseRecord.strictValidation" -CurrentState $releaseIssueCloseRecordValidationState -OwnerAction "Run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady only after real proof records pass."
  New-ProjectionLane -Id "final-owner-readiness" -SourceArtifact "artifacts/final-release/final-owner-close-readiness-checkpoint-validation.json" -TargetField "releaseCloseRecord.finalReadiness" -CurrentState $finalOwnerCheckpointValidationState -OwnerAction "Unblock all final owner close readiness checks."
)

$blockedLanes = @($lanes | Where-Object { -not [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "final-owner-release-close-record-projection"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  projectionState = "blocked-final-owner-release-close-record-projection-owner-input-required"
  laneCount = $lanes.Count
  blockedLaneCount = $blockedLanes.Count
  readyLaneCount = 0
  projectionLanes = @($lanes)
  sourceArtifacts = @($lanes | ForEach-Object { $_.sourceArtifact })
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
  boundary = "Final owner release close record projection is a blocked projection only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-release-close-record-projection.json"
$markdownPath = Join-Path $OutputRoot "final-owner-release-close-record-projection.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $lanes | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.currentState)`` | ``$($_.ready)`` | ``$($_.targetField)`` | $($_.ownerAction.Replace("|", "\|")) |"
}

$markdown = @"
# Final Owner Release Close Record Projection

| Field | Value |
| --- | --- |
| projectionState | ``$($record.projectionState)`` |
| laneCount | ``$($record.laneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Projection Lanes

| ID | Current State | Ready | Target Field | Owner Action |
| --- | --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final owner release close record projection written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ProjectionState=$($record.projectionState) Lanes=$($record.laneCount) Blocked=$($record.blockedLaneCount)"
