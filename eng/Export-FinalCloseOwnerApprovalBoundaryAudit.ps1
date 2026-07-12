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

function New-ApprovalLane {
  param([string]$Id, [string]$SourceArtifact, [string]$CurrentState, [string]$OwnerAction, [string]$Validator)
  [pscustomobject]@{
    id = $Id
    sourceArtifact = $SourceArtifact
    currentState = $CurrentState
    ownerAction = $OwnerAction
    validator = $Validator
    ready = $false
    boundary = "Final close owner approval boundary audit only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$publicPublishDraft = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-record-draft-validation.json"
$postPublishCleanConsumerDraft = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-draft-validation.json"
$finalCloseDecision = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$releaseIssueCloseRecord = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$finalOwnerCheckpoint = Read-JsonOrNull "artifacts\final-release\final-owner-close-readiness-checkpoint-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$lanes = @(
  New-ApprovalLane -Id "public-publish-owner-result" -SourceArtifact "artifacts/final-release/public-publish-real-result-record-draft-validation.json" -CurrentState ([string](Get-PropertyOrDefault -Object $publicPublishDraft -Name "validationState" -DefaultValue "missing-public-publish-real-result-record-draft-validation")) -OwnerAction "Owner must provide real public publish result, package URL/hash, timestamp, transcript, and reviewer." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict"
  New-ApprovalLane -Id "post-publish-clean-consumer-proof" -SourceArtifact "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json" -CurrentState ([string](Get-PropertyOrDefault -Object $postPublishCleanConsumerDraft -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-draft-validation")) -OwnerAction "Owner must provide repository-external clean consumer restore/build/smoke proof." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict"
  New-ApprovalLane -Id "rollback-final-decision" -SourceArtifact "artifacts/final-release/release-issue-final-close-decision-validation.json" -CurrentState ([string](Get-PropertyOrDefault -Object $finalCloseDecision -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")) -OwnerAction "Owner must review rollback plan and final close decision after real proof passes." -Validator "eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict"
  New-ApprovalLane -Id "strict-release-issue-close-record" -SourceArtifact "artifacts/final-release/release-issue-close-record-validation.json" -CurrentState ([string](Get-PropertyOrDefault -Object $releaseIssueCloseRecord -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")) -OwnerAction "Owner must pass strict release issue close record validation." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
  New-ApprovalLane -Id "final-owner-readiness" -SourceArtifact "artifacts/final-release/final-owner-close-readiness-checkpoint-validation.json" -CurrentState ([string](Get-PropertyOrDefault -Object $finalOwnerCheckpoint -Name "validationState" -DefaultValue "missing-final-owner-close-readiness-checkpoint-validation")) -OwnerAction "Owner must unblock final close readiness checkpoint." -Validator "eng\Test-FinalOwnerCloseReadinessCheckpoint.ps1 -Strict"
  New-ApprovalLane -Id "classification-audit-clean" -SourceArtifact "artifacts/final-release/release-evidence-classification-audit.json" -CurrentState ([string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")) -OwnerAction "Keep evidence classification audit clean while preserving non-proof boundaries." -Validator "eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
)

$blockedLanes = @($lanes | Where-Object { -not [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "final-close-owner-approval-boundary-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = "blocked-final-close-owner-approval-boundary-owner-action-required"
  approvalLaneCount = $lanes.Count
  blockedApprovalLaneCount = $blockedLanes.Count
  readyApprovalLaneCount = 0
  approvalLanes = @($lanes)
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
  boundary = "Final close owner approval boundary audit is blocked owner-action review only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-close-owner-approval-boundary-audit.json"
$markdownPath = Join-Path $OutputRoot "final-close-owner-approval-boundary-audit.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $lanes | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.currentState)`` | ``$($_.ready)`` | $($_.ownerAction.Replace("|", "\|")) | ``$($_.validator)`` |"
}

$markdown = @"
# Final Close Owner Approval Boundary Audit

| Field | Value |
| --- | --- |
| auditState | ``$($record.auditState)`` |
| approvalLaneCount | ``$($record.approvalLaneCount)`` |
| blockedApprovalLaneCount | ``$($record.blockedApprovalLaneCount)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Approval Lanes

| ID | Current State | Ready | Owner Action | Validator |
| --- | --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final close owner approval boundary audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$($record.auditState) Lanes=$($record.approvalLaneCount) Blocked=$($record.blockedApprovalLaneCount)"
