[CmdletBinding()]
param(
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-DecisionGate {
  param([string]$Id, [string]$State, [string]$RequiredState, [string]$OwnerAction, [string]$SourceArtifact)
  $ready = [string]::Equals($State, $RequiredState, [StringComparison]::OrdinalIgnoreCase)
  [pscustomobject]@{
    gateId = $Id
    state = $State
    requiredState = $RequiredState
    ready = $ready
    gateState = if ($ready) { "ready" } else { "blocked-release-issue-close-final-owner-decision-required" }
    ownerAction = $OwnerAction
    sourceArtifact = $SourceArtifact
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
  }
}

$publicPackageValidation = Read-JsonOrNull "artifacts\final-release\public-package-proof-owner-input-validation.json"
$postPublishConfirmationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-proof-owner-confirmation-validation.json"
$publicProofBridgeValidation = Read-JsonOrNull "artifacts\final-release\release-close-public-proof-bridge-validation.json"
$closeCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"
$finalDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$strictCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$gates = @(
  New-DecisionGate -Id "public-package-proof-owner-input" -State ([string](Get-PropertyOrDefault -Object $publicPackageValidation -Name "validationState" -DefaultValue "missing-public-package-proof-owner-input-validation")) -RequiredState "public-package-proof-owner-input-ready" -OwnerAction "Owner must fill real public package URLs, registries, nupkg SHA256 values, publish timestamp, and review confirmations." -SourceArtifact "artifacts/final-release/public-package-proof-owner-input-validation.json"
  New-DecisionGate -Id "post-publish-proof-owner-confirmation" -State ([string](Get-PropertyOrDefault -Object $postPublishConfirmationValidation -Name "validationState" -DefaultValue "missing-post-publish-proof-owner-confirmation-validation")) -RequiredState "post-publish-proof-owner-confirmation-ready" -OwnerAction "Owner must satisfy public package proof, post-publish clean consumer proof, owner result import, and close owner bridge gates." -SourceArtifact "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json"
  New-DecisionGate -Id "release-close-public-proof-bridge" -State ([string](Get-PropertyOrDefault -Object $publicProofBridgeValidation -Name "validationState" -DefaultValue "missing-release-close-public-proof-bridge-validation")) -RequiredState "release-close-public-proof-ready" -OwnerAction "All final public proof bridge gates must be ready before close approval can be considered." -SourceArtifact "artifacts/final-release/release-close-public-proof-bridge-validation.json"
  New-DecisionGate -Id "release-issue-close-record-candidate" -State ([string](Get-PropertyOrDefault -Object $closeCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-candidate-validation")) -RequiredState "release-issue-close-record-candidate-ready" -OwnerAction "Close record candidate must contain real post-publish proof, rollback owner input, and candidate validation." -SourceArtifact "artifacts/final-release/release-issue-close-record-candidate-validation.json"
  New-DecisionGate -Id "release-issue-final-close-decision" -State ([string](Get-PropertyOrDefault -Object $finalDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")) -RequiredState "owner-final-close-decision-ready-for-strict-close-validator" -OwnerAction "Owner must fill final close decision, rollback review, public package source, clean consumer and log/hash review fields." -SourceArtifact "artifacts/final-release/release-issue-final-close-decision-validation.json"
  New-DecisionGate -Id "release-issue-close-record" -State ([string](Get-PropertyOrDefault -Object $strictCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")) -RequiredState "ready-for-owner-release-issue-close" -OwnerAction "Strict close validator must pass with Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady." -SourceArtifact "artifacts/final-release/release-issue-close-record-validation.json"
  New-DecisionGate -Id "release-evidence-classification-audit" -State ([string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")) -RequiredState "classification-audit-passed-non-proof-boundaries-intact" -OwnerAction "Keep non-proof boundaries intact while final owner decision remains blocked." -SourceArtifact "artifacts/final-release/release-evidence-classification-audit.json"
)

$blocked = @($gates | Where-Object { -not [bool]$_.ready })
$ready = @($gates | Where-Object { [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "release-issue-close-final-owner-decision-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = if ($blocked.Count -eq 0) { "release-issue-close-final-owner-decision-ready" } else { "blocked-release-issue-close-final-owner-decision-required" }
  finalOwnerDecisionGateCount = $gates.Count
  blockedFinalOwnerDecisionGateCount = $blocked.Count
  readyFinalOwnerDecisionGateCount = $ready.Count
  finalOwnerDecisionGates = $gates
  strictCloseCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  sourceArtifacts = @(
    "artifacts/final-release/public-package-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json",
    "artifacts/final-release/release-close-public-proof-bridge-validation.json",
    "artifacts/final-release/release-issue-close-record-candidate-validation.json",
    "artifacts/final-release/release-issue-final-close-decision-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/release-evidence-classification-audit.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Final owner decision audit is blocked release-close gate aggregation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-issue-close-final-owner-decision-audit.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-final-owner-decision-audit.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.finalOwnerDecisionGates | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.gateId) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.ownerAction) |"
}

$markdown = @"
# Release Issue Close Final Owner Decision Audit

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| auditState | ``$($record.auditState)`` |
| finalOwnerDecisionGateCount | ``$($record.finalOwnerDecisionGateCount)`` |
| blockedFinalOwnerDecisionGateCount | ``$($record.blockedFinalOwnerDecisionGateCount)`` |
| readyFinalOwnerDecisionGateCount | ``$($record.readyFinalOwnerDecisionGateCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Gates

| Gate | State | Required State | Ready | Owner Action |
|---|---|---|---:|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close final owner decision audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$($record.auditState) Gates=$($record.finalOwnerDecisionGateCount) Blocked=$($record.blockedFinalOwnerDecisionGateCount) Ready=$($record.readyFinalOwnerDecisionGateCount)"
