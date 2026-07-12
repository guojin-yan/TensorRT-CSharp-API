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

function New-PublicProofGate {
  param([string]$Id, [string]$State, [string]$RequiredState, [string]$Action, [string]$SourceArtifact)
  $ready = [string]::Equals($State, $RequiredState, [StringComparison]::OrdinalIgnoreCase)
  [pscustomobject]@{
    gateId = $Id
    state = $State
    requiredState = $RequiredState
    gateState = if ($ready) { "ready" } else { "blocked-release-close-public-proof-required" }
    ready = $ready
    requiredAction = $Action
    sourceArtifact = $SourceArtifact
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$postPublishConfirmationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-proof-owner-confirmation-validation.json"
$releaseCloseOwnerBridgeValidation = Read-JsonOrNull "artifacts\final-release\release-close-owner-input-bridge-validation.json"
$releaseIssueCloseCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-candidate-validation.json"
$releaseFinalDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$releaseCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$gates = @(
  New-PublicProofGate -Id "post-publish-proof-owner-confirmation" -State ([string](Get-PropertyOrDefault -Object $postPublishConfirmationValidation -Name "validationState" -DefaultValue "missing-post-publish-proof-owner-confirmation-validation")) -RequiredState "post-publish-proof-owner-confirmation-ready" -Action "Owner must provide public package proof, clean consumer smoke, logs, hashes and host metadata." -SourceArtifact "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json"
  New-PublicProofGate -Id "release-close-owner-input-bridge" -State ([string](Get-PropertyOrDefault -Object $releaseCloseOwnerBridgeValidation -Name "validationState" -DefaultValue "missing-release-close-owner-input-bridge-validation")) -RequiredState "release-close-owner-input-ready" -Action "All release close owner gates must be ready." -SourceArtifact "artifacts/final-release/release-close-owner-input-bridge-validation.json"
  New-PublicProofGate -Id "release-issue-close-record-candidate" -State ([string](Get-PropertyOrDefault -Object $releaseIssueCloseCandidateValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-candidate-validation")) -RequiredState "release-issue-close-record-candidate-ready" -Action "Close record candidate must include real post-publish proof and owner inputs." -SourceArtifact "artifacts/final-release/release-issue-close-record-candidate-validation.json"
  New-PublicProofGate -Id "release-issue-final-close-decision" -State ([string](Get-PropertyOrDefault -Object $releaseFinalDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")) -RequiredState "owner-final-close-decision-ready" -Action "Owner final close decision, rollback owner and rollback trigger must be confirmed." -SourceArtifact "artifacts/final-release/release-issue-final-close-decision-validation.json"
  New-PublicProofGate -Id "release-issue-close-record" -State ([string](Get-PropertyOrDefault -Object $releaseCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")) -RequiredState "release-issue-close-record-ready" -Action "Strict release issue close validator must pass with -FailOnNotCloseReady." -SourceArtifact "artifacts/final-release/release-issue-close-record-validation.json"
  New-PublicProofGate -Id "release-evidence-classification-audit" -State ([string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")) -RequiredState "classification-audit-passed-non-proof-boundaries-intact" -Action "Keep all non-proof boundaries intact while public proof remains blocked." -SourceArtifact "artifacts/final-release/release-evidence-classification-audit.json"
)

$blocked = @($gates | Where-Object { -not [bool]$_.ready })
$ready = @($gates | Where-Object { [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "release-close-public-proof-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = if ($blocked.Count -eq 0) { "release-close-public-proof-ready" } else { "blocked-release-close-public-proof-required" }
  publicProofGateCount = $gates.Count
  blockedPublicProofGateCount = $blocked.Count
  readyPublicProofGateCount = $ready.Count
  publicProofGates = $gates
  strictCloseCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  sourceArtifacts = @(
    "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json",
    "artifacts/final-release/release-close-owner-input-bridge-validation.json",
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
  safetyBoundary = "Release close public proof bridge is blocked public proof gate aggregation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-close-public-proof-bridge.json"
$markdownPath = Join-Path $artifactRoot "release-close-public-proof-bridge.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.publicProofGates | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.gateId) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.requiredAction) |"
}

$markdown = @"
# Release Close Public Proof Bridge

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| bridgeState | ``$($record.bridgeState)`` |
| publicProofGateCount | ``$($record.publicProofGateCount)`` |
| blockedPublicProofGateCount | ``$($record.blockedPublicProofGateCount)`` |
| readyPublicProofGateCount | ``$($record.readyPublicProofGateCount)`` |
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

Write-Host "Release close public proof bridge written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "BridgeState=$($record.bridgeState) Gates=$($record.publicProofGateCount) Blocked=$($record.blockedPublicProofGateCount) Ready=$($record.readyPublicProofGateCount)"
