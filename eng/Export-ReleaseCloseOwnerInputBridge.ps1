[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

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

function New-BridgeGate {
  param([string]$Id, [string]$State, [string]$RequiredState, [string]$RequiredAction, [string]$SourceArtifact)

  $ready = $State -eq $RequiredState
  [pscustomobject]@{
    gateId = $Id
    state = $State
    requiredState = $RequiredState
    gateState = if ($ready) { "ready" } else { "blocked-release-close-owner-input-required" }
    ready = $ready
    requiredAction = $RequiredAction
    sourceArtifact = $SourceArtifact
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$realImportValidation = Read-JsonOrNull "artifacts\final-release\real-external-proof-record-import-validator-validation.json"
$ownerResultImportValidation = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseOwnerProofValidation = Read-JsonOrNull "artifacts\final-release\release-owner-proof-input-record-validation.json"
$releaseFinalDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-final-close-decision-validation.json"
$releaseCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"

$gates = @(
  New-BridgeGate -Id "owner-external-proof-result-import" -State ([string](Get-PropertyOrDefault -Object $ownerResultImportValidation -Name "validationState" -DefaultValue "missing-owner-external-proof-execution-result-import-validation")) -RequiredState "owner-external-proof-execution-result-ready" -RequiredAction "Backfill real owner execution result files, hashes, metadata, validator output, reviewer and timestamps." -SourceArtifact "artifacts/final-release/owner-external-proof-execution-result-import-validation.json"
  New-BridgeGate -Id "real-external-proof-record-import" -State ([string](Get-PropertyOrDefault -Object $realImportValidation -Name "validationState" -DefaultValue "missing-real-external-proof-record-import-validator-validation")) -RequiredState "real-external-proof-record-import-ready" -RequiredAction "Convert owner result import into strict real external proof record candidates without substitutes." -SourceArtifact "artifacts/final-release/real-external-proof-record-import-validator-validation.json"
  New-BridgeGate -Id "post-publish-verification" -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")) -RequiredState "post-publish-verification-ready" -RequiredAction "Provide real public-channel post-publish proof with package hashes and clean consumer evidence." -SourceArtifact "artifacts/final-release/post-publish-verification-validation.json"
  New-BridgeGate -Id "release-owner-proof-input" -State ([string](Get-PropertyOrDefault -Object $releaseOwnerProofValidation -Name "validationState" -DefaultValue "missing-release-owner-proof-input-record-validation")) -RequiredState "release-owner-proof-input-ready" -RequiredAction "Provide owner-approved release proof inputs and required non-substitute confirmations." -SourceArtifact "artifacts/final-release/release-owner-proof-input-record-validation.json"
  New-BridgeGate -Id "rollback-and-final-close-decision" -State ([string](Get-PropertyOrDefault -Object $releaseFinalDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-final-close-decision-validation")) -RequiredState "owner-final-close-decision-ready" -RequiredAction "Provide rollback plan, rollback owner, rollback trigger and final close approval." -SourceArtifact "artifacts/final-release/release-issue-final-close-decision-validation.json"
  New-BridgeGate -Id "release-issue-close-record" -State ([string](Get-PropertyOrDefault -Object $releaseCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")) -RequiredState "release-issue-close-record-ready" -RequiredAction "Run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady with real proof inputs." -SourceArtifact "artifacts/final-release/release-issue-close-record-validation.json"
  New-BridgeGate -Id "release-evidence-classification-audit" -State ([string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")) -RequiredState "classification-audit-passed-non-proof-boundaries-intact" -RequiredAction "Keep all non-proof boundaries intact while real proof import remains blocked." -SourceArtifact "artifacts/final-release/release-evidence-classification-audit.json"
)

$blockedGates = @($gates | Where-Object { -not [bool]$_.ready })
$readyGates = @($gates | Where-Object { [bool]$_.ready })
$releaseBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")

$record = [pscustomobject]@{
  recordKind = "release-close-owner-input-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = if ($blockedGates.Count -eq 0) { "release-close-owner-input-ready" } else { "blocked-release-close-owner-input-required" }
  releaseEvidenceBundleState = $releaseBundleState
  bridgeGateCount = $gates.Count
  blockedBridgeGateCount = $blockedGates.Count
  readyBridgeGateCount = $readyGates.Count
  bridgeGates = $gates
  strictCloseCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  sourceArtifacts = @(
    "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
    "artifacts/final-release/real-external-proof-record-import-validator-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/release-owner-proof-input-record-validation.json",
    "artifacts/final-release/release-issue-final-close-decision-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/release-evidence-classification-audit.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Release close owner input bridge is a blocked prerequisite map. It cannot substitute real proof, public package publication, post-publish verification, rollback approval, final owner close decision, or strict release issue close validation."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-close-owner-input-bridge.json"
$markdownPath = Join-Path $artifactRoot "release-close-owner-input-bridge.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Close Owner Input Bridge")
$lines.Add("")
$lines.Add("| 项目 | 当前值 |")
$lines.Add("|---|---|")
$lines.Add("| bridgeState | ``$(ConvertTo-MarkdownCell $record.bridgeState)`` |")
$lines.Add("| bridgeGateCount | ``$($record.bridgeGateCount)`` |")
$lines.Add("| blockedBridgeGateCount | ``$($record.blockedBridgeGateCount)`` |")
$lines.Add("| readyBridgeGateCount | ``$($record.readyBridgeGateCount)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Gates")
$lines.Add("")
$lines.Add("| Gate | State | Required State | Ready |")
$lines.Add("|---|---|---|---:|")
foreach ($gate in $gates) {
  $lines.Add("| $(ConvertTo-MarkdownCell $gate.gateId) | $(ConvertTo-MarkdownCell $gate.state) | $(ConvertTo-MarkdownCell $gate.requiredState) | ``$($gate.ready)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.safetyBoundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close owner input bridge written to $jsonPath"
Write-Host "Release close owner input bridge markdown written to $markdownPath"
Write-Host "BridgeState=$($record.bridgeState) Gates=$($record.bridgeGateCount) Blocked=$($record.blockedBridgeGateCount) Ready=$($record.readyBridgeGateCount)"
