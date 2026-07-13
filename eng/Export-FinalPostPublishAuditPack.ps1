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

function New-AuditLane {
  param([string]$Id, [string]$State, [string]$RequiredState, [string]$EvidenceRequired, [string]$SourceArtifact)
  $ready = [string]::Equals($State, $RequiredState, [StringComparison]::OrdinalIgnoreCase)
  [pscustomobject]@{
    laneId = $Id
    state = $State
    requiredState = $RequiredState
    laneState = if ($ready) { "ready" } else { "blocked-final-post-publish-audit-required" }
    ready = $ready
    evidenceRequired = $EvidenceRequired
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
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$postPublishConfirmationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-proof-owner-confirmation-validation.json"
$packageConsumerRecordValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-record-validation.json"
$releaseClosePublicProofBridgeValidation = Read-JsonOrNull "artifacts\final-release\release-close-public-proof-bridge-validation.json"
$finalDecisionAuditValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-final-owner-decision-audit-validation.json"
$strictCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$lanes = @(
  New-AuditLane -Id "public-package-proof" -State ([string](Get-PropertyOrDefault -Object $publicPackageValidation -Name "validationState" -DefaultValue "missing-public-package-proof-owner-input-validation")) -RequiredState "public-package-proof-owner-input-ready" -EvidenceRequired "Public package source, registry URL, package URLs, nupkg SHA256 values, and Owner review fields." -SourceArtifact "artifacts/final-release/public-package-proof-owner-input-validation.json"
  New-AuditLane -Id "post-publish-verification-proof" -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")) -RequiredState "post-publish-verification-ready" -EvidenceRequired "Real public-channel clean consumer restore/build/smoke logs, SHA256 values, and host metadata." -SourceArtifact "artifacts/final-release/post-publish-verification-validation.json"
  New-AuditLane -Id "post-publish-proof-owner-confirmation" -State ([string](Get-PropertyOrDefault -Object $postPublishConfirmationValidation -Name "validationState" -DefaultValue "missing-post-publish-proof-owner-confirmation-validation")) -RequiredState "post-publish-proof-owner-confirmation-ready" -EvidenceRequired "All Owner confirmation gates for public package proof and post-publish proof." -SourceArtifact "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json"
  New-AuditLane -Id "package-consumer-runtime-proof" -State ([string](Get-PropertyOrDefault -Object $packageConsumerRecordValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-record-validation")) -RequiredState "package-consumer-runtime-proof-ready" -EvidenceRequired "Clean external package consumer runtime smoke proof." -SourceArtifact "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
  New-AuditLane -Id "release-close-public-proof-bridge" -State ([string](Get-PropertyOrDefault -Object $releaseClosePublicProofBridgeValidation -Name "validationState" -DefaultValue "missing-release-close-public-proof-bridge-validation")) -RequiredState "release-close-public-proof-ready" -EvidenceRequired "All public proof bridge gates ready." -SourceArtifact "artifacts/final-release/release-close-public-proof-bridge-validation.json"
  New-AuditLane -Id "final-owner-decision-audit" -State ([string](Get-PropertyOrDefault -Object $finalDecisionAuditValidation -Name "validationState" -DefaultValue "missing-release-issue-close-final-owner-decision-audit-validation")) -RequiredState "release-issue-close-final-owner-decision-ready" -EvidenceRequired "Final owner decision audit gates ready." -SourceArtifact "artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json"
  New-AuditLane -Id "strict-release-close-validator" -State ([string](Get-PropertyOrDefault -Object $strictCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")) -RequiredState "ready-for-owner-release-issue-close" -EvidenceRequired "Strict close validator must pass with -FailOnNotCloseReady." -SourceArtifact "artifacts/final-release/release-issue-close-record-validation.json"
)

$blocked = @($lanes | Where-Object { -not [bool]$_.ready })
$ready = @($lanes | Where-Object { [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "final-post-publish-audit-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = if ($blocked.Count -eq 0) { "final-post-publish-audit-ready" } else { "blocked-final-post-publish-audit-required" }
  auditLaneCount = $lanes.Count
  blockedAuditLaneCount = $blocked.Count
  readyAuditLaneCount = $ready.Count
  auditLanes = $lanes
  strictCloseCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  sourceArtifacts = @(
    "artifacts/final-release/public-package-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
    "artifacts/final-release/release-close-public-proof-bridge-validation.json",
    "artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Final post-publish audit pack is blocked audit aggregation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "final-post-publish-audit-pack.json"
$markdownPath = Join-Path $artifactRoot "final-post-publish-audit-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.auditLanes | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.laneId) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.evidenceRequired) |"
}

$markdown = @"
# Final Post-Publish Audit Pack

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| auditState | ``$($record.auditState)`` |
| auditLaneCount | ``$($record.auditLaneCount)`` |
| blockedAuditLaneCount | ``$($record.blockedAuditLaneCount)`` |
| readyAuditLaneCount | ``$($record.readyAuditLaneCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Audit Lanes

| Lane | State | Required State | Ready | Evidence Required |
|---|---|---|---:|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final post-publish audit pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$($record.auditState) Lanes=$($record.auditLaneCount) Blocked=$($record.blockedAuditLaneCount) Ready=$($record.readyAuditLaneCount)"
