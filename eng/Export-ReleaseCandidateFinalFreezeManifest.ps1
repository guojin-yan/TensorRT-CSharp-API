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

function Get-FileHashOrEmpty {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "" }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-FreezeArtifact {
  param([string]$RelativePath, [string]$Kind, [string]$Boundary)
  $path = Join-Path $RepositoryRoot $RelativePath
  $exists = Test-Path -LiteralPath $path -PathType Leaf
  [pscustomobject]@{
    relativePath = $RelativePath.Replace("\", "/")
    kind = $Kind
    exists = $exists
    length = if ($exists) { (Get-Item -LiteralPath $path).Length } else { 0 }
    sha256 = if ($exists) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
    boundary = $Boundary
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
  }
}

$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$finalOwnerDecisionAudit = Read-JsonOrNull "artifacts\final-release\release-issue-close-final-owner-decision-audit-validation.json"
$finalPostPublishAudit = Read-JsonOrNull "artifacts\final-release\final-post-publish-audit-pack-validation.json"
$releaseCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$candidateArtifacts = @(
  New-FreezeArtifact -RelativePath "README.md" -Kind "readme" -Boundary "Readme is documentation only and is not release proof."
  New-FreezeArtifact -RelativePath "README.zh-CN.md" -Kind "readme" -Boundary "Readme is documentation only and is not release proof."
  New-FreezeArtifact -RelativePath "docs\index.md" -Kind "docs-index" -Boundary "Docs index is navigation only and is not release proof."
  New-FreezeArtifact -RelativePath "docs\toc.yml" -Kind "docs-toc" -Boundary "Docs table of contents is navigation only and is not release proof."
  New-FreezeArtifact -RelativePath "artifacts\final-release\release-evidence-bundle.json" -Kind "release-evidence" -Boundary "Release evidence bundle aggregates evidence and cannot substitute missing proof."
  New-FreezeArtifact -RelativePath "artifacts\final-release\release-evidence-classification-audit.json" -Kind "classification-audit" -Boundary "Classification audit checks non-proof boundaries and cannot approve release close."
  New-FreezeArtifact -RelativePath "artifacts\final-release\release-issue-close-final-owner-decision-audit.json" -Kind "final-owner-decision-audit" -Boundary "Final owner decision audit is gate aggregation only and is not release proof."
  New-FreezeArtifact -RelativePath "artifacts\final-release\release-issue-close-final-owner-decision-audit-validation.json" -Kind "final-owner-decision-audit-validation" -Boundary "Final owner decision audit validation is non-proof validation only and is not release proof."
  New-FreezeArtifact -RelativePath "artifacts\final-release\final-post-publish-audit-pack.json" -Kind "post-publish-audit" -Boundary "Final post-publish audit pack is audit aggregation only and is not release proof."
  New-FreezeArtifact -RelativePath "artifacts\final-release\final-post-publish-audit-pack-validation.json" -Kind "post-publish-audit-validation" -Boundary "Final post-publish audit validation is non-proof validation only and is not release proof."
  New-FreezeArtifact -RelativePath "artifacts\test-results\targeted\release-issue-close-final-owner-decision-and-post-publish-audit.trx" -Kind "targeted-test-result" -Boundary "Targeted TRX proves local quality gate execution only and is not package publish proof."
)

$missingArtifacts = @($candidateArtifacts | Where-Object { -not [bool]$_.exists })
$classificationFindingCount = [int](Get-PropertyOrDefault -Object $classificationAudit -Name "findingCount" -DefaultValue -1)
$finalOwnerBlocked = [int](Get-PropertyOrDefault -Object $finalOwnerDecisionAudit -Name "blockedFinalOwnerDecisionGateCount" -DefaultValue -1)
$postPublishBlocked = [int](Get-PropertyOrDefault -Object $finalPostPublishAudit -Name "blockedAuditLaneCount" -DefaultValue -1)
$closeValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")

$record = [pscustomobject]@{
  recordKind = "release-candidate-final-freeze-manifest"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  freezeState = "blocked-owner-public-publish-required"
  artifactCount = $candidateArtifacts.Count
  existingArtifactCount = @($candidateArtifacts | Where-Object { [bool]$_.exists }).Count
  missingArtifactCount = $missingArtifacts.Count
  classificationAuditFindingCount = $classificationFindingCount
  finalOwnerDecisionBlockedGateCount = $finalOwnerBlocked
  finalPostPublishBlockedLaneCount = $postPublishBlocked
  strictCloseValidationState = $closeValidationState
  freezeArtifacts = $candidateArtifacts
  sourceArtifacts = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-evidence-classification-audit.json",
    "artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json",
    "artifacts/final-release/final-post-publish-audit-pack-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Release candidate final freeze manifest records local artifact hashes for owner handoff only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-candidate-final-freeze-manifest.json"
$markdownPath = Join-Path $artifactRoot "release-candidate-final-freeze-manifest.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.freezeArtifacts | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.relativePath) | $(ConvertTo-MarkdownCell $_.kind) | ``$($_.exists)`` | ``$($_.sha256)`` | $(ConvertTo-MarkdownCell $_.boundary) |"
}

$markdown = @"
# Release Candidate Final Freeze Manifest

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| freezeState | ``$($record.freezeState)`` |
| artifactCount | ``$($record.artifactCount)`` |
| existingArtifactCount | ``$($record.existingArtifactCount)`` |
| missingArtifactCount | ``$($record.missingArtifactCount)`` |
| classificationAuditFindingCount | ``$($record.classificationAuditFindingCount)`` |
| finalOwnerDecisionBlockedGateCount | ``$($record.finalOwnerDecisionBlockedGateCount)`` |
| finalPostPublishBlockedLaneCount | ``$($record.finalPostPublishBlockedLaneCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Freeze Artifacts

| Artifact | Kind | Exists | SHA256 | Boundary |
|---|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate final freeze manifest written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "FreezeState=$($record.freezeState) Artifacts=$($record.artifactCount) Missing=$($record.missingArtifactCount)"
