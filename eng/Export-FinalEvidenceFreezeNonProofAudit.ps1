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
  param([string]$Id, [AllowNull()][object]$Record, [string]$StateProperty, [string]$DefaultState)

  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
  $performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
  $canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
  $canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
  $isRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  $isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
  $isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)

  [pscustomobject]@{
    id = $Id
    state = $state
    passed = $false
    blocked = $true
    performsPublish = $performsPublish
    canPublishPublicly = $canPublishPublicly
    canCloseReleaseIssue = $canCloseReleaseIssue
    isRuntimeExecutionProof = $isRuntimeProof
    isReleaseCloseProof = $isReleaseCloseProof
    isPostPublishProof = $isPostPublishProof
    boundaryOk = (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue -and -not $isRuntimeProof -and -not $isReleaseCloseProof -and -not $isPostPublishProof)
    boundary = "Final freeze non-proof audit lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$finalEvidenceFreezeValidation = Read-JsonOrNull "artifacts\final-release\final-evidence-freeze-validation.json"
$publicPublishFinalPackValidation = Read-JsonOrNull "artifacts\final-release\public-publish-final-owner-execution-pack-validation.json"
$publicPublishCommandCrossCheckValidation = Read-JsonOrNull "artifacts\final-release\public-publish-command-cross-check-validation.json"
$releaseIssueCloseOwnerDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input-validation.json"
$publicPublishResultOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-owner-input-validation.json"
$postPublishCleanConsumerConvergenceValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-result-convergence-validation.json"
$strictCloseReadyValidation = Read-JsonOrNull "artifacts\final-release\strict-close-ready-convergence-dashboard-validation.json"
$releaseEvidenceClassificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$auditLanes = @(
  New-AuditLane -Id "final-evidence-freeze" -Record $finalEvidenceFreezeValidation -StateProperty "validationState" -DefaultState "missing-final-evidence-freeze-validation"
  New-AuditLane -Id "public-publish-final-owner-execution-pack" -Record $publicPublishFinalPackValidation -StateProperty "validationState" -DefaultState "missing-public-publish-final-owner-execution-pack-validation"
  New-AuditLane -Id "public-publish-command-cross-check" -Record $publicPublishCommandCrossCheckValidation -StateProperty "validationState" -DefaultState "missing-public-publish-command-cross-check-validation"
  New-AuditLane -Id "release-issue-close-owner-decision-input" -Record $releaseIssueCloseOwnerDecisionValidation -StateProperty "validationState" -DefaultState "missing-release-issue-close-owner-decision-input-validation"
  New-AuditLane -Id "public-publish-result-owner-input" -Record $publicPublishResultOwnerInputValidation -StateProperty "validationState" -DefaultState "missing-public-publish-result-owner-input-validation"
  New-AuditLane -Id "post-publish-clean-consumer-result-convergence" -Record $postPublishCleanConsumerConvergenceValidation -StateProperty "validationState" -DefaultState "missing-post-publish-clean-consumer-result-convergence-validation"
  New-AuditLane -Id "strict-close-ready-convergence-dashboard" -Record $strictCloseReadyValidation -StateProperty "validationState" -DefaultState "missing-strict-close-ready-convergence-dashboard-validation"
  New-AuditLane -Id "release-evidence-classification-audit" -Record $releaseEvidenceClassificationAudit -StateProperty "auditState" -DefaultState "missing-release-evidence-classification-audit"
)

$boundaryFailures = @($auditLanes | Where-Object { -not [bool]$_.boundaryOk })

$record = [pscustomobject]@{
  recordKind = "final-evidence-freeze-non-proof-audit"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = "blocked-final-evidence-freeze-non-proof-audit"
  auditLaneCount = $auditLanes.Count
  blockedAuditLaneCount = @($auditLanes | Where-Object { [bool]$_.blocked }).Count
  boundaryFailureCount = $boundaryFailures.Count
  auditLanes = $auditLanes
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Final evidence freeze non-proof audit verifies blocked/non-proof boundaries only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "final-evidence-freeze-non-proof-audit.json"
$markdownPath = Join-Path $artifactRoot "final-evidence-freeze-non-proof-audit.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.auditLanes | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | $(ConvertTo-MarkdownCell $_.state) | ``$($_.boundaryOk)`` |"
}

$markdown = @"
# Final Evidence Freeze Non-Proof Audit

| 项目 | 当前值 |
|---|---|
| auditState | ``$($record.auditState)`` |
| auditLaneCount | ``$($record.auditLaneCount)`` |
| blockedAuditLaneCount | ``$($record.blockedAuditLaneCount)`` |
| boundaryFailureCount | ``$($record.boundaryFailureCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Audit Lanes

| Lane | State | Boundary OK |
|---|---|---:|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final evidence freeze non-proof audit written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "AuditState=$($record.auditState) Lanes=$($record.auditLaneCount) BoundaryFailures=$($record.boundaryFailureCount)"
