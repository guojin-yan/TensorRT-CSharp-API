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

function Get-RelativeFileSha256OrPlaceholder {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return "<owner-fill-$($RelativePath.Replace('\','-').Replace('/','-'))-sha256>"
  }

  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
$postPublishValidationPath = "artifacts/final-release/post-publish-verification-validation.json"
$strictCloseReadyPath = "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json"
$classificationAuditPath = "artifacts/final-release/release-evidence-classification-audit.json"

$template = [pscustomobject]@{
  recordKind = "release-issue-close-owner-decision-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ownerDecisionInputState = "blocked-release-issue-close-owner-decision-input-required"
  ownerName = "<owner-fill-name>"
  ownerEmail = "<owner-fill-email>"
  ownerDecisionTimestampUtc = "<owner-fill-owner-decision-timestamp-utc>"
  releaseIssueUrl = "<owner-fill-release-issue-url>"
  selectedChannel = "<owner-fill-selected-public-channel>"
  approvedPublicPackageProofHash = "<owner-fill-public-package-proof-sha256>"
  approvedPostPublishProofHash = "<owner-fill-post-publish-proof-sha256>"
  approvedReleaseEvidenceBundleHash = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath
  approvedStrictCloseReadyHash = Get-RelativeFileSha256OrPlaceholder -RelativePath $strictCloseReadyPath
  approvedClassificationAuditHash = Get-RelativeFileSha256OrPlaceholder -RelativePath $classificationAuditPath
  postPublishValidationPath = $postPublishValidationPath
  postPublishValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishValidationPath
  releaseEvidenceBundlePath = $releaseEvidenceBundlePath
  releaseEvidenceBundleSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath
  rollbackPlan = "<owner-fill-rollback-plan>"
  rollbackOwner = "<owner-fill-rollback-owner>"
  rollbackTrigger = "<owner-fill-rollback-trigger>"
  knownLimitationsAcknowledgement = "<owner-fill-known-limitations-acknowledgement>"
  finalCloseDecision = "<owner-fill-final-close-decision>"
  finalCloseDecisionAllowedValues = @(
    "approved-close-release-issue",
    "blocked-keep-release-issue-open"
  )
  strictCloseValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  requiredOwnerFields = @(
    "ownerName",
    "ownerEmail",
    "ownerDecisionTimestampUtc",
    "releaseIssueUrl",
    "selectedChannel",
    "approvedPublicPackageProofHash",
    "approvedPostPublishProofHash",
    "approvedReleaseEvidenceBundleHash",
    "rollbackPlan",
    "rollbackOwner",
    "rollbackTrigger",
    "knownLimitationsAcknowledgement",
    "finalCloseDecision"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Release issue close owner decision input template only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $artifactRoot "release-issue-close-owner-decision-input.template.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-owner-decision-input.template.md"
$template | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$fieldRows = $template.requiredOwnerFields | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_) | owner-filled after real public publish, post-publish proof, and strict close validation |"
}

$markdown = @"
# Release Issue Close Owner Decision Input Template

生成时间：$($template.generatedAtUtc)

该模板只定义 release owner 最终关闭 issue 前必须回填的决策字段。它不会执行发布，不会关闭 issue，也不会把本地 artifact 或 dashboard 提升为 proof。

| 项目 | 当前值 |
|---|---|
| ownerDecisionInputState | ``$($template.ownerDecisionInputState)`` |
| notExecutedByAutomation | ``$($template.notExecutedByAutomation)`` |
| ownerExecutionOnly | ``$($template.ownerExecutionOnly)`` |
| performsPublish | ``$($template.performsPublish)`` |
| canPublishPublicly | ``$($template.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($template.canCloseReleaseIssue)`` |

## Required Owner Fields

| Field | Required Source |
|---|---|
$($fieldRows -join "`r`n")

## Boundary

$($template.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close owner decision input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "OwnerDecisionInputState=$($template.ownerDecisionInputState) PerformsPublish=False CanCloseReleaseIssue=False"
