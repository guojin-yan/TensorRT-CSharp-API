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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Resolve-RepositoryPath -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Get-RelativeFileSha256OrPlaceholder {
  param([string]$RelativePath)

  $path = Resolve-RepositoryPath -Path $RelativePath
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
$postPublishProofResultValidationPath = "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json"
$strictCloseReadyPath = "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json"
$classificationAuditPath = "artifacts/final-release/release-evidence-classification-audit.json"
$finalPublicReleaseClosureBridgePath = "artifacts/final-release/final-public-release-closure-bridge.json"
$finalPublicReleaseClosureBridgeValidationPath = "artifacts/final-release/final-public-release-closure-bridge-validation.json"

$bridge = Read-JsonOrNull -RelativePath $finalPublicReleaseClosureBridgePath
$bridgeValidation = Read-JsonOrNull -RelativePath $finalPublicReleaseClosureBridgeValidationPath
$postPublishProofResultValidation = Read-JsonOrNull -RelativePath $postPublishProofResultValidationPath
$summary = Get-PropertyOrDefault -Object $bridge -Name "closureProofSourceSummary" -DefaultValue ([pscustomobject]@{})

$template = [pscustomobject]@{
  recordKind = "release-issue-close-owner-decision-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ownerDecisionInputState = "blocked-release-issue-close-owner-decision-input-required"
  ownerName = "<owner-fill-name>"
  ownerEmail = "<owner-fill-email>"
  ownerDecisionTimestampUtc = "<owner-fill-owner-decision-timestamp-utc>"
  releaseIssueUrl = "<owner-fill-release-issue-url>"
  releaseIssueNumber = "<owner-fill-release-issue-number>"
  selectedChannel = "<owner-fill-selected-public-channel>"
  approvedPublicPackageProofHash = "<owner-fill-public-package-proof-sha256>"
  approvedPostPublishProofHash = "<owner-fill-post-publish-proof-sha256>"
  approvedReleaseEvidenceBundleHash = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath
  approvedStrictCloseReadyHash = Get-RelativeFileSha256OrPlaceholder -RelativePath $strictCloseReadyPath
  approvedClassificationAuditHash = Get-RelativeFileSha256OrPlaceholder -RelativePath $classificationAuditPath
  finalPublicReleaseClosureBridgePath = $finalPublicReleaseClosureBridgePath
  finalPublicReleaseClosureBridgeSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $finalPublicReleaseClosureBridgePath
  finalPublicReleaseClosureBridgeValidationPath = $finalPublicReleaseClosureBridgeValidationPath
  finalPublicReleaseClosureBridgeValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $finalPublicReleaseClosureBridgeValidationPath
  finalPublicReleaseClosureBridgeValidationState = [string](Get-PropertyOrDefault -Object $bridgeValidation -Name "validationState" -DefaultValue "missing-final-public-release-closure-bridge-validation")
  closureLaneCount = [int](Get-PropertyOrDefault -Object $bridge -Name "laneCount" -DefaultValue 0)
  closureBlockedLaneCount = [int](Get-PropertyOrDefault -Object $bridge -Name "blockedLaneCount" -DefaultValue 0)
  closureFailedConsistencyBlockerCount = [int](Get-PropertyOrDefault -Object $bridge -Name "failedConsistencyBlockerCount" -DefaultValue 0)
  closureFailedConsistencyActionRequiredCount = [int](Get-PropertyOrDefault -Object $bridge -Name "failedConsistencyActionRequiredCount" -DefaultValue 0)
  postPublishValidationPath = $postPublishValidationPath
  postPublishValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishValidationPath
  postPublishProofResultValidationPath = $postPublishProofResultValidationPath
  postPublishProofResultValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishProofResultValidationPath
  postPublishProofCandidateReady = [bool](Get-PropertyOrDefault -Object $postPublishProofResultValidation -Name "proofCandidateReady" -DefaultValue $false)
  postPublishProofSourceLinkageReady = [bool](Get-PropertyOrDefault -Object $postPublishProofResultValidation -Name "sourceProofLinkageReady" -DefaultValue $false)
  githubActionsRunId = [string](Get-PropertyOrDefault -Object $summary -Name "githubActionsRunId" -DefaultValue "")
  githubActionsRunUrl = [string](Get-PropertyOrDefault -Object $summary -Name "githubActionsRunUrl" -DefaultValue "")
  githubActionsHeadSha = [string](Get-PropertyOrDefault -Object $summary -Name "githubActionsHeadSha" -DefaultValue "")
  ownerPublicPublishResultReady = [bool](Get-PropertyOrDefault -Object $summary -Name "ownerPublicPublishResultReady" -DefaultValue $false)
  publicPackageDownloadProofReady = [bool](Get-PropertyOrDefault -Object $summary -Name "publicDownloadProofReady" -DefaultValue $false)
  publicPackageUrl = [string](Get-PropertyOrDefault -Object $summary -Name "ownerPublicPackageUrl" -DefaultValue "")
  publicPackageVersion = [string](Get-PropertyOrDefault -Object $summary -Name "ownerPublicPackageVersion" -DefaultValue "")
  publicPackageSha256 = [string](Get-PropertyOrDefault -Object $summary -Name "ownerPublicPackageSha256" -DefaultValue "")
  runtimePackageUrl = [string](Get-PropertyOrDefault -Object $summary -Name "publicDownloadRuntimePackageUrl" -DefaultValue "")
  runtimePackageDownloadUrl = [string](Get-PropertyOrDefault -Object $summary -Name "publicDownloadRuntimePackageDownloadUrl" -DefaultValue "")
  githubReleaseAssetUrl = [string](Get-PropertyOrDefault -Object $summary -Name "ownerGitHubReleaseAssetUrl" -DefaultValue "")
  githubReleaseAssetSha256 = [string](Get-PropertyOrDefault -Object $summary -Name "ownerGitHubReleaseAssetSha256" -DefaultValue "")
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
    "releaseIssueNumber",
    "selectedChannel",
    "approvedPublicPackageProofHash",
    "approvedPostPublishProofHash",
    "approvedReleaseEvidenceBundleHash",
    "finalPublicReleaseClosureBridgeSha256",
    "publicPackageUrl",
    "publicPackageVersion",
    "publicPackageSha256",
    "githubActionsRunId",
    "githubActionsRunUrl",
    "githubActionsHeadSha",
    "rollbackPlan",
    "rollbackOwner",
    "rollbackTrigger",
    "knownLimitationsAcknowledgement",
    "finalCloseDecision"
  )
  sourceArtifacts = @(
    $finalPublicReleaseClosureBridgePath,
    $finalPublicReleaseClosureBridgeValidationPath,
    $postPublishProofResultValidationPath,
    $postPublishValidationPath,
    $releaseEvidenceBundlePath,
    $strictCloseReadyPath,
    $classificationAuditPath
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
  safetyBoundary = "Release issue close owner decision input template only; it records final bridge and source proof snapshots for owner review, but is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $artifactRoot "release-issue-close-owner-decision-input.template.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-owner-decision-input.template.md"
$template | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$fieldRows = $template.requiredOwnerFields | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_) | owner-filled after real public publish, post-publish proof, final closure bridge, and strict close validation |"
}

$markdown = @"
# Release Issue Close Owner Decision Input Template

生成时间：$($template.generatedAtUtc)

该模板只定义 release owner 最终关闭 issue 前必须回填的决策字段。它不会执行发布，不会关闭 issue，也不会把本地 artifact 或 dashboard 提升为 proof。

| 项目 | 当前值 |
|---|---|
| ownerDecisionInputState | ``$($template.ownerDecisionInputState)`` |
| finalPublicReleaseClosureBridgeValidationState | ``$($template.finalPublicReleaseClosureBridgeValidationState)`` |
| closureLaneCount | ``$($template.closureLaneCount)`` |
| closureBlockedLaneCount | ``$($template.closureBlockedLaneCount)`` |
| closureFailedConsistencyBlockerCount | ``$($template.closureFailedConsistencyBlockerCount)`` |
| closureFailedConsistencyActionRequiredCount | ``$($template.closureFailedConsistencyActionRequiredCount)`` |
| postPublishProofCandidateReady | ``$($template.postPublishProofCandidateReady)`` |
| postPublishProofSourceLinkageReady | ``$($template.postPublishProofSourceLinkageReady)`` |
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
