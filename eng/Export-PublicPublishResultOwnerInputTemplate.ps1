[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$template = [pscustomobject]@{
  recordKind = "public-publish-result-owner-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  templateState = "blocked-public-publish-result-owner-input-required"
  ownerName = "<owner-fill-name>"
  ownerEmail = "<owner-fill-email>"
  publishedAtUtc = "<owner-fill-public-publish-timestamp-utc>"
  selectedChannel = "<owner-fill-nuget.org-or-github-packages>"
  nugetPackageSource = "<owner-fill-public-nuget-package-source>"
  nugetPackageUrl = "<owner-fill-nuget-package-url>"
  githubPackageUrl = "<owner-fill-github-package-url-or-empty-if-unused>"
  githubRelease = [pscustomobject]@{
    releaseUrl = "<owner-fill-github-release-url>"
    tagName = "<owner-fill-github-release-tag>"
    managedAssetPath = "<owner-fill-github-release-managed-asset-path>"
    managedAssetSha256 = "<owner-fill-github-release-managed-asset-sha256>"
    runtimeAssetPath = "<owner-fill-github-release-runtime-asset-path>"
    runtimeAssetSha256 = "<owner-fill-github-release-runtime-asset-sha256>"
  }
  packageId = "JYPPX.TensorRT.CSharp.API"
  packageVersion = "<owner-fill-managed-package-version>"
  managedNupkgPath = "<owner-fill-managed-nupkg-path>"
  managedNupkgSha256 = "<owner-fill-managed-nupkg-sha256>"
  managedPackage = [pscustomobject]@{
    packageId = "JYPPX.TensorRT.CSharp.API"
    version = "<owner-fill-managed-package-version>"
    packageUrl = "<owner-fill-managed-public-package-url>"
    publicDownloadUrl = "<owner-fill-managed-public-download-url>"
    publicDownloadSha256 = "<owner-fill-managed-public-download-sha256>"
  }
  runtimePackageKey = $RuntimePackageKey
  runtimePackagePaths = @(
    "<owner-fill-runtime-nupkg-path-or-empty-if-not-published>"
  )
  runtimePackageSha256 = @(
    "<owner-fill-runtime-nupkg-sha256-or-empty-if-not-published>"
  )
  runtimePackage = [pscustomobject]@{
    packageId = "<owner-fill-runtime-package-id>"
    version = "<owner-fill-runtime-package-version>"
    packageUrl = "<owner-fill-runtime-public-package-url>"
    publicDownloadUrl = "<owner-fill-runtime-public-download-url>"
    publicDownloadSha256 = "<owner-fill-runtime-public-download-sha256>"
  }
  publishCommandTranscriptPath = "<owner-fill-publish-command-transcript-path>"
  publishCommandTranscriptSha256 = "<owner-fill-publish-command-transcript-sha256>"
  ownerReview = [pscustomobject]@{
    reviewer = "<owner-fill-reviewer>"
    reviewedAtUtc = "<owner-fill-review-timestamp-utc>"
    approvalState = "<owner-fill-approved-or-blocked>"
  }
  rollbackReview = [pscustomobject]@{
    reviewedBy = "<owner-fill-rollback-reviewer>"
    reviewedAtUtc = "<owner-fill-rollback-review-timestamp-utc>"
    rollbackPlanSha256 = "<owner-fill-rollback-plan-sha256>"
    decision = "<owner-fill-rollback-approved-or-blocked>"
  }
  finalCloseDecision = [pscustomobject]@{
    decision = "<owner-fill-close-or-keep-open>"
    decidedAtUtc = "<owner-fill-final-close-decision-timestamp-utc>"
    ownerReviewer = "<owner-fill-final-close-owner>"
    releaseIssueUrl = "<owner-fill-release-issue-url>"
  }
  ownerReviewedPackageHash = $false
  ownerReviewedPublicUrl = $false
  rollbackPlanReviewed = $false
  requiredRealInputFields = @(
    "ownerName",
    "ownerEmail",
    "publishedAtUtc",
    "selectedChannel",
    "nugetPackageSource",
    "nugetPackageUrl or githubPackageUrl",
    "githubRelease.releaseUrl",
    "githubRelease.tagName",
    "githubRelease.managedAssetPath",
    "githubRelease.managedAssetSha256",
    "githubRelease.runtimeAssetPath",
    "githubRelease.runtimeAssetSha256",
    "packageId",
    "packageVersion",
    "managedNupkgPath",
    "managedNupkgSha256",
    "managedPackage.packageUrl",
    "managedPackage.publicDownloadUrl",
    "managedPackage.publicDownloadSha256",
    "runtimePackage.packageUrl",
    "runtimePackage.publicDownloadUrl",
    "runtimePackage.publicDownloadSha256",
    "publishCommandTranscriptPath",
    "publishCommandTranscriptSha256",
    "ownerReview.reviewer",
    "ownerReview.reviewedAtUtc",
    "rollbackReview.rollbackPlanSha256",
    "finalCloseDecision.decision",
    "finalCloseDecision.releaseIssueUrl",
    "ownerReviewedPackageHash",
    "ownerReviewedPublicUrl",
    "rollbackPlanReviewed"
  )
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Public publish result owner input template only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $artifactRoot "public-publish-result-owner-input.template.json"
$markdownPath = Join-Path $artifactRoot "public-publish-result-owner-input.template.md"
$template | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$fieldRows = $template.requiredRealInputFields | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_) | owner-filled after real public publish |"
}

$markdown = @"
# Public Publish Result Owner Input Template

生成时间：$($template.generatedAtUtc)

该模板只定义 Owner 在真实公开发布完成后需要回填的结果字段。自动化不会执行 publish，也不会把模板提升为 proof。

| 项目 | 当前值 |
|---|---|
| recordKind | ``$($template.recordKind)`` |
| templateState | ``$($template.templateState)`` |
| runtimePackageKey | ``$($template.runtimePackageKey)`` |
| notExecutedByAutomation | ``$($template.notExecutedByAutomation)`` |
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

Write-Host "Public publish result owner input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "TemplateState=$($template.templateState) NotExecutedByAutomation=$($template.notExecutedByAutomation) PerformsPublish=False CanCloseReleaseIssue=False"
