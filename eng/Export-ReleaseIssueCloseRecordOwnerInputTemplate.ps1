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

function Get-RelativeFileSha256OrPlaceholder {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return "<owner-fill-$($RelativePath.Replace('\','-').Replace('/','-'))-sha256>"
  }

  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
$releaseClosePreflightPath = "artifacts/final-release/release-close-preflight.json"
$staleClaimsAuditPath = "artifacts/final-release/stale-release-claims-audit.json"
$postPublishProofValidationPath = "artifacts/final-release/post-publish-verification-validation.json"

$template = [pscustomobject]@{
  recordKind = "release-issue-close-record-owner-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ownerInputState = "template-owner-input-required"
  proofLineId = "release-issue-close-record"
  releaseEvidenceBundlePath = $releaseEvidenceBundlePath
  releaseEvidenceBundleSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseEvidenceBundlePath
  releaseClosePreflightPath = $releaseClosePreflightPath
  releaseClosePreflightSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $releaseClosePreflightPath
  staleClaimsAuditPath = $staleClaimsAuditPath
  staleClaimsAuditSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $staleClaimsAuditPath
  postPublishProofValidationPath = $postPublishProofValidationPath
  postPublishProofValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $postPublishProofValidationPath
  postPublishProofValidationState = "<owner-fill-real-post-publish-proof-validation-state>"
  rollbackPlan = "<owner-fill-rollback-plan>"
  rollbackOwner = "<owner-fill-rollback-owner>"
  rollbackTrigger = "<owner-fill-rollback-trigger>"
  ownerFinalCloseDecision = "<owner-fill-final-close-decision>"
  ownerDecisionTimestamp = "<owner-fill-owner-decision-timestamp>"
  releaseIssueId = "<owner-fill-release-issue-id>"
  releaseIssueUrl = "<owner-fill-release-issue-url>"
  strictCloseValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Owner input template only. It does not publish packages, close the release issue, or replace real post-publish proof."
}

$jsonPath = Join-Path $artifactRoot "release-issue-close-record-owner-input.template.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-record-owner-input.template.md"

$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Release Issue Close Record Owner Input Template

生成时间：$($template.generatedAtUtc)

## 用途

该模板用于 Owner 回填 release issue close record candidate 所需真实输入：release evidence bundle、release close preflight、stale claims audit、post-publish proof validation、rollback plan 和 owner final close decision。

它不是 close proof，不关闭 release issue，也不能替代 post-publish proof validation 或最终 owner 决策。

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOwnerInput.ps1 -Strict
```

## Safety Boundary

$($template.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue close record owner input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
