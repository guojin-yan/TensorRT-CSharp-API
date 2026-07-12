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

$template = [pscustomobject]@{
  recordKind = "release-issue-final-close-decision"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  decisionState = "template-owner-input-required"
  proofLineId = "release-issue-final-close-decision"
  finalEvidenceFreezePath = "artifacts/final-release/final-evidence-freeze.json"
  finalEvidenceFreezeSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath "artifacts/final-release/final-evidence-freeze.json"
  finalEvidenceFreezeValidationPath = "artifacts/final-release/final-evidence-freeze-validation.json"
  finalEvidenceFreezeValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath "artifacts/final-release/final-evidence-freeze-validation.json"
  releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
  releaseEvidenceBundleSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath "artifacts/final-release/release-evidence-bundle.json"
  postPublishVerificationValidationPath = "artifacts/final-release/post-publish-verification-validation.json"
  postPublishVerificationValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath "artifacts/final-release/post-publish-verification-validation.json"
  releaseIssueCloseRecordCandidateValidationPath = "artifacts/final-release/release-issue-close-record-candidate-validation.json"
  releaseIssueCloseRecordCandidateValidationSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath "artifacts/final-release/release-issue-close-record-candidate-validation.json"
  ownerName = "<owner-fill-owner-name>"
  ownerDecisionTimestampUtc = "<owner-fill-owner-decision-timestamp-utc>"
  releaseIssueId = "<owner-fill-release-issue-id>"
  releaseIssueUrl = "<owner-fill-release-issue-url>"
  ownerFinalCloseDecision = "<owner-fill-approved-to-close-after-real-proof>"
  rollbackPlanReviewed = $false
  rollbackOwner = "<owner-fill-rollback-owner>"
  rollbackTrigger = "<owner-fill-rollback-trigger>"
  confirmsRealPostPublishProof = $false
  confirmsPublicPackageSource = $false
  confirmsCleanConsumerOutsideRepository = $false
  confirmsNoProjectReference = $false
  confirmsNoLocalPackageSource = $false
  confirmsNoDirectNupkgReference = $false
  confirmsRuntimeSmokePassed = $false
  runtimeSmokeExitCode = $null
  confirmsLogsAndSha256Reviewed = $false
  strictCloseValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Owner final close decision template only. It cannot publish packages or close the release issue."
}

$jsonPath = Join-Path $artifactRoot "release-issue-final-close-decision.template.json"
$markdownPath = Join-Path $artifactRoot "release-issue-final-close-decision.template.md"
$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Release Issue Final Close Decision Template

生成时间：$($template.generatedAtUtc)

该模板用于 Owner 在真实 post-publish proof、final evidence freeze、rollback plan 和 release close strict validator 都准备好后回填最终关闭决定。模板本身不是 close proof，不关闭 issue，也不执行发布。

## 必须确认

- post-publish proof 来自真实公开包源。
- clean consumer 在仓库外。
- 无 ProjectReference、local feed 或 direct `.nupkg`。
- runtime smoke 真实执行并且 exit code 为 0。
- 所有日志与 SHA256 已经审阅。
- rollback plan、rollback owner 和 rollback trigger 已经明确。

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict
```

## Safety Boundary

$($template.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release issue final close decision template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
