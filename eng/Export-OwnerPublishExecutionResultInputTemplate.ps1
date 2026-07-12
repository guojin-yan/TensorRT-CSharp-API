[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$runtimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22"
$runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$runtimePackageKey"

$record = [pscustomobject]@{
  recordKind = "owner-publish-execution-result-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "blocked-owner-publish-execution-result-required"
  ownerExecutionResultReady = $false
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  runtimePackageId = $runtimePackageId
  runtimePackageKey = $runtimePackageKey
  managedPackageVersion = "<owner-fill-managed-package-version>"
  runtimePackageVersion = "<owner-fill-runtime-package-version>"
  ownerName = "<owner-fill-owner-name>"
  ownerReviewedAtUtc = "<owner-fill-owner-reviewed-at-utc>"
  ownerApprovalReference = "<owner-fill-owner-approval-reference>"
  publishCommandReviewed = "<owner-fill-true>"
  publishExecutedByOwner = "<owner-fill-true>"
  publishStartedAtUtc = "<owner-fill-publish-started-at-utc>"
  publishCompletedAtUtc = "<owner-fill-publish-completed-at-utc>"
  publishExitCode = "<owner-fill-0>"
  publishCommand = "<owner-fill-redacted-command-without-token>"
  pushTranscriptPath = "<owner-fill-push-transcript-path>"
  pushTranscriptSha256 = "<owner-fill-push-transcript-sha256>"
  pushStdoutPath = "<owner-fill-push-stdout-path>"
  pushStdoutSha256 = "<owner-fill-push-stdout-sha256>"
  pushStderrPath = "<owner-fill-push-stderr-path>"
  pushStderrSha256 = "<owner-fill-push-stderr-sha256>"
  publicManagedPackageUrl = "<owner-fill-https-public-managed-package-url>"
  publicRuntimePackageUrl = "<owner-fill-https-public-runtime-package-url>"
  nugetPackageMetadataUrl = "<owner-fill-https-nuget-package-metadata-url>"
  githubPackagesMetadataUrl = "<owner-fill-https-github-packages-metadata-url>"
  downloadedManagedNupkgPath = "<owner-fill-downloaded-managed-nupkg-path>"
  downloadedManagedNupkgSha256 = "<owner-fill-downloaded-managed-nupkg-sha256>"
  downloadedRuntimeNupkgPath = "<owner-fill-downloaded-runtime-nupkg-path>"
  downloadedRuntimeNupkgSha256 = "<owner-fill-downloaded-runtime-nupkg-sha256>"
  releaseNotesPath = "<owner-fill-release-notes-path>"
  releaseNotesSha256 = "<owner-fill-release-notes-sha256>"
  rollbackPlanPath = "<owner-fill-rollback-plan-path>"
  rollbackPlanSha256 = "<owner-fill-rollback-plan-sha256>"
  rollbackDecision = "<owner-fill-rollback-decision>"
  confirmsNoTokenPersisted = "<owner-fill-true>"
  confirmsNoTokenInTranscripts = "<owner-fill-true>"
  confirmsNoDryRunArtifactSubstitution = "<owner-fill-true>"
  confirmsNoLocalFeedSubstitution = "<owner-fill-true>"
  confirmsNoDirectNupkgSubstitution = "<owner-fill-true>"
  confirmsNoGitHubActionsArtifactSubstitution = "<owner-fill-true>"
  confirmsPublicPackageDownloadProofStillRequired = "<owner-fill-true>"
  confirmsPostPublishProofStillRequired = "<owner-fill-true>"
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  forbiddenSubstitutes = @(
    "local feed",
    "direct .nupkg",
    "artifacts/package-managed-dry-run",
    "GitHub Actions package artifact",
    "ProjectReference",
    "build-only",
    "dependency-probe-only",
    "dashboard-only",
    "template-only"
  )
  safetyBoundary = "Owner publish execution result input template only. It records owner-supplied publish result evidence after an owner-run publish; automation does not publish, does not use tokens, does not claim runtime proof, does not claim post-publish proof, and cannot close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-publish-execution-result-input.template.json"
$markdownPath = Join-Path $OutputRoot "owner-publish-execution-result-input.template.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$forbiddenRows = $record.forbiddenSubstitutes | ForEach-Object { "| $_ | rejected as real Owner publish execution result proof substitute |" }
$markdown = @"
# Owner Publish Execution Result Input Template

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($record.validationState)`` |
| ownerExecutionResultReady | ``$($record.ownerExecutionResultReady)`` |
| managedPackageId | ``$($record.managedPackageId)`` |
| runtimePackageId | ``$($record.runtimePackageId)`` |
| performsPublish | ``$($record.performsPublish)`` |
| usesPublishToken | ``$($record.usesPublishToken)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Forbidden Substitutes

| Substitute | Boundary |
|---|---|
$($forbiddenRows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner publish execution result input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($record.validationState) PerformsPublish=False UsesPublishToken=False CanCloseReleaseIssue=False"
