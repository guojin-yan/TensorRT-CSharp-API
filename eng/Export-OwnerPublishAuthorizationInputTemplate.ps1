[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
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

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$template = [pscustomobject]@{
  recordKind = "owner-publish-authorization-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "blocked-owner-publish-authorization-required"
  ownerName = "<owner-fill-owner-name>"
  ownerDecisionTimestampUtc = "<owner-fill-decision-timestamp-utc>"
  authorizationDecision = "owner-authorization-required"
  authorizedRoutes = @()
  publishTargetChannels = @(
    "nuget-small-bridge-core",
    "github-packages-full-runtime"
  )
  ownerAuthorizationId = "<owner-fill-owner-authorization-id>"
  ownerAuthorizationScope = "manual-owner-run-only"
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  managedPackageVersion = "<owner-fill-managed-package-version>"
  managedNupkgPath = "<owner-fill-managed-nupkg-path>"
  managedNupkgSha256 = "<owner-fill-managed-nupkg-sha256>"
  runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey"
  runtimePackageVersion = "<owner-fill-runtime-package-version>"
  runtimePackageKey = $RuntimePackageKey
  runtimeNupkgPath = "<owner-fill-runtime-nupkg-path>"
  runtimeNupkgSha256 = "<owner-fill-runtime-nupkg-sha256>"
  nugetSourceUrl = "https://api.nuget.org/v3/index.json"
  githubPackagesSourceUrl = "https://nuget.pkg.github.com/<owner>/index.json"
  releaseNotesPath = "<owner-fill-release-notes-path>"
  releaseNotesSha256 = "<owner-fill-release-notes-sha256>"
  rollbackPlanPath = "<owner-fill-rollback-plan-path>"
  rollbackPlanSha256 = "<owner-fill-rollback-plan-sha256>"
  publishCommandPlanPath = "artifacts/final-release/owner-authorized-publish-command-plan.json"
  publishCommandPlanSha256 = "<owner-fill-publish-command-plan-sha256>"
  managedPublishCommandSha256 = "<owner-fill-managed-publish-command-template-sha256>"
  runtimePublishCommandSha256 = "<owner-fill-runtime-publish-command-template-sha256>"
  sourceRunnerQueueStatus = "<owner-fill-completed-not-queued>"
  sourceRunnerInfrastructureStatus = "<owner-fill-available-not-missing-self-hosted-runner>"
  sourceRunnerOwnerAction = "owner-infra-action-required-until-runner-completed-and-available"
  confirmsNoTokenPersisted = "<owner-fill-true>"
  confirmsNoDryRunArtifactSubstitution = "<owner-fill-true>"
  confirmsPackageHashesReviewed = "<owner-fill-true>"
  confirmsPublishCommandReviewed = "<owner-fill-true>"
  confirmsPublishCommandHashesReviewed = "<owner-fill-true>"
  confirmsNoForcePublish = "<owner-fill-true>"
  confirmsNoQueuedRunOrMissingRunnerSubstitution = "<owner-fill-true>"
  confirmsPublicPackageDownloadProofStillRequired = "<owner-fill-true>"
  performsPublish = $false
  usesPublishToken = $false
  requiresOwnerAuthorization = $true
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimPackageConsumerRuntimeProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  blockedReasons = @(
    "owner-authorization-required",
    "publish-command-owner-review-required",
    "public-package-download-proof-still-required",
    "post-publish-proof-required"
  )
  forbiddenSubstitutes = @(
    "publish token persisted in JSON or Markdown",
    "GitHub Actions dry-run artifact path",
    "package-managed-dry-run artifact",
    "github-actions-runs artifact path",
    "template placeholder",
    "local feed",
    "direct .nupkg as public proof",
    "queued GitHub Actions run",
    "missing self-hosted runner",
    "--force publish",
    "--skip-duplicate as authorization substitute"
  )
  publishCommandTemplates = @(
    "dotnet nuget push <managed-nupkg> --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json",
    "dotnet nuget push <runtime-nupkg> --api-key <GITHUB_TOKEN> --source https://nuget.pkg.github.com/<owner>/index.json"
  )
  proofBoundary = "Owner authorization input only. This template does not execute dotnet nuget push, does not use a publish token, does not publish NuGet or GitHub Packages, and cannot close the release issue. Even an approved owner run still requires post-publish public package download proof and clean external consumer smoke proof. queued GitHub Actions run, missing self-hosted runner, --force publish, and --skip-duplicate-only outcomes cannot substitute owner authorization or post-publish proof."
}

$jsonPath = Join-Path $artifactRoot "owner-publish-authorization-input.template.json"
$markdownPath = Join-Path $artifactRoot "owner-publish-authorization-input.template.md"

$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$blockedRows = $template.blockedReasons | ForEach-Object { "- ``$_``" }
$forbiddenRows = $template.forbiddenSubstitutes | ForEach-Object { "- ``$_``" }
$commandRows = $template.publishCommandTemplates | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner Publish Authorization Input Template

生成时间：$($template.generatedAtUtc)

## 用途

该模板用于 Owner 审阅发布命令、包路径、SHA256、release notes 与 rollback plan。它只生成 Owner 授权输入，不执行发布、不保存 token、不关闭 release issue。

| 字段 | 当前值 |
|---|---|
| validationState | ``$($template.validationState)`` |
| ownerAuthorizationId | ``$($template.ownerAuthorizationId)`` |
| authorizationDecision | ``$($template.authorizationDecision)`` |
| ownerAuthorizationScope | ``$($template.ownerAuthorizationScope)`` |
| managedPackageId | ``$($template.managedPackageId)`` |
| runtimePackageId | ``$($template.runtimePackageId)`` |
| runtimePackageKey | ``$($template.runtimePackageKey)`` |
| publishCommandPlanPath | ``$($template.publishCommandPlanPath)`` |
| sourceRunnerQueueStatus | ``$($template.sourceRunnerQueueStatus)`` |
| sourceRunnerInfrastructureStatus | ``$($template.sourceRunnerInfrastructureStatus)`` |
| performsPublish | ``$($template.performsPublish)`` |
| usesPublishToken | ``$($template.usesPublishToken)`` |
| requiresOwnerAuthorization | ``$($template.requiresOwnerAuthorization)`` |
| canPublishPublicly | ``$($template.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($template.canCloseReleaseIssue)`` |
| isPostPublishProof | ``$($template.isPostPublishProof)`` |

## Publish Command Templates

$($commandRows -join "`r`n")

## Blocked Reasons

$($blockedRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerPublishAuthorizationInput.ps1 -Strict
```

## Boundary

$($template.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner publish authorization input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($template.validationState) PerformsPublish=False UsesPublishToken=False CanPublishPublicly=False"
