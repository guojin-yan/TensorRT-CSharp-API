[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$PreReleaseReadinessMatrixPath = "artifacts\final-release\pre-release-package-proof-readiness-matrix.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Get-BoolPropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [bool]$DefaultValue)
  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) { return [bool]$value }
  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) { return $parsed }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$preReleaseReadinessMatrix = Read-JsonOrNull -Path $PreReleaseReadinessMatrixPath
$preReleaseReadinessMatrixState = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "matrixState" -DefaultValue "missing-pre-release-package-proof-readiness-matrix")
$preReleaseReadinessMatrixReady = $preReleaseReadinessMatrixState.Equals("pre-release-package-proof-ready", [StringComparison]::OrdinalIgnoreCase)
$preReleaseReadyLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "readyLaneCount" -DefaultValue 0)
$preReleaseBlockedLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "blockedLaneCount" -DefaultValue 0)
$preReleaseCurrentHead = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "currentHead" -DefaultValue "")
$preReleaseSourceQualityRunId = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "sourceQualityRunId" -DefaultValue "")
$preReleaseLanes = @((Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "lanes" -DefaultValue @()))
$blockedReadinessLanes = @($preReleaseLanes | Where-Object { -not (Get-BoolPropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) } | ForEach-Object {
    [pscustomobject]@{
      id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
      state = [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "")
      requiredEvidence = [string](Get-PropertyOrDefault -Object $_ -Name "requiredEvidence" -DefaultValue "")
      validatorPath = [string](Get-PropertyOrDefault -Object $_ -Name "validatorPath" -DefaultValue "")
      blockedReason = [string](Get-PropertyOrDefault -Object $_ -Name "blockedReason" -DefaultValue "")
    }
  })
$readinessRequiredEvidence = @($preReleaseLanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "requiredEvidence" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
$readinessValidatorPaths = @($preReleaseLanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "validatorPath" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)

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
  preReleaseReadinessMatrixPath = $PreReleaseReadinessMatrixPath
  preReleaseReadinessMatrixState = $preReleaseReadinessMatrixState
  preReleaseReadinessMatrixReady = $preReleaseReadinessMatrixReady
  preReleaseReadinessCurrentHead = $preReleaseCurrentHead
  preReleaseReadinessSourceQualityRunId = $preReleaseSourceQualityRunId
  preReleaseReadyLaneCount = $preReleaseReadyLaneCount
  preReleaseBlockedLaneCount = $preReleaseBlockedLaneCount
  preReleaseBlockedLanes = @($blockedReadinessLanes)
  preReleaseRequiredEvidence = @($readinessRequiredEvidence)
  preReleaseValidatorPaths = @($readinessValidatorPaths)
  confirmsPreReleaseReadinessMatrixReviewed = "<owner-fill-true>"
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
    "pre-release-readiness-matrix-not-ready",
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
| preReleaseReadinessMatrixState | ``$($template.preReleaseReadinessMatrixState)`` |
| preReleaseReadinessMatrixReady | ``$($template.preReleaseReadinessMatrixReady)`` |
| preReleaseBlockedLaneCount | ``$($template.preReleaseBlockedLaneCount)`` |
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

## Pre-Release Readiness Blocked Lanes

| Lane | State | Validator | Required Evidence |
|---|---|---|---|
$(@($template.preReleaseBlockedLanes | ForEach-Object { "| ``$(ConvertTo-MarkdownCell $_.id)`` | ``$(ConvertTo-MarkdownCell $_.state)`` | ``$(ConvertTo-MarkdownCell $_.validatorPath)`` | $(ConvertTo-MarkdownCell $_.requiredEvidence) |" }) -join "`r`n")

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
