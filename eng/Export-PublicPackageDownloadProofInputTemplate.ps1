[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$GitHubActionsRunEvidenceImportPath = "artifacts\final-release\github-actions-run-evidence-import.json",
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

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Resolve-RepoPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $Path
  }

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolvedPath = Resolve-RepoPath -Path $Path
  if ([string]::IsNullOrWhiteSpace($resolvedPath) -or -not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

$githubActionsEvidence = Read-JsonOrNull -Path $GitHubActionsRunEvidenceImportPath
$dryRunPackages = if ($null -eq $githubActionsEvidence) {
  @()
}
else {
  @(Get-PropertyOrDefault -Object $githubActionsEvidence -Name "nupkgPackages" -DefaultValue @())
}

$dryRunManagedPackage = @($dryRunPackages | Where-Object {
    $fileName = [string](Get-PropertyOrDefault -Object $_ -Name "fileName" -DefaultValue "")
    $fileName.StartsWith("JYPPX.TensorRT.CSharp.API.", [StringComparison]::OrdinalIgnoreCase) -and
    -not $fileName.Contains(".runtime.", [StringComparison]::OrdinalIgnoreCase)
  } | Select-Object -First 1)
if ($dryRunManagedPackage.Count -eq 0 -and $dryRunPackages.Count -gt 0) {
  $dryRunManagedPackage = @($dryRunPackages[0])
}

$sourceRunId = if ($null -eq $githubActionsEvidence) { "<no-github-actions-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $githubActionsEvidence -Name "runId" -DefaultValue "<missing-run-id>") }
$sourceRunUrl = if ($null -eq $githubActionsEvidence) { "<no-github-actions-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $githubActionsEvidence -Name "runUrl" -DefaultValue "<missing-run-url>") }
$sourceHeadSha = if ($null -eq $githubActionsEvidence) { "<no-github-actions-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $githubActionsEvidence -Name "headSha" -DefaultValue "<missing-head-sha>") }
$packageDryRunCanClaimPack = if ($null -eq $githubActionsEvidence) { $false } else { [bool](Get-PropertyOrDefault -Object $githubActionsEvidence -Name "canClaimGitHubActionsPackageDryRunPackForRun" -DefaultValue $false) }
$packageDryRunArtifactPath = if ($dryRunManagedPackage.Count -eq 0) { "<no-package-managed-dry-run-artifact>" } else { [string](Get-PropertyOrDefault -Object $dryRunManagedPackage[0] -Name "fullPath" -DefaultValue "<missing-dry-run-package-path>") }
$packageDryRunManagedNupkgSha256 = if ($dryRunManagedPackage.Count -eq 0) { "<no-package-managed-dry-run-sha256>" } else { [string](Get-PropertyOrDefault -Object $dryRunManagedPackage[0] -Name "sha256" -DefaultValue "<missing-dry-run-package-sha256>") }

$template = [pscustomobject]@{
  recordKind = "public-package-download-proof-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "blocked-public-package-download-proof-required"
  proofLineId = "public-package-download"
  sourceGitHubActionsRunEvidenceImportPath = $GitHubActionsRunEvidenceImportPath
  sourceGitHubActionsRunId = $sourceRunId
  sourceGitHubActionsRunUrl = $sourceRunUrl
  sourceHeadSha = $sourceHeadSha
  packageDryRunArtifactPath = $packageDryRunArtifactPath
  packageDryRunManagedNupkgSha256 = $packageDryRunManagedNupkgSha256
  packageDryRunCanClaimPack = $packageDryRunCanClaimPack
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  managedPackageVersion = "<owner-fill-managed-package-version>"
  publicPackageSourceUrl = "<owner-fill-public-package-source-url>"
  publicPackageSourceKind = "<owner-fill-nuget.org-or-github-packages-or-private-feed>"
  downloadedManagedNupkgPath = "<owner-fill-downloaded-managed-nupkg-path>"
  downloadedManagedNupkgSha256 = "<owner-fill-downloaded-managed-nupkg-sha256>"
  runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey"
  runtimePackageVersion = "<owner-fill-runtime-package-version>"
  runtimePackageKey = $RuntimePackageKey
  downloadedRuntimeNupkgPath = "<owner-fill-downloaded-runtime-nupkg-path>"
  downloadedRuntimeNupkgSha256 = "<owner-fill-downloaded-runtime-nupkg-sha256>"
  downloadCommand = "dotnet package search/download from public feed; owner must replace with exact command"
  downloadedAtUtc = "<owner-fill-downloaded-at-utc>"
  ownerName = "<owner-fill-owner-name>"
  ownerAuthorizationState = "owner-authorization-required"
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimRuntimeProof = $false
  canClaimPackageConsumerRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isDryRunOnly = $true
  isPublishedPackageProof = $false
  blockedReasons = @(
    "owner-authorization-required",
    "public-package-download-proof-required",
    "downloaded-managed-nupkg-sha256-required",
    "clean-consumer-runtime-proof-missing",
    "post-publish-proof-missing"
  )
  forbiddenSubstitutes = @(
    "local path",
    "artifacts folder",
    "direct .nupkg reference",
    "GitHub Actions package-managed-dry-run artifact",
    "github-actions-runs artifact path",
    "template placeholder",
    "local feed",
    "ProjectReference"
  )
  proofBoundary = "Owner input template only. GitHub Actions dry-run context is included for traceability, but dry-run artifacts and hashes are not public package download proof. This template does not publish packages, does not use a publish token, does not close release issues, and is not package-consumer runtime proof or post-publish proof."
}

$jsonPath = Join-Path $artifactRoot "public-package-download-proof-input.template.json"
$markdownPath = Join-Path $artifactRoot "public-package-download-proof-input.template.md"

$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$blockedReasonRows = $template.blockedReasons | ForEach-Object { "- ``$_``" }
$forbiddenRows = $template.forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# Public Package Download Proof Input Template

生成时间：$($template.generatedAtUtc)

## 用途

该模板用于 Owner 回填真实 public package 下载证明。它只记录下载来源、下载命令、包路径和 SHA256，不执行发布，不使用 publish token，不关闭 release issue，也不提升 package-consumer runtime proof。

| 字段 | 当前值 |
|---|---|
| validationState | ``$($template.validationState)`` |
| managedPackageId | ``$($template.managedPackageId)`` |
| runtimePackageId | ``$($template.runtimePackageId)`` |
| runtimePackageKey | ``$($template.runtimePackageKey)`` |
| publicPackageSourceUrl | ``$($template.publicPackageSourceUrl)`` |
| publicPackageSourceKind | ``$($template.publicPackageSourceKind)`` |
| packageDryRunArtifactPath | ``$($template.packageDryRunArtifactPath)`` |
| packageDryRunManagedNupkgSha256 | ``$($template.packageDryRunManagedNupkgSha256)`` |
| packageDryRunCanClaimPack | ``$($template.packageDryRunCanClaimPack)`` |
| performsPublish | ``$($template.performsPublish)`` |
| usesPublishToken | ``$($template.usesPublishToken)`` |
| canPublishPublicly | ``$($template.canPublishPublicly)`` |
| isPackageConsumerRuntimeProof | ``$($template.isPackageConsumerRuntimeProof)`` |
| isPostPublishProof | ``$($template.isPostPublishProof)`` |

## Blocked Reasons

$($blockedReasonRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPackageDownloadProofInput.ps1 -Strict
```

## Boundary

$($template.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public package download proof input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($template.validationState) PerformsPublish=False UsesPublishToken=False CanPublishPublicly=False"
