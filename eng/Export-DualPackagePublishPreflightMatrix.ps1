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

function Read-JsonOrNull {
  param([string]$Path)

  $resolvedPath = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepositoryRoot $Path }
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function New-EvidenceRequirement {
  param(
    [string]$Id,
    [string]$Label,
    [bool]$Satisfied,
    [string]$Source,
    [string]$BlockingReason
  )

  [pscustomobject]@{
    id = $Id
    label = $Label
    satisfied = $Satisfied
    source = $Source
    blockingReason = $BlockingReason
  }
}

function New-PackageRoute {
  param(
    [string]$Id,
    [string]$DisplayName,
    [string]$PackageId,
    [string]$DistributionChannel,
    [string]$PackageContents,
    [string]$DependencyStrategy,
    [bool]$PackageDryRunPackSatisfied,
    [object[]]$AdditionalRequirements,
    [string[]]$BlockedReasons,
    [string]$ProofBoundary
  )

  $requirements = @(
    New-EvidenceRequirement -Id "package-dry-run-pack-success" -Label "GitHub Actions package dry-run pack success" -Satisfied $PackageDryRunPackSatisfied -Source "artifacts/final-release/github-actions-run-evidence-import.json" -BlockingReason "package-dry-run-pack-missing-or-failed"
  ) + $AdditionalRequirements

  [pscustomobject]@{
    id = $Id
    displayName = $DisplayName
    packageId = $PackageId
    runtimePackageKey = $RuntimePackageKey
    distributionChannel = $DistributionChannel
    packageContents = $PackageContents
    dependencyStrategy = $DependencyStrategy
    evidenceRequirements = @($requirements)
    blockedReasons = @($BlockedReasons)
    performsPublish = $false
    usesPublishToken = $false
    requiresOwnerAuthorization = $true
    canPublishPublicly = $false
    canPublishGitHubPackages = $false
    canClaimRuntimeProof = $false
    canClaimPackageConsumerRuntimeProof = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPackageConsumerRuntimeProof = $false
    isPostPublishProof = $false
    proofBoundary = $ProofBoundary
  }
}

$githubActionsEvidence = Read-JsonOrNull -Path "artifacts\final-release\github-actions-run-evidence-import.json"
$canClaimDryRunPack = if ($null -eq $githubActionsEvidence) {
  $false
}
else {
  [bool](Get-PropertyOrDefault -Object $githubActionsEvidence -Name "canClaimGitHubActionsPackageDryRunPackForRun" -DefaultValue $false)
}

$sourceRunId = if ($null -eq $githubActionsEvidence) { "" } else { [string](Get-PropertyOrDefault -Object $githubActionsEvidence -Name "runId" -DefaultValue "") }
$sourceHeadSha = if ($null -eq $githubActionsEvidence) { "" } else { [string](Get-PropertyOrDefault -Object $githubActionsEvidence -Name "headSha" -DefaultValue "") }

$commonBlockedReasons = @(
  "owner-authorization-required",
  "public-package-download-proof-missing",
  "clean-consumer-runtime-proof-missing",
  "post-publish-proof-missing"
)

$routes = @(
  New-PackageRoute `
    -Id "nuget-small-bridge-core" `
    -DisplayName "NuGet small bridge/core package" `
    -PackageId "JYPPX.TensorRT.CSharp.API" `
    -DistributionChannel "nuget.org" `
    -PackageContents "C# public API, XML documentation, and small native bridge assets only; no bundled NVIDIA full runtime claim." `
    -DependencyStrategy "The consumer installs CUDA, TensorRT, and cuDNN separately and exposes them through standard probing paths or environment variables." `
    -PackageDryRunPackSatisfied $canClaimDryRunPack `
    -AdditionalRequirements @(
      New-EvidenceRequirement -Id "public-package-metadata-ready" -Label "nuget.org public package metadata" -Satisfied $false -Source "owner public publish result input" -BlockingReason "public-package-metadata-missing"
      New-EvidenceRequirement -Id "public-package-download-proof" -Label "downloaded public nupkg path and SHA256" -Satisfied $false -Source "owner package consumer proof input" -BlockingReason "public-package-download-proof-missing"
      New-EvidenceRequirement -Id "clean-consumer-restore-build-smoke" -Label "clean external consumer restore/build/runtime smoke" -Satisfied $false -Source "owner package consumer proof input" -BlockingReason "clean-consumer-runtime-proof-missing"
      New-EvidenceRequirement -Id "post-publish-proof" -Label "post-publish restore/build/smoke proof" -Satisfied $false -Source "post-publish owner proof input" -BlockingReason "post-publish-proof-missing"
    ) `
    -BlockedReasons $commonBlockedReasons `
    -ProofBoundary "NuGet small bridge/core route preflight does not execute dotnet nuget push, does not use a publish token, does not bundle or claim full NVIDIA runtime, and is not package-consumer runtime proof."
  New-PackageRoute `
    -Id "github-packages-full-runtime" `
    -DisplayName "GitHub Packages full runtime package" `
    -PackageId "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey" `
    -DistributionChannel "GitHub Packages" `
    -PackageContents "Runtime package lane for heavier native assets, runtime manifest, and version-pinned TensorRT/CUDA/cuDNN dependency layout." `
    -DependencyStrategy "The consumer restores from GitHub Packages with credentials and validates runtime DLL resolution plus clean external runtime smoke." `
    -PackageDryRunPackSatisfied $canClaimDryRunPack `
    -AdditionalRequirements @(
      New-EvidenceRequirement -Id "runtime-package-split-pack-success" -Label "runtime split package pack success" -Satisfied $false -Source "runtime package workflow artifact" -BlockingReason "runtime-package-split-pack-proof-missing"
      New-EvidenceRequirement -Id "github-packages-restore-source-ready" -Label "GitHub Packages restore source and permission evidence" -Satisfied $false -Source "owner publish and consumer input" -BlockingReason "github-packages-restore-source-proof-missing"
      New-EvidenceRequirement -Id "runtime-dll-resolution-report" -Label "runtime DLL resolution report" -Satisfied $false -Source "clean external consumer runtime proof" -BlockingReason "runtime-dll-resolution-proof-missing"
      New-EvidenceRequirement -Id "clean-consumer-runtime-smoke" -Label "clean external runtime smoke" -Satisfied $false -Source "owner package consumer proof input" -BlockingReason "clean-consumer-runtime-proof-missing"
      New-EvidenceRequirement -Id "post-publish-proof" -Label "post-publish restore/build/smoke proof" -Satisfied $false -Source "post-publish owner proof input" -BlockingReason "post-publish-proof-missing"
    ) `
    -BlockedReasons (@("owner-authorization-required", "github-packages-restore-source-proof-missing", "runtime-package-split-pack-proof-missing", "clean-consumer-runtime-proof-missing", "post-publish-proof-missing")) `
    -ProofBoundary "GitHub Packages full runtime route preflight does not publish packages, does not use a publish token, does not prove public NuGet publication, and is not runtime execution proof until Owner-provided restore, DLL resolution, and smoke evidence are accepted."
)

$allBlockedReasons = @($routes | ForEach-Object { $_.blockedReasons } | Sort-Object -Unique)

$matrix = [pscustomobject]@{
  recordKind = "dual-package-publish-preflight-matrix"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runtimePackageKey = $RuntimePackageKey
  sourceGitHubActionsRunId = $sourceRunId
  sourceHeadSha = $sourceHeadSha
  routeCount = $routes.Count
  routes = @($routes)
  blockedReasons = @($allBlockedReasons)
  performsPublish = $false
  usesPublishToken = $false
  requiresOwnerAuthorization = $true
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimRuntimeProof = $false
  canClaimPackageConsumerRuntimeProof = $false
  proofBoundary = "Preflight matrix only. It can carry package dry-run pack evidence as an input signal, but it does not publish NuGet, does not publish GitHub Packages, does not execute dotnet nuget push, and is not package-consumer runtime proof or post-publish proof."
}

$jsonPath = Join-Path $artifactRoot "dual-package-publish-preflight-matrix.json"
$markdownPath = Join-Path $artifactRoot "dual-package-publish-preflight-matrix.md"

$matrix | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$routeRows = $matrix.routes | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.distributionChannel)`` | ``$($_.packageId)`` | ``$($_.canPublishPublicly)`` | ``$($_.canPublishGitHubPackages)`` | ``$($_.canClaimPackageConsumerRuntimeProof)`` | ``$(($_.blockedReasons -join ', '))`` |"
}

$requirementRows = $matrix.routes | ForEach-Object {
  $route = $_
  $route.evidenceRequirements | ForEach-Object {
    "| ``$($route.id)`` | ``$($_.id)`` | ``$($_.satisfied)`` | $($_.label.Replace('|', '\|')) | ``$($_.blockingReason)`` |"
  }
}

$markdown = @"
# Dual Package Publish Preflight Matrix

生成时间：$($matrix.generatedAtUtc)

## 用途

该矩阵将 TensorRtSharp4.0 的发布路线拆成 NuGet small bridge/core 与 GitHub Packages full runtime 两条路线。它只做发布前证据差距预检，不执行 `dotnet nuget push`，不使用 publish token，不上传 NuGet，不上传 GitHub Packages，也不能把 dry-run artifact、local feed、ProjectReference、direct `.nupkg`、template 或 dashboard 当作 package-consumer runtime proof。

| 项目 | 当前值 |
|---|---|
| runtimePackageKey | ``$($matrix.runtimePackageKey)`` |
| sourceGitHubActionsRunId | ``$($matrix.sourceGitHubActionsRunId)`` |
| sourceHeadSha | ``$($matrix.sourceHeadSha)`` |
| routeCount | ``$($matrix.routeCount)`` |
| performsPublish | ``$($matrix.performsPublish)`` |
| usesPublishToken | ``$($matrix.usesPublishToken)`` |
| requiresOwnerAuthorization | ``$($matrix.requiresOwnerAuthorization)`` |
| canPublishPublicly | ``$($matrix.canPublishPublicly)`` |
| canPublishGitHubPackages | ``$($matrix.canPublishGitHubPackages)`` |
| canCloseReleaseIssue | ``$($matrix.canCloseReleaseIssue)`` |

## Routes

| Route | Channel | Package ID | Can Publish Publicly | Can Publish GitHub Packages | Can Claim Package Consumer Runtime Proof | Blocked Reasons |
|---|---|---|---|---|---|---|
$($routeRows -join "`r`n")

## Evidence Requirements

| Route | Requirement | Satisfied | Label | Blocking Reason |
|---|---|---|---|---|
$($requirementRows -join "`r`n")

## Boundary

$($matrix.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Dual package publish preflight matrix written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "Routes=$($matrix.routeCount) PerformsPublish=False UsesPublishToken=False CanPublishPublicly=False CanPublishGitHubPackages=False"
