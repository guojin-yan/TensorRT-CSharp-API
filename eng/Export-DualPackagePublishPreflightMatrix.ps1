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
    [string]$BlockingReason,
    [string]$MissingProofKind = "owner-external-proof",
    [string]$OwnerAction = "owner must import real external proof before publish or close",
    [bool]$ExternalProofRequired = $true
  )

  [pscustomobject]@{
    id = $Id
    label = $Label
    satisfied = $Satisfied
    source = $Source
    blockingReason = $BlockingReason
    missingProofKind = $MissingProofKind
    ownerAction = $OwnerAction
    externalProofRequired = $ExternalProofRequired
    acceptsSubstituteProof = $false
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
    [string]$NextOwnerAction,
    [string]$ExternalProofMissingReason,
    [string]$PostPublishProofMissingReason,
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
    ownerActionRequired = $true
    nextOwnerAction = $NextOwnerAction
    externalProofRequired = $true
    externalProofMissingReason = $ExternalProofMissingReason
    postPublishProofRequired = $true
    postPublishProofMissingReason = $PostPublishProofMissingReason
    acceptsSubstituteProof = $false
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
    -DisplayName "NuGet managed C# API package" `
    -PackageId "JYPPX.TensorRT.CSharp.API" `
    -DistributionChannel "nuget.org" `
    -PackageContents "C# public API assemblies and XML documentation only; no native runtime assets." `
    -DependencyStrategy "The consumer installs a matching project-owned bridge package plus CUDA, TensorRT, cuDNN, and optional NVRTC separately." `
    -PackageDryRunPackSatisfied $canClaimDryRunPack `
    -AdditionalRequirements @(
      New-EvidenceRequirement -Id "public-package-metadata-ready" -Label "nuget.org public package metadata" -Satisfied $false -Source "owner public publish result input" -BlockingReason "public-package-metadata-missing" -MissingProofKind "owner-publish-authorization" -OwnerAction "owner must approve public package metadata and publish lane before any push"
      New-EvidenceRequirement -Id "public-package-download-proof" -Label "downloaded public nupkg path and SHA256" -Satisfied $false -Source "owner package consumer proof input" -BlockingReason "public-package-download-proof-missing" -MissingProofKind "public-package-download-proof" -OwnerAction "owner must import public nupkg download URL path and SHA256 from nuget.org"
      New-EvidenceRequirement -Id "clean-consumer-restore-build-smoke" -Label "clean external consumer restore/build/runtime smoke" -Satisfied $false -Source "owner package consumer proof input" -BlockingReason "clean-consumer-runtime-proof-missing" -MissingProofKind "package-consumer-runtime-proof" -OwnerAction "owner must run clean external consumer restore build and runtime smoke without local feed or ProjectReference"
      New-EvidenceRequirement -Id "post-publish-proof" -Label "post-publish restore/build/smoke proof" -Satisfied $false -Source "post-publish owner proof input" -BlockingReason "post-publish-proof-missing" -MissingProofKind "post-publish-proof" -OwnerAction "owner must import post-publish restore build smoke logs hashes and host metadata"
    ) `
    -BlockedReasons $commonBlockedReasons `
    -NextOwnerAction "owner-authorize-public-nuget-publish-and-import-clean-external-consumer-proof" `
    -ExternalProofMissingReason "public-package-download-and-clean-consumer-runtime-proof-missing" `
    -PostPublishProofMissingReason "post-publish-clean-consumer-proof-missing" `
    -ProofBoundary "NuGet managed API route preflight does not execute dotnet nuget push, does not use a publish token, does not contain native runtime assets, and is not package-consumer runtime proof."
  New-PackageRoute `
    -Id "github-packages-bridge" `
    -DisplayName "GitHub Packages bridge package" `
    -PackageId "JYPPX.TensorRT.CSharp.API.Runtime.$RuntimePackageKey.Bridge" `
    -DistributionChannel "GitHub Packages" `
    -PackageContents "One project-owned native bridge binary for the selected TensorRT/CUDA build line; no NVIDIA vendor runtime libraries." `
    -DependencyStrategy "The consumer restores the bridge from GitHub Packages, installs matching NVIDIA dependencies separately, and validates bridge resolution plus clean external runtime smoke." `
    -PackageDryRunPackSatisfied $canClaimDryRunPack `
    -AdditionalRequirements @(
      New-EvidenceRequirement -Id "bridge-package-pack-success" -Label "bridge package pack success" -Satisfied $false -Source "bridge package workflow artifact" -BlockingReason "bridge-package-pack-proof-missing" -MissingProofKind "bridge-package-pack-proof" -OwnerAction "owner must import successful bridge package pack artifact and SHA256"
      New-EvidenceRequirement -Id "github-packages-restore-source-ready" -Label "GitHub Packages restore source and permission evidence" -Satisfied $false -Source "owner publish and consumer input" -BlockingReason "github-packages-restore-source-proof-missing" -MissingProofKind "github-packages-restore-source-proof" -OwnerAction "owner must import GitHub Packages restore source and credentialed restore evidence"
      New-EvidenceRequirement -Id "bridge-and-external-dependency-resolution-report" -Label "bridge and external dependency resolution report" -Satisfied $false -Source "clean external consumer runtime proof" -BlockingReason "bridge-dependency-resolution-proof-missing" -MissingProofKind "bridge-dependency-resolution-proof" -OwnerAction "owner must import bridge resolution and system-installed NVIDIA dependency diagnostics"
      New-EvidenceRequirement -Id "clean-consumer-runtime-smoke" -Label "clean external runtime smoke" -Satisfied $false -Source "owner package consumer proof input" -BlockingReason "clean-consumer-runtime-proof-missing" -MissingProofKind "package-consumer-runtime-proof" -OwnerAction "owner must run clean external runtime smoke from GitHub Packages without local feed or ProjectReference"
      New-EvidenceRequirement -Id "post-publish-proof" -Label "post-publish restore/build/smoke proof" -Satisfied $false -Source "post-publish owner proof input" -BlockingReason "post-publish-proof-missing" -MissingProofKind "post-publish-proof" -OwnerAction "owner must import post-publish restore build smoke logs hashes and host metadata"
    ) `
    -BlockedReasons (@("owner-authorization-required", "github-packages-restore-source-proof-missing", "bridge-package-pack-proof-missing", "clean-consumer-runtime-proof-missing", "post-publish-proof-missing")) `
    -NextOwnerAction "owner-authorize-github-bridge-publish-and-import-clean-runtime-proof" `
    -ExternalProofMissingReason "github-bridge-restore-external-dependency-resolution-clean-smoke-missing" `
    -PostPublishProofMissingReason "post-publish-github-bridge-clean-consumer-proof-missing" `
    -ProofBoundary "GitHub Packages bridge route preflight does not publish packages, does not use a publish token, does not bundle NVIDIA runtime libraries, and is not runtime execution proof until Owner-provided restore, dependency resolution, and smoke evidence are accepted."
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
  ownerActionRequired = $true
  nextOwnerAction = "collect-owner-authorization-external-consumer-and-post-publish-proof-before-any-publish-or-close"
  externalProofRequired = $true
  externalProofMissingReason = "clean-external-package-consumer-runtime-proof-missing"
  postPublishProofRequired = $true
  postPublishProofMissingReason = "post-publish-restore-build-smoke-proof-missing"
  acceptsSubstituteProof = $false
  proofBoundary = "Preflight matrix only. It can carry package dry-run pack evidence as an input signal, but it does not publish NuGet, does not publish GitHub Packages, does not execute dotnet nuget push, and is not package-consumer runtime proof or post-publish proof."
}

$jsonPath = Join-Path $artifactRoot "dual-package-publish-preflight-matrix.json"
$markdownPath = Join-Path $artifactRoot "dual-package-publish-preflight-matrix.md"

$matrix | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$routeRows = $matrix.routes | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.distributionChannel)`` | ``$($_.packageId)`` | ``$($_.canPublishPublicly)`` | ``$($_.canPublishGitHubPackages)`` | ``$($_.canClaimPackageConsumerRuntimeProof)`` | ``$($_.externalProofMissingReason)`` | ``$($_.postPublishProofMissingReason)`` | ``$($_.nextOwnerAction)`` | ``$(($_.blockedReasons -join ', '))`` |"
}

$requirementRows = $matrix.routes | ForEach-Object {
  $route = $_
  $route.evidenceRequirements | ForEach-Object {
    "| ``$($route.id)`` | ``$($_.id)`` | ``$($_.satisfied)`` | $($_.label.Replace('|', '\|')) | ``$($_.missingProofKind)`` | ``$($_.blockingReason)`` | ``$($_.ownerAction)`` | ``$($_.acceptsSubstituteProof)`` |"
  }
}

$markdown = @"
# Dual Package Publish Preflight Matrix

生成时间：$($matrix.generatedAtUtc)

## 用途

该矩阵将 TensorRtSharp4.0 的发布路线拆成 NuGet managed package 与 GitHub Packages bridge-only package 两条路线。两条路线都禁止 NVIDIA vendor runtime 资产。它只做发布前证据差距预检，不执行 `dotnet nuget push`，不使用 publish token，不上传 NuGet，不上传 GitHub Packages，也不能把 dry-run artifact、local feed、ProjectReference、direct `.nupkg`、template 或 dashboard 当作 package-consumer runtime proof。

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
| ownerActionRequired | ``$($matrix.ownerActionRequired)`` |
| nextOwnerAction | ``$($matrix.nextOwnerAction)`` |
| externalProofMissingReason | ``$($matrix.externalProofMissingReason)`` |
| postPublishProofMissingReason | ``$($matrix.postPublishProofMissingReason)`` |
| acceptsSubstituteProof | ``$($matrix.acceptsSubstituteProof)`` |

## Routes

| Route | Channel | Package ID | Can Publish Publicly | Can Publish GitHub Packages | Can Claim Package Consumer Runtime Proof | External Proof Missing Reason | Post-Publish Proof Missing Reason | Next Owner Action | Blocked Reasons |
|---|---|---|---|---|---|---|---|---|---|
$($routeRows -join "`r`n")

## Evidence Requirements

| Route | Requirement | Satisfied | Label | Missing Proof Kind | Blocking Reason | Owner Action | Accepts Substitute Proof |
|---|---|---|---|---|---|---|---|
$($requirementRows -join "`r`n")

## Boundary

$($matrix.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Dual package publish preflight matrix written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "Routes=$($matrix.routeCount) PerformsPublish=False UsesPublishToken=False CanPublishPublicly=False CanPublishGitHubPackages=False"
