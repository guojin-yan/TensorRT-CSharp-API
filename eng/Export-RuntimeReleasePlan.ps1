[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$Ref = "TensorRtSharp4.0",
  [string]$SuggestedVersion,
  [string]$InventoryJsonPath,
  [string]$RuntimePublicationTargetCoverageJsonPath,
  [string]$ReleaseReadinessJsonPath,
  [string]$RepositoryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($Repository)) { $Repository = $env:GITHUB_REPOSITORY }
if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
  throw "Repository must use owner/name form: '$Repository'."
}
if ([string]::IsNullOrWhiteSpace($Ref)) { $Ref = $env:GITHUB_REF_NAME }
if ([string]::IsNullOrWhiteSpace($Ref)) { $Ref = "TensorRtSharp4.0" }
if ([string]::IsNullOrWhiteSpace($SuggestedVersion)) { $SuggestedVersion = "4.0.0" }

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-OptionalPath {
  param([AllowNull()][string]$Path, [string]$DefaultRelativePath)

  $value = if ([string]::IsNullOrWhiteSpace($Path)) { $DefaultRelativePath } else { $Path }
  if (-not [System.IO.Path]::IsPathRooted($value)) { $value = Join-Path $RepositoryRoot $value }
  return [System.IO.Path]::GetFullPath($value)
}

function Get-FileEvidence {
  param([Parameter(Mandatory = $true)][string]$Path)

  $exists = Test-Path -LiteralPath $Path -PathType Leaf
  [pscustomobject]@{
    path = $Path
    exists = $exists
    sha256 = if ($exists) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
  }
}

function Format-CommandArgument {
  param([Parameter(Mandatory = $true)][string]$Value)
  if ($Value -notmatch '[\s`"'']') { return $Value }
  return "'" + $Value.Replace("'", "''") + "'"
}

function New-WorkflowPlan {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Workflow,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Purpose
  )

  $parts = @("gh", "workflow", "run", $Workflow, "--repo", $Repository, "--ref", $Ref) + $Arguments
  [pscustomobject]@{
    id = $Id
    workflow = $Workflow
    purpose = $Purpose
    command = ($parts | ForEach-Object { Format-CommandArgument -Value ([string]$_) }) -join " "
    dryRunOnly = $true
    performsPublish = $false
    canPublishPublicly = $false
    requiresOwnerAuthorizationForPublishFlags = $true
  }
}

$policyPath = Join-Path $RepositoryRoot "pack\external-vendor-runtime-policy.json"
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$linuxTargetCatalogPath = Join-Path $RepositoryRoot "pack\runtime\linux-runtime-targets.manifest.json"
foreach ($requiredPath in @($policyPath, $runtimeManifestPath, $splitManifestPath, $linuxTargetCatalogPath)) {
  if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
    throw "Required release planning contract was not found: $requiredPath"
  }
}

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$linuxTargetCatalog = Get-Content -LiteralPath $linuxTargetCatalogPath -Raw -Encoding utf8 | ConvertFrom-Json

$bridgeByRuntimeKey = @{}
foreach ($package in @($splitManifest.packages | Where-Object { [string]$_.role -eq "bridge" })) {
  $bridgeByRuntimeKey[[string]$package.sourceRuntimeKey] = $package
}

$targets = New-Object System.Collections.Generic.List[object]
foreach ($package in @($runtimeManifest.packages | Sort-Object key)) {
  $runtimeKey = [string]$package.key
  $staticBridge = if ($bridgeByRuntimeKey.ContainsKey($runtimeKey)) { $bridgeByRuntimeKey[$runtimeKey] } else { $null }
  $basePackageId = [string]$package.packageId
  $bridgePackageId = if ($null -ne $staticBridge) { [string]$staticBridge.packageId } else { "$basePackageId.Bridge" }
  $bridgeAsset = if ($null -ne $staticBridge) {
    @($staticBridge.assets | ForEach-Object { [string]$_ })
  }
  elseif ([string]$package.platform -eq "linux") {
    @("libjyppxtrtbridge.so")
  }
  else {
    @("jyppxtrtbridge.dll")
  }

  $targets.Add([pscustomobject]@{
      runtimeKey = $runtimeKey
      platform = [string]$package.platform
      rid = [string]$package.rid
      bridgePackageId = $bridgePackageId
      allowedRole = "bridge"
      expectedNativeAssets = @($bridgeAsset)
      vendorRuntimeBundled = $false
      systemInstalledVendorDependenciesRequired = $true
      manifestValidationState = [string]$package.validationState
      planState = "bridge-build-validation-candidate"
      performsPublish = $false
      canPublishPublicly = $false
    }) | Out-Null
}

$windowsKeys = @($targets | Where-Object platform -eq "windows" | ForEach-Object runtimeKey)
$linuxKeys = @($targets | Where-Object platform -eq "linux" | ForEach-Object runtimeKey)
$windowsKeyText = $windowsKeys -join ","
$linuxKeyText = $linuxKeys -join ","
$formalRepository = "guojin-yan/TensorRT-CSharp-API"
$isFormalRepository = $Repository.Equals($formalRepository, [System.StringComparison]::OrdinalIgnoreCase)

$commands = @(
  New-WorkflowPlan -Id "managed-dry-run" -Workflow "package-managed.yml" -Purpose "Pack and validate the managed C# API without publication." -Arguments @(
    "-f", "version=$SuggestedVersion",
    "-f", "publish_to_nuget=false",
    "-f", "publish_to_github_packages=false",
    "-f", "attach_to_github_release=false"
  )
  New-WorkflowPlan -Id "windows-bridge-dry-run" -Workflow "runtime-windows.yml" -Purpose "Build and consume all modeled Windows bridge-only packages without publication." -Arguments @(
    "-f", "version=$SuggestedVersion",
    "-f", "runtime_keys=$windowsKeyText",
    "-f", "runtime_delivery_mode=split",
    "-f", "split_package_roles=bridge",
    "-f", "include_meta_package=false",
    "-f", "publish_to_nuget=false",
    "-f", "publish_to_github_packages=false",
    "-f", "attach_to_github_release=false"
  )
  New-WorkflowPlan -Id "linux-bridge-dry-run" -Workflow "runtime-linux.yml" -Purpose "Build modeled Linux bridge-only packages without publication." -Arguments @(
    "-f", "version=$SuggestedVersion",
    "-f", "runtime_keys=$linuxKeyText",
    "-f", "runtime_delivery_mode=split",
    "-f", "split_package_roles=bridge",
    "-f", "include_meta_package=false",
    "-f", "publish_to_nuget=false",
    "-f", "publish_to_github_packages=false",
    "-f", "attach_to_github_release=false"
  )
  New-WorkflowPlan -Id "ubuntu20-hosted-container" -Workflow "runtime-linux.yml" -Purpose "Validate the modeled Ubuntu 20.04 bridge-only line in its hosted container." -Arguments @(
    "-f", "version=$SuggestedVersion",
    "-f", "runtime_key_set=hosted-container-ubuntu20",
    "-f", "runner_mode=hosted-container",
    "-f", "runtime_delivery_mode=split",
    "-f", "split_package_roles=bridge",
    "-f", "include_meta_package=false",
    "-f", "publish_to_nuget=false",
    "-f", "publish_to_github_packages=false",
    "-f", "attach_to_github_release=false"
  )
  New-WorkflowPlan -Id "source-dry-run" -Workflow "package-source.yml" -Purpose "Create a tracked-files-only source archive without Release upload." -Arguments @(
    "-f", "version=$SuggestedVersion",
    "-f", "attach_to_github_release=false"
  )
)

$inventoryEvidence = Get-FileEvidence -Path (Resolve-OptionalPath -Path $InventoryJsonPath -DefaultRelativePath "artifacts\publication-inventory\github-publication-inventory.json")
$coverageEvidence = Get-FileEvidence -Path (Resolve-OptionalPath -Path $RuntimePublicationTargetCoverageJsonPath -DefaultRelativePath "artifacts\runtime-publication-target-coverage\runtime-publication-target-coverage.json")
$readinessEvidence = Get-FileEvidence -Path (Resolve-OptionalPath -Path $ReleaseReadinessJsonPath -DefaultRelativePath "artifacts\release-readiness\release-readiness.json")

$result = [pscustomobject]@{
  schemaVersion = 2
  recordKind = "runtime-release-plan"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  planState = "bridge-only-owner-authorization-required"
  repository = $Repository
  ref = $Ref
  formalRepository = $formalRepository
  isFormalRepository = $isFormalRepository
  suggestedVersion = $SuggestedVersion
  publicationPolicy = "bridge-only"
  publicationPolicyId = [string]$policy.policyId
  allowedPackageKinds = @($policy.allowedPackageKinds)
  allowedReleaseAssets = @("managed nupkg", "bridge nupkg", "tracked source archive")
  retiredPackageKinds = @("full-runtime", "cuda-cudnn", "tensorrt", "cuda-rtc", "collection", "meta")
  vendorRuntimePackagesForbidden = $true
  systemInstalledVendorDependenciesRequired = $true
  runtimeTargetCount = $targets.Count
  windowsTargetCount = $windowsKeys.Count
  linuxTargetCount = $linuxKeys.Count
  dispatchableNextTargets = @($targets.ToArray())
  workflowPlans = @($commands)
  retiredStableDependencyPinMaps = [pscustomobject]@{
    state = "retired-empty"
    vendorPackagePinningAllowed = $false
    windows = [pscustomobject]@{}
    linux = [pscustomobject]@{}
  }
  sourceEvidence = [pscustomobject]@{
    publicationInventory = $inventoryEvidence
    runtimePublicationCoverage = $coverageEvidence
    releaseReadiness = $readinessEvidence
    linuxTargetCatalogPath = $linuxTargetCatalogPath
    linuxTargetCatalogSha256 = (Get-FileHash -LiteralPath $linuxTargetCatalogPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  futureTargetBoundary = "ARM/SBSA, Jetson/L4T, and non-Ubuntu targets remain future separate package lines requiring dedicated bridge identities, runners, and proof."
  ownerPublishDelta = [pscustomobject]@{
    required = $true
    formalRepositoryOnly = $true
    instructions = "After Owner authorization and all pre-publish gates pass, change only the managed/bridge/source publish or attach flags for the approved version and Release tag. Never populate retired vendor/meta inputs."
  }
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "Runtime Release Plan is a bridge-only dry-run action map. It does not dispatch workflows, publish packages, upload Release assets, redistribute NVIDIA libraries, establish runtime proof, or close the release."
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-release-plan"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "runtime-release-plan.json"
$markdownPath = Join-Path $outputRoot "runtime-release-plan.md"
$result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$targetRows = @($result.dispatchableNextTargets | ForEach-Object {
    "| ``$($_.runtimeKey)`` | ``$($_.platform)`` | ``$($_.bridgePackageId)`` | ``$($_.manifestValidationState)`` |"
  })
$commandRows = @($result.workflowPlans | ForEach-Object {
    "| ``$($_.id)`` | $($_.purpose.Replace('|', '\|')) | ``$($_.command.Replace('|', '\|'))`` |"
  })
$markdown = @"
# Runtime Release Plan

- State: ``$($result.planState)``
- Repository: ``$Repository``
- Formal repository: ``$isFormalRepository``
- Publication policy: ``$($result.publicationPolicy)``
- Runtime targets: ``$($result.runtimeTargetCount)``
- Vendor runtime packages forbidden: ``True``
- Performs publish: ``False``

## Bridge Targets

| Runtime key | Platform | Bridge package | Manifest state |
| --- | --- | --- | --- |
$($targetRows -join "`r`n")

## Dry-run Commands

| ID | Purpose | Command |
| --- | --- | --- |
$($commandRows -join "`r`n")

## Owner Boundary

$($result.ownerPublishDelta.instructions)

$($result.boundary)
"@
Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Host "Runtime release plan written: $jsonPath"
Write-Host "Runtime release plan written: $markdownPath"
$result | ConvertTo-Json -Depth 10
