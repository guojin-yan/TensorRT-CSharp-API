[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$CapabilityMatrixPath,
  [string]$LocalManifestPath,
  [string]$OutputPath,
  [switch]$RequireMaterializationReady
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
$RepositoryRoot = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
if ([string]::IsNullOrWhiteSpace($CapabilityMatrixPath)) {
  $CapabilityMatrixPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\capability-matrix.json'
}
if ([string]::IsNullOrWhiteSpace($LocalManifestPath)) {
  $LocalManifestPath = Join-Path $RepositoryRoot 'pack\runtime\runtime-packages.local.json'
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\full-runtime-packaging-preflight.json'
}

$runtimeManifestPath = Join-Path $RepositoryRoot 'pack\runtime\runtime-packages.manifest.json'
$splitManifestPath = Join-Path $RepositoryRoot 'pack\runtime-split\split-runtime-packages.manifest.json'
foreach ($requiredPath in @($runtimeManifestPath, $splitManifestPath, $CapabilityMatrixPath)) {
  if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
    throw "Required CUDA RTC packaging input was not found: $requiredPath"
  }
}

function Read-Json {
  param([Parameter(Mandatory = $true)][string]$Path)
  return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Write-Utf8Text {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Value
  )
  [System.IO.File]::WriteAllText($Path, $Value, $utf8)
}

function Add-UniqueText {
  param(
    [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.List[string]]$List,
    [Parameter(Mandatory = $true)][string]$Value
  )
  if (-not $List.Contains($Value)) {
    $List.Add($Value)
  }
}

function Get-NamedPropertyValue {
  param(
    [Parameter(Mandatory = $true)][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name
  )
  $property = $Object.PSObject.Properties | Where-Object { $_.Name -eq $Name } | Select-Object -First 1
  if ($null -eq $property) {
    return $null
  }
  return $property.Value
}

function Get-ToolkitRoot {
  param(
    [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$LocalPackages,
    [Parameter(Mandatory = $true)][string[]]$RuntimeKeys,
    [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.List[string]]$StructuralFindings
  )

  $roots = @(
    $LocalPackages |
      Where-Object { [string]$_.key -in $RuntimeKeys -and -not [string]::IsNullOrWhiteSpace([string]$_.defaultCudaRoot) } |
      ForEach-Object { [System.IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables([string]$_.defaultCudaRoot)).TrimEnd('\', '/') } |
      Sort-Object -Unique
  )
  if ($roots.Count -gt 1) {
    Add-UniqueText -List $StructuralFindings -Value "Runtime keys '$($RuntimeKeys -join ', ')' resolve to multiple CUDA roots."
    return $null
  }
  if ($roots.Count -eq 0) {
    return $null
  }
  return $roots[0]
}

function New-AssetRecord {
  param(
    [Parameter(Mandatory = $true)][string]$Kind,
    [Parameter(Mandatory = $true)][string]$RelativePath,
    [AllowNull()][string]$ToolkitRoot,
    [long]$ExpectedSizeBytes,
    [AllowNull()][string]$ExpectedSha256
  )

  $path = if ([string]::IsNullOrWhiteSpace($ToolkitRoot)) {
    $null
  }
  else {
    Join-Path $ToolkitRoot ($RelativePath.Replace('/', '\'))
  }
  $exists = -not [string]::IsNullOrWhiteSpace($path) -and (Test-Path -LiteralPath $path -PathType Leaf)
  $actualSize = if ($exists) { (Get-Item -LiteralPath $path).Length } else { 0L }
  $actualSha = if ($exists) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
  $expectedSha = if ([string]::IsNullOrWhiteSpace($ExpectedSha256)) { $null } else { $ExpectedSha256.ToLowerInvariant() }
  $sizeMatches = $exists -and $ExpectedSizeBytes -gt 0 -and $actualSize -eq $ExpectedSizeBytes
  $hashMatches = $exists -and $null -ne $expectedSha -and [string]::Equals($actualSha, $expectedSha, [System.StringComparison]::OrdinalIgnoreCase)

  return [ordered]@{
    kind = $Kind
    relativePath = $RelativePath.Replace('\', '/')
    localPath = $path
    exists = $exists
    expectedSizeBytes = $ExpectedSizeBytes
    actualSizeBytes = $actualSize
    sizeMatches = $sizeMatches
    expectedSha256 = $expectedSha
    actualSha256 = $actualSha
    hashMatches = $hashMatches
    integrityReady = $exists -and $sizeMatches -and $hashMatches
  }
}

$runtimeManifest = Read-Json -Path $runtimeManifestPath
$splitManifest = Read-Json -Path $splitManifestPath
$capabilityMatrix = Read-Json -Path $CapabilityMatrixPath
$localManifest = if (Test-Path -LiteralPath $LocalManifestPath -PathType Leaf) {
  Read-Json -Path $LocalManifestPath
}
else {
  [pscustomobject]@{ packages = @() }
}

$policy = $runtimeManifest.cudaRtcPackaging
$role = $splitManifest.cudaRtcSplitRole
$structuralFindings = [System.Collections.Generic.List[string]]::new()
$blockers = [System.Collections.Generic.List[string]]::new()

if ($null -eq $policy) {
  Add-UniqueText -List $structuralFindings -Value 'runtime-packages.manifest.json is missing cudaRtcPackaging.'
}
if ($null -eq $role) {
  Add-UniqueText -List $structuralFindings -Value 'split-runtime-packages.manifest.json is missing cudaRtcSplitRole.'
}
if ($null -ne $policy -and [string]$policy.dependencyMode -ne 'optional-dynamic') {
  Add-UniqueText -List $structuralFindings -Value 'CUDA RTC dependencyMode must remain optional-dynamic.'
}
if ($null -ne $policy -and [bool]$policy.bridgeOnlyBundlesNvrtc) {
  Add-UniqueText -List $structuralFindings -Value 'Bridge-only packages must not bundle NVRTC.'
}
if ($null -ne $role -and [string]$role.role -ne 'cuda-rtc') {
  Add-UniqueText -List $structuralFindings -Value 'The split CUDA RTC role must be named cuda-rtc.'
}
if ($null -ne $role -and [bool]$role.bridgePackagesReferenceRole) {
  Add-UniqueText -List $structuralFindings -Value 'Bridge packages must not reference the cuda-rtc role.'
}

$approvalState = [string]$policy.redistributionApprovalState
$redistributionApproved = $approvalState -in @('approved', 'owner-approved', 'approved-for-selected-channel')
if (-not $redistributionApproved) {
  Add-UniqueText -List $blockers -Value 'redistribution-approval-pending'
}
$packageHostSizeReviewState = [string]$policy.packageHostSizeReviewState
$packageHostSizeReviewApproved = $packageHostSizeReviewState -in @('approved', 'owner-approved', 'approved-for-selected-channel')
if (-not $packageHostSizeReviewApproved) {
  Add-UniqueText -List $blockers -Value 'package-host-size-review-pending'
}

$runtimePackages = @($runtimeManifest.packages)
$localPackages = @($localManifest.packages)
$windowsRows = [System.Collections.Generic.List[object]]::new()
$windowsGroups = @($runtimePackages | Where-Object { [string]$_.platform -eq 'windows' } | Group-Object cudaVersion | Sort-Object Name)
foreach ($group in $windowsGroups) {
  $version = [string]$group.Name
  $runtimeKeys = @($group.Group | ForEach-Object { [string]$_.key } | Sort-Object -Unique)
  $capability = @($capabilityMatrix.windows | Where-Object { [string]$_.toolkitVersion -eq $version })
  if ($capability.Count -ne 1) {
    Add-UniqueText -List $structuralFindings -Value "CUDA $version must have exactly one Windows capability row."
    continue
  }
  $capability = $capability[0]

  $policyAssets = @(Get-NamedPropertyValue -Object $policy.windowsAssetsByCudaVersion -Name $version)
  if ($policyAssets.Count -ne 2) {
    Add-UniqueText -List $structuralFindings -Value "CUDA $version must define exactly one NVRTC and one builtins asset."
  }
  $capabilityAssets = @([string]$capability.runtimeRelativePath, [string]$capability.builtinsRelativePath)
  $policyAssetKey = (@($policyAssets | ForEach-Object { ([string]$_).Replace('\', '/') } | Sort-Object) -join '|')
  $capabilityAssetKey = (@($capabilityAssets | ForEach-Object { ([string]$_).Replace('\', '/') } | Sort-Object) -join '|')
  $pairMatchesCapability = [string]::Equals($policyAssetKey, $capabilityAssetKey, [System.StringComparison]::Ordinal)
  if (-not $pairMatchesCapability) {
    Add-UniqueText -List $structuralFindings -Value "CUDA $version packaging asset pair differs from the capability matrix."
  }

  $toolkitRoot = Get-ToolkitRoot -LocalPackages $localPackages -RuntimeKeys $runtimeKeys -StructuralFindings $structuralFindings
  $runtimeAsset = New-AssetRecord -Kind 'nvrtc' -RelativePath ([string]$capability.runtimeRelativePath) `
    -ToolkitRoot $toolkitRoot -ExpectedSizeBytes ([long]$capability.runtimeSizeBytes) -ExpectedSha256 ([string]$capability.runtimeSha256)
  $builtinsAsset = New-AssetRecord -Kind 'nvrtc-builtins' -RelativePath ([string]$capability.builtinsRelativePath) `
    -ToolkitRoot $toolkitRoot -ExpectedSizeBytes ([long]$capability.builtinsSizeBytes) -ExpectedSha256 ([string]$capability.builtinsSha256)

  $licenseCandidates = if ([string]::IsNullOrWhiteSpace($toolkitRoot)) {
    @()
  }
  else {
    @('EULA.txt', 'LICENSE') | ForEach-Object { Join-Path $toolkitRoot $_ } | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
  }
  $licensePath = @($licenseCandidates | Select-Object -First 1)
  $licensePresent = $licensePath.Count -eq 1
  $licenseSha = if ($licensePresent) { (Get-FileHash -LiteralPath $licensePath[0] -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
  $assetIntegrityReady = [bool]$runtimeAsset.integrityReady -and [bool]$builtinsAsset.integrityReady
  if (-not $assetIntegrityReady) {
    Add-UniqueText -List $blockers -Value "windows-cuda-$version-assets-not-ready"
  }
  if (-not $licensePresent) {
    Add-UniqueText -List $blockers -Value "windows-cuda-$version-license-text-not-found"
  }
  $assetPairSizeBytes = [long]$runtimeAsset.actualSizeBytes + [long]$builtinsAsset.actualSizeBytes

  $windowsRows.Add([ordered]@{
      toolkitVersion = $version
      runtimeKeys = $runtimeKeys
      toolkitRoot = $toolkitRoot
      capabilityEvidenceState = [string]$capability.evidenceState
      expectedAssetPair = @($policyAssets | ForEach-Object { ([string]$_).Replace('\', '/') })
      pairMatchesCapabilityMatrix = $pairMatchesCapability
      assets = @($runtimeAsset, $builtinsAsset)
      assetPairSizeBytes = $assetPairSizeBytes
      assetPairSizeMiB = [Math]::Round($assetPairSizeBytes / 1MB, 2)
      assetIntegrityReady = $assetIntegrityReady
      licenseTextPresent = $licensePresent
      licensePath = if ($licensePresent) { $licensePath[0] } else { $null }
      licenseSha256 = $licenseSha
      redistributionApprovalState = $approvalState
      localAssetStagingReady = $pairMatchesCapability -and $assetIntegrityReady -and $licensePresent
      packageMaterializationReady = $pairMatchesCapability -and $assetIntegrityReady -and $licensePresent -and
        $redistributionApproved -and $packageHostSizeReviewApproved
    })
}

$linuxRows = [System.Collections.Generic.List[object]]::new()
$linuxGroups = @($runtimePackages | Where-Object { [string]$_.platform -eq 'linux' } | Group-Object cudaVersion | Sort-Object Name)
$linuxPatterns = @($policy.linuxAssetPatterns)
$hasLinuxNvrtcPattern = @($linuxPatterns | Where-Object { [string]$_ -match 'libnvrtc\.so' -and [string]$_ -notmatch 'builtins' }).Count -gt 0
$hasLinuxBuiltinsPattern = @($linuxPatterns | Where-Object { [string]$_ -match 'libnvrtc-builtins\.so' }).Count -gt 0
if (-not $hasLinuxNvrtcPattern -or -not $hasLinuxBuiltinsPattern) {
  Add-UniqueText -List $structuralFindings -Value 'Linux CUDA RTC policy must include NVRTC and builtins asset patterns.'
}
foreach ($group in $linuxGroups) {
  $version = [string]$group.Name
  $runtimeKeys = @($group.Group | ForEach-Object { [string]$_.key } | Sort-Object -Unique)
  $capability = @($capabilityMatrix.linux | Where-Object { [string]$_.toolkitVersion -eq $version })
  if ($capability.Count -ne 1) {
    Add-UniqueText -List $structuralFindings -Value "CUDA $version must have exactly one Linux capability row."
    continue
  }
  $capability = $capability[0]
  $assetsVerified = [string]$capability.evidenceState -ne 'unverified-local-assets-not-found' -and
    [int]$capability.localLibNvrtcAssetCount -gt 0
  if (-not $assetsVerified) {
    Add-UniqueText -List $blockers -Value 'linux-assets-unverified'
  }
  $linuxRows.Add([ordered]@{
      toolkitVersion = $version
      runtimeKeys = $runtimeKeys
      expectedAssetPatterns = $linuxPatterns
      capabilityEvidenceState = [string]$capability.evidenceState
      localAssetCount = [int]$capability.localLibNvrtcAssetCount
      soname = $capability.soname
      assetsVerified = $assetsVerified
      localAssetStagingReady = $assetsVerified
      packageMaterializationReady = $assetsVerified -and $redistributionApproved
      boundary = [string]$capability.boundary
    })
}

$roleMaterialized = [string]$role.prototypeState -ne 'planned-not-materialized' -and
  @($splitManifest.packages | Where-Object { [string]$_.role -eq 'cuda-rtc' }).Count -gt 0
if (-not $roleMaterialized) {
  Add-UniqueText -List $blockers -Value 'cuda-rtc-role-not-materialized'
}
$windowsAssetStagingReady = $windowsRows.Count -gt 0 -and @($windowsRows | Where-Object { -not [bool]$_.localAssetStagingReady }).Count -eq 0
$linuxAssetStagingReady = $linuxRows.Count -gt 0 -and @($linuxRows | Where-Object { -not [bool]$_.localAssetStagingReady }).Count -eq 0
$windowsUniqueAssetBytes = [long](($windowsRows | ForEach-Object { [long]$_.assetPairSizeBytes } | Measure-Object -Sum).Sum)
$canMaterialize = $structuralFindings.Count -eq 0 -and $windowsAssetStagingReady -and $linuxAssetStagingReady -and
  $redistributionApproved -and $packageHostSizeReviewApproved -and $roleMaterialized

$report = [ordered]@{
  schemaVersion = 1
  recordKind = 'cuda-rtc-full-runtime-packaging-preflight'
  generatedLocalDate = (Get-Date -Format 'yyyy-MM-dd')
  proofClassification = 'local-full-runtime-packaging-preflight'
  performsDownload = $false
  performsAssetCopy = $false
  performsPackaging = $false
  performsPublish = $false
  inputs = [ordered]@{
    runtimeManifest = 'pack/runtime/runtime-packages.manifest.json'
    splitManifest = 'pack/runtime-split/split-runtime-packages.manifest.json'
    capabilityMatrix = 'artifacts/cuda-runtime-compilation/capability-matrix.json'
    localManifestPresent = Test-Path -LiteralPath $LocalManifestPath -PathType Leaf
    localManifestPath = $LocalManifestPath
  }
  contract = [ordered]@{
    dependencyMode = [string]$policy.dependencyMode
    bridgeOnlyBundlesNvrtc = [bool]$policy.bridgeOnlyBundlesNvrtc
    fullRuntimeBundleState = [string]$policy.fullRuntimeBundleState
    splitRole = [string]$role.role
    splitRolePrototypeState = [string]$role.prototypeState
    bridgePackagesReferenceRole = [bool]$role.bridgePackagesReferenceRole
    redistributionApprovalState = $approvalState
    packageHostSizeReviewState = $packageHostSizeReviewState
  }
  windows = @($windowsRows)
  linux = @($linuxRows)
  summary = [ordered]@{
    runtimeKeyCount = $runtimePackages.Count
    windowsRuntimeKeyCount = @($runtimePackages | Where-Object { [string]$_.platform -eq 'windows' }).Count
    linuxRuntimeKeyCount = @($runtimePackages | Where-Object { [string]$_.platform -eq 'linux' }).Count
    windowsToolkitVersionCount = $windowsRows.Count
    windowsAssetPairReadyCount = @($windowsRows | Where-Object { [bool]$_.assetIntegrityReady }).Count
    windowsLicenseTextPresentCount = @($windowsRows | Where-Object { [bool]$_.licenseTextPresent }).Count
    windowsUniqueAssetBytes = $windowsUniqueAssetBytes
    windowsUniqueAssetMiB = [Math]::Round($windowsUniqueAssetBytes / 1MB, 2)
    windowsAssetStagingReady = $windowsAssetStagingReady
    linuxToolkitVersionCount = $linuxRows.Count
    linuxAssetReadyCount = @($linuxRows | Where-Object { [bool]$_.assetsVerified }).Count
    linuxAssetStagingReady = $linuxAssetStagingReady
    redistributionApproved = $redistributionApproved
    packageHostSizeReviewApproved = $packageHostSizeReviewApproved
    roleMaterialized = $roleMaterialized
    structuralFindingCount = $structuralFindings.Count
    materializationBlockerCount = $blockers.Count
    canMaterializeFullRuntimeCudaRtcRole = $canMaterialize
    canPublish = $false
  }
  structuralFindings = @($structuralFindings)
  materializationBlockers = @($blockers)
  proofBoundary = 'This preflight verifies local asset identity, manifest pairing, and license-text presence. It does not grant redistribution approval, materialize a package, prove Linux assets, publish anything, or promote public/post-publish proof.'
}

[void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $OutputPath))
Write-Utf8Text -Path $OutputPath -Value ($report | ConvertTo-Json -Depth 20)
$markdownPath = [System.IO.Path]::ChangeExtension($OutputPath, '.md')
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# CUDA RTC Full-Runtime Packaging Preflight')
$lines.Add('')
$lines.Add("- Classification: ``$($report.proofClassification)``")
$lines.Add("- Runtime keys: ``$($report.summary.runtimeKeyCount)`` (Windows ``$($report.summary.windowsRuntimeKeyCount)``, Linux ``$($report.summary.linuxRuntimeKeyCount)``)")
$lines.Add("- Windows asset pairs ready: ``$($report.summary.windowsAssetPairReadyCount)/$($report.summary.windowsToolkitVersionCount)``")
$lines.Add("- Linux asset versions ready: ``$($report.summary.linuxAssetReadyCount)/$($report.summary.linuxToolkitVersionCount)``")
$lines.Add("- Redistribution approved: ``$($report.summary.redistributionApproved)``")
$lines.Add("- Package-host size review approved: ``$($report.summary.packageHostSizeReviewApproved)``")
$lines.Add("- Role materialized: ``$($report.summary.roleMaterialized)``")
$lines.Add("- Can materialize full-runtime cuda-rtc role: ``$($report.summary.canMaterializeFullRuntimeCudaRtcRole)``")
$lines.Add('')
$lines.Add('| CUDA | Runtime keys | Asset pair | License text | Redistribution |')
$lines.Add('| --- | ---: | --- | --- | --- |')
foreach ($row in $windowsRows) {
  $lines.Add("| $($row.toolkitVersion) | $($row.runtimeKeys.Count) | $($row.assetIntegrityReady) | $($row.licenseTextPresent) | $($row.redistributionApprovalState) |")
}
$lines.Add('')
$lines.Add('## Blockers')
$lines.Add('')
foreach ($blocker in $blockers) {
  $lines.Add("- ``$blocker``")
}
$lines.Add('')
$lines.Add($report.proofBoundary)
Write-Utf8Text -Path $markdownPath -Value (($lines -join [Environment]::NewLine) + [Environment]::NewLine)

if ($structuralFindings.Count -gt 0) {
  throw "CUDA RTC packaging preflight found $($structuralFindings.Count) structural finding(s): $($structuralFindings -join ' | ')"
}
if ($RequireMaterializationReady.IsPresent -and -not $canMaterialize) {
  throw "CUDA RTC full-runtime role is not materialization-ready: $($blockers -join ', ')"
}

Write-Host "CUDA RTC full-runtime packaging preflight completed: $OutputPath"
