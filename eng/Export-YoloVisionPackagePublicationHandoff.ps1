[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$ManagedPackagePath,
  [Parameter(Mandatory = $true)][string]$YoloVisionPackagePath,
  [Parameter(Mandatory = $true)][string]$BridgePackagePath,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$PackageVersion = "4.0.0",
  [string]$ExpectedSourceCommit,
  [string]$OutputPath = "artifacts\yolovision\publication-handoff\yolovision-package-publication-handoff.json",
  [string]$MarkdownOutputPath = "artifacts\yolovision\publication-handoff\yolovision-package-publication-handoff.md",
  [string]$RepositoryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }
  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

foreach ($variableName in @("ManagedPackagePath", "YoloVisionPackagePath", "BridgePackagePath")) {
  $resolved = Resolve-RepositoryPath -Path (Get-Variable -Name $variableName -ValueOnly)
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    throw "$variableName does not exist: $resolved"
  }
  Set-Variable -Name $variableName -Value $resolved
}

if ([string]::IsNullOrWhiteSpace($ExpectedSourceCommit)) {
  $ExpectedSourceCommit = (& git -C $RepositoryRoot rev-parse HEAD).Trim()
}
if ($ExpectedSourceCommit -notmatch '^[a-fA-F0-9]{40}$') {
  throw "ExpectedSourceCommit must be a 40-character Git commit SHA."
}
$ExpectedSourceCommit = $ExpectedSourceCommit.ToLowerInvariant()

$OutputPath = Resolve-RepositoryPath -Path $OutputPath
$MarkdownOutputPath = Resolve-RepositoryPath -Path $MarkdownOutputPath
$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Read-Package {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Role
  )

  $file = Get-Item -LiteralPath $Path
  $archive = [IO.Compression.ZipFile]::OpenRead($file.FullName)
  try {
    $nuspecEntries = @($archive.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) })
    if ($nuspecEntries.Count -ne 1) {
      throw "Package '$Path' must contain exactly one nuspec."
    }
    $reader = [IO.StreamReader]::new($nuspecEntries[0].Open(), [Text.Encoding]::UTF8)
    try {
      [xml]$nuspec = $reader.ReadToEnd()
    }
    finally {
      $reader.Dispose()
    }

    $entryNames = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
    $nativeEntries = @($entryNames | Where-Object { $_ -match '^runtimes/[^/]+/native/[^/]+$' })
    $dependencies = @(
      foreach ($dependency in @($nuspec.SelectNodes("//*[local-name()='dependency']"))) {
        [pscustomobject][ordered]@{
          id = [string]$dependency.id
          version = [string]$dependency.version
        }
      }
    )

    return [pscustomobject][ordered]@{
      role = $Role
      packageId = [string]$nuspec.package.metadata.id
      packageVersion = [string]$nuspec.package.metadata.version
      repositoryUrl = [string]$nuspec.package.metadata.repository.url
      repositoryCommit = ([string]$nuspec.package.metadata.repository.commit).ToLowerInvariant()
      fileName = $file.Name
      path = $file.FullName
      length = $file.Length
      sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
      nativeEntries = $nativeEntries
      dependencies = $dependencies
    }
  }
  finally {
    $archive.Dispose()
  }
}

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeDefinitions = @($splitManifest.packages | Where-Object {
    [string]::Equals([string]$_.sourceRuntimeKey, $RuntimePackageKey, [StringComparison]::OrdinalIgnoreCase) -and
    [string]::Equals([string]$_.role, "bridge", [StringComparison]::OrdinalIgnoreCase)
  })
if ($bridgeDefinitions.Count -ne 1) {
  throw "RuntimePackageKey '$RuntimePackageKey' must resolve to exactly one bridge package."
}
$expectedBridgePackageId = [string]$bridgeDefinitions[0].packageId
$expectedBridgeNativeEntry = "runtimes/$([string]$bridgeDefinitions[0].rid)/native/$([string]$bridgeDefinitions[0].assets[0])"

$managed = Read-Package -Path $ManagedPackagePath -Role "managed-api"
$yoloVision = Read-Package -Path $YoloVisionPackagePath -Role "managed-extension"
$bridge = Read-Package -Path $BridgePackagePath -Role "bridge-only"
$packages = @($managed, $yoloVision, $bridge)
$expectedIds = @(
  "JYPPX.TensorRT.CSharp.API",
  "JYPPX.TensorRT.CSharp.API.YoloVision",
  $expectedBridgePackageId
)

for ($index = 0; $index -lt $packages.Count; $index++) {
  $package = $packages[$index]
  if (-not [string]::Equals([string]$package.packageId, $expectedIds[$index], [StringComparison]::Ordinal)) {
    throw "Package role '$($package.role)' has id '$($package.packageId)', expected '$($expectedIds[$index])'."
  }
  if (-not [string]::Equals([string]$package.packageVersion, $PackageVersion, [StringComparison]::Ordinal)) {
    throw "Package '$($package.packageId)' version '$($package.packageVersion)' does not match '$PackageVersion'."
  }
  if (-not [string]::Equals([string]$package.repositoryUrl, "https://github.com/guojin-yan/TensorRT-CSharp-API", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Package '$($package.packageId)' repository URL is invalid."
  }
  if (-not [string]::Equals([string]$package.repositoryCommit, $ExpectedSourceCommit, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Package '$($package.packageId)' repository commit '$($package.repositoryCommit)' does not match '$ExpectedSourceCommit'."
  }
  if ([string]$package.sha256 -notmatch '^[a-f0-9]{64}$') {
    throw "Package '$($package.packageId)' SHA256 is invalid."
  }
}

if (@($managed.nativeEntries).Count -ne 0 -or @($yoloVision.nativeEntries).Count -ne 0) {
  throw "Managed packages must not contain native entries."
}
if (@($bridge.nativeEntries).Count -ne 1 -or
    -not [string]::Equals([string]$bridge.nativeEntries[0], $expectedBridgeNativeEntry, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Bridge package must contain exactly '$expectedBridgeNativeEntry'."
}

$managedDependencies = @($yoloVision.dependencies | Where-Object {
    [string]::Equals([string]$_.id, [string]$managed.packageId, [StringComparison]::Ordinal)
  })
if ($managedDependencies.Count -ne 1 -or
    -not [string]::Equals([string]$managedDependencies[0].version, $PackageVersion, [StringComparison]::Ordinal)) {
  throw "YoloVision dependency must match the managed API package version."
}

& (Join-Path $RepositoryRoot "eng\Test-ExternalVendorRuntimePackagePolicy.ps1") `
  -RepositoryRoot $RepositoryRoot `
  -PackagePath @($managed.path, $yoloVision.path, $bridge.path) `
  -ExpectedPackageId $expectedIds `
  -ExpectedPackageVersion $PackageVersion `
  -RequireExactPackageSet | Out-Host

$record = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-package-publication-handoff"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  handoffState = "ready-local-three-package-handoff"
  runtimePackageKey = $RuntimePackageKey
  packageVersion = $PackageVersion
  sourceCommit = $ExpectedSourceCommit
  packageCount = $packages.Count
  expectedPackageIds = $expectedIds
  packageIdsExact = $true
  packageVersionsAligned = $true
  packageSourceCommitsAligned = $true
  packageHashesReady = $true
  yoloVisionManagedDependencyAligned = $true
  vendorRuntimeEntryCount = 0
  bridgeNativeEntryCount = 1
  packages = @($packages | ForEach-Object {
    [pscustomobject][ordered]@{
      role = $_.role
      packageId = $_.packageId
      packageVersion = $_.packageVersion
      fileName = $_.fileName
      length = $_.length
      sha256 = $_.sha256
      repositoryUrl = $_.repositoryUrl
      repositoryCommit = $_.repositoryCommit
      nativeEntries = @($_.nativeEntries)
      dependencies = @($_.dependencies)
    }
  })
  ownerHandoff = [pscustomobject][ordered]@{
    ownerReviewRequired = $true
    explicitPublishAuthorizationRequired = $true
    publicChannelSelectionRequired = $true
    publicUrlsRequiredAfterPublish = $true
    downloadedHashesRequiredAfterPublish = $true
    postPublishCleanConsumerRequired = $true
  }
  boundary = [pscustomobject][ordered]@{
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    isLocalPackageHandoff = $true
    isPublicPackageProof = $false
    isPostPublishProof = $false
    ownerReleaseAcceptance = $false
    canCloseReleaseIssue = $false
  }
}

New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($OutputPath)), ([IO.Path]::GetDirectoryName($MarkdownOutputPath)) -Force | Out-Null
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutputPath -Encoding utf8

$packageRows = @($record.packages | ForEach-Object {
  "| ``$($_.role)`` | ``$($_.packageId)`` | ``$($_.packageVersion)`` | ``$($_.repositoryCommit)`` | ``$($_.sha256)`` | ``$(@($_.nativeEntries).Count)`` |"
})
@(
  "# YoloVision Package Publication Handoff",
  "",
  "- state: ``$($record.handoffState)``",
  "- runtime package key: ``$RuntimePackageKey``",
  "- source commit: ``$ExpectedSourceCommit``",
  "- package version: ``$PackageVersion``",
  "- package IDs exact: ``True``",
  "- versions/source commits aligned: ``True / True``",
  "- vendor runtime entries: ``0``",
  "- performs publish: ``False``",
  "",
  "| Role | Package ID | Version | Source commit | SHA256 | Native entries |",
  "|---|---|---|---|---|---:|",
  $packageRows,
  "",
  "This local handoff pins the three candidate package files. Owner authorization, public URLs/download hashes, and a post-publish clean consumer remain required before any public or release claim."
) | Set-Content -LiteralPath $MarkdownOutputPath -Encoding utf8

Write-Host "HandoffState=$($record.handoffState) PackageCount=3 SourceCommitsAligned=True PackageHashesReady=True"
Write-Host "PerformsPublish=False OwnerReviewRequired=True PostPublishCleanConsumerRequired=True"
Write-Host "Report=$OutputPath"
