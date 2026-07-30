[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$SourceRuntimeKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [Parameter(Mandatory = $true)]
  [string]$ManagedReleaseTag,
  [Parameter(Mandatory = $true)]
  [string]$BridgeReleaseTag,
  [string]$ManagedPackageVersion,
  [string]$BridgePackageVersion,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
  [string]$GhCommand = "gh",
  [ValidateRange(1, 10)]
  [int]$DownloadMaxAttempts = 3,
  [ValidateRange(30, 3600)]
  [int]$DownloadTimeoutSeconds = 300,
  [switch]$AllowCrossCommitPair,
  [switch]$AllowRuntimeSmokeFailure,
  [switch]$KeepConsumerOutput,
  [string]$RepositoryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Net.Http

function Test-PathWithin {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Parent
  )

  $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
  return $resolvedPath.StartsWith(
    $resolvedParent + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase)
}

function Remove-SafeChildDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$AllowedRoot
  )

  if (-not (Test-PathWithin -Path $Path -Parent $AllowedRoot)) {
    throw "Refusing to remove path outside the public Release consumer root: $Path"
  }

  if (Test-Path -LiteralPath $Path) {
    try {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
    }
    catch {
      $isWindowsPlatform = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
      if (-not $isWindowsPlatform) {
        throw
      }

      $fullPath = [System.IO.Path]::GetFullPath($Path)
      $extendedPath = if ($fullPath.StartsWith("\\?\", [System.StringComparison]::Ordinal)) {
        $fullPath
      }
      else {
        "\\?\$fullPath"
      }
      [System.IO.Directory]::Delete($extendedPath, $true)
    }
  }
}

function Get-BridgeDefinition {
  param([Parameter(Mandatory = $true)][string]$RuntimeKey)

  $manifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
  $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $matches = @($manifest.packages | Where-Object {
      $_.sourceRuntimeKey -eq $RuntimeKey -and $_.role -eq "bridge"
    })
  if ($matches.Count -ne 1) {
    throw "Runtime key '$RuntimeKey' must resolve to exactly one bridge package. Found $($matches.Count)."
  }

  return $matches[0]
}

function Get-GitHubRelease {
  param([Parameter(Mandatory = $true)][string]$ReleaseTag)

  $output = @(& $GhCommand api "repos/$Repository/releases/tags/$ReleaseTag" 2>&1)
  if ($LASTEXITCODE -ne 0) {
    throw "Unable to query GitHub Release '$ReleaseTag' from '$Repository': $($output -join ' ')"
  }

  try {
    return (($output -join "`n") | ConvertFrom-Json)
  }
  catch {
    throw "GitHub Release '$ReleaseTag' returned invalid JSON: $($_.Exception.Message)"
  }
}

function Resolve-NuGetReleaseAsset {
  param(
    [Parameter(Mandatory = $true)][object]$Release,
    [Parameter(Mandatory = $true)][string]$PackageId,
    [string]$PackageVersion
  )

  $prefix = "$PackageId."
  $matches = @($Release.assets | Where-Object {
      $name = [string]$_.name
      $name.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase) -and
        $name.EndsWith(".nupkg", [System.StringComparison]::OrdinalIgnoreCase)
    })
  if (-not [string]::IsNullOrWhiteSpace($PackageVersion)) {
    $expectedName = "$PackageId.$PackageVersion.nupkg"
    $matches = @($matches | Where-Object {
        [string]::Equals([string]$_.name, $expectedName, [System.StringComparison]::OrdinalIgnoreCase)
      })
  }

  if ($matches.Count -ne 1) {
    $found = @($matches | ForEach-Object { [string]$_.name }) -join ", "
    throw "Release '$($Release.tag_name)' must contain exactly one '$PackageId' nupkg for the requested version. Found $($matches.Count): $found"
  }

  $asset = $matches[0]
  $downloadUrl = [string]$asset.browser_download_url
  $expectedPrefix = "https://github.com/$Repository/releases/download/"
  if (-not $downloadUrl.StartsWith($expectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Release asset URL is not an immutable download URL for '$Repository': $downloadUrl"
  }

  $digest = [string]$asset.digest
  if ($digest -notmatch '^sha256:[0-9a-fA-F]{64}$') {
    throw "Release asset '$($asset.name)' has no usable GitHub SHA256 digest. Actual='$digest'."
  }

  return $asset
}

function Save-PublicReleaseAsset {
  param(
    [Parameter(Mandatory = $true)][object]$Asset,
    [Parameter(Mandatory = $true)][string]$Directory
  )

  New-Item -ItemType Directory -Path $Directory -Force | Out-Null
  $targetPath = Join-Path $Directory ([string]$Asset.name)
  if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
    $existingFile = Get-Item -LiteralPath $targetPath
    $existingSha256 = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedSha256 = ([string]$Asset.digest).Substring("sha256:".Length).ToLowerInvariant()
    if ([long]$existingFile.Length -eq [long]$Asset.size -and $existingSha256 -eq $expectedSha256) {
      Write-Host "Reusing downloaded asset after remote digest verification: $targetPath"
      return [pscustomobject]@{
        path = $existingFile.FullName
        lengthBytes = [long]$existingFile.Length
        sha256 = $existingSha256
        downloadTransport = "verified-cache"
      }
    }

    Remove-Item -LiteralPath $targetPath -Force
  }

  for ($attempt = 1; $attempt -le $DownloadMaxAttempts; $attempt++) {
    Write-Host "Downloading public GitHub Release asset (attempt $attempt/$DownloadMaxAttempts): $($Asset.browser_download_url)"
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds($DownloadTimeoutSeconds)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd("TensorRtSharp4.0-public-release-consumer/1.0")
    try {
      $bytes = $client.GetByteArrayAsync([string]$Asset.browser_download_url).GetAwaiter().GetResult()
      [System.IO.File]::WriteAllBytes($targetPath, $bytes)
      break
    }
    catch {
      if ($attempt -ge $DownloadMaxAttempts) {
        throw "Failed to download public Release asset '$($Asset.name)' after $DownloadMaxAttempts attempt(s): $($_.Exception.Message)"
      }
      Write-Warning "Public Release asset download attempt $attempt failed: $($_.Exception.Message)"
      Start-Sleep -Seconds ([Math]::Min(5 * $attempt, 15))
    }
    finally {
      $client.Dispose()
    }
  }

  $file = Get-Item -LiteralPath $targetPath
  if ([long]$file.Length -ne [long]$Asset.size) {
    throw "Downloaded asset size mismatch for '$($Asset.name)'. Expected=$($Asset.size) Actual=$($file.Length)."
  }

  $actualSha256 = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $expectedSha256 = ([string]$Asset.digest).Substring("sha256:".Length).ToLowerInvariant()
  if ($actualSha256 -ne $expectedSha256) {
    throw "Downloaded asset SHA256 mismatch for '$($Asset.name)'. Expected=$expectedSha256 Actual=$actualSha256."
  }

  return [pscustomobject]@{
    path = $file.FullName
    lengthBytes = [long]$file.Length
    sha256 = $actualSha256
    downloadTransport = "public-https"
  }
}

function Get-NuGetPackageMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)

  $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspecEntry = @($archive.Entries | Where-Object {
        $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase)
      }) | Select-Object -First 1
    if ($null -eq $nuspecEntry) {
      throw "NuGet package has no nuspec: $Path"
    }

    $reader = [System.IO.StreamReader]::new($nuspecEntry.Open())
    try { [xml]$nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $idNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']/*[local-name()='id']")
    $versionNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']/*[local-name()='version']")
    if ($null -eq $idNode -or $null -eq $versionNode) {
      throw "NuGet package nuspec is missing id or version: $Path"
    }

    $nativeEntries = @($archive.Entries | Where-Object {
        $_.FullName -match '^runtimes/[^/]+/native/[^/]+$'
      } | ForEach-Object { $_.FullName })
    $repositoryNode = $nuspec.SelectSingleNode("//*[local-name()='metadata']/*[local-name()='repository']")
    $repositoryUrl = if ($null -eq $repositoryNode) { "" } else { [string]$repositoryNode.GetAttribute("url") }
    $repositoryCommit = if ($null -eq $repositoryNode) { "" } else { [string]$repositoryNode.GetAttribute("commit") }
    return [pscustomobject]@{
      id = [string]$idNode.InnerText
      version = [string]$versionNode.InnerText
      entryCount = @($archive.Entries).Count
      nativeEntries = @($nativeEntries)
      repositoryUrl = $repositoryUrl
      repositoryCommit = $repositoryCommit
    }
  }
  finally {
    $archive.Dispose()
  }
}

function Assert-PackageIdentity {
  param(
    [Parameter(Mandatory = $true)][object]$Metadata,
    [Parameter(Mandatory = $true)][string]$ExpectedId,
    [string]$ExpectedVersion
  )

  if (-not [string]::Equals([string]$Metadata.id, $ExpectedId, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Downloaded package id mismatch. Expected='$ExpectedId' Actual='$($Metadata.id)'."
  }
  if (-not [string]::IsNullOrWhiteSpace($ExpectedVersion) -and
      -not [string]::Equals([string]$Metadata.version, $ExpectedVersion, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Downloaded package version mismatch for '$ExpectedId'. Expected='$ExpectedVersion' Actual='$($Metadata.version)'."
  }
}

function Invoke-PackagePolicyGate {
  param([Parameter(Mandatory = $true)][string[]]$PackagePaths)

  $policyScript = Join-Path $RepositoryRoot "eng\Test-ExternalVendorRuntimePackagePolicy.ps1"
  $output = @(& $policyScript -PackagePath $PackagePaths -RepositoryRoot $RepositoryRoot 2>&1)

  return @($output)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
  throw "Repository must use owner/name form: $Repository"
}

$bridgeDefinition = Get-BridgeDefinition -RuntimeKey $SourceRuntimeKey
$managedPackageId = "JYPPX.TensorRT.CSharp.API"
$bridgePackageId = [string]$bridgeDefinition.packageId

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path ([System.IO.Path]::GetTempPath()) "jyppx-public-release-bridge-consumer"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-PathWithin -Path $OutputRoot -Parent $RepositoryRoot) {
  throw "Public Release consumer output must be outside the source repository: $OutputRoot"
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\public-release-consumer\$SourceRuntimeKey"
}
elseif (-not [System.IO.Path]::IsPathRooted($ReportDirectory)) {
  $ReportDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ReportDirectory))
}

$safeRuntimeKey = $SourceRuntimeKey -replace '[^A-Za-z0-9._-]', '-'
$executionRoot = Join-Path $OutputRoot $safeRuntimeKey
$managedAssetDirectory = Join-Path $executionRoot "assets\managed"
$bridgeAssetDirectory = Join-Path $executionRoot "assets\bridge"
$runtimeConsumerRoot = Join-Path ([System.IO.Path]::GetTempPath()) "jypr-$PID"
Remove-SafeChildDirectory -Path $runtimeConsumerRoot -AllowedRoot ([System.IO.Path]::GetTempPath())
$runtimeReportDirectory = Join-Path $ReportDirectory "runtime"
New-Item -ItemType Directory -Path $managedAssetDirectory, $bridgeAssetDirectory, $ReportDirectory -Force | Out-Null

$managedRelease = Get-GitHubRelease -ReleaseTag $ManagedReleaseTag
$bridgeRelease = if ($BridgeReleaseTag -eq $ManagedReleaseTag) { $managedRelease } else { Get-GitHubRelease -ReleaseTag $BridgeReleaseTag }
$managedAsset = Resolve-NuGetReleaseAsset -Release $managedRelease -PackageId $managedPackageId -PackageVersion $ManagedPackageVersion
$bridgeAsset = Resolve-NuGetReleaseAsset -Release $bridgeRelease -PackageId $bridgePackageId -PackageVersion $BridgePackageVersion

$managedDownload = Save-PublicReleaseAsset -Asset $managedAsset -Directory $managedAssetDirectory
$bridgeDownload = Save-PublicReleaseAsset -Asset $bridgeAsset -Directory $bridgeAssetDirectory
$managedMetadata = Get-NuGetPackageMetadata -Path $managedDownload.path
$bridgeMetadata = Get-NuGetPackageMetadata -Path $bridgeDownload.path
Assert-PackageIdentity -Metadata $managedMetadata -ExpectedId $managedPackageId -ExpectedVersion $ManagedPackageVersion
Assert-PackageIdentity -Metadata $bridgeMetadata -ExpectedId $bridgePackageId -ExpectedVersion $BridgePackageVersion

$expectedRepositoryUrl = "https://github.com/$Repository"
foreach ($packageMetadata in @($managedMetadata, $bridgeMetadata)) {
  if (-not [string]::Equals([string]$packageMetadata.repositoryUrl, $expectedRepositoryUrl, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Downloaded package repository URL mismatch. Expected='$expectedRepositoryUrl' Actual='$($packageMetadata.repositoryUrl)'."
  }
  if ([string]$packageMetadata.repositoryCommit -notmatch '^[0-9a-fA-F]{40}$') {
    throw "Downloaded package '$($packageMetadata.id)' has no usable 40-character repository commit."
  }
}
$packageSourceCommitAligned = [string]::Equals(
  [string]$managedMetadata.repositoryCommit,
  [string]$bridgeMetadata.repositoryCommit,
  [System.StringComparison]::OrdinalIgnoreCase)
if ($packageSourceCommitAligned -and $AllowCrossCommitPair.IsPresent) {
  throw "-AllowCrossCommitPair is valid only for an explicitly mismatched diagnostic pair; remove it for same-commit public asset evidence."
}
if (-not $packageSourceCommitAligned -and -not $AllowCrossCommitPair.IsPresent) {
  throw "Managed and bridge packages were built from different source commits. Managed=$($managedMetadata.repositoryCommit) Bridge=$($bridgeMetadata.repositoryCommit). Use a same-commit package pair; -AllowCrossCommitPair is diagnostic-only."
}

$policyOutput = @(Invoke-PackagePolicyGate -PackagePaths @($managedDownload.path, $bridgeDownload.path))

$shell = if ($PSVersionTable.PSEdition -eq "Core") { "pwsh" } else { "powershell" }
$runtimeArguments = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $RepositoryRoot "eng\Test-BridgePackageRuntimeConsumer.ps1"),
  "-SourceRuntimeKey", $SourceRuntimeKey,
  "-ManagedPackageDirectory", $managedAssetDirectory,
  "-BridgePackageDirectory", $bridgeAssetDirectory,
  "-OutputRoot", $runtimeConsumerRoot,
  "-ReportDirectory", $runtimeReportDirectory,
  "-RepositoryRoot", $RepositoryRoot,
  "-SkipBaselineValidation",
  "-AllowRuntimeSmokeFailure",
  "-KeepConsumerOutput"
)
if (-not [string]::IsNullOrWhiteSpace($TensorRtRoot)) { $runtimeArguments += @("-TensorRtRoot", $TensorRtRoot) }
if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) { $runtimeArguments += @("-CudaRoot", $CudaRoot) }
if (-not [string]::IsNullOrWhiteSpace($CudnnRoot)) { $runtimeArguments += @("-CudnnRoot", $CudnnRoot) }
if ($AllowCrossCommitPair.IsPresent) { $runtimeArguments += "-SkipInstalledVendorAssetHashing" }

Write-Host "Running repository-external consumer from verified public Release assets."
New-Item -ItemType Directory -Path $runtimeReportDirectory -Force | Out-Null
$runtimeInvocationStdoutPath = Join-Path $runtimeReportDirectory "runtime-consumer-invocation.stdout.log"
$runtimeInvocationStderrPath = Join-Path $runtimeReportDirectory "runtime-consumer-invocation.stderr.log"
Remove-Item -LiteralPath $runtimeInvocationStdoutPath, $runtimeInvocationStderrPath -Force -ErrorAction SilentlyContinue
& $shell @runtimeArguments 1> $runtimeInvocationStdoutPath 2> $runtimeInvocationStderrPath
$runtimeScriptExitCode = $LASTEXITCODE
$runtimeProofPath = Join-Path $runtimeReportDirectory "bridge-package-runtime-consumer-proof.json"
if (-not (Test-Path -LiteralPath $runtimeProofPath -PathType Leaf)) {
  throw "Runtime consumer did not write its structured report: $runtimeProofPath"
}
$runtimeProofSha256 = (Get-FileHash -LiteralPath $runtimeProofPath -Algorithm SHA256).Hash.ToLowerInvariant()
$runtimeProof = Get-Content -LiteralPath $runtimeProofPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeSmokePassed = [bool]$runtimeProof.isRuntimeExecutionProof -and [int]$runtimeProof.exitCode -eq 0
$runtimeDiagnosticMode = [bool]$runtimeProof.installedVendorAssetHashingSkipped -or
  [bool]$runtimeProof.installedVendorAssetInventorySkipped -or
  -not [bool]$runtimeProof.nativeAssetHashesComplete
if ($AllowCrossCommitPair.IsPresent -ne $runtimeDiagnosticMode) {
  throw "Runtime diagnostic mode did not match the cross-commit override. AllowCrossCommitPair=$($AllowCrossCommitPair.IsPresent) RuntimeDiagnosticMode=$runtimeDiagnosticMode."
}
$publicReleaseAssetConsumerEvidence = $runtimeSmokePassed -and
  $packageSourceCommitAligned -and
  -not $runtimeDiagnosticMode -and
  [bool]$runtimeProof.canPromoteCompatibleHostRuntimeProof

$result = [ordered]@{
  schemaVersion = 1
  recordKind = "public-release-bridge-package-consumer"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  repository = $Repository
  sourceRuntimeKey = $SourceRuntimeKey
  publicationPolicy = "bridge-only"
  channel = "github-release-assets"
  proofClassification = if ($publicReleaseAssetConsumerEvidence) { "public-release-assets-compatible-host-runtime" } elseif (-not $packageSourceCommitAligned) { "cross-commit-public-assets-diagnostic-only" } else { "public-release-assets-runtime-failed" }
  publicReleaseAssetProvenanceVerified = $true
  remoteDigestVerified = $true
  packageIdentityVerified = $true
  packageSourceCommitAligned = $packageSourceCommitAligned
  crossCommitDiagnosticOverride = $AllowCrossCommitPair.IsPresent
  externalVendorRuntimePackagePolicyPassed = $true
  restoreUsesDownloadedAssetStaging = $true
  stagingIsLocallyBuiltPackageFeed = $false
  directNupkgReferenceUsed = $false
  packageReferenceOnly = $true
  vendorRuntimeBundled = $false
  systemInstalledVendorDependenciesRequired = $true
  consumerRootOutsideRepository = -not (Test-PathWithin -Path $runtimeConsumerRoot -Parent $RepositoryRoot)
  runtimeSmokePassed = $runtimeSmokePassed
  runtimeScriptExitCode = $runtimeScriptExitCode
  isRuntimeExecutionProof = $runtimeSmokePassed
  isPublicReleaseAssetConsumerEvidence = $publicReleaseAssetConsumerEvidence
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  currentHeadBindingVerified = $false
  canPromoteCurrentHeadPackageConsumerProof = $false
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  releaseAssets = [ordered]@{
    managed = [ordered]@{
      releaseTag = [string]$managedRelease.tag_name
      releaseUrl = [string]$managedRelease.html_url
      assetId = [long]$managedAsset.id
      assetName = [string]$managedAsset.name
      assetUrl = [string]$managedAsset.browser_download_url
      githubDigest = [string]$managedAsset.digest
      downloadedPath = [string]$managedDownload.path
      downloadedSha256 = [string]$managedDownload.sha256
      downloadTransport = [string]$managedDownload.downloadTransport
      lengthBytes = [long]$managedDownload.lengthBytes
      packageId = [string]$managedMetadata.id
      packageVersion = [string]$managedMetadata.version
      repositoryUrl = [string]$managedMetadata.repositoryUrl
      repositoryCommit = [string]$managedMetadata.repositoryCommit
      nativeEntries = @($managedMetadata.nativeEntries)
    }
    bridge = [ordered]@{
      releaseTag = [string]$bridgeRelease.tag_name
      releaseUrl = [string]$bridgeRelease.html_url
      assetId = [long]$bridgeAsset.id
      assetName = [string]$bridgeAsset.name
      assetUrl = [string]$bridgeAsset.browser_download_url
      githubDigest = [string]$bridgeAsset.digest
      downloadedPath = [string]$bridgeDownload.path
      downloadedSha256 = [string]$bridgeDownload.sha256
      downloadTransport = [string]$bridgeDownload.downloadTransport
      lengthBytes = [long]$bridgeDownload.lengthBytes
      packageId = [string]$bridgeMetadata.id
      packageVersion = [string]$bridgeMetadata.version
      repositoryUrl = [string]$bridgeMetadata.repositoryUrl
      repositoryCommit = [string]$bridgeMetadata.repositoryCommit
      nativeEntries = @($bridgeMetadata.nativeEntries)
    }
  }
  consumer = [ordered]@{
    outputRoot = $runtimeConsumerRoot
    runtimeReportPath = $runtimeProofPath
    runtimeReportSha256 = $runtimeProofSha256
    projectPath = [string]$runtimeProof.consumer.projectPath
    projectSha256 = [string]$runtimeProof.consumer.projectSha256
    smokeExitCode = [int]$runtimeProof.exitCode
    smokeStatus = [string]$runtimeProof.smokeStatus
    identityOutputMatch = [bool]$runtimeProof.identityOutputMatch
    enqueueCompleted = [bool]$runtimeProof.enqueueCompleted
    invocationStdoutPath = $runtimeInvocationStdoutPath
    invocationStdoutSha256 = (Get-FileHash -LiteralPath $runtimeInvocationStdoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
    invocationStderrPath = $runtimeInvocationStderrPath
    invocationStderrSha256 = (Get-FileHash -LiteralPath $runtimeInvocationStderrPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  policyValidationOutput = @($policyOutput | ForEach-Object { [string]$_ })
  boundary = if ($publicReleaseAssetConsumerEvidence) {
    "This record proves the named public GitHub Release managed and bridge assets were downloaded from immutable URLs, matched GitHub SHA256 digests and package identities, passed the bridge-only package policy, and completed compatible-host runtime consumption outside the repository with machine-installed NVIDIA dependencies. It does not prove current HEAD publication, Owner authorization, another runtime line, package-consumer-runtime promotion, or post-publish release closure."
  }
  elseif (-not $packageSourceCommitAligned) {
    "This cross-commit record is diagnostic-only. It verifies each named public asset independently and may record runtime behavior, but the managed and bridge packages are not a coherent source pair and cannot be promoted as public Release asset consumer evidence, package-consumer-runtime proof, or post-publish proof."
  }
  else {
    "This record verifies the named public GitHub Release asset provenance and bridge-only policy, but the compatible-host runtime smoke did not pass. It is not runtime proof, package-consumer-runtime proof, or post-publish proof."
  }
}

$jsonPath = Join-Path $ReportDirectory "public-release-bridge-package-consumer.json"
$markdownPath = Join-Path $ReportDirectory "public-release-bridge-package-consumer.md"
[pscustomobject]$result | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Public Release Bridge Package Consumer

| Runtime key | Channel | Managed | Bridge | Runtime smoke | Identity output |
| --- | --- | --- | --- | --- | --- |
| $(ConvertTo-MarkdownCell $SourceRuntimeKey) | GitHub Release assets | ``$(ConvertTo-MarkdownCell $managedMetadata.version)`` | ``$(ConvertTo-MarkdownCell $bridgeMetadata.version)`` | ``$runtimeSmokePassed`` | ``$($runtimeProof.identityOutputMatch)`` |

- Managed asset: ``$($managedAsset.browser_download_url)``
- Managed SHA256: ``$($managedDownload.sha256)``
- Bridge asset: ``$($bridgeAsset.browser_download_url)``
- Bridge SHA256: ``$($bridgeDownload.sha256)``
- Package source commit aligned: ``$packageSourceCommitAligned``
- Restore staging contains downloaded public assets, not locally built packages.
- NVIDIA TensorRT, CUDA, cuDNN, and optional NVRTC libraries remain machine-installed prerequisites.
- Current HEAD binding: ``False``
- Package-consumer runtime proof promotion: ``False``
- Post-publish proof: ``False``

## Boundary

$($result.boundary)
"@
Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

$validatorArguments = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $RepositoryRoot "eng\Test-PublicReleaseBridgePackageConsumer.ps1"),
  "-InputPath", $jsonPath,
  "-OutputDirectory", $ReportDirectory,
  "-ExpectedSourceRuntimeKey", $SourceRuntimeKey,
  "-RepositoryRoot", $RepositoryRoot,
  "-RequireReferencedFiles",
  "-Strict"
)
if ($publicReleaseAssetConsumerEvidence) { $validatorArguments += "-FailOnNotEvidence" }
& $shell @validatorArguments
if ($LASTEXITCODE -ne 0) {
  throw "Public Release bridge package consumer validation failed with exit code $LASTEXITCODE."
}

Write-Host "Public Release bridge package consumer report written: $jsonPath"
Write-Host "Public Release bridge package consumer report written: $markdownPath"

if (-not $KeepConsumerOutput.IsPresent) {
  Remove-SafeChildDirectory -Path $executionRoot -AllowedRoot $OutputRoot
  Remove-SafeChildDirectory -Path $runtimeConsumerRoot -AllowedRoot ([System.IO.Path]::GetTempPath())
}

if (-not $runtimeSmokePassed -and -not $AllowRuntimeSmokeFailure.IsPresent) {
  throw "Public Release bridge package runtime smoke failed. See $runtimeProofPath"
}
