[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$PackageDirectory,
  [string]$PackageVersion = "4.0.0",
  [string]$ExpectedSourceCommit,
  [string]$Configuration = "Release",
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [switch]$KeepWorkspace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)

if ([string]::IsNullOrWhiteSpace($PackageDirectory)) {
  $PackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
elseif (-not [IO.Path]::IsPathRooted($PackageDirectory)) {
  $PackageDirectory = Join-Path $RepositoryRoot $PackageDirectory
}
$PackageDirectory = [IO.Path]::GetFullPath($PackageDirectory)

if ([string]::IsNullOrWhiteSpace($ExpectedSourceCommit)) {
  $ExpectedSourceCommit = (& git -C $RepositoryRoot rev-parse HEAD).Trim()
}
if ($ExpectedSourceCommit -notmatch '^[a-fA-F0-9]{40}$') {
  throw "ExpectedSourceCommit must be a 40-character Git commit SHA."
}
$ExpectedSourceCommit = $ExpectedSourceCommit.ToLowerInvariant()

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $baseOutputRoot = if (-not [string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    $env:RUNNER_TEMP
  }
  else {
    Join-Path ([IO.Directory]::GetParent($RepositoryRoot).FullName) "consumer-workspaces"
  }
  $OutputRoot = Join-Path $baseOutputRoot "yolovision-managed-package-dry-run"
}
elseif (-not [IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$repositoryPrefix = $RepositoryRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if ($OutputRoot.Equals($RepositoryRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $OutputRoot.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "OutputRoot must be outside the repository for a clean managed package consumer."
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\yolovision\managed-package-dry-run"
}
elseif (-not [IO.Path]::IsPathRooted($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot $ReportDirectory
}
$ReportDirectory = [IO.Path]::GetFullPath($ReportDirectory)

$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function ConvertTo-XmlAttributeValue {
  param([string]$Value)
  return [Security.SecurityElement]::Escape($Value)
}

function Get-NupkgMetadata {
  param([Parameter(Mandatory = $true)][IO.FileInfo]$File)

  $archive = [IO.Compression.ZipFile]::OpenRead($File.FullName)
  try {
    $nuspecEntries = @($archive.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) })
    if ($nuspecEntries.Count -ne 1) {
      throw "Package must contain exactly one nuspec: $($File.FullName)"
    }

    $reader = [IO.StreamReader]::new($nuspecEntries[0].Open(), [Text.Encoding]::UTF8)
    try {
      [xml]$nuspec = $reader.ReadToEnd()
    }
    finally {
      $reader.Dispose()
    }

    $dependencyNodes = @($nuspec.SelectNodes("//*[local-name()='dependency']"))
    $dependencies = @(
      foreach ($dependency in $dependencyNodes) {
        [pscustomobject][ordered]@{
          id = [string]$dependency.id
          version = [string]$dependency.version
        }
      }
    )
    $entryNames = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
    $nativeEntries = @($entryNames | Where-Object { $_ -match '^runtimes/[^/]+/native/[^/]+$' })

    return [pscustomobject][ordered]@{
      id = [string]$nuspec.package.metadata.id
      version = [string]$nuspec.package.metadata.version
      repositoryUrl = [string]$nuspec.package.metadata.repository.url
      repositoryCommit = ([string]$nuspec.package.metadata.repository.commit).ToLowerInvariant()
      dependencies = $dependencies
      entryNames = $entryNames
      nativeEntries = $nativeEntries
      path = $File.FullName
      fileName = $File.Name
      length = $File.Length
      sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
  }
  finally {
    $archive.Dispose()
  }
}

function Find-RequiredPackage {
  param(
    [Parameter(Mandatory = $true)][object[]]$Packages,
    [Parameter(Mandatory = $true)][string]$PackageId
  )

  $matches = @($Packages | Where-Object {
      [string]::Equals([string]$_.id, $PackageId, [StringComparison]::Ordinal) -and
      [string]::Equals([string]$_.version, $PackageVersion, [StringComparison]::Ordinal)
    })
  if ($matches.Count -ne 1) {
    throw "Expected exactly one '$PackageId' package at version '$PackageVersion', found $($matches.Count)."
  }
  return $matches[0]
}

function Invoke-DotNetStep {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$LogPath
  )

  $previousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $output = @(& dotnet @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  $output | ForEach-Object { [string]$_ } | Set-Content -LiteralPath $LogPath -Encoding utf8
  if ($exitCode -ne 0) {
    throw "$Name failed with exit code $exitCode. See $LogPath"
  }
  return [pscustomobject]@{ exitCode = $exitCode; output = @($output | ForEach-Object { [string]$_ }) }
}

if (-not (Test-Path -LiteralPath $PackageDirectory -PathType Container)) {
  throw "PackageDirectory does not exist: $PackageDirectory"
}
New-Item -ItemType Directory -Path $OutputRoot, $ReportDirectory -Force | Out-Null

$packageFiles = @(Get-ChildItem -LiteralPath $PackageDirectory -Filter *.nupkg -File)
$packages = @($packageFiles | ForEach-Object { Get-NupkgMetadata -File $_ })
$managed = Find-RequiredPackage -Packages $packages -PackageId "JYPPX.TensorRT.CSharp.API"
$yoloVision = Find-RequiredPackage -Packages $packages -PackageId "JYPPX.TensorRT.CSharp.API.YoloVision"
$classification = Find-RequiredPackage -Packages $packages -PackageId "JYPPX.TensorRT.CSharp.API.Classification"
$selectedPackages = @($managed, $yoloVision, $classification)

foreach ($package in $selectedPackages) {
  if (-not [string]::Equals([string]$package.repositoryUrl, "https://github.com/guojin-yan/TensorRT-CSharp-API", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Package '$($package.id)' repository URL is not the formal source repository."
  }
  if (-not [string]::Equals([string]$package.repositoryCommit, $ExpectedSourceCommit, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Package '$($package.id)' repository commit '$($package.repositoryCommit)' does not match '$ExpectedSourceCommit'."
  }
  if (@($package.nativeEntries).Count -ne 0) {
    throw "Managed package '$($package.id)' contains native runtime entries."
  }
}

$yoloManagedDependencies = @($yoloVision.dependencies | Where-Object {
    [string]::Equals([string]$_.id, "JYPPX.TensorRT.CSharp.API", [StringComparison]::Ordinal)
  })
if ($yoloManagedDependencies.Count -ne 1 -or
    -not [string]::Equals([string]$yoloManagedDependencies[0].version, $PackageVersion, [StringComparison]::Ordinal)) {
  throw "YoloVision must declare exactly one managed API dependency at version '$PackageVersion'."
}
$classificationManagedDependencies = @($classification.dependencies | Where-Object {
    [string]::Equals([string]$_.id, "JYPPX.TensorRT.CSharp.API", [StringComparison]::Ordinal)
  })
if ($classificationManagedDependencies.Count -ne 1 -or
    -not [string]::Equals([string]$classificationManagedDependencies[0].version, $PackageVersion, [StringComparison]::Ordinal)) {
  throw "Classification must declare exactly one managed API dependency at version '$PackageVersion'."
}
if (@($classification.entryNames | Where-Object { $_ -eq "lib/net8.0/Classification.dll" }).Count -ne 1) {
  throw "Classification package must contain lib/net8.0/Classification.dll."
}

& (Join-Path $RepositoryRoot "eng\Test-ExternalVendorRuntimePackagePolicy.ps1") `
  -RepositoryRoot $RepositoryRoot `
  -PackagePath $PackageDirectory `
  -ExpectedPackageId @($selectedPackages | ForEach-Object { $_.id }) `
  -ExpectedPackageVersion $PackageVersion `
  -RequireExactPackageSet | Out-Host

$workspace = Join-Path $OutputRoot ("workspace-" + [Guid]::NewGuid().ToString("N"))
$packageCache = Join-Path $workspace "packages"
$dotnetHome = Join-Path $workspace "dotnet-home"
$projectPath = Join-Path $workspace "YoloVision.ManagedPackageConsumer.csproj"
$programPath = Join-Path $workspace "Program.cs"
$nugetConfigPath = Join-Path $workspace "NuGet.Config"
$restoreLogPath = Join-Path $ReportDirectory "restore.log"
$buildLogPath = Join-Path $ReportDirectory "build.log"
$runLogPath = Join-Path $ReportDirectory "run.log"

$oldNuGetPackages = $env:NUGET_PACKAGES
$oldDotnetHome = $env:DOTNET_CLI_HOME
$oldTelemetry = $env:DOTNET_CLI_TELEMETRY_OPTOUT
try {
  New-Item -ItemType Directory -Path $workspace, $packageCache, $dotnetHome -Force | Out-Null
  $templateRoot = Join-Path $RepositoryRoot "samples\YoloVision.ManagedPackageConsumer"
  Copy-Item -LiteralPath (Join-Path $templateRoot "Program.cs") -Destination $programPath
  $project = Get-Content -LiteralPath (Join-Path $templateRoot "YoloVision.ManagedPackageConsumer.csproj.template") -Raw -Encoding utf8
  $project = $project.Replace("__MANAGED_PACKAGE_VERSION__", $PackageVersion).Replace("__YOLOVISION_PACKAGE_VERSION__", $PackageVersion)
  [IO.File]::WriteAllText($projectPath, $project, $utf8)

  $nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="jyppx-managed-dry-run" value="$(ConvertTo-XmlAttributeValue -Value $PackageDirectory)" />
  </packageSources>
</configuration>
"@
  [IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)

  $env:NUGET_PACKAGES = $packageCache
  $env:DOTNET_CLI_HOME = $dotnetHome
  $env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"

  $restore = Invoke-DotNetStep -Name "restore" -Arguments @(
    "restore", $projectPath,
    "--configfile", $nugetConfigPath,
    "--packages", $packageCache,
    "--force-evaluate"
  ) -LogPath $restoreLogPath
  $build = Invoke-DotNetStep -Name "build" -Arguments @(
    "build", $projectPath,
    "-c", $Configuration,
    "--no-restore"
  ) -LogPath $buildLogPath
  $run = Invoke-DotNetStep -Name "run" -Arguments @(
    "run", "--project", $projectPath,
    "-c", $Configuration,
    "--no-build"
  ) -LogPath $runLogPath

  $assetsPath = Join-Path $workspace "obj\project.assets.json"
  if (-not (Test-Path -LiteralPath $assetsPath -PathType Leaf)) {
    throw "Consumer restore graph is missing: $assetsPath"
  }
  $assets = Get-Content -LiteralPath $assetsPath -Raw -Encoding utf8 | ConvertFrom-Json
  $libraries = @($assets.libraries.PSObject.Properties | ForEach-Object { $_.Value })
  $projectLibraryCount = @($libraries | Where-Object { [string]$_.type -eq "project" }).Count
  $packageLibraryCount = @($libraries | Where-Object { [string]$_.type -eq "package" }).Count
  if ($projectLibraryCount -ne 0) {
    throw "Consumer restore graph contains project libraries."
  }
  if ($packageLibraryCount -ne 2) {
    throw "Consumer restore graph must contain exactly two package libraries, found $packageLibraryCount."
  }
  if ($project.IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $project.IndexOf("<Reference", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
    throw "Consumer project must use PackageReference only."
  }

  $runText = $run.output -join [Environment]::NewLine
  if ($runText.IndexOf("YoloVisionManagedPackageConsumer Passed=True PackageReferenceOnly=True NativeRuntimeLoaded=False", [StringComparison]::Ordinal) -lt 0) {
    throw "Consumer did not emit the required managed package success marker."
  }

  $report = [pscustomobject][ordered]@{
    schemaVersion = 1
    recordKind = "yolovision-managed-package-dry-run"
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    validationState = "passed-managed-package-pack-content-surface-consumer"
    sourceCommit = $ExpectedSourceCommit
    packageVersion = $PackageVersion
    packages = @($selectedPackages | ForEach-Object {
      [pscustomobject][ordered]@{
        id = $_.id
        version = $_.version
        fileName = $_.fileName
        length = $_.length
        sha256 = $_.sha256
        repositoryUrl = $_.repositoryUrl
        repositoryCommit = $_.repositoryCommit
        nativeEntryCount = @($_.nativeEntries).Count
        dependencies = @($_.dependencies)
      }
    })
    packageSet = [pscustomobject][ordered]@{
      expectedPackageCount = 3
      selectedPackageCount = $selectedPackages.Count
      packageIdsExact = $true
      packageVersionsAligned = $true
      packageSourceCommitsAligned = $true
      yoloVisionManagedDependencyAligned = $true
      classificationManagedDependencyAligned = $true
      vendorRuntimeEntryCount = 0
      nativeEntryCount = 0
    }
    consumer = [pscustomobject][ordered]@{
      template = "samples/YoloVision.ManagedPackageConsumer"
      workspaceOutsideRepository = -not $workspace.StartsWith($RepositoryRoot, [StringComparison]::OrdinalIgnoreCase)
      packageReferenceCount = 2
      projectReferenceCount = 0
      directAssemblyReferenceCount = 0
      restoredProjectLibraryCount = $projectLibraryCount
      restoredPackageLibraryCount = $packageLibraryCount
      remoteSourcesCleared = $true
      isolatedPackageCache = $true
      restoreExitCode = $restore.exitCode
      buildExitCode = $build.exitCode
      runExitCode = $run.exitCode
      passedMarker = "YoloVisionManagedPackageConsumer Passed=True PackageReferenceOnly=True NativeRuntimeLoaded=False"
      nativeRuntimeLoaded = $false
    }
    logs = [pscustomobject][ordered]@{
      restorePath = $restoreLogPath
      restoreSha256 = (Get-FileHash -LiteralPath $restoreLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
      buildPath = $buildLogPath
      buildSha256 = (Get-FileHash -LiteralPath $buildLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
      runPath = $runLogPath
      runSha256 = (Get-FileHash -LiteralPath $runLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    boundary = [pscustomobject][ordered]@{
      performsPublish = $false
      usesPublishToken = $false
      isManagedPackageDryRun = $true
      isTensorRtRuntimeProof = $false
      isPublicPackageProof = $false
      isPostPublishProof = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
    }
  }

  $jsonPath = Join-Path $ReportDirectory "yolovision-managed-package-dry-run.json"
  $markdownPath = Join-Path $ReportDirectory "yolovision-managed-package-dry-run.md"
  $report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
  @(
    "# YoloVision Managed Package Dry Run",
    "",
    "- state: ``$($report.validationState)``",
    "- source commit: ``$ExpectedSourceCommit``",
    "- package version: ``$PackageVersion``",
    "- selected package count: ``$($selectedPackages.Count)``",
    "- PackageReference count: ``2``",
    "- ProjectReference/direct assembly reference: ``0 / 0``",
    "- restored project libraries: ``$projectLibraryCount``",
    "- restored package libraries: ``$packageLibraryCount``",
    "- vendor/native package entries: ``0 / 0``",
    "- native runtime loaded: ``False``",
    "- performs publish: ``False``",
    "",
    "This record proves a local managed-package pack/content/dependency/clean-consumer dry run. It is not TensorRT runtime, public-feed, post-publish, redistribution, Owner acceptance, or release proof."
  ) | Set-Content -LiteralPath $markdownPath -Encoding utf8

  Write-Host "ValidationState=$($report.validationState) PackageCount=3 ProjectReferenceCount=0 RestoredProjectLibraryCount=$projectLibraryCount"
  Write-Host "SourceCommit=$ExpectedSourceCommit PackageSourceCommitsAligned=True PerformsPublish=False NativeRuntimeLoaded=False"
  Write-Host "Report=$jsonPath"
}
finally {
  $env:NUGET_PACKAGES = $oldNuGetPackages
  $env:DOTNET_CLI_HOME = $oldDotnetHome
  $env:DOTNET_CLI_TELEMETRY_OPTOUT = $oldTelemetry
  if (-not $KeepWorkspace.IsPresent -and (Test-Path -LiteralPath $workspace)) {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
      Remove-Item -LiteralPath $workspace -Recurse -Force -ErrorAction SilentlyContinue
      if (-not (Test-Path -LiteralPath $workspace)) {
        break
      }
      Start-Sleep -Milliseconds (200 * $attempt)
    }
    if (Test-Path -LiteralPath $workspace) {
      throw "Managed package consumer workspace could not be removed: $workspace"
    }
  }
}
