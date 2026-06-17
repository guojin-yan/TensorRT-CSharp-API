param(
  [Parameter(Mandatory = $true)]
  [string]$SourceRuntimeKey,
  [string[]]$SplitPackageRole = @("all"),
  [string]$Version,
  [string]$BridgePackageVersion,
  [string]$CudaCudnnPackageVersion,
  [string]$CudaCudnnPackageVersionMap,
  [string]$CudaCudnnPackageReleaseTag,
  [string]$CudaCudnnPackageReleaseTagMap,
  [string]$TensorRtPackageVersion,
  [string]$TensorRtPackageVersionMap,
  [string]$TensorRtPackageReleaseTag,
  [string]$TensorRtPackageReleaseTagMap,
  [string]$Repository,
  [string]$OutputRoot,
  [string]$OutputPathFile,
  [int]$DownloadMaxAttempts = 3,
  [int]$DownloadTimeoutSeconds = 1800,
  [int]$DownloadStallSeconds = 180,
  [int]$DownloadProgressPollSeconds = 10,
  [switch]$SkipGlobalPackageCache,
  [string]$RepositoryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Expand-KeyList {
  param(
    [AllowNull()]
    [string[]]$Values
  )

  $items = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in $value -split '[,;]') {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $items.Add($trimmed)
      }
    }
  }

  return @($items)
}

function Resolve-RolePackageVersion {
  param(
    [string]$Value,
    [string]$Fallback
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $Fallback
  }

  return $Value
}

function Resolve-ReleaseTag {
  param(
    [string]$ExplicitTag,
    [string]$ExplicitPackageVersion,
    [string]$PackageVersion
  )

  if (-not [string]::IsNullOrWhiteSpace($ExplicitTag)) {
    return $ExplicitTag
  }

  if (-not [string]::IsNullOrWhiteSpace($ExplicitPackageVersion)) {
    return "v$PackageVersion"
  }

  return ""
}

function Resolve-SplitPackagePins {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeKey
  )

  $powerShellCommand = if ($PSVersionTable.PSEdition -eq "Core") { "pwsh" } else { "powershell" }
  $arguments = @(
    "-NoProfile",
    "-File",
    (Join-Path $RepositoryRoot "eng\Resolve-SplitPackagePins.ps1"),
    "-SourceRuntimeKey",
    $RuntimeKey,
    "-Version",
    $Version
  )

  if (-not [string]::IsNullOrWhiteSpace($BridgePackageVersion)) {
    $arguments += @("-BridgePackageVersion", $BridgePackageVersion)
  }
  if (-not [string]::IsNullOrWhiteSpace($CudaCudnnPackageVersion)) {
    $arguments += @("-CudaCudnnPackageVersion", $CudaCudnnPackageVersion)
  }
  if (-not [string]::IsNullOrWhiteSpace($CudaCudnnPackageVersionMap)) {
    $arguments += @("-CudaCudnnPackageVersionMap", $CudaCudnnPackageVersionMap)
  }
  if (-not [string]::IsNullOrWhiteSpace($CudaCudnnPackageReleaseTag)) {
    $arguments += @("-CudaCudnnPackageReleaseTag", $CudaCudnnPackageReleaseTag)
  }
  if (-not [string]::IsNullOrWhiteSpace($CudaCudnnPackageReleaseTagMap)) {
    $arguments += @("-CudaCudnnPackageReleaseTagMap", $CudaCudnnPackageReleaseTagMap)
  }
  if (-not [string]::IsNullOrWhiteSpace($TensorRtPackageVersion)) {
    $arguments += @("-TensorRtPackageVersion", $TensorRtPackageVersion)
  }
  if (-not [string]::IsNullOrWhiteSpace($TensorRtPackageVersionMap)) {
    $arguments += @("-TensorRtPackageVersionMap", $TensorRtPackageVersionMap)
  }
  if (-not [string]::IsNullOrWhiteSpace($TensorRtPackageReleaseTag)) {
    $arguments += @("-TensorRtPackageReleaseTag", $TensorRtPackageReleaseTag)
  }
  if (-not [string]::IsNullOrWhiteSpace($TensorRtPackageReleaseTagMap)) {
    $arguments += @("-TensorRtPackageReleaseTagMap", $TensorRtPackageReleaseTagMap)
  }

  & $powerShellCommand @arguments | ConvertFrom-Json
}

function Test-SplitPackageRequested {
  param(
    [Parameter(Mandatory = $true)]
    [psobject]$SplitPackage,
    [Parameter(Mandatory = $true)]
    [string[]]$RequestedRoles
  )

  if ($RequestedRoles -contains "all") {
    return $true
  }

  $role = ([string]$SplitPackage.role).ToLowerInvariant()
  $key = ([string]$SplitPackage.key).ToLowerInvariant()
  return (
    ($RequestedRoles -contains $role) -or
    ($RequestedRoles -contains $key) -or
    (($RequestedRoles -contains "stable-dependencies" -or $RequestedRoles -contains "nvidia-dependencies") -and
      ($role -eq "cuda-cudnn" -or $role -eq "tensorrt"))
  )
}

function Get-SplitPackageVersion {
  param(
    [Parameter(Mandatory = $true)]
    [psobject]$SplitPackage,
    [Parameter(Mandatory = $true)]
    [string]$BridgeVersion,
    [Parameter(Mandatory = $true)]
    [string]$CudaCudnnVersion,
    [Parameter(Mandatory = $true)]
    [string]$TensorRtVersion
  )

  switch ([string]$SplitPackage.role) {
    "bridge" { return $BridgeVersion }
    "cuda-cudnn" { return $CudaCudnnVersion }
    "tensorrt" { return $TensorRtVersion }
    default { return $Version }
  }
}

function Get-SplitPackageReleaseTag {
  param(
    [Parameter(Mandatory = $true)]
    [psobject]$SplitPackage,
    [string]$CudaCudnnReleaseTag,
    [string]$TensorRtReleaseTag
  )

  switch ([string]$SplitPackage.role) {
    "cuda-cudnn" { return $CudaCudnnReleaseTag }
    "tensorrt" { return $TensorRtReleaseTag }
    default { return "" }
  }
}

function Get-ExpectedNativeFileNames {
  param(
    [object[]]$RelativeFiles
  )

  $fileNames = New-Object System.Collections.Generic.List[string]
  foreach ($relativePath in @($RelativeFiles)) {
    if ([string]::IsNullOrWhiteSpace([string]$relativePath)) {
      continue
    }

    $fileNames.Add([System.IO.Path]::GetFileName([string]$relativePath))
  }

  return @($fileNames | Sort-Object -Unique)
}

function New-LinuxDynamicSplitPackage {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SourcePackage,
    [Parameter(Mandatory = $true)]
    [string]$Role,
    [Parameter(Mandatory = $true)]
    [string[]]$Assets,
    [string]$Suffix,
    [string]$PackageSuffix
  )

  $effectiveSuffix = if ([string]::IsNullOrWhiteSpace($Suffix)) { $Role } else { $Suffix }
  $effectivePackageSuffix = if (-not [string]::IsNullOrWhiteSpace($PackageSuffix)) {
    $PackageSuffix
  }
  else {
    switch ($Role) {
      "bridge" { "Bridge" }
      "cuda-cudnn" { "CudaCudnn" }
      "tensorrt" { "TensorRt" }
      default { $effectiveSuffix }
    }
  }

  return [pscustomobject]@{
    key = "$($SourcePackage.key)-$effectiveSuffix"
    sourceRuntimeKey = $SourcePackage.key
    packageId = "$($SourcePackage.packageId).$effectivePackageSuffix"
    rid = $SourcePackage.rid
    platform = $SourcePackage.platform
    tensorRtLine = $SourcePackage.tensorRtLine
    cudaLine = $SourcePackage.cudaLine
    role = $Role
    prototypeState = $SourcePackage.validationState
    assets = @($Assets | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Sort-Object -Unique)
    generated = $true
  }
}

function New-DynamicSplitPackagesForRuntime {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SourcePackage
  )

  if ($SourcePackage.platform -ne "linux") {
    return @()
  }

  $packages = New-Object System.Collections.Generic.List[object]
  $packages.Add((New-LinuxDynamicSplitPackage -SourcePackage $SourcePackage -Role "bridge" -Suffix "bridge" -Assets @($SourcePackage.bridgeFile)))
  $packages.Add((New-LinuxDynamicSplitPackage -SourcePackage $SourcePackage -Role "cuda-cudnn" -Suffix "cuda-cudnn" -Assets @(
        (Get-ExpectedNativeFileNames -RelativeFiles @($SourcePackage.cudaFiles + $SourcePackage.cudnnFiles))
      )))

  $tensorRtAssets = @(Get-ExpectedNativeFileNames -RelativeFiles @($SourcePackage.tensorRtFiles))
  $tensorRtBuilderAssets = @($tensorRtAssets | Where-Object { [string]$_ -like "*builder_resource*" })
  $tensorRtRuntimeAssets = @($tensorRtAssets | Where-Object { [string]$_ -notlike "*builder_resource*" })
  if ($tensorRtBuilderAssets.Count -gt 0) {
    $packages.Add((New-LinuxDynamicSplitPackage -SourcePackage $SourcePackage -Role "tensorrt" -Suffix "tensorrt-runtime" -PackageSuffix "TensorRtRuntime" -Assets $tensorRtRuntimeAssets))
    $packages.Add((New-LinuxDynamicSplitPackage -SourcePackage $SourcePackage -Role "tensorrt" -Suffix "tensorrt-builder" -PackageSuffix "TensorRtBuilder" -Assets $tensorRtBuilderAssets))
  }
  else {
    $packages.Add((New-LinuxDynamicSplitPackage -SourcePackage $SourcePackage -Role "tensorrt" -Suffix "tensorrt" -PackageSuffix "TensorRt" -Assets $tensorRtAssets))
  }

  return @($packages.ToArray())
}

function Write-ResolvedPackageSource {
  param(
    [string]$SourceDirectory
  )

  if ([string]::IsNullOrWhiteSpace($OutputPathFile)) {
    return
  }

  $outputDirectory = Split-Path -Parent $OutputPathFile
  if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
  }

  Set-Content -LiteralPath $OutputPathFile -Value $SourceDirectory -Encoding utf8
}

function ConvertTo-CommandLineArgument {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  if ($Value -notmatch '[\s"]') {
    return $Value
  }

  return '"' + ($Value -replace '"', '\"') + '"'
}

function Test-NuGetPackageFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return $false
  }

  try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
    $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
      foreach ($entry in $archive.Entries) {
        if ($entry.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase)) {
          return $true
        }
      }
    }
    finally {
      $archive.Dispose()
    }
  }
  catch {
    Write-Warning "Package file '$Path' is not a complete NuGet package yet: $($_.Exception.Message)"
    return $false
  }

  Write-Warning "Package file '$Path' does not contain a .nuspec entry."
  return $false
}

function Get-GlobalPackageCacheFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageId,
    [Parameter(Mandatory = $true)]
    [string]$PackageVersion
  )

  if ($SkipGlobalPackageCache.IsPresent) {
    return $null
  }

  $globalPackagesRoot = $env:NUGET_PACKAGES
  if ([string]::IsNullOrWhiteSpace($globalPackagesRoot)) {
    $userProfile = $env:USERPROFILE
    if ([string]::IsNullOrWhiteSpace($userProfile)) {
      return $null
    }

    $globalPackagesRoot = Join-Path $userProfile ".nuget\packages"
  }

  $packageIdLower = $PackageId.ToLowerInvariant()
  $packageVersionLower = $PackageVersion.ToLowerInvariant()
  $candidate = Join-Path $globalPackagesRoot (Join-Path $packageIdLower (Join-Path $packageVersionLower "$packageIdLower.$packageVersionLower.nupkg"))
  if (Test-NuGetPackageFile -Path $candidate) {
    return $candidate
  }

  return $null
}

function Invoke-GhReleaseDownloadWithRetry {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseTag,
    [Parameter(Mandatory = $true)]
    [string]$Repository,
    [Parameter(Mandatory = $true)]
    [string]$PackageFileName,
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory,
    [Parameter(Mandatory = $true)]
    [string]$TargetPath
  )

  for ($attempt = 1; $attempt -le $DownloadMaxAttempts; $attempt++) {
    if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
      Remove-Item -LiteralPath $TargetPath -Force
    }

    Write-Host "Downloading published split package '$PackageFileName' from GitHub Release '$ReleaseTag' (attempt $attempt/$DownloadMaxAttempts)."
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = "gh"
    $processInfo.UseShellExecute = $false
    $processInfo.Arguments = (@(
        "release",
        "download",
        $ReleaseTag,
        "--repo",
        $Repository,
        "--pattern",
        $PackageFileName,
        "--dir",
        $SourceDirectory,
        "--clobber"
      ) | ForEach-Object { ConvertTo-CommandLineArgument -Value $_ }) -join " "

    $process = [System.Diagnostics.Process]::Start($processInfo)
    $startedAt = Get-Date
    $lastSize = -1
    $lastProgressAt = Get-Date
    $timedOut = $false
    $stalled = $false

    while (-not $process.WaitForExit($DownloadProgressPollSeconds * 1000)) {
      $elapsedSeconds = ((Get-Date) - $startedAt).TotalSeconds
      if ($elapsedSeconds -gt $DownloadTimeoutSeconds) {
        $timedOut = $true
        break
      }

      if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
        $currentSize = (Get-Item -LiteralPath $TargetPath).Length
        if ($currentSize -ne $lastSize) {
          $lastSize = $currentSize
          $lastProgressAt = Get-Date
          Write-Host ("Download progress for {0}: {1} MB" -f $PackageFileName, [Math]::Round($currentSize / 1MB, 2))
        }
        elseif ($currentSize -gt 0 -and ((Get-Date) - $lastProgressAt).TotalSeconds -gt $DownloadStallSeconds) {
          $stalled = $true
          break
        }
      }
    }

    if ($timedOut -or $stalled) {
      try {
        $process.Kill()
      }
      catch {
      }

      $reason = if ($timedOut) { "timed out" } else { "stalled" }
      Write-Warning "Download $reason for '$PackageFileName'."
    }
    elseif ($process.ExitCode -eq 0 -and (Test-NuGetPackageFile -Path $TargetPath)) {
      return
    }
    else {
      Write-Warning "gh release download failed for '$PackageFileName' with exit code $($process.ExitCode)."
    }

    if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
      Remove-Item -LiteralPath $TargetPath -Force -ErrorAction SilentlyContinue
    }

    if ($attempt -lt $DownloadMaxAttempts) {
      Start-Sleep -Seconds 30
    }
  }

  throw "Failed to download a complete NuGet package '$PackageFileName' from release '$ReleaseTag' after $DownloadMaxAttempts attempt(s)."
}

$requestedRoles = @(Expand-KeyList -Values $SplitPackageRole | ForEach-Object { $_.ToLowerInvariant() })
if ($requestedRoles.Count -eq 0) {
  $requestedRoles = @("all")
}

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = $PSScriptRoot
  if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($Version)) {
  $Version = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1")
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  $Repository = $env:GH_REPO
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    $OutputRoot = Join-Path $RepositoryRoot "artifacts\stable-runtime-package-source"
  }
  else {
    $OutputRoot = Join-Path $env:RUNNER_TEMP "jyppx-release-package-source"
  }
}

$splitPackagePins = Resolve-SplitPackagePins -RuntimeKey $SourceRuntimeKey
$resolvedCudaCudnnPackageVersion = [string]$splitPackagePins.cudaCudnnPackageVersion
$resolvedTensorRtPackageVersion = [string]$splitPackagePins.tensorRtPackageVersion
$resolvedBridgePackageVersion = [string]$splitPackagePins.bridgePackageVersion
$resolvedCudaCudnnPackageReleaseTag = [string]$splitPackagePins.cudaCudnnPackageReleaseTag
$resolvedTensorRtPackageReleaseTag = [string]$splitPackagePins.tensorRtPackageReleaseTag

$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$sourcePackage = $runtimeManifest.packages | Where-Object { [string]$_.key -eq $SourceRuntimeKey } | Select-Object -First 1
if (-not $sourcePackage) {
  throw "Runtime package key '$SourceRuntimeKey' was not found."
}

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$allSplitPackages = @($splitManifest.packages | Where-Object { [string]$_.sourceRuntimeKey -eq $SourceRuntimeKey })
if ($allSplitPackages.Count -eq 0 -and $sourcePackage.platform -eq "linux") {
  $allSplitPackages = @(New-DynamicSplitPackagesForRuntime -SourcePackage $sourcePackage)
}
if ($allSplitPackages.Count -eq 0) {
  throw "No split runtime packages were defined for source runtime '$SourceRuntimeKey'."
}

$selectedKeys = @(
  $allSplitPackages |
    Where-Object { Test-SplitPackageRequested -SplitPackage $_ -RequestedRoles $requestedRoles } |
    ForEach-Object { [string]$_.key }
)

$missingPackages = @(
  $allSplitPackages |
    Where-Object { $selectedKeys -notcontains [string]$_.key }
)

if ($missingPackages.Count -eq 0) {
  Write-Host "All split component packages for '$SourceRuntimeKey' are built in this run; no published stable dependency package source is needed."
  Write-ResolvedPackageSource -SourceDirectory ""
  return
}

$downloadPackages = @($missingPackages | Where-Object {
    -not [string]::IsNullOrWhiteSpace((Get-SplitPackageReleaseTag `
          -SplitPackage $_ `
          -CudaCudnnReleaseTag $resolvedCudaCudnnPackageReleaseTag `
          -TensorRtReleaseTag $resolvedTensorRtPackageReleaseTag))
  })

if ($downloadPackages.Count -eq 0) {
  Write-Host "No stable dependency package release tag was resolved; relying on configured NuGet feeds for previously published split packages."
  Write-ResolvedPackageSource -SourceDirectory ""
  return
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  throw "Repository was not provided. Pass -Repository or set GH_REPO before downloading GitHub Release assets."
}

$safeKey = $SourceRuntimeKey -replace '[^A-Za-z0-9._-]', '-'
$sourceDirectory = Join-Path $OutputRoot $safeKey
New-Item -ItemType Directory -Path $sourceDirectory -Force | Out-Null

foreach ($package in $missingPackages) {
  $packageVersion = Get-SplitPackageVersion `
    -SplitPackage $package `
    -BridgeVersion $resolvedBridgePackageVersion `
    -CudaCudnnVersion $resolvedCudaCudnnPackageVersion `
    -TensorRtVersion $resolvedTensorRtPackageVersion
  $packageFileName = "$($package.packageId).$packageVersion.nupkg"
  $targetPath = Join-Path $sourceDirectory $packageFileName
  if (Test-NuGetPackageFile -Path $targetPath) {
    Write-Host "Using cached release asset package: $packageFileName"
    continue
  }

  if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
    Write-Warning "Removing incomplete cached package file: $targetPath"
    Remove-Item -LiteralPath $targetPath -Force
  }

  $globalPackageFile = Get-GlobalPackageCacheFile -PackageId $package.packageId -PackageVersion $packageVersion
  if (-not [string]::IsNullOrWhiteSpace($globalPackageFile)) {
    Write-Host "Copying published split package from NuGet global cache: $globalPackageFile"
    Copy-Item -LiteralPath $globalPackageFile -Destination $targetPath -Force
    continue
  }

  $releaseTag = Get-SplitPackageReleaseTag `
    -SplitPackage $package `
    -CudaCudnnReleaseTag $resolvedCudaCudnnPackageReleaseTag `
    -TensorRtReleaseTag $resolvedTensorRtPackageReleaseTag
  if ([string]::IsNullOrWhiteSpace($releaseTag)) {
    Write-Host "No release tag was provided for '$($package.packageId)'; relying on configured NuGet feeds."
    continue
  }

  Invoke-GhReleaseDownloadWithRetry `
    -ReleaseTag $releaseTag `
    -Repository $Repository `
    -PackageFileName $packageFileName `
    -SourceDirectory $sourceDirectory `
    -TargetPath $targetPath
}

Write-Host "Published stable dependency package source: $sourceDirectory"
Write-ResolvedPackageSource -SourceDirectory $sourceDirectory
