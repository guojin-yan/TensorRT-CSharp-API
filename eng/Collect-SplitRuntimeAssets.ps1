[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$SplitPackageKey,
  [string]$SourceAssetsRoot,
  [string]$BridgeConfiguration,
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$isWindowsHost = $env:OS -eq "Windows_NT"

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json

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

function Resolve-LinuxRuntimeAssetCopySource {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if ($isWindowsHost) {
    return (Resolve-Path -LiteralPath $Path).Path
  }

  try {
    $fileInfo = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ($fileInfo.LinkType -eq "SymbolicLink" -and $fileInfo.Target) {
      $targetPath = if ([System.IO.Path]::IsPathRooted([string]$fileInfo.Target)) {
        [string]$fileInfo.Target
      }
      else {
        Join-Path $fileInfo.DirectoryName ([string]$fileInfo.Target)
      }

      if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
        return (Resolve-Path -LiteralPath $targetPath).Path
      }
    }
  }
  catch {
    Write-Warning "Unable to inspect Linux split runtime asset '$Path': $($_.Exception.Message)"
  }

  return (Resolve-Path -LiteralPath $Path).Path
}

function Get-LinuxRuntimeAssetSortRank {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  $fileName = [System.IO.Path]::GetFileName($Path)
  if ($fileName -match '\.so\.\d+$') {
    return 0
  }

  if ($fileName -match '\.so\.\d+\.') {
    return 1
  }

  if ($fileName -match '\.so$') {
    return 2
  }

  return 3
}

$splitPackage = $splitManifest.packages | Where-Object { $_.key -eq $SplitPackageKey } | Select-Object -First 1
$sourcePackage = $null
if ($splitPackage) {
  $sourcePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $splitPackage.sourceRuntimeKey } | Select-Object -First 1
}
else {
  foreach ($runtimePackage in @($runtimeManifest.packages | Where-Object { $_.platform -eq "linux" })) {
    $dynamicSplitPackage = New-DynamicSplitPackagesForRuntime -SourcePackage $runtimePackage |
      Where-Object { $_.key -eq $SplitPackageKey } |
      Select-Object -First 1
    if ($dynamicSplitPackage) {
      $splitPackage = $dynamicSplitPackage
      $sourcePackage = $runtimePackage
      break
    }
  }
}

if (-not $splitPackage) {
  throw "Split package key '$SplitPackageKey' was not found."
}
if (-not $sourcePackage) {
  throw "Source runtime key '$($splitPackage.sourceRuntimeKey)' was not found."
}

if ([string]::IsNullOrWhiteSpace($SourceAssetsRoot)) {
  $SourceAssetsRoot = Join-Path $RepositoryRoot "pack\runtime\$($splitPackage.sourceRuntimeKey)\assets\runtimes\$($sourcePackage.rid)\native"
  if (-not (Test-Path -LiteralPath $SourceAssetsRoot -PathType Container) -and [string]$splitPackage.role -eq "bridge") {
    $configuration = if (-not [string]::IsNullOrWhiteSpace($BridgeConfiguration)) {
      $BridgeConfiguration
    }
    elseif ($sourcePackage.PSObject.Properties.Name.Contains("bridgeConfiguration") -and -not [string]::IsNullOrWhiteSpace([string]$sourcePackage.bridgeConfiguration)) {
      [string]$sourcePackage.bridgeConfiguration
    }
    else {
      "Release"
    }

    $buildPreset = if ($sourcePackage.PSObject.Properties.Name.Contains("buildPreset") -and -not [string]::IsNullOrWhiteSpace([string]$sourcePackage.buildPreset)) {
      [string]$sourcePackage.buildPreset
    }
    else {
      [string]$splitPackage.sourceRuntimeKey
    }

    $bridgeSourceRoot = Join-Path $RepositoryRoot "build-out\$buildPreset\bin\$configuration"
    $bridgeFile = [string]($splitPackage.assets | Select-Object -First 1)
    if (-not [string]::IsNullOrWhiteSpace([string]$bridgeFile) -and (Test-Path -LiteralPath (Join-Path $bridgeSourceRoot ([string]$bridgeFile)) -PathType Leaf)) {
      $SourceAssetsRoot = $bridgeSourceRoot
      Write-Host "Using bridge build output as split source assets root: $SourceAssetsRoot"
    }
  }
}

if (-not (Test-Path -LiteralPath $SourceAssetsRoot -PathType Container)) {
  throw "Source runtime native asset folder was not found: $SourceAssetsRoot"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\runtime-split\$SplitPackageKey"
}

$artifactNativeOutput = Join-Path $OutputRoot "runtimes\$($splitPackage.rid)\native"
$packageAssetsRoot = Join-Path $RepositoryRoot "pack\runtime-split\$SplitPackageKey\assets"
$packageNativeOutput = Join-Path $packageAssetsRoot "runtimes\$($splitPackage.rid)\native"

if (Test-Path -LiteralPath $packageAssetsRoot) {
  Remove-Item -LiteralPath $packageAssetsRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $artifactNativeOutput -Force | Out-Null
New-Item -ItemType Directory -Path $packageNativeOutput -Force | Out-Null

$copiedFiles = New-Object System.Collections.Generic.List[string]
$copiedDestinationNames = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
$copiedRealPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
foreach ($asset in @($splitPackage.assets)) {
  $sourcePath = Join-Path $SourceAssetsRoot $asset
  $matchingFiles = @()
  if ($asset.IndexOfAny(@('*', '?')) -ge 0) {
    $matchingFiles = @(Resolve-Path -Path $sourcePath -ErrorAction SilentlyContinue | ForEach-Object { $_.Path })
  }
  elseif (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
    $matchingFiles = @((Resolve-Path -LiteralPath $sourcePath).Path)
  }

  if ($matchingFiles.Count -eq 0) {
    throw "Expected split runtime asset was not found: $sourcePath"
  }

  if (-not $isWindowsHost) {
    $matchingFiles = @(
      $matchingFiles |
        Sort-Object `
          @{ Expression = { Get-LinuxRuntimeAssetSortRank -Path ([string]$_) } },
          @{ Expression = { [System.IO.Path]::GetFileName([string]$_) } }
    )
  }

  foreach ($resolvedSourcePath in @($matchingFiles | Sort-Object)) {
    $copySourcePath = Resolve-LinuxRuntimeAssetCopySource -Path $resolvedSourcePath
    if (-not $isWindowsHost) {
      $realPath = try { (Resolve-Path -LiteralPath $copySourcePath).Path } catch { $copySourcePath }
      if (-not $copiedRealPaths.Add($realPath)) {
        Write-Host "Skipping duplicate Linux split runtime asset resolved through symlink: $resolvedSourcePath -> $realPath"
        continue
      }
    }

    $destinationFileName = [System.IO.Path]::GetFileName($resolvedSourcePath)
    if (-not $copiedDestinationNames.Add($destinationFileName)) {
      Write-Host "Skipping duplicate split runtime asset destination: $destinationFileName"
      continue
    }

    $artifactDestination = Join-Path $artifactNativeOutput $destinationFileName
    $packageDestination = Join-Path $packageNativeOutput $destinationFileName
    Copy-Item -LiteralPath $copySourcePath -Destination $artifactDestination -Force -ErrorAction Stop
    Copy-Item -LiteralPath $copySourcePath -Destination $packageDestination -Force -ErrorAction Stop
    $copiedFiles.Add($artifactDestination)
  }
}

$artifactManifest = [ordered]@{
  splitPackageKey = $splitPackage.key
  sourceRuntimeKey = $splitPackage.sourceRuntimeKey
  packageId = $splitPackage.packageId
  rid = $splitPackage.rid
  role = $splitPackage.role
  prototypeState = $splitPackage.prototypeState
  sourceAssetsRoot = $SourceAssetsRoot
  packageAssetsRoot = $packageAssetsRoot
  files = @($copiedFiles)
}

$artifactManifestPath = Join-Path $OutputRoot "artifact-manifest.json"
$artifactManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $artifactManifestPath -Encoding utf8

Write-Host "Collected split runtime assets for $($splitPackage.packageId)"
Write-Host "Output: $OutputRoot"
