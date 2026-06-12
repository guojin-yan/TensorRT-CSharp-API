[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [string]$BridgeConfiguration,
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
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

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

function Merge-LocalRuntimeOverrides {
  param(
    [object]$Manifest,
    [string]$Root
  )

  $localManifestPath = Join-Path $Root "pack\runtime\runtime-packages.local.json"
  if (-not (Test-Path -LiteralPath $localManifestPath)) {
    return
  }

  $localManifest = Get-Content -LiteralPath $localManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  foreach ($localPackage in @($localManifest.packages)) {
    $targetPackage = $Manifest.packages | Where-Object { $_.key -eq $localPackage.key } | Select-Object -First 1
    if (-not $targetPackage) {
      continue
    }

    foreach ($property in $localPackage.PSObject.Properties) {
      if ($property.Name -eq "key") {
        continue
      }

      $existing = $targetPackage.PSObject.Properties[$property.Name]
      if ($existing) {
        $existing.Value = $property.Value
      }
      else {
        $targetPackage | Add-Member -NotePropertyName $property.Name -NotePropertyValue $property.Value
      }
    }
  }
}

Merge-LocalRuntimeOverrides -Manifest $manifest -Root $RepositoryRoot
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found in $manifestPath."
}

if (-not $BridgeConfiguration) {
  $BridgeConfiguration = if ($package.bridgeConfiguration) { $package.bridgeConfiguration } else { "Release" }
}

if (-not $TensorRtRoot -and $package.defaultTensorRtRoot) {
  $TensorRtRoot = $package.defaultTensorRtRoot
}

if (-not $CudaRoot -and $package.defaultCudaRoot) {
  $CudaRoot = $package.defaultCudaRoot
}

if (-not $CudnnRoot -and $package.defaultCudnnRoot) {
  $CudnnRoot = $package.defaultCudnnRoot
}

if (-not $OutputRoot) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\runtime\$RuntimePackageKey"
}

$nativeOutput = Join-Path $OutputRoot "runtimes\$($package.rid)\native"
New-Item -ItemType Directory -Path $nativeOutput -Force | Out-Null
$packageAssetsRoot = Join-Path $RepositoryRoot "pack\runtime\$RuntimePackageKey\assets"
$packageNativeOutput = Join-Path $packageAssetsRoot "runtimes\$($package.rid)\native"
if (Test-Path -LiteralPath $packageAssetsRoot) {
  Remove-Item -LiteralPath $packageAssetsRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $packageNativeOutput -Force | Out-Null

$bridgeName = $package.bridgeFile
$buildPreset = if ($package.buildPreset) { $package.buildPreset } else { $RuntimePackageKey }
$bridgeSource = Join-Path $RepositoryRoot "build-out\$buildPreset\bin\$BridgeConfiguration\$bridgeName"
if (-not (Test-Path -LiteralPath $bridgeSource)) {
  throw "Bridge binary was not found: $bridgeSource"
}

Copy-Item -LiteralPath $bridgeSource -Destination (Join-Path $nativeOutput $bridgeName) -Force
Copy-Item -LiteralPath $bridgeSource -Destination (Join-Path $packageNativeOutput $bridgeName) -Force

$copiedFiles = New-Object System.Collections.Generic.List[string]
$copiedFiles.Add((Join-Path $nativeOutput $bridgeName))

function Unblock-CopiedRuntimeAsset {
  param(
    [string]$Path
  )

  if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
    return
  }

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return
  }

  $unblockCommand = Get-Command Unblock-File -ErrorAction SilentlyContinue
  if (-not $unblockCommand) {
    return
  }

  try {
    Unblock-File -LiteralPath $Path -ErrorAction Stop
  }
  catch {
    Write-Warning "Unable to unblock runtime asset '$Path': $($_.Exception.Message)"
  }
}

Unblock-CopiedRuntimeAsset -Path (Join-Path $nativeOutput $bridgeName)
Unblock-CopiedRuntimeAsset -Path (Join-Path $packageNativeOutput $bridgeName)

function Copy-RelativeFiles {
  param(
    [string]$BaseRoot,
    [object[]]$RelativeFiles,
    [string]$Label,
    [string]$ArtifactOutput,
  [string]$PackageOutput
  )

  $requestedRelativeFiles = @($RelativeFiles | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })

  if (-not $BaseRoot) {
    if ($requestedRelativeFiles.Count -gt 0) {
      throw "$Label root was not provided for runtime package '$RuntimePackageKey'."
    }

    return
  }

  foreach ($relativePath in $requestedRelativeFiles) {
    $normalizedRelativePath = $relativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar
    $sourcePath = Join-Path $BaseRoot $normalizedRelativePath

    $matchingFiles = @()
    if ($normalizedRelativePath.IndexOfAny(@('*', '?')) -ge 0) {
      $matchingFiles = Resolve-Path -Path $sourcePath -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }
    }
    elseif (Test-Path -LiteralPath $sourcePath) {
      $matchingFiles = @((Resolve-Path -LiteralPath $sourcePath).Path)
    }

    if (-not $matchingFiles -or $matchingFiles.Count -eq 0) {
      throw "Expected $Label asset was not found: $sourcePath"
    }

    foreach ($resolvedSourcePath in $matchingFiles) {
      $destinationPath = Join-Path $nativeOutput ([System.IO.Path]::GetFileName($resolvedSourcePath))
      $packageDestinationPath = Join-Path $PackageOutput ([System.IO.Path]::GetFileName($resolvedSourcePath))
      Copy-Item -LiteralPath $resolvedSourcePath -Destination $destinationPath -Force
      Copy-Item -LiteralPath $resolvedSourcePath -Destination $packageDestinationPath -Force
      Unblock-CopiedRuntimeAsset -Path $destinationPath
      Unblock-CopiedRuntimeAsset -Path $packageDestinationPath
      $copiedFiles.Add($destinationPath)
    }
  }
}

Copy-RelativeFiles -BaseRoot $TensorRtRoot -RelativeFiles @($package.tensorRtFiles) -Label "TensorRT" -ArtifactOutput $nativeOutput -PackageOutput $packageNativeOutput
Copy-RelativeFiles -BaseRoot $CudaRoot -RelativeFiles @($package.cudaFiles) -Label "CUDA" -ArtifactOutput $nativeOutput -PackageOutput $packageNativeOutput
Copy-RelativeFiles -BaseRoot $CudnnRoot -RelativeFiles @($package.cudnnFiles) -Label "cuDNN" -ArtifactOutput $nativeOutput -PackageOutput $packageNativeOutput

$artifactManifest = [ordered]@{
  key = $package.key
  packageId = $package.packageId
  rid = $package.rid
  bridgeConfiguration = $BridgeConfiguration
  buildPreset = $buildPreset
  tensorRtRoot = $TensorRtRoot
  cudaRoot = $CudaRoot
  cudnnRoot = $CudnnRoot
  packageAssetsRoot = $packageAssetsRoot
  files = $copiedFiles
}

$artifactManifestPath = Join-Path $OutputRoot "artifact-manifest.json"
$artifactManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $artifactManifestPath -Encoding utf8

Write-Host "Collected runtime assets for $($package.packageId)"
Write-Host "Output: $OutputRoot"
