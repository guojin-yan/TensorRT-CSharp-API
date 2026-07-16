[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt8.6-cuda11.8-cudnn8.9",
  [int]$Iterations = 3,
  [ValidateSet("Debug", "Release")]
  [string]$Configuration = "Debug",
  [switch]$UseRuntimeAssets,
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
$localManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.local.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "windows") {
  throw "Runtime package key '$RuntimePackageKey' is not a Windows package."
}

if (-not (Test-Path -LiteralPath $localManifestPath -PathType Leaf)) {
  throw "Local runtime root override was not found: $localManifestPath"
}

$localManifest = Get-Content -LiteralPath $localManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$localPackage = $localManifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $localPackage) {
  throw "Local runtime root override does not contain '$RuntimePackageKey'."
}

$bridgeRoot = Join-Path $RepositoryRoot "build-out\$($package.buildPreset)\bin\$($package.bridgeConfiguration)"
if (-not (Test-Path -LiteralPath (Join-Path $bridgeRoot $package.bridgeFile) -PathType Leaf)) {
  throw "Bridge file was not found for $RuntimePackageKey under $bridgeRoot."
}

$runtimeNativeRoot = Join-Path $RepositoryRoot "artifacts\runtime\$RuntimePackageKey\runtimes\$($package.rid)\native"
$nativeSearchRoot = $bridgeRoot
if ($UseRuntimeAssets -and (Test-Path -LiteralPath (Join-Path $runtimeNativeRoot $package.bridgeFile) -PathType Leaf)) {
  $nativeSearchRoot = $runtimeNativeRoot
}

$runnerPath = Join-Path $RepositoryRoot "smoke\LifecycleSmokeRunner\bin\$Configuration\net8.0\LifecycleSmokeRunner.dll"
if (-not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
  throw "LifecycleSmokeRunner was not built: $runnerPath"
}

$previousBridgePath = $env:JYPPX_NATIVE_BRIDGE_PATH
$previousTensorRtRoot = $env:JYPPX_TENSORRT_ROOT
$previousCudaRoot = $env:JYPPX_CUDA_ROOT
$previousCudnnRoot = $env:JYPPX_CUDNN_ROOT
$previousPath = $env:PATH

try {
  $env:JYPPX_NATIVE_BRIDGE_PATH = $nativeSearchRoot
  $env:JYPPX_TENSORRT_ROOT = $localPackage.defaultTensorRtRoot
  $env:JYPPX_CUDA_ROOT = $localPackage.defaultCudaRoot
  $env:JYPPX_CUDNN_ROOT = $localPackage.defaultCudnnRoot
  $pathEntries = New-Object System.Collections.Generic.List[string]
  $pathEntries.Add($nativeSearchRoot)
  $tensorRtLibPath = Join-Path $env:JYPPX_TENSORRT_ROOT "lib"
  if (Test-Path -LiteralPath $tensorRtLibPath -PathType Container) {
    $pathEntries.Add($tensorRtLibPath)
  }

  foreach ($cudaBinPath in @(
      (Join-Path $env:JYPPX_CUDA_ROOT "bin\x64"),
      (Join-Path $env:JYPPX_CUDA_ROOT "bin")
    )) {
    if (Test-Path -LiteralPath $cudaBinPath -PathType Container) {
      $pathEntries.Add($cudaBinPath)
    }
  }

  if (-not [string]::IsNullOrWhiteSpace($env:JYPPX_CUDNN_ROOT)) {
    $cudnnBinPath = Join-Path $env:JYPPX_CUDNN_ROOT "bin"
    if (Test-Path -LiteralPath $cudnnBinPath -PathType Container) {
      $pathEntries.Add($cudnnBinPath)
    }
  }

  $pathEntries.Add($previousPath)
  $env:PATH = $pathEntries -join [System.IO.Path]::PathSeparator

  Write-Host "Running LifecycleSmokeRunner for $RuntimePackageKey"
  Write-Host "  Configuration: $Configuration"
  Write-Host "  Bridge: $nativeSearchRoot"
  Write-Host "  Asset source: $(if ($UseRuntimeAssets) { 'runtime-assets' } else { 'build-output' })"
  Write-Host "  TensorRT: $($env:JYPPX_TENSORRT_ROOT)"
  Write-Host "  CUDA: $($env:JYPPX_CUDA_ROOT)"
  if (-not [string]::IsNullOrWhiteSpace($env:JYPPX_CUDNN_ROOT)) {
    Write-Host "  cuDNN: $($env:JYPPX_CUDNN_ROOT)"
  }

  dotnet $runnerPath --iterations $Iterations --tensor-rt-line $package.tensorRtLine
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
finally {
  $env:JYPPX_NATIVE_BRIDGE_PATH = $previousBridgePath
  $env:JYPPX_TENSORRT_ROOT = $previousTensorRtRoot
  $env:JYPPX_CUDA_ROOT = $previousCudaRoot
  $env:JYPPX_CUDNN_ROOT = $previousCudnnRoot
  $env:PATH = $previousPath
}
