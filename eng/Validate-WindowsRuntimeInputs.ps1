[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [Parameter(Mandatory = $true)]
  [string]$TensorRtRoot,
  [Parameter(Mandatory = $true)]
  [string]$CudaRoot,
  [string]$CudnnRoot,
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
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "windows") {
  throw "Runtime package key '$RuntimePackageKey' is not a Windows package."
}

$errors = New-Object System.Collections.Generic.List[string]
if (-not (Test-Path -LiteralPath $TensorRtRoot -PathType Container)) {
  $errors.Add("TensorRT root does not exist: $TensorRtRoot")
}
if (-not (Test-Path -LiteralPath $CudaRoot -PathType Container)) {
  $errors.Add("CUDA root does not exist: $CudaRoot")
}
if (@($package.cudnnFiles).Count -gt 0) {
  if ([string]::IsNullOrWhiteSpace($CudnnRoot)) {
    $errors.Add("cuDNN root is required for runtime package '$RuntimePackageKey'.")
  }
  elseif (-not (Test-Path -LiteralPath $CudnnRoot -PathType Container)) {
    $errors.Add("cuDNN root does not exist: $CudnnRoot")
  }
}

function Test-RelativeAssetPresent {
  param(
    [string]$BaseRoot,
    [string]$RelativePath
  )

  $normalized = $RelativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar
  $path = Join-Path $BaseRoot $normalized
  if ($normalized.IndexOfAny(@('*', '?')) -ge 0) {
    return @(Resolve-Path -Path $path -ErrorAction SilentlyContinue).Count -gt 0
  }

  return Test-Path -LiteralPath $path -PathType Leaf
}

function Get-RelativeAssetDisplayPath {
  param(
    [string]$BaseRoot,
    [string]$RelativePath
  )

  return Join-Path $BaseRoot ($RelativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar)
}

foreach ($relativePath in @($package.tensorRtFiles)) {
  if (-not (Test-RelativeAssetPresent -BaseRoot $TensorRtRoot -RelativePath $relativePath)) {
    $errors.Add("Expected TensorRT asset was not found: $(Get-RelativeAssetDisplayPath -BaseRoot $TensorRtRoot -RelativePath $relativePath)")
  }
}

foreach ($relativePath in @($package.cudaFiles)) {
  if (-not (Test-RelativeAssetPresent -BaseRoot $CudaRoot -RelativePath $relativePath)) {
    $errors.Add("Expected CUDA asset was not found: $(Get-RelativeAssetDisplayPath -BaseRoot $CudaRoot -RelativePath $relativePath)")
  }
}

foreach ($relativePath in @($package.cudnnFiles)) {
  if ([string]::IsNullOrWhiteSpace($CudnnRoot)) {
    continue
  }

  if (-not (Test-RelativeAssetPresent -BaseRoot $CudnnRoot -RelativePath $relativePath)) {
    $errors.Add("Expected cuDNN asset was not found: $(Get-RelativeAssetDisplayPath -BaseRoot $CudnnRoot -RelativePath $relativePath)")
  }
}

if ($errors.Count -gt 0) {
  $errors | ForEach-Object { Write-Error $_ }
  exit 1
}

Write-Host "Windows runtime inputs are valid for $RuntimePackageKey."
