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
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

if (-not (Test-Path -LiteralPath $TensorRtRoot)) {
  throw "TensorRT root was not found: $TensorRtRoot"
}

if (-not (Test-Path -LiteralPath $CudaRoot)) {
  throw "CUDA root was not found: $CudaRoot"
}

if (-not [string]::IsNullOrWhiteSpace($CudnnRoot) -and -not (Test-Path -LiteralPath $CudnnRoot)) {
  throw "cuDNN root was not found: $CudnnRoot"
}

function Test-GlobMatches {
  param(
    [string]$BaseRoot,
    [string[]]$Patterns,
    [string]$Label
  )

  foreach ($pattern in $Patterns) {
    $normalizedPattern = $pattern -replace '\\', '/'
    $combined = Join-Path $BaseRoot ($normalizedPattern -replace '/', [System.IO.Path]::DirectorySeparatorChar)
    $matches = Resolve-Path -Path $combined -ErrorAction SilentlyContinue
    if (-not $matches) {
      throw "$Label pattern did not match any files: $combined"
    }
  }
}

function Find-CudaHostDefinesHeader {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Root
  )

  $includeCandidates = @(
    (Join-Path $Root "include"),
    (Join-Path $Root "targets\x86_64-linux\include"),
    (Join-Path $Root "targets\aarch64-linux\include")
  )

  foreach ($includeRoot in $includeCandidates) {
    foreach ($relativePath in @("crt\host_defines.h", "host_defines.h")) {
      $candidate = Join-Path $includeRoot $relativePath
      if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return (Resolve-Path -LiteralPath $candidate).Path
      }
    }
  }

  return $null
}

Test-GlobMatches -BaseRoot $TensorRtRoot -Patterns $package.tensorRtFiles -Label "TensorRT"
Test-GlobMatches -BaseRoot $CudaRoot -Patterns $package.cudaFiles -Label "CUDA"
if (-not [string]::IsNullOrWhiteSpace($CudnnRoot)) {
  Test-GlobMatches -BaseRoot $CudnnRoot -Patterns $package.cudnnFiles -Label "cuDNN"
}

$cudaHostDefinesHeader = Find-CudaHostDefinesHeader -Root $CudaRoot
if (-not $cudaHostDefinesHeader) {
  throw "CUDA host compiler headers were not found for '$RuntimePackageKey'. Expected host_defines.h or crt/host_defines.h under '$CudaRoot'."
}

Write-Host "Linux runtime input validation passed for $RuntimePackageKey"
