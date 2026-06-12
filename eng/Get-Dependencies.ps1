[CmdletBinding()]
param(
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [ValidateSet("8", "10", "11")]
  [string]$TensorRtLine,
  [ValidateSet("11", "12", "13")]
  [string]$CudaLine,
  [ValidateSet("x64", "arm64")]
  [string]$Architecture = "x64",
  [switch]$AsJson,
  [switch]$RequireAll
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-FirstExistingPath {
  param(
    [string[]]$Candidates
  )

  foreach ($candidate in $Candidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) {
      continue
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($candidate)
    if (Test-Path -LiteralPath $expanded) {
      return (Resolve-Path -LiteralPath $expanded).Path
    }
  }

  return $null
}

function Get-CudaInstallationRoots {
  param([string]$BaseDirectory)

  $results = New-Object System.Collections.Generic.List[string]

  if (Test-CudaRoot -Path $BaseDirectory) {
    $results.Add((Resolve-Path -LiteralPath $BaseDirectory).Path)
  }

  if (Test-Path -LiteralPath $BaseDirectory) {
    Get-ChildItem -LiteralPath $BaseDirectory -Directory -ErrorAction SilentlyContinue |
      Sort-Object Name |
      ForEach-Object {
        if (Test-CudaRoot -Path $_.FullName) {
          $results.Add($_.FullName)
        }
      }
  }

  return $results
}

function Get-TensorRtInstallationRoots {
  param([string]$BaseDirectory)

  $results = New-Object System.Collections.Generic.List[string]

  if (Test-TensorRtRoot -Path $BaseDirectory) {
    $results.Add((Resolve-Path -LiteralPath $BaseDirectory).Path)
  }

  if (Test-Path -LiteralPath $BaseDirectory) {
    Get-ChildItem -LiteralPath $BaseDirectory -Directory -ErrorAction SilentlyContinue |
      Sort-Object Name |
      ForEach-Object {
        if (Test-TensorRtRoot -Path $_.FullName) {
          $results.Add($_.FullName)
        }
      }
  }

  return $results
}

function Get-SelectedTensorRtInstallation {
  param(
    [object[]]$Installations,
    [string]$RequestedTensorRtLine,
    [string]$RequestedCudaLine
  )

  if (-not $RequestedTensorRtLine -or -not $RequestedCudaLine) {
    return $null
  }

  $matches = $Installations | Where-Object {
    $_.name -match "^TensorRT-$RequestedTensorRtLine(\.|-)" -and
    $_.name -match "cuda $RequestedCudaLine(\.|$)"
  }

  return $matches | Select-Object -First 1
}

function Get-SelectedCudaInstallation {
  param(
    [object[]]$Installations,
    [string]$RequestedCudaLine,
    [object]$SelectedTensorRtInstallation
  )

  if (-not $RequestedCudaLine) {
    return $null
  }

  $matches = $Installations | Where-Object { $_.name -like "v$RequestedCudaLine.*" } | Sort-Object -Property name
  if (-not $matches) {
    return $null
  }

  if ($SelectedTensorRtInstallation -and $SelectedTensorRtInstallation.name -like "TensorRT-8.6.1.6-cuda 12.0*" -and $RequestedCudaLine -eq "12") {
    $bounded = $matches | Where-Object { $_.name -le "v12.1" }
    if ($bounded) {
      return $bounded | Select-Object -Last 1
    }
  }

  return $matches | Select-Object -Last 1
}

function Test-TensorRtRoot {
  param([string]$Path)

  if (-not $Path) {
    return $false
  }

  return (Test-Path -LiteralPath (Join-Path $Path "include\\NvInfer.h")) -or
         (Test-Path -LiteralPath (Join-Path $Path "include\\NvInferVersion.h"))
}

function Test-CudaRoot {
  param([string]$Path)

  if (-not $Path) {
    return $false
  }

  return Test-Path -LiteralPath (Join-Path $Path "include\\cuda_runtime.h")
}

$tensorRtCandidates = @(
  $TensorRtRoot,
  $env:JYPPX_TENSORRT_ROOT,
  $env:TENSORRT_ROOT,
  $env:TensorRT_ROOT,
  (Join-Path $repoRoot "third_party\nvidia\tensorrt"),
  (Join-Path $repoRoot "third_party\nvidia"),
  "C:\TensorRT",
  "C:\Program Files\TensorRT",
  "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT"
)

$cudaCandidates = @(
  $CudaRoot,
  $env:JYPPX_CUDA_ROOT,
  $env:CUDA_PATH,
  $env:CUDAToolkit_ROOT,
  (Join-Path $repoRoot "third_party\nvidia\cuda"),
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
)

$resolvedTensorRtRoot = $null
$tensorRtInstallations = New-Object System.Collections.Generic.List[object]
foreach ($candidate in $tensorRtCandidates) {
  if ([string]::IsNullOrWhiteSpace($candidate)) {
    continue
  }

  $expanded = [Environment]::ExpandEnvironmentVariables($candidate)
  foreach ($root in (Get-TensorRtInstallationRoots -BaseDirectory $expanded)) {
    if (-not ($tensorRtInstallations | Where-Object { $_.root -eq $root })) {
      $tensorRtInstallations.Add([pscustomobject]@{
        root = $root
        name = [System.IO.Path]::GetFileName($root)
      })
    }
  }
}

if ($TensorRtRoot -and (Test-TensorRtRoot -Path $TensorRtRoot)) {
  $resolvedTensorRtRoot = (Resolve-Path -LiteralPath $TensorRtRoot).Path
}
elseif ($env:JYPPX_TENSORRT_ROOT -and (Test-TensorRtRoot -Path $env:JYPPX_TENSORRT_ROOT)) {
  $resolvedTensorRtRoot = (Resolve-Path -LiteralPath $env:JYPPX_TENSORRT_ROOT).Path
}
elseif (($selectedTensorRtInstallation = Get-SelectedTensorRtInstallation -Installations $tensorRtInstallations -RequestedTensorRtLine $TensorRtLine -RequestedCudaLine $CudaLine)) {
  $resolvedTensorRtRoot = $selectedTensorRtInstallation.root
}
elseif ($tensorRtInstallations.Count -eq 1) {
  $resolvedTensorRtRoot = $tensorRtInstallations[0].root
}

$resolvedCudaRoot = $null
$cudaInstallations = New-Object System.Collections.Generic.List[object]
foreach ($candidate in $cudaCandidates) {
  if ([string]::IsNullOrWhiteSpace($candidate)) {
    continue
  }

  $expanded = [Environment]::ExpandEnvironmentVariables($candidate)
  foreach ($root in (Get-CudaInstallationRoots -BaseDirectory $expanded)) {
    if (-not ($cudaInstallations | Where-Object { $_.root -eq $root })) {
      $cudaInstallations.Add([pscustomobject]@{
        root = $root
        name = [System.IO.Path]::GetFileName($root)
      })
    }
  }
}

if ($CudaRoot -and (Test-CudaRoot -Path $CudaRoot)) {
  $resolvedCudaRoot = (Resolve-Path -LiteralPath $CudaRoot).Path
}
elseif ($env:JYPPX_CUDA_ROOT -and (Test-CudaRoot -Path $env:JYPPX_CUDA_ROOT)) {
  $resolvedCudaRoot = (Resolve-Path -LiteralPath $env:JYPPX_CUDA_ROOT).Path
}
elseif (($selectedCudaInstallation = Get-SelectedCudaInstallation -Installations $cudaInstallations -RequestedCudaLine $CudaLine -SelectedTensorRtInstallation $selectedTensorRtInstallation)) {
  $resolvedCudaRoot = $selectedCudaInstallation.root
}
elseif ($env:CUDA_PATH -and (Test-CudaRoot -Path $env:CUDA_PATH)) {
  $resolvedCudaRoot = (Resolve-Path -LiteralPath $env:CUDA_PATH).Path
}
elseif ($cudaInstallations.Count -gt 0) {
  $resolvedCudaRoot = ($cudaInstallations | Sort-Object root | Select-Object -Last 1).root
}

$result = [ordered]@{
  architecture = $Architecture
  requested = [ordered]@{
    tensorRtLine = $TensorRtLine
    cudaLine = $CudaLine
  }
  tensorRt = [ordered]@{
    found = [bool]$resolvedTensorRtRoot
    root = $resolvedTensorRtRoot
    include = if ($resolvedTensorRtRoot) { Join-Path $resolvedTensorRtRoot "include" } else { $null }
    installations = $tensorRtInstallations
  }
  cuda = [ordered]@{
    found = [bool]$resolvedCudaRoot
    root = $resolvedCudaRoot
    include = if ($resolvedCudaRoot) { Join-Path $resolvedCudaRoot "include" } else { $null }
    installations = $cudaInstallations
  }
}

if ($AsJson) {
  $result | ConvertTo-Json -Depth 5
}
else {
  Write-Host "Architecture : $($result.architecture)"
  if ($result.requested.tensorRtLine -or $result.requested.cudaLine) {
    Write-Host "Requested    : TensorRT=$($result.requested.tensorRtLine) CUDA=$($result.requested.cudaLine)"
  }
  Write-Host "TensorRT     : $(if ($result.tensorRt.found) { 'FOUND' } else { 'MISSING' })"
  if ($result.tensorRt.root) {
    Write-Host "  Root       : $($result.tensorRt.root)"
  }
  else {
    Write-Host "  Root       : <not found>"
  }
  if ($result.tensorRt.installations.Count -gt 0) {
    Write-Host "  Installs   :"
    $result.tensorRt.installations | ForEach-Object { Write-Host "    - $($_.name) => $($_.root)" }
  }

  Write-Host "CUDA         : $(if ($result.cuda.found) { 'FOUND' } else { 'MISSING' })"
  if ($result.cuda.root) {
    Write-Host "  Root       : $($result.cuda.root)"
  }
  else {
    Write-Host "  Root       : <not found>"
  }
  if ($result.cuda.installations.Count -gt 0) {
    Write-Host "  Installs   :"
    $result.cuda.installations | ForEach-Object { Write-Host "    - $($_.name) => $($_.root)" }
  }
}

if ($RequireAll -and (-not $result.tensorRt.found -or -not $result.cuda.found)) {
  Write-Error "One or more required dependencies are missing. Provide explicit roots or install the vendor packages."
  exit 1
}
