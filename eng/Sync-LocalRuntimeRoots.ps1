[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [string]$TensorRtRoot = $env:JYPPX_TENSORRT_ROOT,
  [string]$CudaRoot = $env:JYPPX_CUDA_ROOT,
  [string]$CudnnRoot = $env:JYPPX_CUDNN_ROOT,
  [switch]$WriteRepositoryLocalFile,
  [switch]$WriteUserProfileFile = $true,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$userProfileConfigPath = Join-Path $env:USERPROFILE ".jyppx\runtime-packages.local.json"
$repositoryLocalConfigPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.local.json"

foreach ($requiredRoot in @(
  @{ Name = "TensorRT"; Path = $TensorRtRoot; Marker = "include\NvInfer.h" },
  @{ Name = "CUDA"; Path = $CudaRoot; Marker = "include\cuda_runtime.h" }
)) {
  if ([string]::IsNullOrWhiteSpace($requiredRoot.Path) -or
      -not (Test-Path -LiteralPath (Join-Path $requiredRoot.Path $requiredRoot.Marker) -PathType Leaf)) {
    throw "$($requiredRoot.Name) root is missing or invalid. Pass the installed SDK root explicitly."
  }
}

$TensorRtRoot = (Resolve-Path -LiteralPath $TensorRtRoot).Path
$CudaRoot = (Resolve-Path -LiteralPath $CudaRoot).Path
if (-not [string]::IsNullOrWhiteSpace($CudnnRoot)) {
  if (-not (Test-Path -LiteralPath $CudnnRoot -PathType Container)) {
    throw "cuDNN root is invalid: $CudnnRoot"
  }
  $CudnnRoot = (Resolve-Path -LiteralPath $CudnnRoot).Path
}

$document = @{
  packages = @(@{
    key = $RuntimePackageKey
    defaultTensorRtRoot = $TensorRtRoot
    defaultCudaRoot = $CudaRoot
    defaultCudnnRoot = $CudnnRoot
  })
}

function Write-ConfigFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  $directory = Split-Path -Path $Path -Parent
  if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $document | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Path -Encoding utf8
  Write-Host "Wrote runtime root overrides: $Path"
}

if ($WriteRepositoryLocalFile.IsPresent) {
  Write-ConfigFile -Path $repositoryLocalConfigPath
}

if ($WriteUserProfileFile.IsPresent) {
  Write-ConfigFile -Path $userProfileConfigPath
}
