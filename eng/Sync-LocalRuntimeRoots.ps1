[CmdletBinding()]
param(
  [switch]$WriteRepositoryLocalFile,
  [switch]$WriteUserProfileFile = $true,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$thirdPartyRoot = Join-Path $RepositoryRoot "third_party\nvidia"
$userProfileConfigPath = Join-Path $env:USERPROFILE ".jyppx\runtime-packages.local.json"
$repositoryLocalConfigPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.local.json"

if (-not (Test-Path -LiteralPath $thirdPartyRoot -PathType Container)) {
  throw "NVIDIA third-party directory was not found: $thirdPartyRoot"
}

$packages = @(
  @{
    key = "win-x64-trt8.6-cuda11.8-cudnn8.9"
    tensorRtRoot = Join-Path $thirdPartyRoot "TensorRT-8.6.1.6-cuda 11.8"
    cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    cudnnRoot = Join-Path $thirdPartyRoot "cudnn-windows-x86_64-8.9.7.29_cuda11-archive"
  }
  @{
    key = "win-x64-trt8.6-cuda12.1-cudnn8.9"
    tensorRtRoot = Join-Path $thirdPartyRoot "TensorRT-8.6.1.6-cuda 12.1"
    cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    cudnnRoot = Join-Path $thirdPartyRoot "cudnn-windows-x86_64-8.9.7.29_cuda12-archive"
  }
  @{
    key = "win-x64-trt10.11-cuda11.8-cudnn8.9"
    tensorRtRoot = Join-Path $thirdPartyRoot "TensorRT-10.11.0.33-cuda 11.8"
    cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    cudnnRoot = Join-Path $thirdPartyRoot "cudnn-windows-x86_64-8.9.7.29_cuda11-archive"
  }
  @{
    key = "win-x64-trt10.11-cuda12.9-cudnn9.22"
    tensorRtRoot = Join-Path $thirdPartyRoot "TensorRT-10.11.0.33-cuda 12.9"
    cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
    cudnnRoot = Join-Path $thirdPartyRoot "cudnn_windows_x86_64_9.22.0_cuda12\v9.22"
  }
  @{
    key = "win-x64-trt11.0-cuda12.9-cudnn9.22"
    tensorRtRoot = Join-Path $thirdPartyRoot "TensorRT-11.0.0.114-cuda 12.9"
    cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
    cudnnRoot = Join-Path $thirdPartyRoot "cudnn_windows_x86_64_9.22.0_cuda12\v9.22"
  }
  @{
    key = "win-x64-trt11.0-cuda13.2-cudnn9.22"
    tensorRtRoot = Join-Path $thirdPartyRoot "TensorRT-11.0.0.114-cuda 13.2"
    cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
    cudnnRoot = Join-Path $thirdPartyRoot "cudnn_windows_x86_64_9.22.0_cuda13"
  }
)

$document = @{
  packages = @(
    foreach ($package in $packages) {
      @{
        key = $package.key
        defaultTensorRtRoot = $package.tensorRtRoot
        defaultCudaRoot = $package.cudaRoot
        defaultCudnnRoot = $package.cudnnRoot
      }
    }
  )
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
