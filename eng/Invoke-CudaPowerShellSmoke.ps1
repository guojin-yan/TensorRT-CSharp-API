[CmdletBinding()]
param(
  [string]$BridgePath,
  [string]$CudaRoot,
  [string]$Configuration = "Debug",
  [string]$TargetFramework = "net48",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [string]::IsNullOrWhiteSpace($BridgePath)) {
  $env:JYPPX_NATIVE_BRIDGE_PATH = $BridgePath
}

if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) {
  $env:JYPPX_CUDA_ROOT = $CudaRoot
}

$sharedAssembly = Join-Path $RepositoryRoot "src\JYPPX.Shared\bin\$Configuration\$TargetFramework\JYPPX.Shared.dll"
$cudaAssembly = Join-Path $RepositoryRoot "src\JYPPX.CudaSharp\bin\$Configuration\$TargetFramework\JYPPX.CudaSharp.dll"

if (-not (Test-Path -LiteralPath $sharedAssembly)) {
  throw "JYPPX.Shared assembly was not found: $sharedAssembly. Run dotnet build first."
}

if (-not (Test-Path -LiteralPath $cudaAssembly)) {
  throw "JYPPX.CudaSharp assembly was not found: $cudaAssembly. Run dotnet build first."
}

Add-Type -Path $sharedAssembly
Add-Type -Path $cudaAssembly

$snapshot = [JYPPX.CudaSharp.CudaEnvironmentProbe]::GetCurrent()
Write-Host "Bridge=$($snapshot.BuildInfo.BridgeName) CUDA=$($snapshot.BuildInfo.CudaToolkitVersion) Devices=$($snapshot.CudaRuntimeInfo.DeviceCount) Vendor=$($snapshot.CudaRuntimeInfo.VendorDependencyAvailable)"

if (-not $snapshot.CudaRuntimeInfo.VendorDependencyAvailable) {
  throw $snapshot.CudaRuntimeInfo.StatusMessage
}

$stream = [JYPPX.CudaSharp.CudaStream]::new([JYPPX.CudaSharp.CudaStreamCreationFlags]::NonBlocking)
$event = [JYPPX.CudaSharp.CudaEvent]::new()
$memory = [JYPPX.CudaSharp.CudaMemory]::new([uint64]64)

try {
  $memory.Fill(90, [uint64]64)
  $bytes = $memory.ToArray([uint64]64)
  $fillOk = ($bytes | Where-Object { $_ -ne 90 }).Count -eq 0

  $event.Record($stream)
  $stream.Synchronize()
  $event.Synchronize()

  if (-not $fillOk) {
    throw "CUDA fill verification failed."
  }

  Write-Host "CudaPowerShellSmoke Fill=$fillOk StreamReady=$($stream.IsReady()) EventReady=$($event.IsReady())"
}
finally {
  $memory.Dispose()
  $event.Dispose()
  $stream.Dispose()
}
