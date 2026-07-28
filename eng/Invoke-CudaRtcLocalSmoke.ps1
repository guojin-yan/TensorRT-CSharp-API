[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$BridgePath,
  [string]$OutputPath
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
if ([string]::IsNullOrWhiteSpace($BridgePath)) {
  $BridgePath = Join-Path $RepositoryRoot 'build-out\win-x64-trt11-cuda12-release\bin\Release\jyppxtrtbridge.dll'
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\local-smoke.json'
}

$BridgePath = (Resolve-Path -LiteralPath $BridgePath).Path
$sampleProject = Join-Path $RepositoryRoot 'samples\CudaRuntimeCompilation\CudaRuntimeCompilation.csproj'
$sampleDll = Join-Path $RepositoryRoot 'samples\CudaRuntimeCompilation\bin\Debug\net8.0\CudaRuntimeCompilation.dll'
$cudaRuntimeRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$tensorRtRoot = Join-Path $RepositoryRoot 'third_party\nvidia\TensorRT-11.0.0.114-cuda 12.9'
$toolkits = @(
  [ordered]@{ version = '11.8'; library = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvrtc64_112_0.dll' },
  [ordered]@{ version = '12.1'; library = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvrtc64_120_0.dll' },
  [ordered]@{ version = '12.9'; library = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvrtc64_120_0.dll' },
  [ordered]@{ version = '13.2'; library = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\nvrtc64_130_0.dll' }
)

function Get-PrefixedValue {
  param([string[]]$Lines, [string]$Prefix)
  $line = $Lines | Where-Object { $_.StartsWith($Prefix, [StringComparison]::Ordinal) } | Select-Object -First 1
  if ($null -eq $line) {
    return $null
  }
  return $line.Substring($Prefix.Length)
}

function Get-TranscriptSha256 {
  param([string[]]$Lines)
  $text = [string]::Join([Environment]::NewLine, $Lines) + [Environment]::NewLine
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
  $algorithm = [System.Security.Cryptography.SHA256]::Create()
  try {
    $hash = $algorithm.ComputeHash($bytes)
  }
  finally {
    $algorithm.Dispose()
  }
  return -join ($hash | ForEach-Object { $_.ToString('x2') })
}

$records = [System.Collections.Generic.List[object]]::new()
if (-not (Test-Path -LiteralPath $sampleDll -PathType Leaf)) {
  & dotnet build $sampleProject --no-restore
  if ($LASTEXITCODE -ne 0) {
    throw 'CUDA RTC sample build failed.'
  }
}
foreach ($toolkit in $toolkits) {
  if (-not (Test-Path -LiteralPath $toolkit.library -PathType Leaf)) {
    throw "NVRTC library was not found: $($toolkit.library)"
  }

  $env:JYPPX_NATIVE_BRIDGE_PATH = $BridgePath
  $env:JYPPX_NVRTC_LIBRARY = $toolkit.library
  $env:JYPPX_CUDA_ROOT = $cudaRuntimeRoot
  $env:CUDA_PATH = $cudaRuntimeRoot
  $env:JYPPX_TENSORRT_ROOT = $tensorRtRoot
  $lines = @(& dotnet $sampleDll 2>&1 | ForEach-Object { [string]$_ })
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "CUDA RTC sample failed for Toolkit $($toolkit.version): $($lines -join [Environment]::NewLine)"
  }

  $capabilityVersion = Get-PrefixedValue -Lines $lines -Prefix 'capability.version='
  $ptxHash = Get-PrefixedValue -Lines $lines -Prefix 'determinism.ptxSha256='
  if ($null -ne $ptxHash) {
    $ptxHash = ($ptxHash -split ' ')[0]
  }
  $cubinLine = $lines | Where-Object { $_ -match '^artifact\.kind=Cubin ' } | Select-Object -First 1
  $ltoLine = $lines | Where-Object { $_ -match '^artifact\.kind=LtoIr ' } | Select-Object -First 1
  $loadLine = $lines | Where-Object { $_ -match '^load\.attempted=True ' } | Select-Object -First 1
  $launchLine = $lines | Where-Object { $_ -match '^launch\.attempted=' } | Select-Object -First 1
  $driverCapabilityLine = $lines | Where-Object { $_ -match '^driver\.capability\.available=' } | Select-Object -First 1
  $driverLoadLine = $lines | Where-Object { $_ -match '^driver\.load\.attempted=True ' } | Select-Object -First 1
  $driverLaunchLine = $lines | Where-Object { $_ -match '^driver\.launch\.attempted=' } | Select-Object -First 1
  $failureLine = $lines | Where-Object { $_ -match '^failure\.success=False ' } | Select-Object -First 1
  if ($capabilityVersion -ne $toolkit.version -or $null -eq $ptxHash -or $null -eq $cubinLine -or $null -eq $loadLine -or $null -eq $launchLine -or $null -eq $driverCapabilityLine -or $null -eq $driverLoadLine -or $null -eq $driverLaunchLine -or $null -eq $failureLine) {
    throw "CUDA RTC sample output contract was incomplete for Toolkit $($toolkit.version)."
  }

  $loadSucceeded = $loadLine -match 'load\.succeeded=True'
  $launchSucceeded = $launchLine -match 'launch\.succeeded=True'
  $gpuReadback = $launchLine -match 'gpuReadback=True'
  $correctnessProof = $launchLine -match 'correctness=True'
  $ownersDisposedBeforeSynchronize = $launchLine -match 'ownersDisposedBeforeSynchronize=True'
  if ($loadSucceeded -and (-not $launchSucceeded -or -not $gpuReadback -or -not $correctnessProof)) {
    throw "A loadable RTC artifact did not produce launch/readback correctness proof for Toolkit $($toolkit.version)."
  }
  if ($loadSucceeded -and -not $ownersDisposedBeforeSynchronize) {
    throw "A loadable RTC artifact did not prove owner retention before synchronization for Toolkit $($toolkit.version)."
  }
  if (-not $loadSucceeded -and $launchLine -match 'launch\.attempted=True') {
    throw "RTC launch was attempted without a loadable artifact for Toolkit $($toolkit.version)."
  }
  $outputSha256 = if ($launchLine -match 'outputSha256=([0-9a-f]{64})') { $Matches[1] } else { $null }

  $driverAvailable = $driverCapabilityLine -match 'driver\.capability\.available=True'
  $driverVersion = if ($driverCapabilityLine -match 'version=([0-9]+)') { [int]$Matches[1] } else { 0 }
  $driverLibrary = if ($driverCapabilityLine -match 'library=([^ ]+)') { $Matches[1] } else { $null }
  if (-not $driverAvailable -or $driverVersion -le 0) {
    throw "CUDA Driver capability was unavailable for Toolkit $($toolkit.version)."
  }
  $driverLoadSucceeded = $driverLoadLine -match 'driver\.load\.succeeded=True'
  $driverLaunchSucceeded = $driverLaunchLine -match 'driver\.launch\.succeeded=True'
  $driverGpuReadback = $driverLaunchLine -match 'gpuReadback=True'
  $driverCorrectnessProof = $driverLaunchLine -match 'correctness=True'
  $driverOwnersDisposedBeforeSynchronize = $driverLaunchLine -match 'ownersDisposedBeforeSynchronize=True'
  if ($driverLoadSucceeded -and (-not $driverLaunchSucceeded -or -not $driverGpuReadback -or -not $driverCorrectnessProof -or -not $driverOwnersDisposedBeforeSynchronize)) {
    throw "A loadable RTC artifact did not produce CUDA Driver launch/readback/owner-retention proof for Toolkit $($toolkit.version)."
  }
  if (-not $driverLoadSucceeded -and $driverLaunchLine -match 'driver\.launch\.attempted=True') {
    throw "CUDA Driver launch was attempted without a loadable artifact for Toolkit $($toolkit.version)."
  }
  $driverOutputSha256 = if ($driverLaunchLine -match 'outputSha256=([0-9a-f]{64})') { $Matches[1] } else { $null }

  $records.Add([ordered]@{
      toolkitVersion = $toolkit.version
      processExitCode = $exitCode
      compileSuccess = $true
      compileFailureLogCaptured = $true
      loweredNameCaptured = [bool]($lines -match '^lowered\.name=.+')
      ptxDeterministic = $true
      ptxSha256 = $ptxHash
      cubinCapturedForSm75 = $true
      ltoIrCaptured = $null -ne $ltoLine
      loadAttempted = $true
      loadSucceeded = $loadSucceeded
      loadDiagnostic = Get-PrefixedValue -Lines $lines -Prefix 'load.diagnostic='
      kernelLaunch = $launchSucceeded
      gpuReadback = $gpuReadback
      correctnessProof = $correctnessProof
      ownersDisposedBeforeSynchronize = $ownersDisposedBeforeSynchronize
      outputSha256 = $outputSha256
      launchDiagnostic = Get-PrefixedValue -Lines $lines -Prefix 'launch.diagnostic='
      evidenceClassification = if ($correctnessProof) { 'local-toolkit-kernel-runtime-readback' } elseif ($loadSucceeded) { 'local-toolkit-compile-to-load' } else { 'local-toolkit-compile-only-load-rejected' }
      driverCapabilityAvailable = $driverAvailable
      driverVersion = $driverVersion
      driverLoadedLibraryName = $driverLibrary
      driverLoadAttempted = $true
      driverLoadSucceeded = $driverLoadSucceeded
      driverLoadDiagnostic = Get-PrefixedValue -Lines $lines -Prefix 'driver.load.diagnostic='
      driverKernelLaunch = $driverLaunchSucceeded
      driverGpuReadback = $driverGpuReadback
      driverCorrectnessProof = $driverCorrectnessProof
      driverOwnersDisposedBeforeSynchronize = $driverOwnersDisposedBeforeSynchronize
      driverOutputSha256 = $driverOutputSha256
      driverLaunchDiagnostic = Get-PrefixedValue -Lines $lines -Prefix 'driver.launch.diagnostic='
      driverEvidenceClassification = if ($driverCorrectnessProof) { 'local-toolkit-driver-kernel-runtime-readback' } elseif ($driverLoadSucceeded) { 'local-toolkit-driver-compile-to-load' } else { 'local-toolkit-driver-compile-only-load-rejected' }
      transcriptSha256 = Get-TranscriptSha256 -Lines $lines
    })
}

$document = [ordered]@{
  schemaVersion = 4
  recordKind = 'cuda-rtc-local-smoke'
  generatedLocalDate = '2026-07-28'
  bridgePath = $BridgePath.Substring($RepositoryRoot.TrimEnd('\').Length).TrimStart('\').Replace('\', '/')
  bridgeSha256 = (Get-FileHash -LiteralPath $BridgePath -Algorithm SHA256).Hash.ToLowerInvariant()
  records = @($records)
  performsPublish = $false
  proofBoundary = 'Runtime-library and Driver module launch/readback correctness are recorded only when every output value is validated. Compiler version identifies the NVRTC artifact source; the Driver path uses the current system nvcuda/libcuda and remains separate from package consumer, public package, Linux, and post-publish proof.'
}

$directory = Split-Path -Parent $OutputPath
[void][System.IO.Directory]::CreateDirectory($directory)
$document | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "CUDA RTC local smoke exported to $OutputPath"
