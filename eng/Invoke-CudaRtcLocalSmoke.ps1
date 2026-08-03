[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$BridgePath,
  [string]$CudaRuntimeRoot,
  [string]$TensorRtRoot,
  [string[]]$CudaToolkitRoots = @(),
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

if ([string]::IsNullOrWhiteSpace($CudaRuntimeRoot)) {
  $CudaRuntimeRoot = if (-not [string]::IsNullOrWhiteSpace($env:JYPPX_CUDA_ROOT)) { $env:JYPPX_CUDA_ROOT } else { $env:CUDA_PATH }
}
if ([string]::IsNullOrWhiteSpace($TensorRtRoot)) {
  $TensorRtRoot = if (-not [string]::IsNullOrWhiteSpace($env:JYPPX_TENSORRT_ROOT)) { $env:JYPPX_TENSORRT_ROOT } else { $env:TENSORRT_ROOT }
}
if ([string]::IsNullOrWhiteSpace($CudaRuntimeRoot) -or -not (Test-Path -LiteralPath $CudaRuntimeRoot -PathType Container)) {
  throw 'CudaRuntimeRoot must point to a user-installed CUDA Toolkit. Pass -CudaRuntimeRoot or set JYPPX_CUDA_ROOT.'
}
if ([string]::IsNullOrWhiteSpace($TensorRtRoot) -or -not (Test-Path -LiteralPath $TensorRtRoot -PathType Container)) {
  throw 'TensorRtRoot must point to a user-installed TensorRT SDK/runtime. Pass -TensorRtRoot or set JYPPX_TENSORRT_ROOT.'
}
$CudaRuntimeRoot = (Resolve-Path -LiteralPath $CudaRuntimeRoot).Path
$TensorRtRoot = (Resolve-Path -LiteralPath $TensorRtRoot).Path

if ($CudaToolkitRoots.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($env:JYPPX_CUDA_TOOLKIT_ROOTS)) {
  $CudaToolkitRoots = @($env:JYPPX_CUDA_TOOLKIT_ROOTS -split [IO.Path]::PathSeparator | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}
if ($CudaToolkitRoots.Count -eq 0) {
  $CudaToolkitRoots = @($CudaRuntimeRoot)
}

$toolkits = @(
  foreach ($rootValue in $CudaToolkitRoots) {
    if (-not (Test-Path -LiteralPath $rootValue -PathType Container)) {
      throw "CUDA Toolkit root was not found: $rootValue"
    }

    $root = (Resolve-Path -LiteralPath $rootValue).Path
    $version = $null
    $versionJsonPath = Join-Path $root 'version.json'
    if (Test-Path -LiteralPath $versionJsonPath -PathType Leaf) {
      $versionDocument = Get-Content -LiteralPath $versionJsonPath -Raw -Encoding utf8 | ConvertFrom-Json
      if ($null -ne $versionDocument.cuda -and [string]$versionDocument.cuda.version -match '^(\d+\.\d+)') {
        $version = $Matches[1]
      }
    }
    if ([string]::IsNullOrWhiteSpace($version) -and (Split-Path -Leaf $root) -match 'v?(\d+\.\d+)') {
      $version = $Matches[1]
    }
    if ([string]::IsNullOrWhiteSpace($version)) {
      throw "Unable to determine the CUDA Toolkit major.minor version from: $root"
    }

    $library = @(
      Get-ChildItem -Path (Join-Path $root 'bin\nvrtc64_*.dll') -File -ErrorAction SilentlyContinue
      Get-ChildItem -Path (Join-Path $root 'bin\x64\nvrtc64_*.dll') -File -ErrorAction SilentlyContinue
    ) | Where-Object { $_.Name -notlike '*.alt.dll' } | Sort-Object FullName -Unique | Select-Object -First 1
    if ($null -eq $library) {
      throw "NVRTC library was not found under user-installed CUDA Toolkit root: $root"
    }

    [ordered]@{ version = $version; root = $root; library = $library.FullName }
  }
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
  $env:JYPPX_CUDA_ROOT = $CudaRuntimeRoot
  $env:CUDA_PATH = $CudaRuntimeRoot
  $env:JYPPX_TENSORRT_ROOT = $TensorRtRoot
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
  generatedLocalDate = (Get-Date -Format 'yyyy-MM-dd')
  bridgePath = [IO.Path]::GetRelativePath($RepositoryRoot, $BridgePath).Replace('\', '/')
  bridgeSha256 = (Get-FileHash -LiteralPath $BridgePath -Algorithm SHA256).Hash.ToLowerInvariant()
  records = @($records)
  performsPublish = $false
  proofBoundary = 'Runtime-library and Driver module launch/readback correctness are recorded only when every output value is validated. Compiler version identifies the NVRTC artifact source; the Driver path uses the current system nvcuda/libcuda and remains separate from package consumer, public package, Linux, and post-publish proof.'
}

$directory = Split-Path -Parent $OutputPath
[void][System.IO.Directory]::CreateDirectory($directory)
$document | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "CUDA RTC local smoke exported to $OutputPath"
