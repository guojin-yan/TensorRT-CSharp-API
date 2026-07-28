[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\driver-capability-matrix.json'
}

function Resolve-DumpbinPath {
  $command = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($null -ne $command) {
    return $command.Source
  }

  $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'dumpbin.exe was not found on PATH and vswhere.exe is unavailable.'
  }

  $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  $candidate = Get-ChildItem -LiteralPath (Join-Path $installationPath 'VC\Tools\MSVC') -Recurse -Filter dumpbin.exe -File |
    Where-Object FullName -match '[\\/]bin[\\/]Hostx64[\\/]x64[\\/]dumpbin\.exe$' |
    Sort-Object FullName -Descending |
    Select-Object -First 1
  if ($null -eq $candidate) {
    throw "dumpbin.exe was not found under '$installationPath'."
  }
  return $candidate.FullName
}

function Test-Token {
  param([string[]]$Lines, [string]$Token)
  return [bool]($Lines -match ('(?m)\b' + [regex]::Escape($Token) + '\b'))
}

$dumpbin = Resolve-DumpbinPath
$driverPath = 'C:\Windows\System32\nvcuda.dll'
$driverExists = Test-Path -LiteralPath $driverPath -PathType Leaf
$driverExportLines = @()
if ($driverExists) {
  $driverExportLines = @(& $dumpbin /nologo /exports $driverPath 2>&1)
  if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /exports failed for '$driverPath'."
  }
}

$symbols = @(
  'cuInit',
  'cuDriverGetVersion',
  'cuGetErrorName',
  'cuGetErrorString',
  'cuDeviceGet',
  'cuDevicePrimaryCtxRetain',
  'cuDevicePrimaryCtxRelease_v2',
  'cuCtxPushCurrent_v2',
  'cuCtxPopCurrent_v2',
  'cuModuleLoadDataEx',
  'cuModuleUnload',
  'cuModuleGetFunction',
  'cuLaunchKernel',
  'cuStreamSynchronize',
  'cuEventCreate',
  'cuEventRecord',
  'cuEventQuery',
  'cuEventSynchronize',
  'cuEventDestroy_v2'
)

$toolkits = @(
  [ordered]@{ version = '11.8'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8' },
  [ordered]@{ version = '12.1'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1' },
  [ordered]@{ version = '12.9'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9' },
  [ordered]@{ version = '13.2'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2' }
)

$rows = [System.Collections.Generic.List[object]]::new()
foreach ($toolkit in $toolkits) {
  $headerPath = Join-Path $toolkit.root 'include\cuda.h'
  $libraryPath = Join-Path $toolkit.root 'lib\x64\cuda.lib'
  $headerLines = if (Test-Path -LiteralPath $headerPath -PathType Leaf) { @(Get-Content -LiteralPath $headerPath) } else { @() }
  $importLines = if (Test-Path -LiteralPath $libraryPath -PathType Leaf) { @(& $dumpbin /nologo /linkermember:1 $libraryPath 2>&1) } else { @() }
  if ((Test-Path -LiteralPath $libraryPath -PathType Leaf) -and $LASTEXITCODE -ne 0) {
    throw "dumpbin /linkermember:1 failed for '$libraryPath'."
  }

  $symbolRows = [ordered]@{}
  foreach ($symbol in $symbols) {
    $symbolRows[$symbol] = [ordered]@{
      headerDeclared = Test-Token -Lines $headerLines -Token $symbol
      importLibraryMember = Test-Token -Lines $importLines -Token $symbol
      driverExported = Test-Token -Lines $driverExportLines -Token $symbol
    }
  }

  $allHeader = @($symbolRows.Values | Where-Object { -not $_.headerDeclared }).Count -eq 0
  $allImport = @($symbolRows.Values | Where-Object { -not $_.importLibraryMember }).Count -eq 0
  $allExport = $driverExists -and @($symbolRows.Values | Where-Object { -not $_.driverExported }).Count -eq 0
  $rows.Add([ordered]@{
      toolkitVersion = $toolkit.version
      platform = 'windows'
      architecture = 'x64'
      evidenceState = if ($allHeader -and $allImport -and $allExport) { 'local-header-import-lib-driver-export-verified' } else { 'local-assets-partially-verified' }
      headerRelativePath = 'include/cuda.h'
      importLibraryRelativePath = 'lib/x64/cuda.lib'
      driverLibraryPath = if ($driverExists) { $driverPath } else { $null }
      driverLibrarySha256 = if ($driverExists) { (Get-FileHash -LiteralPath $driverPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
      symbols = $symbolRows
      capabilities = [ordered]@{
        moduleLoad = $symbolRows.cuModuleLoadDataEx.headerDeclared -and $symbolRows.cuModuleLoadDataEx.importLibraryMember -and $symbolRows.cuModuleLoadDataEx.driverExported
        functionLookup = $symbolRows.cuModuleGetFunction.headerDeclared -and $symbolRows.cuModuleGetFunction.importLibraryMember -and $symbolRows.cuModuleGetFunction.driverExported
        typedLaunch = $symbolRows.cuLaunchKernel.headerDeclared -and $symbolRows.cuLaunchKernel.importLibraryMember -and $symbolRows.cuLaunchKernel.driverExported
        failureCleanupSynchronization = $symbolRows.cuStreamSynchronize.headerDeclared -and $symbolRows.cuStreamSynchronize.importLibraryMember -and $symbolRows.cuStreamSynchronize.driverExported
        contextInterop = $symbolRows.cuCtxPushCurrent_v2.headerDeclared -and $symbolRows.cuCtxPushCurrent_v2.importLibraryMember -and $symbolRows.cuCtxPushCurrent_v2.driverExported -and $symbolRows.cuCtxPopCurrent_v2.headerDeclared -and $symbolRows.cuCtxPopCurrent_v2.importLibraryMember -and $symbolRows.cuCtxPopCurrent_v2.driverExported
        completionEvent = $symbolRows.cuEventCreate.headerDeclared -and $symbolRows.cuEventCreate.importLibraryMember -and $symbolRows.cuEventCreate.driverExported -and $symbolRows.cuEventRecord.headerDeclared -and $symbolRows.cuEventRecord.importLibraryMember -and $symbolRows.cuEventRecord.driverExported -and $symbolRows.cuEventQuery.headerDeclared -and $symbolRows.cuEventQuery.importLibraryMember -and $symbolRows.cuEventQuery.driverExported -and $symbolRows.cuEventSynchronize.headerDeclared -and $symbolRows.cuEventSynchronize.importLibraryMember -and $symbolRows.cuEventSynchronize.driverExported -and $symbolRows.cuEventDestroy_v2.headerDeclared -and $symbolRows.cuEventDestroy_v2.importLibraryMember -and $symbolRows.cuEventDestroy_v2.driverExported
      }
    })
}

$linuxRows = foreach ($version in @('11.8', '12.1', '12.9', '13.2')) {
  [ordered]@{
    toolkitVersion = $version
    platform = 'linux'
    architecture = 'x64'
    evidenceState = 'unverified-local-assets-not-found'
    localAssetSearchRoots = @('E:/', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA')
    libcudaAssets = @()
    soname = $null
    symbols = $null
    capabilities = $null
    boundary = 'No local Linux libcuda.so or package payload was found on the audited Windows host. Loader candidates are implementation probes, not SONAME proof.'
  }
}

$document = [ordered]@{
  schemaVersion = 1
  recordKind = 'cuda-driver-vendor-capability-matrix'
  auditedLocalDate = (Get-Date -Format 'yyyy-MM-dd')
  performsDownload = $false
  performsPublish = $false
  windows = @($rows)
  linux = @($linuxRows)
  loaderContract = [ordered]@{
    dependencyMode = 'optional-dynamic'
    exactOverrideEnvironmentVariable = 'JYPPX_CUDA_DRIVER_LIBRARY'
    windowsCandidates = @('nvcuda.dll')
    linuxCandidates = @('libcuda.so.1', 'libcuda.so')
    coreBridgeStaticDriverLink = $false
  }
  ownerContract = [ordered]@{
    moduleRetainsPrimaryContext = $true
    borrowedFunctionStaysInsideBridge = $true
    launchRetainsManagedModuleStreamAndMemory = $true
    publicSurfaceExposesRawPointer = $false
  }
  proofBoundary = [ordered]@{
    vendorAuditIsRuntimeProof = $false
    localWindowsDriverIsLinuxProof = $false
    compileToLaunchRequiresSeparateGpuSmoke = $true
  }
}

$outputDirectory = Split-Path -Parent $OutputPath
[void][System.IO.Directory]::CreateDirectory($outputDirectory)
$document | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "CUDA Driver capability matrix exported to $OutputPath"
