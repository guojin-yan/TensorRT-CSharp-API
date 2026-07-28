[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\capability-matrix.json'
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
    throw "Visual Studio dumpbin.exe was not found under '$installationPath'."
  }
  return $candidate.FullName
}

function Test-Token {
  param([string[]]$Lines, [string]$Token)
  return [bool]($Lines -match ('(?m)\b' + [regex]::Escape($Token) + '\b'))
}

$dumpbin = Resolve-DumpbinPath
$symbols = @(
  'nvrtcVersion',
  'nvrtcGetErrorString',
  'nvrtcCreateProgram',
  'nvrtcDestroyProgram',
  'nvrtcCompileProgram',
  'nvrtcGetProgramLogSize',
  'nvrtcGetProgramLog',
  'nvrtcGetPTXSize',
  'nvrtcGetPTX',
  'nvrtcGetCUBINSize',
  'nvrtcGetCUBIN',
  'nvrtcGetLTOIRSize',
  'nvrtcGetLTOIR',
  'nvrtcGetNVVMSize',
  'nvrtcGetNVVM',
  'nvrtcAddNameExpression',
  'nvrtcGetLoweredName'
)
$toolkits = @(
  [ordered]@{ version = '11.8'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8'; runtime = 'bin\nvrtc64_112_0.dll'; builtins = 'bin\nvrtc-builtins64_118.dll' },
  [ordered]@{ version = '12.1'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1'; runtime = 'bin\nvrtc64_120_0.dll'; builtins = 'bin\nvrtc-builtins64_121.dll' },
  [ordered]@{ version = '12.9'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'; runtime = 'bin\nvrtc64_120_0.dll'; builtins = 'bin\nvrtc-builtins64_129.dll' },
  [ordered]@{ version = '13.2'; root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'; runtime = 'bin\x64\nvrtc64_130_0.dll'; builtins = 'bin\x64\nvrtc-builtins64_132.dll' }
)

$windowsRows = [System.Collections.Generic.List[object]]::new()
foreach ($toolkit in $toolkits) {
  $headerPath = Join-Path $toolkit.root 'include\nvrtc.h'
  $libraryPath = Join-Path $toolkit.root 'lib\x64\nvrtc.lib'
  $runtimePath = Join-Path $toolkit.root $toolkit.runtime
  $builtinsPath = Join-Path $toolkit.root $toolkit.builtins
  foreach ($path in @($headerPath, $libraryPath, $runtimePath, $builtinsPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      throw "Required local CUDA RTC audit asset is missing: $path"
    }
  }

  $headerLines = @(Get-Content -LiteralPath $headerPath)
  $exportLines = @(& $dumpbin /nologo /exports $runtimePath 2>&1)
  if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /exports failed for '$runtimePath'."
  }
  $importLines = @(& $dumpbin /nologo /linkermember:1 $libraryPath 2>&1)
  if ($LASTEXITCODE -ne 0) {
    throw "dumpbin /linkermember:1 failed for '$libraryPath'."
  }

  $symbolRows = [ordered]@{}
  foreach ($symbol in $symbols) {
    $symbolRows[$symbol] = [ordered]@{
      headerDeclared = Test-Token -Lines $headerLines -Token $symbol
      importLibraryMember = Test-Token -Lines $importLines -Token $symbol
      runtimeExported = Test-Token -Lines $exportLines -Token $symbol
    }
  }

  $runtimeFile = Get-Item -LiteralPath $runtimePath
  $builtinsFile = Get-Item -LiteralPath $builtinsPath
  $windowsRows.Add([ordered]@{
      toolkitVersion = $toolkit.version
      platform = 'windows'
      architecture = 'x64'
      evidenceState = 'local-header-import-lib-dll-export-verified'
      headerRelativePath = 'include/nvrtc.h'
      importLibraryRelativePath = 'lib/x64/nvrtc.lib'
      runtimeRelativePath = $toolkit.runtime.Replace('\', '/')
      runtimeSizeBytes = $runtimeFile.Length
      runtimeSha256 = (Get-FileHash -LiteralPath $runtimePath -Algorithm SHA256).Hash.ToLowerInvariant()
      builtinsRelativePath = $toolkit.builtins.Replace('\', '/')
      builtinsSizeBytes = $builtinsFile.Length
      builtinsSha256 = (Get-FileHash -LiteralPath $builtinsPath -Algorithm SHA256).Hash.ToLowerInvariant()
      symbols = $symbolRows
      capabilities = [ordered]@{
        compileLog = $symbolRows.nvrtcGetProgramLogSize.runtimeExported -and $symbolRows.nvrtcGetProgramLog.runtimeExported
        ptx = $symbolRows.nvrtcGetPTXSize.runtimeExported -and $symbolRows.nvrtcGetPTX.runtimeExported
        cubin = $symbolRows.nvrtcGetCUBINSize.runtimeExported -and $symbolRows.nvrtcGetCUBIN.runtimeExported
        ltoIr = $symbolRows.nvrtcGetLTOIRSize.runtimeExported -and $symbolRows.nvrtcGetLTOIR.runtimeExported
        deprecatedNvvm = $symbolRows.nvrtcGetNVVMSize.runtimeExported -and $symbolRows.nvrtcGetNVVM.runtimeExported
        nameExpressions = $symbolRows.nvrtcAddNameExpression.runtimeExported -and $symbolRows.nvrtcGetLoweredName.runtimeExported
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
    localLibNvrtcAssetCount = 0
    soname = $null
    symbols = $null
    capabilities = $null
    boundary = 'No local Linux .so or package payload was found on the audited Windows host. Loader candidates are implementation probes, not SONAME proof.'
  }
}

$document = [ordered]@{
  schemaVersion = 1
  recordKind = 'cuda-rtc-vendor-capability-matrix'
  auditedLocalDate = '2026-07-28'
  performsDownload = $false
  performsPublish = $false
  windows = @($windowsRows)
  linux = @($linuxRows)
  loaderContract = [ordered]@{
    dependencyMode = 'optional-dynamic'
    exactOverrideEnvironmentVariable = 'JYPPX_NVRTC_LIBRARY'
    windowsCandidates = @('nvrtc64_130_0.dll', 'nvrtc64_120_0.dll', 'nvrtc64_112_0.dll')
    linuxCandidates = @('libnvrtc.so', 'libnvrtc.so.13', 'libnvrtc.so.12', 'libnvrtc.so.11.2')
    coreBridgeStaticNvrtcLink = $false
  }
  packagingContract = [ordered]@{
    bridgeOnlyBundlesNvrtc = $false
    bridgeOnlyConsumerInstallsToolkit = $true
    fullRuntimeRequiresNvrtcAndMatchingBuiltins = $true
    redistributionApprovalState = 'pending-owner-and-license-review'
  }
  proofBoundary = [ordered]@{
    vendorAuditIsRuntimeProof = $false
    compileOnlyIsKernelRuntimeProof = $false
    artifactHashIsCorrectnessProof = $false
    localWindowsIsLinuxProof = $false
  }
}

$outputDirectory = Split-Path -Parent $OutputPath
[void][System.IO.Directory]::CreateDirectory($outputDirectory)
$document | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "CUDA RTC capability matrix exported to $OutputPath"
