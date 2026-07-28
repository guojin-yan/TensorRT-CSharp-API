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
  $OutputPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\driver-native-abi-surface.json'
}

function Resolve-DumpbinPath {
  $command = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($null -ne $command) { return $command.Source }
  $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
  $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  return (Get-ChildItem -LiteralPath (Join-Path $installationPath 'VC\Tools\MSVC') -Recurse -Filter dumpbin.exe -File |
      Where-Object FullName -match '[\\/]bin[\\/]Hostx64[\\/]x64[\\/]dumpbin\.exe$' |
      Sort-Object FullName -Descending |
      Select-Object -First 1 -ExpandProperty FullName)
}

$manifestRelativePath = 'native/manifests/cuda/cuda-sixty-fifth-batch-driver-module-owner.manifest.json'
$manifestPath = Join-Path $RepositoryRoot $manifestRelativePath
$headerPath = Join-Path $RepositoryRoot 'native\include\jyppx\cuda\runtime.h'
$BridgePath = (Resolve-Path -LiteralPath $BridgePath).Path
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$header = [System.IO.File]::ReadAllText($headerPath)
$entryPoints = @($manifest.apis | ForEach-Object { [string]$_.entryPoint } | Sort-Object -Unique)
$missingDeclarations = @($entryPoints | Where-Object {
    $pattern = 'JYPPX_C_API\s*\(\s*JYPPX_StatusCode\s*\)\s+' + [regex]::Escape($_) + '\s*\('
    -not [regex]::IsMatch($header, $pattern)
  })

$dumpbin = Resolve-DumpbinPath
$exportLines = @(& $dumpbin /nologo /exports $BridgePath 2>&1)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /exports failed for '$BridgePath'." }
$exports = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($line in $exportLines) {
  if ([string]$line -match '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$') {
    [void]$exports.Add($Matches[1])
  }
}
$missingExports = @($entryPoints | Where-Object { -not $exports.Contains($_) })

$document = [ordered]@{
  schemaVersion = 1
  recordKind = 'cuda-driver-module-native-abi-surface'
  generatedLocalDate = (Get-Date -Format 'yyyy-MM-dd')
  manifest = $manifestRelativePath
  header = 'native/include/jyppx/cuda/runtime.h'
  manifestEntryPointCount = $entryPoints.Count
  declaredEntryPointCount = $entryPoints.Count - $missingDeclarations.Count
  missingDeclarationCount = $missingDeclarations.Count
  missingDeclarations = @($missingDeclarations)
  bridge = $BridgePath.Substring($RepositoryRoot.TrimEnd('\').Length).TrimStart('\').Replace('\', '/')
  discoveredPeExportCount = $exports.Count
  matchedPeExportCount = $entryPoints.Count - $missingExports.Count
  missingPeExportCount = $missingExports.Count
  missingPeExports = @($missingExports)
  passed = $missingDeclarations.Count -eq 0 -and $missingExports.Count -eq 0
  performsPublish = $false
  proofBoundary = 'This checks CUDA Driver owner manifest/header/PE export parity only. Vendor symbols and runtime correctness are recorded by separate capability and local smoke artifacts.'
}

$directory = Split-Path -Parent $OutputPath
[void][System.IO.Directory]::CreateDirectory($directory)
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
if (-not $document.passed) {
  throw "CUDA Driver native ABI surface failed: declarations=$($missingDeclarations.Count), exports=$($missingExports.Count)."
}
Write-Host "CUDA Driver native ABI surface passed: $($entryPoints.Count)/$($entryPoints.Count) declarations and PE exports."
