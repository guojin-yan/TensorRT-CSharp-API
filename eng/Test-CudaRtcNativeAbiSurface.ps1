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
  $OutputPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\native-abi-surface.json'
}

function Resolve-DumpbinPath {
  $command = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($null -ne $command) {
    return $command.Source
  }
  $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
  $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  return (Get-ChildItem -LiteralPath (Join-Path $installationPath 'VC\Tools\MSVC') -Recurse -Filter dumpbin.exe -File |
      Where-Object FullName -match '[\\/]bin[\\/]Hostx64[\\/]x64[\\/]dumpbin\.exe$' |
      Sort-Object FullName -Descending |
      Select-Object -First 1 -ExpandProperty FullName)
}

$manifestPath = Join-Path $RepositoryRoot 'native\manifests\cuda\cuda-sixty-third-batch-runtime-compilation-owner.manifest.json'
$headerPath = Join-Path $RepositoryRoot 'native\include\jyppx\cuda\runtime.h'
$BridgePath = (Resolve-Path -LiteralPath $BridgePath).Path
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$header = [System.IO.File]::ReadAllText($headerPath)
$entryPoints = @($manifest.apis | ForEach-Object { [string]$_.entryPoint } | Sort-Object -Unique)
$missingDeclarations = [System.Collections.Generic.List[string]]::new()
foreach ($entryPoint in $entryPoints) {
  $pattern = 'JYPPX_C_API\s*\(\s*JYPPX_StatusCode\s*\)\s+' + [regex]::Escape($entryPoint) + '\s*\('
  if (-not [regex]::IsMatch($header, $pattern)) {
    $missingDeclarations.Add($entryPoint)
  }
}

$dumpbin = Resolve-DumpbinPath
$exportLines = @(& $dumpbin /nologo /exports $BridgePath 2>&1)
if ($LASTEXITCODE -ne 0) {
  throw "dumpbin /exports failed for '$BridgePath'."
}
$exports = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($line in $exportLines) {
  if ([string]$line -match '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$') {
    [void]$exports.Add($Matches[1])
  }
}
$missingExports = [System.Collections.Generic.List[string]]::new()
foreach ($entryPoint in $entryPoints) {
  if (-not $exports.Contains($entryPoint)) {
    $missingExports.Add($entryPoint)
  }
}

$document = [ordered]@{
  schemaVersion = 1
  recordKind = 'cuda-rtc-native-abi-surface'
  generatedLocalDate = '2026-07-28'
  manifest = 'native/manifests/cuda/cuda-sixty-third-batch-runtime-compilation-owner.manifest.json'
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
  proofBoundary = 'This checks RTC manifest/header/PE export parity only. It is not compiler, kernel runtime, package, Linux, or post-publish proof.'
}

$directory = Split-Path -Parent $OutputPath
[void][System.IO.Directory]::CreateDirectory($directory)
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
if (-not $document.passed) {
  throw "CUDA RTC native ABI surface failed: declarations=$($missingDeclarations.Count), exports=$($missingExports.Count)."
}
Write-Host "CUDA RTC native ABI surface passed: $($entryPoints.Count)/$($entryPoints.Count) declarations and PE exports."
