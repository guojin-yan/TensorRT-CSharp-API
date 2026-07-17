[CmdletBinding()]
param(
  [string]$RepositoryRoot = "",
  [ValidateSet("8", "10", "11")]
  [string[]]$TensorRtLines = @("8", "10", "11"),
  [string]$BridgePath = "",
  [string]$DumpbinPath = "",
  [string]$OutputPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}

$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "artifacts/native-abi/tensorrt-native-abi-surface.json"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot $OutputPath
}

function Resolve-DumpbinPath {
  param([string]$RequestedPath)

  if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
    return (Resolve-Path -LiteralPath $RequestedPath).Path
  }

  $command = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($null -ne $command) {
    return $command.Source
  }

  $programFilesX86 = [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86)
  $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio/Installer/vswhere.exe"
  if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw "dumpbin.exe was not found on PATH and vswhere.exe is unavailable."
  }

  $installationPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($installationPath)) {
    throw "Visual Studio C++ tools were not found."
  }

  $candidate = Get-ChildItem -LiteralPath (Join-Path $installationPath "VC/Tools/MSVC") -Recurse -Filter dumpbin.exe -File |
    Where-Object { $_.FullName -match '[\\/]bin[\\/]Hostx64[\\/]x64[\\/]dumpbin\.exe$' } |
    Sort-Object FullName -Descending |
    Select-Object -First 1
  if ($null -eq $candidate) {
    throw "Visual Studio dumpbin.exe was not found under '$installationPath'."
  }

  return $candidate.FullName
}

function Get-ManifestEntryPoints {
  param([string]$Line)

  $manifestRoot = Join-Path $RepositoryRoot "native/manifests/tensorrt/v$Line"
  $entryPoints = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
  foreach ($manifestPath in Get-ChildItem -LiteralPath $manifestRoot -Filter *.manifest.json -File | Sort-Object FullName) {
    $manifest = Get-Content -LiteralPath $manifestPath.FullName -Raw | ConvertFrom-Json
    foreach ($api in @($manifest.apis)) {
      $entryPoint = [string]$api.entryPoint
      if ([string]::IsNullOrWhiteSpace($entryPoint)) {
        throw "Manifest '$($manifestPath.FullName)' contains an API without an entryPoint."
      }

      [void]$entryPoints.Add($entryPoint)
    }
  }

  return @($entryPoints | Sort-Object)
}

$lineSummaries = [System.Collections.Generic.List[object]]::new()
$missingDeclarations = [System.Collections.Generic.List[object]]::new()
$entryPointsByLine = @{}

foreach ($line in @($TensorRtLines | Sort-Object -Unique)) {
  $headerPath = Join-Path $RepositoryRoot "native/include/jyppx/tensorrt/trt$line.h"
  $header = [System.IO.File]::ReadAllText($headerPath)
  $entryPoints = @(Get-ManifestEntryPoints -Line $line)
  $entryPointsByLine[$line] = $entryPoints

  foreach ($entryPoint in $entryPoints) {
    $escapedEntryPoint = [regex]::Escape($entryPoint)
    $explicitPattern = 'JYPPX_C_API\s*\(\s*JYPPX_StatusCode\s*\)\s+' + $escapedEntryPoint + '\s*\('
    $macroPattern = '(?m)^\s*[A-Z][A-Z0-9_]*_DECL\s*\(\s*' + $escapedEntryPoint + '\s*\)\s*;?\s*$'
    $declared =
      [regex]::IsMatch($header, $explicitPattern, [System.Text.RegularExpressions.RegexOptions]::CultureInvariant) -or
      [regex]::IsMatch($header, $macroPattern, [System.Text.RegularExpressions.RegexOptions]::CultureInvariant)
    if (-not $declared) {
      $missingDeclarations.Add([ordered]@{
          tensorRtLine = $line
          entryPoint = $entryPoint
          header = [System.IO.Path]::GetRelativePath($RepositoryRoot, $headerPath).Replace('\', '/')
        })
    }
  }

  $lineSummaries.Add([ordered]@{
      tensorRtLine = $line
      manifestEntryPointCount = $entryPoints.Count
      declaredEntryPointCount = $entryPoints.Count - @($missingDeclarations | Where-Object tensorRtLine -EQ $line).Count
      missingDeclarationCount = @($missingDeclarations | Where-Object tensorRtLine -EQ $line).Count
    })
}

$exportSummary = $null
$missingExports = [System.Collections.Generic.List[string]]::new()
if (-not [string]::IsNullOrWhiteSpace($BridgePath)) {
  $selectedLines = @($TensorRtLines | Sort-Object -Unique)
  if ($selectedLines.Count -ne 1) {
    throw "-BridgePath requires exactly one -TensorRtLines value."
  }

  $resolvedBridgePath = (Resolve-Path -LiteralPath $BridgePath).Path
  $resolvedDumpbinPath = Resolve-DumpbinPath -RequestedPath $DumpbinPath
  $dumpbinOutput = @(& $resolvedDumpbinPath /nologo /exports $resolvedBridgePath 2>&1)
  if ($LASTEXITCODE -ne 0) {
    throw "dumpbin export inspection failed for '$resolvedBridgePath': $($dumpbinOutput -join [Environment]::NewLine)"
  }

  $exportedNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
  foreach ($lineText in $dumpbinOutput) {
    if ([string]$lineText -match '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)\s*$') {
      [void]$exportedNames.Add($Matches[1])
    }
  }

  $selectedLine = $selectedLines[0]
  foreach ($entryPoint in @($entryPointsByLine[$selectedLine])) {
    if (-not $exportedNames.Contains($entryPoint)) {
      $missingExports.Add($entryPoint)
    }
  }

  $exportSummary = [ordered]@{
    tensorRtLine = $selectedLine
    bridgePath = $resolvedBridgePath
    dumpbinPath = $resolvedDumpbinPath
    discoveredExportCount = $exportedNames.Count
    expectedManifestExportCount = @($entryPointsByLine[$selectedLine]).Count
    matchedManifestExportCount = @($entryPointsByLine[$selectedLine]).Count - $missingExports.Count
    missingExportCount = $missingExports.Count
  }
}

$failed = $missingDeclarations.Count -ne 0 -or $missingExports.Count -ne 0
$state = if ($failed) { "tensorrt-native-abi-surface-failed" } else { "tensorrt-native-abi-surface-ready" }
$result = [ordered]@{
  recordKind = "tensorrt-native-abi-surface"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("o")
  state = $state
  tensorRtLines = @($TensorRtLines | Sort-Object -Unique)
  lineSummaries = @($lineSummaries)
  missingDeclarationCount = $missingDeclarations.Count
  missingDeclarations = @($missingDeclarations)
  exportVerificationRequested = -not [string]::IsNullOrWhiteSpace($BridgePath)
  exportSummary = $exportSummary
  missingExportCount = $missingExports.Count
  missingExports = @($missingExports)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "This gate validates manifest/header declarations and optional PE exports only. It does not publish packages, execute inference, or promote runtime/post-publish proof."
}

$outputDirectory = Split-Path -Parent $OutputPath
[void][System.IO.Directory]::CreateDirectory($outputDirectory)
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8

$markdownPath = [System.IO.Path]::ChangeExtension($OutputPath, ".md")
$markdown = [System.Collections.Generic.List[string]]::new()
$markdown.Add("# TensorRT Native ABI Surface")
$markdown.Add("")
$markdown.Add("State: ``$state``")
$markdown.Add("")
$markdown.Add("| TensorRT line | Manifest entry points | Header declarations | Missing declarations |")
$markdown.Add("| --- | ---: | ---: | ---: |")
foreach ($summary in $lineSummaries) {
  $markdown.Add("| TRT$($summary.tensorRtLine) | $($summary.manifestEntryPointCount) | $($summary.declaredEntryPointCount) | $($summary.missingDeclarationCount) |")
}
$markdown.Add("")
if ($null -ne $exportSummary) {
  $markdown.Add("PE export verification: TRT$($exportSummary.tensorRtLine), matched $($exportSummary.matchedManifestExportCount)/$($exportSummary.expectedManifestExportCount), missing $($exportSummary.missingExportCount).")
} else {
  $markdown.Add("PE export verification was not requested; this record contains source declaration parity only.")
}
$markdown.Add("")
$markdown.Add("This is ABI surface evidence only; it is not runtime execution proof, post-publish proof, or authorization to publish.")
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "TensorRtNativeAbiSurfaceState=$state MissingDeclarations=$($missingDeclarations.Count) MissingExports=$($missingExports.Count)"
Write-Output "Json=$OutputPath"
Write-Output "Markdown=$markdownPath"

if ($failed) {
  $details = @($missingDeclarations | ForEach-Object { "TRT$($_.tensorRtLine):$($_.entryPoint)" }) + @($missingExports | ForEach-Object { "export:$_" })
  throw "TensorRT native ABI surface validation failed: $($details -join ', ')"
}
