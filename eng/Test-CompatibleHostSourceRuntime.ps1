[CmdletBinding()]
param(
  [ValidateSet("8", "10", "11")]
  [string]$TensorRtLine = "10",
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [string]$Preset = "win-x64-trt10-cuda11-release",
  [string]$OutputDirectory = "artifacts\real-case\tensorrt10-compatible-host-source-runtime",
  [string]$ScreenshotPath = "docs\images\tensorrt10-compatible-host-source-runtime-terminal.png",
  [string]$RepositoryRoot,
  [switch]$SkipConfigure,
  [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
  $RepositoryRoot = (Resolve-Path $RepositoryRoot).Path
}

$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RequiredDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$RequiredChild
  )

  if ([string]::IsNullOrWhiteSpace($Path)) {
    throw "$Name must be provided explicitly."
  }

  $resolved = [IO.Path]::GetFullPath($Path)
  if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
    throw "$Name does not exist: $resolved"
  }

  if (-not (Test-Path -LiteralPath (Join-Path $resolved $RequiredChild) -PathType Leaf)) {
    throw "$Name is missing $RequiredChild under: $resolved"
  }

  return $resolved
}

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $resolved = if ([IO.Path]::IsPathRooted($Path)) {
    [IO.Path]::GetFullPath($Path)
  }
  else {
    [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
  }

  $rootPrefix = $RepositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  if (-not $resolved.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Evidence output must remain inside the repository: $resolved"
  }

  return $resolved
}

function Get-RepositoryRelativePath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $resolved = [IO.Path]::GetFullPath($Path)
  $rootPrefix = $RepositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  if (-not $resolved.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Path is outside the repository: $resolved"
  }

  return $resolved.Substring($rootPrefix.Length).Replace('\', '/')
}

function Invoke-CapturedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [Parameter(Mandatory = $true)][string]$Description
  )

  $lines = New-Object Collections.Generic.List[string]
  & $FilePath @ArgumentList 2>&1 | ForEach-Object {
    $line = $_.ToString()
    $lines.Add($line)
    Write-Host $line
  }
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "$Description failed with exit code $exitCode."
  }

  return @($lines)
}

function Get-CMakeCacheValue {
  param(
    [Parameter(Mandatory = $true)][string]$CachePath,
    [Parameter(Mandatory = $true)][string]$Name
  )

  $pattern = '^' + [Regex]::Escape($Name) + ':[^=]*=(.*)$'
  foreach ($line in Get-Content -LiteralPath $CachePath -Encoding utf8) {
    $match = [Regex]::Match($line, $pattern)
    if ($match.Success) {
      return $match.Groups[1].Value
    }
  }

  throw "CMake cache entry was not found: $Name"
}

function Test-PathInside {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Root
  )

  $resolvedPath = [IO.Path]::GetFullPath($Path)
  $rootPrefix = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  return $resolvedPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Get-ArtifactRecord {
  param([Parameter(Mandatory = $true)][string]$Path)

  $item = Get-Item -LiteralPath $Path
  return [ordered]@{
    path = Get-RepositoryRelativePath $item.FullName
    length = [int64]$item.Length
    sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

function Split-TerminalLine {
  param(
    [Parameter(Mandatory = $true)][string]$Line,
    [int]$Width = 118
  )

  $remaining = $Line
  $parts = New-Object Collections.Generic.List[string]
  while ($remaining.Length -gt $Width) {
    $parts.Add($remaining.Substring(0, $Width))
    $remaining = "  " + $remaining.Substring($Width)
  }
  $parts.Add($remaining)
  return @($parts)
}

function Write-TerminalScreenshot {
  param(
    [Parameter(Mandatory = $true)][string[]]$Lines,
    [Parameter(Mandatory = $true)][string]$Path
  )

  Add-Type -AssemblyName System.Drawing
  $renderLines = New-Object Collections.Generic.List[string]
  $renderLines.Add('TensorRtSharp4.0 compatible-host source runtime')
  $renderLines.Add('')
  foreach ($line in $Lines) {
    foreach ($part in Split-TerminalLine -Line $line) {
      $renderLines.Add($part)
    }
  }

  $width = 1500
  $lineHeight = 23
  $height = [Math]::Max(320, 54 + ($renderLines.Count * $lineHeight))
  $bitmap = [Drawing.Bitmap]::new($width, $height)
  $graphics = [Drawing.Graphics]::FromImage($bitmap)
  $background = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(15, 24, 31))
  $foreground = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(226, 233, 238))
  $accent = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(91, 211, 164))
  $muted = [Drawing.SolidBrush]::new([Drawing.Color]::FromArgb(133, 151, 162))
  $font = [Drawing.Font]::new('Consolas', 14, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
  $titleFont = [Drawing.Font]::new('Consolas', 15, [Drawing.FontStyle]::Bold, [Drawing.GraphicsUnit]::Pixel)
  try {
    $graphics.FillRectangle($background, 0, 0, $width, $height)
    $graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    for ($index = 0; $index -lt $renderLines.Count; $index++) {
      $line = $renderLines[$index]
      $brush = if ($index -eq 0) {
        $accent
      }
      elseif ($line.IndexOf('Passed=True', [StringComparison]::Ordinal) -ge 0 -or $line -match '=True:') {
        $accent
      }
      elseif ($line.StartsWith('Boundary=', [StringComparison]::Ordinal)) {
        $muted
      }
      else {
        $foreground
      }
      $selectedFont = if ($index -eq 0) { $titleFont } else { $font }
      $graphics.DrawString($line, $selectedFont, $brush, 24, 22 + ($index * $lineHeight))
    }

    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $bitmap.Save($Path, [Drawing.Imaging.ImageFormat]::Png)
  }
  finally {
    $titleFont.Dispose()
    $font.Dispose()
    $muted.Dispose()
    $accent.Dispose()
    $foreground.Dispose()
    $background.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
  }
}

$resolvedTensorRtRoot = Resolve-RequiredDirectory -Name "TensorRtRoot" -Path $TensorRtRoot -RequiredChild "include\NvInfer.h"
$resolvedCudaRoot = Resolve-RequiredDirectory -Name "CudaRoot" -Path $CudaRoot -RequiredChild "bin\nvcc.exe"
$resolvedOutputDirectory = Resolve-RepositoryPath $OutputDirectory
$resolvedScreenshotPath = Resolve-RepositoryPath $ScreenshotPath
New-Item -ItemType Directory -Path $resolvedOutputDirectory -Force | Out-Null

$configureArguments = @(
  "--preset", $Preset,
  "-DJYPPX_TENSORRT_ROOT=$resolvedTensorRtRoot",
  "-DJYPPX_CUDA_ROOT=$resolvedCudaRoot"
)
if (-not $SkipConfigure.IsPresent) {
  Invoke-CapturedCommand -FilePath "cmake" -ArgumentList $configureArguments -Description "CMake configure" | Out-Null
}

$buildDirectory = Join-Path $RepositoryRoot ("build-out\" + $Preset)
$cachePath = Join-Path $buildDirectory "CMakeCache.txt"
if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf)) {
  throw "CMake cache was not found for preset '$Preset'."
}

$libraryCacheNames = @(
  "TensorRT_NVINFER_LIBRARY",
  "TensorRT_NVINFER_PLUGIN_LIBRARY",
  "TensorRT_NVONNXPARSER_LIBRARY"
)
$importLibraries = New-Object Collections.Generic.List[object]
foreach ($cacheName in $libraryCacheNames) {
  $libraryPath = Get-CMakeCacheValue -CachePath $cachePath -Name $cacheName
  if (-not (Test-Path -LiteralPath $libraryPath -PathType Leaf)) {
    throw "$cacheName does not point to an existing file: $libraryPath"
  }
  if (-not (Test-PathInside -Path $libraryPath -Root $resolvedTensorRtRoot)) {
    throw "$cacheName is outside the selected TensorRT root: $libraryPath"
  }
  $importLibraries.Add([ordered]@{
      cacheName = $cacheName
      fileName = [IO.Path]::GetFileName($libraryPath)
      sha256 = (Get-FileHash -LiteralPath $libraryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}

if (-not $SkipBuild.IsPresent) {
  Invoke-CapturedCommand -FilePath "cmake" -ArgumentList @("--build", "--preset", $Preset, "--parallel") -Description "CMake build" | Out-Null
}

$bridgePath = Join-Path $buildDirectory "bin\Release\jyppxtrtbridge.dll"
if (-not (Test-Path -LiteralPath $bridgePath -PathType Leaf)) {
  throw "Bridge output was not found: $bridgePath"
}

$previousBridgePath = $env:JYPPX_NATIVE_BRIDGE_PATH
$previousTensorRtRoot = $env:JYPPX_TENSORRT_ROOT
$previousCudaRoot = $env:JYPPX_CUDA_ROOT
$previousDevelopmentProbing = $env:JYPPX_ENABLE_DEVELOPMENT_PROBING
$previousPath = $env:PATH
try {
  $env:JYPPX_NATIVE_BRIDGE_PATH = $bridgePath
  $env:JYPPX_TENSORRT_ROOT = $resolvedTensorRtRoot
  $env:JYPPX_CUDA_ROOT = $resolvedCudaRoot
  $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "true"
  $runtimeDirectories = @(
    (Join-Path $resolvedTensorRtRoot "lib"),
    (Join-Path $resolvedTensorRtRoot "bin"),
    (Join-Path $resolvedCudaRoot "bin")
  ) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
  $env:PATH = (($runtimeDirectories + @($previousPath)) -join [IO.Path]::PathSeparator)

  $smokeProject = Join-Path $RepositoryRoot "smoke\TensorRtSmokeRunner\TensorRtSmokeRunner.csproj"
  $smokeLines = @(Invoke-CapturedCommand -FilePath "dotnet" -ArgumentList @(
      "run", "--project", $smokeProject, "-c", "Release", "--", "--tensor-rt-line", $TensorRtLine
    ) -Description "TensorRtSmokeRunner")
}
finally {
  $env:JYPPX_NATIVE_BRIDGE_PATH = $previousBridgePath
  $env:JYPPX_TENSORRT_ROOT = $previousTensorRtRoot
  $env:JYPPX_CUDA_ROOT = $previousCudaRoot
  $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $previousDevelopmentProbing
  $env:PATH = $previousPath
}

$smokeText = $smokeLines -join [Environment]::NewLine
$environmentMatch = [Regex]::Match($smokeText, 'Bridge=\S+ TRT=([0-9]+\.[0-9]+\.[0-9]+) CUDA=([0-9]+\.[0-9]+)')
if (-not $environmentMatch.Success) {
  throw "TensorRtSmokeRunner did not report the expected bridge environment line."
}

$requiredMarkers = @(
  "TryCreateRuntime$TensorRtLine=True:",
  "TryCreateBuilder$TensorRtLine=True:",
  "TryBuildSerializedNetwork$TensorRtLine=True:",
  "TryRunMinimalBuildChain$TensorRtLine=True:",
  "HighLevelChain$TensorRtLine=True:"
)
$missingMarkers = @($requiredMarkers | Where-Object { $smokeText.IndexOf($_, [StringComparison]::Ordinal) -lt 0 })
if ($missingMarkers.Count -gt 0) {
  throw "TensorRtSmokeRunner missed required success markers: $($missingMarkers -join ', ')"
}

$highLevelLine = @($smokeLines | Where-Object { $_.StartsWith("HighLevelChain$TensorRtLine=True:", [StringComparison]::Ordinal) })[0]
$enqueueCompleted = $highLevelLine.IndexOf("Enqueue=True", [StringComparison]::Ordinal) -ge 0
if (-not $enqueueCompleted) {
  throw "The high-level TensorRT chain did not report Enqueue=True."
}

$tensorRtVersion = $environmentMatch.Groups[1].Value
$cudaVersion = $environmentMatch.Groups[2].Value
$exactPackageMatrix = $false
$summaryLines = @(
  "CompatibleHostSourceRuntime Passed=True",
  "ImportLibrariesInsideTensorRtRoot=True",
  "TensorRtLine=$TensorRtLine TensorRtVersion=$tensorRtVersion CudaVersion=$cudaVersion",
  "RuntimeCreate=True BuilderCreate=True SerializedNetwork=True MinimalBuildChain=True HighLevelChain=True Enqueue=True",
  "Boundary=SourceRuntimeOnly ExactPackageMatrix=$exactPackageMatrix PackageConsumer=False PerformsPublish=False"
)
foreach ($line in $summaryLines) {
  Write-Host $line
}

$transcriptPath = Join-Path $resolvedOutputDirectory "tensorrt-smoke-transcript.txt"
$transcriptLines = @($smokeLines + $summaryLines)
[IO.File]::WriteAllText($transcriptPath, (($transcriptLines -join [Environment]::NewLine) + [Environment]::NewLine), $utf8)

$screenshotLines = @($transcriptLines | Where-Object {
    $_.StartsWith("TensorRtSmokeRunner ", [StringComparison]::Ordinal) -or
    $_.StartsWith("Bridge=", [StringComparison]::Ordinal) -or
    $_.StartsWith("TRT$TensorRtLine Vendor=", [StringComparison]::Ordinal) -or
    $_.StartsWith("TryCreateRuntime$TensorRtLine=", [StringComparison]::Ordinal) -or
    $_.StartsWith("TryCreateBuilder$TensorRtLine=", [StringComparison]::Ordinal) -or
    $_.StartsWith("TryBuildSerializedNetwork$TensorRtLine=", [StringComparison]::Ordinal) -or
    $_.StartsWith("TryRunMinimalBuildChain$TensorRtLine=", [StringComparison]::Ordinal) -or
    $_.StartsWith("HighLevelChain$TensorRtLine=", [StringComparison]::Ordinal) -or
    $_.StartsWith("CompatibleHostSourceRuntime ", [StringComparison]::Ordinal) -or
    $_.StartsWith("ImportLibrariesInsideTensorRtRoot=", [StringComparison]::Ordinal) -or
    $_.StartsWith("Boundary=", [StringComparison]::Ordinal)
  })
Write-TerminalScreenshot -Lines $screenshotLines -Path $resolvedScreenshotPath

$gpuName = "unavailable"
$driverVersion = "unavailable"
try {
  $gpuLine = @(& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)[0]
  if (-not [string]::IsNullOrWhiteSpace($gpuLine)) {
    $gpuParts = $gpuLine -split ',', 2
    $gpuName = $gpuParts[0].Trim()
    if ($gpuParts.Count -gt 1) {
      $driverVersion = $gpuParts[1].Trim()
    }
  }
}
catch {
  Write-Warning "nvidia-smi host metadata was unavailable: $($_.Exception.Message)"
}

$sourceCommit = (& git -C $RepositoryRoot rev-parse HEAD).Trim()
$record = [ordered]@{
  recordKind = "compatible-host-source-runtime"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "passed-compatible-host-source-runtime"
  sourceCommit = $sourceCommit
  preset = $Preset
  tensorRtLine = [int]$TensorRtLine
  tensorRtVersion = $tensorRtVersion
  cudaToolkitVersion = $cudaVersion
  tensorRtSdkLabel = Split-Path -Leaf $resolvedTensorRtRoot
  cudaSdkLabel = Split-Path -Leaf $resolvedCudaRoot
  host = [ordered]@{
    operatingSystem = [Environment]::OSVersion.VersionString
    gpuName = $gpuName
    driverVersion = $driverVersion
  }
  build = [ordered]@{
    configured = -not $SkipConfigure.IsPresent
    built = -not $SkipBuild.IsPresent
    importLibrariesInsideSelectedTensorRtRoot = $true
    importLibraries = @($importLibraries | ForEach-Object { $_ })
    bridgeFileName = [IO.Path]::GetFileName($bridgePath)
    bridgeSha256 = (Get-FileHash -LiteralPath $bridgePath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  runtime = [ordered]@{
    runtimeCreated = $true
    builderCreated = $true
    serializedNetworkBuilt = $true
    minimalBuildChainCompleted = $true
    highLevelChainCompleted = $true
    enqueueCompleted = $true
    requiredMarkers = $requiredMarkers
    missingMarkers = @()
  }
  model = [ordered]@{
    name = "in-process TensorRT identity network"
    acquisition = "not-applicable"
    conversion = "not-applicable"
    onnxRequired = $false
    externalModelRequired = $false
    outerModelsDirectoryUsed = $false
  }
  artifacts = [ordered]@{
    transcript = Get-ArtifactRecord $transcriptPath
    screenshot = Get-ArtifactRecord $resolvedScreenshotPath
  }
  boundary = [ordered]@{
    isCurrentSourceBridgeRuntimeEvidence = $true
    isSameMajorCompatibleHostEvidence = $true
    isExactPackageMatrixEvidence = $false
    isPackageConsumerRuntimeProof = $false
    isPublicPackageProof = $false
    isReleaseProof = $false
    performsPack = $false
    performsPublish = $false
    createsTag = $false
    createsRelease = $false
    statement = "This proves the current source bridge on one TensorRT same-major compatible host. It does not prove an exact release package key, a package consumer, a public package, a release, or permission to publish."
  }
}

$jsonPath = Join-Path $resolvedOutputDirectory "compatible-host-source-runtime-evidence.json"
[IO.File]::WriteAllText($jsonPath, (($record | ConvertTo-Json -Depth 12) + [Environment]::NewLine), $utf8)

$markdown = @"
# TensorRT Compatible-Host Source Runtime Evidence

| Field | Value |
|---|---|
| validation state | ``$($record.validationState)`` |
| source commit | ``$sourceCommit`` |
| preset | ``$Preset`` |
| TensorRT | ``$tensorRtVersion`` |
| CUDA | ``$cudaVersion`` |
| GPU | ``$gpuName`` |
| import libraries inside selected root | ``true`` |
| runtime / builder | ``true / true`` |
| serialized network / minimal chain | ``true / true`` |
| high-level chain / enqueue | ``true / true`` |
| exact package matrix evidence | ``false`` |
| package consumer proof | ``false`` |
| performs publish | ``false`` |

## Artifacts

- Transcript: ``$((Get-RepositoryRelativePath $transcriptPath))``
- Screenshot: ``$((Get-RepositoryRelativePath $resolvedScreenshotPath))``
- Bridge SHA256: ``$($record.build.bridgeSha256)``

## Boundary

$($record.boundary.statement)
"@
$markdownPath = Join-Path $resolvedOutputDirectory "compatible-host-source-runtime-evidence.md"
[IO.File]::WriteAllText($markdownPath, ($markdown + [Environment]::NewLine), $utf8)

Write-Host "Compatible-host source runtime evidence written:"
Write-Host "  Transcript=$(Get-RepositoryRelativePath $transcriptPath)"
Write-Host "  Evidence=$(Get-RepositoryRelativePath $jsonPath)"
Write-Host "  Screenshot=$(Get-RepositoryRelativePath $resolvedScreenshotPath)"
Write-Host "CompatibleHostSourceRuntimeEvidence Passed=True PerformsPublish=False"
