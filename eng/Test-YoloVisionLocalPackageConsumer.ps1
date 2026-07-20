[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$ManagedPackageDirectory,
  [string]$YoloVisionPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$BridgePackageId,
  [ValidateSet("8", "10", "11")][string]$TensorRtLine,
  [string]$ModelPath,
  [string]$LabelsPath,
  [string]$ImagePath,
  [string]$TensorRtRoot,
  [string]$TensorRtRuntimeRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
  [string]$PackageVersion = "4.0.0",
  [switch]$KeepWorkspace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$outerRoot = [IO.Path]::GetFullPath((Split-Path -Parent $RepositoryRoot))
$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Resolve-PathValue {
  param(
    [string]$Value,
    [string]$DefaultValue,
    [string]$RelativeRoot
  )

  $candidate = if ([string]::IsNullOrWhiteSpace($Value)) { $DefaultValue } else { $Value }
  if ([IO.Path]::IsPathRooted($candidate)) {
    return [IO.Path]::GetFullPath($candidate)
  }

  return [IO.Path]::GetFullPath((Join-Path $RelativeRoot $candidate))
}

function Assert-PathUnderRoot {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Description
  )

  $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  if (-not $fullPath.StartsWith($fullRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "$Description must remain under '$fullRoot': $fullPath"
  }
}

function Assert-NonCDrivePath {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Description
  )

  $fullPath = [IO.Path]::GetFullPath($Path)
  if ([IO.Path]::GetPathRoot($fullPath).TrimEnd('\') -ieq "C:") {
    throw "$Description must not use the C drive: $fullPath"
  }
}

function ConvertTo-XmlAttributeValue {
  param([Parameter(Mandatory = $true)][string]$Value)
  return [Security.SecurityElement]::Escape($Value)
}

function Get-NupkgMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)

  $zip = [IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = $zip.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
    if ($null -eq $nuspec) {
      throw "Package does not contain a nuspec: $Path"
    }

    $stream = $nuspec.Open()
    try {
      $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
      try {
        [xml]$xml = $reader.ReadToEnd()
      }
      finally {
        $reader.Dispose()
      }
    }
    finally {
      $stream.Dispose()
    }

    $namespaceManager = [Xml.XmlNamespaceManager]::new($xml.NameTable)
    $namespaceManager.AddNamespace("n", $xml.package.NamespaceURI)
    return [pscustomobject]@{
      Path = [IO.Path]::GetFullPath($Path)
      Id = $xml.SelectSingleNode("//n:metadata/n:id", $namespaceManager).InnerText
      Version = $xml.SelectSingleNode("//n:metadata/n:version", $namespaceManager).InnerText
      Length = (Get-Item -LiteralPath $Path).Length
      Sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
  }
  finally {
    $zip.Dispose()
  }
}

function Find-Package {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$PackageId,
    [Parameter(Mandatory = $true)][string]$ExpectedVersion
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }

  $matches = @(
    foreach ($package in Get-ChildItem -LiteralPath $Directory -Filter *.nupkg -File) {
      $metadata = Get-NupkgMetadata -Path $package.FullName
      if ([string]::Equals($metadata.Id, $PackageId, [StringComparison]::Ordinal) -and
          [string]::Equals($metadata.Version, $ExpectedVersion, [StringComparison]::Ordinal)) {
        $metadata
      }
    }
  )
  if ($matches.Count -eq 0) {
    throw "Package '$PackageId' version '$ExpectedVersion' was not found under $Directory."
  }

  return @($matches | Sort-Object Version, Path -Descending)[0]
}

function Test-PackageEntry {
  param(
    [Parameter(Mandatory = $true)][string]$PackagePath,
    [Parameter(Mandatory = $true)][string]$ExpectedEntry
  )

  $normalizedExpected = $ExpectedEntry.Replace('\', '/')
  $zip = [IO.Compression.ZipFile]::OpenRead($PackagePath)
  try {
    return @($zip.Entries | Where-Object {
      [string]::Equals($_.FullName.Replace('\', '/'), $normalizedExpected, [StringComparison]::OrdinalIgnoreCase)
    }).Count -eq 1
  }
  finally {
    $zip.Dispose()
  }
}

function Invoke-CapturedProcess {
  param(
    [Parameter(Mandatory = $true)][string]$FileName,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [hashtable]$Environment = @{}
  )

  $startInfo = [Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $FileName
  $startInfo.WorkingDirectory = $WorkingDirectory
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.CreateNoWindow = $true
  foreach ($argument in $Arguments) {
    $startInfo.ArgumentList.Add($argument)
  }
  foreach ($name in $Environment.Keys) {
    $startInfo.Environment[$name] = [string]$Environment[$name]
  }

  $process = [Diagnostics.Process]::new()
  $process.StartInfo = $startInfo
  if (-not $process.Start()) {
    throw "Failed to start process: $FileName"
  }

  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  $process.WaitForExit()
  $stdout = $stdoutTask.GetAwaiter().GetResult()
  $stderr = $stderrTask.GetAwaiter().GetResult()
  $exitCode = $process.ExitCode
  $process.Dispose()
  return [pscustomobject]@{
    ExitCode = $exitCode
    Stdout = $stdout
    Stderr = $stderr
  }
}

function Invoke-CheckedDotNet {
  param(
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$LogPath
  )

  $result = Invoke-CapturedProcess -FileName "dotnet" -Arguments $Arguments -WorkingDirectory $WorkingDirectory
  ($result.Stdout + $result.Stderr) | Set-Content -LiteralPath $LogPath -Encoding utf8
  if ($result.ExitCode -ne 0) {
    throw "dotnet $($Arguments -join ' ') failed with exit code $($result.ExitCode). See $LogPath"
  }

  return $result
}

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeManifestEntries = @($splitManifest.packages | Where-Object {
  [string]::Equals([string]$_.sourceRuntimeKey, $RuntimePackageKey, [StringComparison]::OrdinalIgnoreCase) -and
  [string]::Equals([string]$_.role, "bridge", [StringComparison]::OrdinalIgnoreCase)
})
if ($bridgeManifestEntries.Count -ne 1) {
  throw "Runtime package key '$RuntimePackageKey' must resolve to exactly one bridge package in $splitManifestPath. Found $($bridgeManifestEntries.Count)."
}

$bridgeManifest = $bridgeManifestEntries[0]
$expectedBridgePackageId = [string]$bridgeManifest.packageId
$expectedTensorRtLine = [string]$bridgeManifest.tensorRtLine
if (-not [string]::Equals([string]$bridgeManifest.platform, "windows", [StringComparison]::OrdinalIgnoreCase)) {
  throw "YoloVision local package consumer currently requires a Windows runtime package key: $RuntimePackageKey"
}
if (-not [string]::IsNullOrWhiteSpace($BridgePackageId) -and
    -not [string]::Equals($BridgePackageId, $expectedBridgePackageId, [StringComparison]::Ordinal)) {
  throw "Bridge package id '$BridgePackageId' does not match runtime key '$RuntimePackageKey' manifest id '$expectedBridgePackageId'."
}
if (-not [string]::IsNullOrWhiteSpace($TensorRtLine) -and
    -not [string]::Equals($TensorRtLine, $expectedTensorRtLine, [StringComparison]::Ordinal)) {
  throw "TensorRT line '$TensorRtLine' does not match runtime key '$RuntimePackageKey' manifest line '$expectedTensorRtLine'."
}
$BridgePackageId = $expectedBridgePackageId
$TensorRtLine = $expectedTensorRtLine

$resolvedRuntimeRoots = $null
if ([string]::IsNullOrWhiteSpace($TensorRtRoot) -or
    [string]::IsNullOrWhiteSpace($CudaRoot) -or
    [string]::IsNullOrWhiteSpace($CudnnRoot)) {
  $runtimeRootsJson = (& pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $RuntimePackageKey -RepositoryRoot $RepositoryRoot | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve vendor roots for runtime package key '$RuntimePackageKey'."
  }
  $resolvedRuntimeRoots = $runtimeRootsJson | ConvertFrom-Json
}
$defaultTensorRtRoot = if ($null -eq $resolvedRuntimeRoots) { "" } else { [string]$resolvedRuntimeRoots.tensorRtRoot }
$defaultCudaRoot = if ($null -eq $resolvedRuntimeRoots) { "" } else { [string]$resolvedRuntimeRoots.cudaRoot }
$defaultCudnnRoot = if ($null -eq $resolvedRuntimeRoots) { "" } else { [string]$resolvedRuntimeRoots.cudnnRoot }

$OutputRoot = Resolve-PathValue -Value $OutputRoot -DefaultValue (Join-Path $outerRoot "consumer-workspaces\yolovision-yolox-local-package-trt$TensorRtLine") -RelativeRoot $outerRoot
$ReportDirectory = Resolve-PathValue -Value $ReportDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\yolovision\yolox-local-package-consumer\$RuntimePackageKey") -RelativeRoot $RepositoryRoot
$ManagedPackageDirectory = Resolve-PathValue -Value $ManagedPackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\managed") -RelativeRoot $RepositoryRoot
$YoloVisionPackageDirectory = Resolve-PathValue -Value $YoloVisionPackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\yolovision-nupkg") -RelativeRoot $RepositoryRoot
$BridgePackageDirectory = Resolve-PathValue -Value $BridgePackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$RuntimePackageKey") -RelativeRoot $RepositoryRoot
$ModelPath = Resolve-PathValue -Value $ModelPath -DefaultValue (Join-Path $outerRoot "downloads\yolox-apache\source\yolox_s.onnx") -RelativeRoot $outerRoot
$LabelsPath = Resolve-PathValue -Value $LabelsPath -DefaultValue (Join-Path $outerRoot "downloads\yolox-apache\derived\coco.names") -RelativeRoot $outerRoot
$ImagePath = Resolve-PathValue -Value $ImagePath -DefaultValue (Join-Path $outerRoot "downloads\yolox-apache\derived\dog.ppm") -RelativeRoot $outerRoot
$TensorRtRoot = Resolve-PathValue -Value $TensorRtRoot -DefaultValue $defaultTensorRtRoot -RelativeRoot $outerRoot
$TensorRtRuntimeRoot = Resolve-PathValue -Value $TensorRtRuntimeRoot -DefaultValue $TensorRtRoot -RelativeRoot $outerRoot
$CudaRoot = Resolve-PathValue -Value $CudaRoot -DefaultValue $defaultCudaRoot -RelativeRoot $outerRoot
$CudnnRoot = Resolve-PathValue -Value $CudnnRoot -DefaultValue $defaultCudnnRoot -RelativeRoot $outerRoot

Assert-PathUnderRoot -Path $OutputRoot -Root $outerRoot -Description "Consumer workspace"
Assert-PathUnderRoot -Path $ReportDirectory -Root $outerRoot -Description "Consumer report directory"
foreach ($item in @(
  @{ Path = $OutputRoot; Description = "Consumer workspace" },
  @{ Path = $ReportDirectory; Description = "Consumer report directory" },
  @{ Path = $ManagedPackageDirectory; Description = "Managed package feed" },
  @{ Path = $YoloVisionPackageDirectory; Description = "YoloVision package feed" },
  @{ Path = $BridgePackageDirectory; Description = "Bridge package feed" },
  @{ Path = $ModelPath; Description = "YOLOX model" },
  @{ Path = $LabelsPath; Description = "YOLOX labels" },
  @{ Path = $ImagePath; Description = "YOLOX image" }
)) {
  Assert-NonCDrivePath -Path $item.Path -Description $item.Description
}

foreach ($requiredPath in @($ModelPath, $LabelsPath, $ImagePath, $TensorRtRoot, $TensorRtRuntimeRoot, $CudaRoot, $CudnnRoot)) {
  if (-not (Test-Path -LiteralPath $requiredPath)) {
    throw "Required local dependency does not exist: $requiredPath"
  }
}

$managedPackage = Find-Package -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API" -ExpectedVersion $PackageVersion
$yoloVisionPackage = Find-Package -Directory $YoloVisionPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API.YoloVision" -ExpectedVersion $PackageVersion
$bridgePackage = Find-Package -Directory $BridgePackageDirectory -PackageId $BridgePackageId -ExpectedVersion $PackageVersion
if (-not (Test-PackageEntry -PackagePath $managedPackage.Path -ExpectedEntry "lib/net8.0/JYPPX.TensorRtSharp.dll")) {
  throw "Managed package does not contain the net8.0 TensorRT assembly."
}
if (-not (Test-PackageEntry -PackagePath $yoloVisionPackage.Path -ExpectedEntry "lib/net8.0/YoloVision.dll")) {
  throw "YoloVision package does not contain lib/net8.0/YoloVision.dll."
}
if (-not (Test-PackageEntry -PackagePath $bridgePackage.Path -ExpectedEntry "runtimes/win-x64/native/jyppxtrtbridge.dll")) {
  throw "Bridge package does not contain the win-x64 native bridge."
}

if (Test-Path -LiteralPath $OutputRoot) {
  $resolvedExistingOutput = (Resolve-Path -LiteralPath $OutputRoot).Path
  Assert-PathUnderRoot -Path $resolvedExistingOutput -Root $outerRoot -Description "Existing consumer workspace"
  Remove-Item -LiteralPath $resolvedExistingOutput -Recurse -Force
}
New-Item -ItemType Directory -Path $OutputRoot, $ReportDirectory -Force | Out-Null

$workspace = Join-Path $OutputRoot "workspace"
$packageCache = Join-Path $OutputRoot "packages"
$runOutput = Join-Path $OutputRoot "run-output"
New-Item -ItemType Directory -Path $workspace, $packageCache, $runOutput -Force | Out-Null
$templateRoot = Join-Path $RepositoryRoot "samples\YoloVision.PackageConsumer"
$consumerProjectPath = Join-Path $workspace "YoloVision.PackageConsumer.csproj"
$consumerProgramPath = Join-Path $workspace "Program.cs"
Copy-Item -LiteralPath (Join-Path $templateRoot "Program.cs") -Destination $consumerProgramPath
$projectTemplate = Get-Content -LiteralPath (Join-Path $templateRoot "YoloVision.PackageConsumer.csproj.template") -Raw -Encoding utf8
$projectContent = $projectTemplate.Replace("__MANAGED_PACKAGE_VERSION__", $managedPackage.Version)
$projectContent = $projectContent.Replace("__YOLOVISION_PACKAGE_VERSION__", $yoloVisionPackage.Version)
$projectContent = $projectContent.Replace("__BRIDGE_PACKAGE_ID__", $bridgePackage.Id)
$projectContent = $projectContent.Replace("__BRIDGE_PACKAGE_VERSION__", $bridgePackage.Version)
[IO.File]::WriteAllText($consumerProjectPath, $projectContent, $utf8)
if ($projectContent.Contains("ProjectReference", [StringComparison]::OrdinalIgnoreCase)) {
  throw "Consumer project must not contain ProjectReference."
}

$nugetConfigPath = Join-Path $workspace "NuGet.config"
$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="jyppx-managed-local" value="$(ConvertTo-XmlAttributeValue -Value $ManagedPackageDirectory)" />
    <add key="jyppx-yolovision-local" value="$(ConvertTo-XmlAttributeValue -Value $YoloVisionPackageDirectory)" />
    <add key="jyppx-bridge-local" value="$(ConvertTo-XmlAttributeValue -Value $BridgePackageDirectory)" />
  </packageSources>
</configuration>
"@
[IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)

$restoreLogPath = Join-Path $ReportDirectory "restore.log"
$buildLogPath = Join-Path $ReportDirectory "build.log"
$restoreResult = Invoke-CheckedDotNet -Arguments @("restore", $consumerProjectPath, "--configfile", $nugetConfigPath, "--packages", $packageCache, "--force", "--no-cache", "--verbosity", "minimal") -WorkingDirectory $workspace -LogPath $restoreLogPath
$buildResult = Invoke-CheckedDotNet -Arguments @("build", $consumerProjectPath, "-c", "Release", "--no-restore", "--verbosity", "minimal") -WorkingDirectory $workspace -LogPath $buildLogPath

$assetsPath = Join-Path $workspace "obj\project.assets.json"
$assets = Get-Content -LiteralPath $assetsPath -Raw -Encoding utf8 | ConvertFrom-Json
$projectLibraryCount = @($assets.libraries.PSObject.Properties | Where-Object { $_.Value.type -eq "project" }).Count
if ($projectLibraryCount -ne 0) {
  throw "Consumer restore graph contains project libraries."
}

$consumerOutputDirectory = Join-Path $workspace "bin\Release\net8.0"
$consumerAssemblyPath = Join-Path $consumerOutputDirectory "YoloVision.PackageConsumer.dll"
$nativeBridgePaths = @(Get-ChildItem -LiteralPath $consumerOutputDirectory -Recurse -Filter jyppxtrtbridge.dll -File)
if ($nativeBridgePaths.Count -ne 1) {
  throw "Expected one copied native bridge in consumer output, found $($nativeBridgePaths.Count)."
}

$tensorPath = Join-Path $runOutput "dog-yolox-s.fp32.bin"
$outputJsonPath = Join-Path $runOutput "yolovision-output.json"
$visualizationPath = Join-Path $runOutput "yolovision-output.svg"
$runArguments = @(
  $consumerAssemblyPath,
  "--model", $ModelPath,
  "--labels", $LabelsPath,
  "--image", $ImagePath,
  "--preprocessed-output", $tensorPath,
  "--output-json", $outputJsonPath,
  "--visualization", $visualizationPath,
  "--input-shape", "1x3x640x640",
  "--tensor-rt-line", $TensorRtLine,
  "--family", "yolox",
  "--task", "det",
  "--layout", "boxes-first",
  "--has-objectness", "true",
  "--nms-mode", "class-aware",
  "--confidence", "0.3",
  "--iou-threshold", "0.45",
  "--top-k", "20"
)
$nativePathEntries = @(
  $consumerOutputDirectory,
  $nativeBridgePaths[0].DirectoryName,
  (Join-Path $TensorRtRuntimeRoot "bin"),
  (Join-Path $TensorRtRuntimeRoot "lib"),
  (Join-Path $TensorRtRoot "bin"),
  (Join-Path $TensorRtRoot "lib"),
  (Join-Path $CudaRoot "bin"),
  $CudnnRoot,
  (Join-Path $CudnnRoot "bin")
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_) }
$cudnnDllDirectories = @(Get-ChildItem -LiteralPath $CudnnRoot -Recurse -File -Filter *.dll -ErrorAction SilentlyContinue | Select-Object -ExpandProperty DirectoryName -Unique)
$pathEntries = @($nativePathEntries + $cudnnDllDirectories + $env:PATH)
$runEnvironment = @{
  PATH = ($pathEntries -join ';')
  NUGET_PACKAGES = $packageCache
}
$runResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $runArguments -WorkingDirectory $workspace -Environment $runEnvironment
$stdoutPath = Join-Path $ReportDirectory "runtime.stdout.log"
$stderrPath = Join-Path $ReportDirectory "runtime.stderr.log"
[IO.File]::WriteAllText($stdoutPath, $runResult.Stdout, $utf8)
[IO.File]::WriteAllText($stderrPath, $runResult.Stderr, $utf8)
Write-Host $runResult.Stdout
if (-not [string]::IsNullOrWhiteSpace($runResult.Stderr)) {
  Write-Warning $runResult.Stderr
}
if ($runResult.ExitCode -ne 0) {
  throw "YoloVision package consumer exited with code $($runResult.ExitCode)."
}
if (-not $runResult.Stdout.Contains("YoloVisionPackageConsumer ProjectReference=False", [StringComparison]::Ordinal) -or
    -not $runResult.Stdout.Contains("YoloVision Passed=True", [StringComparison]::Ordinal)) {
  throw "YoloVision package consumer did not emit the required package/runtime success markers."
}

if ($runResult.Stdout -notmatch 'YoloVisionPackageConsumer BridgeTensorRt=(?<trt>\S+) BridgeCuda=(?<cuda>\S+)') {
  throw "YoloVision package consumer did not emit bridge build version metadata."
}
$bridgeTensorRtVersion = $Matches.trt
$bridgeCudaToolkitVersion = $Matches.cuda
if ($bridgeTensorRtVersion -notmatch '^(?<major>[0-9]+)' -or
    -not [string]::Equals($Matches.major, $TensorRtLine, [StringComparison]::Ordinal)) {
  throw "Bridge TensorRT build version '$bridgeTensorRtVersion' does not match requested TensorRT line '$TensorRtLine'."
}

$predictionLines = @($runResult.Stdout -split "`r?`n" | Where-Object { $_.StartsWith("Detection Class=", [StringComparison]::Ordinal) })
$predictions = @(
  foreach ($line in $predictionLines) {
    if ($line -match '^Detection Class=(?<class>.+?) Score=(?<score>[0-9.]+) ') {
      [pscustomobject]@{
        className = $Matches.class
        score = [double]::Parse($Matches.score, [Globalization.CultureInfo]::InvariantCulture)
        line = $line
      }
    }
  }
)
if ($predictions.Count -eq 0) {
  throw "YoloVision package consumer did not produce detections."
}
$elapsedMilliseconds = 0.0
if ($runResult.Stdout -match 'Execution .* ElapsedMs=(?<elapsed>[0-9.]+)') {
  $elapsedMilliseconds = [double]::Parse($Matches.elapsed, [Globalization.CultureInfo]::InvariantCulture)
}

foreach ($file in @($tensorPath, $outputJsonPath, $visualizationPath)) {
  if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
    throw "Expected runtime output was not created: $file"
  }
}
$yoloOutputReport = Get-Content -LiteralPath $outputJsonPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([int]$yoloOutputReport.runtime.tensorRtLine -ne [int]$TensorRtLine) {
  throw "YoloVision output report TensorRT line '$($yoloOutputReport.runtime.tensorRtLine)' does not match requested line '$TensorRtLine'."
}
$copiedOutputJson = Join-Path $ReportDirectory "yolovision-output.json"
$copiedVisualization = Join-Path $ReportDirectory "yolovision-output.svg"
Copy-Item -LiteralPath $outputJsonPath -Destination $copiedOutputJson -Force
Copy-Item -LiteralPath $visualizationPath -Destination $copiedVisualization -Force

$gpuName = ""
$driverVersion = ""
try {
  $gpuLine = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
  if (-not [string]::IsNullOrWhiteSpace($gpuLine)) {
    $gpuParts = $gpuLine -split ',', 2
    $gpuName = $gpuParts[0].Trim()
    if ($gpuParts.Count -gt 1) {
      $driverVersion = $gpuParts[1].Trim()
    }
  }
}
catch {
}

$packageCacheFileCount = @(Get-ChildItem -LiteralPath $packageCache -Recurse -File).Count
$packageCacheBytes = (Get-ChildItem -LiteralPath $packageCache -Recurse -File | Measure-Object Length -Sum).Sum
$nativeBridgeLength = $nativeBridgePaths[0].Length
$nativeBridgeSha256 = (Get-FileHash -LiteralPath $nativeBridgePaths[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$tensorLength = (Get-Item -LiteralPath $tensorPath).Length
$tensorSha256 = (Get-FileHash -LiteralPath $tensorPath -Algorithm SHA256).Hash.ToLowerInvariant()
$workspaceRemoved = $false
if (-not $KeepWorkspace.IsPresent) {
  $resolvedCleanupTarget = (Resolve-Path -LiteralPath $OutputRoot).Path
  Assert-PathUnderRoot -Path $resolvedCleanupTarget -Root $outerRoot -Description "Consumer cleanup target"
  Remove-Item -LiteralPath $resolvedCleanupTarget -Recurse -Force
  $workspaceRemoved = -not (Test-Path -LiteralPath $resolvedCleanupTarget)
}

$report = [pscustomobject][ordered]@{
  schemaVersion = 2
  recordKind = "yolovision-yolox-local-package-consumer-runtime"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  validationState = "passed-local-package-consumer-runtime"
  evidenceClassification = "local-package-consumer-runtime"
  runtimePackageKey = $RuntimePackageKey
  tensorRtLine = $TensorRtLine
  consumer = [pscustomobject][ordered]@{
    template = "samples/YoloVision.PackageConsumer"
    targetFramework = "net8.0"
    projectReferenceCount = 0
    restoredProjectLibraryCount = $projectLibraryCount
    packageSourceKind = "local-file-feed-only"
    packageSourceCount = 3
    packageCacheDrive = [IO.Path]::GetPathRoot($packageCache).TrimEnd('\')
    packageCacheFileCount = $packageCacheFileCount
    packageCacheBytes = $packageCacheBytes
    workspaceDrive = [IO.Path]::GetPathRoot($OutputRoot).TrimEnd('\')
    workspaceRemovedAfterValidation = $workspaceRemoved
    restoreExitCode = $restoreResult.ExitCode
    buildExitCode = $buildResult.ExitCode
    runtimeExitCode = $runResult.ExitCode
    restoreCommand = "dotnet restore $consumerProjectPath --configfile $nugetConfigPath --packages $packageCache --force --no-cache --verbosity minimal"
    buildCommand = "dotnet build $consumerProjectPath -c Release --no-restore --verbosity minimal"
    runtimeCommand = "dotnet " + ($runArguments -join ' ')
    restoreLogSha256 = (Get-FileHash -LiteralPath $restoreLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    buildLogSha256 = (Get-FileHash -LiteralPath $buildLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  packages = @(
    [pscustomobject][ordered]@{ role = "managed-api"; id = $managedPackage.Id; version = $managedPackage.Version; source = $ManagedPackageDirectory; length = $managedPackage.Length; sha256 = $managedPackage.Sha256 },
    [pscustomobject][ordered]@{ role = "yolovision"; id = $yoloVisionPackage.Id; version = $yoloVisionPackage.Version; source = $YoloVisionPackageDirectory; length = $yoloVisionPackage.Length; sha256 = $yoloVisionPackage.Sha256 },
    [pscustomobject][ordered]@{ role = "bridge-only"; id = $bridgePackage.Id; version = $bridgePackage.Version; source = $BridgePackageDirectory; length = $bridgePackage.Length; sha256 = $bridgePackage.Sha256 }
  )
  nativeDependency = [pscustomobject][ordered]@{
    bridgeCopiedByNuGet = $true
    bridgeFileName = "jyppxtrtbridge.dll"
    bridgeLength = $nativeBridgeLength
    bridgeSha256 = $nativeBridgeSha256
    tensorRtRoot = $TensorRtRoot
    tensorRtRuntimeRoot = $TensorRtRuntimeRoot
    cudaRoot = $CudaRoot
    cudnnRoot = $CudnnRoot
    bridgeBuildTensorRtVersion = $bridgeTensorRtVersion
    bridgeBuildCudaToolkitVersion = $bridgeCudaToolkitVersion
    bridgeBuildTensorRtLineMatches = $true
    tensorRtCudaAndCudnnAreExternalDependencies = $true
  }
  assets = [pscustomobject][ordered]@{
    modelSha256 = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    labelsSha256 = (Get-FileHash -LiteralPath $LabelsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    imageSha256 = (Get-FileHash -LiteralPath $ImagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    tensorLength = $tensorLength
    tensorSha256 = $tensorSha256
    assetsRemainOnEDrive = $true
  }
  runtime = [pscustomobject][ordered]@{
    passedMarker = "YoloVision Passed=True"
    packageConsumerMarker = "YoloVisionPackageConsumer ProjectReference=False"
    tensorRtLine = $TensorRtLine
    elapsedMilliseconds = $elapsedMilliseconds
    predictionCount = $predictions.Count
    predictions = $predictions
    stdoutSha256 = (Get-FileHash -LiteralPath $stdoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
    stderrSha256 = (Get-FileHash -LiteralPath $stderrPath -Algorithm SHA256).Hash.ToLowerInvariant()
    outputJsonSha256 = (Get-FileHash -LiteralPath $copiedOutputJson -Algorithm SHA256).Hash.ToLowerInvariant()
    visualizationSha256 = (Get-FileHash -LiteralPath $copiedVisualization -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  host = [pscustomobject][ordered]@{
    os = [Runtime.InteropServices.RuntimeInformation]::OSDescription
    processArchitecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
    dotnetVersion = (& dotnet --version).Trim()
    gpu = $gpuName
    driverVersion = $driverVersion
  }
  boundary = [pscustomobject][ordered]@{
    isRuntimeExecutionEvidence = $true
    isRealModelRuntimeEvidence = $true
    isLocalPackageConsumerRuntimeEvidence = $true
    isPackageConsumerRuntimeProof = $false
    packagesDownloadedFromPublicFeed = $false
    publicRedistributionOwnerApproval = $false
    canPromotePackageConsumerRuntime = $false
    canPublishPublicly = $false
    isPostPublishProof = $false
    canCloseReleaseIssue = $false
    performsPublish = $false
  }
}

$reportPath = Join-Path $ReportDirectory "yolox-local-package-consumer-runtime.json"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($reportPath, ".md")
$markdown = @(
  "# YOLOX Local Package Consumer Runtime",
  "",
  "- state: ``$($report.validationState)``",
  "- classification: ``$($report.evidenceClassification)``",
  "- runtime package key: ``$RuntimePackageKey``",
  "- TensorRT line: ``$TensorRtLine``",
  "- bridge build: TensorRT ``$bridgeTensorRtVersion`` / CUDA ``$bridgeCudaToolkitVersion``",
  "- ProjectReference count: ``0``",
  "- local package count: ``3``",
  "- prediction count: ``$($predictions.Count)``",
  "- runtime marker: ``YoloVision Passed=True``",
  "- workspace removed: ``$workspaceRemoved``",
  "- package-consumer runtime proof: ``False``",
  "- can publish publicly: ``False``",
  "",
  "This record proves a clean local-feed PackageReference restore/build/run with the official YOLOX assets. It does not prove public-feed download, redistribution approval, post-publish verification, or release closure."
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "ValidationState=$($report.validationState) RuntimePackageKey=$RuntimePackageKey TensorRtLine=$TensorRtLine PredictionCount=$($predictions.Count) ProjectReferenceCount=0"
Write-Host "EvidenceClassification=$($report.evidenceClassification) IsPackageConsumerRuntimeProof=False CanPublishPublicly=False"
Write-Host "Report=$reportPath"
