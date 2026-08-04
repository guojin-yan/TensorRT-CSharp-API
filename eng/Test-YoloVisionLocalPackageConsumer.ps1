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
  [ValidateSet("yolox-detection", "yolov8-detection", "yolov10-detection", "yolov8-segmentation", "torchvision-lraspp-semantic", "yolov8-classification", "yolov8-pose", "yolov8-obb")][string]$Scenario = "yolox-detection",
  [string]$ModelPath,
  [string]$ModelWeightsPath,
  [string]$LabelsPath,
  [string]$ImagePath,
  [string]$ReferenceOutput0Path,
  [string]$ReferenceOutput1Path,
  [string]$ReferenceClassIndexPath,
  [string]$ReferenceInputTensorPath,
  [string]$PythonPath,
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

function Remove-DirectoryTree {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Attempts = 6
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  $fullPath = [IO.Path]::GetFullPath($Path)
  $extendedPath = if ($fullPath.StartsWith("\\", [StringComparison]::Ordinal)) {
    "\\?\UNC\" + $fullPath.Substring(2)
  }
  else {
    "\\?\" + $fullPath
  }
  $lastError = $null
  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    try {
      Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction Stop
      return
    }
    catch {
      $lastError = $_
    }

    try {
      [GC]::Collect()
      [GC]::WaitForPendingFinalizers()
      [IO.Directory]::Delete($extendedPath, $true)
      return
    }
    catch {
      $lastError = $_
      Start-Sleep -Milliseconds (250 * $attempt)
    }

    if (-not (Test-Path -LiteralPath $fullPath)) {
      return
    }
  }

  throw "Failed to remove directory '$fullPath': $($lastError.Exception.Message)"
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

function Get-PackageEntryNames {
  param([Parameter(Mandatory = $true)][string]$PackagePath)

  $zip = [IO.Compression.ZipFile]::OpenRead($PackagePath)
  try {
    return @($zip.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
  }
  finally {
    $zip.Dispose()
  }
}

function Assert-FileSha256 {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256,
    [Parameter(Mandatory = $true)][string]$Description
  )

  $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if (-not [string]::Equals($actualSha256, $ExpectedSha256, [StringComparison]::Ordinal)) {
    throw "$Description SHA256 mismatch. Expected='$ExpectedSha256' Actual='$actualSha256' Path='$Path'."
  }
}

function Invoke-CapturedProcess {
  param(
    [Parameter(Mandatory = $true)][string]$FileName,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [hashtable]$Environment = @{},
    [string[]]$EnvironmentVariablesToRemove = @()
  )

  function ConvertTo-NativeProcessArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
      return $Value
    }

    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashCount = 0
    foreach ($character in $Value.ToCharArray()) {
      if ($character -eq '\') {
        $backslashCount++
        continue
      }

      if ($character -eq '"') {
        [void]$builder.Append(('\' * (($backslashCount * 2) + 1)))
        [void]$builder.Append('"')
      }
      else {
        if ($backslashCount -gt 0) {
          [void]$builder.Append(('\' * $backslashCount))
        }
        [void]$builder.Append($character)
      }
      $backslashCount = 0
    }
    if ($backslashCount -gt 0) {
      [void]$builder.Append(('\' * ($backslashCount * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
  }

  $startInfo = [Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $FileName
  $startInfo.WorkingDirectory = $WorkingDirectory
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.CreateNoWindow = $true
  $startInfo.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeProcessArgument -Value $_ }) -join ' ')
  foreach ($name in $EnvironmentVariablesToRemove) {
    [void]$startInfo.EnvironmentVariables.Remove($name)
  }
  foreach ($name in $Environment.Keys) {
    $startInfo.EnvironmentVariables[$name] = [string]$Environment[$name]
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

function Set-NamedArgumentValue {
  param(
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value
  )

  $copy = @($Arguments)
  $matches = @()
  for ($index = 0; $index -lt $copy.Count; $index++) {
    if ([string]::Equals($copy[$index], $Name, [StringComparison]::Ordinal)) {
      $matches += $index
    }
  }
  if ($matches.Count -ne 1 -or $matches[0] + 1 -ge $copy.Count) {
    throw "Argument '$Name' must occur exactly once with a value."
  }

  $copy[$matches[0] + 1] = $Value
  return $copy
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
$isSegmentationScenario = [string]::Equals($Scenario, "yolov8-segmentation", [StringComparison]::Ordinal)
$isSemanticScenario = [string]::Equals($Scenario, "torchvision-lraspp-semantic", [StringComparison]::Ordinal)
$isClassificationScenario = [string]::Equals($Scenario, "yolov8-classification", [StringComparison]::Ordinal)
$isPoseScenario = [string]::Equals($Scenario, "yolov8-pose", [StringComparison]::Ordinal)
$isObbScenario = [string]::Equals($Scenario, "yolov8-obb", [StringComparison]::Ordinal)
$isOfficialDetectionScenario = [string]::Equals($Scenario, "yolov8-detection", [StringComparison]::Ordinal)
$isYoloV10DetectionScenario = [string]::Equals($Scenario, "yolov10-detection", [StringComparison]::Ordinal)
$scenarioSlug = if ($isSegmentationScenario) { "yolov8n-seg" } elseif ($isSemanticScenario) { "lraspp-semantic" } elseif ($isClassificationScenario) { "yolov8n-cls" } elseif ($isPoseScenario) { "yolov8n-pose" } elseif ($isObbScenario) { "yolov8n-obb" } elseif ($isOfficialDetectionScenario) { "yolov8n-det" } elseif ($isYoloV10DetectionScenario) { "yolov10n-det" } else { "yolox" }

$resolvedRuntimeRoots = $null
if ([string]::IsNullOrWhiteSpace($TensorRtRoot) -or
    [string]::IsNullOrWhiteSpace($CudaRoot) -or
    [string]::IsNullOrWhiteSpace($CudnnRoot)) {
  $runtimeRootShell = Get-Command pwsh -ErrorAction SilentlyContinue
  if ($null -eq $runtimeRootShell) {
    $runtimeRootShell = Get-Command powershell.exe -ErrorAction Stop
  }
  $runtimeRootsJson = (& $runtimeRootShell.Source -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $RuntimePackageKey -RepositoryRoot $RepositoryRoot | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve vendor roots for runtime package key '$RuntimePackageKey'."
  }
  $resolvedRuntimeRoots = $runtimeRootsJson | ConvertFrom-Json
}
$defaultTensorRtRoot = if ($null -eq $resolvedRuntimeRoots) { "" } else { [string]$resolvedRuntimeRoots.tensorRtRoot }
$defaultCudaRoot = if ($null -eq $resolvedRuntimeRoots) { "" } else { [string]$resolvedRuntimeRoots.cudaRoot }
$defaultCudnnRoot = if ($null -eq $resolvedRuntimeRoots) { "" } else { [string]$resolvedRuntimeRoots.cudnnRoot }

$defaultConsumerOutputRoot = if ($isSegmentationScenario) {
  Join-Path $outerRoot "consumer-workspaces\yolovision-yolov8n-seg-local-package-trt$TensorRtLine"
}
elseif ($isSemanticScenario) {
  Join-Path $outerRoot "consumer-workspaces\yolovision-lraspp-semantic-local-package-trt$TensorRtLine"
}
elseif ($isClassificationScenario) {
  Join-Path $outerRoot "consumer-workspaces\yolovision-yolov8n-cls-local-package-trt$TensorRtLine"
}
elseif ($isPoseScenario) {
  Join-Path $outerRoot "consumer-workspaces\yolovision-yolov8n-pose-local-package-trt$TensorRtLine"
}
elseif ($isObbScenario) {
  Join-Path $outerRoot "consumer-workspaces\yolovision-yolov8n-obb-local-package-trt$TensorRtLine"
}
elseif ($isOfficialDetectionScenario) {
  Join-Path $outerRoot "consumer-workspaces\yolovision-yolov8n-det-local-package-trt$TensorRtLine"
}
elseif ($isYoloV10DetectionScenario) {
  Join-Path $outerRoot "consumer-workspaces\yv-yolov10-pkg-trt$TensorRtLine"
}
else {
  Join-Path $outerRoot "consumer-workspaces\yv-yolox-pkg-trt$TensorRtLine"
}
$OutputRoot = Resolve-PathValue -Value $OutputRoot -DefaultValue $defaultConsumerOutputRoot -RelativeRoot $outerRoot
$ReportDirectory = Resolve-PathValue -Value $ReportDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\yolovision\$scenarioSlug-local-package-consumer\$RuntimePackageKey") -RelativeRoot $RepositoryRoot
$ManagedPackageDirectory = Resolve-PathValue -Value $ManagedPackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\managed") -RelativeRoot $RepositoryRoot
$YoloVisionPackageDirectory = Resolve-PathValue -Value $YoloVisionPackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\yolovision-nupkg") -RelativeRoot $RepositoryRoot
$BridgePackageDirectory = Resolve-PathValue -Value $BridgePackageDirectory -DefaultValue (Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$RuntimePackageKey") -RelativeRoot $RepositoryRoot
$defaultModelPath = if ($isSegmentationScenario) {
  Join-Path $outerRoot "downloads\yolov8n-seg-ultralytics-v8.3.0\source\yolov8n-seg.onnx"
}
elseif ($isSemanticScenario) {
  Join-Path $outerRoot "models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0\lraspp-mobilenet-v3-large-320.onnx"
}
elseif ($isClassificationScenario) {
  Join-Path $outerRoot "models\YoloVision\Classification\yolov8n-cls-ultralytics-v8.3.0\yolov8n-cls.onnx"
}
elseif ($isPoseScenario) {
  Join-Path $outerRoot "models\YoloVision\Pose\yolov8n-pose-ultralytics-v8.3.0\yolov8n-pose.onnx"
}
elseif ($isObbScenario) {
  Join-Path $outerRoot "models\YoloVision\OrientedBoundingBox\yolov8n-obb-ultralytics-v8.3.0\yolov8n-obb.onnx"
}
elseif ($isOfficialDetectionScenario) {
  Join-Path $outerRoot "models\YoloVision\Detection\yolov8n-ultralytics-v8.3.0\yolov8n.onnx"
}
elseif ($isYoloV10DetectionScenario) {
  Join-Path $outerRoot "models\YoloVision\Detection\yolov10n-thu-mig-v1.1\yolov10n.onnx"
}
else {
  Join-Path $outerRoot "downloads\yolox-apache\source\yolox_s.onnx"
}
$ModelPath = Resolve-PathValue -Value $ModelPath -DefaultValue $defaultModelPath -RelativeRoot $outerRoot
$defaultLabelsPath = if ($isSemanticScenario) {
  Join-Path $RepositoryRoot "artifacts\yolovision\semantic-lraspp-reference\voc-semantic.names"
}
elseif ($isClassificationScenario) {
  Join-Path $outerRoot "downloads\yolov8n-cls-ultralytics-v8.3.0\reports\independent-reference\imagenet-yolov8n-cls.names"
}
elseif ($isObbScenario) {
  Join-Path $outerRoot "downloads\yolov8n-obb-ultralytics-v8.3.0\derived\dota.names"
}
elseif ($isOfficialDetectionScenario) {
  Join-Path $outerRoot "downloads\yolov8n-det-ultralytics-v8.3.0\derived\coco.names"
}
else {
  Join-Path $outerRoot "downloads\yolox-apache\derived\coco.names"
}
$defaultImagePath = if ($isSemanticScenario) {
  Join-Path $RepositoryRoot "artifacts\yolovision\semantic-lraspp-reference\dog.ppm"
}
elseif ($isClassificationScenario) {
  Join-Path $outerRoot "downloads\yolov8n-cls-ultralytics-v8.3.0\source\bus.jpg"
}
elseif ($isPoseScenario) {
  Join-Path $outerRoot "downloads\yolov8n-pose-ultralytics-v8.3.0\derived\bus.ppm"
}
elseif ($isObbScenario) {
  Join-Path $outerRoot "downloads\yolov8n-obb-ultralytics-v8.3.0\derived\boats.ppm"
}
elseif ($isOfficialDetectionScenario) {
  Join-Path $outerRoot "downloads\yolov8n-det-ultralytics-v8.3.0\derived\bus.ppm"
}
elseif ($isYoloV10DetectionScenario) {
  Join-Path $outerRoot "downloads\article-assets\yolovision-detection-cc0-bus-station\liverpool-street-bus-station-1280.ppm"
}
else {
  Join-Path $outerRoot "downloads\yolox-apache\derived\dog.ppm"
}
$LabelsPath = Resolve-PathValue -Value $LabelsPath -DefaultValue $defaultLabelsPath -RelativeRoot $outerRoot
$ImagePath = Resolve-PathValue -Value $ImagePath -DefaultValue $defaultImagePath -RelativeRoot $outerRoot
if ($isSegmentationScenario) {
  $ModelWeightsPath = Resolve-PathValue -Value $ModelWeightsPath -DefaultValue (Join-Path $outerRoot "downloads\yolov8n-seg-ultralytics-v8.3.0\source\yolov8n-seg.pt") -RelativeRoot $outerRoot
  $ReferenceOutput0Path = Resolve-PathValue -Value $ReferenceOutput0Path -DefaultValue (Join-Path $outerRoot "downloads\yolov8n-seg-ultralytics-v8.3.0\reference\output0.reference.json") -RelativeRoot $outerRoot
  $ReferenceOutput1Path = Resolve-PathValue -Value $ReferenceOutput1Path -DefaultValue (Join-Path $outerRoot "downloads\yolov8n-seg-ultralytics-v8.3.0\reference\output1.reference.json") -RelativeRoot $outerRoot
  $defaultPythonPath = Join-Path $env:USERPROFILE ".conda\envs\ultralytics\python.exe"
  $PythonPath = Resolve-PathValue -Value $PythonPath -DefaultValue $defaultPythonPath -RelativeRoot $outerRoot
}
elseif ($isSemanticScenario) {
  $semanticAssetRoot = Join-Path $RepositoryRoot "artifacts\yolovision\semantic-lraspp-reference"
  $semanticModelRoot = Join-Path $outerRoot "models\YoloVision\SemanticSegmentation\lraspp-mobilenet-v3-large-torchvision-v0.25.0"
  $ModelWeightsPath = Resolve-PathValue -Value $ModelWeightsPath -DefaultValue (Join-Path $semanticModelRoot "lraspp_mobilenet_v3_large-d234d4ea.pth") -RelativeRoot $outerRoot
  $ReferenceOutput0Path = Resolve-PathValue -Value $ReferenceOutput0Path -DefaultValue (Join-Path $semanticAssetRoot "semantic.reference.json") -RelativeRoot $outerRoot
  $ReferenceClassIndexPath = Resolve-PathValue -Value $ReferenceClassIndexPath -DefaultValue (Join-Path $semanticAssetRoot "semantic-class-index-onnxruntime.i32.bin") -RelativeRoot $outerRoot
  $defaultPythonPath = Join-Path $env:USERPROFILE ".conda\envs\ultralytics\python.exe"
  $PythonPath = Resolve-PathValue -Value $PythonPath -DefaultValue $defaultPythonPath -RelativeRoot $outerRoot
}
elseif ($isClassificationScenario) {
  $classificationRoot = Join-Path $outerRoot "downloads\yolov8n-cls-ultralytics-v8.3.0"
  $ModelWeightsPath = Resolve-PathValue -Value $ModelWeightsPath -DefaultValue (Join-Path $classificationRoot "source\yolov8n-cls.pt") -RelativeRoot $outerRoot
  $ReferenceOutput0Path = Resolve-PathValue -Value $ReferenceOutput0Path -DefaultValue (Join-Path $classificationRoot "reports\independent-reference\output0.reference.json") -RelativeRoot $outerRoot
  $ReferenceInputTensorPath = Resolve-PathValue -Value $ReferenceInputTensorPath -DefaultValue (Join-Path $classificationRoot "reports\independent-reference\input-ultralytics-1x3x224x224.fp32.bin") -RelativeRoot $outerRoot
  $defaultPythonPath = Join-Path $env:USERPROFILE ".conda\envs\ultralytics\python.exe"
  $PythonPath = Resolve-PathValue -Value $PythonPath -DefaultValue $defaultPythonPath -RelativeRoot $outerRoot
}
elseif ($isPoseScenario) {
  $poseRoot = Join-Path $outerRoot "downloads\yolov8n-pose-ultralytics-v8.3.0"
  $ModelWeightsPath = Resolve-PathValue -Value $ModelWeightsPath -DefaultValue (Join-Path $poseRoot "source\yolov8n-pose.pt") -RelativeRoot $outerRoot
  $ReferenceOutput0Path = Resolve-PathValue -Value $ReferenceOutput0Path -DefaultValue (Join-Path $poseRoot "reference\output0.reference.json") -RelativeRoot $outerRoot
  $ReferenceInputTensorPath = Resolve-PathValue -Value $ReferenceInputTensorPath -DefaultValue (Join-Path $poseRoot "runtime\bus-1x3x640x640-rgb-letterbox.fp32.bin") -RelativeRoot $outerRoot
  $poseSourceImagePath = Join-Path $poseRoot "source\bus.jpg"
  $defaultPythonPath = Join-Path $env:USERPROFILE ".conda\envs\ultralytics\python.exe"
  $PythonPath = Resolve-PathValue -Value $PythonPath -DefaultValue $defaultPythonPath -RelativeRoot $outerRoot
}
elseif ($isObbScenario) {
  $obbRoot = Join-Path $outerRoot "downloads\yolov8n-obb-ultralytics-v8.3.0"
  $ModelWeightsPath = Resolve-PathValue -Value $ModelWeightsPath -DefaultValue (Join-Path $obbRoot "source\yolov8n-obb.pt") -RelativeRoot $outerRoot
  $ReferenceOutput0Path = Resolve-PathValue -Value $ReferenceOutput0Path -DefaultValue (Join-Path $obbRoot "reference\output0.reference.json") -RelativeRoot $outerRoot
  $ReferenceInputTensorPath = Resolve-PathValue -Value $ReferenceInputTensorPath -DefaultValue (Join-Path $obbRoot "runtime\boats-1x3x1024x1024-rgb-letterbox.fp32.bin") -RelativeRoot $outerRoot
  $obbSourceImagePath = Join-Path $obbRoot "source\boats.jpg"
  $obbPinnedIndependentReferencePath = Join-Path $obbRoot "reference\ultralytics-pytorch-reference.json"
  $defaultPythonPath = Join-Path $env:USERPROFILE ".conda\envs\ultralytics\python.exe"
  $PythonPath = Resolve-PathValue -Value $PythonPath -DefaultValue $defaultPythonPath -RelativeRoot $outerRoot
}
elseif ($isOfficialDetectionScenario) {
  $officialDetectionRoot = Join-Path $outerRoot "downloads\yolov8n-det-ultralytics-v8.3.0"
  $ModelWeightsPath = Resolve-PathValue -Value $ModelWeightsPath -DefaultValue (Join-Path $officialDetectionRoot "source\yolov8n.pt") -RelativeRoot $outerRoot
  $ReferenceOutput0Path = Resolve-PathValue -Value $ReferenceOutput0Path -DefaultValue (Join-Path $officialDetectionRoot "reports\independent-reference\output0.reference.json") -RelativeRoot $outerRoot
  $ReferenceInputTensorPath = Resolve-PathValue -Value $ReferenceInputTensorPath -DefaultValue (Join-Path $officialDetectionRoot "runtime\bus-csharp-letterbox-1x3x640x640.fp32.bin") -RelativeRoot $outerRoot
  $officialDetectionSourceImagePath = Join-Path $officialDetectionRoot "source\bus.jpg"
  $officialDetectionCocoYamlPath = Join-Path $officialDetectionRoot "source\coco.yaml"
  $officialDetectionPinnedIndependentReferencePath = Join-Path $officialDetectionRoot "reports\independent-reference\ultralytics-pytorch-reference.json"
  $defaultPythonPath = Join-Path $env:USERPROFILE ".conda\envs\ultralytics\python.exe"
  $PythonPath = Resolve-PathValue -Value $PythonPath -DefaultValue $defaultPythonPath -RelativeRoot $outerRoot
}
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
  @{ Path = $ModelPath; Description = "YoloVision model" },
  @{ Path = $LabelsPath; Description = "YoloVision labels" },
  @{ Path = $ImagePath; Description = "YoloVision image" }
)) {
  Assert-NonCDrivePath -Path $item.Path -Description $item.Description
}

if ($isSegmentationScenario) {
  foreach ($item in @(
    @{ Path = $ModelWeightsPath; Description = "YOLOv8 segmentation weights" },
    @{ Path = $ReferenceOutput0Path; Description = "YOLOv8 output0 reference" },
    @{ Path = $ReferenceOutput1Path; Description = "YOLOv8 output1 reference" }
  )) {
    Assert-NonCDrivePath -Path $item.Path -Description $item.Description
  }
}
elseif ($isOfficialDetectionScenario) {
  foreach ($item in @(
    @{ Path = $ModelWeightsPath; Description = "YOLOv8n detection source weights" },
    @{ Path = $ReferenceOutput0Path; Description = "YOLOv8n detection output0 reference" },
    @{ Path = $ReferenceInputTensorPath; Description = "YOLOv8n detection authoritative input tensor" },
    @{ Path = $officialDetectionSourceImagePath; Description = "YOLOv8n detection source image" },
    @{ Path = $officialDetectionCocoYamlPath; Description = "YOLOv8n detection COCO YAML" },
    @{ Path = $officialDetectionPinnedIndependentReferencePath; Description = "YOLOv8n detection pinned independent reference" }
  )) {
    Assert-NonCDrivePath -Path $item.Path -Description $item.Description
  }
}
elseif ($isSemanticScenario) {
  foreach ($item in @(
    @{ Path = $ModelWeightsPath; Description = "LRASPP source weights" },
    @{ Path = $ReferenceOutput0Path; Description = "LRASPP semantic raw reference" },
    @{ Path = $ReferenceClassIndexPath; Description = "LRASPP semantic class-index reference" }
  )) {
    Assert-NonCDrivePath -Path $item.Path -Description $item.Description
  }
}
elseif ($isClassificationScenario) {
  foreach ($item in @(
    @{ Path = $ModelWeightsPath; Description = "YOLOv8n-cls source weights" },
    @{ Path = $ReferenceOutput0Path; Description = "YOLOv8n-cls output0 reference" },
    @{ Path = $ReferenceInputTensorPath; Description = "YOLOv8n-cls authoritative input tensor" }
  )) {
    Assert-NonCDrivePath -Path $item.Path -Description $item.Description
  }
}
elseif ($isPoseScenario) {
  foreach ($item in @(
    @{ Path = $ModelWeightsPath; Description = "YOLOv8n-pose source weights" },
    @{ Path = $ReferenceOutput0Path; Description = "YOLOv8n-pose output0 reference" },
    @{ Path = $ReferenceInputTensorPath; Description = "YOLOv8n-pose authoritative input tensor" },
    @{ Path = $poseSourceImagePath; Description = "YOLOv8n-pose source image" }
  )) {
    Assert-NonCDrivePath -Path $item.Path -Description $item.Description
  }
}
elseif ($isObbScenario) {
  foreach ($item in @(
    @{ Path = $ModelWeightsPath; Description = "YOLOv8n-obb source weights" },
    @{ Path = $ReferenceOutput0Path; Description = "YOLOv8n-obb output0 reference" },
    @{ Path = $ReferenceInputTensorPath; Description = "YOLOv8n-obb authoritative input tensor" },
    @{ Path = $obbSourceImagePath; Description = "YOLOv8n-obb source image" },
    @{ Path = $obbPinnedIndependentReferencePath; Description = "YOLOv8n-obb pinned independent reference" }
  )) {
    Assert-NonCDrivePath -Path $item.Path -Description $item.Description
  }
}

$requiredPaths = @($ModelPath, $LabelsPath, $ImagePath, $TensorRtRoot, $TensorRtRuntimeRoot, $CudaRoot, $CudnnRoot)
if ($isSegmentationScenario) {
  $requiredPaths += @($ModelWeightsPath, $ReferenceOutput0Path, $ReferenceOutput1Path, $PythonPath)
}
elseif ($isSemanticScenario) {
  $requiredPaths += @($ModelWeightsPath, $ReferenceOutput0Path, $ReferenceClassIndexPath, $PythonPath)
}
elseif ($isClassificationScenario) {
  $requiredPaths += @($ModelWeightsPath, $ReferenceOutput0Path, $ReferenceInputTensorPath, $PythonPath)
}
elseif ($isPoseScenario) {
  $requiredPaths += @($ModelWeightsPath, $ReferenceOutput0Path, $ReferenceInputTensorPath, $poseSourceImagePath, $PythonPath)
}
elseif ($isObbScenario) {
  $requiredPaths += @($ModelWeightsPath, $ReferenceOutput0Path, $ReferenceInputTensorPath, $obbSourceImagePath, $obbPinnedIndependentReferencePath, $PythonPath)
}
elseif ($isOfficialDetectionScenario) {
  $requiredPaths += @($ModelWeightsPath, $ReferenceOutput0Path, $ReferenceInputTensorPath, $officialDetectionSourceImagePath, $officialDetectionCocoYamlPath, $officialDetectionPinnedIndependentReferencePath, $PythonPath)
}
foreach ($requiredPath in $requiredPaths) {
  if (-not (Test-Path -LiteralPath $requiredPath)) {
    throw "Required local dependency does not exist: $requiredPath"
  }
}

if ($isSegmentationScenario) {
  Assert-FileSha256 -Path $ModelPath -ExpectedSha256 "08b5c61368d4ddec5e647522fc55a93c42a9e0c581770aae48b87bba65a9b21d" -Description "YOLOv8n-seg ONNX"
  Assert-FileSha256 -Path $ModelWeightsPath -ExpectedSha256 "a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c" -Description "YOLOv8n-seg weights"
  Assert-FileSha256 -Path $ReferenceOutput0Path -ExpectedSha256 "a607dd1d80dbb85a1a77f5354c7669e357c2caf8d0a9cd93328397d1858bde5f" -Description "YOLOv8n-seg output0 reference"
  Assert-FileSha256 -Path $ReferenceOutput1Path -ExpectedSha256 "3cc8387483187bc5d86ef8e82943e215caadefbbf41a8523d9ffc4c6b040f8d5" -Description "YOLOv8n-seg output1 reference"
  Assert-FileSha256 -Path $LabelsPath -ExpectedSha256 "4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930" -Description "COCO labels"
  Assert-FileSha256 -Path $ImagePath -ExpectedSha256 "6cb94c9cd0781412598fe179246b09041af4303d388a5ba3c55f760dff11ec2c" -Description "YOLOX dog image"
}
elseif ($isSemanticScenario) {
  Assert-FileSha256 -Path $ModelPath -ExpectedSha256 "3cb94e561bdefe606ed7d1a2c4d0296409bec066f3a39a9fe9dabd72b23728f8" -Description "LRASPP ONNX"
  Assert-FileSha256 -Path $ModelWeightsPath -ExpectedSha256 "d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280" -Description "LRASPP source weights"
  Assert-FileSha256 -Path $ReferenceOutput0Path -ExpectedSha256 "09a27bfe1ee7cd48413806f057bcface6c3c8ed224ed4766e1618d9766b14dff" -Description "LRASPP semantic raw reference"
  Assert-FileSha256 -Path $ReferenceClassIndexPath -ExpectedSha256 "fdd15b95222eadf137fc6880e56990aa507ee7d2429f471deaae9eab31268414" -Description "LRASPP semantic class-index reference"
  Assert-FileSha256 -Path $LabelsPath -ExpectedSha256 "82b3b65943e7865cf31b8c5e9a720edd5eddff176d2b146ae09795a734595781" -Description "VOC semantic labels"
  Assert-FileSha256 -Path $ImagePath -ExpectedSha256 "58d4301e1ccf0d60b890b73980e4a00b0316840d618b7c741484e7514e71ff4b" -Description "LRASPP dog PPM image"
}
elseif ($isClassificationScenario) {
  Assert-FileSha256 -Path $ModelPath -ExpectedSha256 "630c022a99885d59f633ab5a614738f8a49be7f361e340fd3ff89b8c19b0768f" -Description "YOLOv8n-cls ONNX"
  Assert-FileSha256 -Path $ModelWeightsPath -ExpectedSha256 "11fa19f2aea79bc960d680a13f82f22105982b325eb9e17a4a5e1a9f8245980a" -Description "YOLOv8n-cls weights"
  Assert-FileSha256 -Path $ReferenceOutput0Path -ExpectedSha256 "ded68dd048acdee3a38e95dc46490517e85ece8346f1ebe627bdd88cdb307697" -Description "YOLOv8n-cls output0 reference"
  Assert-FileSha256 -Path $ReferenceInputTensorPath -ExpectedSha256 "05e47521b07652eee70942902ab5bf070edc9da7067466a5433c7cdd29fb1a62" -Description "YOLOv8n-cls authoritative input tensor"
  Assert-FileSha256 -Path $LabelsPath -ExpectedSha256 "dcc60e7297d33ea2b0efeab10074e4ac07d3fdd702fb1fb7ace169ee684240dd" -Description "YOLOv8n-cls ImageNet labels"
  Assert-FileSha256 -Path $ImagePath -ExpectedSha256 "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63" -Description "YOLOv8n-cls bus image"
}
elseif ($isPoseScenario) {
  Assert-FileSha256 -Path $ModelPath -ExpectedSha256 "ed1e8d2d2aeb8a2c66e642a16295a72a2990393e3a3843325537da7e11c8a899" -Description "YOLOv8n-pose ONNX"
  Assert-FileSha256 -Path $ModelWeightsPath -ExpectedSha256 "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212" -Description "YOLOv8n-pose weights"
  Assert-FileSha256 -Path $ReferenceOutput0Path -ExpectedSha256 "73752d18797b5c9336359b99eef4cadfb8d263e5925fd489648ce3072ef440f1" -Description "YOLOv8n-pose output0 reference"
  Assert-FileSha256 -Path $ReferenceInputTensorPath -ExpectedSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" -Description "YOLOv8n-pose authoritative input tensor"
  Assert-FileSha256 -Path $LabelsPath -ExpectedSha256 "4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930" -Description "COCO labels"
  Assert-FileSha256 -Path $ImagePath -ExpectedSha256 "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688" -Description "YOLOv8n-pose bus PPM image"
  Assert-FileSha256 -Path $poseSourceImagePath -ExpectedSha256 "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63" -Description "YOLOv8n-pose bus source image"
}
elseif ($isObbScenario) {
  Assert-FileSha256 -Path $ModelPath -ExpectedSha256 "5f2701ef5326fb5a691999438cfc55a69656323c21ffddebaff8968ab6de2e92" -Description "YOLOv8n-obb ONNX"
  Assert-FileSha256 -Path $ModelWeightsPath -ExpectedSha256 "fa6e4cd2691f132875c143135affaa66b5d89394ebb1d07d19770a9b6382c1b8" -Description "YOLOv8n-obb weights"
  Assert-FileSha256 -Path $ReferenceOutput0Path -ExpectedSha256 "53f00a488c44227a3e5fbc86b530355acfd0c17cf9fdfc771a6d1f99f72fd65d" -Description "YOLOv8n-obb output0 reference"
  Assert-FileSha256 -Path $ReferenceInputTensorPath -ExpectedSha256 "c56c027619088bce94f9160a3f602b4ad81fe323001867b1ee456100040fec6e" -Description "YOLOv8n-obb authoritative input tensor"
  Assert-FileSha256 -Path $LabelsPath -ExpectedSha256 "4b1932ade5050f71f3a8ff9031993d8bb2891b2c31291e29cbe2331820221272" -Description "DOTA labels"
  Assert-FileSha256 -Path $ImagePath -ExpectedSha256 "9156edd731c3dcc7baaa59d93d0753c5098003c6bffc5eecaca1087713d07c76" -Description "YOLOv8n-obb boats PPM image"
  Assert-FileSha256 -Path $obbSourceImagePath -ExpectedSha256 "8c5ada657cf8110a9f8aaac954c1dd96cde0187315b581276c32b0d1863e756f" -Description "YOLOv8n-obb boats source image"
  Assert-FileSha256 -Path $obbPinnedIndependentReferencePath -ExpectedSha256 "97e777e1bcecec7fdb3b15e8d4d1e34468505947522c8881c435b484afc10045" -Description "Pinned Ultralytics OBB reference"
}
elseif ($isOfficialDetectionScenario) {
  Assert-FileSha256 -Path $ModelPath -ExpectedSha256 "db28a49ffbb0425f39ae56252e7e0b43d06b357416c7da58872e285560b4221e" -Description "YOLOv8n detection ONNX"
  Assert-FileSha256 -Path $ModelWeightsPath -ExpectedSha256 "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36" -Description "YOLOv8n detection weights"
  Assert-FileSha256 -Path $ReferenceOutput0Path -ExpectedSha256 "6b7d8acb790fde1d41aa503e8d6a09ead9e4ea8e447f503b8fddab998c7387d6" -Description "YOLOv8n detection output0 reference"
  Assert-FileSha256 -Path $ReferenceInputTensorPath -ExpectedSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" -Description "YOLOv8n detection authoritative input tensor"
  Assert-FileSha256 -Path $LabelsPath -ExpectedSha256 "bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a" -Description "YOLOv8n detection COCO labels"
  Assert-FileSha256 -Path $ImagePath -ExpectedSha256 "6cdb4b6728a36516826f9adb9387774a6b5db0a49837d515c9045324e04e8688" -Description "YOLOv8n detection bus PPM image"
  Assert-FileSha256 -Path $officialDetectionSourceImagePath -ExpectedSha256 "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63" -Description "YOLOv8n detection bus source image"
  Assert-FileSha256 -Path $officialDetectionCocoYamlPath -ExpectedSha256 "bd6f98a2e18775c39a4d5214080c87fcb163d367c18a2fcf2609371bab00c0b8" -Description "YOLOv8n detection COCO YAML"
  Assert-FileSha256 -Path $officialDetectionPinnedIndependentReferencePath -ExpectedSha256 "eae23c95dafee3904fb8adc519e90045929491ccbea613a32ed69f2290e405c0" -Description "Pinned Ultralytics detection reference"
}
elseif ($isYoloV10DetectionScenario) {
  Assert-FileSha256 -Path $ModelPath -ExpectedSha256 "7025ea1913f9a259cf8a8465ed608e10610d1bb376db2e0348b13e3bd286e0d3" -Description "YOLOv10n v1.1 ONNX"
  Assert-FileSha256 -Path $LabelsPath -ExpectedSha256 "4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930" -Description "COCO labels"
  Assert-FileSha256 -Path $ImagePath -ExpectedSha256 "80715af66669147b049fec9386152bd45505f4e494a65079fdc55404d4589b8e" -Description "YOLOv10n CC0 bus-station PPM image"
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
$packageEntries = @(
  foreach ($package in @($managedPackage, $yoloVisionPackage, $bridgePackage)) {
    foreach ($entryName in Get-PackageEntryNames -PackagePath $package.Path) {
      [pscustomobject]@{ packageId = $package.Id; entryName = $entryName }
    }
  }
)
$vendorRuntimeEntries = @($packageEntries | Where-Object {
  [IO.Path]::GetFileName([string]$_.entryName) -match '^(?:cudart|cudnn|nvinfer|nvonnxparser|nvrtc|nvJitLink|cublas|cufft|curand|cusolver|cusparse|nvToolsExt|zlibwapi).*(?:\.dll|\.so(?:\.[0-9.]+)?|\.dylib)$'
})
if ($vendorRuntimeEntries.Count -ne 0) {
  throw "Local package set contains forbidden NVIDIA vendor runtime entries: $($vendorRuntimeEntries.entryName -join ', ')."
}
$bridgeNativeEntries = @(Get-PackageEntryNames -PackagePath $bridgePackage.Path | Where-Object {
  $_.StartsWith("runtimes/win-x64/native/", [StringComparison]::OrdinalIgnoreCase) -and -not $_.EndsWith("/", [StringComparison]::Ordinal)
})
if ($bridgeNativeEntries.Count -ne 1 -or
    -not [string]::Equals($bridgeNativeEntries[0], "runtimes/win-x64/native/jyppxtrtbridge.dll", [StringComparison]::OrdinalIgnoreCase)) {
  throw "Bridge package native surface must contain exactly runtimes/win-x64/native/jyppxtrtbridge.dll."
}

if (Test-Path -LiteralPath $OutputRoot) {
  $resolvedExistingOutput = (Resolve-Path -LiteralPath $OutputRoot).Path
  Assert-PathUnderRoot -Path $resolvedExistingOutput -Root $outerRoot -Description "Existing consumer workspace"
  Remove-DirectoryTree -Path $resolvedExistingOutput
}
New-Item -ItemType Directory -Path $OutputRoot, $ReportDirectory -Force | Out-Null

$workspace = Join-Path $OutputRoot "workspace"
$packageCache = Join-Path $OutputRoot "packages"
$runOutput = Join-Path $OutputRoot "run-output"
$isolatedFeedRoot = Join-Path $OutputRoot "isolated-local-feeds"
$isolatedManagedFeed = Join-Path $isolatedFeedRoot "managed-api"
$isolatedYoloVisionFeed = Join-Path $isolatedFeedRoot "yolovision"
$isolatedBridgeFeed = Join-Path $isolatedFeedRoot "bridge-only"
New-Item -ItemType Directory -Path $workspace, $packageCache, $runOutput, $isolatedManagedFeed, $isolatedYoloVisionFeed, $isolatedBridgeFeed -Force | Out-Null
Copy-Item -LiteralPath $managedPackage.Path -Destination $isolatedManagedFeed -Force
Copy-Item -LiteralPath $yoloVisionPackage.Path -Destination $isolatedYoloVisionFeed -Force
Copy-Item -LiteralPath $bridgePackage.Path -Destination $isolatedBridgeFeed -Force
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
if ($projectContent.IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
  throw "Consumer project must not contain ProjectReference."
}
if ($projectContent.IndexOf("<Reference ", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $projectContent.IndexOf("<HintPath>", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
  throw "Consumer project must not contain direct assembly references or HintPath entries."
}

$nugetConfigPath = Join-Path $workspace "NuGet.config"
$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="jyppx-managed-local" value="$(ConvertTo-XmlAttributeValue -Value $isolatedManagedFeed)" />
    <add key="jyppx-yolovision-local" value="$(ConvertTo-XmlAttributeValue -Value $isolatedYoloVisionFeed)" />
    <add key="jyppx-bridge-local" value="$(ConvertTo-XmlAttributeValue -Value $isolatedBridgeFeed)" />
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
$restoredPackageHashChecks = @(
  foreach ($package in @($managedPackage, $yoloVisionPackage, $bridgePackage)) {
    $normalizedId = $package.Id.ToLowerInvariant()
    $normalizedVersion = $package.Version.ToLowerInvariant()
    $restoredNupkgPath = Join-Path $packageCache "$normalizedId\$normalizedVersion\$normalizedId.$normalizedVersion.nupkg"
    if (-not (Test-Path -LiteralPath $restoredNupkgPath -PathType Leaf)) {
      throw "Restored package cache does not contain the selected nupkg: $restoredNupkgPath"
    }
    $restoredSha256 = (Get-FileHash -LiteralPath $restoredNupkgPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals($restoredSha256, $package.Sha256, [StringComparison]::Ordinal)) {
      throw "Restored package '$($package.Id)' SHA256 does not match the selected isolated-feed package. Expected='$($package.Sha256)' Actual='$restoredSha256'."
    }
    [pscustomobject][ordered]@{
      id = $package.Id
      selectedSha256 = $package.Sha256
      restoredSha256 = $restoredSha256
      matches = $true
    }
  }
)

$consumerOutputDirectory = Join-Path $workspace "bin\Release\net8.0"
$consumerAssemblyPath = Join-Path $consumerOutputDirectory "YoloVision.PackageConsumer.dll"
$nativeBridgePaths = @(Get-ChildItem -LiteralPath $consumerOutputDirectory -Recurse -Filter jyppxtrtbridge.dll -File)
if ($nativeBridgePaths.Count -ne 1) {
  throw "Expected one copied native bridge in consumer output, found $($nativeBridgePaths.Count)."
}

$tensorFileName = if ($isSegmentationScenario) { "dog-yolov8n-seg.fp32.bin" } elseif ($isSemanticScenario) { "dog-lraspp-semantic.fp32.bin" } elseif ($isClassificationScenario) { "bus-yolov8n-cls.fp32.bin" } elseif ($isPoseScenario) { "bus-yolov8n-pose.fp32.bin" } elseif ($isObbScenario) { "boats-yolov8n-obb.fp32.bin" } elseif ($isOfficialDetectionScenario) { "bus-yolov8n-det.fp32.bin" } elseif ($isYoloV10DetectionScenario) { "bus-yolov10n-det.fp32.bin" } else { "dog-yolox-s.fp32.bin" }
$tensorPath = if ($isClassificationScenario) { $ReferenceInputTensorPath } else { Join-Path $runOutput $tensorFileName }
$outputJsonPath = Join-Path $runOutput "yolovision-output.json"
$visualizationPath = Join-Path $runOutput "yolovision-output.svg"
$segmentationMaskDirectory = Join-Path $runOutput "segmentation-masks"
$semanticArtifactDirectory = Join-Path $runOutput "semantic-map-artifacts"
if ($isClassificationScenario) {
  $runArguments = @(
    $consumerAssemblyPath,
    "--model", $ModelPath,
    "--labels", $LabelsPath,
    "--input-data", $ReferenceInputTensorPath,
    "--output-json", $outputJsonPath,
    "--visualization", $visualizationPath,
    "--input-shape", "1x3x224x224",
    "--input-name", "images",
    "--output-name", "output0",
    "--tensor-rt-line", $TensorRtLine,
    "--noTF32",
    "--family", "v8",
    "--task", "cls",
    "--classification-output", "output0",
    "--class-count", "1000",
    "--classification-score-mode", "probabilities",
    "--no-nms",
    "--nms-mode", "none",
    "--confidence", "0",
    "--top-k", "5",
    "--reference-outputs", "output0:$ReferenceOutput0Path",
    "--reference-abs-tolerance", "0.001",
    "--reference-rel-tolerance", "0.001"
  )
}
elseif ($isPoseScenario) {
  $runArguments = @(
    $consumerAssemblyPath,
    "--model", $ModelPath,
    "--labels", $LabelsPath,
    "--image", $ImagePath,
    "--preprocessed-output", $tensorPath,
    "--output-json", $outputJsonPath,
    "--visualization", $visualizationPath,
    "--input-shape", "1x3x640x640",
    "--input-name", "images",
    "--output-name", "output0",
    "--tensor-rt-line", $TensorRtLine,
    "--family", "v8",
    "--task", "pose",
    "--class-count", "1",
    "--layout", "channels-first",
    "--has-objectness", "auto",
    "--nms-mode", "class-aware",
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--top-k", "10",
    "--keypoint-count", "17",
    "--keypoint-stride", "3",
    "--aux-channel-start", "5",
    "--aux-layout", "channels-first",
    "--reference-outputs", "output0:$ReferenceOutput0Path",
    "--reference-abs-tolerance", "1.25",
    "--reference-rel-tolerance", "0.05"
  )
}
elseif ($isObbScenario) {
  $runArguments = @(
    $consumerAssemblyPath,
    "--model", $ModelPath,
    "--labels", $LabelsPath,
    "--image", $ImagePath,
    "--preprocessed-output", $tensorPath,
    "--output-json", $outputJsonPath,
    "--visualization", $visualizationPath,
    "--input-shape", "1x3x1024x1024",
    "--input-name", "images",
    "--output-name", "output0",
    "--tensor-rt-line", $TensorRtLine,
    "--family", "v8",
    "--task", "obb",
    "--class-count", "15",
    "--layout", "channels-first",
    "--has-objectness", "auto",
    "--nms-mode", "class-aware",
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--top-k", "40",
    "--aux-channel-start", "19",
    "--aux-layout", "channels-first",
    "--angle-radians",
    "--reference-outputs", "output0:$ReferenceOutput0Path",
    "--reference-abs-tolerance", "4.25",
    "--reference-rel-tolerance", "0.05"
  )
}
elseif ($isOfficialDetectionScenario) {
  $runArguments = @(
    $consumerAssemblyPath,
    "--model", $ModelPath,
    "--labels", $LabelsPath,
    "--image", $ImagePath,
    "--preprocessed-output", $tensorPath,
    "--output-json", $outputJsonPath,
    "--visualization", $visualizationPath,
    "--input-shape", "1x3x640x640",
    "--input-name", "images",
    "--output-name", "output0",
    "--tensor-rt-line", $TensorRtLine,
    "--family", "v8",
    "--task", "det",
    "--class-count", "80",
    "--layout", "channels-first",
    "--has-objectness", "false",
    "--nms-mode", "class-aware",
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--top-k", "10",
    "--reference-outputs", "output0:$ReferenceOutput0Path",
    "--reference-abs-tolerance", "0.02",
    "--reference-rel-tolerance", "0.05"
  )
}
elseif ($isYoloV10DetectionScenario) {
  $runArguments = @(
    $consumerAssemblyPath,
    "--model", $ModelPath,
    "--labels", $LabelsPath,
    "--image", $ImagePath,
    "--preprocessed-output", $tensorPath,
    "--output-json", $outputJsonPath,
    "--visualization", $visualizationPath,
    "--input-shape", "1x3x640x640",
    "--input-name", "images",
    "--output-name", "output0",
    "--tensor-rt-line", $TensorRtLine,
    "--family", "v10",
    "--task", "det",
    "--layout", "end2end",
    "--class-count", "80",
    "--confidence", "0.25",
    "--top-k", "100"
  )
}
elseif ($isSemanticScenario) {
  $runArguments = @(
    $consumerAssemblyPath,
    "--model", $ModelPath,
    "--labels", $LabelsPath,
    "--image", $ImagePath,
    "--preprocessed-output", $tensorPath,
    "--output-json", $outputJsonPath,
    "--semantic-artifact-output-directory", $semanticArtifactDirectory,
    "--visualization", $visualizationPath,
    "--input-shape", "1x3x320x320",
    "--input-name", "images",
    "--output-name", "semantic",
    "--tensor-rt-line", $TensorRtLine,
    "--noTF32",
    "--family", "custom",
    "--task", "sem",
    "--class-count", "21",
    "--tensor-layout", "NCHW",
    "--color-order", "RGB",
    "--resize", "stretch",
    "--scale", "0.003921568627451",
    "--mean", "0.485,0.456,0.406",
    "--std", "0.229,0.224,0.225",
    "--reference-outputs", "semantic:$ReferenceOutput0Path",
    "--reference-abs-tolerance", "0.0001",
    "--reference-rel-tolerance", "0.0001"
  )
}
elseif ($isSegmentationScenario) {
  $runArguments = @(
    $consumerAssemblyPath,
    "--model", $ModelPath,
    "--labels", $LabelsPath,
    "--image", $ImagePath,
    "--preprocessed-output", $tensorPath,
    "--output-json", $outputJsonPath,
    "--segmentation-mask-output-directory", $segmentationMaskDirectory,
    "--visualization", $visualizationPath,
    "--input-shape", "1x3x640x640",
    "--tensor-rt-line", $TensorRtLine,
    "--family", "v8",
    "--task", "seg",
    "--output-role-map", "output0:det,output1:mask-prototypes",
    "--mask-coefficient-count", "32",
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--top-k", "10",
    "--mask-threshold", "0.5",
    "--mask-spatial-transform",
    "--mask-coordinate-space", "model-input",
    "--mask-crop-to-box", "true",
    "--reference-outputs", "output0:$ReferenceOutput0Path,output1:$ReferenceOutput1Path",
    "--reference-abs-tolerance", "0.02",
    "--reference-rel-tolerance", "0.03"
  )
}
else {
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
}
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
$runResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $runArguments -WorkingDirectory $workspace -Environment $runEnvironment -EnvironmentVariablesToRemove @("JYPPX_NATIVE_BRIDGE_PATH")
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
if ($runResult.Stdout.IndexOf("YoloVisionPackageConsumer ProjectReference=False", [StringComparison]::Ordinal) -lt 0 -or
    $runResult.Stdout.IndexOf("YoloVision Passed=True", [StringComparison]::Ordinal) -lt 0) {
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

$predictions = @()
if ($isClassificationScenario) {
  $predictionPrefix = "Classification Class="
  $predictionLines = @($runResult.Stdout -split "`r?`n" | Where-Object { $_.StartsWith($predictionPrefix, [StringComparison]::Ordinal) })
  $predictions = @(
    foreach ($line in $predictionLines) {
      if ($line -match '^Classification Class=(?<class>.+?) Score=(?<score>[0-9.]+)$') {
        [pscustomobject]@{
          kind = "classification"
          className = $Matches.class
          score = [double]::Parse($Matches.score, [Globalization.CultureInfo]::InvariantCulture)
          line = $line
        }
      }
    }
  )
  $expectedClasses = @("minibus", "police_van", "trolleybus", "golfcart", "jinrikisha")
  $actualClasses = @($predictions | ForEach-Object { $_.className })
  if ($predictions.Count -ne 5 -or ($actualClasses -join ',') -ne ($expectedClasses -join ',')) {
    throw "YOLOv8n-cls package consumer Top-5 order mismatch. Actual='$($actualClasses -join ',')'."
  }
}
elseif ($isPoseScenario) {
  $predictionPrefix = "Pose Class="
  $predictionLines = @($runResult.Stdout -split "`r?`n" | Where-Object { $_.StartsWith($predictionPrefix, [StringComparison]::Ordinal) })
  $predictions = @(
    foreach ($line in $predictionLines) {
      if ($line -match '^Pose Class=(?<class>.+?) Score=(?<score>[0-9.]+) Keypoints=(?<keypoints>[0-9]+)$') {
        [pscustomobject]@{
          kind = "pose"
          className = $Matches.class
          score = [double]::Parse($Matches.score, [Globalization.CultureInfo]::InvariantCulture)
          keypointCount = [int]$Matches.keypoints
          line = $line
        }
      }
    }
  )
  if ($predictions.Count -ne 4 -or
      @($predictions | Where-Object { $_.className -ne "person" -or $_.keypointCount -ne 17 }).Count -ne 0) {
    throw "YOLOv8n-pose package consumer must emit four person poses with 17 keypoints each."
  }
}
elseif ($isObbScenario) {
  $predictionPrefix = "Obb Class="
  $predictionLines = @($runResult.Stdout -split "`r?`n" | Where-Object { $_.StartsWith($predictionPrefix, [StringComparison]::Ordinal) })
  $predictions = @(
    foreach ($line in $predictionLines) {
      if ($line -match '^Obb Class=(?<class>.+?) Score=(?<score>[0-9.]+) AngleRadians=(?<angle>[-0-9.]+)$') {
        [pscustomobject]@{
          kind = "obb"
          className = $Matches.class
          score = [double]::Parse($Matches.score, [Globalization.CultureInfo]::InvariantCulture)
          angleRadians = [double]::Parse($Matches.angle, [Globalization.CultureInfo]::InvariantCulture)
          line = $line
        }
      }
    }
  )
  if ($predictions.Count -ne 40 -or @($predictions | Where-Object { $_.className -ne "ship" }).Count -ne 0) {
    throw "YOLOv8n-obb package consumer must emit exactly 40 ship oriented boxes."
  }
}
elseif ($isSemanticScenario) {
  if ($runResult.Stdout -notmatch 'SemanticMap Classes=(?<classes>[0-9]+) Width=(?<width>[0-9]+) Height=(?<height>[0-9]+) Values=(?<values>[0-9]+)') {
    throw "YoloVision semantic package consumer did not emit the SemanticMap summary."
  }
  $predictions = @(
    [pscustomobject]@{
      kind = "semantic"
      classCount = [int]$Matches.classes
      width = [int]$Matches.width
      height = [int]$Matches.height
      valueCount = [long]$Matches.values
      line = $Matches[0]
    }
  )
  if ($predictions[0].classCount -ne 21 -or $predictions[0].width -ne 320 -or
      $predictions[0].height -ne 320 -or $predictions[0].valueCount -ne 2150400) {
    throw "LRASPP semantic package consumer summary does not match [1,21,320,320]."
  }
}
else {
  $predictionPrefix = if ($isSegmentationScenario) { "Segmentation Class=" } else { "Detection Class=" }
  $predictionKind = if ($isSegmentationScenario) { "segmentation" } else { "detection" }
  $predictionLines = @($runResult.Stdout -split "`r?`n" | Where-Object { $_.StartsWith($predictionPrefix, [StringComparison]::Ordinal) })
  $predictions = @(
    foreach ($line in $predictionLines) {
      if ($line -match '^(?:Detection|Segmentation) Class=(?<class>.+?) Score=(?<score>[0-9.]+) ') {
        [pscustomobject]@{
          kind = $predictionKind
          className = $Matches.class
          score = [double]::Parse($Matches.score, [Globalization.CultureInfo]::InvariantCulture)
          line = $line
        }
      }
    }
  )
}
if ($predictions.Count -eq 0) {
  throw "YoloVision package consumer did not produce a task result."
}
if ($isSegmentationScenario) {
  $expectedClasses = @("dog", "bicycle", "truck", "car")
  $actualClasses = @($predictions | ForEach-Object { $_.className })
  if ($predictions.Count -ne 4 -or @(Compare-Object -ReferenceObject $expectedClasses -DifferenceObject $actualClasses).Count -ne 0) {
    throw "YOLOv8 segmentation package consumer predictions must be dog, bicycle, truck, and car exactly once. Actual='$($actualClasses -join ',')'."
  }
}
elseif ($isOfficialDetectionScenario) {
  $expectedClasses = @("person", "person", "person", "bus", "person")
  $actualClasses = @($predictions | ForEach-Object { $_.className })
  if ($predictions.Count -ne 5 -or ($actualClasses -join ',') -ne ($expectedClasses -join ',')) {
    throw "YOLOv8n detection package consumer predictions must be four persons and one bus in the canonical order. Actual='$($actualClasses -join ',')'."
  }
}
elseif ($isYoloV10DetectionScenario) {
  $actualClasses = @($predictions | ForEach-Object { $_.className })
  if ($predictions.Count -ne 6 -or
      @($actualClasses | Where-Object { $_ -eq "bus" }).Count -ne 1 -or
      @($actualClasses | Where-Object { $_ -eq "person" }).Count -ne 5 -or
      @($actualClasses | Where-Object { $_ -notin @("bus", "person") }).Count -ne 0) {
    throw "YOLOv10n detection package consumer predictions must contain one bus and five persons. Actual='$($actualClasses -join ',')'."
  }
  $topBus = @($predictions | Where-Object { $_.className -eq "bus" } | Sort-Object score -Descending | Select-Object -First 1)
  if ($topBus.Count -ne 1 -or [double]$topBus[0].score -lt 0.94) {
    throw "YOLOv10n detection package consumer must contain a bus score of at least 0.94."
  }
}
$elapsedMilliseconds = 0.0
if ($runResult.Stdout -match 'Execution .* ElapsedMs=(?<elapsed>[0-9.]+)') {
  $elapsedMilliseconds = [double]::Parse($Matches.elapsed, [Globalization.CultureInfo]::InvariantCulture)
}

$expectedRuntimeFiles = @($tensorPath, $outputJsonPath, $visualizationPath)
if ($isSegmentationScenario) {
  $segmentationMaskManifestPath = Join-Path $segmentationMaskDirectory "segmentation-mask-artifacts.manifest.json"
  $expectedRuntimeFiles += $segmentationMaskManifestPath
}
elseif ($isSemanticScenario) {
  $semanticArtifactManifestPath = Join-Path $semanticArtifactDirectory "semantic-map-artifacts.manifest.json"
  $expectedRuntimeFiles += $semanticArtifactManifestPath
}
foreach ($file in $expectedRuntimeFiles) {
  if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
    throw "Expected runtime output was not created: $file"
  }
}
$yoloOutputReport = Get-Content -LiteralPath $outputJsonPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([int]$yoloOutputReport.runtime.tensorRtLine -ne [int]$TensorRtLine) {
  throw "YoloVision output report TensorRT line '$($yoloOutputReport.runtime.tensorRtLine)' does not match requested line '$TensorRtLine'."
}
$referenceComparisons = @()
if ($isClassificationScenario) {
  $actualOutputs = @($yoloOutputReport.outputs)
  if ($actualOutputs.Count -ne 1 -or [string]$actualOutputs[0].name -ne "output0" -or
      (@($actualOutputs[0].shape) -join 'x') -ne "1x1000") {
    throw "YOLOv8n-cls output report must contain output0:[1,1000]."
  }
  if (-not $yoloOutputReport.referenceValidation.requested -or
      -not $yoloOutputReport.referenceValidation.completed -or
      -not $yoloOutputReport.referenceValidation.passed) {
    throw "YOLOv8n-cls raw tensor reference validation did not pass."
  }
  $referenceComparisons = @($yoloOutputReport.referenceValidation.tensorComparisons)
  if ($referenceComparisons.Count -ne 1 -or
      [long]$referenceComparisons[0].comparedElementCount -ne 1000 -or
      [long]$referenceComparisons[0].mismatchCount -ne 0) {
    throw "YOLOv8n-cls raw tensor reference validation must compare 1,000 values with zero mismatches."
  }
  $classificationPredictions = @($yoloOutputReport.predictions | Where-Object { [string]$_.task -eq "cls" })
  if ($classificationPredictions.Count -ne 5 -or
      (@($classificationPredictions | ForEach-Object { [string]$_.className }) -join ',') -ne "minibus,police_van,trolleybus,golfcart,jinrikisha") {
    throw "YOLOv8n-cls output report Top-5 order does not match the independent reference."
  }
  if ([bool]$yoloOutputReport.postprocess.applyNms -or [string]$yoloOutputReport.postprocess.nmsMode -ne "None" -or
      [string]$yoloOutputReport.postprocess.classificationScoreMode -ne "probabilities" -or
      [int]$yoloOutputReport.postprocess.classCount -ne 1000) {
    throw "YOLOv8n-cls output report must declare ApplyNms=False, NmsMode=None, 1,000 classes, and probability scores."
  }
}
elseif ($isPoseScenario) {
  $actualOutputs = @($yoloOutputReport.outputs)
  if ($actualOutputs.Count -ne 1 -or [string]$actualOutputs[0].name -ne "output0" -or
      (@($actualOutputs[0].shape) -join 'x') -ne "1x56x8400") {
    throw "YOLOv8n-pose output report must contain output0:[1,56,8400]."
  }
  if (-not $yoloOutputReport.referenceValidation.requested -or
      -not $yoloOutputReport.referenceValidation.completed -or
      -not $yoloOutputReport.referenceValidation.passed) {
    throw "YOLOv8n-pose raw tensor reference validation did not pass."
  }
  $referenceComparisons = @($yoloOutputReport.referenceValidation.tensorComparisons)
  if ($referenceComparisons.Count -ne 1 -or
      [long]$referenceComparisons[0].comparedElementCount -ne 470400 -or
      [long]$referenceComparisons[0].mismatchCount -ne 0) {
    throw "YOLOv8n-pose raw tensor reference validation must compare 470,400 values with zero mismatches."
  }
  $posePredictions = @($yoloOutputReport.predictions | Where-Object { [string]$_.task -eq "pose" })
  if ($posePredictions.Count -ne 4 -or
      @($posePredictions | Where-Object { [int]$_.classId -ne 0 -or [string]$_.className -ne "person" -or @($_.keypoints).Count -ne 17 }).Count -ne 0) {
    throw "YOLOv8n-pose output report must contain four person poses with 17 keypoints each."
  }
  if (-not [bool]$yoloOutputReport.postprocess.applyNms -or
      [string]$yoloOutputReport.postprocess.nmsMode -ne "ClassAware" -or
      [string]$yoloOutputReport.postprocess.layout -ne "ChannelsFirst") {
    throw "YOLOv8n-pose output report must declare class-aware NMS and channels-first layout."
  }
  if ([string]$yoloOutputReport.input.preprocessedTensor.sha256 -ne "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" -or
      [long]$yoloOutputReport.input.preprocessedTensor.elementCount -ne 1228800) {
    throw "YOLOv8n-pose C# preprocessing tensor does not match the authoritative input tensor."
  }
}
elseif ($isObbScenario) {
  $actualOutputs = @($yoloOutputReport.outputs)
  if ($actualOutputs.Count -ne 1 -or [string]$actualOutputs[0].name -ne "output0" -or
      (@($actualOutputs[0].shape) -join 'x') -ne "1x20x21504") {
    throw "YOLOv8n-obb output report must contain output0:[1,20,21504]."
  }
  if (-not $yoloOutputReport.referenceValidation.requested -or
      -not $yoloOutputReport.referenceValidation.completed -or
      -not $yoloOutputReport.referenceValidation.passed) {
    throw "YOLOv8n-obb raw tensor reference validation did not pass."
  }
  $referenceComparisons = @($yoloOutputReport.referenceValidation.tensorComparisons)
  if ($referenceComparisons.Count -ne 1 -or
      [long]$referenceComparisons[0].comparedElementCount -ne 430080 -or
      [long]$referenceComparisons[0].mismatchCount -ne 0) {
    throw "YOLOv8n-obb raw tensor reference validation must compare 430,080 values with zero mismatches."
  }
  $obbPredictions = @($yoloOutputReport.predictions | Where-Object { [string]$_.task -eq "obb" })
  if ($obbPredictions.Count -ne 40 -or
      @($obbPredictions | Where-Object { [int]$_.classId -ne 1 -or [string]$_.className -ne "ship" -or [string]$_.angleUnit -ne "radian" }).Count -ne 0) {
    throw "YOLOv8n-obb output report must contain exactly 40 ship boxes with radian angles."
  }
  if (-not [bool]$yoloOutputReport.postprocess.applyNms -or
      [string]$yoloOutputReport.postprocess.nmsMode -ne "ClassAware" -or
      [string]$yoloOutputReport.postprocess.layout -ne "ChannelsFirst") {
    throw "YOLOv8n-obb output report must declare class-aware NMS and channels-first layout."
  }
  if ([string]$yoloOutputReport.input.preprocessedTensor.sha256 -ne "c56c027619088bce94f9160a3f602b4ad81fe323001867b1ee456100040fec6e" -or
      [long]$yoloOutputReport.input.preprocessedTensor.elementCount -ne 3145728) {
    throw "YOLOv8n-obb C# preprocessing tensor does not match the authoritative input tensor."
  }
}
elseif ($isOfficialDetectionScenario) {
  $actualOutputs = @($yoloOutputReport.outputs)
  if ($actualOutputs.Count -ne 1 -or [string]$actualOutputs[0].name -ne "output0" -or
      (@($actualOutputs[0].shape) -join 'x') -ne "1x84x8400") {
    throw "YOLOv8n detection output report must contain output0:[1,84,8400]."
  }
  if (-not $yoloOutputReport.referenceValidation.requested -or
      -not $yoloOutputReport.referenceValidation.completed -or
      -not $yoloOutputReport.referenceValidation.passed) {
    throw "YOLOv8n detection raw tensor reference validation did not pass."
  }
  $referenceComparisons = @($yoloOutputReport.referenceValidation.tensorComparisons)
  if ($referenceComparisons.Count -ne 1 -or
      [long]$referenceComparisons[0].comparedElementCount -ne 705600 -or
      [long]$referenceComparisons[0].mismatchCount -ne 0) {
    throw "YOLOv8n detection raw tensor reference validation must compare 705,600 values with zero mismatches."
  }
  $detectionPredictions = @($yoloOutputReport.predictions | Where-Object { [string]$_.task -eq "det" })
  $actualDetectionClasses = @($detectionPredictions | ForEach-Object { [string]$_.className })
  if ($detectionPredictions.Count -ne 5 -or ($actualDetectionClasses -join ',') -ne "person,person,person,bus,person") {
    throw "YOLOv8n detection output report must contain four persons and one bus in the canonical order."
  }
  if (-not [bool]$yoloOutputReport.postprocess.applyNms -or
      [string]$yoloOutputReport.postprocess.nmsMode -ne "ClassAware" -or
      [string]$yoloOutputReport.postprocess.layout -ne "ChannelsFirst" -or
      [int]$yoloOutputReport.postprocess.classCount -ne 80) {
    throw "YOLOv8n detection output report must declare 80 classes, class-aware NMS, and channels-first layout."
  }
  if ([string]$yoloOutputReport.input.preprocessedTensor.sha256 -ne "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" -or
      [long]$yoloOutputReport.input.preprocessedTensor.elementCount -ne 1228800) {
    throw "YOLOv8n detection C# preprocessing tensor does not match the authoritative input tensor."
  }
}
elseif ($isYoloV10DetectionScenario) {
  $actualOutputs = @($yoloOutputReport.outputs)
  if ($actualOutputs.Count -ne 1 -or [string]$actualOutputs[0].name -ne "output0" -or
      (@($actualOutputs[0].shape) -join 'x') -ne "1x300x6") {
    throw "YOLOv10n detection output report must contain output0:[1,300,6]."
  }
  if ([bool]$yoloOutputReport.referenceValidation.requested) {
    throw "YOLOv10n local package evidence must not claim an independent raw tensor reference."
  }
  $detectionPredictions = @($yoloOutputReport.predictions | Where-Object { [string]$_.task -eq "det" })
  $detectionClasses = @($detectionPredictions | ForEach-Object { [string]$_.className })
  if ($detectionPredictions.Count -ne 6 -or
      @($detectionClasses | Where-Object { $_ -eq "bus" }).Count -ne 1 -or
      @($detectionClasses | Where-Object { $_ -eq "person" }).Count -ne 5 -or
      @($detectionClasses | Where-Object { $_ -notin @("bus", "person") }).Count -ne 0) {
    throw "YOLOv10n detection output report must contain one bus and five persons."
  }
  if ([bool]$yoloOutputReport.postprocess.applyNms -or
      [string]$yoloOutputReport.postprocess.nmsMode -ne "None" -or
      [string]$yoloOutputReport.postprocess.layout -ne "EndToEndNms" -or
      [int]$yoloOutputReport.postprocess.classCount -ne 80) {
    throw "YOLOv10n detection output report must declare 80 classes, no application-side NMS, and end-to-end layout."
  }
  if ([string]$yoloOutputReport.input.preprocessedTensor.sha256 -ne "050935ebf471ec32ab4327d9f5643f0fe1a203289088895205e732e448a8d225" -or
      [long]$yoloOutputReport.input.preprocessedTensor.elementCount -ne 1228800) {
    throw "YOLOv10n C# preprocessing tensor does not match the pinned CC0 input contract."
  }
}
elseif ($isSemanticScenario) {
  $actualOutputs = @($yoloOutputReport.outputs)
  if ($actualOutputs.Count -ne 1 -or [string]$actualOutputs[0].name -ne "semantic" -or
      (@($actualOutputs[0].shape) -join 'x') -ne "1x21x320x320") {
    throw "LRASPP semantic output report must contain semantic:[1,21,320,320]."
  }
  if (-not $yoloOutputReport.referenceValidation.requested -or
      -not $yoloOutputReport.referenceValidation.completed -or
      -not $yoloOutputReport.referenceValidation.passed) {
    throw "LRASPP semantic raw tensor reference validation did not pass."
  }
  $referenceComparisons = @($yoloOutputReport.referenceValidation.tensorComparisons)
  if ($referenceComparisons.Count -ne 1 -or
      [long]$referenceComparisons[0].comparedElementCount -ne 2150400 -or
      [long]$referenceComparisons[0].mismatchCount -ne 0) {
    throw "LRASPP semantic raw tensor reference validation must compare 2,150,400 values with zero mismatches."
  }
  $semanticPredictions = @($yoloOutputReport.predictions | Where-Object { [string]$_.task -eq "sem" })
  if ($semanticPredictions.Count -ne 1 -or [int]$semanticPredictions[0].classCount -ne 21 -or
      [long]$semanticPredictions[0].classIndexValueCount -ne 102400) {
    throw "LRASPP semantic output report must contain one 21-class, 102,400-pixel summary."
  }
}
elseif ($isSegmentationScenario) {
  $expectedOutputShapes = @{
    output0 = "1x116x8400"
    output1 = "1x32x160x160"
  }
  $actualOutputs = @($yoloOutputReport.outputs)
  if ($actualOutputs.Count -ne 2) {
    throw "YOLOv8 segmentation output report must contain exactly two outputs. Found $($actualOutputs.Count)."
  }
  foreach ($output in $actualOutputs) {
    $shapeText = (@($output.shape) -join 'x')
    if (-not $expectedOutputShapes.ContainsKey([string]$output.name) -or
        -not [string]::Equals($shapeText, $expectedOutputShapes[[string]$output.name], [StringComparison]::Ordinal)) {
      throw "Unexpected YOLOv8 segmentation output contract: $($output.name):$shapeText."
    }
  }

  if (-not $yoloOutputReport.referenceValidation.requested -or
      -not $yoloOutputReport.referenceValidation.completed -or
      -not $yoloOutputReport.referenceValidation.passed) {
    throw "YOLOv8 segmentation raw tensor reference validation did not pass."
  }
  $referenceComparisons = @($yoloOutputReport.referenceValidation.tensorComparisons)
  if ($referenceComparisons.Count -ne 2 -or
      ($referenceComparisons | Measure-Object comparedElementCount -Sum).Sum -ne 1793600 -or
      ($referenceComparisons | Measure-Object mismatchCount -Sum).Sum -ne 0) {
    throw "YOLOv8 segmentation raw tensor reference validation must compare 1,793,600 values with zero mismatches."
  }
}
$copiedOutputJson = Join-Path $ReportDirectory "yolovision-output.json"
$copiedVisualization = Join-Path $ReportDirectory "yolovision-output.svg"
Copy-Item -LiteralPath $outputJsonPath -Destination $copiedOutputJson -Force
Copy-Item -LiteralPath $visualizationPath -Destination $copiedVisualization -Force

$archivedMaskManifestPath = $null
$runtimeMaskManifestSha256 = $null
$archivedMaskManifestSha256 = $null
$independentReference = $null
$independentComparison = $null
$independentReferencePath = $null
$independentComparisonPath = $null
$independentRuntimeResult = $null
$controlledReferenceResult = $null
$controlledReferenceReport = $null
$controlledReferenceMutationSha256 = $null
$controlledMaskResult = $null
$controlledMaskOriginalSha256 = $null
$controlledMaskMutatedSha256 = $null
$controlledMaskExpectedSha256 = $null
$semanticArtifactValidation = $null
$semanticArtifactValidationResult = $null
$semanticArtifactValidationPath = $null
$archivedSemanticArtifactDirectory = $null
$archivedSemanticManifestPath = $null
$semanticArtifactManifestSha256 = $null
$semanticClassIndexSha256 = $null
$controlledSemanticArtifactResult = $null
$controlledSemanticArtifactValidation = $null
$controlledSemanticOriginalSha256 = $null
$controlledSemanticMutatedSha256 = $null

if ($isSegmentationScenario) {
  $runtimeMaskManifestSha256 = (Get-FileHash -LiteralPath $segmentationMaskManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $archivedMaskDirectory = Join-Path $ReportDirectory "segmentation-masks"
  if (Test-Path -LiteralPath $archivedMaskDirectory) {
    Remove-Item -LiteralPath $archivedMaskDirectory -Recurse -Force
  }
  Copy-Item -LiteralPath $segmentationMaskDirectory -Destination $archivedMaskDirectory -Recurse -Force
  $archivedMaskManifestPath = Join-Path $archivedMaskDirectory "segmentation-mask-artifacts.manifest.json"
  $archivedMaskManifest = Get-Content -LiteralPath $archivedMaskManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  foreach ($prediction in @($archivedMaskManifest.predictions)) {
    foreach ($propertyName in @("prototypeProbability", "sourceProbability", "sourceThresholded")) {
      $artifact = $prediction.$propertyName
      if ($null -ne $artifact) {
        $artifact.path = Join-Path $archivedMaskDirectory ([string]$artifact.fileName)
      }
    }
  }
  [IO.File]::WriteAllText($archivedMaskManifestPath, ($archivedMaskManifest | ConvertTo-Json -Depth 16), $utf8)
  $archivedMaskManifestSha256 = (Get-FileHash -LiteralPath $archivedMaskManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

  $independentDirectory = Join-Path $ReportDirectory "independent-pytorch-reference"
  if (Test-Path -LiteralPath $independentDirectory) {
    Remove-Item -LiteralPath $independentDirectory -Recurse -Force
  }
  New-Item -ItemType Directory -Path $independentDirectory -Force | Out-Null
  $independentArguments = @(
    (Join-Path $RepositoryRoot "eng\Invoke-YoloVisionSegmentationReference.py"),
    "--model", $ModelWeightsPath,
    "--image", $ImagePath,
    "--output-directory", $independentDirectory,
    "--actual-manifest", $archivedMaskManifestPath,
    "--image-size", "640",
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--max-detections", "10",
    "--mask-threshold", "0.5",
    "--maximum-box-coordinate-error", "1.0",
    "--maximum-score-error", "0.01",
    "--minimum-box-iou", "0.995",
    "--minimum-mask-iou", "0.99",
    "--evidence-classification", "local-package-consumer-runtime"
  )
  $independentRuntimeResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments $independentArguments -WorkingDirectory $RepositoryRoot
  $independentStdoutPath = Join-Path $ReportDirectory "independent-pytorch.stdout.log"
  $independentStderrPath = Join-Path $ReportDirectory "independent-pytorch.stderr.log"
  [IO.File]::WriteAllText($independentStdoutPath, $independentRuntimeResult.Stdout, $utf8)
  [IO.File]::WriteAllText($independentStderrPath, $independentRuntimeResult.Stderr, $utf8)
  if ($independentRuntimeResult.ExitCode -ne 0) {
    throw "Independent Ultralytics/PyTorch comparison failed with exit code $($independentRuntimeResult.ExitCode). See $independentStderrPath"
  }
  $independentReferencePath = Join-Path $independentDirectory "ultralytics-pytorch-reference.json"
  $independentComparisonPath = Join-Path $independentDirectory "yolovision-independent-comparison.json"
  $independentReference = Get-Content -LiteralPath $independentReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
  $independentComparison = Get-Content -LiteralPath $independentComparisonPath -Raw -Encoding utf8 | ConvertFrom-Json
  if (-not $independentComparison.completed -or -not $independentComparison.passed -or
      [int]$independentComparison.actualPredictionCount -ne 4 -or
      @($independentComparison.comparisons | Where-Object { -not $_.passed }).Count -ne 0) {
    throw "Independent Ultralytics/PyTorch mask comparison did not pass all four predictions."
  }

  $controlledReferenceDirectory = Join-Path $runOutput "controlled-reference-negative"
  New-Item -ItemType Directory -Path $controlledReferenceDirectory -Force | Out-Null
  $controlledReferencePath = Join-Path $controlledReferenceDirectory "output0.single-value-mutated.reference.json"
  $mutationResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments @(
    (Join-Path $RepositoryRoot "eng\New-YoloVisionReferenceMutation.py"),
    "--input", $ReferenceOutput0Path,
    "--output", $controlledReferencePath,
    "--index", "0",
    "--delta", "10000"
  ) -WorkingDirectory $RepositoryRoot
  $mutationLogPath = Join-Path $ReportDirectory "controlled-reference-mutation.log"
  [IO.File]::WriteAllText($mutationLogPath, ($mutationResult.Stdout + $mutationResult.Stderr), $utf8)
  if ($mutationResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $controlledReferencePath -PathType Leaf)) {
    throw "Failed to create the controlled raw-reference mutation. See $mutationLogPath"
  }
  $controlledReferenceMutationSha256 = (Get-FileHash -LiteralPath $controlledReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()

  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $runArguments -Name "--reference-outputs" -Value "output0:$controlledReferencePath,output1:$ReferenceOutput1Path"
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--output-json" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.json")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--visualization" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.svg")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--segmentation-mask-output-directory" -Value (Join-Path $controlledReferenceDirectory "segmentation-masks")
  $controlledReferenceResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $controlledReferenceArguments -WorkingDirectory $workspace -Environment $runEnvironment -EnvironmentVariablesToRemove @("JYPPX_NATIVE_BRIDGE_PATH")
  $controlledReferenceStdoutPath = Join-Path $ReportDirectory "controlled-reference.stdout.log"
  $controlledReferenceStderrPath = Join-Path $ReportDirectory "controlled-reference.stderr.log"
  [IO.File]::WriteAllText($controlledReferenceStdoutPath, $controlledReferenceResult.Stdout, $utf8)
  [IO.File]::WriteAllText($controlledReferenceStderrPath, $controlledReferenceResult.Stderr, $utf8)
  $controlledReferenceOutputPath = Join-Path $controlledReferenceDirectory "yolovision-output.json"
  if ($controlledReferenceResult.ExitCode -ne 1 -or
      $controlledReferenceResult.Stdout.IndexOf("YoloVision Passed=False", [StringComparison]::Ordinal) -lt 0 -or
      -not (Test-Path -LiteralPath $controlledReferenceOutputPath -PathType Leaf)) {
    throw "Controlled raw-reference mutation must fail closed and still emit its diagnostic report."
  }
  $controlledReferenceReport = Get-Content -LiteralPath $controlledReferenceOutputPath -Raw -Encoding utf8 | ConvertFrom-Json
  $controlledReferenceComparisons = @($controlledReferenceReport.referenceValidation.tensorComparisons)
  $controlledOutput0 = @($controlledReferenceComparisons | Where-Object { $_.tensorName -eq "output0" })
  if ($controlledOutput0.Count -ne 1 -or [int]$controlledOutput0[0].mismatchCount -ne 1 -or
      [long]$controlledOutput0[0].firstMismatchIndex -ne 0 -or $controlledReferenceReport.referenceValidation.passed) {
    throw "Controlled raw-reference mutation did not produce exactly one output0 mismatch at index zero."
  }
  Copy-Item -LiteralPath $controlledReferenceOutputPath -Destination (Join-Path $ReportDirectory "controlled-reference-output.json") -Force

  $controlledMaskDirectory = Join-Path $ReportDirectory "controlled-mask-tamper"
  if (Test-Path -LiteralPath $controlledMaskDirectory) {
    Remove-Item -LiteralPath $controlledMaskDirectory -Recurse -Force
  }
  Copy-Item -LiteralPath $archivedMaskDirectory -Destination $controlledMaskDirectory -Recurse -Force
  $controlledMaskManifestPath = Join-Path $controlledMaskDirectory "segmentation-mask-artifacts.manifest.json"
  $controlledMaskManifest = Get-Content -LiteralPath $controlledMaskManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  foreach ($prediction in @($controlledMaskManifest.predictions)) {
    foreach ($propertyName in @("prototypeProbability", "sourceProbability", "sourceThresholded")) {
      $artifact = $prediction.$propertyName
      if ($null -ne $artifact) {
        $artifact.path = Join-Path $controlledMaskDirectory ([string]$artifact.fileName)
      }
    }
  }
  [IO.File]::WriteAllText($controlledMaskManifestPath, ($controlledMaskManifest | ConvertTo-Json -Depth 16), $utf8)
  $controlledMaskArtifact = $controlledMaskManifest.predictions[0].sourceThresholded
  $controlledMaskPath = [string]$controlledMaskArtifact.path
  $controlledMaskExpectedSha256 = [string]$controlledMaskArtifact.sha256
  $controlledMaskOriginalSha256 = (Get-FileHash -LiteralPath $controlledMaskPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $controlledMaskBytes = [IO.File]::ReadAllBytes($controlledMaskPath)
  if ($controlledMaskBytes.Length -eq 0) {
    throw "Controlled thresholded mask is empty."
  }
  $controlledMaskBytes[0] = if ($controlledMaskBytes[0] -eq 0) { 1 } else { 0 }
  [IO.File]::WriteAllBytes($controlledMaskPath, $controlledMaskBytes)
  $controlledMaskMutatedSha256 = (Get-FileHash -LiteralPath $controlledMaskPath -Algorithm SHA256).Hash.ToLowerInvariant()
  if (-not [string]::Equals($controlledMaskOriginalSha256, $controlledMaskExpectedSha256, [StringComparison]::Ordinal) -or
      [string]::Equals($controlledMaskMutatedSha256, $controlledMaskExpectedSha256, [StringComparison]::Ordinal)) {
    throw "Controlled mask mutation did not preserve the original manifest digest while changing one byte."
  }

  $controlledMaskReferenceDirectory = Join-Path $runOutput "controlled-mask-reference"
  $controlledMaskArguments = @($independentArguments)
  $controlledMaskArguments = Set-NamedArgumentValue -Arguments $controlledMaskArguments -Name "--output-directory" -Value $controlledMaskReferenceDirectory
  $controlledMaskArguments = Set-NamedArgumentValue -Arguments $controlledMaskArguments -Name "--actual-manifest" -Value $controlledMaskManifestPath
  $controlledMaskResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments $controlledMaskArguments -WorkingDirectory $RepositoryRoot
  $controlledMaskStdoutPath = Join-Path $ReportDirectory "controlled-mask.stdout.log"
  $controlledMaskStderrPath = Join-Path $ReportDirectory "controlled-mask.stderr.log"
  [IO.File]::WriteAllText($controlledMaskStdoutPath, $controlledMaskResult.Stdout, $utf8)
  [IO.File]::WriteAllText($controlledMaskStderrPath, $controlledMaskResult.Stderr, $utf8)
  if ($controlledMaskResult.ExitCode -ne 1 -or
      $controlledMaskResult.Stderr.IndexOf("Thresholded mask SHA256 does not match the manifest.", [StringComparison]::Ordinal) -lt 0) {
    throw "Controlled mask mutation did not fail closed on the manifest SHA256 mismatch."
  }
}
elseif ($isPoseScenario) {
  Assert-FileSha256 -Path $tensorPath -ExpectedSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" -Description "YOLOv8n-pose generated input tensor"

  $independentDirectory = Join-Path $ReportDirectory "independent-pose-reference"
  if (Test-Path -LiteralPath $independentDirectory) {
    Remove-Item -LiteralPath $independentDirectory -Recurse -Force
  }
  New-Item -ItemType Directory -Path $independentDirectory -Force | Out-Null
  $independentArguments = @(
    (Join-Path $RepositoryRoot "eng\Invoke-YoloVisionPoseReference.py"),
    "--onnx-model", $ModelPath,
    "--weights", $ModelWeightsPath,
    "--input-tensor", $tensorPath,
    "--image", $poseSourceImagePath,
    "--output-directory", $independentDirectory,
    "--actual-output", $copiedOutputJson,
    "--input-name", "images",
    "--output-name", "output0",
    "--input-shape", "1x3x640x640",
    "--output-shape", "1x56x8400",
    "--image-size", "640",
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--max-detections", "10",
    "--keypoint-count", "17",
    "--minimum-box-iou", "0.98",
    "--maximum-score-error", "0.03",
    "--minimum-keypoint-score", "0.25",
    "--maximum-keypoint-coordinate-error", "5.0",
    "--maximum-keypoint-score-error", "0.03"
  )
  $independentRuntimeResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments $independentArguments -WorkingDirectory $RepositoryRoot
  $independentStdoutPath = Join-Path $ReportDirectory "independent-pose.stdout.log"
  $independentStderrPath = Join-Path $ReportDirectory "independent-pose.stderr.log"
  [IO.File]::WriteAllText($independentStdoutPath, $independentRuntimeResult.Stdout, $utf8)
  [IO.File]::WriteAllText($independentStderrPath, $independentRuntimeResult.Stderr, $utf8)
  if ($independentRuntimeResult.ExitCode -ne 0) {
    throw "Independent Ultralytics/PyTorch pose comparison failed with exit code $($independentRuntimeResult.ExitCode). See $independentStderrPath"
  }
  $independentRawReferencePath = Join-Path $independentDirectory "output0.reference.json"
  $independentReferencePath = Join-Path $independentDirectory "ultralytics-pytorch-reference.json"
  $independentComparisonPath = Join-Path $independentDirectory "yolovision-independent-comparison.json"
  Assert-FileSha256 -Path $independentRawReferencePath -ExpectedSha256 "73752d18797b5c9336359b99eef4cadfb8d263e5925fd489648ce3072ef440f1" -Description "Regenerated YOLOv8n-pose raw reference"
  Assert-FileSha256 -Path $independentReferencePath -ExpectedSha256 "771f90902bc7b16acaef8b029035c9d4f4a8981526bed278cdc114e2e6872192" -Description "Regenerated Ultralytics pose reference"
  $independentReference = Get-Content -LiteralPath $independentReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
  $independentComparison = Get-Content -LiteralPath $independentComparisonPath -Raw -Encoding utf8 | ConvertFrom-Json
  if (-not $independentComparison.completed -or -not $independentComparison.passed -or
      [int]$independentComparison.referencePredictionCount -ne 4 -or
      [int]$independentComparison.actualPredictionCount -ne 4 -or
      @($independentComparison.comparisons | Where-Object { -not $_.passed }).Count -ne 0) {
    throw "Independent Ultralytics/PyTorch pose comparison did not pass all four predictions."
  }

  $controlledReferenceDirectory = Join-Path $runOutput "controlled-reference-negative"
  New-Item -ItemType Directory -Path $controlledReferenceDirectory -Force | Out-Null
  $controlledReferencePath = Join-Path $controlledReferenceDirectory "output0.single-value-mutated.reference.json"
  $mutationResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments @(
    (Join-Path $RepositoryRoot "eng\New-YoloVisionReferenceMutation.py"),
    "--input", $ReferenceOutput0Path,
    "--output", $controlledReferencePath,
    "--index", "0",
    "--delta", "10000"
  ) -WorkingDirectory $RepositoryRoot
  $mutationLogPath = Join-Path $ReportDirectory "controlled-reference-mutation.log"
  [IO.File]::WriteAllText($mutationLogPath, ($mutationResult.Stdout + $mutationResult.Stderr), $utf8)
  if ($mutationResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $controlledReferencePath -PathType Leaf)) {
    throw "Failed to create the controlled pose raw-reference mutation. See $mutationLogPath"
  }
  $controlledReferenceMutationSha256 = (Get-FileHash -LiteralPath $controlledReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()

  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $runArguments -Name "--reference-outputs" -Value "output0:$controlledReferencePath"
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--output-json" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.json")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--visualization" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.svg")
  $controlledReferenceResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $controlledReferenceArguments -WorkingDirectory $workspace -Environment $runEnvironment -EnvironmentVariablesToRemove @("JYPPX_NATIVE_BRIDGE_PATH")
  $controlledReferenceStdoutPath = Join-Path $ReportDirectory "controlled-reference.stdout.log"
  $controlledReferenceStderrPath = Join-Path $ReportDirectory "controlled-reference.stderr.log"
  [IO.File]::WriteAllText($controlledReferenceStdoutPath, $controlledReferenceResult.Stdout, $utf8)
  [IO.File]::WriteAllText($controlledReferenceStderrPath, $controlledReferenceResult.Stderr, $utf8)
  $controlledReferenceOutputPath = Join-Path $controlledReferenceDirectory "yolovision-output.json"
  if ($controlledReferenceResult.ExitCode -ne 1 -or
      $controlledReferenceResult.Stdout.IndexOf("YoloVision Passed=False", [StringComparison]::Ordinal) -lt 0 -or
      -not (Test-Path -LiteralPath $controlledReferenceOutputPath -PathType Leaf)) {
    throw "Controlled pose raw-reference mutation must fail closed and emit its diagnostic report."
  }
  $controlledReferenceReport = Get-Content -LiteralPath $controlledReferenceOutputPath -Raw -Encoding utf8 | ConvertFrom-Json
  $controlledReferenceComparisons = @($controlledReferenceReport.referenceValidation.tensorComparisons)
  if ($controlledReferenceComparisons.Count -ne 1 -or [long]$controlledReferenceComparisons[0].mismatchCount -ne 1 -or
      [long]$controlledReferenceComparisons[0].firstMismatchIndex -ne 0 -or $controlledReferenceReport.referenceValidation.passed) {
    throw "Controlled pose raw-reference mutation did not produce one mismatch at index zero."
  }
  Copy-Item -LiteralPath $controlledReferenceOutputPath -Destination (Join-Path $ReportDirectory "controlled-reference-output.json") -Force
}
elseif ($isObbScenario) {
  Assert-FileSha256 -Path $tensorPath -ExpectedSha256 "c56c027619088bce94f9160a3f602b4ad81fe323001867b1ee456100040fec6e" -Description "YOLOv8n-obb generated input tensor"

  $independentDirectory = Join-Path $ReportDirectory "independent-obb-reference"
  if (Test-Path -LiteralPath $independentDirectory) {
    Remove-Item -LiteralPath $independentDirectory -Recurse -Force
  }
  New-Item -ItemType Directory -Path $independentDirectory -Force | Out-Null
  $independentArguments = @(
    (Join-Path $RepositoryRoot "eng\Invoke-YoloVisionObbReference.py"),
    "--onnx-model", $ModelPath,
    "--weights", $ModelWeightsPath,
    "--image", $obbSourceImagePath,
    "--input-tensor", $tensorPath,
    "--output-directory", $independentDirectory,
    "--actual-output", $copiedOutputJson,
    "--input-name", "images",
    "--output-name", "output0",
    "--input-shape", "1", "3", "1024", "1024",
    "--output-shape", "1", "20", "21504",
    "--image-size", "1024",
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--max-detections", "40",
    "--minimum-rotated-iou", "0.98",
    "--maximum-coordinate-error", "5.0",
    "--maximum-angle-error", "0.02",
    "--maximum-score-error", "0.03"
  )
  $independentRuntimeResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments $independentArguments -WorkingDirectory $RepositoryRoot
  $independentStdoutPath = Join-Path $ReportDirectory "independent-obb.stdout.log"
  $independentStderrPath = Join-Path $ReportDirectory "independent-obb.stderr.log"
  [IO.File]::WriteAllText($independentStdoutPath, $independentRuntimeResult.Stdout, $utf8)
  [IO.File]::WriteAllText($independentStderrPath, $independentRuntimeResult.Stderr, $utf8)
  if ($independentRuntimeResult.ExitCode -ne 0) {
    throw "Independent Ultralytics/PyTorch OBB comparison failed with exit code $($independentRuntimeResult.ExitCode). See $independentStderrPath"
  }
  $independentRawReferencePath = Join-Path $independentDirectory "output0.reference.json"
  $independentReferencePath = Join-Path $independentDirectory "ultralytics-pytorch-reference.json"
  $independentComparisonPath = Join-Path $independentDirectory "yolovision-independent-comparison.json"
  Assert-FileSha256 -Path $independentRawReferencePath -ExpectedSha256 "53f00a488c44227a3e5fbc86b530355acfd0c17cf9fdfc771a6d1f99f72fd65d" -Description "Regenerated YOLOv8n-obb raw reference"
  $independentReference = Get-Content -LiteralPath $independentReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
  $pinnedIndependentReference = Get-Content -LiteralPath $obbPinnedIndependentReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
  $independentComparison = Get-Content -LiteralPath $independentComparisonPath -Raw -Encoding utf8 | ConvertFrom-Json
  $regeneratedPredictionJson = @($independentReference.predictions) | ConvertTo-Json -Depth 12 -Compress
  $pinnedPredictionJson = @($pinnedIndependentReference.predictions) | ConvertTo-Json -Depth 12 -Compress
  if (-not [string]::Equals($regeneratedPredictionJson, $pinnedPredictionJson, [StringComparison]::Ordinal) -or
      [string]$independentReference.annotatedImage.sha256 -ne "7231b76afc3d823e4dafb41c3c37f8e150c58c842f1c7334def386fa1c686ce1") {
    throw "Regenerated Ultralytics/PyTorch OBB predictions or annotated image do not match the pinned independent reference."
  }
  if (-not $independentComparison.completed -or -not $independentComparison.passed -or
      [int]$independentComparison.referencePredictionCount -ne 40 -or
      [int]$independentComparison.actualPredictionCount -ne 40 -or
      @($independentComparison.comparisons | Where-Object { -not $_.passed }).Count -ne 0) {
    throw "Independent Ultralytics/PyTorch OBB comparison did not pass all 40 predictions."
  }

  $controlledReferenceDirectory = Join-Path $runOutput "controlled-reference-negative"
  New-Item -ItemType Directory -Path $controlledReferenceDirectory -Force | Out-Null
  $controlledReferencePath = Join-Path $controlledReferenceDirectory "output0.single-value-mutated.reference.json"
  $mutationResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments @(
    (Join-Path $RepositoryRoot "eng\New-YoloVisionReferenceMutation.py"),
    "--input", $ReferenceOutput0Path,
    "--output", $controlledReferencePath,
    "--index", "0",
    "--delta", "10000"
  ) -WorkingDirectory $RepositoryRoot
  $mutationLogPath = Join-Path $ReportDirectory "controlled-reference-mutation.log"
  [IO.File]::WriteAllText($mutationLogPath, ($mutationResult.Stdout + $mutationResult.Stderr), $utf8)
  if ($mutationResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $controlledReferencePath -PathType Leaf)) {
    throw "Failed to create the controlled OBB raw-reference mutation. See $mutationLogPath"
  }
  $controlledReferenceMutationSha256 = (Get-FileHash -LiteralPath $controlledReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()

  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $runArguments -Name "--reference-outputs" -Value "output0:$controlledReferencePath"
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--output-json" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.json")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--visualization" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.svg")
  $controlledReferenceResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $controlledReferenceArguments -WorkingDirectory $workspace -Environment $runEnvironment -EnvironmentVariablesToRemove @("JYPPX_NATIVE_BRIDGE_PATH")
  $controlledReferenceStdoutPath = Join-Path $ReportDirectory "controlled-reference.stdout.log"
  $controlledReferenceStderrPath = Join-Path $ReportDirectory "controlled-reference.stderr.log"
  [IO.File]::WriteAllText($controlledReferenceStdoutPath, $controlledReferenceResult.Stdout, $utf8)
  [IO.File]::WriteAllText($controlledReferenceStderrPath, $controlledReferenceResult.Stderr, $utf8)
  $controlledReferenceOutputPath = Join-Path $controlledReferenceDirectory "yolovision-output.json"
  if ($controlledReferenceResult.ExitCode -ne 1 -or
      $controlledReferenceResult.Stdout.IndexOf("YoloVision Passed=False", [StringComparison]::Ordinal) -lt 0 -or
      -not (Test-Path -LiteralPath $controlledReferenceOutputPath -PathType Leaf)) {
    throw "Controlled OBB raw-reference mutation must fail closed and emit its diagnostic report."
  }
  $controlledReferenceReport = Get-Content -LiteralPath $controlledReferenceOutputPath -Raw -Encoding utf8 | ConvertFrom-Json
  $controlledReferenceComparisons = @($controlledReferenceReport.referenceValidation.tensorComparisons)
  if ($controlledReferenceComparisons.Count -ne 1 -or [long]$controlledReferenceComparisons[0].mismatchCount -ne 1 -or
      [long]$controlledReferenceComparisons[0].firstMismatchIndex -ne 0 -or $controlledReferenceReport.referenceValidation.passed) {
    throw "Controlled OBB raw-reference mutation did not produce one mismatch at index zero."
  }
  Copy-Item -LiteralPath $controlledReferenceOutputPath -Destination (Join-Path $ReportDirectory "controlled-reference-output.json") -Force
}
elseif ($isOfficialDetectionScenario) {
  Assert-FileSha256 -Path $tensorPath -ExpectedSha256 "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d" -Description "YOLOv8n detection generated input tensor"

  $independentDirectory = Join-Path $ReportDirectory "independent-detection-reference"
  if (Test-Path -LiteralPath $independentDirectory) {
    Remove-Item -LiteralPath $independentDirectory -Recurse -Force
  }
  New-Item -ItemType Directory -Path $independentDirectory -Force | Out-Null
  $independentArguments = @(
    (Join-Path $RepositoryRoot "eng\Invoke-YoloVisionDetectionReference.py"),
    "--weights", $ModelWeightsPath,
    "--coco-yaml", $officialDetectionCocoYamlPath,
    "--image", $officialDetectionSourceImagePath,
    "--onnx", $ModelPath,
    "--output-directory", $independentDirectory,
    "--csharp-tensor", $tensorPath,
    "--actual-output", $copiedOutputJson,
    "--confidence", "0.25",
    "--iou-threshold", "0.45",
    "--max-detections", "10",
    "--maximum-score-error", "0.01",
    "--minimum-box-iou", "0.995"
  )
  $independentRuntimeResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments $independentArguments -WorkingDirectory $RepositoryRoot
  $independentStdoutPath = Join-Path $ReportDirectory "independent-detection.stdout.log"
  $independentStderrPath = Join-Path $ReportDirectory "independent-detection.stderr.log"
  [IO.File]::WriteAllText($independentStdoutPath, $independentRuntimeResult.Stdout, $utf8)
  [IO.File]::WriteAllText($independentStderrPath, $independentRuntimeResult.Stderr, $utf8)
  if ($independentRuntimeResult.ExitCode -ne 0) {
    throw "Independent Ultralytics/PyTorch detection comparison failed with exit code $($independentRuntimeResult.ExitCode). See $independentStderrPath"
  }
  $independentRawReferencePath = Join-Path $independentDirectory "output0.reference.json"
  $independentReferencePath = Join-Path $independentDirectory "ultralytics-pytorch-reference.json"
  $independentComparisonPath = Join-Path $independentDirectory "yolovision-independent-comparison.json"
  Assert-FileSha256 -Path $independentRawReferencePath -ExpectedSha256 "6b7d8acb790fde1d41aa503e8d6a09ead9e4ea8e447f503b8fddab998c7387d6" -Description "Regenerated YOLOv8n detection raw reference"
  $independentReference = Get-Content -LiteralPath $independentReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
  $pinnedIndependentReference = Get-Content -LiteralPath $officialDetectionPinnedIndependentReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
  $independentComparison = Get-Content -LiteralPath $independentComparisonPath -Raw -Encoding utf8 | ConvertFrom-Json
  $regeneratedPredictionJson = @($independentReference.predictions) | ConvertTo-Json -Depth 12 -Compress
  $pinnedPredictionJson = @($pinnedIndependentReference.predictions) | ConvertTo-Json -Depth 12 -Compress
  if (-not [string]::Equals($regeneratedPredictionJson, $pinnedPredictionJson, [StringComparison]::Ordinal) -or
      [string]$independentReference.annotatedImage.sha256 -ne "909011789107f1573a75efa0ecd84dbc92ffd032cc2936f22f0319bede5c1dd3") {
    throw "Regenerated Ultralytics/PyTorch detection predictions or annotated image do not match the pinned independent reference."
  }
  if (-not $independentComparison.completed -or -not $independentComparison.passed -or
      [int]$independentComparison.referencePredictionCount -ne 5 -or
      [int]$independentComparison.actualPredictionCount -ne 5 -or
      @($independentComparison.comparisons | Where-Object { -not $_.passed }).Count -ne 0) {
    throw "Independent Ultralytics/PyTorch detection comparison did not pass all five predictions."
  }

  $controlledReferenceDirectory = Join-Path $runOutput "controlled-reference-negative"
  New-Item -ItemType Directory -Path $controlledReferenceDirectory -Force | Out-Null
  $controlledReferencePath = Join-Path $controlledReferenceDirectory "output0.single-value-mutated.reference.json"
  $mutationResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments @(
    (Join-Path $RepositoryRoot "eng\New-YoloVisionReferenceMutation.py"),
    "--input", $ReferenceOutput0Path,
    "--output", $controlledReferencePath,
    "--index", "0",
    "--delta", "125"
  ) -WorkingDirectory $RepositoryRoot
  $mutationLogPath = Join-Path $ReportDirectory "controlled-reference-mutation.log"
  [IO.File]::WriteAllText($mutationLogPath, ($mutationResult.Stdout + $mutationResult.Stderr), $utf8)
  if ($mutationResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $controlledReferencePath -PathType Leaf)) {
    throw "Failed to create the controlled detection raw-reference mutation. See $mutationLogPath"
  }
  $controlledReferenceMutationSha256 = (Get-FileHash -LiteralPath $controlledReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()

  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $runArguments -Name "--reference-outputs" -Value "output0:$controlledReferencePath"
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--output-json" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.json")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--visualization" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.svg")
  $controlledReferenceResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $controlledReferenceArguments -WorkingDirectory $workspace -Environment $runEnvironment -EnvironmentVariablesToRemove @("JYPPX_NATIVE_BRIDGE_PATH")
  $controlledReferenceStdoutPath = Join-Path $ReportDirectory "controlled-reference.stdout.log"
  $controlledReferenceStderrPath = Join-Path $ReportDirectory "controlled-reference.stderr.log"
  [IO.File]::WriteAllText($controlledReferenceStdoutPath, $controlledReferenceResult.Stdout, $utf8)
  [IO.File]::WriteAllText($controlledReferenceStderrPath, $controlledReferenceResult.Stderr, $utf8)
  $controlledReferenceOutputPath = Join-Path $controlledReferenceDirectory "yolovision-output.json"
  if ($controlledReferenceResult.ExitCode -ne 1 -or
      $controlledReferenceResult.Stdout.IndexOf("YoloVision Passed=False", [StringComparison]::Ordinal) -lt 0 -or
      -not (Test-Path -LiteralPath $controlledReferenceOutputPath -PathType Leaf)) {
    throw "Controlled detection raw-reference mutation must fail closed and emit its diagnostic report."
  }
  $controlledReferenceReport = Get-Content -LiteralPath $controlledReferenceOutputPath -Raw -Encoding utf8 | ConvertFrom-Json
  $controlledReferenceComparisons = @($controlledReferenceReport.referenceValidation.tensorComparisons)
  if ($controlledReferenceComparisons.Count -ne 1 -or [long]$controlledReferenceComparisons[0].mismatchCount -ne 1 -or
      [long]$controlledReferenceComparisons[0].firstMismatchIndex -ne 0 -or $controlledReferenceReport.referenceValidation.passed) {
    throw "Controlled detection raw-reference mutation did not produce one mismatch at index zero."
  }
  Copy-Item -LiteralPath $controlledReferenceOutputPath -Destination (Join-Path $ReportDirectory "controlled-reference-output.json") -Force
}
elseif ($isSemanticScenario) {
  $archivedSemanticArtifactDirectory = Join-Path $ReportDirectory "semantic-map-artifacts"
  if (Test-Path -LiteralPath $archivedSemanticArtifactDirectory) {
    Remove-Item -LiteralPath $archivedSemanticArtifactDirectory -Recurse -Force
  }
  Copy-Item -LiteralPath $semanticArtifactDirectory -Destination $archivedSemanticArtifactDirectory -Recurse -Force
  $archivedSemanticManifestPath = Join-Path $archivedSemanticArtifactDirectory "semantic-map-artifacts.manifest.json"
  $archivedSemanticManifest = Get-Content -LiteralPath $archivedSemanticManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $archivedSemanticClassIndexPath = Join-Path $archivedSemanticArtifactDirectory ([string]$archivedSemanticManifest.classIndexArtifact.fileName)
  $archivedSemanticManifest.classIndexArtifact.path = $archivedSemanticClassIndexPath
  [IO.File]::WriteAllText($archivedSemanticManifestPath, ($archivedSemanticManifest | ConvertTo-Json -Depth 12) + "`n", $utf8)

  $semanticArtifactManifestSha256 = (Get-FileHash -LiteralPath $archivedSemanticManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $semanticClassIndexSha256 = (Get-FileHash -LiteralPath $archivedSemanticClassIndexPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $semanticArtifactValidationPath = Join-Path $ReportDirectory "semantic-map-artifact-validation.json"
  $validatorShell = Get-Command pwsh -ErrorAction SilentlyContinue
  if ($null -eq $validatorShell) {
    $validatorShell = Get-Command powershell.exe -ErrorAction Stop
  }
  $semanticArtifactValidationResult = Invoke-CapturedProcess -FileName $validatorShell.Source -Arguments @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $RepositoryRoot "eng\Test-YoloVisionSemanticMapArtifact.ps1"),
    "-ManifestPath", $archivedSemanticManifestPath,
    "-ExpectedClassIndexPath", $ReferenceClassIndexPath,
    "-OutputPath", $semanticArtifactValidationPath
  ) -WorkingDirectory $RepositoryRoot
  if ($semanticArtifactValidationResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $semanticArtifactValidationPath -PathType Leaf)) {
    throw "LRASPP semantic class-index artifact validation failed: $($semanticArtifactValidationResult.Stdout)$($semanticArtifactValidationResult.Stderr)"
  }
  $semanticArtifactValidation = Get-Content -LiteralPath $semanticArtifactValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
  if (-not $semanticArtifactValidation.passed -or -not $semanticArtifactValidation.classIndexMatches -or
      -not $semanticArtifactValidation.histogramMatches) {
    throw "LRASPP semantic class-index artifact validation did not pass all checks."
  }

  $semanticHistogram = @($archivedSemanticManifest.classHistogram)
  $backgroundRow = @($semanticHistogram | Where-Object { [int]$_.classId -eq 0 })
  $dogRow = @($semanticHistogram | Where-Object { [int]$_.classId -eq 12 })
  $unexpectedRows = @($semanticHistogram | Where-Object { [int]$_.classId -notin @(0, 12) -and [long]$_.pixelCount -ne 0 })
  if ($semanticHistogram.Count -ne 21 -or $backgroundRow.Count -ne 1 -or $dogRow.Count -ne 1 -or
      [long]$backgroundRow[0].pixelCount -ne 65193 -or [long]$dogRow[0].pixelCount -ne 37207 -or
      $unexpectedRows.Count -ne 0) {
    throw "LRASPP semantic histogram must contain 65,193 background and 37,207 dog pixels only."
  }

  $controlledReferenceDirectory = Join-Path $runOutput "controlled-reference-negative"
  New-Item -ItemType Directory -Path $controlledReferenceDirectory -Force | Out-Null
  $controlledReferencePath = Join-Path $controlledReferenceDirectory "semantic.single-value-mutated.reference.json"
  $mutationResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments @(
    (Join-Path $RepositoryRoot "eng\New-YoloVisionReferenceMutation.py"),
    "--input", $ReferenceOutput0Path,
    "--output", $controlledReferencePath,
    "--index", "0",
    "--delta", "10"
  ) -WorkingDirectory $RepositoryRoot
  $mutationLogPath = Join-Path $ReportDirectory "controlled-reference-mutation.log"
  [IO.File]::WriteAllText($mutationLogPath, ($mutationResult.Stdout + $mutationResult.Stderr), $utf8)
  if ($mutationResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $controlledReferencePath -PathType Leaf)) {
    throw "Failed to create the controlled semantic raw-reference mutation. See $mutationLogPath"
  }
  $controlledReferenceMutationSha256 = (Get-FileHash -LiteralPath $controlledReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()

  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $runArguments -Name "--reference-outputs" -Value "semantic:$controlledReferencePath"
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--output-json" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.json")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--visualization" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.svg")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--semantic-artifact-output-directory" -Value (Join-Path $controlledReferenceDirectory "semantic-map-artifacts")
  $controlledReferenceResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $controlledReferenceArguments -WorkingDirectory $workspace -Environment $runEnvironment -EnvironmentVariablesToRemove @("JYPPX_NATIVE_BRIDGE_PATH")
  $controlledReferenceStdoutPath = Join-Path $ReportDirectory "controlled-reference.stdout.log"
  $controlledReferenceStderrPath = Join-Path $ReportDirectory "controlled-reference.stderr.log"
  [IO.File]::WriteAllText($controlledReferenceStdoutPath, $controlledReferenceResult.Stdout, $utf8)
  [IO.File]::WriteAllText($controlledReferenceStderrPath, $controlledReferenceResult.Stderr, $utf8)
  $controlledReferenceOutputPath = Join-Path $controlledReferenceDirectory "yolovision-output.json"
  if ($controlledReferenceResult.ExitCode -ne 1 -or
      $controlledReferenceResult.Stdout.IndexOf("YoloVision Passed=False", [StringComparison]::Ordinal) -lt 0 -or
      -not (Test-Path -LiteralPath $controlledReferenceOutputPath -PathType Leaf)) {
    throw "Controlled semantic raw-reference mutation must fail closed and emit its diagnostic report."
  }
  $controlledReferenceReport = Get-Content -LiteralPath $controlledReferenceOutputPath -Raw -Encoding utf8 | ConvertFrom-Json
  $controlledReferenceComparisons = @($controlledReferenceReport.referenceValidation.tensorComparisons)
  if ($controlledReferenceComparisons.Count -ne 1 -or [long]$controlledReferenceComparisons[0].mismatchCount -ne 1 -or
      [long]$controlledReferenceComparisons[0].firstMismatchIndex -ne 0 -or $controlledReferenceReport.referenceValidation.passed) {
    throw "Controlled semantic raw-reference mutation did not produce one mismatch at index zero."
  }
  Copy-Item -LiteralPath $controlledReferenceOutputPath -Destination (Join-Path $ReportDirectory "controlled-reference-output.json") -Force

  $controlledSemanticDirectory = Join-Path $ReportDirectory "controlled-semantic-artifact-tamper"
  if (Test-Path -LiteralPath $controlledSemanticDirectory) {
    Remove-Item -LiteralPath $controlledSemanticDirectory -Recurse -Force
  }
  Copy-Item -LiteralPath $archivedSemanticArtifactDirectory -Destination $controlledSemanticDirectory -Recurse -Force
  $controlledSemanticManifestPath = Join-Path $controlledSemanticDirectory "semantic-map-artifacts.manifest.json"
  $controlledSemanticManifest = Get-Content -LiteralPath $controlledSemanticManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $controlledSemanticClassIndexPath = Join-Path $controlledSemanticDirectory ([string]$controlledSemanticManifest.classIndexArtifact.fileName)
  $controlledSemanticManifest.classIndexArtifact.path = $controlledSemanticClassIndexPath
  [IO.File]::WriteAllText($controlledSemanticManifestPath, ($controlledSemanticManifest | ConvertTo-Json -Depth 12) + "`n", $utf8)
  $controlledSemanticOriginalSha256 = (Get-FileHash -LiteralPath $controlledSemanticClassIndexPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $controlledSemanticBytes = [IO.File]::ReadAllBytes($controlledSemanticClassIndexPath)
  $controlledSemanticBytes[0] = $controlledSemanticBytes[0] -bxor 1
  [IO.File]::WriteAllBytes($controlledSemanticClassIndexPath, $controlledSemanticBytes)
  $controlledSemanticMutatedSha256 = (Get-FileHash -LiteralPath $controlledSemanticClassIndexPath -Algorithm SHA256).Hash.ToLowerInvariant()
  $controlledSemanticValidationPath = Join-Path $controlledSemanticDirectory "semantic-map-artifact-validation.json"
  $controlledSemanticArtifactResult = Invoke-CapturedProcess -FileName $validatorShell.Source -Arguments @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $RepositoryRoot "eng\Test-YoloVisionSemanticMapArtifact.ps1"),
    "-ManifestPath", $controlledSemanticManifestPath,
    "-ExpectedClassIndexPath", $ReferenceClassIndexPath,
    "-OutputPath", $controlledSemanticValidationPath
  ) -WorkingDirectory $RepositoryRoot
  if ($controlledSemanticArtifactResult.ExitCode -ne 1 -or -not (Test-Path -LiteralPath $controlledSemanticValidationPath -PathType Leaf)) {
    throw "Controlled semantic class-index mutation must fail closed."
  }
  $controlledSemanticArtifactValidation = Get-Content -LiteralPath $controlledSemanticValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
  if ($controlledSemanticArtifactValidation.passed -or
      @($controlledSemanticArtifactValidation.findings | Where-Object { $_ -eq "artifact-sha256" }).Count -ne 1) {
    throw "Controlled semantic class-index mutation did not report the manifest SHA256 mismatch."
  }
}
elseif ($isClassificationScenario) {
  $controlledReferenceDirectory = Join-Path $runOutput "controlled-reference-negative"
  New-Item -ItemType Directory -Path $controlledReferenceDirectory -Force | Out-Null
  $controlledReferencePath = Join-Path $controlledReferenceDirectory "output0.single-value-mutated.reference.json"
  $mutationResult = Invoke-CapturedProcess -FileName $PythonPath -Arguments @(
    (Join-Path $RepositoryRoot "eng\New-YoloVisionReferenceMutation.py"),
    "--input", $ReferenceOutput0Path,
    "--output", $controlledReferencePath,
    "--index", "0",
    "--delta", "0.125"
  ) -WorkingDirectory $RepositoryRoot
  $mutationLogPath = Join-Path $ReportDirectory "controlled-reference-mutation.log"
  [IO.File]::WriteAllText($mutationLogPath, ($mutationResult.Stdout + $mutationResult.Stderr), $utf8)
  if ($mutationResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $controlledReferencePath -PathType Leaf)) {
    throw "Failed to create the controlled classification raw-reference mutation. See $mutationLogPath"
  }
  $controlledReferenceMutationSha256 = (Get-FileHash -LiteralPath $controlledReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()

  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $runArguments -Name "--reference-outputs" -Value "output0:$controlledReferencePath"
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--output-json" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.json")
  $controlledReferenceArguments = Set-NamedArgumentValue -Arguments $controlledReferenceArguments -Name "--visualization" -Value (Join-Path $controlledReferenceDirectory "yolovision-output.svg")
  $controlledReferenceResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $controlledReferenceArguments -WorkingDirectory $workspace -Environment $runEnvironment -EnvironmentVariablesToRemove @("JYPPX_NATIVE_BRIDGE_PATH")
  $controlledReferenceStdoutPath = Join-Path $ReportDirectory "controlled-reference.stdout.log"
  $controlledReferenceStderrPath = Join-Path $ReportDirectory "controlled-reference.stderr.log"
  [IO.File]::WriteAllText($controlledReferenceStdoutPath, $controlledReferenceResult.Stdout, $utf8)
  [IO.File]::WriteAllText($controlledReferenceStderrPath, $controlledReferenceResult.Stderr, $utf8)
  $controlledReferenceOutputPath = Join-Path $controlledReferenceDirectory "yolovision-output.json"
  if ($controlledReferenceResult.ExitCode -ne 1 -or
      $controlledReferenceResult.Stdout.IndexOf("YoloVision Passed=False", [StringComparison]::Ordinal) -lt 0 -or
      -not (Test-Path -LiteralPath $controlledReferenceOutputPath -PathType Leaf)) {
    throw "Controlled classification raw-reference mutation must fail closed and emit its diagnostic report."
  }
  $controlledReferenceReport = Get-Content -LiteralPath $controlledReferenceOutputPath -Raw -Encoding utf8 | ConvertFrom-Json
  $controlledReferenceComparisons = @($controlledReferenceReport.referenceValidation.tensorComparisons)
  if ($controlledReferenceComparisons.Count -ne 1 -or [long]$controlledReferenceComparisons[0].mismatchCount -ne 1 -or
      [long]$controlledReferenceComparisons[0].firstMismatchIndex -ne 0 -or $controlledReferenceReport.referenceValidation.passed) {
    throw "Controlled classification raw-reference mutation did not produce one mismatch at index zero."
  }
  Copy-Item -LiteralPath $controlledReferenceOutputPath -Destination (Join-Path $ReportDirectory "controlled-reference-output.json") -Force
}

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
  Remove-DirectoryTree -Path $resolvedCleanupTarget
  $workspaceRemoved = -not (Test-Path -LiteralPath $resolvedCleanupTarget)
}

$segmentationEvidence = $null
if ($isSegmentationScenario) {
  $rawTensorSummaries = @(
    foreach ($comparison in $referenceComparisons) {
      [pscustomobject][ordered]@{
        tensorName = [string]$comparison.tensorName
        actualShape = @($comparison.actualShape)
        referenceShape = @($comparison.referenceShape)
        comparedElementCount = [long]$comparison.comparedElementCount
        mismatchCount = [long]$comparison.mismatchCount
        firstMismatchIndex = [long]$comparison.firstMismatchIndex
        maximumAbsoluteError = [double]$comparison.maximumAbsoluteError
        maximumRelativeError = [double]$comparison.maximumRelativeError
        referenceSha256 = [string]$comparison.referenceSha256
        sourceClassification = [string]$comparison.sourceClassification
        passed = [bool]$comparison.passed
      }
    }
  )
  $independentComparisonSummaries = @(
    foreach ($comparison in @($independentComparison.comparisons)) {
      [pscustomobject][ordered]@{
        classId = [int]$comparison.classId
        className = [string]$comparison.className
        maximumBoxCoordinateAbsoluteError = [double]$comparison.maximumBoxCoordinateAbsoluteError
        scoreAbsoluteError = [double]$comparison.scoreAbsoluteError
        boxIoU = [double]$comparison.boxIoU
        maskIoU = [double]$comparison.maskIoU
        passed = [bool]$comparison.passed
      }
    }
  )
  $archivedMaskFiles = @(Get-ChildItem -LiteralPath $archivedMaskDirectory -File)
  $segmentationEvidence = [pscustomobject][ordered]@{
    modelContract = [pscustomobject][ordered]@{
      input = [pscustomobject][ordered]@{ name = "images"; shape = @(1, 3, 640, 640); dataType = "float32" }
      outputs = @(
        [pscustomobject][ordered]@{ name = "output0"; shape = @(1, 116, 8400); role = "detection-rows-with-32-mask-coefficients" },
        [pscustomobject][ordered]@{ name = "output1"; shape = @(1, 32, 160, 160); role = "mask-prototypes" }
      )
    }
    rawTensorReferenceValidation = [pscustomobject][ordered]@{
      sourceClassification = "independent-onnxruntime-cpu-execution-provider"
      absoluteTolerance = 0.02
      relativeTolerance = 0.03
      tensorCount = $rawTensorSummaries.Count
      comparedElementCount = [long](($rawTensorSummaries | Measure-Object comparedElementCount -Sum).Sum)
      mismatchCount = [long](($rawTensorSummaries | Measure-Object mismatchCount -Sum).Sum)
      tensors = $rawTensorSummaries
      completed = $true
      passed = $true
    }
    maskArtifacts = [pscustomobject][ordered]@{
      runtimeManifestSha256 = $runtimeMaskManifestSha256
      archivedManifestSha256 = $archivedMaskManifestSha256
      archivedFileCount = $archivedMaskFiles.Count
      archivedBytes = [long](($archivedMaskFiles | Measure-Object Length -Sum).Sum)
      predictionCount = [int]$archivedMaskManifest.predictionCount
      spatialTransformApplied = [bool]$archivedMaskManifest.spatialTransformApplied
      sourceThresholdedMaskCount = @($archivedMaskManifest.predictions | Where-Object { $null -ne $_.sourceThresholded }).Count
    }
    independentPostprocessValidation = [pscustomobject][ordered]@{
      evidenceClassification = [string]$independentComparison.evidenceClassification
      referenceFramework = [string]$independentReference.runtime.framework
      pythonVersion = [string]$independentReference.runtime.pythonVersion
      ultralyticsVersion = [string]$independentReference.runtime.ultralyticsVersion
      torchVersion = [string]$independentReference.runtime.torchVersion
      referenceSha256 = (Get-FileHash -LiteralPath $independentReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()
      comparisonSha256 = (Get-FileHash -LiteralPath $independentComparisonPath -Algorithm SHA256).Hash.ToLowerInvariant()
      thresholds = $independentComparison.thresholds
      predictionCount = [int]$independentComparison.actualPredictionCount
      comparisons = $independentComparisonSummaries
      completed = [bool]$independentComparison.completed
      passed = [bool]$independentComparison.passed
    }
    controlledRawReferenceValidation = [pscustomobject][ordered]@{
      kind = "single-reference-value-mutation"
      mutationIndex = 0
      mutationDelta = 10000.0
      mutatedReferenceSha256 = $controlledReferenceMutationSha256
      exitCode = $controlledReferenceResult.ExitCode
      tensorName = [string]$controlledOutput0[0].tensorName
      comparedElementCount = [long]$controlledOutput0[0].comparedElementCount
      mismatchCount = [long]$controlledOutput0[0].mismatchCount
      firstMismatchIndex = [long]$controlledOutput0[0].firstMismatchIndex
      validationPassed = [bool]$controlledReferenceReport.referenceValidation.passed
      failClosed = $true
    }
    controlledMaskIntegrityValidation = [pscustomobject][ordered]@{
      kind = "source-thresholded-mask-single-byte-mutation-with-unchanged-manifest-sha256"
      expectedSha256 = $controlledMaskExpectedSha256
      originalSha256 = $controlledMaskOriginalSha256
      mutatedSha256 = $controlledMaskMutatedSha256
      exitCode = $controlledMaskResult.ExitCode
      diagnostic = "Thresholded mask SHA256 does not match the manifest."
      failClosed = $true
    }
  }
}

$semanticEvidence = $null
if ($isSemanticScenario) {
  $rawTensorComparison = $referenceComparisons[0]
  $semanticEvidence = [pscustomobject][ordered]@{
    modelContract = [pscustomobject][ordered]@{
      input = [pscustomobject][ordered]@{
        name = "images"
        shape = @(1, 3, 320, 320)
        dataType = "float32"
        preprocess = "stretch-320x320; RGB; NCHW; scale=1/255; mean=0.485,0.456,0.406; std=0.229,0.224,0.225"
      }
      outputs = @(
        [pscustomobject][ordered]@{
          name = "semantic"
          shape = @(1, 21, 320, 320)
          role = "semantic-class-logits"
          layout = "NCHW"
          classCount = 21
        }
      )
      postprocess = [pscustomobject][ordered]@{
        operation = "argmax-over-class-dimension"
        tieRule = "lowest class index wins"
        classIndexShape = @(320, 320)
        classIndexElementCount = 102400
        classIndexDataType = "int32-little-endian"
      }
    }
    rawTensorReferenceValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$rawTensorComparison.sourceClassification
      absoluteTolerance = 0.0001
      relativeTolerance = 0.0001
      tensorCount = 1
      comparedElementCount = [long]$rawTensorComparison.comparedElementCount
      mismatchCount = [long]$rawTensorComparison.mismatchCount
      tensor = [pscustomobject][ordered]@{
        tensorName = [string]$rawTensorComparison.tensorName
        actualShape = @($rawTensorComparison.actualShape)
        referenceShape = @($rawTensorComparison.referenceShape)
        maximumAbsoluteError = [double]$rawTensorComparison.maximumAbsoluteError
        maximumRelativeError = [double]$rawTensorComparison.maximumRelativeError
        referenceSha256 = [string]$rawTensorComparison.referenceSha256
      }
      completed = $true
      passed = $true
    }
    classIndexArtifactValidation = [pscustomobject][ordered]@{
      manifestSha256 = $semanticArtifactManifestSha256
      classIndexSha256 = $semanticClassIndexSha256
      expectedClassIndexSha256 = [string]$semanticArtifactValidation.expectedClassIndexSha256
      pixelCount = 102400
      mismatchCount = 0
      firstMismatchIndex = -1
      classIndexMatches = [bool]$semanticArtifactValidation.classIndexMatches
      histogramMatches = [bool]$semanticArtifactValidation.histogramMatches
      histogram = @($archivedSemanticManifest.classHistogram)
      validationReportSha256 = (Get-FileHash -LiteralPath $semanticArtifactValidationPath -Algorithm SHA256).Hash.ToLowerInvariant()
      passed = [bool]$semanticArtifactValidation.passed
    }
    controlledRawReferenceValidation = [pscustomobject][ordered]@{
      kind = "single-reference-value-mutation"
      mutationIndex = 0
      mutationDelta = 10.0
      mutatedReferenceSha256 = $controlledReferenceMutationSha256
      exitCode = $controlledReferenceResult.ExitCode
      tensorName = [string]$controlledReferenceComparisons[0].tensorName
      comparedElementCount = [long]$controlledReferenceComparisons[0].comparedElementCount
      mismatchCount = [long]$controlledReferenceComparisons[0].mismatchCount
      firstMismatchIndex = [long]$controlledReferenceComparisons[0].firstMismatchIndex
      validationPassed = [bool]$controlledReferenceReport.referenceValidation.passed
      failClosed = $true
    }
    controlledClassIndexIntegrityValidation = [pscustomobject][ordered]@{
      kind = "semantic-class-index-single-byte-mutation-with-unchanged-manifest-sha256"
      originalSha256 = $controlledSemanticOriginalSha256
      mutatedSha256 = $controlledSemanticMutatedSha256
      exitCode = $controlledSemanticArtifactResult.ExitCode
      findings = @($controlledSemanticArtifactValidation.findings)
      failClosed = $true
    }
  }
}

$classificationEvidence = $null
if ($isClassificationScenario) {
  $rawTensorComparison = $referenceComparisons[0]
  $classificationEvidence = [pscustomobject][ordered]@{
    modelContract = [pscustomobject][ordered]@{
      input = [pscustomobject][ordered]@{
        name = "images"
        shape = @(1, 3, 224, 224)
        dataType = "float32"
        preprocess = "authoritative Ultralytics shorter-side-to-224 center-crop tensor; RGB; NCHW; scale=1/255"
      }
      outputs = @(
        [pscustomobject][ordered]@{
          name = "output0"
          shape = @(1, 1000)
          role = "probabilities"
          lastOnnxNode = "Softmax"
          classificationScoreMode = "probabilities"
        }
      )
      postprocess = [pscustomobject][ordered]@{
        classCount = 1000
        confidenceThreshold = 0.0
        topK = 5
        applyNms = $false
        nmsMode = "None"
      }
    }
    rawTensorReferenceValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$rawTensorComparison.sourceClassification
      absoluteTolerance = 0.001
      relativeTolerance = 0.001
      tensorCount = 1
      comparedElementCount = [long]$rawTensorComparison.comparedElementCount
      mismatchCount = [long]$rawTensorComparison.mismatchCount
      tensor = [pscustomobject][ordered]@{
        tensorName = [string]$rawTensorComparison.tensorName
        actualShape = @($rawTensorComparison.actualShape)
        referenceShape = @($rawTensorComparison.referenceShape)
        maximumAbsoluteError = [double]$rawTensorComparison.maximumAbsoluteError
        maximumRelativeError = [double]$rawTensorComparison.maximumRelativeError
        referenceSha256 = [string]$rawTensorComparison.referenceSha256
      }
      completed = $true
      passed = $true
    }
    top5Validation = [pscustomobject][ordered]@{
      sameIndicesAndOrderAsIndependentReference = $true
      predictionCount = $predictions.Count
      predictions = @($yoloOutputReport.predictions | ForEach-Object {
        [pscustomobject][ordered]@{
          classId = [int]$_.classId
          className = [string]$_.className
          score = [double]$_.score
        }
      })
      passed = $true
    }
    controlledRawReferenceValidation = [pscustomobject][ordered]@{
      kind = "single-reference-value-mutation"
      mutationIndex = 0
      mutationDelta = 0.125
      mutatedReferenceSha256 = $controlledReferenceMutationSha256
      exitCode = $controlledReferenceResult.ExitCode
      tensorName = [string]$controlledReferenceComparisons[0].tensorName
      comparedElementCount = [long]$controlledReferenceComparisons[0].comparedElementCount
      mismatchCount = [long]$controlledReferenceComparisons[0].mismatchCount
      firstMismatchIndex = [long]$controlledReferenceComparisons[0].firstMismatchIndex
      validationPassed = [bool]$controlledReferenceReport.referenceValidation.passed
      failClosed = $true
    }
  }
}

$poseEvidence = $null
if ($isPoseScenario) {
  $rawTensorComparison = $referenceComparisons[0]
  $poseComparisonSummaries = @(
    foreach ($comparison in @($independentComparison.comparisons)) {
      [pscustomobject][ordered]@{
        referenceIndex = [int]$comparison.referenceIndex
        actualIndex = [int]$comparison.actualIndex
        classId = [int]$comparison.classId
        className = [string]$comparison.className
        boxIoU = [double]$comparison.boxIoU
        scoreAbsoluteError = [double]$comparison.scoreAbsoluteError
        visibleKeypointCount = [int]$comparison.visibleKeypointCount
        maximumKeypointCoordinateError = [double]$comparison.maximumKeypointCoordinateError
        maximumKeypointScoreError = [double]$comparison.maximumKeypointScoreError
        passed = [bool]$comparison.passed
      }
    }
  )
  $poseEvidence = [pscustomobject][ordered]@{
    modelContract = [pscustomobject][ordered]@{
      input = [pscustomobject][ordered]@{
        name = "images"
        shape = @(1, 3, 640, 640)
        dataType = "float32"
        preprocess = "center-letterbox-640x640; RGB; NCHW; scale=1/255; fill=114"
      }
      outputs = @(
        [pscustomobject][ordered]@{
          name = "output0"
          shape = @(1, 56, 8400)
          role = "detection-rows-with-embedded-pose-keypoints"
          boxChannelCount = 4
          classCount = 1
          hasObjectness = $false
          auxiliaryChannelStart = 5
          keypointCount = 17
          keypointStride = 3
          layout = "channels-first"
        }
      )
      postprocess = [pscustomobject][ordered]@{
        confidenceThreshold = 0.25
        iouThreshold = 0.45
        topK = 10
        applyNms = $true
        nmsMode = "ClassAware"
      }
    }
    preprocessingValidation = [pscustomobject][ordered]@{
      tensorSha256 = $tensorSha256
      authoritativeTensorSha256 = "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d"
      elementCount = 1228800
      matchesAuthoritativeTensor = $true
      passed = $true
    }
    rawTensorReferenceValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$rawTensorComparison.sourceClassification
      absoluteTolerance = 1.25
      relativeTolerance = 0.05
      tensorCount = 1
      comparedElementCount = [long]$rawTensorComparison.comparedElementCount
      mismatchCount = [long]$rawTensorComparison.mismatchCount
      tensor = [pscustomobject][ordered]@{
        tensorName = [string]$rawTensorComparison.tensorName
        actualShape = @($rawTensorComparison.actualShape)
        referenceShape = @($rawTensorComparison.referenceShape)
        maximumAbsoluteError = [double]$rawTensorComparison.maximumAbsoluteError
        maximumRelativeError = [double]$rawTensorComparison.maximumRelativeError
        referenceSha256 = [string]$rawTensorComparison.referenceSha256
      }
      completed = $true
      passed = $true
    }
    independentPostprocessValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$independentReference.sourceClassification
      referenceFramework = [string]$independentReference.runtime.framework
      ultralyticsVersion = [string]$independentReference.runtime.ultralyticsVersion
      torchVersion = [string]$independentReference.runtime.torchVersion
      referenceSha256 = (Get-FileHash -LiteralPath $independentReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()
      comparisonSha256 = (Get-FileHash -LiteralPath $independentComparisonPath -Algorithm SHA256).Hash.ToLowerInvariant()
      thresholds = $independentComparison.thresholds
      predictionCount = [int]$independentComparison.actualPredictionCount
      minimumObservedBoxIoU = [double](($poseComparisonSummaries | Measure-Object boxIoU -Minimum).Minimum)
      maximumObservedScoreError = [double](($poseComparisonSummaries | Measure-Object scoreAbsoluteError -Maximum).Maximum)
      maximumObservedKeypointCoordinateError = [double](($poseComparisonSummaries | Measure-Object maximumKeypointCoordinateError -Maximum).Maximum)
      maximumObservedKeypointScoreError = [double](($poseComparisonSummaries | Measure-Object maximumKeypointScoreError -Maximum).Maximum)
      comparisons = $poseComparisonSummaries
      completed = [bool]$independentComparison.completed
      passed = [bool]$independentComparison.passed
    }
    controlledRawReferenceValidation = [pscustomobject][ordered]@{
      kind = "single-reference-value-mutation"
      mutationIndex = 0
      mutationDelta = 10000.0
      mutatedReferenceSha256 = $controlledReferenceMutationSha256
      exitCode = $controlledReferenceResult.ExitCode
      tensorName = [string]$controlledReferenceComparisons[0].tensorName
      comparedElementCount = [long]$controlledReferenceComparisons[0].comparedElementCount
      mismatchCount = [long]$controlledReferenceComparisons[0].mismatchCount
      firstMismatchIndex = [long]$controlledReferenceComparisons[0].firstMismatchIndex
      validationPassed = [bool]$controlledReferenceReport.referenceValidation.passed
      failClosed = $true
    }
  }
}

$obbEvidence = $null
if ($isObbScenario) {
  $rawTensorComparison = $referenceComparisons[0]
  $obbComparisonSummaries = @(
    foreach ($comparison in @($independentComparison.comparisons)) {
      [pscustomobject][ordered]@{
        referenceIndex = [int]$comparison.referenceIndex
        actualIndex = [int]$comparison.actualIndex
        classId = [int]$comparison.classId
        className = [string]$comparison.className
        rotatedIoU = [double]$comparison.rotatedIoU
        maximumCoordinateAbsoluteError = [double]$comparison.maximumCoordinateAbsoluteError
        anglePeriodicAbsoluteErrorRadians = [double]$comparison.anglePeriodicAbsoluteErrorRadians
        scoreAbsoluteError = [double]$comparison.scoreAbsoluteError
        passed = [bool]$comparison.passed
      }
    }
  )
  $obbEvidence = [pscustomobject][ordered]@{
    modelContract = [pscustomobject][ordered]@{
      input = [pscustomobject][ordered]@{
        name = "images"
        shape = @(1, 3, 1024, 1024)
        dataType = "float32"
        preprocess = "center-letterbox-1024x1024; RGB; NCHW; scale=1/255; fill=114"
      }
      outputs = @(
        [pscustomobject][ordered]@{
          name = "output0"
          shape = @(1, 20, 21504)
          role = "detection-rows-with-embedded-obb-angle"
          boxChannelCount = 4
          classCount = 15
          hasObjectness = $false
          auxiliaryChannelStart = 19
          angleChannelCount = 1
          angleUnit = "radian"
          angleRange = "[-pi/4,3pi/4]"
          layout = "channels-first"
        }
      )
      postprocess = [pscustomobject][ordered]@{
        confidenceThreshold = 0.25
        iouThreshold = 0.45
        topK = 40
        applyNms = $true
        nmsMode = "ClassAware"
        overlapMetric = "probabilistic-rotated-iou"
      }
    }
    preprocessingValidation = [pscustomobject][ordered]@{
      tensorSha256 = $tensorSha256
      authoritativeTensorSha256 = "c56c027619088bce94f9160a3f602b4ad81fe323001867b1ee456100040fec6e"
      elementCount = 3145728
      matchesAuthoritativeTensor = $true
      passed = $true
    }
    rawTensorReferenceValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$rawTensorComparison.sourceClassification
      absoluteTolerance = 4.25
      relativeTolerance = 0.05
      tensorCount = 1
      comparedElementCount = [long]$rawTensorComparison.comparedElementCount
      mismatchCount = [long]$rawTensorComparison.mismatchCount
      tensor = [pscustomobject][ordered]@{
        tensorName = [string]$rawTensorComparison.tensorName
        actualShape = @($rawTensorComparison.actualShape)
        referenceShape = @($rawTensorComparison.referenceShape)
        maximumAbsoluteError = [double]$rawTensorComparison.maximumAbsoluteError
        maximumRelativeError = [double]$rawTensorComparison.maximumRelativeError
        referenceSha256 = [string]$rawTensorComparison.referenceSha256
      }
      completed = $true
      passed = $true
    }
    independentPostprocessValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$independentReference.sourceClassification
      referenceFramework = [string]$independentReference.runtime.framework
      ultralyticsVersion = [string]$independentReference.runtime.ultralyticsVersion
      torchVersion = [string]$independentReference.runtime.torchVersion
      referenceSha256 = (Get-FileHash -LiteralPath $independentReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()
      comparisonSha256 = (Get-FileHash -LiteralPath $independentComparisonPath -Algorithm SHA256).Hash.ToLowerInvariant()
      thresholds = $independentComparison.thresholds
      predictionCount = [int]$independentComparison.actualPredictionCount
      minimumObservedRotatedIoU = [double](($obbComparisonSummaries | Measure-Object rotatedIoU -Minimum).Minimum)
      maximumObservedCoordinateError = [double](($obbComparisonSummaries | Measure-Object maximumCoordinateAbsoluteError -Maximum).Maximum)
      maximumObservedAngleErrorRadians = [double](($obbComparisonSummaries | Measure-Object anglePeriodicAbsoluteErrorRadians -Maximum).Maximum)
      maximumObservedScoreError = [double](($obbComparisonSummaries | Measure-Object scoreAbsoluteError -Maximum).Maximum)
      comparisons = $obbComparisonSummaries
      completed = [bool]$independentComparison.completed
      passed = [bool]$independentComparison.passed
    }
    controlledRawReferenceValidation = [pscustomobject][ordered]@{
      kind = "single-reference-value-mutation"
      mutationIndex = 0
      mutationDelta = 10000.0
      mutatedReferenceSha256 = $controlledReferenceMutationSha256
      exitCode = $controlledReferenceResult.ExitCode
      tensorName = [string]$controlledReferenceComparisons[0].tensorName
      comparedElementCount = [long]$controlledReferenceComparisons[0].comparedElementCount
      mismatchCount = [long]$controlledReferenceComparisons[0].mismatchCount
      firstMismatchIndex = [long]$controlledReferenceComparisons[0].firstMismatchIndex
      validationPassed = [bool]$controlledReferenceReport.referenceValidation.passed
      failClosed = $true
    }
  }
}

$officialDetectionEvidence = $null
if ($isOfficialDetectionScenario) {
  $rawTensorComparison = $referenceComparisons[0]
  $detectionComparisonSummaries = @(
    foreach ($comparison in @($independentComparison.comparisons)) {
      [pscustomobject][ordered]@{
        referenceIndex = [int]$comparison.referenceIndex
        actualIndex = [int]$comparison.actualIndex
        classId = [int]$comparison.classId
        className = [string]$comparison.className
        boxIoU = [double]$comparison.boxIoU
        scoreAbsoluteError = [double]$comparison.scoreAbsoluteError
        passed = [bool]$comparison.passed
      }
    }
  )
  $officialDetectionEvidence = [pscustomobject][ordered]@{
    modelContract = [pscustomobject][ordered]@{
      input = [pscustomobject][ordered]@{
        name = "images"
        shape = @(1, 3, 640, 640)
        dataType = "float32"
        preprocess = "center-letterbox-640x640; RGB; NCHW; scale=1/255; fill=114"
      }
      outputs = @(
        [pscustomobject][ordered]@{
          name = "output0"
          shape = @(1, 84, 8400)
          role = "detection-rows"
          boxChannelCount = 4
          classCount = 80
          hasObjectness = $false
          layout = "channels-first"
        }
      )
      postprocess = [pscustomobject][ordered]@{
        confidenceThreshold = 0.25
        iouThreshold = 0.45
        topK = 10
        applyNms = $true
        nmsMode = "ClassAware"
      }
    }
    preprocessingValidation = [pscustomobject][ordered]@{
      tensorSha256 = $tensorSha256
      authoritativeTensorSha256 = "46a0278967f1230ef8db59b0b8311a3aba3dce45233f1ba68821418ae02a574d"
      elementCount = 1228800
      matchesAuthoritativeTensor = $true
      passed = $true
    }
    rawTensorReferenceValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$rawTensorComparison.sourceClassification
      absoluteTolerance = 0.02
      relativeTolerance = 0.05
      tensorCount = 1
      comparedElementCount = [long]$rawTensorComparison.comparedElementCount
      mismatchCount = [long]$rawTensorComparison.mismatchCount
      tensor = [pscustomobject][ordered]@{
        tensorName = [string]$rawTensorComparison.tensorName
        actualShape = @($rawTensorComparison.actualShape)
        referenceShape = @($rawTensorComparison.referenceShape)
        maximumAbsoluteError = [double]$rawTensorComparison.maximumAbsoluteError
        maximumRelativeError = [double]$rawTensorComparison.maximumRelativeError
        referenceSha256 = [string]$rawTensorComparison.referenceSha256
      }
      completed = $true
      passed = $true
    }
    independentPostprocessValidation = [pscustomobject][ordered]@{
      sourceClassification = [string]$independentReference.sourceClassification
      referenceFramework = [string]$independentReference.runtime.framework
      ultralyticsVersion = [string]$independentReference.runtime.ultralyticsVersion
      torchVersion = [string]$independentReference.runtime.torchVersion
      referenceSha256 = (Get-FileHash -LiteralPath $independentReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()
      comparisonSha256 = (Get-FileHash -LiteralPath $independentComparisonPath -Algorithm SHA256).Hash.ToLowerInvariant()
      thresholds = $independentComparison.thresholds
      predictionCount = [int]$independentComparison.actualPredictionCount
      minimumObservedBoxIoU = [double](($detectionComparisonSummaries | Measure-Object boxIoU -Minimum).Minimum)
      maximumObservedScoreError = [double](($detectionComparisonSummaries | Measure-Object scoreAbsoluteError -Maximum).Maximum)
      comparisons = $detectionComparisonSummaries
      completed = [bool]$independentComparison.completed
      passed = [bool]$independentComparison.passed
    }
    controlledRawReferenceValidation = [pscustomobject][ordered]@{
      kind = "single-reference-value-mutation"
      mutationIndex = 0
      mutationDelta = 125.0
      mutatedReferenceSha256 = $controlledReferenceMutationSha256
      exitCode = $controlledReferenceResult.ExitCode
      tensorName = [string]$controlledReferenceComparisons[0].tensorName
      comparedElementCount = [long]$controlledReferenceComparisons[0].comparedElementCount
      mismatchCount = [long]$controlledReferenceComparisons[0].mismatchCount
      firstMismatchIndex = [long]$controlledReferenceComparisons[0].firstMismatchIndex
      validationPassed = [bool]$controlledReferenceReport.referenceValidation.passed
      failClosed = $true
    }
  }
}

$yoloV10DetectionEvidence = $null
if ($isYoloV10DetectionScenario) {
  $yoloV10DetectionEvidence = [pscustomobject][ordered]@{
    modelContract = [pscustomobject][ordered]@{
      input = [pscustomobject][ordered]@{
        name = "images"
        shape = @(1, 3, 640, 640)
        dataType = "float32"
        preprocess = "center-letterbox-640x640; RGB; NCHW; scale=1/255; fill=114"
      }
      outputs = @(
        [pscustomobject][ordered]@{
          name = "output0"
          shape = @(1, 300, 6)
          role = "end-to-end-detection-rows"
          columns = @("x1", "y1", "x2", "y2", "score", "classId")
          classCount = 80
          hasObjectness = $false
          layout = "end-to-end"
        }
      )
      postprocess = [pscustomobject][ordered]@{
        confidenceThreshold = 0.25
        topK = 100
        applyNms = $false
        nmsMode = "None"
      }
    }
    preprocessingValidation = [pscustomobject][ordered]@{
      tensorSha256 = $tensorSha256
      authoritativeTensorSha256 = "050935ebf471ec32ab4327d9f5643f0fe1a203289088895205e732e448a8d225"
      elementCount = 1228800
      matchesAuthoritativeTensor = $true
      passed = $true
    }
    outputContractValidation = [pscustomobject][ordered]@{
      tensorName = "output0"
      shape = @(1, 300, 6)
      predictionCount = 6
      busCount = 1
      personCount = 5
      topBusScore = [double]$topBus[0].score
      passed = $true
    }
    rawTensorReferenceValidation = [pscustomobject][ordered]@{
      requested = $false
      completed = $false
      passed = $false
      sourceClassification = "not-available-for-this-package-consumer-run"
      claimMade = $false
    }
  }
}

$reportRecordKind = if ($isSegmentationScenario) { "yolovision-yolov8n-seg-local-package-consumer-runtime" } elseif ($isSemanticScenario) { "yolovision-lraspp-semantic-local-package-consumer-runtime" } elseif ($isClassificationScenario) { "yolovision-yolov8n-cls-local-package-consumer-runtime" } elseif ($isPoseScenario) { "yolovision-yolov8n-pose-local-package-consumer-runtime" } elseif ($isObbScenario) { "yolovision-yolov8n-obb-local-package-consumer-runtime" } elseif ($isOfficialDetectionScenario) { "yolovision-yolov8n-det-local-package-consumer-runtime" } elseif ($isYoloV10DetectionScenario) { "yolovision-yolov10n-det-local-package-consumer-runtime" } else { "yolovision-yolox-local-package-consumer-runtime" }
$reportFileName = if ($isSegmentationScenario) { "yolov8n-seg-local-package-consumer-runtime.json" } elseif ($isSemanticScenario) { "lraspp-semantic-local-package-consumer-runtime.json" } elseif ($isClassificationScenario) { "yolov8n-cls-local-package-consumer-runtime.json" } elseif ($isPoseScenario) { "yolov8n-pose-local-package-consumer-runtime.json" } elseif ($isObbScenario) { "yolov8n-obb-local-package-consumer-runtime.json" } elseif ($isOfficialDetectionScenario) { "yolov8n-det-local-package-consumer-runtime.json" } elseif ($isYoloV10DetectionScenario) { "yolov10n-det-local-package-consumer-runtime.json" } else { "yolox-local-package-consumer-runtime.json" }
$reportTitle = if ($isSegmentationScenario) { "YOLOv8n-seg Local Package Consumer Runtime" } elseif ($isSemanticScenario) { "TorchVision LRASPP Semantic Local Package Consumer Runtime" } elseif ($isClassificationScenario) { "YOLOv8n-cls Local Package Consumer Runtime" } elseif ($isPoseScenario) { "YOLOv8n-pose Local Package Consumer Runtime" } elseif ($isObbScenario) { "YOLOv8n-obb Local Package Consumer Runtime" } elseif ($isOfficialDetectionScenario) { "YOLOv8n Detection Local Package Consumer Runtime" } elseif ($isYoloV10DetectionScenario) { "YOLOv10n Detection Local Package Consumer Runtime" } else { "YOLOX Local Package Consumer Runtime" }
$sourceCommit = (& git -C $RepositoryRoot rev-parse HEAD).Trim()

$report = [pscustomobject][ordered]@{
  schemaVersion = if ($isYoloV10DetectionScenario) { 9 } elseif ($isOfficialDetectionScenario) { 8 } elseif ($isObbScenario) { 7 } elseif ($isPoseScenario) { 6 } elseif ($isClassificationScenario) { 5 } elseif ($isSemanticScenario) { 4 } elseif ($isSegmentationScenario) { 3 } else { 2 }
  recordKind = $reportRecordKind
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  validationState = "passed-local-package-consumer-runtime"
  evidenceClassification = "local-package-consumer-runtime"
  scenario = $Scenario
  sourceCommit = $sourceCommit
  runtimePackageKey = $RuntimePackageKey
  tensorRtLine = $TensorRtLine
  consumer = [pscustomobject][ordered]@{
    template = "samples/YoloVision.PackageConsumer"
    targetFramework = "net8.0"
    projectReferenceCount = 0
    directAssemblyReferenceCount = 0
    restoredProjectLibraryCount = $projectLibraryCount
    restoredPackageHashesMatchSelected = (@($restoredPackageHashChecks | Where-Object { -not $_.matches }).Count -eq 0)
    restoredPackageHashChecks = $restoredPackageHashChecks
    packageSourceKind = "local-file-feed-only"
    packageSourceIsolation = "one-selected-nupkg-per-feed"
    packageSourceCount = 3
    packageCacheDrive = [IO.Path]::GetPathRoot($packageCache).TrimEnd('\')
    packageCacheFileCount = $packageCacheFileCount
    packageCacheBytes = $packageCacheBytes
    workspaceDrive = [IO.Path]::GetPathRoot($OutputRoot).TrimEnd('\')
    workspaceRemovedAfterValidation = $workspaceRemoved
    restoreExitCode = $restoreResult.ExitCode
    buildExitCode = $buildResult.ExitCode
    runtimeExitCode = $runResult.ExitCode
    nativeBridgePathEnvironmentVariableSet = $false
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
    packageContainsTensorRtCudaOrCudnn = ($vendorRuntimeEntries.Count -ne 0)
    vendorRuntimePackageEntryCount = $vendorRuntimeEntries.Count
    bridgeNativePackageEntryCount = $bridgeNativeEntries.Count
  }
  assets = [pscustomobject][ordered]@{
    modelSha256 = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    labelsSha256 = (Get-FileHash -LiteralPath $LabelsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    imageSha256 = (Get-FileHash -LiteralPath $ImagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    modelWeightsSha256 = if ($isSegmentationScenario -or $isSemanticScenario -or $isClassificationScenario -or $isPoseScenario -or $isObbScenario -or $isOfficialDetectionScenario) { (Get-FileHash -LiteralPath $ModelWeightsPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    referenceOutput0Sha256 = if ($isSegmentationScenario -or $isSemanticScenario -or $isClassificationScenario -or $isPoseScenario -or $isObbScenario -or $isOfficialDetectionScenario) { (Get-FileHash -LiteralPath $ReferenceOutput0Path -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    referenceOutput1Sha256 = if ($isSegmentationScenario) { (Get-FileHash -LiteralPath $ReferenceOutput1Path -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    referenceClassIndexSha256 = if ($isSemanticScenario) { (Get-FileHash -LiteralPath $ReferenceClassIndexPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    referenceInputTensorSha256 = if ($isClassificationScenario -or $isPoseScenario -or $isObbScenario -or $isOfficialDetectionScenario) { (Get-FileHash -LiteralPath $ReferenceInputTensorPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    sourceImageSha256 = if ($isPoseScenario) { (Get-FileHash -LiteralPath $poseSourceImagePath -Algorithm SHA256).Hash.ToLowerInvariant() } elseif ($isObbScenario) { (Get-FileHash -LiteralPath $obbSourceImagePath -Algorithm SHA256).Hash.ToLowerInvariant() } elseif ($isOfficialDetectionScenario) { (Get-FileHash -LiteralPath $officialDetectionSourceImagePath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
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
  segmentation = $segmentationEvidence
  semantic = $semanticEvidence
  classification = $classificationEvidence
  pose = $poseEvidence
  obb = $obbEvidence
  officialDetection = $officialDetectionEvidence
  yoloV10Detection = $yoloV10DetectionEvidence
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
    isSourceTreeRuntimeProof = $false
    isPackageConsumerRuntimeProof = $false
    isPublicPackageProof = $false
    packagesDownloadedFromPublicFeed = $false
    publicRedistributionOwnerApproval = $false
    canPromotePackageConsumerRuntime = $false
    canPublishPublicly = $false
    isPostPublishProof = $false
    ownerReleaseAcceptance = $false
    releaseProof = $false
    canCloseReleaseIssue = $false
    performsPublish = $false
    uploadsAssets = $false
  }
}

$reportPath = Join-Path $ReportDirectory $reportFileName
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($reportPath, ".md")
$markdown = @(
  "# $reportTitle",
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
  $(if ($isSegmentationScenario) { "- raw tensor values compared: ``$($segmentationEvidence.rawTensorReferenceValidation.comparedElementCount)``" } elseif ($isSemanticScenario) { "- raw tensor values compared: ``$($semanticEvidence.rawTensorReferenceValidation.comparedElementCount)``" } elseif ($isClassificationScenario) { "- raw tensor values compared: ``$($classificationEvidence.rawTensorReferenceValidation.comparedElementCount)``" } elseif ($isPoseScenario) { "- raw tensor values compared: ``$($poseEvidence.rawTensorReferenceValidation.comparedElementCount)``" } elseif ($isObbScenario) { "- raw tensor values compared: ``$($obbEvidence.rawTensorReferenceValidation.comparedElementCount)``" } elseif ($isOfficialDetectionScenario) { "- raw tensor values compared: ``$($officialDetectionEvidence.rawTensorReferenceValidation.comparedElementCount)``" } else { $null }),
  $(if ($isSegmentationScenario -or $isSemanticScenario -or $isClassificationScenario -or $isPoseScenario -or $isObbScenario -or $isOfficialDetectionScenario) { "- raw tensor mismatches: ``0``" } else { $null }),
  $(if ($isSegmentationScenario) { "- independent mask comparison passed: ``True``" } else { $null }),
  $(if ($isSemanticScenario) { "- semantic class-index pixels compared: ``102400``" } else { $null }),
  $(if ($isSemanticScenario) { "- semantic class-index SHA256: ``$semanticClassIndexSha256``" } else { $null }),
  $(if ($isClassificationScenario) { "- classification Top-5: ``$((@($predictions | ForEach-Object { $_.className })) -join ', ')``" } else { $null }),
  $(if ($isPoseScenario) { "- pose predictions / keypoints: ``4 / 17``" } else { $null }),
  $(if ($isPoseScenario) { "- independent minimum box IoU: ``$($poseEvidence.independentPostprocessValidation.minimumObservedBoxIoU)``" } else { $null }),
  $(if ($isObbScenario) { "- OBB predictions / class: ``40 / ship``" } else { $null }),
  $(if ($isObbScenario) { "- independent minimum rotated IoU: ``$($obbEvidence.independentPostprocessValidation.minimumObservedRotatedIoU)``" } else { $null }),
  $(if ($isOfficialDetectionScenario) { "- detection predictions / classes: ``5 / 4 person + 1 bus``" } else { $null }),
  $(if ($isOfficialDetectionScenario) { "- independent minimum box IoU: ``$($officialDetectionEvidence.independentPostprocessValidation.minimumObservedBoxIoU)``" } else { $null }),
  $(if ($isYoloV10DetectionScenario) { "- detection predictions / classes: ``6 / 5 person + 1 bus``" } else { $null }),
  $(if ($isYoloV10DetectionScenario) { "- output contract: ``output0:[1,300,6] / end-to-end / application NMS disabled``" } else { $null }),
  $(if ($isYoloV10DetectionScenario) { "- independent raw tensor reference: ``not available; no comparison claim made``" } else { $null }),
  $(if ($isSegmentationScenario -or $isSemanticScenario -or $isClassificationScenario -or $isPoseScenario -or $isObbScenario -or $isOfficialDetectionScenario) { "- raw-reference negative exit: ``$($controlledReferenceResult.ExitCode)``" } else { $null }),
  $(if ($isSegmentationScenario) { "- mask-integrity negative exit: ``$($controlledMaskResult.ExitCode)``" } else { $null }),
  $(if ($isSemanticScenario) { "- class-index-integrity negative exit: ``$($controlledSemanticArtifactResult.ExitCode)``" } else { $null }),
  "",
  $(if ($isSegmentationScenario) { "This record proves a clean local-feed PackageReference restore/build/run with the pinned YOLOv8n-seg assets, two raw tensor references, source-image mask artifacts, an independent PyTorch comparison, and two fail-closed negatives." } elseif ($isSemanticScenario) { "This record proves a clean local-feed PackageReference restore/build/run with the official torchvision LRASPP assets, one raw tensor reference, a full-resolution semantic class-index artifact, and two fail-closed negatives." } elseif ($isClassificationScenario) { "This record proves a clean local-feed PackageReference restore/build/run with the official YOLOv8n-cls assets, all 1,000 probabilities, the independent Top-5 order, and a fail-closed raw-reference negative." } elseif ($isPoseScenario) { "This record proves a clean local-feed PackageReference restore/build/run with the official YOLOv8n-pose assets, all 470,400 raw values, four 17-keypoint poses, an independent PyTorch comparison, and a fail-closed raw-reference negative." } elseif ($isObbScenario) { "This record proves a clean local-feed PackageReference restore/build/run with the official YOLOv8n-obb assets, all 430,080 raw values, 40 ship oriented boxes, an independent PyTorch rotated-IoU comparison, and a fail-closed raw-reference negative." } elseif ($isOfficialDetectionScenario) { "This record proves a clean local-feed PackageReference restore/build/run with the official YOLOv8n detection assets, all 705,600 raw values, four person detections and one bus, an independent PyTorch box-IoU comparison, and a fail-closed raw-reference negative." } elseif ($isYoloV10DetectionScenario) { "This record proves a clean local-feed PackageReference restore/build/run with the official YOLOv10n v1.1 ONNX, a pinned CC0 image, the fixed end-to-end output contract, and one bus plus five persons. No independent raw tensor reference was available, so no raw comparison claim is made." } else { "This record proves a clean local-feed PackageReference restore/build/run with the official YOLOX assets." }),
  "It does not prove public-feed download, redistribution approval, post-publish verification, Owner release acceptance, or release closure."
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "ValidationState=$($report.validationState) RuntimePackageKey=$RuntimePackageKey TensorRtLine=$TensorRtLine PredictionCount=$($predictions.Count) ProjectReferenceCount=0"
Write-Host "EvidenceClassification=$($report.evidenceClassification) IsPackageConsumerRuntimeProof=False CanPublishPublicly=False"
Write-Host "Report=$reportPath"
