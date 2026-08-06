[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$ProofInputPath,
  [Parameter(Mandatory = $true)][string]$ExpectedHandoffPath,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$PackageVersion = "4.0.0",
  [string]$PublicFeedUrl = "https://api.nuget.org/v3/index.json",
  [string]$ModelPath,
  [string]$LabelsPath,
  [string]$ImagePath,
  [string]$TensorRtRoot,
  [string]$TensorRtRuntimeRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
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

function Resolve-PathValue {
  param([string]$Value, [string]$DefaultValue, [string]$RelativeRoot)
  $candidate = if ([string]::IsNullOrWhiteSpace($Value)) { $DefaultValue } else { $Value }
  if ([IO.Path]::IsPathRooted($candidate)) { return [IO.Path]::GetFullPath($candidate) }
  return [IO.Path]::GetFullPath((Join-Path $RelativeRoot $candidate))
}
function Assert-NonCDrivePath {
  param([string]$Path, [string]$Description)
  if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)).TrimEnd('\') -ieq "C:") {
    throw "$Description must not use the C drive: $Path"
  }
}
function Assert-PathUnderRoot {
  param([string]$Path, [string]$Root, [string]$Description)
  $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd('\')
  $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
  if (-not $fullPath.StartsWith($fullRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "$Description must remain under '$fullRoot': $fullPath"
  }
}
function ConvertTo-XmlAttributeValue {
  param([string]$Value)
  return [Security.SecurityElement]::Escape($Value)
}
function Invoke-CapturedProcess {
  param([string]$FileName, [string[]]$Arguments, [string]$WorkingDirectory, [hashtable]$Environment = @{})
  $startInfo = [Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $FileName
  $startInfo.WorkingDirectory = $WorkingDirectory
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.CreateNoWindow = $true
  foreach ($argument in $Arguments) { $startInfo.ArgumentList.Add($argument) }
  foreach ($name in $Environment.Keys) { $startInfo.Environment[$name] = [string]$Environment[$name] }
  $process = [Diagnostics.Process]::new()
  $process.StartInfo = $startInfo
  if (-not $process.Start()) { throw "Failed to start process: $FileName" }
  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  $process.WaitForExit()
  $result = [pscustomobject]@{ ExitCode = $process.ExitCode; Stdout = $stdoutTask.GetAwaiter().GetResult(); Stderr = $stderrTask.GetAwaiter().GetResult() }
  $process.Dispose()
  return $result
}
function Invoke-CheckedDotNet {
  param([string[]]$Arguments, [string]$WorkingDirectory, [string]$LogPath)
  $result = Invoke-CapturedProcess -FileName "dotnet" -Arguments $Arguments -WorkingDirectory $WorkingDirectory
  [IO.File]::WriteAllText($LogPath, $result.Stdout + $result.Stderr, $utf8)
  if ($result.ExitCode -ne 0) { throw "dotnet $($Arguments -join ' ') failed with exit code $($result.ExitCode). See $LogPath" }
  return $result
}
function Remove-DirectoryWithRetry {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return $true }
  for ($attempt = 1; $attempt -le 20; $attempt++) {
    try {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
      if (-not (Test-Path -LiteralPath $Path)) { return $true }
    }
    catch {
      if ($attempt -eq 20) { Write-Warning "Failed to remove '$Path': $($_.Exception.Message)" }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
    Start-Sleep -Milliseconds 500
  }
  return -not (Test-Path -LiteralPath $Path)
}
function Get-RestoredPackage {
  param([string]$PackageCache, [string]$PackageId, [string]$Version, [string]$ExpectedSource)
  $packageRoot = Join-Path (Join-Path $PackageCache $PackageId.ToLowerInvariant()) $Version.ToLowerInvariant()
  $metadataPath = Join-Path $packageRoot ".nupkg.metadata"
  $nupkgPath = Join-Path $packageRoot "$($PackageId.ToLowerInvariant()).$($Version.ToLowerInvariant()).nupkg"
  foreach ($path in @($metadataPath, $nupkgPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Restored public package evidence is missing: $path" }
  }
  $metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding utf8 | ConvertFrom-Json
  if (-not [string]::Equals([string]$metadata.source, $ExpectedSource, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Package '$PackageId' restored from '$($metadata.source)' instead of '$ExpectedSource'."
  }
  return [pscustomobject]@{
    Id = $PackageId
    Version = $Version
    NupkgPath = $nupkgPath
    MetadataPath = $metadataPath
    Length = (Get-Item -LiteralPath $nupkgPath).Length
    Sha256 = (Get-FileHash -LiteralPath $nupkgPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Source = [string]$metadata.source
  }
}
function Get-MissingRuntimePatterns {
  param([string]$Root, [object[]]$Patterns)
  if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return @($Patterns) }
  return @(foreach ($pattern in $Patterns) { if (-not (Test-Path -Path (Join-Path $Root ([string]$pattern)) -PathType Leaf)) { [string]$pattern } })
}
function Get-PublicPackageUrl {
  param([string]$PackageId, [string]$Version)
  $flat = "https://api.nuget.org/v3-flatcontainer"
  return "$flat/$($PackageId.ToLowerInvariant())/$Version/$($PackageId.ToLowerInvariant()).$Version.nupkg"
}

if (-not [string]::Equals($PublicFeedUrl, "https://api.nuget.org/v3/index.json", [StringComparison]::OrdinalIgnoreCase)) {
  throw "Public YoloVision proof currently accepts only the official NuGet v3 source."
}
$splitManifest = Get-Content -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json") -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeEntries = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $RuntimePackageKey -and $_.role -eq "bridge" })
if ($bridgeEntries.Count -ne 1) { throw "Runtime package key '$RuntimePackageKey' must resolve to exactly one bridge package." }
$bridge = $bridgeEntries[0]
$runtimeManifest = Get-Content -LiteralPath (Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json") -Raw -Encoding utf8 | ConvertFrom-Json
$runtimePackages = @($runtimeManifest.packages | Where-Object { $_.key -eq $RuntimePackageKey })
if ($runtimePackages.Count -ne 1) { throw "Runtime package key '$RuntimePackageKey' must resolve to exactly one runtime package." }
$runtimePackage = $runtimePackages[0]

$rootsJson = (& pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $RuntimePackageKey -RepositoryRoot $RepositoryRoot | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "Failed to resolve runtime roots for '$RuntimePackageKey'." }
$roots = $rootsJson | ConvertFrom-Json
$TensorRtRoot = Resolve-PathValue -Value $TensorRtRoot -DefaultValue ([string]$roots.tensorRtRoot) -RelativeRoot $outerRoot
$CudaRoot = Resolve-PathValue -Value $CudaRoot -DefaultValue ([string]$roots.cudaRoot) -RelativeRoot $outerRoot
$CudnnRoot = Resolve-PathValue -Value $CudnnRoot -DefaultValue ([string]$roots.cudnnRoot) -RelativeRoot $outerRoot
if ([string]::IsNullOrWhiteSpace($TensorRtRuntimeRoot)) {
  $runtimeCandidates = [Collections.Generic.List[string]]::new()
  $runtimeCandidates.Add($TensorRtRoot)
  $downloadsRoot = Join-Path $outerRoot "downloads"
  if (Test-Path -LiteralPath $downloadsRoot -PathType Container) {
    foreach ($candidate in Get-ChildItem -LiteralPath $downloadsRoot -Directory -Recurse -Filter "assembled-runtime" -ErrorAction SilentlyContinue) {
      $runtimeCandidates.Add($candidate.FullName)
    }
  }
  foreach ($candidate in $runtimeCandidates) {
    if (@(Get-MissingRuntimePatterns -Root $candidate -Patterns @($runtimePackage.tensorRtFiles)).Count -eq 0) {
      $TensorRtRuntimeRoot = $candidate
      break
    }
  }
}
$TensorRtRuntimeRoot = Resolve-PathValue -Value $TensorRtRuntimeRoot -DefaultValue $TensorRtRoot -RelativeRoot $outerRoot
$missingRuntimePatterns = @(Get-MissingRuntimePatterns -Root $TensorRtRuntimeRoot -Patterns @($runtimePackage.tensorRtFiles))
if ($missingRuntimePatterns.Count -ne 0) { throw "TensorRT runtime root is incomplete: $($missingRuntimePatterns -join ', ')" }

$OutputRoot = Resolve-PathValue -Value $OutputRoot -DefaultValue (Join-Path $outerRoot "consumer-workspaces\yolovision-public-trt$($bridge.tensorRtLine)") -RelativeRoot $outerRoot
$ProofInputPath = Resolve-PathValue -Value $ProofInputPath -DefaultValue (Join-Path $outerRoot "public-proof\yolovision\$RuntimePackageKey\public-proof-record.json") -RelativeRoot $outerRoot
$ModelPath = Resolve-PathValue -Value $ModelPath -DefaultValue (Join-Path $outerRoot "downloads\yolox-apache\source\yolox_s.onnx") -RelativeRoot $outerRoot
$LabelsPath = Resolve-PathValue -Value $LabelsPath -DefaultValue (Join-Path $outerRoot "downloads\yolox-apache\derived\coco.names") -RelativeRoot $outerRoot
$ImagePath = Resolve-PathValue -Value $ImagePath -DefaultValue (Join-Path $outerRoot "downloads\yolox-apache\derived\dog.ppm") -RelativeRoot $outerRoot
Assert-PathUnderRoot -Path $OutputRoot -Root $outerRoot -Description "Public consumer workspace"
foreach ($item in @(
  @{ Path = $OutputRoot; Description = "Public consumer workspace" },
  @{ Path = $ProofInputPath; Description = "Public proof input" },
  @{ Path = $ModelPath; Description = "YOLOX model" },
  @{ Path = $LabelsPath; Description = "YOLOX labels" },
  @{ Path = $ImagePath; Description = "YOLOX image" }
)) { Assert-NonCDrivePath -Path $item.Path -Description $item.Description }
foreach ($path in @($ModelPath, $LabelsPath, $ImagePath, $TensorRtRoot, $TensorRtRuntimeRoot, $CudaRoot, $CudnnRoot)) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Required public consumer dependency does not exist: $path" }
}

if (Test-Path -LiteralPath $OutputRoot) {
  $existingOutput = (Resolve-Path -LiteralPath $OutputRoot).Path
  Assert-PathUnderRoot -Path $existingOutput -Root $outerRoot -Description "Existing public consumer workspace"
  if (-not (Remove-DirectoryWithRetry -Path $existingOutput)) { throw "Existing public consumer workspace could not be removed." }
}
$workspace = Join-Path $OutputRoot "workspace"
$packageCache = Join-Path $OutputRoot "packages"
$runOutput = Join-Path $OutputRoot "run-output"
$logRoot = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Path $workspace, $packageCache, $runOutput, $logRoot -Force | Out-Null
$templateRoot = Join-Path $RepositoryRoot "tests\fixtures\legacy-package-consumers\YoloVision.PackageConsumer"
$projectPath = Join-Path $workspace "YoloVision.PackageConsumer.csproj"
Copy-Item -LiteralPath (Join-Path $templateRoot "Program.cs") -Destination (Join-Path $workspace "Program.cs")
$project = Get-Content -LiteralPath (Join-Path $templateRoot "YoloVision.PackageConsumer.csproj.template") -Raw -Encoding utf8
$project = $project.Replace("__MANAGED_PACKAGE_VERSION__", $PackageVersion).Replace("__YOLOVISION_PACKAGE_VERSION__", $PackageVersion).Replace("__BRIDGE_PACKAGE_ID__", [string]$bridge.packageId).Replace("__BRIDGE_PACKAGE_VERSION__", $PackageVersion)
[IO.File]::WriteAllText($projectPath, $project, $utf8)
if ($project.Contains("ProjectReference", [StringComparison]::OrdinalIgnoreCase)) { throw "Public consumer project must not contain ProjectReference." }
$nugetConfigPath = Join-Path $workspace "NuGet.config"
$nugetConfig = "<?xml version=`"1.0`" encoding=`"utf-8`"?>`n<configuration>`n  <packageSources>`n    <clear />`n    <add key=`"nuget-org-public`" value=`"$(ConvertTo-XmlAttributeValue $PublicFeedUrl)`" />`n  </packageSources>`n</configuration>`n"
[IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)
$restoreResult = Invoke-CheckedDotNet -Arguments @("restore", $projectPath, "--configfile", $nugetConfigPath, "--packages", $packageCache, "--force", "--no-cache", "--verbosity", "minimal") -WorkingDirectory $workspace -LogPath (Join-Path $logRoot "restore.log")
$buildResult = Invoke-CheckedDotNet -Arguments @("build", $projectPath, "-c", "Release", "--no-restore", "--verbosity", "minimal") -WorkingDirectory $workspace -LogPath (Join-Path $logRoot "build.log")
$assets = Get-Content -LiteralPath (Join-Path $workspace "obj\project.assets.json") -Raw -Encoding utf8 | ConvertFrom-Json
$projectLibraryCount = @($assets.libraries.PSObject.Properties | Where-Object { $_.Value.type -eq "project" }).Count
if ($projectLibraryCount -ne 0) { throw "Public consumer restore graph contains project libraries." }
$restoredPackages = @(
  Get-RestoredPackage -PackageCache $packageCache -PackageId "JYPPX.TensorRT.CSharp.API" -Version $PackageVersion -ExpectedSource $PublicFeedUrl
  Get-RestoredPackage -PackageCache $packageCache -PackageId "JYPPX.TensorRT.CSharp.API.YoloVision" -Version $PackageVersion -ExpectedSource $PublicFeedUrl
  Get-RestoredPackage -PackageCache $packageCache -PackageId ([string]$bridge.packageId) -Version $PackageVersion -ExpectedSource $PublicFeedUrl
)

$consumerOutput = Join-Path $workspace "bin\Release\net8.0"
$bridgePaths = @(Get-ChildItem -LiteralPath $consumerOutput -Recurse -File -Filter "jyppxtrtbridge.dll")
if ($bridgePaths.Count -ne 1) { throw "Expected one bridge DLL in public consumer output, found $($bridgePaths.Count)." }
$tensorPath = Join-Path $runOutput "dog-yolox-s.fp32.bin"
$outputJsonPath = Join-Path $runOutput "yolovision-output.json"
$visualizationPath = Join-Path $runOutput "yolovision-output.svg"
$runArguments = @(
  (Join-Path $consumerOutput "YoloVision.PackageConsumer.dll"), "--model", $ModelPath, "--labels", $LabelsPath, "--image", $ImagePath,
  "--preprocessed-output", $tensorPath, "--output-json", $outputJsonPath, "--visualization", $visualizationPath,
  "--input-shape", "1x3x640x640", "--tensor-rt-line", [string]$bridge.tensorRtLine, "--family", "yolox", "--task", "det",
  "--layout", "boxes-first", "--has-objectness", "true", "--nms-mode", "class-aware", "--confidence", "0.3", "--iou-threshold", "0.45", "--top-k", "20"
)
$nativePaths = @(
  $consumerOutput, $bridgePaths[0].DirectoryName, (Join-Path $TensorRtRuntimeRoot "bin"), (Join-Path $TensorRtRuntimeRoot "lib"),
  (Join-Path $TensorRtRoot "bin"), (Join-Path $TensorRtRoot "lib"), (Join-Path $CudaRoot "bin"), $CudnnRoot, (Join-Path $CudnnRoot "bin")
) | Where-Object { Test-Path -LiteralPath $_ }
$cudnnDllDirectories = @(Get-ChildItem -LiteralPath $CudnnRoot -Recurse -File -Filter *.dll -ErrorAction SilentlyContinue | Select-Object -ExpandProperty DirectoryName -Unique)
$runResult = Invoke-CapturedProcess -FileName "dotnet" -Arguments $runArguments -WorkingDirectory $workspace -Environment @{ PATH = (@($nativePaths + $cudnnDllDirectories + $env:PATH) -join ';'); NUGET_PACKAGES = $packageCache }
[IO.File]::WriteAllText((Join-Path $logRoot "runtime.stdout.log"), $runResult.Stdout, $utf8)
[IO.File]::WriteAllText((Join-Path $logRoot "runtime.stderr.log"), $runResult.Stderr, $utf8)
Write-Host $runResult.Stdout
if ($runResult.ExitCode -ne 0 -or -not $runResult.Stdout.Contains("YoloVision Passed=True", [StringComparison]::Ordinal) -or -not $runResult.Stdout.Contains("YoloVisionPackageConsumer ProjectReference=False", [StringComparison]::Ordinal)) {
  throw "Public YoloVision package consumer did not complete the required runtime markers."
}
if ($runResult.Stdout -notmatch 'YoloVisionPackageConsumer BridgeTensorRt=(?<trt>\S+) BridgeCuda=(?<cuda>\S+)') { throw "Public consumer did not emit bridge build metadata." }
$bridgeTensorRtVersion = $Matches.trt
$bridgeCudaVersion = $Matches.cuda
if ($bridgeTensorRtVersion -notmatch '^(?<major>[0-9]+)' -or $Matches.major -ne [string]$bridge.tensorRtLine) { throw "Public consumer bridge line does not match the requested runtime key." }
$predictions = @($runResult.Stdout -split "`r?`n" | Where-Object { $_.StartsWith("Detection Class=", [StringComparison]::Ordinal) })
if ($predictions.Count -eq 0) { throw "Public consumer produced no detections." }
$elapsed = 0.0
if ($runResult.Stdout -match 'Execution .* ElapsedMs=(?<elapsed>[0-9.]+)') { $elapsed = [double]::Parse($Matches.elapsed, [Globalization.CultureInfo]::InvariantCulture) }

$proofDirectory = Split-Path -Parent $ProofInputPath
$proofPackageDirectory = Join-Path $proofDirectory "packages"
New-Item -ItemType Directory -Path $proofPackageDirectory -Force | Out-Null
$proofPackages = foreach ($package in $restoredPackages) {
  $destination = Join-Path $proofPackageDirectory (Split-Path -Leaf $package.NupkgPath)
  $metadataDestination = Join-Path $proofPackageDirectory "$($package.Id.ToLowerInvariant()).$($package.Version.ToLowerInvariant()).nupkg.metadata.json"
  Copy-Item -LiteralPath $package.NupkgPath -Destination $destination -Force
  Copy-Item -LiteralPath $package.MetadataPath -Destination $metadataDestination -Force
  [pscustomobject][ordered]@{
    id = $package.Id
    version = $package.Version
    sourceUrl = Get-PublicPackageUrl -PackageId $package.Id -Version $package.Version
    downloadedFromPublicFeed = $true
    downloadedNupkgPath = $destination
    nugetMetadataPath = $metadataDestination
    nugetMetadataSha256 = (Get-FileHash -LiteralPath $metadataDestination -Algorithm SHA256).Hash.ToLowerInvariant()
    length = (Get-Item -LiteralPath $destination).Length
    sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$workspaceRemoved = $false
if (-not $KeepWorkspace.IsPresent) { $workspaceRemoved = Remove-DirectoryWithRetry -Path $OutputRoot }
$proof = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-public-package-consumer-runtime-proof-input"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  evidenceClassification = "public-package-consumer-runtime"
  publicFeedUrl = $PublicFeedUrl
  packageSourceMode = "public-feed-only"
  runtimePackageKey = $RuntimePackageKey
  packageVersion = $PackageVersion
  packages = @($proofPackages)
  consumer = [pscustomobject][ordered]@{
    projectReferenceCount = 0
    restoredProjectLibraryCount = $projectLibraryCount
    packageSources = "public-feed-only"
    workspaceDrive = [IO.Path]::GetPathRoot($OutputRoot).TrimEnd('\')
    workspaceRemovedAfterValidation = $workspaceRemoved
    restoreExitCode = $restoreResult.ExitCode
    buildExitCode = $buildResult.ExitCode
  }
  runtime = [pscustomobject][ordered]@{
    passed = $true
    tensorRtLine = [string]$bridge.tensorRtLine
    bridgeBuildTensorRtVersion = $bridgeTensorRtVersion
    bridgeBuildCudaToolkitVersion = $bridgeCudaVersion
    predictionCount = $predictions.Count
    elapsedMilliseconds = $elapsed
    passedMarker = "YoloVision Passed=True"
    packageConsumerMarker = "YoloVisionPackageConsumer ProjectReference=False"
  }
  boundary = [pscustomobject][ordered]@{
    isPackageConsumerRuntimeProof = $true
    packagesDownloadedFromPublicFeed = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}
$proof | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ProofInputPath -Encoding utf8
Write-Host "EvidenceClassification=public-package-consumer-runtime RuntimePackageKey=$RuntimePackageKey PredictionCount=$($predictions.Count) PerformsPublish=False"
Write-Host "ProofInput=$ProofInputPath"
& pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "eng\Test-YoloVisionPublicPackageProof.ps1") -RepositoryRoot $RepositoryRoot -InputPath $ProofInputPath -ExpectedRuntimePackageKey $RuntimePackageKey -ExpectedPackageVersion $PackageVersion -ExpectedHandoffPath $ExpectedHandoffPath
if ($LASTEXITCODE -ne 0) { throw "Strict public package proof validation failed." }
