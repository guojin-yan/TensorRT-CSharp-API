[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$ManagedPackageDirectory,
  [string]$ClassificationPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$PackageVersion = "4.0.0",
  [string]$ModelPath,
  [string]$LabelsPath,
  [string]$ImagePath,
  [string]$TaskReferencePath,
  [string]$RawReferencePath,
  [string]$TensorRtRoot,
  [string]$TensorRtRuntimeRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
  [switch]$KeepWorkspace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$outerRoot = [IO.Path]::GetFullPath((Split-Path -Parent $RepositoryRoot))

function Resolve-PathValue {
  param([string]$Value, [string]$DefaultValue, [string]$RelativeRoot)
  $candidate = if ([string]::IsNullOrWhiteSpace($Value)) { $DefaultValue } else { $Value }
  if ([IO.Path]::IsPathRooted($candidate)) { return [IO.Path]::GetFullPath($candidate) }
  return [IO.Path]::GetFullPath((Join-Path $RelativeRoot $candidate))
}

function Get-NupkgMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)
  $archive = [IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = @($archive.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) })
    if ($nuspec.Count -ne 1) { throw "Package must contain exactly one nuspec: $Path" }
    $reader = [IO.StreamReader]::new($nuspec[0].Open(), [Text.Encoding]::UTF8)
    try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $entryNames = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
    return [pscustomobject][ordered]@{
      id = [string]$xml.package.metadata.id
      version = [string]$xml.package.metadata.version
      path = [IO.Path]::GetFullPath($Path)
      fileName = [IO.Path]::GetFileName($Path)
      length = (Get-Item -LiteralPath $Path).Length
      sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
      entryNames = $entryNames
    }
  }
  finally { $archive.Dispose() }
}

function Find-Package {
  param([string]$Directory, [string]$PackageId)
  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { throw "Package directory is missing: $Directory" }
  $matches = @(
    Get-ChildItem -LiteralPath $Directory -Filter *.nupkg -File | ForEach-Object { Get-NupkgMetadata $_.FullName } |
      Where-Object { $_.id -ceq $PackageId -and $_.version -ceq $PackageVersion }
  )
  if ($matches.Count -ne 1) { throw "Expected one $PackageId $PackageVersion package under $Directory; found $($matches.Count)." }
  return $matches[0]
}

function Assert-Sha256 {
  param([string]$Path, [string]$Expected, [string]$Description)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Description is missing: $Path" }
  $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actual -cne $Expected) { throw "$Description SHA256 mismatch. Expected=$Expected Actual=$actual Path=$Path" }
}

function Remove-DirectoryTree {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return }
  $fullPath = [IO.Path]::GetFullPath($Path)
  $outerPrefix = $outerRoot.TrimEnd('\') + '\'
  if (-not $fullPath.StartsWith($outerPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove a directory outside the outer workspace: $fullPath"
  }
  $extendedPath = if ($fullPath.StartsWith("\\", [StringComparison]::Ordinal)) { "\\?\UNC\" + $fullPath.Substring(2) } else { "\\?\" + $fullPath }
  $lastError = $null
  for ($attempt = 1; $attempt -le 6; $attempt++) {
    try { [IO.Directory]::Delete($extendedPath, $true); return } catch { $lastError = $_ }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
    Start-Sleep -Milliseconds (200 * $attempt)
    if (-not (Test-Path -LiteralPath $fullPath)) { return }
  }
  throw "Failed to remove directory '$fullPath': $($lastError.Exception.Message)"
}

function ConvertTo-NativeProcessArgument {
  param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)
  if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
  $builder = [Text.StringBuilder]::new()
  [void]$builder.Append('"')
  $slashes = 0
  foreach ($character in $Value.ToCharArray()) {
    if ($character -eq '\') { $slashes++; continue }
    if ($character -eq '"') {
      [void]$builder.Append(('\' * (($slashes * 2) + 1)))
      [void]$builder.Append('"')
    }
    else {
      if ($slashes -gt 0) { [void]$builder.Append(('\' * $slashes)) }
      [void]$builder.Append($character)
    }
    $slashes = 0
  }
  if ($slashes -gt 0) { [void]$builder.Append(('\' * ($slashes * 2))) }
  [void]$builder.Append('"')
  return $builder.ToString()
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
  $startInfo.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeProcessArgument $_ }) -join ' ')
  foreach ($name in $Environment.Keys) { $startInfo.EnvironmentVariables[$name] = [string]$Environment[$name] }
  [void]$startInfo.EnvironmentVariables.Remove("JYPPX_NATIVE_BRIDGE_PATH")
  $process = [Diagnostics.Process]::new()
  $process.StartInfo = $startInfo
  if (-not $process.Start()) { throw "Failed to start $FileName." }
  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  $process.WaitForExit()
  $result = [pscustomobject]@{
    exitCode = $process.ExitCode
    stdout = $stdoutTask.GetAwaiter().GetResult()
    stderr = $stderrTask.GetAwaiter().GetResult()
  }
  $process.Dispose()
  return $result
}

function Invoke-CheckedDotNet {
  param([string[]]$Arguments, [string]$WorkingDirectory, [string]$LogPath)
  $result = Invoke-CapturedProcess -FileName "dotnet" -Arguments $Arguments -WorkingDirectory $WorkingDirectory
  [IO.File]::WriteAllText($LogPath, $result.stdout + $result.stderr, $utf8)
  if ($result.exitCode -ne 0) { throw "dotnet $($Arguments -join ' ') failed. See $LogPath" }
  return $result
}

$manifest = Get-Content -LiteralPath (Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json") -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeRows = @($manifest.packages | Where-Object { $_.sourceRuntimeKey -eq $RuntimePackageKey -and $_.role -eq "bridge" })
if ($bridgeRows.Count -ne 1) { throw "RuntimePackageKey must resolve to exactly one bridge-only package: $RuntimePackageKey" }
$bridgeId = [string]$bridgeRows[0].packageId
$tensorRtLine = [string]$bridgeRows[0].tensorRtLine
if ([string]$bridgeRows[0].platform -ne "windows") { throw "This real-model consumer currently requires a Windows bridge package." }

$resolvedRoots = $null
if ([string]::IsNullOrWhiteSpace($TensorRtRoot) -or [string]::IsNullOrWhiteSpace($CudaRoot) -or [string]::IsNullOrWhiteSpace($CudnnRoot)) {
  $shell = Get-Command pwsh -ErrorAction SilentlyContinue
  if ($null -eq $shell) { $shell = Get-Command powershell.exe -ErrorAction Stop }
  $rootsJson = (& $shell.Source -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $RuntimePackageKey -RepositoryRoot $RepositoryRoot | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) { throw "Failed to resolve user-installed runtime roots for $RuntimePackageKey." }
  $resolvedRoots = $rootsJson | ConvertFrom-Json
}
$defaultTensorRtRoot = if ($null -ne $resolvedRoots -and -not [string]::IsNullOrWhiteSpace([string]$resolvedRoots.tensorRtRoot)) { [string]$resolvedRoots.tensorRtRoot } elseif (-not [string]::IsNullOrWhiteSpace($env:TENSORRT_PATH)) { $env:TENSORRT_PATH } else { "" }
$defaultCudaRoot = if ($null -eq $resolvedRoots) { "" } else { [string]$resolvedRoots.cudaRoot }
$defaultCudnnRoot = if ($null -eq $resolvedRoots) { "" } else { [string]$resolvedRoots.cudnnRoot }

$OutputRoot = Resolve-PathValue $OutputRoot (Join-Path $outerRoot "consumer-workspaces\classification-resnet18-local-package-trt$tensorRtLine") $outerRoot
$ReportDirectory = Resolve-PathValue $ReportDirectory (Join-Path $OutputRoot "report") $outerRoot
$ManagedPackageDirectory = Resolve-PathValue $ManagedPackageDirectory (Join-Path $RepositoryRoot "artifacts\classification-package-consumer\managed") $RepositoryRoot
$ClassificationPackageDirectory = Resolve-PathValue $ClassificationPackageDirectory (Join-Path $RepositoryRoot "artifacts\classification-package-consumer\classification") $RepositoryRoot
$BridgePackageDirectory = Resolve-PathValue $BridgePackageDirectory (Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$RuntimePackageKey") $RepositoryRoot
$ModelPath = Resolve-PathValue $ModelPath (Join-Path $outerRoot "models\Classification\resnet18-torchvision-v0.25.0\resnet18-imagenet1k-v1.onnx") $outerRoot
$LabelsPath = Resolve-PathValue $LabelsPath (Join-Path $outerRoot "models\Classification\resnet18-torchvision-v0.25.0\imagenet1k.names") $outerRoot
$assetRoot = Join-Path $outerRoot "downloads\article-assets\classification-resnet18-cc0-dog"
$ImagePath = Resolve-PathValue $ImagePath (Join-Path $assetRoot "dog-norre-vorupor-1280.bmp") $outerRoot
$TaskReferencePath = Resolve-PathValue $TaskReferencePath (Join-Path $assetRoot "reference\classification.onnxruntime.reference.json") $outerRoot
$RawReferencePath = Resolve-PathValue $RawReferencePath (Join-Path $assetRoot "reference\logits.onnxruntime.reference.json") $outerRoot
$TensorRtRoot = Resolve-PathValue $TensorRtRoot $defaultTensorRtRoot $outerRoot
$TensorRtRuntimeRoot = Resolve-PathValue $TensorRtRuntimeRoot $TensorRtRoot $outerRoot
$CudaRoot = Resolve-PathValue $CudaRoot $defaultCudaRoot $outerRoot
$cudnnCandidate = if ([string]::IsNullOrWhiteSpace($CudnnRoot)) { $defaultCudnnRoot } else { $CudnnRoot }
$CudnnRoot = if ([string]::IsNullOrWhiteSpace($cudnnCandidate)) { "" } else { Resolve-PathValue $cudnnCandidate $cudnnCandidate $outerRoot }

Assert-Sha256 $ModelPath "ead3558569edd88aa73a4eb46acbe6c38dee113933234547f04a0f6e48169903" "ResNet18 ONNX"
Assert-Sha256 $LabelsPath "ee4fa67b1dd46919ef87529b70e25bdd1ec9ebd350de3ad426e6640eb70320a1" "ImageNet labels"
Assert-Sha256 $ImagePath "0516933cf05213539e0a128ad5df5668acc6d32157ce50cc55c38e4125a21ead" "CC0 input BMP"
Assert-Sha256 $TaskReferencePath "ada4ff9f3685029f552aad5b0fe004f5c5e97641a7b2e45bcbc546d819205565" "Task reference"
Assert-Sha256 $RawReferencePath "f7bcdf6a835b7d6d25154ea35a452375daf2af59af12fad7b6e3f5b7d98b769c" "Raw logits reference"
foreach ($root in @($TensorRtRoot, $TensorRtRuntimeRoot, $CudaRoot)) {
  if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw "User-installed runtime root is missing: $root" }
}
if (-not [string]::IsNullOrWhiteSpace($CudnnRoot) -and -not (Test-Path -LiteralPath $CudnnRoot -PathType Container)) { throw "User-installed cuDNN root is missing: $CudnnRoot" }

$managed = Find-Package $ManagedPackageDirectory "JYPPX.TensorRT.CSharp.API"
$classification = Find-Package $ClassificationPackageDirectory "JYPPX.TensorRT.CSharp.API.Classification"
$bridge = Find-Package $BridgePackageDirectory $bridgeId
if ($managed.entryNames -notcontains "lib/net8.0/JYPPX.TensorRtSharp.dll") { throw "Managed package is missing the net8.0 TensorRT assembly." }
if ($classification.entryNames -notcontains "lib/net8.0/Classification.dll") { throw "Classification package is missing lib/net8.0/Classification.dll." }
if ($bridge.entryNames -notcontains "runtimes/win-x64/native/jyppxtrtbridge.dll") { throw "Bridge package is missing its only native bridge asset." }
$allEntries = @($managed.entryNames + $classification.entryNames + $bridge.entryNames)
$vendorEntries = @($allEntries | Where-Object { [IO.Path]::GetFileName($_) -match '^(cudart|cudnn|nvinfer|nvonnxparser|nvrtc|cublas|cufft|curand|cusolver|cusparse).*(\.dll|\.so|\.dylib)$' })
if ($vendorEntries.Count -ne 0) { throw "Candidate packages contain forbidden NVIDIA vendor runtimes: $($vendorEntries -join ', ')" }

if (Test-Path -LiteralPath $OutputRoot) { Remove-DirectoryTree $OutputRoot }
$workspace = Join-Path $OutputRoot "workspace"
$runOutput = Join-Path $OutputRoot "run-output"
$packageCache = Join-Path $OutputRoot "packages"
$feedRoot = Join-Path $OutputRoot "feeds"
$managedFeed = Join-Path $feedRoot "managed"
$classificationFeed = Join-Path $feedRoot "classification"
$bridgeFeed = Join-Path $feedRoot "bridge"
New-Item -ItemType Directory -Path $workspace, $runOutput, $packageCache, $ReportDirectory, $managedFeed, $classificationFeed, $bridgeFeed -Force | Out-Null
Copy-Item -LiteralPath $managed.path -Destination $managedFeed
Copy-Item -LiteralPath $classification.path -Destination $classificationFeed
Copy-Item -LiteralPath $bridge.path -Destination $bridgeFeed

$templateRoot = Join-Path $RepositoryRoot "samples\Classification.PackageConsumer"
$projectPath = Join-Path $workspace "Classification.PackageConsumer.csproj"
Copy-Item -LiteralPath (Join-Path $templateRoot "Program.cs") -Destination (Join-Path $workspace "Program.cs")
$project = Get-Content -LiteralPath (Join-Path $templateRoot "Classification.PackageConsumer.csproj.template") -Raw -Encoding utf8
$project = $project.Replace("__MANAGED_PACKAGE_VERSION__", $PackageVersion).Replace("__CLASSIFICATION_PACKAGE_VERSION__", $PackageVersion).Replace("__BRIDGE_PACKAGE_ID__", $bridge.id).Replace("__BRIDGE_PACKAGE_VERSION__", $PackageVersion)
if ($project.IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or $project.IndexOf("<Reference ", [StringComparison]::OrdinalIgnoreCase) -ge 0) { throw "Consumer template must contain PackageReference only." }
[IO.File]::WriteAllText($projectPath, $project, $utf8)
$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="managed" value="$([Security.SecurityElement]::Escape($managedFeed))" />
    <add key="classification" value="$([Security.SecurityElement]::Escape($classificationFeed))" />
    <add key="bridge" value="$([Security.SecurityElement]::Escape($bridgeFeed))" />
  </packageSources>
</configuration>
"@
$nugetConfigPath = Join-Path $workspace "NuGet.Config"
[IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)

$restore = Invoke-CheckedDotNet @("restore", $projectPath, "--configfile", $nugetConfigPath, "--packages", $packageCache, "--force", "--no-cache", "--verbosity", "minimal") $workspace (Join-Path $ReportDirectory "restore.log")
$build = Invoke-CheckedDotNet @("build", $projectPath, "-c", "Release", "--no-restore", "--verbosity", "minimal") $workspace (Join-Path $ReportDirectory "build.log")
$assets = Get-Content -LiteralPath (Join-Path $workspace "obj\project.assets.json") -Raw -Encoding utf8 | ConvertFrom-Json
$projectLibraryCount = @($assets.libraries.PSObject.Properties | Where-Object { $_.Value.type -eq "project" }).Count
if ($projectLibraryCount -ne 0) { throw "Consumer restore graph contains project libraries." }

$consumerOutput = Join-Path $workspace "bin\Release\net8.0"
$consumerAssembly = Join-Path $consumerOutput "Classification.PackageConsumer.dll"
$nativeBridges = @(Get-ChildItem -LiteralPath $consumerOutput -Recurse -Filter jyppxtrtbridge.dll -File)
if ($nativeBridges.Count -ne 1) { throw "Expected exactly one copied bridge in consumer output; found $($nativeBridges.Count)." }
$cudnnPathEntries = if ([string]::IsNullOrWhiteSpace($CudnnRoot)) { @() } else { @($CudnnRoot, (Join-Path $CudnnRoot "bin")) }
$pathEntries = @(
  $consumerOutput, $nativeBridges[0].DirectoryName,
  (Join-Path $TensorRtRuntimeRoot "bin"), (Join-Path $TensorRtRuntimeRoot "lib"),
  (Join-Path $TensorRtRoot "bin"), (Join-Path $TensorRtRoot "lib"),
  (Join-Path $CudaRoot "bin")
  $cudnnPathEntries
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_) }
$cudnnDirectories = if ([string]::IsNullOrWhiteSpace($CudnnRoot)) { @() } else { @(Get-ChildItem -LiteralPath $CudnnRoot -Recurse -Filter *.dll -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty DirectoryName -Unique) }
$runtimeEnvironment = @{ PATH = (@($pathEntries + $cudnnDirectories + $env:PATH) -join ';'); NUGET_PACKAGES = $packageCache }

$tensorPath = Join-Path $runOutput "dog-resnet18-imagenet.fp32.bin"
$outputJson = Join-Path $runOutput "classification-output.json"
$visualization = Join-Path $runOutput "classification-annotated.svg"
$baseArguments = @(
  $consumerAssembly,
  "--model", $ModelPath, "--labels", $LabelsPath, "--image", $ImagePath,
  "--preprocessed-output", $tensorPath, "--output-json", $outputJson, "--visualization", $visualization,
  "--input-shape", "1x3x224x224", "--input-name", "images", "--output-name", "logits",
  "--tensor-rt-line", $tensorRtLine, "--noTF32", "--score-transform", "softmax", "--top-k", "5",
  "--reference-output", $TaskReferencePath, "--reference-abs", "0.00001", "--reference-rel", "0.0001",
  "--reference-outputs", "logits:$RawReferencePath", "--reference-abs-tolerance", "0.0001", "--reference-rel-tolerance", "0.001"
)
$positive = Invoke-CapturedProcess "dotnet" $baseArguments $workspace $runtimeEnvironment
[IO.File]::WriteAllText((Join-Path $ReportDirectory "runtime.stdout.log"), $positive.stdout, $utf8)
[IO.File]::WriteAllText((Join-Path $ReportDirectory "runtime.stderr.log"), $positive.stderr, $utf8)
Write-Host $positive.stdout
if ($positive.exitCode -ne 0 -or $positive.stdout.IndexOf("ClassificationPackageConsumer ProjectReference=False", [StringComparison]::Ordinal) -lt 0 -or $positive.stdout.IndexOf("Classification Passed=True", [StringComparison]::Ordinal) -lt 0) {
  throw "Positive Classification package consumer failed with exit code $($positive.exitCode)."
}
Assert-Sha256 $tensorPath "43de394443f6fc3ccfd08cd9df61ee645ee5c51d1954c52267c221a438252f9e" "C# preprocessed tensor"
$output = Get-Content -LiteralPath $outputJson -Raw -Encoding utf8 | ConvertFrom-Json

$tamperedReference = Join-Path $runOutput "classification.tampered.reference.json"
$tampered = Get-Content -LiteralPath $TaskReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
$tampered.values[0] = [double]$tampered.values[0] + 0.125
[IO.File]::WriteAllText($tamperedReference, ($tampered | ConvertTo-Json -Depth 8), $utf8)
$negativeArguments = @($baseArguments)
$negativeOutputJson = Join-Path $runOutput "classification-negative-output.json"
$negativeVisualization = Join-Path $runOutput "classification-negative-annotated.svg"
for ($index = 0; $index -lt $negativeArguments.Count; $index++) {
  if ($negativeArguments[$index] -eq "--reference-output") { $negativeArguments[$index + 1] = $tamperedReference }
  elseif ($negativeArguments[$index] -eq "--output-json") { $negativeArguments[$index + 1] = $negativeOutputJson }
  elseif ($negativeArguments[$index] -eq "--visualization") { $negativeArguments[$index + 1] = $negativeVisualization }
}
$negative = Invoke-CapturedProcess "dotnet" $negativeArguments $workspace $runtimeEnvironment
[IO.File]::WriteAllText((Join-Path $ReportDirectory "negative.stdout.log"), $negative.stdout, $utf8)
[IO.File]::WriteAllText((Join-Path $ReportDirectory "negative.stderr.log"), $negative.stderr, $utf8)
if ($negative.exitCode -ne 1 -or $negative.stdout.IndexOf("Mismatches=1", [StringComparison]::Ordinal) -lt 0 -or $negative.stdout.IndexOf("FirstMismatch=0", [StringComparison]::Ordinal) -lt 0 -or $negative.stdout.IndexOf("Classification Passed=False", [StringComparison]::Ordinal) -lt 0) {
  throw "Controlled Classification reference mutation did not fail closed as expected. ExitCode=$($negative.exitCode)"
}

$top5 = @($output.output.topK)
$rawComparison = @($output.runtimeReferenceValidation.tensorComparisons)[0]
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "classification-resnet18-local-package-consumer-runtime"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  proofClassification = "local-package-consumer-runtime"
  runtimePackageKey = $RuntimePackageKey
  packageVersion = $PackageVersion
  packages = @(@($managed, $classification, $bridge) | ForEach-Object { [pscustomobject][ordered]@{ id = $_.id; version = $_.version; length = $_.length; sha256 = $_.sha256 } })
  consumer = [pscustomobject][ordered]@{
    template = "samples/Classification.PackageConsumer"
    packageReferenceCount = 3
    projectReferenceCount = 0
    restoredProjectLibraryCount = $projectLibraryCount
    remoteSourcesCleared = $true
    isolatedPackageCache = $true
    bridgeCopyCount = $nativeBridges.Count
  }
  assets = [pscustomobject][ordered]@{
    modelSha256 = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    labelsSha256 = (Get-FileHash -LiteralPath $LabelsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    imageSha256 = (Get-FileHash -LiteralPath $ImagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    inputTensorSha256 = (Get-FileHash -LiteralPath $tensorPath -Algorithm SHA256).Hash.ToLowerInvariant()
    taskReferenceSha256 = (Get-FileHash -LiteralPath $TaskReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()
    rawReferenceSha256 = (Get-FileHash -LiteralPath $RawReferencePath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  runtime = [pscustomobject][ordered]@{
    exitCode = $positive.exitCode
    tensorRtLine = $tensorRtLine
    tensorRtVersion = "10.11.0"
    elapsedMilliseconds = [double]$output.runtime.elapsedMilliseconds
    outputValidated = [bool]$output.outputValidated
    taskReference = $output.referenceValidation
    rawReference = $output.runtimeReferenceValidation
    top5 = $top5
    passed = $true
  }
  controlledNegative = [pscustomobject][ordered]@{
    exitCode = $negative.exitCode
    mismatchCount = 1
    firstMismatchIndex = 0
    failClosed = $true
  }
  outputs = [pscustomobject][ordered]@{
    outputJson = $outputJson
    visualization = $visualization
    stdout = (Join-Path $ReportDirectory "runtime.stdout.log")
    stderr = (Join-Path $ReportDirectory "runtime.stderr.log")
  }
  boundary = [pscustomobject][ordered]@{
    localPackageConsumerRuntimeEvidence = $true
    vendorRuntimeEntryCount = $vendorEntries.Count
    userInstalledTensorRtCudaCudnn = $true
    publicPackageProof = $false
    postPublishProof = $false
    publicWeightRedistributionApproved = $false
    canPublishPublicly = $false
    performsPublish = $false
    uploadsAssets = $false
  }
}
$reportPath = Join-Path $ReportDirectory "classification-resnet18-local-package-consumer-runtime.json"
[IO.File]::WriteAllText($reportPath, ($report | ConvertTo-Json -Depth 12), $utf8)
Write-Host "ClassificationLocalPackageConsumer Passed=True PackageReferenceCount=3 ProjectReferenceCount=0"
Write-Host "Top5=$((@($top5 | ForEach-Object { $_.label })) -join ', ') RawCompared=$($rawComparison.comparedElementCount) RawMismatches=$($rawComparison.mismatchCount)"
Write-Host "ControlledNegativeExitCode=$($negative.exitCode) MismatchCount=1 FirstMismatchIndex=0"
Write-Host "Report=$reportPath PerformsPublish=False"

if (-not $KeepWorkspace.IsPresent -and (Test-Path -LiteralPath $workspace)) {
  Remove-DirectoryTree $workspace
}
