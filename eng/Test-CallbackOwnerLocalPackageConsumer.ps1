[CmdletBinding()]
param(
  [ValidateSet("GpuAllocator", "OutputAllocator", "DebugListener", "ProgressMonitor", "Profiler", "Logger", "StreamReader")][string]$Scenario = "GpuAllocator",
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$PackageVersion = "4.0.0",
  [ValidateSet("8", "10", "11")][string]$TensorRtLine = "10",
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [switch]$KeepWorkspace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$scenarioConfig = switch ($Scenario) {
  "GpuAllocator" {
    [pscustomobject][ordered]@{
      slug = "gpu-allocator"
      sampleDirectory = "GpuAllocator.PackageConsumer"
      projectFileName = "GpuAllocatorPackageConsumer.csproj"
      projectTemplateFileName = "GpuAllocator.PackageConsumer.csproj.template"
      runtimeSwitch = "--gpu-allocator-runtime-smoke-only"
      runtimeMarkerPrefix = "GpuAllocatorRealRuntime=Passed "
      finalMarker = "Passed=True Mode=GpuAllocatorRuntimeSmokeOnly"
      publicSurfaceMarkers = @(
        "TensorRtGpuAllocatorCallbackOwner",
        "TensorRtGpuAllocatorCallbackRequest",
        "TensorRtGpuAllocatorRuntimeSnapshot",
        "SetGpuAllocator"
      )
    }
  }
  "OutputAllocator" {
    [pscustomobject][ordered]@{
      slug = "output-allocator"
      sampleDirectory = "OutputAllocator.PackageConsumer"
      projectFileName = "OutputAllocatorPackageConsumer.csproj"
      projectTemplateFileName = "OutputAllocator.PackageConsumer.csproj.template"
      runtimeSwitch = "--output-allocator-runtime-smoke-only"
      runtimeMarkerPrefix = "OutputAllocatorRealRuntime=Passed "
      finalMarker = "Passed=True Mode=OutputAllocatorRuntimeSmokeOnly"
      publicSurfaceMarkers = @(
        "TensorRtOutputAllocatorCallbackOwner",
        "TensorRtOutputAllocatorCallbackRequest",
        "TensorRtOutputAllocatorRuntimeSnapshot",
        "SetOutputAllocator",
        "ClearOutputAllocator"
      )
    }
  }
  "DebugListener" {
    [pscustomobject][ordered]@{
      slug = "debug-listener"
      sampleDirectory = "DebugListener.PackageConsumer"
      projectFileName = "DebugListenerPackageConsumer.csproj"
      projectTemplateFileName = "DebugListener.PackageConsumer.csproj.template"
      runtimeSwitch = "--debug-listener-runtime-smoke-only"
      runtimeMarkerPrefix = "DebugListenerRealRuntime=Passed "
      finalMarker = "Passed=True Mode=DebugListenerRuntimeSmokeOnly"
      publicSurfaceMarkers = @(
        "TensorRtDebugListenerCallbackOwner",
        "TensorRtDebugListenerCallbackRequest",
        "TensorRtDebugListenerRuntimeSnapshot",
        "TensorRtDebugTensorMetadataSnapshot",
        "SetDebugListener",
        "ClearDebugListener",
        "SetTensorDebugState"
      )
    }
  }
  "ProgressMonitor" {
    [pscustomobject][ordered]@{
      slug = "progress-monitor"
      sampleDirectory = "ProgressMonitor.PackageConsumer"
      projectFileName = "ProgressMonitorPackageConsumer.csproj"
      projectTemplateFileName = "ProgressMonitor.PackageConsumer.csproj.template"
      runtimeSwitch = "--progress-monitor-runtime-smoke-only"
      runtimeMarkerPrefix = "ProgressMonitorRealRuntime=Passed "
      finalMarker = "Passed=True Mode=ProgressMonitorRuntimeSmokeOnly"
      publicSurfaceMarkers = @(
        "TensorRtProgressMonitor",
        "TensorRtProgressMonitorEvent",
        "TensorRtProgressMonitorEventKind",
        "SetProgressMonitor",
        "ClearProgressMonitor",
        "HasProgressMonitor"
      )
    }
  }
  "Profiler" {
    [pscustomobject][ordered]@{
      slug = "profiler"
      sampleDirectory = "Profiler.PackageConsumer"
      projectFileName = "ProfilerPackageConsumer.csproj"
      projectTemplateFileName = "Profiler.PackageConsumer.csproj.template"
      runtimeSwitch = "--profiler-runtime-smoke-only"
      runtimeMarkerPrefix = "ProfilerRealRuntime=Passed "
      finalMarker = "Passed=True Mode=ProfilerRuntimeSmokeOnly"
      publicSurfaceMarkers = @(
        "TensorRtProfiler",
        "TensorRtProfilerHandler",
        "SetProfiler",
        "ClearProfiler",
        "ReportToProfiler",
        "EnqueueEmitsProfile",
        "HasNativeProfiler"
      )
    }
  }
  "Logger" {
    [pscustomobject][ordered]@{
      slug = "logger"
      sampleDirectory = "Logger.PackageConsumer"
      projectFileName = "LoggerPackageConsumer.csproj"
      projectTemplateFileName = "Logger.PackageConsumer.csproj.template"
      runtimeSwitch = "--logger-runtime-smoke-only"
      runtimeMarkerPrefix = "LoggerRealRuntime=Passed "
      finalMarker = "LoggerPackageConsumer Passed=True Mode=LoggerRuntimeSmokeOnly"
      publicSurfaceMarkers = @(
        "TensorRtLogger",
        "TensorRtLogHandler",
        "TensorRtLogSeverity",
        "CallbackInvocationCount",
        "CallbackFailureCount",
        "LastCallbackException",
        "IsAttached"
      )
    }
  }
  "StreamReader" {
    [pscustomobject][ordered]@{
      slug = "stream-reader"
      sampleDirectory = "StreamReader.PackageConsumer"
      projectFileName = "StreamReaderPackageConsumer.csproj"
      projectTemplateFileName = "StreamReader.PackageConsumer.csproj.template"
      runtimeSwitch = "--stream-reader-runtime-smoke-only"
      runtimeMarkerPrefix = "StreamReaderRealRuntime=Passed "
      finalMarker = "StreamReaderPackageConsumer Passed=True Mode=StreamReaderRuntimeSmokeOnly"
      publicSurfaceMarkers = @(
        "TensorRtStreamReader",
        "TensorRtStreamReaderRuntimeSnapshot",
        "TensorRtStreamSeekPosition",
        "Deserialize(JYPPX.TensorRtSharp.TensorRtStreamReader)"
      )
    }
  }
}

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
$outerRoot = [System.IO.Directory]::GetParent($RepositoryRoot).FullName

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $outerRoot "package-consumers\$($scenarioConfig.slug)-$RuntimePackageKey"
}
if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\callbacks\$($scenarioConfig.slug)-local-package-consumer\$RuntimePackageKey"
}
if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
if ([string]::IsNullOrWhiteSpace($BridgePackageDirectory)) {
  $BridgePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$RuntimePackageKey"
}
if ([string]::IsNullOrWhiteSpace($TensorRtRoot)) {
  $TensorRtRoot = $env:TENSORRT_PATH
}
if ([string]::IsNullOrWhiteSpace($CudaRoot)) {
  $CudaRoot = $env:CUDA_PATH
}

$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd('\', '/')
$ReportDirectory = [System.IO.Path]::GetFullPath($ReportDirectory).TrimEnd('\', '/')
$ManagedPackageDirectory = [System.IO.Path]::GetFullPath($ManagedPackageDirectory)
$BridgePackageDirectory = [System.IO.Path]::GetFullPath($BridgePackageDirectory)
$TensorRtRoot = [System.IO.Path]::GetFullPath($TensorRtRoot)
$CudaRoot = [System.IO.Path]::GetFullPath($CudaRoot)

function Test-PathWithin {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Parent
  )

  $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
  return $resolvedPath.StartsWith(
    $resolvedParent + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase)
}

function Remove-ConsumerDirectory {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-PathWithin -Path $Path -Parent $outerRoot) -or (Test-PathWithin -Path $Path -Parent $RepositoryRoot)) {
    throw "Refusing to remove package-consumer directory outside the outer workspace or inside the repository: $Path"
  }
  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  $lastError = $null
  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $extendedPath = if ($fullPath.StartsWith("\\", [System.StringComparison]::Ordinal)) {
    "\\?\UNC\" + $fullPath.Substring(2)
  }
  else {
    "\\?\" + $fullPath
  }

  for ($attempt = 1; $attempt -le 6; $attempt++) {
    try {
      Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction Stop
      return
    }
    catch {
      $lastError = $_
      if ($attempt -eq 1) {
        try { & dotnet build-server shutdown *> $null } catch { }
      }
      [System.GC]::Collect()
      [System.GC]::WaitForPendingFinalizers()
    }

    try {
      [System.IO.Directory]::Delete($extendedPath, $true)
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

  if (Test-Path -LiteralPath $fullPath) {
    $reason = if ($lastError) { $lastError.Exception.Message } else { "unknown error" }
    throw "Unable to remove package-consumer directory: $fullPath. Last error: $reason"
  }
}

function Write-Utf8Text {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Value
  )
  [System.IO.File]::WriteAllText($Path, $Value, $utf8)
}

function ConvertTo-XmlAttributeValue {
  param([Parameter(Mandatory = $true)][string]$Value)
  return [System.Security.SecurityElement]::Escape($Value)
}

function Get-StreamSha256 {
  param([Parameter(Mandatory = $true)][System.IO.Stream]$Stream)

  $algorithm = [System.Security.Cryptography.SHA256]::Create()
  try {
    return -join ($algorithm.ComputeHash($Stream) | ForEach-Object { $_.ToString("x2") })
  }
  finally {
    $algorithm.Dispose()
  }
}

function Get-NupkgInfo {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$PackageId,
    [Parameter(Mandatory = $true)][string]$ExpectedVersion
  )

  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($file in @(Get-ChildItem -LiteralPath $Directory -Filter "*.nupkg" -File)) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($file.FullName)
    try {
      $nuspec = $zip.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
      if ($null -eq $nuspec) { continue }
      $stream = $nuspec.Open()
      try {
        $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
        try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
      }
      finally {
        $stream.Dispose()
      }

      $manager = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
      $manager.AddNamespace("n", $xml.package.NamespaceURI)
      $id = $xml.SelectSingleNode("//n:metadata/n:id", $manager).InnerText
      $version = $xml.SelectSingleNode("//n:metadata/n:version", $manager).InnerText
      if ($id -eq $PackageId -and $version -eq $ExpectedVersion) {
        $entryHashes = [ordered]@{}
        foreach ($entry in @($zip.Entries | Where-Object { -not [string]::IsNullOrWhiteSpace($_.Name) })) {
          $entryStream = $entry.Open()
          try { $entryHashes[$entry.FullName] = Get-StreamSha256 -Stream $entryStream } finally { $entryStream.Dispose() }
        }
        $matches.Add([pscustomobject][ordered]@{
          path = $file.FullName
          id = $id
          version = $version
          length = $file.Length
          sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
          entries = @($zip.Entries | ForEach-Object { $_.FullName })
          entryHashes = $entryHashes
        })
      }
    }
    finally {
      $zip.Dispose()
    }
  }

  if ($matches.Count -ne 1) {
    throw "Expected one package '$PackageId' version '$ExpectedVersion' under '$Directory'; found $($matches.Count)."
  }
  return $matches[0]
}

function Invoke-CapturedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [hashtable]$Environment = @{},
    [string[]]$RemoveEnvironment = @()
  )

  $names = @(@($Environment.Keys) + @($RemoveEnvironment) | Select-Object -Unique)
  $previous = @{}
  foreach ($name in $names) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
  }

  $oldPreference = $ErrorActionPreference
  Push-Location $WorkingDirectory
  try {
    foreach ($name in $Environment.Keys) {
      [Environment]::SetEnvironmentVariable($name, [string]$Environment[$name], "Process")
    }
    foreach ($name in $RemoveEnvironment) {
      [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
    $ErrorActionPreference = "Continue"
    $lines = @(& $FilePath @ArgumentList 2>&1 | ForEach-Object { [string]$_ })
    $exitCode = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $oldPreference
    Pop-Location
    foreach ($name in $names) {
      [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
    }
  }

  return [pscustomobject][ordered]@{
    exitCode = $exitCode
    lines = $lines
    text = $lines -join [Environment]::NewLine
    command = "$FilePath $($ArgumentList -join ' ')"
  }
}

function Get-MarkerField {
  param(
    [Parameter(Mandatory = $true)][string]$Line,
    [Parameter(Mandatory = $true)][string]$Name
  )
  $match = [regex]::Match($Line, "(?:^|\s)" + [regex]::Escape($Name) + "=([^\s]+)")
  if (-not $match.Success) {
    throw "Marker field '$Name' was not found in: $Line"
  }
  return $match.Groups[1].Value
}

foreach ($required in @(
  @{ Path = $ManagedPackageDirectory; Kind = "managed package directory" },
  @{ Path = $BridgePackageDirectory; Kind = "bridge package directory" },
  @{ Path = $TensorRtRoot; Kind = "TensorRT root" },
  @{ Path = $CudaRoot; Kind = "CUDA root" }
)) {
  if (-not (Test-Path -LiteralPath $required.Path -PathType Container)) {
    throw "$($required.Kind) does not exist: $($required.Path)"
  }
}
if (-not (Test-PathWithin -Path $OutputRoot -Parent $outerRoot) -or (Test-PathWithin -Path $OutputRoot -Parent $RepositoryRoot)) {
  throw "OutputRoot must be outside the Git repository and below '$outerRoot': $OutputRoot"
}

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeRecord = @($splitManifest.packages | Where-Object {
    $_.sourceRuntimeKey -eq $RuntimePackageKey -and $_.role -eq "bridge"
  }) | Select-Object -First 1
if ($null -eq $bridgeRecord) {
  throw "Bridge-only package metadata for '$RuntimePackageKey' was not found."
}

$managedPackage = Get-NupkgInfo -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API" -ExpectedVersion $PackageVersion
$bridgePackage = Get-NupkgInfo -Directory $BridgePackageDirectory -PackageId $bridgeRecord.packageId -ExpectedVersion $PackageVersion
$bridgeNativeEntries = @($bridgePackage.entries | Where-Object { $_ -match '^runtimes/[^/]+/native/[^/]+$' })
if ($bridgeNativeEntries.Count -ne 1 -or $bridgeNativeEntries[0] -ne "runtimes/win-x64/native/jyppxtrtbridge.dll") {
  throw "Bridge package must contain exactly runtimes/win-x64/native/jyppxtrtbridge.dll."
}
$vendorBinaryEntries = @(
  @($managedPackage.entries + $bridgePackage.entries) |
    Where-Object { $_ -match '(?i)(^|/)(nvinfer|nvonnxparser|cudnn|cudart|nvrtc)[^/]*\.(dll|so)(\.|$)' }
)
if ($vendorBinaryEntries.Count -ne 0) {
  throw "Local package set unexpectedly bundles vendor runtime binaries: $($vendorBinaryEntries -join ', ')"
}

$managedSurfaceEntries = @($managedPackage.entries | Where-Object { $_ -match '^lib/net8\.0/JYPPX\..*\.xml$' })
Add-Type -AssemblyName System.IO.Compression.FileSystem
$managedZip = [System.IO.Compression.ZipFile]::OpenRead($managedPackage.path)
try {
  $surface = [System.Text.StringBuilder]::new()
  foreach ($entry in @($managedZip.Entries | Where-Object { $managedSurfaceEntries -contains $_.FullName })) {
    $entryStream = $entry.Open()
    try {
      $reader = [System.IO.StreamReader]::new($entryStream, [System.Text.Encoding]::UTF8)
      try { [void]$surface.AppendLine($reader.ReadToEnd()) } finally { $reader.Dispose() }
    }
    finally {
      $entryStream.Dispose()
    }
  }
  foreach ($marker in @($scenarioConfig.publicSurfaceMarkers)) {
    if ($surface.ToString().IndexOf($marker, [StringComparison]::Ordinal) -lt 0) {
      throw "Managed package XML surface is missing '$marker'."
    }
  }
}
finally {
  $managedZip.Dispose()
}

Remove-ConsumerDirectory -Path $OutputRoot
New-Item -ItemType Directory -Force -Path $OutputRoot, $ReportDirectory | Out-Null
$packagesPath = Join-Path $OutputRoot "packages"
$projectPath = Join-Path $OutputRoot $scenarioConfig.projectFileName
$programPath = Join-Path $OutputRoot "Program.cs"
$nugetConfigPath = Join-Path $OutputRoot "NuGet.config"
$stdoutPath = Join-Path $ReportDirectory "$($scenarioConfig.slug)-local-package-consumer.stdout.txt"
$reportPath = Join-Path $ReportDirectory "$($scenarioConfig.slug)-local-package-consumer-runtime.json"

$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="jyppx-managed" value="$(ConvertTo-XmlAttributeValue -Value $ManagedPackageDirectory)" />
    <add key="jyppx-bridge" value="$(ConvertTo-XmlAttributeValue -Value $BridgePackageDirectory)" />
  </packageSources>
</configuration>
"@
$sampleRoot = Join-Path $RepositoryRoot "samples\$($scenarioConfig.sampleDirectory)"
$project = Get-Content -LiteralPath (Join-Path $sampleRoot $scenarioConfig.projectTemplateFileName) -Raw
$project = $project.
  Replace("__MANAGED_PACKAGE_ID__", $managedPackage.id).
  Replace("__MANAGED_PACKAGE_VERSION__", $managedPackage.version).
  Replace("__BRIDGE_PACKAGE_ID__", $bridgePackage.id).
  Replace("__BRIDGE_PACKAGE_VERSION__", $bridgePackage.version)
Write-Utf8Text -Path $nugetConfigPath -Value $nugetConfig
Write-Utf8Text -Path $projectPath -Value $project
Copy-Item -LiteralPath (Join-Path $sampleRoot "Program.cs") -Destination $programPath -Force

$restore = Invoke-CapturedCommand -FilePath "dotnet" -ArgumentList @(
  "restore", $projectPath,
  "--configfile", $nugetConfigPath,
  "--packages", $packagesPath,
  "--force-evaluate"
) -WorkingDirectory $OutputRoot -RemoveEnvironment @("JYPPX_NATIVE_BRIDGE_PATH", "JYPPX_ENABLE_DEVELOPMENT_PROBING")
if ($restore.exitCode -ne 0) {
  throw "External consumer restore failed:`n$($restore.text)"
}

$build = Invoke-CapturedCommand -FilePath "dotnet" -ArgumentList @(
  "build", $projectPath, "-c", "Release", "--no-restore"
) -WorkingDirectory $OutputRoot -RemoveEnvironment @("JYPPX_NATIVE_BRIDGE_PATH", "JYPPX_ENABLE_DEVELOPMENT_PROBING")
if ($build.exitCode -ne 0) {
  throw "External consumer build failed:`n$($build.text)"
}

$tensorRtLib = Join-Path $TensorRtRoot "lib"
$cudaBin = Join-Path $CudaRoot "bin"
$runtimeEnvironment = @{
  "TENSORRT_PATH" = $TensorRtRoot
  "JYPPX_TENSORRT_ROOT" = $TensorRtRoot
  "CUDA_PATH" = $CudaRoot
  "PATH" = "$tensorRtLib;$cudaBin;$env:PATH"
}
$run = Invoke-CapturedCommand -FilePath "dotnet" -ArgumentList @(
  "run", "--project", $projectPath, "-c", "Release", "--no-build", "--",
  "--tensor-rt-line", $TensorRtLine,
  "--runtime-package-key", $RuntimePackageKey,
  $scenarioConfig.runtimeSwitch
) -WorkingDirectory $OutputRoot -Environment $runtimeEnvironment -RemoveEnvironment @(
  "JYPPX_NATIVE_BRIDGE_PATH",
  "JYPPX_ENABLE_DEVELOPMENT_PROBING"
)
Write-Utf8Text -Path $stdoutPath -Value ($run.text + [Environment]::NewLine)
if ($run.exitCode -ne 0) {
  throw "External $Scenario package consumer failed:`n$($run.text)"
}

$marker = $run.lines | Where-Object { $_.StartsWith($scenarioConfig.runtimeMarkerPrefix, [StringComparison]::Ordinal) } | Select-Object -First 1
if ($null -eq $marker) {
  throw "$($scenarioConfig.runtimeMarkerPrefix.Trim()) marker was not emitted."
}
$finalMarker = $run.lines | Where-Object { $_.IndexOf($scenarioConfig.finalMarker, [StringComparison]::Ordinal) -ge 0 } | Select-Object -First 1
if ($null -eq $finalMarker) {
  throw "Final package-consumer pass marker was not emitted."
}
$environmentMarker = $run.lines | Where-Object { $_.StartsWith("RuntimeEnvironment ", [StringComparison]::Ordinal) } | Select-Object -First 1
if ($null -eq $environmentMarker) {
  throw "RuntimeEnvironment marker was not emitted."
}
$runtimeTensorRtVersion = Get-MarkerField -Line $environmentMarker -Name "TRT"
$runtimeCudaToolkitVersion = Get-MarkerField -Line $environmentMarker -Name "CUDA"
if (-not $runtimeTensorRtVersion.StartsWith("$TensorRtLine.", [StringComparison]::Ordinal)) {
  throw "Runtime TensorRT version '$runtimeTensorRtVersion' does not match requested line '$TensorRtLine'."
}

$scenarioRuntime = $null
$scenarioNegatives = $null
$resultSummary = @()
switch ($Scenario) {
  "GpuAllocator" {
    $builderInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "BuilderInvocationCount")
    $builderAllocateCount = [uint64](Get-MarkerField -Line $marker -Name "BuilderAllocateCount")
    $builderReallocateCount = [uint64](Get-MarkerField -Line $marker -Name "BuilderReallocateCount")
    $builderDeallocateCount = [uint64](Get-MarkerField -Line $marker -Name "BuilderDeallocateCount")
    $builderLiveAllocationCount = [uint64](Get-MarkerField -Line $marker -Name "BuilderLiveAllocationCount")
    $rejectedBuildFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "RejectedBuildFailed"))
    $rejectedCount = [uint64](Get-MarkerField -Line $marker -Name "RejectedCount")
    $exceptionBuildFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "ExceptionBuildFailed"))
    $exceptionFailureCount = [uint64](Get-MarkerField -Line $marker -Name "ExceptionCallbackFailureCount")
    $pointerExposed = [bool]::Parse((Get-MarkerField -Line $marker -Name "PointerExposed"))
    $runtimeAttachLifecycle = [bool]::Parse((Get-MarkerField -Line $marker -Name "RuntimeAttachLifecycle"))
    $realCallbackRuntime = [bool]::Parse((Get-MarkerField -Line $marker -Name "RealCallbackRuntime"))
    if ($builderInvocationCount -eq 0 -or ($builderAllocateCount + $builderReallocateCount) -eq 0 -or
        $builderDeallocateCount -eq 0 -or $builderLiveAllocationCount -ne 0 -or
        -not $rejectedBuildFailed -or $rejectedCount -eq 0 -or
        -not $exceptionBuildFailed -or $exceptionFailureCount -eq 0 -or
        $pointerExposed -or -not $runtimeAttachLifecycle -or -not $realCallbackRuntime) {
      throw "External GPU allocator marker did not satisfy callback, fail-closed, pointer-free, and zero-leak invariants: $marker"
    }

    $scenarioRuntime = [pscustomobject][ordered]@{
      passed = $true
      runtimeAttachLifecycle = $runtimeAttachLifecycle
      builderInvocationCount = $builderInvocationCount
      builderAllocateCount = $builderAllocateCount
      builderReallocateCount = $builderReallocateCount
      builderDeallocateCount = $builderDeallocateCount
      builderPeakLiveAllocationBytes = [uint64](Get-MarkerField -Line $marker -Name "BuilderPeakLiveAllocationBytes")
      builderLiveAllocationCount = $builderLiveAllocationCount
      nativePointerExposed = $pointerExposed
      realCallbackRuntime = $realCallbackRuntime
    }
    $scenarioNegatives = [pscustomobject][ordered]@{
      rejection = [pscustomobject][ordered]@{ passed = $true; buildFailed = $rejectedBuildFailed; rejectedCount = $rejectedCount }
      exception = [pscustomobject][ordered]@{ passed = $true; buildFailed = $exceptionBuildFailed; callbackFailureCount = $exceptionFailureCount }
    }
    $resultSummary = @(
      "BuilderCallbacks=$builderInvocationCount LiveAllocations=$builderLiveAllocationCount",
      "RejectedCount=$rejectedCount ExceptionCallbackFailures=$exceptionFailureCount"
    )
  }
  "OutputAllocator" {
    $invocationCount = [uint64](Get-MarkerField -Line $marker -Name "InvocationCount")
    $notifyShapeCount = [uint64](Get-MarkerField -Line $marker -Name "NotifyShapeCount")
    $reallocateOutputCount = [uint64](Get-MarkerField -Line $marker -Name "ReallocateOutputCount")
    $allocationCount = [uint64](Get-MarkerField -Line $marker -Name "AllocationCount")
    $releaseCount = [uint64](Get-MarkerField -Line $marker -Name "ReleaseCount")
    $liveAllocationCount = [uint64](Get-MarkerField -Line $marker -Name "LiveAllocationCount")
    $peakLiveAllocationBytes = [uint64](Get-MarkerField -Line $marker -Name "PeakLiveAllocationBytes")
    $pointerExposed = [bool]::Parse((Get-MarkerField -Line $marker -Name "PointerExposed"))
    $negativeEnqueueFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeEnqueueFailed"))
    $negativeAllocationCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeAllocationCount")
    $negativeFailureCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeFailureCount")
    $realCallbackRuntime = [bool]::Parse((Get-MarkerField -Line $marker -Name "RealCallbackRuntime"))
    if ($invocationCount -eq 0 -or $notifyShapeCount -eq 0 -or $reallocateOutputCount -eq 0 -or
        $allocationCount -eq 0 -or $releaseCount -eq 0 -or $liveAllocationCount -ne 0 -or
        $peakLiveAllocationBytes -eq 0 -or $pointerExposed -or -not $negativeEnqueueFailed -or
        $negativeAllocationCount -ne 0 -or $negativeFailureCount -eq 0 -or -not $realCallbackRuntime) {
      throw "External output allocator marker did not satisfy callback, fail-closed, pointer-free, and zero-leak invariants: $marker"
    }

    $scenarioRuntime = [pscustomobject][ordered]@{
      passed = $true
      invocationCount = $invocationCount
      notifyShapeCount = $notifyShapeCount
      reallocateOutputCount = $reallocateOutputCount
      allocationCount = $allocationCount
      releaseCount = $releaseCount
      liveAllocationCount = $liveAllocationCount
      peakLiveAllocationBytes = $peakLiveAllocationBytes
      nativePointerExposed = $pointerExposed
      realCallbackRuntime = $realCallbackRuntime
    }
    $scenarioNegatives = [pscustomobject][ordered]@{
      rejection = [pscustomobject][ordered]@{
        passed = $true
        enqueueFailed = $negativeEnqueueFailed
        allocationCount = $negativeAllocationCount
        failureCount = $negativeFailureCount
      }
    }
    $resultSummary = @(
      "Callbacks=$invocationCount Allocations=$allocationCount Releases=$releaseCount LiveAllocations=$liveAllocationCount",
      "NegativeEnqueueFailed=$negativeEnqueueFailed NegativeFailures=$negativeFailureCount"
    )
  }
  "DebugListener" {
    $nativeVTableInstalled = [bool]::Parse((Get-MarkerField -Line $marker -Name "NativeVTableInstalled"))
    $processDebugTensorInvoked = [bool]::Parse((Get-MarkerField -Line $marker -Name "ProcessDebugTensorInvoked"))
    $invocationCount = [uint64](Get-MarkerField -Line $marker -Name "InvocationCount")
    $failureCount = [uint64](Get-MarkerField -Line $marker -Name "FailureCount")
    $inFlightCallbackCount = [uint64](Get-MarkerField -Line $marker -Name "InFlightCallbackCount")
    $tensorName = Get-MarkerField -Line $marker -Name "TensorName"
    $shape = Get-MarkerField -Line $marker -Name "Shape"
    $metadataCopied = [bool]::Parse((Get-MarkerField -Line $marker -Name "MetadataCopied"))
    $borrowedPointerExposed = [bool]::Parse((Get-MarkerField -Line $marker -Name "BorrowedPointerExposed"))
    $detachCount = [uint64](Get-MarkerField -Line $marker -Name "DetachCount")
    $negativeCallbackRejected = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeCallbackRejected"))
    $negativeEnqueueFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeEnqueueFailed"))
    $negativeInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeInvocationCount")
    $negativeFailureCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeFailureCount")
    $realCallbackRuntime = [bool]::Parse((Get-MarkerField -Line $marker -Name "IsRealCallbackRuntimeProof"))
    if (-not $nativeVTableInstalled -or -not $processDebugTensorInvoked -or
        $invocationCount -eq 0 -or $failureCount -ne 0 -or $inFlightCallbackCount -ne 0 -or
        $tensorName -ne "debug_output" -or $shape -ne "[1,4]" -or -not $metadataCopied -or
        $borrowedPointerExposed -or $detachCount -eq 0 -or -not $negativeCallbackRejected -or
        $negativeInvocationCount -eq 0 -or $negativeFailureCount -eq 0 -or -not $realCallbackRuntime) {
      throw "External debug listener marker did not satisfy callback, copied-metadata, pointer-free, detach, and rejection invariants: $marker"
    }

    $scenarioRuntime = [pscustomobject][ordered]@{
      passed = $true
      nativeVTableInstalled = $nativeVTableInstalled
      processDebugTensorInvoked = $processDebugTensorInvoked
      invocationCount = $invocationCount
      failureCount = $failureCount
      inFlightCallbackCount = $inFlightCallbackCount
      tensorName = $tensorName
      shape = @(1, 4)
      metadataCopied = $metadataCopied
      borrowedPointerExposed = $borrowedPointerExposed
      detachCount = $detachCount
      realCallbackRuntime = $realCallbackRuntime
    }
    $scenarioNegatives = [pscustomobject][ordered]@{
      rejection = [pscustomobject][ordered]@{
        passed = $true
        callbackRejected = $negativeCallbackRejected
        enqueueFailed = $negativeEnqueueFailed
        invocationCount = $negativeInvocationCount
        failureCount = $negativeFailureCount
      }
    }
    $resultSummary = @(
      "Callbacks=$invocationCount Failures=$failureCount InFlight=$inFlightCallbackCount DetachCount=$detachCount",
      "Tensor=$tensorName Shape=$shape MetadataCopied=$metadataCopied PointerExposed=$borrowedPointerExposed",
      "NegativeEnqueueFailed=$negativeEnqueueFailed NegativeFailures=$negativeFailureCount"
    )
  }
  "ProgressMonitor" {
    $attachedDuringBuild = [bool]::Parse((Get-MarkerField -Line $marker -Name "AttachedDuringBuild"))
    $invocationCount = [uint64](Get-MarkerField -Line $marker -Name "InvocationCount")
    $phaseStartCount = [uint64](Get-MarkerField -Line $marker -Name "PhaseStartCount")
    $stepCompleteCount = [uint64](Get-MarkerField -Line $marker -Name "StepCompleteCount")
    $phaseFinishCount = [uint64](Get-MarkerField -Line $marker -Name "PhaseFinishCount")
    $distinctPhaseCount = [uint64](Get-MarkerField -Line $marker -Name "DistinctPhaseCount")
    $failureCount = [uint64](Get-MarkerField -Line $marker -Name "FailureCount")
    $metadataCopied = [bool]::Parse((Get-MarkerField -Line $marker -Name "MetadataCopied"))
    $detachVerified = [bool]::Parse((Get-MarkerField -Line $marker -Name "DetachVerified"))
    $negativeCancellationRequested = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeCancellationRequested"))
    $negativeBuildFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeBuildFailed"))
    $negativeInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeInvocationCount")
    $negativeFailureCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeFailureCount")
    $negativeDetachVerified = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeDetachVerified"))
    $realCallbackRuntime = [bool]::Parse((Get-MarkerField -Line $marker -Name "RealCallbackRuntime"))
    if (-not $attachedDuringBuild -or $invocationCount -eq 0 -or $phaseStartCount -eq 0 -or
        $stepCompleteCount -eq 0 -or $phaseFinishCount -eq 0 -or $distinctPhaseCount -eq 0 -or
        $failureCount -ne 0 -or -not $metadataCopied -or -not $detachVerified -or
        -not $negativeCancellationRequested -or -not $negativeBuildFailed -or
        $negativeInvocationCount -eq 0 -or $negativeFailureCount -ne 0 -or
        -not $negativeDetachVerified -or -not $realCallbackRuntime) {
      throw "External progress monitor marker did not satisfy real callback, copied-metadata, cancellation, and detach invariants: $marker"
    }

    $scenarioRuntime = [pscustomobject][ordered]@{
      passed = $true
      attachedDuringBuild = $attachedDuringBuild
      invocationCount = $invocationCount
      phaseStartCount = $phaseStartCount
      stepCompleteCount = $stepCompleteCount
      phaseFinishCount = $phaseFinishCount
      distinctPhaseCount = $distinctPhaseCount
      failureCount = $failureCount
      metadataCopied = $metadataCopied
      detachVerified = $detachVerified
      realCallbackRuntime = $realCallbackRuntime
    }
    $scenarioNegatives = [pscustomobject][ordered]@{
      cancellation = [pscustomobject][ordered]@{
        passed = $true
        cancellationRequested = $negativeCancellationRequested
        buildFailed = $negativeBuildFailed
        invocationCount = $negativeInvocationCount
        failureCount = $negativeFailureCount
        detachVerified = $negativeDetachVerified
      }
    }
    $resultSummary = @(
      "Callbacks=$invocationCount Start=$phaseStartCount Step=$stepCompleteCount Finish=$phaseFinishCount Phases=$distinctPhaseCount",
      "CancellationRequested=$negativeCancellationRequested BuildFailed=$negativeBuildFailed NegativeFailures=$negativeFailureCount",
      "MetadataCopied=$metadataCopied DetachVerified=$detachVerified"
    )
  }
  "Profiler" {
    $immediateMode = [bool]::Parse((Get-MarkerField -Line $marker -Name "ImmediateMode"))
    $immediateInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "ImmediateInvocationCount")
    $immediateLayerCount = [uint64](Get-MarkerField -Line $marker -Name "ImmediateLayerCount")
    $immediateFailureCount = [uint64](Get-MarkerField -Line $marker -Name "ImmediateFailureCount")
    $immediateDetachVerified = [bool]::Parse((Get-MarkerField -Line $marker -Name "ImmediateDetachVerified"))
    $deferredMode = [bool]::Parse((Get-MarkerField -Line $marker -Name "DeferredMode"))
    $deferredBeforeReportCount = [uint64](Get-MarkerField -Line $marker -Name "DeferredBeforeReportCount")
    $deferredReported = [bool]::Parse((Get-MarkerField -Line $marker -Name "DeferredReported"))
    $deferredInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "DeferredInvocationCount")
    $deferredLayerCount = [uint64](Get-MarkerField -Line $marker -Name "DeferredLayerCount")
    $deferredFailureCount = [uint64](Get-MarkerField -Line $marker -Name "DeferredFailureCount")
    $deferredDetachVerified = [bool]::Parse((Get-MarkerField -Line $marker -Name "DeferredDetachVerified"))
    $metadataCopied = [bool]::Parse((Get-MarkerField -Line $marker -Name "MetadataCopied"))
    $negativeEnqueueFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeEnqueueFailed"))
    $negativeInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeInvocationCount")
    $negativeFailureCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeFailureCount")
    $negativeDetachVerified = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeDetachVerified"))
    $realCallbackRuntime = [bool]::Parse((Get-MarkerField -Line $marker -Name "RealCallbackRuntime"))
    if (-not $immediateMode -or $immediateInvocationCount -eq 0 -or $immediateLayerCount -eq 0 -or
        $immediateFailureCount -ne 0 -or -not $immediateDetachVerified -or $deferredMode -or
        $deferredBeforeReportCount -ne 0 -or -not $deferredReported -or
        $deferredInvocationCount -eq 0 -or $deferredLayerCount -eq 0 -or
        $deferredFailureCount -ne 0 -or -not $deferredDetachVerified -or -not $metadataCopied -or
        $negativeInvocationCount -eq 0 -or $negativeFailureCount -eq 0 -or
        -not $negativeDetachVerified -or -not $realCallbackRuntime) {
      throw "External profiler marker did not satisfy immediate, deferred-report, copied-metadata, exception, and detach invariants: $marker"
    }

    $scenarioRuntime = [pscustomobject][ordered]@{
      passed = $true
      immediate = [pscustomobject][ordered]@{
        enqueueEmitsProfile = $immediateMode
        invocationCount = $immediateInvocationCount
        layerCount = $immediateLayerCount
        failureCount = $immediateFailureCount
        detachVerified = $immediateDetachVerified
      }
      deferred = [pscustomobject][ordered]@{
        enqueueEmitsProfile = $deferredMode
        beforeReportCount = $deferredBeforeReportCount
        reportToProfilerReturned = $deferredReported
        invocationCount = $deferredInvocationCount
        layerCount = $deferredLayerCount
        failureCount = $deferredFailureCount
        detachVerified = $deferredDetachVerified
      }
      metadataCopied = $metadataCopied
      realCallbackRuntime = $realCallbackRuntime
    }
    $scenarioNegatives = [pscustomobject][ordered]@{
      handlerException = [pscustomobject][ordered]@{
        passed = $true
        enqueueFailed = $negativeEnqueueFailed
        invocationCount = $negativeInvocationCount
        failureCount = $negativeFailureCount
        detachVerified = $negativeDetachVerified
      }
    }
    $resultSummary = @(
      "ImmediateCallbacks=$immediateInvocationCount Layers=$immediateLayerCount Failures=$immediateFailureCount",
      "DeferredBeforeReport=$deferredBeforeReportCount Reported=$deferredReported Callbacks=$deferredInvocationCount",
      "NegativeEnqueueFailed=$negativeEnqueueFailed NegativeFailures=$negativeFailureCount MetadataCopied=$metadataCopied"
    )
  }
  "Logger" {
    $beforeOwnerCount = [uint64](Get-MarkerField -Line $marker -Name "BeforeOwnerCount")
    $afterBuildCount = [uint64](Get-MarkerField -Line $marker -Name "AfterBuildCount")
    $positiveInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "PositiveInvocationCount")
    $positiveSeverityCount = [uint64](Get-MarkerField -Line $marker -Name "PositiveSeverityCount")
    $positiveFailureCount = [uint64](Get-MarkerField -Line $marker -Name "PositiveFailureCount")
    $metadataCopied = [bool]::Parse((Get-MarkerField -Line $marker -Name "MetadataCopied"))
    $builderAttached = [bool]::Parse((Get-MarkerField -Line $marker -Name "BuilderAttached"))
    $builderDetached = [bool]::Parse((Get-MarkerField -Line $marker -Name "BuilderDetached"))
    $runtimeAttached = [bool]::Parse((Get-MarkerField -Line $marker -Name "RuntimeAttached"))
    $runtimeDetached = [bool]::Parse((Get-MarkerField -Line $marker -Name "RuntimeDetached"))
    $lifecycleAttachedBeforeDispose = [bool]::Parse((Get-MarkerField -Line $marker -Name "LifecycleAttachedBeforeDispose"))
    $lifecycleAttachedAfterDispose = [bool]::Parse((Get-MarkerField -Line $marker -Name "LifecycleAttachedAfterDispose"))
    $lifecyclePostDisposeCallbacks = [bool]::Parse((Get-MarkerField -Line $marker -Name "LifecyclePostDisposeCallbacks"))
    $lifecycleDetached = [bool]::Parse((Get-MarkerField -Line $marker -Name "LifecycleDetached"))
    $disposedRejectsNewBorrower = [bool]::Parse((Get-MarkerField -Line $marker -Name "DisposedRejectsNewBorrower"))
    $negativeOperationFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeOperationFailed"))
    $negativeInvocationCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeInvocationCount")
    $negativeFailureCount = [uint64](Get-MarkerField -Line $marker -Name "NegativeFailureCount")
    $negativeDetachVerified = [bool]::Parse((Get-MarkerField -Line $marker -Name "NegativeDetachVerified"))
    $threadSafeHandlerState = [bool]::Parse((Get-MarkerField -Line $marker -Name "ThreadSafeHandlerState"))
    $nativeFailureFlagAtomic = [bool]::Parse((Get-MarkerField -Line $marker -Name "NativeFailureFlagAtomic"))
    $realCallbackRuntime = [bool]::Parse((Get-MarkerField -Line $marker -Name "RealCallbackRuntime"))
    $syntheticDiagnosticUsed = [bool]::Parse((Get-MarkerField -Line $marker -Name "SyntheticDiagnosticUsed"))
    if ($beforeOwnerCount -ne 0 -or $afterBuildCount -eq 0 -or
        $positiveInvocationCount -lt $afterBuildCount -or $positiveSeverityCount -eq 0 -or
        $positiveFailureCount -ne 0 -or -not $metadataCopied -or
        -not $builderAttached -or -not $builderDetached -or -not $runtimeAttached -or -not $runtimeDetached -or
        -not $lifecycleAttachedBeforeDispose -or -not $lifecycleAttachedAfterDispose -or
        -not $lifecyclePostDisposeCallbacks -or -not $lifecycleDetached -or -not $disposedRejectsNewBorrower -or
        $negativeInvocationCount -eq 0 -or $negativeFailureCount -ne $negativeInvocationCount -or
        -not $negativeDetachVerified -or -not $threadSafeHandlerState -or -not $nativeFailureFlagAtomic -or
        -not $realCallbackRuntime -or $syntheticDiagnosticUsed) {
      throw "External logger marker did not satisfy real-message, copied-metadata, owner-lifetime, exception, and detach invariants: $marker"
    }

    $scenarioRuntime = [pscustomobject][ordered]@{
      passed = $true
      beforeOwnerCount = $beforeOwnerCount
      afterBuildCount = $afterBuildCount
      invocationCount = $positiveInvocationCount
      distinctSeverityCount = $positiveSeverityCount
      failureCount = $positiveFailureCount
      metadataCopied = $metadataCopied
      builderAttached = $builderAttached
      builderDetached = $builderDetached
      runtimeAttached = $runtimeAttached
      runtimeDetached = $runtimeDetached
      threadSafeHandlerState = $threadSafeHandlerState
      nativeFailureFlagAtomic = $nativeFailureFlagAtomic
      realCallbackRuntime = $realCallbackRuntime
      syntheticDiagnosticUsed = $syntheticDiagnosticUsed
    }
    $scenarioNegatives = [pscustomobject][ordered]@{
      deferredDispose = [pscustomobject][ordered]@{
        passed = $true
        attachedBeforeDispose = $lifecycleAttachedBeforeDispose
        attachedAfterDispose = $lifecycleAttachedAfterDispose
        postDisposeCallbacks = $lifecyclePostDisposeCallbacks
        detached = $lifecycleDetached
        rejectsNewBorrower = $disposedRejectsNewBorrower
      }
      handlerException = [pscustomobject][ordered]@{
        passed = $true
        operationFailed = $negativeOperationFailed
        invocationCount = $negativeInvocationCount
        failureCount = $negativeFailureCount
        detachVerified = $negativeDetachVerified
      }
    }
    $resultSummary = @(
      "Callbacks=$positiveInvocationCount Severities=$positiveSeverityCount Failures=$positiveFailureCount",
      "BuilderAttached=$builderAttached BuilderDetached=$builderDetached RuntimeAttached=$runtimeAttached RuntimeDetached=$runtimeDetached",
      "DeferredDisposeCallbacks=$lifecyclePostDisposeCallbacks NegativeOperationFailed=$negativeOperationFailed NegativeFailures=$negativeFailureCount"
    )
  }
  "StreamReader" {
    $planBytes = [uint64](Get-MarkerField -Line $marker -Name "PlanBytes")
    $planSha256 = Get-MarkerField -Line $marker -Name "PlanSha256"
    $attemptCount = [uint64](Get-MarkerField -Line $marker -Name "AttemptCount")
    $successfulDeserializeCount = [uint64](Get-MarkerField -Line $marker -Name "SuccessfulDeserializeCount")
    $failedDeserializeCount = [uint64](Get-MarkerField -Line $marker -Name "FailedDeserializeCount")
    $readCount = [uint64](Get-MarkerField -Line $marker -Name "ReadCount")
    $seekCount = [uint64](Get-MarkerField -Line $marker -Name "SeekCount")
    $hostReadCount = [uint64](Get-MarkerField -Line $marker -Name "HostReadCount")
    $deviceReadCount = [uint64](Get-MarkerField -Line $marker -Name "DeviceReadCount")
    $bytesRead = [uint64](Get-MarkerField -Line $marker -Name "BytesRead")
    $failureCount = [uint64](Get-MarkerField -Line $marker -Name "FailureCount")
    $inFlightCallbackCount = [uint64](Get-MarkerField -Line $marker -Name "InFlightCallbackCount")
    $metadataMatched = [bool]::Parse((Get-MarkerField -Line $marker -Name "MetadataMatched"))
    $sourceCopied = [bool]::Parse((Get-MarkerField -Line $marker -Name "SourceCopied"))
    $reusable = [bool]::Parse((Get-MarkerField -Line $marker -Name "Reusable"))
    $lifecycleDisposeDeferred = [bool]::Parse((Get-MarkerField -Line $marker -Name "LifecycleDisposeDeferred"))
    $lifecycleRejectsNewDeserialize = [bool]::Parse((Get-MarkerField -Line $marker -Name "LifecycleRejectsNewDeserialize"))
    $lifecycleReleasedAfterEngine = [bool]::Parse((Get-MarkerField -Line $marker -Name "LifecycleReleasedAfterEngine"))
    $truncatedDeserializeFailed = [bool]::Parse((Get-MarkerField -Line $marker -Name "TruncatedDeserializeFailed"))
    $truncatedAttemptCount = [uint64](Get-MarkerField -Line $marker -Name "TruncatedAttemptCount")
    $truncatedFailedCount = [uint64](Get-MarkerField -Line $marker -Name "TruncatedFailedCount")
    $truncatedFailureCount = [uint64](Get-MarkerField -Line $marker -Name "TruncatedFailureCount")
    $trt8Rejected = [bool]::Parse((Get-MarkerField -Line $marker -Name "Trt8Rejected"))
    $pointerExposed = [bool]::Parse((Get-MarkerField -Line $marker -Name "PointerExposed"))
    $realCallbackRuntime = [bool]::Parse((Get-MarkerField -Line $marker -Name "RealCallbackRuntime"))
    $legacyStreamReaderDeferred = [bool]::Parse((Get-MarkerField -Line $marker -Name "LegacyStreamReaderDeferred"))
    $streamWriterVerified = [bool]::Parse((Get-MarkerField -Line $marker -Name "StreamWriterVerified"))
    if ($planBytes -eq 0 -or $planSha256 -notmatch '^[0-9a-f]{64}$' -or
        $attemptCount -ne 2 -or $successfulDeserializeCount -ne 2 -or $failedDeserializeCount -ne 0 -or
        $readCount -eq 0 -or ($hostReadCount + $deviceReadCount) -eq 0 -or $bytesRead -eq 0 -or
        $failureCount -ne 0 -or $inFlightCallbackCount -ne 0 -or
        -not $metadataMatched -or -not $sourceCopied -or -not $reusable -or
        -not $lifecycleDisposeDeferred -or -not $lifecycleRejectsNewDeserialize -or -not $lifecycleReleasedAfterEngine -or
        -not $truncatedDeserializeFailed -or $truncatedAttemptCount -ne 1 -or $truncatedFailedCount -ne 1 -or $truncatedFailureCount -eq 0 -or
        -not $trt8Rejected -or $pointerExposed -or -not $realCallbackRuntime -or
        -not $legacyStreamReaderDeferred -or $streamWriterVerified) {
      throw "External stream reader marker did not satisfy real callback, immutable-source, lifecycle, fail-closed, and version-boundary invariants: $marker"
    }

    $scenarioRuntime = [pscustomobject][ordered]@{
      passed = $true
      planBytes = $planBytes
      planSha256 = $planSha256
      deserializeAttemptCount = $attemptCount
      successfulDeserializeCount = $successfulDeserializeCount
      readCount = $readCount
      seekCount = $seekCount
      hostReadCount = $hostReadCount
      deviceReadCount = $deviceReadCount
      bytesRead = $bytesRead
      failureCount = $failureCount
      inFlightCallbackCount = $inFlightCallbackCount
      metadataMatched = $metadataMatched
      sourceCopied = $sourceCopied
      reusable = $reusable
      nativePointerExposed = $pointerExposed
      realCallbackRuntime = $realCallbackRuntime
    }
    $scenarioNegatives = [pscustomobject][ordered]@{
      deferredDispose = [pscustomobject][ordered]@{
        passed = $true
        disposeDeferred = $lifecycleDisposeDeferred
        rejectsNewDeserialize = $lifecycleRejectsNewDeserialize
        releasedAfterEngine = $lifecycleReleasedAfterEngine
      }
      truncatedInput = [pscustomobject][ordered]@{
        passed = $true
        deserializeFailed = $truncatedDeserializeFailed
        attemptCount = $truncatedAttemptCount
        failedCount = $truncatedFailedCount
        failureCount = $truncatedFailureCount
      }
      versionBoundary = [pscustomobject][ordered]@{
        passed = $true
        tensorRt8Rejected = $trt8Rejected
        legacyStreamReaderDeferred = $legacyStreamReaderDeferred
        streamWriterVerified = $streamWriterVerified
      }
    }
    $resultSummary = @(
      "Attempts=$attemptCount Success=$successfulDeserializeCount Reads=$readCount Seeks=$seekCount Bytes=$bytesRead",
      "HostReads=$hostReadCount DeviceReads=$deviceReadCount Failures=$failureCount InFlight=$inFlightCallbackCount",
      "DeferredDispose=$lifecycleDisposeDeferred TruncatedFailed=$truncatedDeserializeFailed PointerExposed=$pointerExposed"
    )
  }
}

$projectText = Get-Content -LiteralPath $projectPath -Raw -Encoding utf8
if ($projectText -match "<ProjectReference\b" -or $projectText -match "RepositoryRoot" -or $projectText -match "HintPath") {
  throw "External consumer project contains a source-tree dependency."
}
$consumerOutput = Join-Path $OutputRoot "bin\Release\net8.0\win-x64"
$consumerBridge = Join-Path $consumerOutput "jyppxtrtbridge.dll"
if (-not (Test-Path -LiteralPath $consumerBridge -PathType Leaf)) {
  throw "Bridge package did not place jyppxtrtbridge.dll in the consumer output."
}
$consumerVendorBinaries = @(
  Get-ChildItem -LiteralPath $consumerOutput -File |
    Where-Object { $_.Name -match '(?i)^(nvinfer|nvonnxparser|cudnn|cudart|nvrtc).+\.dll$' }
)
if ($consumerVendorBinaries.Count -ne 0) {
  throw "External consumer output unexpectedly contains vendor runtime binaries."
}
$bridgeEntryHash = [string]$bridgePackage.entryHashes["runtimes/win-x64/native/jyppxtrtbridge.dll"]
$consumerBridgeHash = (Get-FileHash -LiteralPath $consumerBridge -Algorithm SHA256).Hash.ToLowerInvariant()
if ($consumerBridgeHash -ne $bridgeEntryHash) {
  throw "Consumer bridge hash does not match the bridge-only package entry."
}

$stdoutFile = Get-Item -LiteralPath $stdoutPath
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "$($scenarioConfig.slug)-local-package-consumer-runtime"
  scenario = $Scenario
  validationState = "passed-local-package-consumer-runtime"
  evidenceClassification = "local-package-consumer-runtime"
  recordedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runtimePackageKey = $RuntimePackageKey
  tensorRtLine = [int]$TensorRtLine
  tensorRtVersion = $runtimeTensorRtVersion
  cudaToolkitVersion = $runtimeCudaToolkitVersion
  packages = @(
    [pscustomobject][ordered]@{ role = "managed-api"; id = $managedPackage.id; version = $managedPackage.version; length = $managedPackage.length; sha256 = $managedPackage.sha256 },
    [pscustomobject][ordered]@{ role = "bridge-only"; id = $bridgePackage.id; version = $bridgePackage.version; length = $bridgePackage.length; sha256 = $bridgePackage.sha256; nativeEntry = $bridgeNativeEntries[0]; nativeEntrySha256 = $bridgeEntryHash }
  )
  isolation = [pscustomobject][ordered]@{
    outputRootOutsideRepository = $true
    usesProjectReference = $false
    usesHintPath = $false
    usesSourceTreeBinary = $false
    developmentBridgePathRemoved = $true
    developmentProbingRemoved = $true
    consumerBridgeSha256 = $consumerBridgeHash
    consumerBridgeMatchesPackage = $true
    vendorRuntimeBinaryCountInPackages = $vendorBinaryEntries.Count
    vendorRuntimeBinaryCountInConsumerOutput = $consumerVendorBinaries.Count
    vendorDependenciesProvidedByHost = $true
  }
  runtime = $scenarioRuntime
  negatives = $scenarioNegatives
  artifacts = [pscustomobject][ordered]@{
    stdoutPath = $stdoutPath
    stdoutLength = $stdoutFile.Length
    stdoutSha256 = (Get-FileHash -LiteralPath $stdoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
    reportPath = $reportPath
  }
  boundary = [pscustomobject][ordered]@{
    isLocalPackageConsumerRuntimeEvidence = $true
    isPublicPackageConsumerProof = $false
    isReleaseProof = $false
    isPostPublishProof = $false
    performsPublish = $false
    statement = "Repository-external local-feed consumer proof for managed and bridge-only packages. CUDA and TensorRT are host-installed; no public package or release claim is made."
  }
}
Write-Utf8Text -Path $reportPath -Value (($report | ConvertTo-Json -Depth 12) + [Environment]::NewLine)

Write-Host "$($Scenario)LocalPackageConsumer=Passed"
Write-Host "  Report=$reportPath"
Write-Host "  Stdout=$stdoutPath"
foreach ($line in $resultSummary) {
  Write-Host "  $line"
}
Write-Host "  ProjectReference=False SourceTreeBinary=False VendorRuntimeBundled=False"
Write-Host "  PublicPackageProof=False PerformsPublish=False"

if (-not $KeepWorkspace.IsPresent) {
  Remove-ConsumerDirectory -Path $OutputRoot
}
