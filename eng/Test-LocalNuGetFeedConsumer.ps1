[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$TargetFramework = "net8.0",
  [string]$ManagedPackageDirectory,
  [string]$RuntimePackageDirectory,
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$RunRuntimeProbe,
  [switch]$SkipRun,
  [switch]$AllowSmokeFailure,
  [switch]$PreserveConsumerOutput
)

$ErrorActionPreference = "Stop"
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
elseif (-not [System.IO.Path]::IsPathRooted($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ManagedPackageDirectory))
}

if ([string]::IsNullOrWhiteSpace($RuntimePackageDirectory)) {
  $RuntimePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-nupkg"
}
elseif (-not [System.IO.Path]::IsPathRooted($RuntimePackageDirectory)) {
  $RuntimePackageDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $RuntimePackageDirectory))
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "build-out\local-nuget-feed-consumer"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $OutputRoot))
}

function Join-PathMany {
  param([string[]]$Parts)

  $result = $Parts[0]
  for ($index = 1; $index -lt $Parts.Count; $index++) {
    $result = Join-Path $result $Parts[$index]
  }

  return $result
}

function Read-NupkgIdentity {
  param([Parameter(Mandatory = $true)][string]$Path)

  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $entry = $zip.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
    if (-not $entry) {
      throw "Package '$Path' does not contain a .nuspec file."
    }

    $stream = $entry.Open()
    try {
      $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
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

    return [pscustomobject]@{
      id = [string]$xml.package.metadata.id
      version = [string]$xml.package.metadata.version
      path = $Path
      lastWriteTimeUtc = (Get-Item -LiteralPath $Path).LastWriteTimeUtc
    }
  }
  finally {
    $zip.Dispose()
  }
}

function Find-Nupkg {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$PackageId
  )

  $matches = @(
    Get-ChildItem -LiteralPath $Directory -Filter "*.nupkg" -File -Recurse -ErrorAction SilentlyContinue |
      ForEach-Object { Read-NupkgIdentity -Path $_.FullName } |
      Where-Object { [string]::Equals($_.id, $PackageId, [System.StringComparison]::OrdinalIgnoreCase) }
  )

  if ($matches.Count -eq 0) {
    throw "Package '$PackageId' was not found under '$Directory'."
  }

  return $matches |
    Sort-Object `
      @{ Expression = { $_.lastWriteTimeUtc }; Descending = $true },
      @{ Expression = { $_.version }; Descending = $true },
      @{ Expression = { $_.path }; Descending = $true } |
    Select-Object -First 1
}

function New-NuGetConfigContent {
  param([Parameter(Mandatory = $true)][string]$FeedDirectory)

  $escapedFeed = [System.Security.SecurityElement]::Escape($FeedDirectory)
  return @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local-release-candidate-feed" value="$escapedFeed" />
  </packageSources>
</configuration>
"@
}

function Get-ExpectedNativeFileNames {
  param([object]$RuntimePackage)

  $expected = @($RuntimePackage.bridgeFile)
  foreach ($relativePath in @($RuntimePackage.tensorRtFiles + $RuntimePackage.cudaFiles + $RuntimePackage.cudnnFiles)) {
    if ([string]::IsNullOrWhiteSpace([string]$relativePath)) {
      continue
    }

    $expected += [System.IO.Path]::GetFileName([string]$relativePath)
  }

  return @($expected | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Remove-DirectoryWithRetries {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  for ($attempt = 1; $attempt -le 6; $attempt++) {
    try {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
      return
    }
    catch {
      if ($attempt -eq 1) {
        try { & dotnet build-server shutdown *> $null } catch { }
      }

      [System.GC]::Collect()
      [System.GC]::WaitForPendingFinalizers()
      Start-Sleep -Milliseconds (250 * $attempt)
    }
  }

  if (Test-Path -LiteralPath $Path) {
    throw "Unable to remove local feed consumer directory: $Path"
  }
}

function Classify-RunResult {
  param(
    [int]$ExitCode,
    [string[]]$OutputLines,
    [bool]$RuntimeProbe
  )

  $text = @($OutputLines) -join "`n"
  if ($ExitCode -eq 0 -and $text -match "DependencyProbe BridgeInitialized=True") {
    if ($RuntimeProbe -and $text -match "RuntimeProbe=Succeeded") {
      return [pscustomobject]@{
        status = "runtime-probe-passed"
        diagnostic = "local feed consumer restored, built, copied native assets, and runtime probe succeeded."
      }
    }

    return [pscustomobject]@{
      status = "dependency-probe-passed"
      diagnostic = "local feed consumer restored, built, copied native assets, and dependency probe succeeded."
    }
  }

  if ($text -match "CUDA error 35" -or $text -match "cudaRuntimeGetVersion failed with CUDA error 35") {
    return [pscustomobject]@{
      status = "blocked-by-cuda-driver"
      diagnostic = "local feed consumer reached packaged runtime but CUDA driver/runtime compatibility blocked execution; cudaRuntimeGetVersion reported CUDA error 35."
    }
  }

  if ($text -match "0x800711C7" -or $text -match "Windows application control") {
    return [pscustomobject]@{
      status = "blocked-by-application-control"
      diagnostic = "local feed consumer output was blocked by Windows application control policy."
    }
  }

  return [pscustomobject]@{
    status = if ($ExitCode -eq 0) { "passed-with-unclassified-output" } else { "failed" }
    diagnostic = if ($ExitCode -eq 0) { "local feed consumer completed without a recognized probe marker." } else { "local feed consumer failed with exit code $ExitCode." }
  }
}

$timer = [System.Diagnostics.Stopwatch]::StartNew()
$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimePackage = @($manifest.packages | Where-Object { [string]::Equals([string]$_.key, $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) }) | Select-Object -First 1
if (-not $runtimePackage) {
  throw "Runtime package key '$RuntimePackageKey' was not found in '$manifestPath'."
}

$managedPackage = Find-Nupkg -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
$runtimeNupkg = Find-Nupkg -Directory $RuntimePackageDirectory -PackageId $runtimePackage.packageId

$consumerRoot = Join-Path $OutputRoot $RuntimePackageKey
$resolvedConsumerRoot = [System.IO.Path]::GetFullPath($consumerRoot)
$resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $resolvedConsumerRoot.StartsWith($resolvedOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to clean consumer path outside output root: $resolvedConsumerRoot"
}

Remove-DirectoryWithRetries -Path $resolvedConsumerRoot
New-Item -ItemType Directory -Path $resolvedConsumerRoot -Force | Out-Null

$feedRoot = Join-Path $resolvedConsumerRoot "local-feed"
New-Item -ItemType Directory -Path $feedRoot -Force | Out-Null
Copy-Item -LiteralPath $managedPackage.path -Destination $feedRoot -Force
Copy-Item -LiteralPath $runtimeNupkg.path -Destination $feedRoot -Force

$restorePackagesPath = Join-Path "C:\jyppx-pkgcache" ("local-feed-" + $RuntimePackageKey)
Remove-DirectoryWithRetries -Path $restorePackagesPath

$project = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$TargetFramework</TargetFramework>
    <RuntimeIdentifier>$($runtimePackage.rid)</RuntimeIdentifier>
    <SelfContained>false</SelfContained>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <RestorePackagesPath>$restorePackagesPath</RestorePackagesPath>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="$($managedPackage.id)" Version="$($managedPackage.version)" />
    <PackageReference Include="$($runtimeNupkg.id)" Version="$($runtimeNupkg.version)" />
  </ItemGroup>
</Project>
"@

$program = @"
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

string runtimePackageKey = "$RuntimePackageKey";
Console.WriteLine("LocalFeedConsumer=compiled");
Console.WriteLine("RuntimePackageKey=" + runtimePackageKey);
Console.WriteLine("TensorRtAssemblyBridge=" + typeof(TensorRtEnvironmentProbe).Assembly.GetName().Name);
Console.WriteLine("CudaAssemblyBridge=" + typeof(CudaEnvironmentProbe).Assembly.GetName().Name);
Console.WriteLine("HighLevelWrapperSurface=" + CreateHighLevelWrapperSurfaceSummary());

TensorRtDependencyProbeReport dependencyProbe = TensorRtEnvironmentProbe.ProbeNativeDependencies(TensorRtApiLine.TensorRt11);
Console.WriteLine("DependencyProbe BridgeInitialized=" + dependencyProbe.BridgeInitialized +
    " Candidates=" + dependencyProbe.NativeBridgeCandidates.Count +
    " Loaded=" + dependencyProbe.LoadedModuleCount +
    " SearchPathCandidates=" + dependencyProbe.SearchPathCandidateCount +
    " Diagnostics=" + dependencyProbe.Diagnostics.Count +
    " Message=" + dependencyProbe.BridgeDiagnostic);

if (args.Contains("--runtime-probe", StringComparer.OrdinalIgnoreCase))
{
    try
    {
        TensorRtEnvironmentSnapshot tensorRt = TensorRtEnvironmentProbe.GetCurrent();
        CudaEnvironmentSnapshot cuda = CudaEnvironmentProbe.GetCurrent();
        Console.WriteLine("RuntimeProbe=Succeeded Bridge=" + tensorRt.BuildInfo.BridgeName +
            " TRT=" + tensorRt.BuildInfo.TensorRtVersion +
            " CUDA=" + tensorRt.BuildInfo.CudaToolkitVersion +
            " DeviceCount=" + cuda.CudaRuntimeInfo.DeviceCount);
    }
    catch (Exception exception)
    {
        Console.WriteLine("RuntimeProbe=Blocked");
        Console.WriteLine("RuntimeProbeException=" + exception.GetType().FullName + ": " + exception.Message.Replace(Environment.NewLine, " "));
        return 35;
    }
}

return 0;

static string CreateHighLevelWrapperSurfaceSummary()
{
    string[] markers =
    [
        "compiled:local-feed-consumer",
        "runtime-package-matrix",
        typeof(TensorRtEnvironmentProbe).Name,
        typeof(CudaEnvironmentProbe).Name,
        typeof(TensorRtPluginRegistryInventory).Name,
        nameof(TensorRtPluginRegistryInventory.FindCreator),
        nameof(TensorRtPluginRegistryInventory.TryFindCreator),
        nameof(TensorRtEnvironmentProbe.IsGlobalPluginRegistryAvailable),
        nameof(TensorRtEnvironmentProbe.TryIsGlobalPluginRegistryAvailable),
        nameof(TensorRtOnnxParser.GetError),
        nameof(TensorRtOnnxParser.IsSubgraphSupported),
        nameof(TensorRtOnnxParserRefitter.GetError),
        nameof(TensorRtBuilderConfig.GetDlaCore),
        nameof(TensorRtBuilderConfig.GetL2LimitForTiling),
        nameof(TensorRtBuilderConfig.GetMaxTactics),
        nameof(TensorRtBuilderConfig.GetQuantizationFlag),
        nameof(TensorRtBuilderConfig.GetQuantizationFlags),
        nameof(TensorRtBuilderConfig.GetAverageTimingIterations),
        nameof(TensorRtBuilder.MaxBatchSizeCompatibility),
        nameof(TensorRtBuilder.MaxDlaBatchSize),
        "no-public-borrowed-plugin-creator-pointer",
        "local-feed-is-not-post-publish-proof"
    ];

    return string.Join(";", markers);
}
"@

Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "NuGet.config") -Value (New-NuGetConfigContent -FeedDirectory $feedRoot) -Encoding utf8
Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "LocalNuGetFeedConsumer.csproj") -Value $project -Encoding utf8
Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "Program.cs") -Value $program -Encoding utf8

$restoreOutput = @(& dotnet restore (Join-Path $resolvedConsumerRoot "LocalNuGetFeedConsumer.csproj") --configfile (Join-Path $resolvedConsumerRoot "NuGet.config") 2>&1)
if ($LASTEXITCODE -ne 0) {
  throw "dotnet restore failed for local feed consumer.`n$($restoreOutput -join "`n")"
}

$buildOutput = @(& dotnet build (Join-Path $resolvedConsumerRoot "LocalNuGetFeedConsumer.csproj") -c Release --no-restore 2>&1)
if ($LASTEXITCODE -ne 0) {
  throw "dotnet build failed for local feed consumer.`n$($buildOutput -join "`n")"
}

$outputDirectory = Join-PathMany -Parts @($resolvedConsumerRoot, "bin", "Release", $TargetFramework, $runtimePackage.rid)
$expectedNativeFiles = @(Get-ExpectedNativeFileNames -RuntimePackage $runtimePackage)
$missingNativeFiles = New-Object System.Collections.Generic.List[string]
foreach ($fileName in $expectedNativeFiles) {
  $found = @(Get-ChildItem -LiteralPath $outputDirectory -Recurse -File -Filter $fileName -ErrorAction SilentlyContinue).Count -gt 0
  if (-not $found) {
    $missingNativeFiles.Add($fileName) | Out-Null
  }
}

$runStatus = "not-run"
$runDiagnostic = "local feed consumer execution was not requested."
$runExitCode = $null
$runOutput = @()
$runCommand = ""
if (-not $SkipRun.IsPresent) {
  $arguments = @("run", "--project", (Join-Path $resolvedConsumerRoot "LocalNuGetFeedConsumer.csproj"), "-c", "Release", "--no-build", "--")
  if ($RunRuntimeProbe.IsPresent) {
    $arguments += "--runtime-probe"
  }
  else {
    $arguments += "--dependency-probe-only"
  }

  $runCommand = "dotnet " + ($arguments -join " ")
  $runOutput = @(& dotnet @arguments 2>&1)
  $runExitCode = $LASTEXITCODE
  $classification = Classify-RunResult -ExitCode $runExitCode -OutputLines $runOutput -RuntimeProbe $RunRuntimeProbe.IsPresent
  $runStatus = [string]$classification.status
  $runDiagnostic = [string]$classification.diagnostic
  if ($runExitCode -ne 0 -and -not $AllowSmokeFailure.IsPresent -and $runStatus -notin @("blocked-by-cuda-driver", "blocked-by-application-control")) {
    throw "local feed consumer execution failed with exit code $runExitCode.`n$($runOutput -join "`n")"
  }
}

$timer.Stop()
$projectText = Get-Content -LiteralPath (Join-Path $resolvedConsumerRoot "LocalNuGetFeedConsumer.csproj") -Raw
$summary = [pscustomobject]@{
  RuntimePackageKey = $RuntimePackageKey
  RuntimePackageId = $runtimeNupkg.id
  RuntimePackageVersion = $runtimeNupkg.version
  RuntimePackagePath = $runtimeNupkg.path
  ManagedPackageId = $managedPackage.id
  ManagedPackageVersion = $managedPackage.version
  ManagedPackagePath = $managedPackage.path
  TargetFramework = $TargetFramework
  RuntimeIdentifier = [string]$runtimePackage.rid
  LocalFeedDirectory = $feedRoot
  LocalFeedPackageCount = @(Get-ChildItem -LiteralPath $feedRoot -Filter "*.nupkg" -File).Count
  RestoreSourceMode = "local-feed-only"
  UsesProjectReference = ($projectText -match "<ProjectReference")
  NativeAssetsExpected = $expectedNativeFiles.Count
  NativeAssetsFound = $expectedNativeFiles.Count - $missingNativeFiles.Count
  MissingNativeAssets = @($missingNativeFiles.ToArray())
  RunRequested = (-not $SkipRun.IsPresent)
  RuntimeProbeRequested = $RunRuntimeProbe.IsPresent
  RunStatus = $runStatus
  RunExitCode = $runExitCode
  RunCommand = $runCommand
  RunDiagnostic = $runDiagnostic
  RunOutputLines = @($runOutput)
  ElapsedSeconds = [Math]::Round($timer.Elapsed.TotalSeconds, 2)
  ConsumerOutput = $outputDirectory
  ConsumerOutputPreserved = $PreserveConsumerOutput.IsPresent
}

$reportRoot = Join-Path $RepositoryRoot "artifacts\local-feed-consumer"
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
$jsonPath = Join-Path $reportRoot "local-nuget-feed-consumer-summary.json"
$markdownPath = Join-Path $reportRoot "local-nuget-feed-consumer-summary.md"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Local NuGet Feed Consumer Summary")
$lines.Add("")
$lines.Add("| Runtime key | Managed package | Runtime package | Feed packages | Native assets | Run status | Runtime probe | Diagnostic | Elapsed |")
$lines.Add("| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |")
$lines.Add("| $RuntimePackageKey | ``$($managedPackage.id) $($managedPackage.version)`` | ``$($runtimeNupkg.id) $($runtimeNupkg.version)`` | $($summary.LocalFeedPackageCount) | $($summary.NativeAssetsFound)/$($summary.NativeAssetsExpected) | $runStatus | $($RunRuntimeProbe.IsPresent) | $(ConvertTo-MarkdownCell $runDiagnostic) | $($summary.ElapsedSeconds)s |")
$lines.Add("")
$lines.Add("- restore source mode: ``local-feed-only``")
$lines.Add("- uses project reference: ``$($summary.UsesProjectReference)``")
$lines.Add("- consumer output: ``$outputDirectory``")
$lines.Add("- summary JSON: ``$jsonPath``")
$lines.Add("")
$lines.Add("This gate validates the user-facing NuGet install path. It must not reference source projects, and CUDA error 35 is classified as environment-blocked rather than API proof.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Local NuGet feed consumer summary written to $jsonPath"
Write-Host "Local NuGet feed consumer summary written to $markdownPath"

if (-not $PreserveConsumerOutput.IsPresent) {
  Remove-DirectoryWithRetries -Path $restorePackagesPath
  Remove-DirectoryWithRetries -Path $resolvedConsumerRoot
}

if ($summary.MissingNativeAssets.Count -gt 0) {
  throw "Local feed consumer is missing $($summary.MissingNativeAssets.Count) native asset(s): $($summary.MissingNativeAssets -join ', ')"
}

if ($summary.UsesProjectReference) {
  throw "Local feed consumer must not use ProjectReference."
}

if ($runExitCode -ne 0 -and $runStatus -eq "blocked-by-cuda-driver" -and -not $AllowSmokeFailure.IsPresent) {
  throw "Local feed consumer runtime probe was blocked by CUDA driver/runtime compatibility. Re-run with -AllowSmokeFailure to record it as a non-proof environment blocker."
}
