[CmdletBinding()]
param(
  [string]$SourceRuntimeKey = 'win-x64-trt11.0-cuda12.9-cudnn9.22',
  [string]$Version = '4.0.0-rtc-local.20260728',
  [string]$ManagedPackageVersion,
  [string]$BridgePackageVersion,
  [string]$TargetFramework = 'net8.0',
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9',
  [string]$NvrtcLibrary,
  [string]$OutputRoot,
  [string]$OutputPath,
  [string]$RepositoryRoot,
  [switch]$KeepConsumerOutput
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
$RepositoryRoot = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
$outerRoot = Split-Path -Parent $RepositoryRoot

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot 'artifacts\managed'
}
if ([string]::IsNullOrWhiteSpace($BridgePackageDirectory)) {
  $BridgePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$SourceRuntimeKey"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $outerRoot 'consumer-workspaces\crtc'
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot 'artifacts\cuda-runtime-compilation\bridge-package-consumer.json'
}
if ([string]::IsNullOrWhiteSpace($NvrtcLibrary)) {
  $NvrtcLibrary = Join-Path $CudaRoot 'bin\nvrtc64_120_0.dll'
}

$ManagedPackageDirectory = [System.IO.Path]::GetFullPath($ManagedPackageDirectory)
$BridgePackageDirectory = [System.IO.Path]::GetFullPath($BridgePackageDirectory)
$CudaRoot = [System.IO.Path]::GetFullPath($CudaRoot)
$NvrtcLibrary = [System.IO.Path]::GetFullPath($NvrtcLibrary)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd('\', '/')
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
if ([string]::IsNullOrWhiteSpace($ManagedPackageVersion)) {
  $ManagedPackageVersion = $Version
}
if ([string]::IsNullOrWhiteSpace($BridgePackageVersion)) {
  $BridgePackageVersion = $Version
}

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

function Remove-DirectoryWithRetries {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$AllowedRoot
  )

  if (-not (Test-PathWithin -Path $Path -Parent $AllowedRoot)) {
    throw "Refusing to remove consumer path outside '$AllowedRoot': $Path"
  }
  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  $lastDiagnostic = $null
  for ($attempt = 1; $attempt -le 6; $attempt++) {
    try {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
      return
    }
    catch {
      $lastDiagnostic = $_.Exception.Message
      if ($env:OS -eq 'Windows_NT') {
        try {
          $extendedPath = '\\?\' + [System.IO.Path]::GetFullPath($Path)
          [System.IO.Directory]::Delete($extendedPath, $true)
          if (-not [System.IO.Directory]::Exists($extendedPath)) {
            return
          }
        }
        catch {
          $lastDiagnostic = "$lastDiagnostic | extended-path fallback: $($_.Exception.Message)"
        }
      }
      if ($attempt -eq 1) {
        try { & dotnet build-server shutdown *> $null } catch { }
      }
      [System.GC]::Collect()
      [System.GC]::WaitForPendingFinalizers()
      Start-Sleep -Milliseconds (250 * $attempt)
    }
  }
  throw "Unable to remove consumer path '$Path': $lastDiagnostic"
}

function Get-StreamSha256 {
  param([Parameter(Mandatory = $true)][System.IO.Stream]$Stream)

  $algorithm = [System.Security.Cryptography.SHA256]::Create()
  try {
    return -join ($algorithm.ComputeHash($Stream) | ForEach-Object { $_.ToString('x2') })
  }
  finally {
    $algorithm.Dispose()
  }
}

function Get-NupkgMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)

  $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = $zip.Entries |
      Where-Object { $_.FullName.EndsWith('.nuspec', [System.StringComparison]::OrdinalIgnoreCase) } |
      Select-Object -First 1
    if ($null -eq $nuspec) {
      throw "Package does not contain a nuspec: $Path"
    }

    $stream = $nuspec.Open()
    try {
      $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
      try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
    }
    finally {
      $stream.Dispose()
    }

    $namespaceManager = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
    $namespaceManager.AddNamespace('n', $xml.package.NamespaceURI)
    $entries = @($zip.Entries | ForEach-Object { $_.FullName })
    $entryHashes = [ordered]@{}
    foreach ($entry in @($zip.Entries | Where-Object { -not [string]::IsNullOrEmpty($_.Name) })) {
      $entryStream = $entry.Open()
      try { $entryHashes[$entry.FullName] = Get-StreamSha256 -Stream $entryStream } finally { $entryStream.Dispose() }
    }

    return [pscustomobject]@{
      Path = [System.IO.Path]::GetFullPath($Path)
      Id = $xml.SelectSingleNode('//n:metadata/n:id', $namespaceManager).InnerText
      Version = $xml.SelectSingleNode('//n:metadata/n:version', $namespaceManager).InnerText
      Sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
      Entries = $entries
      EntryHashes = $entryHashes
      LastWriteTimeUtc = (Get-Item -LiteralPath $Path).LastWriteTimeUtc
    }
  }
  finally {
    $zip.Dispose()
  }
}

function Find-Nupkg {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$PackageId,
    [Parameter(Mandatory = $true)][string]$PackageVersion
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }
  $matches = @(
    Get-ChildItem -LiteralPath $Directory -Filter '*.nupkg' -File -Recurse |
      ForEach-Object { Get-NupkgMetadata -Path $_.FullName } |
      Where-Object {
        [string]::Equals($_.Id, $PackageId, [System.StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals($_.Version, $PackageVersion, [System.StringComparison]::OrdinalIgnoreCase)
      }
  )
  if ($matches.Count -ne 1) {
    throw "Expected exactly one package '$PackageId' version '$PackageVersion' under '$Directory'; found $($matches.Count)."
  }
  return $matches[0]
}

function Get-BridgePackageId {
  $manifestPath = Join-Path $RepositoryRoot 'pack\runtime-split\split-runtime-packages.manifest.json'
  $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $package = @($manifest.packages | Where-Object {
      $_.sourceRuntimeKey -eq $SourceRuntimeKey -and $_.role -eq 'bridge'
    }) | Select-Object -First 1
  if ($null -eq $package) {
    throw "Bridge package for '$SourceRuntimeKey' was not found in '$manifestPath'."
  }
  return [string]$package.packageId
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
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
  }

  $oldPreference = $ErrorActionPreference
  Push-Location $WorkingDirectory
  try {
    foreach ($name in $Environment.Keys) {
      [Environment]::SetEnvironmentVariable($name, [string]$Environment[$name], 'Process')
    }
    foreach ($name in $RemoveEnvironment) {
      [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $ErrorActionPreference = 'Continue'
    $lines = @(& $FilePath @ArgumentList 2>&1 | ForEach-Object { [string]$_ })
    $exitCode = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $oldPreference
    Pop-Location
    foreach ($name in $names) {
      [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
  }

  return [pscustomobject]@{
    ExitCode = $exitCode
    Lines = $lines
    Text = $lines -join [Environment]::NewLine
    Command = "$FilePath $($ArgumentList -join ' ')"
  }
}

function Write-Utf8Text {
  param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Value)
  [System.IO.File]::WriteAllText($Path, $Value, $utf8)
}

function Get-MarkerValue {
  param([Parameter(Mandatory = $true)][string[]]$Lines, [Parameter(Mandatory = $true)][string]$Prefix)
  $line = $Lines | Where-Object { $_.StartsWith($Prefix, [System.StringComparison]::Ordinal) } | Select-Object -First 1
  if ($null -eq $line) { return $null }
  return $line.Substring($Prefix.Length)
}

if (-not (Test-Path -LiteralPath $CudaRoot -PathType Container)) {
  throw "CUDA root does not exist: $CudaRoot"
}
if (-not (Test-Path -LiteralPath $NvrtcLibrary -PathType Leaf)) {
  throw "NVRTC library does not exist: $NvrtcLibrary"
}
if (Test-PathWithin -Path $OutputRoot -Parent $RepositoryRoot) {
  throw "Clean consumer output must remain outside the Git repository: $OutputRoot"
}
if (-not (Test-PathWithin -Path $OutputRoot -Parent $outerRoot)) {
  throw "Clean consumer output must remain under the outer workspace root '$outerRoot': $OutputRoot"
}

$bridgePackageId = Get-BridgePackageId
$managedPackage = Find-Nupkg -Directory $ManagedPackageDirectory -PackageId 'JYPPX.TensorRT.CSharp.API' -PackageVersion $ManagedPackageVersion
$bridgePackage = Find-Nupkg -Directory $BridgePackageDirectory -PackageId $bridgePackageId -PackageVersion $BridgePackageVersion
$allPackageEntries = @($managedPackage.Entries + $bridgePackage.Entries)
$nvrtcEntries = @($allPackageEntries | Where-Object { $_ -match '(?i)(^|/)(nvrtc|libnvrtc)[^/]*\.(dll|so)(\.|$)' })
$nvrtcBuiltinsEntries = @($allPackageEntries | Where-Object { $_ -match '(?i)nvrtc-builtins' })
$bridgeNativeEntries = @($bridgePackage.Entries | Where-Object { $_ -match '^runtimes/[^/]+/native/[^/]+$' })
$expectedBridgeEntry = 'runtimes/win-x64/native/jyppxtrtbridge.dll'
if ($nvrtcEntries.Count -ne 0 -or $nvrtcBuiltinsEntries.Count -ne 0) {
  throw 'Bridge-only package set unexpectedly bundles NVRTC or NVRTC builtins.'
}
if ($bridgeNativeEntries.Count -ne 1 -or $bridgeNativeEntries[0] -ne $expectedBridgeEntry) {
  throw "Bridge package native assets must contain only '$expectedBridgeEntry'."
}

$managedXmlEntries = @($managedPackage.Entries | Where-Object { $_ -match '^lib/net8\.0/JYPPX\..*\.xml$' })
if ($managedXmlEntries.Count -lt 3) {
  throw 'Managed package does not contain the expected net8.0 XML documentation assets.'
}
$managedZip = [System.IO.Compression.ZipFile]::OpenRead($managedPackage.Path)
try {
  $surfaceBuilder = [System.Text.StringBuilder]::new()
  foreach ($entry in @($managedZip.Entries | Where-Object { $managedXmlEntries -contains $_.FullName })) {
    $stream = $entry.Open()
    try {
      $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
      try { [void]$surfaceBuilder.AppendLine($reader.ReadToEnd()) } finally { $reader.Dispose() }
    }
    finally { $stream.Dispose() }
  }
  $managedSurface = $surfaceBuilder.ToString()
}
finally { $managedZip.Dispose() }
foreach ($marker in @('CudaRtcCompiler', 'CudaKernelLibrary.Launch', 'CudaDriverModule', 'CudaDriverKernelLaunch')) {
  if ($managedSurface.IndexOf($marker, [System.StringComparison]::Ordinal) -lt 0) {
    throw "Managed package is stale and does not contain public API marker '$marker'."
  }
}

$workspaceToken = 'w-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$workspaceRoot = Join-Path $OutputRoot $workspaceToken
[void][System.IO.Directory]::CreateDirectory($workspaceRoot)
$feedRoot = Join-Path $workspaceRoot 'feed'
$consumerRoot = Join-Path $workspaceRoot 'consumer'
$packageCache = Join-Path $workspaceRoot '.packages'
$dotnetHome = Join-Path $workspaceRoot '.dotnet-home'
foreach ($directory in @($feedRoot, $consumerRoot, $packageCache, $dotnetHome)) {
  [void][System.IO.Directory]::CreateDirectory($directory)
}
Copy-Item -LiteralPath $managedPackage.Path -Destination $feedRoot -Force
Copy-Item -LiteralPath $bridgePackage.Path -Destination $feedRoot -Force

$nugetConfigPath = Join-Path $consumerRoot 'NuGet.config'
$escapedFeed = [System.Security.SecurityElement]::Escape($feedRoot)
Write-Utf8Text -Path $nugetConfigPath -Value @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="jyppx-local-only" value="$escapedFeed" />
  </packageSources>
</configuration>
"@

$projectPath = Join-Path $consumerRoot 'CudaRtcBridgePackageConsumer.csproj'
Write-Utf8Text -Path $projectPath -Value @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$TargetFramework</TargetFramework>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="JYPPX.TensorRT.CSharp.API" Version="$ManagedPackageVersion" />
    <PackageReference Include="$bridgePackageId" Version="$BridgePackageVersion" />
  </ItemGroup>
</Project>
"@

$programPath = Join-Path $consumerRoot 'Program.cs'
Write-Utf8Text -Path $programPath -Value @'
using JYPPX.CudaSharp;
using System.Security.Cryptography;

return args.Contains("--negative", StringComparer.Ordinal) ? RunNegative() : RunPositive();

static int RunNegative()
{
    CudaRtcCapability rtc = CudaRtcCompiler.GetCapability();
    CudaDriverCapability driver = CudaDriver.GetCapability();
    bool diagnosticPresent = !string.IsNullOrWhiteSpace(rtc.DependencyDiagnostic);
    Console.WriteLine($"negative.rtc.available={rtc.IsAvailable} diagnosticPresent={diagnosticPresent} loadedLibrary={rtc.LoadedLibraryName}");
    Console.WriteLine($"negative.driver.available={driver.IsAvailable} version={driver.DriverVersion} loadedLibrary={driver.LoadedLibraryName}");
    if (rtc.IsAvailable || !diagnosticPresent || !driver.IsAvailable)
    {
        return 20;
    }
    return 0;
}

static int RunPositive()
{
    const string sourceText = """
extern "C" __global__ void add_one(const float* input, float* output, int count)
{
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) output[index] = input[index] + 1.0f;
}
""";
    CudaRtcCapability rtc = CudaRtcCompiler.GetCapability();
    CudaDriverCapability driver = CudaDriver.GetCapability();
    Console.WriteLine($"positive.rtc.available={rtc.IsAvailable} version={rtc.Version} loadedLibrary={rtc.LoadedLibraryName}");
    Console.WriteLine($"positive.driver.available={driver.IsAvailable} version={driver.DriverVersion} loadedLibrary={driver.LoadedLibraryName}");
    if (!rtc.IsAvailable || !driver.IsAvailable) return 2;

    var source = new CudaRtcProgramSource(sourceText, "clean-consumer.cu");
    var options = new CudaRtcCompileOptions(targetArchitecture: "compute_75");
    CudaRtcCompilationResult compilation = CudaRtcCompiler.Compile(source, options);
    CudaRtcArtifact? ptx = compilation.FindArtifact(CudaRtcArtifactKind.Ptx);
    Console.WriteLine($"positive.compile.success={compilation.Success} logLength={compilation.Log.Length} ptxPresent={ptx != null}");
    if (!compilation.Success || ptx == null) return 3;

    CudaRtcCompilationResult failure = CudaRtcCompiler.Compile(
        new CudaRtcProgramSource("extern \"C\" __global__ void broken( {", "broken.cu"),
        options);
    bool failureDiagnostic = !failure.Success && failure.ResultCode == CudaRtcResultCode.Compilation && !string.IsNullOrWhiteSpace(failure.Log);
    Console.WriteLine($"positive.failureDiagnostic={failureDiagnostic} result={failure.ResultCode} logLength={failure.Log.Length}");
    if (!failureDiagnostic) return 4;

    byte[] code = ptx.ToArray();
    LaunchResult runtime = RunRuntimeLibrary(code);
    LaunchResult driverResult = RunDriver(code);
    Console.WriteLine($"positive.runtime.launch={runtime.Launch} readback={runtime.Readback} correctness={runtime.Correctness} ownersReleased={runtime.OwnersReleased} outputSha256={runtime.OutputSha256}");
    Console.WriteLine($"positive.driver.launch={driverResult.Launch} readback={driverResult.Readback} correctness={driverResult.Correctness} ownersReleased={driverResult.OwnersReleased} outputSha256={driverResult.OutputSha256}");
    bool hashesMatch = string.Equals(runtime.OutputSha256, driverResult.OutputSha256, StringComparison.Ordinal);
    Console.WriteLine($"positive.outputHashesMatch={hashesMatch}");
    return runtime.IsProof && driverResult.IsProof && hashesMatch ? 0 : 5;
}

static LaunchResult RunRuntimeLibrary(byte[] code)
{
    const int count = 65;
    float[] inputValues = Enumerable.Range(0, count).Select(index => index * 0.5f).ToArray();
    using CudaKernelLibrary library = CudaKernelLibrary.Load(code);
    using var input = new CudaMemory(count * sizeof(float));
    using var output = new CudaMemory(count * sizeof(float));
    using var stream = new CudaStream();
    input.CopyFrom(inputValues);
    output.Fill(0);
    var configuration = new CudaKernelLaunchConfiguration(new CudaDim3(2), new CudaDim3(64));
    using CudaKernelLaunch launch = library.Launch(
        "add_one", configuration, stream,
        CudaKernelArgument.FromDeviceMemory(input),
        CudaKernelArgument.FromDeviceMemory(output),
        CudaKernelArgument.FromInt32(count));
    library.Dispose();
    stream.Dispose();
    input.Dispose();
    bool ownersReleased = true;
    launch.Synchronize();
    float[] actual = output.ToSingleArray(count);
    bool correctness = actual.Select((value, index) => Math.Abs(value - (inputValues[index] + 1.0f)) <= 1e-6f).All(value => value);
    return new LaunchResult(launch.IsCompleted, actual.Length == count, correctness, ownersReleased, Hash(actual));
}

static LaunchResult RunDriver(byte[] code)
{
    const int count = 65;
    float[] inputValues = Enumerable.Range(0, count).Select(index => index * 0.5f).ToArray();
    using CudaDriverModule module = CudaDriverModule.Load(code);
    using var input = new CudaMemory(count * sizeof(float));
    using var output = new CudaMemory(count * sizeof(float));
    using var stream = new CudaStream();
    input.CopyFrom(inputValues);
    output.Fill(0);
    var configuration = new CudaKernelLaunchConfiguration(new CudaDim3(2), new CudaDim3(64));
    using CudaDriverKernelLaunch launch = module.Launch(
        "add_one", configuration, stream,
        CudaKernelArgument.FromDeviceMemory(input),
        CudaKernelArgument.FromDeviceMemory(output),
        CudaKernelArgument.FromInt32(count));
    module.Dispose();
    stream.Dispose();
    input.Dispose();
    bool ownersReleased = true;
    launch.Synchronize();
    float[] actual = output.ToSingleArray(count);
    bool correctness = actual.Select((value, index) => Math.Abs(value - (inputValues[index] + 1.0f)) <= 1e-6f).All(value => value);
    return new LaunchResult(launch.IsCompleted, actual.Length == count, correctness, ownersReleased, Hash(actual));
}

static string Hash(float[] values)
{
    byte[] bytes = new byte[values.Length * sizeof(float)];
    Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
    return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
}

readonly record struct LaunchResult(bool Launch, bool Readback, bool Correctness, bool OwnersReleased, string OutputSha256)
{
    public bool IsProof => Launch && Readback && Correctness && OwnersReleased;
}
'@

$dotnetPath = (Get-Command dotnet -ErrorAction Stop).Source
$commonEnvironment = @{
  NUGET_PACKAGES = $packageCache
  DOTNET_CLI_HOME = $dotnetHome
  DOTNET_NOLOGO = '1'
  DOTNET_SKIP_FIRST_TIME_EXPERIENCE = '1'
}
$restore = Invoke-CapturedCommand -FilePath $dotnetPath -ArgumentList @(
  'restore', $projectPath, '--configfile', $nugetConfigPath,
  "-p:RestorePackagesPath=$packageCache"
) -WorkingDirectory $consumerRoot -Environment $commonEnvironment
if ($restore.ExitCode -ne 0) {
  throw "Clean consumer restore failed: $($restore.Text)"
}
$build = Invoke-CapturedCommand -FilePath $dotnetPath -ArgumentList @(
  'build', $projectPath, '-c', 'Release', '--no-restore', '--verbosity', 'minimal',
  "-p:RestorePackagesPath=$packageCache"
) -WorkingDirectory $consumerRoot -Environment $commonEnvironment
if ($build.ExitCode -ne 0) {
  throw "Clean consumer build failed: $($build.Text)"
}

$outputDirectory = Join-Path $consumerRoot "bin\Release\$TargetFramework\win-x64"
$consumerDll = Join-Path $outputDirectory 'CudaRtcBridgePackageConsumer.dll'
$copiedBridge = Join-Path $outputDirectory 'jyppxtrtbridge.dll'
if (-not (Test-Path -LiteralPath $consumerDll -PathType Leaf)) {
  throw "Clean consumer output was not found: $consumerDll"
}
if (-not (Test-Path -LiteralPath $copiedBridge -PathType Leaf)) {
  throw "Bridge package did not copy its native asset: $copiedBridge"
}
$copiedBridgeHash = (Get-FileHash -LiteralPath $copiedBridge -Algorithm SHA256).Hash.ToLowerInvariant()
$packageBridgeHash = [string]$bridgePackage.EntryHashes[$expectedBridgeEntry]
if (-not [string]::Equals($copiedBridgeHash, $packageBridgeHash, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw 'Copied bridge hash does not match the bridge package entry hash.'
}

$systemPath = @(
  [Environment]::GetFolderPath([Environment+SpecialFolder]::System),
  [Environment]::GetFolderPath([Environment+SpecialFolder]::Windows)
) -join ';'
$negativeEnvironment = @{}
foreach ($entry in $commonEnvironment.GetEnumerator()) { $negativeEnvironment[$entry.Key] = $entry.Value }
$negativeEnvironment['PATH'] = $systemPath
$negativeEnvironment['JYPPX_NVRTC_LIBRARY'] = Join-Path $workspaceRoot 'missing\nvrtc-does-not-exist.dll'
$negative = Invoke-CapturedCommand -FilePath $dotnetPath -ArgumentList @($consumerDll, '--negative') `
  -WorkingDirectory $outputDirectory -Environment $negativeEnvironment `
  -RemoveEnvironment @('CUDA_PATH', 'JYPPX_CUDA_ROOT', 'JYPPX_NATIVE_BRIDGE_PATH', 'JYPPX_CUDA_DRIVER_LIBRARY')
if ($negative.ExitCode -ne 0) {
  throw "Negative dependency diagnostic run failed: $($negative.Text)"
}
$negativeRtcAvailable = Get-MarkerValue -Lines $negative.Lines -Prefix 'negative.rtc.available='
$negativeDriverAvailable = Get-MarkerValue -Lines $negative.Lines -Prefix 'negative.driver.available='
$negativeDriverMatch = [regex]::Match(
  $negativeDriverAvailable,
  '^True version=(?<version>[0-9]+) loadedLibrary=(?<library>.+)$')
if ($negativeRtcAvailable -notmatch '^False diagnosticPresent=True ' -or -not $negativeDriverMatch.Success) {
  throw "Negative dependency diagnostic markers were incomplete: $($negative.Text)"
}

$cudaSearchDirectories = @(
  (Join-Path $CudaRoot 'bin\x64'),
  (Join-Path $CudaRoot 'bin')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
$positiveEnvironment = @{}
foreach ($entry in $commonEnvironment.GetEnumerator()) { $positiveEnvironment[$entry.Key] = $entry.Value }
$positiveEnvironment['PATH'] = (@($cudaSearchDirectories) + @($env:PATH)) -join ';'
$positiveEnvironment['CUDA_PATH'] = $CudaRoot
$positiveEnvironment['JYPPX_CUDA_ROOT'] = $CudaRoot
$positiveEnvironment['JYPPX_NVRTC_LIBRARY'] = $NvrtcLibrary
$positive = Invoke-CapturedCommand -FilePath $dotnetPath -ArgumentList @($consumerDll, '--positive') `
  -WorkingDirectory $outputDirectory -Environment $positiveEnvironment `
  -RemoveEnvironment @('JYPPX_NATIVE_BRIDGE_PATH', 'JYPPX_CUDA_DRIVER_LIBRARY')
if ($positive.ExitCode -ne 0) {
  throw "Positive CUDA RTC package consumer run failed: $($positive.Text)"
}

$positiveRtc = Get-MarkerValue -Lines $positive.Lines -Prefix 'positive.rtc.available='
$positiveDriver = Get-MarkerValue -Lines $positive.Lines -Prefix 'positive.driver.available='
$positiveCompile = Get-MarkerValue -Lines $positive.Lines -Prefix 'positive.compile.success='
$positiveFailure = Get-MarkerValue -Lines $positive.Lines -Prefix 'positive.failureDiagnostic='
$positiveRuntime = Get-MarkerValue -Lines $positive.Lines -Prefix 'positive.runtime.launch='
$positiveDriverLaunch = Get-MarkerValue -Lines $positive.Lines -Prefix 'positive.driver.launch='
$positiveHashesMatch = Get-MarkerValue -Lines $positive.Lines -Prefix 'positive.outputHashesMatch='
$positiveRtcMatch = [regex]::Match(
  $positiveRtc,
  '^True version=(?<version>[0-9]+\.[0-9]+) loadedLibrary=(?<library>.+)$')
$positiveDriverMatch = [regex]::Match(
  $positiveDriver,
  '^True version=(?<version>[0-9]+) loadedLibrary=(?<library>.+)$')
if (-not $positiveRtcMatch.Success -or -not $positiveDriverMatch.Success -or
    $positiveCompile -notmatch '^True ' -or $positiveFailure -notmatch '^True ' -or
    $positiveRuntime -notmatch '^True readback=True correctness=True ownersReleased=True outputSha256=([0-9a-f]{64})$' -or
    $positiveDriverLaunch -notmatch '^True readback=True correctness=True ownersReleased=True outputSha256=([0-9a-f]{64})$' -or
    $positiveHashesMatch -ne 'True') {
  throw "Positive runtime markers were incomplete: $($positive.Text)"
}
$runtimeOutputSha256 = ($positiveRuntime -replace '^.*outputSha256=', '')
$driverOutputSha256 = ($positiveDriverLaunch -replace '^.*outputSha256=', '')

$restoreLogPath = Join-Path $workspaceRoot 'restore.log'
$buildLogPath = Join-Path $workspaceRoot 'build.log'
$negativeLogPath = Join-Path $workspaceRoot 'negative.log'
$positiveLogPath = Join-Path $workspaceRoot 'positive.log'
Write-Utf8Text -Path $restoreLogPath -Value ($restore.Text + [Environment]::NewLine)
Write-Utf8Text -Path $buildLogPath -Value ($build.Text + [Environment]::NewLine)
Write-Utf8Text -Path $negativeLogPath -Value ($negative.Text + [Environment]::NewLine)
Write-Utf8Text -Path $positiveLogPath -Value ($positive.Text + [Environment]::NewLine)

$projectText = [System.IO.File]::ReadAllText($projectPath)
$report = [ordered]@{
  schemaVersion = 1
  recordKind = 'cuda-rtc-bridge-package-clean-consumer'
  generatedLocalDate = (Get-Date -Format 'yyyy-MM-dd')
  sourceRuntimeKey = $SourceRuntimeKey
  proofClassification = 'local-feed-clean-package-consumer-candidate'
  performsDownload = $false
  performsPublish = $false
  canPromotePublicPackageProof = $false
  canPromotePostPublishProof = $false
  packages = [ordered]@{
    managed = [ordered]@{
      id = $managedPackage.Id
      version = $managedPackage.Version
      sha256 = $managedPackage.Sha256
      path = $managedPackage.Path
      containsCurrentCudaRtcSurface = $true
    }
    bridge = [ordered]@{
      id = $bridgePackage.Id
      version = $bridgePackage.Version
      sha256 = $bridgePackage.Sha256
      path = $bridgePackage.Path
      nativeEntries = $bridgeNativeEntries
      containsOnlyBridgeNativeAsset = $true
      copiedBridgeSha256 = $copiedBridgeHash
      packageBridgeEntrySha256 = $packageBridgeHash
    }
    containsNvrtc = $false
    containsNvrtcBuiltins = $false
    nvrtcEntries = $nvrtcEntries
    nvrtcBuiltinsEntries = $nvrtcBuiltinsEntries
  }
  consumer = [ordered]@{
    root = $consumerRoot
    rootOutsideRepository = -not (Test-PathWithin -Path $consumerRoot -Parent $RepositoryRoot)
    cleanWorkspaceCreated = $true
    workspaceToken = $workspaceToken
    targetFramework = $TargetFramework
    runtimeIdentifier = 'win-x64'
    usesPackageReferenceOnly = $projectText -match '<PackageReference' -and $projectText -notmatch '<ProjectReference'
    usesProjectReference = $projectText -match '<ProjectReference'
    nugetSourcesCleared = ([System.IO.File]::ReadAllText($nugetConfigPath)) -match '<clear\s*/>'
    localFeedOnly = ([System.IO.File]::ReadAllText($nugetConfigPath)) -notmatch 'https?://'
    restorePackagesPath = $packageCache
    nativeBridgeEnvironmentOverrideUsed = $false
    restoreSucceeded = $restore.ExitCode -eq 0
    buildSucceeded = $build.ExitCode -eq 0
    projectSha256 = (Get-FileHash -LiteralPath $projectPath -Algorithm SHA256).Hash.ToLowerInvariant()
    programSha256 = (Get-FileHash -LiteralPath $programPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  dependencyDiagnostic = [ordered]@{
    isolatedPath = $true
    cudaSearchPathRemoved = $true
    rtcAvailable = $false
    diagnosticPresent = $true
    driverAvailable = $true
    driverVersion = [int]$negativeDriverMatch.Groups['version'].Value
    driverLoadedLibrary = $negativeDriverMatch.Groups['library'].Value
    processExitCode = $negative.ExitCode
  }
  runtime = [ordered]@{
    cudaRoot = $CudaRoot
    nvrtcLibrary = $NvrtcLibrary
    nvrtcLibrarySha256 = (Get-FileHash -LiteralPath $NvrtcLibrary -Algorithm SHA256).Hash.ToLowerInvariant()
    rtcAvailable = $true
    rtcVersion = $positiveRtcMatch.Groups['version'].Value
    rtcLoadedLibrary = $positiveRtcMatch.Groups['library'].Value
    driverAvailable = $true
    driverVersion = [int]$positiveDriverMatch.Groups['version'].Value
    driverLoadedLibrary = $positiveDriverMatch.Groups['library'].Value
    compileSucceeded = $true
    compileFailureDiagnosticCaptured = $true
    runtimeLibraryLaunch = $true
    runtimeLibraryReadback = $true
    runtimeLibraryCorrectness = $true
    runtimeLibraryOwnersReleasedBeforeSynchronize = $true
    driverLaunch = $true
    driverReadback = $true
    driverCorrectness = $true
    driverOwnersReleasedBeforeSynchronize = $true
    outputHashesMatch = $true
    runtimeOutputSha256 = $runtimeOutputSha256
    driverOutputSha256 = $driverOutputSha256
    processExitCode = $positive.ExitCode
  }
  logs = [ordered]@{
    restoreSha256 = (Get-FileHash -LiteralPath $restoreLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    buildSha256 = (Get-FileHash -LiteralPath $buildLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    negativeSha256 = (Get-FileHash -LiteralPath $negativeLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    positiveSha256 = (Get-FileHash -LiteralPath $positiveLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  proofBoundary = 'This proves a repository-external PackageReference consumer restored from a local-only feed, copied the packaged bridge, diagnosed absent NVRTC, and used a user-installed Toolkit for local compile and dual launch/readback. It is not public-package, post-publish, Linux, or Owner authorization proof.'
}

[void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $OutputPath))
$report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding utf8
$markdownPath = [System.IO.Path]::ChangeExtension($OutputPath, '.md')
$markdown = @"
# CUDA RTC Bridge Package Clean Consumer

- Classification: ``$($report.proofClassification)``
- Managed package: ``$($managedPackage.Id) $($managedPackage.Version)``
- Bridge package: ``$($bridgePackage.Id) $($bridgePackage.Version)``
- Repository-external PackageReference consumer: ``$($report.consumer.rootOutsideRepository)``
- Local-only NuGet feed: ``$($report.consumer.localFeedOnly)``
- Package bundles NVRTC / builtins: ``$($report.packages.containsNvrtc)`` / ``$($report.packages.containsNvrtcBuiltins)``
- Missing-NVRTC diagnostic: ``$($report.dependencyDiagnostic.diagnosticPresent)``
- NVRTC / Driver version: ``$($report.runtime.rtcVersion)`` / ``$($report.runtime.driverVersion)``
- Runtime-library launch/readback/correctness: ``$($report.runtime.runtimeLibraryLaunch)`` / ``$($report.runtime.runtimeLibraryReadback)`` / ``$($report.runtime.runtimeLibraryCorrectness)``
- Driver launch/readback/correctness: ``$($report.runtime.driverLaunch)`` / ``$($report.runtime.driverReadback)`` / ``$($report.runtime.driverCorrectness)``
- Output hashes match: ``$($report.runtime.outputHashesMatch)``
- Public/post-publish promotion: ``$($report.canPromotePublicPackageProof)`` / ``$($report.canPromotePostPublishProof)``

$($report.proofBoundary)
"@
Write-Utf8Text -Path $markdownPath -Value $markdown

if (-not $KeepConsumerOutput.IsPresent) {
  Remove-DirectoryWithRetries -Path $workspaceRoot -AllowedRoot $OutputRoot
}

Write-Host "CUDA RTC bridge package clean consumer passed: $OutputPath"
