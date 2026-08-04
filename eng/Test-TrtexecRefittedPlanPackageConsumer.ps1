[CmdletBinding()]
param(
  [string]$SourceRuntimeKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$TargetFramework = "net8.0",
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$SourcePlanPath,
  [string]$SourceInputPath,
  [string]$SourceReferencePath,
  [string]$ExpectedPlanSha256 = "5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb",
  [string]$ExpectedInputSha256 = "81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564",
  [string]$ExpectedOutputSha256 = "6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041",
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$CompactOutputDirectory,
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
  [switch]$KeepConsumerOutput,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else {
  $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}

$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function ConvertTo-XmlAttributeValue {
  param([Parameter(Mandatory = $true)][string]$Value)
  return [Security.SecurityElement]::Escape($Value)
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
  param([Parameter(Mandatory = $true)][string]$Text)
  $hasher = [Security.Cryptography.SHA256]::Create()
  try {
    return [BitConverter]::ToString($hasher.ComputeHash($utf8.GetBytes($Text))).Replace('-', '').ToLowerInvariant()
  }
  finally { $hasher.Dispose() }
}

function Get-RelativePath {
  param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Path
  )
  $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  $rootUri = [Uri]$rootPath
  $pathUri = [Uri][IO.Path]::GetFullPath($Path)
  return [Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString()).Replace('\', '/')
}

function Test-PathWithin {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Parent
  )
  $resolvedPath = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  $resolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
  return $resolvedPath.StartsWith(
    $resolvedParent + [IO.Path]::DirectorySeparatorChar,
    [StringComparison]::OrdinalIgnoreCase)
}

function Remove-SafeConsumerDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$AllowedRoot
  )
  if (-not (Test-PathWithin -Path $Path -Parent $AllowedRoot)) {
    throw "Refusing to remove a consumer directory outside the allowed root: $Path"
  }
  $resolvedPath = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  if (Test-Path -LiteralPath $resolvedPath) {
    try {
      Remove-Item -LiteralPath $resolvedPath -Recurse -Force -ErrorAction Stop
    }
    catch {
      $isWindowsHost = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
      if (-not (Test-Path -LiteralPath $resolvedPath)) { return }
      if (-not $isWindowsHost) { throw }
      $extendedPath = if ($resolvedPath.StartsWith('\\')) {
        '\\?\UNC\' + $resolvedPath.TrimStart('\')
      }
      else { '\\?\' + $resolvedPath }
      [IO.Directory]::Delete($extendedPath, $true)
    }
    if (Test-Path -LiteralPath $resolvedPath) {
      throw "Consumer directory still exists after cleanup: $resolvedPath"
    }
  }
}

function Get-NupkgMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)
  $zip = [IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = $zip.Entries |
      Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) } |
      Select-Object -First 1
    if (-not $nuspec) { throw "Package does not contain a nuspec: $Path" }
    $stream = $nuspec.Open()
    try {
      $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
      try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
    }
    finally { $stream.Dispose() }

    $namespaceManager = [Xml.XmlNamespaceManager]::new($xml.NameTable)
    $namespaceManager.AddNamespace("n", $xml.package.NamespaceURI)
    return [pscustomobject]@{
      Path = [IO.Path]::GetFullPath($Path)
      Id = $xml.SelectSingleNode("//n:metadata/n:id", $namespaceManager).InnerText
      Version = $xml.SelectSingleNode("//n:metadata/n:version", $namespaceManager).InnerText
      Length = (Get-Item -LiteralPath $Path).Length
      LastWriteTime = (Get-Item -LiteralPath $Path).LastWriteTime
      Sha256 = Get-Sha256 -Path $Path
    }
  }
  finally { $zip.Dispose() }
}

function Find-Package {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$PackageId
  )
  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }
  $matches = foreach ($file in Get-ChildItem -LiteralPath $Directory -Filter *.nupkg -File) {
    $metadata = Get-NupkgMetadata -Path $file.FullName
    if ([string]::Equals($metadata.Id, $PackageId, [StringComparison]::OrdinalIgnoreCase)) { $metadata }
  }
  if (@($matches).Count -eq 0) { throw "Package '$PackageId' was not found under $Directory." }
  return @($matches | Sort-Object LastWriteTime, Version -Descending)[0]
}

function Resolve-BridgePackage {
  param([Parameter(Mandatory = $true)][string]$RuntimeKey)
  $manifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
  $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $package = $manifest.packages |
    Where-Object { $_.sourceRuntimeKey -eq $RuntimeKey -and $_.role -eq "bridge" } |
    Select-Object -First 1
  if (-not $package) { throw "Bridge split package for '$RuntimeKey' was not found." }
  if ([string]$package.tensorRtLine -ne "10") {
    throw "The refitted plan consumer currently requires a TensorRT 10 bridge package. Actual line=$($package.tensorRtLine)."
  }
  return $package
}

function Resolve-RuntimeRootSet {
  $resolved = & (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") `
    -RuntimePackageKey $SourceRuntimeKey -RepositoryRoot $RepositoryRoot | ConvertFrom-Json
  return [pscustomobject]@{
    TensorRtRoot = if ([string]::IsNullOrWhiteSpace($TensorRtRoot)) { [string]$resolved.tensorRtRoot } else { [IO.Path]::GetFullPath($TensorRtRoot) }
    CudaRoot = if ([string]::IsNullOrWhiteSpace($CudaRoot)) { [string]$resolved.cudaRoot } else { [IO.Path]::GetFullPath($CudaRoot) }
    CudnnRoot = if ([string]::IsNullOrWhiteSpace($CudnnRoot)) { [string]$resolved.cudnnRoot } else { [IO.Path]::GetFullPath($CudnnRoot) }
  }
}

function Get-RuntimeSearchDirectories {
  param([Parameter(Mandatory = $true)][object]$Roots)
  $candidates = @(
    (Join-Path $Roots.TensorRtRoot "bin"),
    (Join-Path $Roots.TensorRtRoot "lib"),
    (Join-Path $Roots.CudaRoot "bin\x64"),
    (Join-Path $Roots.CudaRoot "bin")
  )
  if (-not [string]::IsNullOrWhiteSpace($Roots.CudnnRoot)) { $candidates += (Join-Path $Roots.CudnnRoot "bin") }
  return @($candidates |
      Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
      ForEach-Object { (Resolve-Path -LiteralPath $_).Path } |
      Select-Object -Unique)
}

function Find-RuntimeFile {
  param(
    [Parameter(Mandatory = $true)][string[]]$Directories,
    [Parameter(Mandatory = $true)][string]$Pattern
  )
  foreach ($directory in $Directories) {
    $match = Get-ChildItem -LiteralPath $directory -Filter $Pattern -File -ErrorAction SilentlyContinue |
      Sort-Object Name | Select-Object -First 1
    if ($match) { return $match.FullName }
  }
  return ""
}

function New-NativeAssetRecord {
  param(
    [Parameter(Mandatory = $true)][string]$Role,
    [Parameter(Mandatory = $true)][string]$Path
  )
  $file = Get-Item -LiteralPath $Path
  return [pscustomobject][ordered]@{
    role = $Role
    name = $file.Name
    path = $file.FullName
    length = $file.Length
    sha256 = Get-Sha256 -Path $file.FullName
  }
}

function Get-NvidiaHostMetadata {
  $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $command) {
    return [pscustomobject]@{ available = $false; gpuName = ""; driverVersion = ""; diagnostic = "nvidia-smi was not found." }
  }
  $output = & $command.Source --query-gpu=name,driver_version --format=csv,noheader 2>&1
  if ($LASTEXITCODE -ne 0 -or @($output).Count -eq 0) {
    return [pscustomobject]@{ available = $false; gpuName = ""; driverVersion = ""; diagnostic = (@($output) -join " ") }
  }
  $parts = ([string]@($output)[0]).Split(',', 2)
  return [pscustomobject]@{
    available = $true
    gpuName = $parts[0].Trim()
    driverVersion = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
    diagnostic = ""
  }
}

function Get-MarkerValue {
  param(
    [string[]]$Lines,
    [Parameter(Mandatory = $true)][string]$Prefix
  )
  foreach ($line in @($Lines)) {
    if ([string]$line -like "$Prefix*") { return ([string]$line).Substring($Prefix.Length) }
  }
  return ""
}

function ConvertTo-Int32 {
  param([string]$Value)
  $number = 0
  [void][int]::TryParse($Value, [ref]$number)
  return $number
}

function ConvertTo-Int64 {
  param([string]$Value)
  $number = 0L
  [void][long]::TryParse($Value, [ref]$number)
  return $number
}

function Get-NamedCDriveArtifactMatches {
  $matches = [Collections.Generic.List[string]]::new()
  $roots = @(
    [IO.Path]::GetTempPath(),
    (Join-Path $env:USERPROFILE "Downloads"),
    (Join-Path $env:USERPROFILE "Documents"),
    (Join-Path $env:USERPROFILE "Desktop")
  )
  foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { continue }
    foreach ($pattern in @('*refitted-plan-package-consumer*', '*trtexec-refitted-plan-consumer*')) {
      try {
        foreach ($path in [IO.Directory]::EnumerateFileSystemEntries($root, $pattern, [IO.SearchOption]::TopDirectoryOnly)) {
          $matches.Add([IO.Path]::GetFullPath($path))
        }
      }
      catch [UnauthorizedAccessException] { continue }
      catch [IO.IOException] { continue }
    }
  }
  return @($matches.ToArray() | Sort-Object -Unique)
}

$bridgeDefinition = Resolve-BridgePackage -RuntimeKey $SourceRuntimeKey
$rid = [string]$bridgeDefinition.rid
$bridgeFileName = [string]@($bridgeDefinition.assets)[0]

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
else { $ManagedPackageDirectory = [IO.Path]::GetFullPath($ManagedPackageDirectory) }

if ([string]::IsNullOrWhiteSpace($BridgePackageDirectory)) {
  $BridgePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$SourceRuntimeKey"
}
else { $BridgePackageDirectory = [IO.Path]::GetFullPath($BridgePackageDirectory) }

if ([string]::IsNullOrWhiteSpace($SourcePlanPath)) {
  $SourcePlanPath = Join-Path $RepositoryRoot "artifacts\real-case\trtexec-refitted-plan-persistence\mnist-refitted-persisted.plan"
}
else { $SourcePlanPath = [IO.Path]::GetFullPath($SourcePlanPath) }

if ([string]::IsNullOrWhiteSpace($SourceInputPath)) {
  $SourceInputPath = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7-input-f32.bin"
}
else { $SourceInputPath = [IO.Path]::GetFullPath($SourceInputPath) }

if ([string]::IsNullOrWhiteSpace($SourceReferencePath)) {
  $SourceReferencePath = Join-Path $RepositoryRoot "artifacts\real-case\onnx-to-engine-mnist-trt10-runtime\digit-7\mnist-trt10-7.reference.json"
}
else { $SourceReferencePath = [IO.Path]::GetFullPath($SourceReferencePath) }

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path (Split-Path -Parent $RepositoryRoot) ".package-consumer-work"
}
else { $OutputRoot = [IO.Path]::GetFullPath($OutputRoot) }
$consumerRoot = Join-Path $OutputRoot "trtexec-refitted-plan-trt10"

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\package-consumer\trtexec-refitted-plan\$SourceRuntimeKey"
}
elseif (-not [IO.Path]::IsPathRooted($ReportDirectory)) {
  $ReportDirectory = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ReportDirectory))
}

if ([string]::IsNullOrWhiteSpace($CompactOutputDirectory)) {
  $CompactOutputDirectory = Join-Path $RepositoryRoot "artifacts\interface-coverage"
}
elseif (-not [IO.Path]::IsPathRooted($CompactOutputDirectory)) {
  $CompactOutputDirectory = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $CompactOutputDirectory))
}

foreach ($requiredFile in @($SourcePlanPath, $SourceInputPath, $SourceReferencePath)) {
  if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "Required artifact is missing: $requiredFile" }
}

$ExpectedPlanSha256 = $ExpectedPlanSha256.ToLowerInvariant()
$ExpectedInputSha256 = $ExpectedInputSha256.ToLowerInvariant()
$ExpectedOutputSha256 = $ExpectedOutputSha256.ToLowerInvariant()
$sourcePlanSha256 = Get-Sha256 -Path $SourcePlanPath
$sourceInputSha256 = Get-Sha256 -Path $SourceInputPath
$sourceReferenceSha256 = Get-Sha256 -Path $SourceReferencePath
$sourceReference = Get-Content -LiteralPath $SourceReferencePath -Raw -Encoding utf8 | ConvertFrom-Json
if ([int]$sourceReference.schemaVersion -ne 1 -or
    [string]::IsNullOrWhiteSpace([string]$sourceReference.tensorName) -or
    @($sourceReference.shape).Count -eq 0 -or
    @($sourceReference.shape | Where-Object { [int]$_ -le 0 }).Count -gt 0 -or
    @($sourceReference.values).Count -eq 0 -or
    [string]::IsNullOrWhiteSpace([string]$sourceReference.sourceClassification)) {
  throw "Structured MNIST reference is malformed: schemaVersion/name/positive shape/values/sourceClassification are required."
}
if ($sourcePlanSha256 -ne $ExpectedPlanSha256) {
  throw "Persisted plan SHA256 mismatch. Expected=$ExpectedPlanSha256 Actual=$sourcePlanSha256"
}
if ($sourceInputSha256 -ne $ExpectedInputSha256) {
  throw "Input SHA256 mismatch. Expected=$ExpectedInputSha256 Actual=$sourceInputSha256"
}

$consumerRootOutsideRepository = -not (Test-PathWithin -Path $consumerRoot -Parent $RepositoryRoot)
if (-not $consumerRootOutsideRepository) { throw "Consumer root must be outside the source repository: $consumerRoot" }
$workspaceOnSystemDrive = [IO.Path]::GetPathRoot($consumerRoot).TrimEnd('\') -eq $env:SystemDrive.TrimEnd('\')
if ($workspaceOnSystemDrive) { throw "The default proof workspace must not use the system drive: $consumerRoot" }

$managedPackage = Find-Package -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
$bridgePackage = Find-Package -Directory $BridgePackageDirectory -PackageId ([string]$bridgeDefinition.packageId)
$runtimeRoots = Resolve-RuntimeRootSet
foreach ($directory in @($runtimeRoots.TensorRtRoot, $runtimeRoots.CudaRoot)) {
  if (-not (Test-Path -LiteralPath $directory -PathType Container)) { throw "Runtime root is missing: $directory" }
}
$runtimeSearchDirectories = @(Get-RuntimeSearchDirectories -Roots $runtimeRoots)
if ($runtimeSearchDirectories.Count -lt 2) { throw "TensorRT and CUDA runtime search directories were not resolved." }

Remove-SafeConsumerDirectory -Path $consumerRoot -AllowedRoot $OutputRoot
New-Item -ItemType Directory -Path $consumerRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $CompactOutputDirectory -Force | Out-Null

$programSourcePath = Join-Path $RepositoryRoot "samples\RefittedPlan.PackageConsumer\Program.cs"
$projectTemplatePath = Join-Path $RepositoryRoot "samples\RefittedPlan.PackageConsumer\RefittedPlan.PackageConsumer.csproj.template"
$programPath = Join-Path $consumerRoot "Program.cs"
$projectPath = Join-Path $consumerRoot "RefittedPlan.PackageConsumer.csproj"
$nugetConfigPath = Join-Path $consumerRoot "NuGet.config"
$restorePackagesPath = Join-Path $consumerRoot ".nuget\packages"
$assetDirectory = Join-Path $consumerRoot "input"
$planCopyPath = Join-Path $assetDirectory "refitted.plan"
$inputCopyPath = Join-Path $assetDirectory "input-f32.bin"
$referenceCopyPath = Join-Path $assetDirectory "output-reference.json"
$outputCopyPath = Join-Path $consumerRoot "output\logits-f32.bin"
New-Item -ItemType Directory -Path $assetDirectory -Force | Out-Null
Copy-Item -LiteralPath $programSourcePath -Destination $programPath
Copy-Item -LiteralPath $SourcePlanPath -Destination $planCopyPath
Copy-Item -LiteralPath $SourceInputPath -Destination $inputCopyPath
Copy-Item -LiteralPath $SourceReferencePath -Destination $referenceCopyPath

$project = Get-Content -LiteralPath $projectTemplatePath -Raw -Encoding utf8
$project = $project.Replace("__TARGET_FRAMEWORK__", $TargetFramework)
$project = $project.Replace("__RUNTIME_IDENTIFIER__", $rid)
$project = $project.Replace("__RESTORE_PACKAGES_PATH__", (ConvertTo-XmlAttributeValue -Value $restorePackagesPath))
$project = $project.Replace("__MANAGED_PACKAGE_ID__", $managedPackage.Id)
$project = $project.Replace("__MANAGED_PACKAGE_VERSION__", $managedPackage.Version)
$project = $project.Replace("__BRIDGE_PACKAGE_ID__", $bridgePackage.Id)
$project = $project.Replace("__BRIDGE_PACKAGE_VERSION__", $bridgePackage.Version)

$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="jyppx-managed-local" value="$(ConvertTo-XmlAttributeValue -Value $ManagedPackageDirectory)" />
    <add key="jyppx-bridge-local" value="$(ConvertTo-XmlAttributeValue -Value $BridgePackageDirectory)" />
  </packageSources>
</configuration>
"@
[IO.File]::WriteAllText($projectPath, $project, $utf8)
[IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)

$programSource = Get-Content -LiteralPath $programPath -Raw -Encoding utf8
$manualManagedAssemblyLoad = $programSource -match 'Assembly\.(LoadFrom|LoadFile)|AssemblyLoadContext|JYPPX_NATIVE_BRIDGE_PATH\s*='
$usesProjectReference = $project -match '<ProjectReference'
if ($manualManagedAssemblyLoad -or $usesProjectReference) {
  throw "Consumer source violates the clean package contract. ManualLoad=$manualManagedAssemblyLoad ProjectReference=$usesProjectReference"
}

$restoreStdoutPath = Join-Path $ReportDirectory "restore.stdout.log"
$restoreStderrPath = Join-Path $ReportDirectory "restore.stderr.log"
$buildStdoutPath = Join-Path $ReportDirectory "build.stdout.log"
$buildStderrPath = Join-Path $ReportDirectory "build.stderr.log"
$runtimeStdoutPath = Join-Path $ReportDirectory "runtime.stdout.log"
$runtimeStderrPath = Join-Path $ReportDirectory "runtime.stderr.log"
$runtimeCombinedPath = Join-Path $ReportDirectory "runtime.combined.log"

$executionError = $null
$workspaceRemoved = $false
try {
  & dotnet restore $projectPath --configfile $nugetConfigPath --force --no-cache 1> $restoreStdoutPath 2> $restoreStderrPath
  $restoreExitCode = $LASTEXITCODE
  if ($restoreExitCode -ne 0) { throw "dotnet restore failed with exit code $restoreExitCode." }

  & dotnet build $projectPath -c Release --no-restore 1> $buildStdoutPath 2> $buildStderrPath
  $buildExitCode = $LASTEXITCODE
  if ($buildExitCode -ne 0) { throw "dotnet build failed with exit code $buildExitCode." }

  $assetsPath = Join-Path $consumerRoot "obj\project.assets.json"
  $assets = Get-Content -LiteralPath $assetsPath -Raw -Encoding utf8 | ConvertFrom-Json
  $libraryNames = @($assets.libraries.PSObject.Properties.Name)
  $managedLibraryKey = "$($managedPackage.Id)/$($managedPackage.Version)"
  $bridgeLibraryKey = "$($bridgePackage.Id)/$($bridgePackage.Version)"
  $managedResolved = $libraryNames -contains $managedLibraryKey
  $bridgeResolved = $libraryNames -contains $bridgeLibraryKey
  $packageFolders = @($assets.packageFolders.PSObject.Properties.Name | ForEach-Object { [IO.Path]::GetFullPath($_) })
  $isolatedRestorePathUsed = @($packageFolders | Where-Object {
      [string]::Equals($_.TrimEnd('\'), [IO.Path]::GetFullPath($restorePackagesPath).TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)
    }).Count -eq 1
  if (-not $managedResolved -or -not $bridgeResolved -or -not $isolatedRestorePathUsed) {
    throw "Restore did not resolve both target packages into the isolated consumer cache."
  }

  $outputDirectory = Join-Path $consumerRoot "bin\Release\$TargetFramework\$rid"
  $bridgeOutputPath = Join-Path $outputDirectory $bridgeFileName
  if (-not (Test-Path -LiteralPath $bridgeOutputPath -PathType Leaf)) {
    $bridgeOutputPath = Get-ChildItem -LiteralPath $outputDirectory -Recurse -Filter $bridgeFileName -File |
      Select-Object -First 1 -ExpandProperty FullName
  }
  if ([string]::IsNullOrWhiteSpace($bridgeOutputPath) -or -not (Test-Path -LiteralPath $bridgeOutputPath -PathType Leaf)) {
    throw "Bridge asset was not copied to the consumer output: $bridgeFileName"
  }

  $commandLine = "dotnet run --project `"$projectPath`" -c Release --no-build -- `"$planCopyPath`" `"$inputCopyPath`" `"$outputCopyPath`" $ExpectedOutputSha256 `"$referenceCopyPath`" 0.0001 0.0001 reject exact"
  $previousPath = $env:PATH
  $previousBridgePath = $env:JYPPX_NATIVE_BRIDGE_PATH
  $previousDevelopmentProbing = $env:JYPPX_ENABLE_DEVELOPMENT_PROBING
  try {
    $env:PATH = (@($runtimeSearchDirectories) + @($previousPath)) -join [IO.Path]::PathSeparator
    $env:JYPPX_NATIVE_BRIDGE_PATH = $null
    $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $null
    Push-Location $consumerRoot
    try {
      & dotnet run --project $projectPath -c Release --no-build -- `
        $planCopyPath $inputCopyPath $outputCopyPath $ExpectedOutputSha256 $referenceCopyPath 0.0001 0.0001 reject exact `
        1> $runtimeStdoutPath 2> $runtimeStderrPath
      $runtimeExitCode = $LASTEXITCODE
    }
    finally { Pop-Location }
  }
  finally {
    $env:PATH = $previousPath
    $env:JYPPX_NATIVE_BRIDGE_PATH = $previousBridgePath
    $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $previousDevelopmentProbing
  }

  $stdoutLines = if (Test-Path -LiteralPath $runtimeStdoutPath) { @(Get-Content -LiteralPath $runtimeStdoutPath -Encoding utf8) } else { @() }
  $stderrLines = if (Test-Path -LiteralPath $runtimeStderrPath) { @(Get-Content -LiteralPath $runtimeStderrPath -Encoding utf8) } else { @() }
  [IO.File]::WriteAllLines($runtimeCombinedPath, @($stdoutLines + $stderrLines), $utf8)
  foreach ($line in $stdoutLines) { Write-Host $line }
  foreach ($line in $stderrLines) { Write-Warning $line }

  $outputSha256 = if (Test-Path -LiteralPath $outputCopyPath -PathType Leaf) { Get-Sha256 -Path $outputCopyPath } else { "" }
  $runtimePassed = $runtimeExitCode -eq 0 -and
    (Get-MarkerValue -Lines $stdoutLines -Prefix "PackageConsumerRuntime=") -eq "Passed" -and
    (Get-MarkerValue -Lines $stdoutLines -Prefix "OutputExactMatch=") -eq "True" -and
    (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceValidationCompleted=") -eq "True" -and
    (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceValidationPassed=") -eq "True" -and
    (Get-MarkerValue -Lines $stdoutLines -Prefix "OwnerScopeExited=") -eq "True" -and
    $outputSha256 -eq $ExpectedOutputSha256

  $managedAssemblyLocation = Get-MarkerValue -Lines $stdoutLines -Prefix "CoreAssemblyLocation="
  $managedAssemblyOutsideSourceTree = -not (Test-PathWithin -Path $managedAssemblyLocation -Parent $RepositoryRoot)
  $managedAssemblyUnderConsumerRoot = Test-PathWithin -Path $managedAssemblyLocation -Parent $consumerRoot
  $bridgeUnderConsumerRoot = Test-PathWithin -Path $bridgeOutputPath -Parent $consumerRoot
  if (-not $runtimePassed -or -not $managedAssemblyOutsideSourceTree -or -not $managedAssemblyUnderConsumerRoot -or -not $bridgeUnderConsumerRoot) {
    throw "Package consumer runtime contract failed. Runtime=$runtimePassed ManagedOutsideSource=$managedAssemblyOutsideSourceTree ManagedUnderConsumer=$managedAssemblyUnderConsumerRoot BridgeUnderConsumer=$bridgeUnderConsumerRoot"
  }

  $nvinferPath = Find-RuntimeFile -Directories $runtimeSearchDirectories -Pattern "nvinfer_10.dll"
  $cudartPath = Find-RuntimeFile -Directories $runtimeSearchDirectories -Pattern "cudart64_12.dll"
  if ([string]::IsNullOrWhiteSpace($nvinferPath) -or [string]::IsNullOrWhiteSpace($cudartPath)) {
    throw "Required TensorRT/CUDA runtime inventory files were not found."
  }
  $nativeAssets = @(
    (New-NativeAssetRecord -Role "bridge-package-output" -Path $bridgeOutputPath),
    (New-NativeAssetRecord -Role "tensor-rt-runtime" -Path $nvinferPath),
    (New-NativeAssetRecord -Role "cuda-runtime" -Path $cudartPath)
  )
  $nvidia = Get-NvidiaHostMetadata

  $rawEvidence = [ordered]@{
    schemaVersion = "trtexec-refitted-plan-package-consumer-proof.v1"
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    state = "local-package-consumer-refitted-plan-runtime-passed"
    evidenceClassification = "local-package-consumer-refitted-plan-runtime"
    sourceRuntimeKey = $SourceRuntimeKey
    packageSources = [ordered]@{
      managed = $ManagedPackageDirectory
      bridge = $BridgePackageDirectory
      declaredSourceCount = 2
      nugetOrgEnabled = $false
    }
    packages = [ordered]@{
      managed = [ordered]@{ id = $managedPackage.Id; version = $managedPackage.Version; path = $managedPackage.Path; length = $managedPackage.Length; sha256 = $managedPackage.Sha256 }
      bridge = [ordered]@{ id = $bridgePackage.Id; version = $bridgePackage.Version; path = $bridgePackage.Path; length = $bridgePackage.Length; sha256 = $bridgePackage.Sha256 }
    }
    consumer = [ordered]@{
      root = $consumerRoot
      rootOutsideRepository = $consumerRootOutsideRepository
      workspaceOnSystemDrive = $workspaceOnSystemDrive
      projectPath = $projectPath
      projectSha256 = Get-Sha256 -Path $projectPath
      programSha256 = Get-Sha256 -Path $programPath
      usesPackageReferenceOnly = -not $usesProjectReference
      usesProjectReference = $usesProjectReference
      manualManagedAssemblyLoad = $manualManagedAssemblyLoad
      isolatedRestorePath = $restorePackagesPath
      isolatedRestorePathUsed = $isolatedRestorePathUsed
      managedPackageResolved = $managedResolved
      bridgePackageResolved = $bridgeResolved
      managedAssemblyLocation = $managedAssemblyLocation
      managedAssemblyOutsideSourceTree = $managedAssemblyOutsideSourceTree
      managedAssemblyUnderConsumerRoot = $managedAssemblyUnderConsumerRoot
      bridgeOutputPath = $bridgeOutputPath
      bridgeUnderConsumerRoot = $bridgeUnderConsumerRoot
    }
    artifacts = [ordered]@{
      sourcePlanPath = $SourcePlanPath
      copiedPlanPath = $planCopyPath
      planCopyPathDistinct = -not [string]::Equals($SourcePlanPath, $planCopyPath, [StringComparison]::OrdinalIgnoreCase)
      planLengthBytes = (Get-Item -LiteralPath $planCopyPath).Length
      sourcePlanSha256 = $sourcePlanSha256
      copiedPlanSha256 = Get-Sha256 -Path $planCopyPath
      sourceInputPath = $SourceInputPath
      copiedInputPath = $inputCopyPath
      inputCopyPathDistinct = -not [string]::Equals($SourceInputPath, $inputCopyPath, [StringComparison]::OrdinalIgnoreCase)
      inputLengthBytes = (Get-Item -LiteralPath $inputCopyPath).Length
      sourceInputSha256 = $sourceInputSha256
      copiedInputSha256 = Get-Sha256 -Path $inputCopyPath
      sourceReferencePath = $SourceReferencePath
      copiedReferencePath = $referenceCopyPath
      referenceCopyPathDistinct = -not [string]::Equals($SourceReferencePath, $referenceCopyPath, [StringComparison]::OrdinalIgnoreCase)
      sourceReferenceSha256 = $sourceReferenceSha256
      copiedReferenceSha256 = Get-Sha256 -Path $referenceCopyPath
      referenceTensorName = [string]$sourceReference.tensorName
      referenceShape = @($sourceReference.shape | ForEach-Object { [int]$_ })
      referenceElementCount = @($sourceReference.values).Count
      referenceSourceClassification = [string]$sourceReference.sourceClassification
      referenceAbsoluteTolerance = 0.0001
      referenceRelativeTolerance = 0.0001
      referenceNaNPolicy = "reject"
      referenceInfinityPolicy = "exact"
      outputPath = $outputCopyPath
      outputLengthBytes = (Get-Item -LiteralPath $outputCopyPath).Length
      outputSha256 = $outputSha256
      expectedOutputSha256 = $ExpectedOutputSha256
      outputExactMatch = $outputSha256 -eq $ExpectedOutputSha256
    }
    execution = [ordered]@{
      commandLine = $commandLine
      commandSha256 = Get-TextSha256 -Text $commandLine
      restoreExitCode = $restoreExitCode
      buildExitCode = $buildExitCode
      runtimeExitCode = $runtimeExitCode
      runtimePassed = $runtimePassed
      stdoutPath = $runtimeStdoutPath
      stdoutSha256 = Get-Sha256 -Path $runtimeStdoutPath
      stderrPath = $runtimeStderrPath
      stderrSha256 = Get-Sha256 -Path $runtimeStderrPath
      combinedPath = $runtimeCombinedPath
      combinedSha256 = Get-Sha256 -Path $runtimeCombinedPath
      stdout = @($stdoutLines)
      stderr = @($stderrLines)
    }
    runtime = [ordered]@{
      environment = Get-MarkerValue -Lines $stdoutLines -Prefix "RuntimeEnvironment "
      engineRefittable = (Get-MarkerValue -Lines $stdoutLines -Prefix "EngineRefittable=") -eq "True"
      engineIOTensorCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "EngineIOTensorCount=")
      engineLayerCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "EngineLayerCount=")
      engineOptimizationProfileCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "EngineOptimizationProfileCount=")
      inputTensor = Get-MarkerValue -Lines $stdoutLines -Prefix "InputTensor="
      inputShape = Get-MarkerValue -Lines $stdoutLines -Prefix "InputShape="
      inputElementCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "InputElementCount=")
      outputTensor = Get-MarkerValue -Lines $stdoutLines -Prefix "OutputTensor="
      outputShape = Get-MarkerValue -Lines $stdoutLines -Prefix "OutputShape="
      outputElementCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "OutputElementCount=")
      bindingsReadyForEnqueue = (Get-MarkerValue -Lines $stdoutLines -Prefix "BindingsReadyForEnqueue=") -eq "True"
      enqueueCompleted = (Get-MarkerValue -Lines $stdoutLines -Prefix "EnqueueCompleted=") -eq "True"
      ownerScopeExited = (Get-MarkerValue -Lines $stdoutLines -Prefix "OwnerScopeExited=") -eq "True"
      outputExactMatch = (Get-MarkerValue -Lines $stdoutLines -Prefix "OutputExactMatch=") -eq "True"
      referenceValidationCompleted = (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceValidationCompleted=") -eq "True"
      referenceValidationPassed = (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceValidationPassed=") -eq "True"
      referenceComparedElementCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceComparedElementCount=")
      referenceMismatchCount = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceMismatchCount=")
      referenceFirstMismatchIndex = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceFirstMismatchIndex=")
      referenceMaximumAbsoluteError = Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceMaximumAbsoluteError="
      referenceMaximumRelativeError = Get-MarkerValue -Lines $stdoutLines -Prefix "ReferenceMaximumRelativeError="
      predictedIndex = ConvertTo-Int32 (Get-MarkerValue -Lines $stdoutLines -Prefix "PredictedIndex=")
    }
    host = [ordered]@{
      osDescription = [Runtime.InteropServices.RuntimeInformation]::OSDescription.Trim()
      processArchitecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
      gpuName = $nvidia.gpuName
      driverVersion = $nvidia.driverVersion
      nvidiaSmiAvailable = $nvidia.available
      nvidiaSmiDiagnostic = $nvidia.diagnostic
    }
    nativeAssets = @($nativeAssets)
    proofBoundary = [ordered]@{
      isLocalPackageConsumerRuntimeProof = $true
      isPackageConsumerRuntimeProof = $false
      packagesDownloadedFromPublicFeed = $false
      isPostPublishProof = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      performsPublish = $false
      statement = "This proves a copied full-weight refitted plan can be restored and executed by a PackageReference-only consumer from declared local feeds on one compatible host, with the recorded structured reference comparison. The reference is an unreviewed same-runtime MNIST regression baseline, so this is not independent model-accuracy, Owner-approved golden output, public-feed, post-publish, or release-close proof."
    }
  }
}
catch {
  $executionError = $_
}
finally {
  if (-not $KeepConsumerOutput.IsPresent) {
    Remove-SafeConsumerDirectory -Path $consumerRoot -AllowedRoot $OutputRoot
    $workspaceRemoved = -not (Test-Path -LiteralPath $consumerRoot)
    if ($workspaceRemoved -and (Test-Path -LiteralPath $OutputRoot -PathType Container) -and
        @(Get-ChildItem -LiteralPath $OutputRoot -Force).Count -eq 0) {
      Remove-Item -LiteralPath $OutputRoot -Force
    }
  }
}

if ($executionError) { throw $executionError }
if ($null -eq $rawEvidence) { throw "Package consumer evidence was not produced." }
$rawEvidence.consumer["workspaceRemovedAfterValidation"] = $workspaceRemoved
$rawEvidence.consumer["outputPreserved"] = $KeepConsumerOutput.IsPresent
$cDriveMatches = @(Get-NamedCDriveArtifactMatches)
$rawEvidence["cDriveAudit"] = [ordered]@{
  consumerWorkspaceUsedSystemDrive = $workspaceOnSystemDrive
  namedConsumerArtifactMatchCount = $cDriveMatches.Count
  namedConsumerArtifactMatches = @($cDriveMatches)
  unrelatedUserOrSystemCacheTouched = $false
}

$rawJsonPath = Join-Path $ReportDirectory "refitted-plan-package-consumer-proof.json"
$rawMarkdownPath = Join-Path $ReportDirectory "refitted-plan-package-consumer-proof.md"
$jsonDepth = if ($PSVersionTable.PSVersion.Major -lt 6) { 6 } else { 15 }
$rawJson = $rawEvidence | ConvertTo-Json -Depth $jsonDepth
[IO.File]::WriteAllText($rawJsonPath, $rawJson + [Environment]::NewLine, $utf8)
$rawLines = @(
  "# Refitted Plan Package Consumer Proof",
  "",
  "- state: ``$($rawEvidence.state)``",
  "- classification: ``$($rawEvidence.evidenceClassification)``",
  "- restore/build/runtime: ``$($rawEvidence.execution.restoreExitCode)/$($rawEvidence.execution.buildExitCode)/$($rawEvidence.execution.runtimeExitCode)``",
  "- plan/output SHA256: ``$($rawEvidence.artifacts.copiedPlanSha256)`` / ``$($rawEvidence.artifacts.outputSha256)``",
  "- workspace removed: ``$workspaceRemoved``",
  "- public package proof: ``False``",
  "",
  "Generated by ``eng/Test-TrtexecRefittedPlanPackageConsumer.ps1``."
)
$rawLines | Set-Content -LiteralPath $rawMarkdownPath -Encoding utf8

$compactNativeAssets = foreach ($asset in $nativeAssets) {
  [pscustomobject][ordered]@{ role = $asset.role; name = $asset.name; length = $asset.length; sha256 = $asset.sha256 }
}
$compactEvidence = [ordered]@{
  schemaVersion = "trtexec-refitted-plan-package-consumer-evidence.v1"
  generatedDate = [DateTime]::Now.ToString("yyyy-MM-dd")
  state = $rawEvidence.state
  evidenceClassification = $rawEvidence.evidenceClassification
  sourceRuntimeKey = $SourceRuntimeKey
  packageContract = [ordered]@{
    declaredLocalSourceCount = 2
    nugetOrgEnabled = $false
    usesPackageReferenceOnly = $rawEvidence.consumer.usesPackageReferenceOnly
    usesProjectReference = $rawEvidence.consumer.usesProjectReference
    manualManagedAssemblyLoad = $rawEvidence.consumer.manualManagedAssemblyLoad
    isolatedRestorePathUsed = $rawEvidence.consumer.isolatedRestorePathUsed
    managedPackageResolved = $rawEvidence.consumer.managedPackageResolved
    bridgePackageResolved = $rawEvidence.consumer.bridgePackageResolved
  }
  packages = [ordered]@{
    managed = [ordered]@{ id = $managedPackage.Id; version = $managedPackage.Version; length = $managedPackage.Length; sha256 = $managedPackage.Sha256 }
    bridge = [ordered]@{ id = $bridgePackage.Id; version = $bridgePackage.Version; length = $bridgePackage.Length; sha256 = $bridgePackage.Sha256 }
  }
  consumer = [ordered]@{
    rootOutsideRepository = $consumerRootOutsideRepository
    workspaceOnSystemDrive = $workspaceOnSystemDrive
    workspaceRemovedAfterValidation = $workspaceRemoved
    outputPreserved = $KeepConsumerOutput.IsPresent
    managedAssemblyOutsideSourceTree = $rawEvidence.consumer.managedAssemblyOutsideSourceTree
    managedAssemblyUnderConsumerRoot = $rawEvidence.consumer.managedAssemblyUnderConsumerRoot
    bridgeUnderConsumerRoot = $rawEvidence.consumer.bridgeUnderConsumerRoot
    projectSha256 = $rawEvidence.consumer.projectSha256
    programSha256 = $rawEvidence.consumer.programSha256
  }
  artifacts = [ordered]@{
    sourcePlan = Get-RelativePath -Root $RepositoryRoot -Path $SourcePlanPath
    planCopyPathDistinct = $rawEvidence.artifacts.planCopyPathDistinct
    planLengthBytes = $rawEvidence.artifacts.planLengthBytes
    sourcePlanSha256 = $sourcePlanSha256
    copiedPlanSha256 = $rawEvidence.artifacts.copiedPlanSha256
    sourceInput = Get-RelativePath -Root $RepositoryRoot -Path $SourceInputPath
    inputCopyPathDistinct = $rawEvidence.artifacts.inputCopyPathDistinct
    inputLengthBytes = $rawEvidence.artifacts.inputLengthBytes
    sourceInputSha256 = $sourceInputSha256
    copiedInputSha256 = $rawEvidence.artifacts.copiedInputSha256
    sourceReference = Get-RelativePath -Root $RepositoryRoot -Path $SourceReferencePath
    referenceCopyPathDistinct = $rawEvidence.artifacts.referenceCopyPathDistinct
    sourceReferenceSha256 = $sourceReferenceSha256
    copiedReferenceSha256 = $rawEvidence.artifacts.copiedReferenceSha256
    referenceTensorName = $rawEvidence.artifacts.referenceTensorName
    referenceShape = @($rawEvidence.artifacts.referenceShape)
    referenceElementCount = $rawEvidence.artifacts.referenceElementCount
    referenceSourceClassification = $rawEvidence.artifacts.referenceSourceClassification
    referenceAbsoluteTolerance = $rawEvidence.artifacts.referenceAbsoluteTolerance
    referenceRelativeTolerance = $rawEvidence.artifacts.referenceRelativeTolerance
    referenceNaNPolicy = $rawEvidence.artifacts.referenceNaNPolicy
    referenceInfinityPolicy = $rawEvidence.artifacts.referenceInfinityPolicy
    outputLengthBytes = $rawEvidence.artifacts.outputLengthBytes
    outputSha256 = $rawEvidence.artifacts.outputSha256
    expectedOutputSha256 = $ExpectedOutputSha256
    outputExactMatch = $rawEvidence.artifacts.outputExactMatch
  }
  execution = [ordered]@{
    commandShape = "dotnet run --project <consumer-project> -c Release --no-build -- <copied-plan> <copied-input> <raw-output> <expected-output-sha256> <copied-reference-json> <abs-tolerance> <rel-tolerance> <nan-policy> <infinity-policy>"
    restoreExitCode = $rawEvidence.execution.restoreExitCode
    buildExitCode = $rawEvidence.execution.buildExitCode
    runtimeExitCode = $rawEvidence.execution.runtimeExitCode
    runtimePassed = $rawEvidence.execution.runtimePassed
    stdoutSha256 = $rawEvidence.execution.stdoutSha256
    stderrSha256 = $rawEvidence.execution.stderrSha256
    combinedSha256 = $rawEvidence.execution.combinedSha256
  }
  runtime = $rawEvidence.runtime
  host = $rawEvidence.host
  nativeAssets = @($compactNativeAssets)
  cDriveAudit = [ordered]@{
    consumerWorkspaceUsedSystemDrive = $workspaceOnSystemDrive
    namedConsumerArtifactMatchCount = $cDriveMatches.Count
    unrelatedUserOrSystemCacheTouched = $false
  }
  proofBoundary = $rawEvidence.proofBoundary
}

$compactJson = $compactEvidence | ConvertTo-Json -Depth $jsonDepth
if ($compactJson -match '(?i)[A-Z]:\\') {
  throw "Compact package-consumer evidence contains an absolute Windows path."
}
$compactJsonPath = Join-Path $CompactOutputDirectory "trtexec-refitted-plan-package-consumer-evidence.json"
$compactMarkdownPath = Join-Path $CompactOutputDirectory "trtexec-refitted-plan-package-consumer-evidence.md"
[IO.File]::WriteAllText($compactJsonPath, $compactJson + [Environment]::NewLine, $utf8)
$compactLines = @(
  "# TensorRtExec Refitted Plan Local Package Consumer Evidence",
  "",
  "- state: ``$($compactEvidence.state)``",
  "- classification: ``$($compactEvidence.evidenceClassification)``",
  "- package sources: ``2 local / nuget.org disabled``",
  "- ProjectReference / manual assembly load: ``False / False``",
  "- restore/build/runtime: ``0/0/0``",
  "- plan SHA256: ``$($compactEvidence.artifacts.copiedPlanSha256)``",
  "- output/reference SHA256: ``$($compactEvidence.artifacts.outputSha256)`` / ``$($compactEvidence.artifacts.copiedReferenceSha256)``",
  "- enqueue / reference / owner scope / workspace cleanup: ``True / $($compactEvidence.runtime.referenceValidationPassed) / True / $workspaceRemoved``",
  "- public package proof / publish / release close: ``False / False / False``",
  "",
  $compactEvidence.proofBoundary.statement
)
[IO.File]::WriteAllLines($compactMarkdownPath, $compactLines, $utf8)

Write-Host "Refitted plan package consumer proof: RuntimePassed=$($rawEvidence.execution.runtimePassed) OutputExactMatch=$($rawEvidence.artifacts.outputExactMatch) WorkspaceRemoved=$workspaceRemoved"
Write-Host "RawEvidence=$rawJsonPath"
Write-Host "CompactEvidence=$compactJsonPath"

if ($Strict.IsPresent) {
  & (Join-Path $RepositoryRoot "eng\Test-TrtexecRefittedPlanPackageConsumerEvidence.ps1") -Strict -RepositoryRoot $RepositoryRoot
  if ($LASTEXITCODE -ne 0) { throw "Strict package-consumer evidence validation failed with exit code $LASTEXITCODE." }
}
