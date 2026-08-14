[CmdletBinding()]
param(
  [string]$SourceRuntimeKey = "linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22",
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string[]]$AdditionalPackageSource = @(),
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [switch]$KeepConsumerOutput,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function ConvertTo-XmlText {
  param([AllowEmptyString()][string]$Value)
  return [System.Security.SecurityElement]::Escape($Value)
}

function Get-NupkgMetadata {
  param([Parameter(Mandatory = $true)][string]$Path)

  $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = $zip.Entries |
      Where-Object { $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase) } |
      Select-Object -First 1
    if (-not $nuspec) {
      throw "Package does not contain a nuspec: $Path"
    }

    $stream = $nuspec.Open()
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

    $namespaceManager = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
    $namespaceManager.AddNamespace("n", $xml.package.NamespaceURI)
    $idNode = $xml.SelectSingleNode("//n:metadata/n:id", $namespaceManager)
    $versionNode = $xml.SelectSingleNode("//n:metadata/n:version", $namespaceManager)
    return [pscustomobject]@{
      id = [string]$idNode.InnerText
      version = [string]$versionNode.InnerText
      path = [System.IO.Path]::GetFullPath($Path)
      sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
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
    [string]$RequiredVersion
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }

  $matches = @(
    Get-ChildItem -LiteralPath $Directory -Filter "*.nupkg" -File -Recurse |
      Where-Object { -not $_.Name.EndsWith(".symbols.nupkg", [System.StringComparison]::OrdinalIgnoreCase) } |
      ForEach-Object {
        $metadata = Get-NupkgMetadata -Path $_.FullName
        if ([string]::Equals($metadata.id, $PackageId, [System.StringComparison]::OrdinalIgnoreCase) -and
            ([string]::IsNullOrWhiteSpace($RequiredVersion) -or
             [string]::Equals($metadata.version, $RequiredVersion, [System.StringComparison]::OrdinalIgnoreCase))) {
          [pscustomobject]@{
            metadata = $metadata
            lastWriteTimeUtc = $_.LastWriteTimeUtc
          }
        }
      }
  )

  if ($matches.Count -eq 0) {
    $versionSuffix = if ([string]::IsNullOrWhiteSpace($RequiredVersion)) { "" } else { " at version '$RequiredVersion'" }
    throw "Package '$PackageId'$versionSuffix was not found under '$Directory'."
  }

  return ($matches | Sort-Object lastWriteTimeUtc -Descending | Select-Object -First 1).metadata
}

function Invoke-CapturedCommand {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$StdoutPath,
    [Parameter(Mandatory = $true)][string]$StderrPath,
    [string]$WorkingDirectory
  )

  $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
  $startInfo.FileName = $FilePath
  $startInfo.UseShellExecute = $false
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    $startInfo.WorkingDirectory = $WorkingDirectory
  }
  foreach ($argument in $Arguments) {
    $startInfo.ArgumentList.Add($argument)
  }

  $process = [System.Diagnostics.Process]::new()
  $process.StartInfo = $startInfo
  if (-not $process.Start()) {
    throw "Unable to start command: $FilePath"
  }

  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  $process.WaitForExit()
  $stdout = $stdoutTask.GetAwaiter().GetResult()
  $stderr = $stderrTask.GetAwaiter().GetResult()
  $exitCode = $process.ExitCode
  $process.Dispose()

  [System.IO.File]::WriteAllText($StdoutPath, $stdout, $utf8)
  [System.IO.File]::WriteAllText($StderrPath, $stderr, $utf8)
  return [pscustomobject]@{
    exitCode = $exitCode
    stdout = $stdout
    stderr = $stderr
    command = $FilePath + " " + ($Arguments -join " ")
  }
}

function Get-MarkerValue {
  param(
    [AllowEmptyString()][string]$Text,
    [Parameter(Mandatory = $true)][string]$Prefix
  )

  $line = @($Text -split "`r?`n" | Where-Object { $_.StartsWith($Prefix, [System.StringComparison]::Ordinal) } | Select-Object -First 1)
  if ($line.Count -eq 0) {
    return ""
  }
  return $line[0].Substring($Prefix.Length).Trim()
}

function Get-LogEvidence {
  param([Parameter(Mandatory = $true)][string]$Path)

  return [ordered]@{
    path = [System.IO.Path]::GetFullPath($Path)
    lengthBytes = (Get-Item -LiteralPath $Path).Length
    sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

$hostIsLinux = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
  [System.Runtime.InteropServices.OSPlatform]::Linux)
$hostIsX64 = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq
  [System.Runtime.InteropServices.Architecture]::X64
if (-not $hostIsLinux -or -not $hostIsX64) {
  throw "Minimal Linux Bridge runtime consumer requires a Linux x64 host."
}

$manifestPath = Join-Path $RepositoryRoot "pack/runtime/runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimePackage = $manifest.packages | Where-Object { $_.key -eq $SourceRuntimeKey } | Select-Object -First 1
if (-not $runtimePackage) {
  throw "Runtime package key '$SourceRuntimeKey' was not found."
}
if ([string]$runtimePackage.platform -ne "linux" -or [string]$runtimePackage.architecture -ne "x64") {
  throw "Runtime package key '$SourceRuntimeKey' is not a Linux x64 package."
}

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts/managed"
}
if ([string]::IsNullOrWhiteSpace($BridgePackageDirectory)) {
  $BridgePackageDirectory = Join-Path $RepositoryRoot "artifacts/runtime-split-nupkg/$SourceRuntimeKey"
}
if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts/minimal-linux-bridge-runtime/$SourceRuntimeKey"
}

$bridgePackageId = "$([string]$runtimePackage.packageId).Bridge"
$bridgePackage = Find-Nupkg -Directory $BridgePackageDirectory -PackageId $bridgePackageId
$managedPackage = Find-Nupkg `
  -Directory $ManagedPackageDirectory `
  -PackageId "JYPPX.TensorRT.CSharp.API" `
  -RequiredVersion $bridgePackage.version

$ownedConsumerRoot = [string]::IsNullOrWhiteSpace($OutputRoot)
$taskTempRoot = [System.IO.Path]::GetFullPath((Join-Path ([System.IO.Path]::GetTempPath()) "jyppx-minimal-linux-bridge-runtime"))
if ($ownedConsumerRoot) {
  $safeKey = $SourceRuntimeKey -replace '[^A-Za-z0-9._-]', '-'
  $OutputRoot = Join-Path $taskTempRoot "$safeKey/$([Guid]::NewGuid().ToString('N'))"
}
$consumerRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$repositoryRootFull = [System.IO.Path]::GetFullPath($RepositoryRoot)
if ($consumerRoot.StartsWith($repositoryRootFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Runtime proof consumer root must be outside the source repository: $consumerRoot"
}

New-Item -ItemType Directory -Path $consumerRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
$packagesRoot = Join-Path $consumerRoot ".nuget/packages"
$projectPath = Join-Path $consumerRoot "MinimalLinuxBridgePackageRuntimeConsumer.csproj"
$programPath = Join-Path $consumerRoot "Program.cs"
$nugetConfigPath = Join-Path $consumerRoot "NuGet.Config"

$project = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <RuntimeIdentifier>$(ConvertTo-XmlText -Value ([string]$runtimePackage.rid))</RuntimeIdentifier>
    <RestorePackagesPath>$(ConvertTo-XmlText -Value $packagesRoot)</RestorePackagesPath>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="$(ConvertTo-XmlText -Value $managedPackage.id)" Version="$(ConvertTo-XmlText -Value $managedPackage.version)" />
    <PackageReference Include="$(ConvertTo-XmlText -Value $bridgePackage.id)" Version="$(ConvertTo-XmlText -Value $bridgePackage.version)" />
  </ItemGroup>
</Project>
"@
[System.IO.File]::WriteAllText($projectPath, $project, $utf8)
Copy-Item -LiteralPath (Join-Path $RepositoryRoot "eng/templates/MinimalLinuxBridgePackageRuntimeConsumer/Program.cs") -Destination $programPath -Force

$packageSources = [System.Collections.Generic.List[string]]::new()
$packageSources.Add([System.IO.Path]::GetFullPath($ManagedPackageDirectory))
$packageSources.Add([System.IO.Path]::GetFullPath($BridgePackageDirectory))
foreach ($source in @($AdditionalPackageSource)) {
  if (-not [string]::IsNullOrWhiteSpace($source)) {
    $packageSources.Add($source.Trim())
  }
}
$packageSources.Add("https://api.nuget.org/v3/index.json")
$sourceLines = for ($index = 0; $index -lt $packageSources.Count; $index++) {
  "    <add key=`"source-$index`" value=`"$(ConvertTo-XmlText -Value $packageSources[$index])`" />"
}
$nugetConfig = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
$($sourceLines -join "`n")
  </packageSources>
</configuration>
"@
[System.IO.File]::WriteAllText($nugetConfigPath, $nugetConfig, $utf8)

$restoreStdout = Join-Path $ReportDirectory "restore.stdout.log"
$restoreStderr = Join-Path $ReportDirectory "restore.stderr.log"
$buildStdout = Join-Path $ReportDirectory "build.stdout.log"
$buildStderr = Join-Path $ReportDirectory "build.stderr.log"
$runtimeStdout = Join-Path $ReportDirectory "runtime.stdout.log"
$runtimeStderr = Join-Path $ReportDirectory "runtime.stderr.log"

$restore = Invoke-CapturedCommand -FilePath "dotnet" -Arguments @(
  "restore", $projectPath, "--configfile", $nugetConfigPath, "--packages", $packagesRoot
) -StdoutPath $restoreStdout -StderrPath $restoreStderr -WorkingDirectory $consumerRoot

$build = if ($restore.exitCode -eq 0) {
  Invoke-CapturedCommand -FilePath "dotnet" -Arguments @(
    "build", $projectPath, "-c", "Release", "--no-restore"
  ) -StdoutPath $buildStdout -StderrPath $buildStderr -WorkingDirectory $consumerRoot
}
else {
  [System.IO.File]::WriteAllText($buildStdout, "", $utf8)
  [System.IO.File]::WriteAllText($buildStderr, "build skipped because restore failed.", $utf8)
  [pscustomobject]@{ exitCode = -1; stdout = ""; stderr = "build skipped because restore failed."; command = "not-run" }
}

$runtime = if ($build.exitCode -eq 0) {
  Invoke-CapturedCommand -FilePath "dotnet" -Arguments @(
    "run", "--project", $projectPath, "-c", "Release", "--no-build", "--", "--tensor-rt-line", [string]$runtimePackage.tensorRtLine
  ) -StdoutPath $runtimeStdout -StderrPath $runtimeStderr -WorkingDirectory $consumerRoot
}
else {
  [System.IO.File]::WriteAllText($runtimeStdout, "", $utf8)
  [System.IO.File]::WriteAllText($runtimeStderr, "runtime skipped because build failed.", $utf8)
  [pscustomobject]@{ exitCode = -1; stdout = ""; stderr = "runtime skipped because build failed."; command = "not-run" }
}

$kernelRelease = if (Test-Path -LiteralPath "/proc/sys/kernel/osrelease" -PathType Leaf) {
  (Get-Content -LiteralPath "/proc/sys/kernel/osrelease" -Raw).Trim()
}
else {
  ""
}
$containerControlGroup = if (Test-Path -LiteralPath "/proc/1/cgroup" -PathType Leaf) {
  Get-Content -LiteralPath "/proc/1/cgroup" -Raw
}
else {
  ""
}
$hostIsContainer =
  (Test-Path -LiteralPath "/.dockerenv" -PathType Leaf) -or
  (Test-Path -LiteralPath "/run/.containerenv" -PathType Leaf) -or
  $containerControlGroup -match "(?i)docker|containerd|kubepods|libpod"
$wslKernelDetected = $kernelRelease -match "(?i)microsoft|wsl"
$hostIsWsl = $wslKernelDetected -and -not $hostIsContainer
$runningInDedicatedGpuCiWorkflow =
  [string]::Equals($env:GITHUB_ACTIONS, "true", [System.StringComparison]::OrdinalIgnoreCase) -and
  [string]::Equals($env:GITHUB_WORKFLOW, "runtime-linux-gpu-smoke", [System.StringComparison]::Ordinal)
$nvidiaSmi = Invoke-CapturedCommand -FilePath "nvidia-smi" -Arguments @(
  "--query-gpu=name,driver_version", "--format=csv,noheader"
) -StdoutPath (Join-Path $ReportDirectory "nvidia-smi.stdout.log") -StderrPath (Join-Path $ReportDirectory "nvidia-smi.stderr.log") -WorkingDirectory $consumerRoot

$readyForEnqueue = (Get-MarkerValue -Text $runtime.stdout -Prefix "ReadyForEnqueue=") -eq "True"
$enqueueCompleted = (Get-MarkerValue -Text $runtime.stdout -Prefix "EnqueueCompleted=") -eq "True"
$streamSynchronized = (Get-MarkerValue -Text $runtime.stdout -Prefix "StreamSynchronized=") -eq "True"
$identityOutputMatch = (Get-MarkerValue -Text $runtime.stdout -Prefix "IdentityOutputMatch=") -eq "True"
$smokePassed = (Get-MarkerValue -Text $runtime.stdout -Prefix "MinimalPackageRuntimeSmoke=") -eq "Passed"
$engineSerializedBytesText = Get-MarkerValue -Text $runtime.stdout -Prefix "EngineSerializedBytes="
$engineSerializedBytes = if ($engineSerializedBytesText -match '^\d+$') { [long]$engineSerializedBytesText } else { 0 }
$runtimeExecutionProof = $restore.exitCode -eq 0 -and $build.exitCode -eq 0 -and $runtime.exitCode -eq 0 -and
  $readyForEnqueue -and $enqueueCompleted -and $streamSynchronized -and $identityOutputMatch -and $smokePassed

$record = [ordered]@{
  schemaVersion = 1
  recordKind = "minimal-linux-bridge-package-runtime-consumer"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  sourceRuntimeKey = $SourceRuntimeKey
  host = [ordered]@{
    osDescription = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    osArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    processArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
    isLinux = $hostIsLinux
    isX64 = $hostIsX64
    isContainer = $hostIsContainer
    wslKernelDetected = $wslKernelDetected
    isWsl = $hostIsWsl
    kernelRelease = $kernelRelease
    githubActions = [string]::Equals($env:GITHUB_ACTIONS, "true", [System.StringComparison]::OrdinalIgnoreCase)
    githubWorkflow = $env:GITHUB_WORKFLOW
    dedicatedGpuCiWorkflow = $runningInDedicatedGpuCiWorkflow
    runnerName = $env:RUNNER_NAME
    nvidiaSmiExitCode = $nvidiaSmi.exitCode
    nvidiaSmiSummary = $nvidiaSmi.stdout.Trim()
  }
  packages = [ordered]@{
    managed = $managedPackage
    bridge = $bridgePackage
    packageReferenceCount = 2
    usesPackageReferenceOnly = $true
    usesProjectReference = $false
    usesDirectAssemblyReference = $false
  }
  consumer = [ordered]@{
    root = $consumerRoot
    rootOutsideRepository = $true
    projectSha256 = (Get-FileHash -LiteralPath $projectPath -Algorithm SHA256).Hash.ToLowerInvariant()
    programSha256 = (Get-FileHash -LiteralPath $programPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
  execution = [ordered]@{
    restoreExitCode = $restore.exitCode
    buildExitCode = $build.exitCode
    runtimeExitCode = $runtime.exitCode
    engineSerializedBytes = $engineSerializedBytes
    readyForEnqueue = $readyForEnqueue
    enqueueCompleted = $enqueueCompleted
    streamSynchronized = $streamSynchronized
    identityOutputMatch = $identityOutputMatch
    smokePassed = $smokePassed
  }
  logs = [ordered]@{
    restoreStdout = Get-LogEvidence -Path $restoreStdout
    restoreStderr = Get-LogEvidence -Path $restoreStderr
    buildStdout = Get-LogEvidence -Path $buildStdout
    buildStderr = Get-LogEvidence -Path $buildStderr
    runtimeStdout = Get-LogEvidence -Path $runtimeStdout
    runtimeStderr = Get-LogEvidence -Path $runtimeStderr
  }
  runtimeExecutionProof = $runtimeExecutionProof
  proofClassification = if (-not $runtimeExecutionProof) {
    "failed-linux-runtime-attempt"
  }
  elseif ($hostIsContainer) {
    "local-package-linux-container-gpu-runtime-proof"
  }
  elseif ($hostIsWsl) {
    "local-package-wsl-gpu-runtime-proof"
  }
  elseif ($runningInDedicatedGpuCiWorkflow) {
    "local-package-gpu-ci-runtime-proof"
  }
  else {
    "local-package-linux-gpu-runtime-proof"
  }
  canPromoteWslRuntimeProof = $runtimeExecutionProof -and $hostIsWsl
  canPromoteGpuCiRuntimeProof = $runtimeExecutionProof -and $runningInDedicatedGpuCiWorkflow
  isPublicPackageProof = $false
  isPostPublishProof = $false
  performsPublish = $false
  proofBoundary = "This record proves only the selected local managed and Bridge packages on the current Linux GPU host. It is WSL proof only when host.isWsl is true, GPU CI proof only in the dedicated GitHub Actions GPU job, and never public-package or post-publish proof."
}

$jsonPath = Join-Path $ReportDirectory "minimal-linux-bridge-package-runtime-consumer.json"
$markdownPath = Join-Path $ReportDirectory "minimal-linux-bridge-package-runtime-consumer.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Minimal Linux Bridge Package Runtime Consumer

- runtime key: ``$SourceRuntimeKey``
- restore/build/runtime exit codes: ``$($restore.exitCode)/$($build.exitCode)/$($runtime.exitCode)``
- ready/enqueue/synchronized/output match: ``$readyForEnqueue/$enqueueCompleted/$streamSynchronized/$identityOutputMatch``
- runtime execution proof: ``$runtimeExecutionProof``
- WSL proof candidate: ``$($record.canPromoteWslRuntimeProof)``
- GPU CI proof candidate: ``$($record.canPromoteGpuCiRuntimeProof)``
- public/post-publish proof: ``False/False``

## Boundary

$($record.proofBoundary)
"@
[System.IO.File]::WriteAllText($markdownPath, $markdown, $utf8)

Write-Host "Minimal Linux Bridge runtime report written to $jsonPath"
Write-Host "Minimal Linux Bridge runtime report written to $markdownPath"

if ($ownedConsumerRoot -and -not $KeepConsumerOutput.IsPresent) {
  $resolvedTaskTempRoot = [System.IO.Path]::GetFullPath($taskTempRoot)
  if (-not $consumerRoot.StartsWith($resolvedTaskTempRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove consumer root outside the owned task temp directory: $consumerRoot"
  }
  Remove-Item -LiteralPath $consumerRoot -Recurse -Force
}

if (-not $runtimeExecutionProof) {
  throw "Minimal Linux Bridge package runtime consumer failed. See $jsonPath"
}
