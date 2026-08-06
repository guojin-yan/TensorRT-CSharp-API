[CmdletBinding()]
param(
  [string]$SourceRuntimeKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$TargetFramework = "net8.0",
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string[]]$AdditionalPackageSource = @(),
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
  [switch]$SkipBaselineValidation,
  [switch]$SkipInstalledVendorAssetHashing,
  [switch]$KeepConsumerOutput,
  [switch]$AllowRuntimeSmokeFailure,
  [ValidateSet("success", "callback-return-false", "callback-throw", "attempted-no-invocation", "missing-vendor-dependency")]
  [string]$DebugListenerScenario = "success",
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Expand-KeyList {
  param([string[]]$Values)

  $items = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $items.Add($trimmed)
      }
    }
  }

  return @($items | Select-Object -Unique)
}

function ConvertTo-XmlAttributeValue {
  param([string]$Value)
  return [System.Security.SecurityElement]::Escape($Value)
}

function Resolve-PackageSourceValue {
  param([Parameter(Mandatory = $true)][string]$Source)

  if ($Source -match '^[a-zA-Z][a-zA-Z0-9+.-]*://') {
    return $Source
  }

  if ([System.IO.Path]::IsPathRooted($Source)) {
    return [System.IO.Path]::GetFullPath($Source)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Source))
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
    return [pscustomobject]@{
      Path = [System.IO.Path]::GetFullPath($Path)
      Id = $xml.SelectSingleNode("//n:metadata/n:id", $namespaceManager).InnerText
      Version = $xml.SelectSingleNode("//n:metadata/n:version", $namespaceManager).InnerText
      LastWriteTime = (Get-Item -LiteralPath $Path).LastWriteTime
      Sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    }
  }
  finally {
    $zip.Dispose()
  }
}

function Find-Package {
  param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$PackageId
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }

  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($package in Get-ChildItem -LiteralPath $Directory -Filter *.nupkg) {
    $metadata = Get-NupkgMetadata -Path $package.FullName
    if ([string]::Equals($metadata.Id, $PackageId, [System.StringComparison]::OrdinalIgnoreCase)) {
      $matches.Add($metadata)
    }
  }

  if ($matches.Count -eq 0) {
    throw "Package '$PackageId' was not found under $Directory."
  }

  return @($matches | Sort-Object LastWriteTime, Version -Descending)[0]
}

function New-LinuxDynamicBridgePackage {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SourcePackage
  )

  return [pscustomobject]@{
    key = "$($SourcePackage.key)-bridge"
    sourceRuntimeKey = $SourcePackage.key
    packageId = "$($SourcePackage.packageId).Bridge"
    rid = $SourcePackage.rid
    platform = $SourcePackage.platform
    tensorRtLine = $SourcePackage.tensorRtLine
    cudaLine = $SourcePackage.cudaLine
    role = "bridge"
    prototypeState = $SourcePackage.validationState
    assets = @([string]$SourcePackage.bridgeFile)
    generated = $true
  }
}

function Resolve-BridgePackage {
  param([Parameter(Mandatory = $true)][string]$RuntimeKey)

  $manifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
  $runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
  $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $sourcePackage = $runtimeManifest.packages |
    Where-Object { $_.key -eq $RuntimeKey } |
    Select-Object -First 1
  if (-not $sourcePackage) {
    throw "Runtime package '$RuntimeKey' was not found in $runtimeManifestPath."
  }
  $package = $manifest.packages |
    Where-Object { $_.sourceRuntimeKey -eq $RuntimeKey -and $_.role -eq "bridge" } |
    Select-Object -First 1
  if (-not $package -and [string]$sourcePackage.platform -eq "linux") {
    $package = New-LinuxDynamicBridgePackage -SourcePackage $sourcePackage
  }
  if (-not $package) {
    throw "Bridge split package for '$RuntimeKey' was not found in $manifestPath."
  }

  return $package
}

function Resolve-RuntimeRootSet {
  $resolved = & (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") `
    -RuntimePackageKey $SourceRuntimeKey `
    -RepositoryRoot $RepositoryRoot | ConvertFrom-Json

  return [pscustomobject]@{
    TensorRtRoot = if ([string]::IsNullOrWhiteSpace($TensorRtRoot)) { [string]$resolved.tensorRtRoot } else { [System.IO.Path]::GetFullPath($TensorRtRoot) }
    CudaRoot = if ([string]::IsNullOrWhiteSpace($CudaRoot)) { [string]$resolved.cudaRoot } else { [System.IO.Path]::GetFullPath($CudaRoot) }
    CudnnRoot = if ([string]::IsNullOrWhiteSpace($CudnnRoot)) { [string]$resolved.cudnnRoot } else { [System.IO.Path]::GetFullPath($CudnnRoot) }
  }
}

function Assert-ExistingDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Path,
    [switch]$Optional
  )

  if ([string]::IsNullOrWhiteSpace($Path)) {
    if ($Optional.IsPresent) {
      return
    }

    throw "$Name was not resolved."
  }

  if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    throw "$Name does not exist: $Path"
  }
}

function Get-SafePathName {
  param([Parameter(Mandatory = $true)][string]$Value)
  return ($Value -replace '[^a-zA-Z0-9._-]', '-')
}

function Get-ShortRuntimeConsumerRoot {
  param(
    [Parameter(Mandatory = $true)][string]$RuntimeKey,
    [Parameter(Mandatory = $true)][string]$Rid
  )

  $shortBase = Join-Path ([System.IO.Path]::GetTempPath()) "jybr"
  $runtimeToken = ($RuntimeKey -replace "win-x64-", "" -replace "linux-x64-", "" -replace "[^a-zA-Z0-9]+", "")
  if ($runtimeToken.Length -gt 28) {
    $runtimeToken = $runtimeToken.Substring(0, 28)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $shortBase "$Rid-$runtimeToken"))
}

function Test-PathWithin {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Parent
  )

  $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
  $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
  return $resolvedPath.StartsWith($resolvedParent + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

function Remove-ConsumerDirectory {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$AllowedRoot
  )

  if (-not (Test-PathWithin -Path $Path -Parent $AllowedRoot)) {
    throw "Refusing to remove consumer path outside the allowed root: $Path"
  }

  if (Test-Path -LiteralPath $Path) {
    Remove-Item -LiteralPath $Path -Recurse -Force
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
  if (-not [string]::IsNullOrWhiteSpace($Roots.CudnnRoot)) {
    $candidates += (Join-Path $Roots.CudnnRoot "bin")
  }

  return @($candidates |
      Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
      ForEach-Object { (Resolve-Path -LiteralPath $_).Path } |
      Select-Object -Unique)
}

function Get-NativeAssetEvidence {
  param(
    [Parameter(Mandatory = $true)][string]$BridgeOutputPath,
    [Parameter(Mandatory = $true)][object]$Roots,
    [switch]$SkipVendorHashing
  )

  $assets = New-Object System.Collections.Generic.List[object]
  $candidates = New-Object System.Collections.Generic.List[string]
  $candidates.Add($BridgeOutputPath)

  if (-not $SkipVendorHashing.IsPresent) {
    foreach ($directoryAndPattern in @(
        [pscustomobject]@{ Directory = (Join-Path $Roots.TensorRtRoot "bin"); Pattern = "nvinfer*.dll" },
        [pscustomobject]@{ Directory = (Join-Path $Roots.TensorRtRoot "bin"); Pattern = "nvonnxparser*.dll" },
        [pscustomobject]@{ Directory = (Join-Path $Roots.TensorRtRoot "lib"); Pattern = "nvinfer*.dll" },
        [pscustomobject]@{ Directory = (Join-Path $Roots.TensorRtRoot "lib"); Pattern = "nvonnxparser*.dll" },
        [pscustomobject]@{ Directory = (Join-Path $Roots.CudaRoot "bin\x64"); Pattern = "cudart64*.dll" },
        [pscustomobject]@{ Directory = (Join-Path $Roots.CudaRoot "bin"); Pattern = "cudart64*.dll" },
        [pscustomobject]@{ Directory = (Join-Path $Roots.CudnnRoot "bin"); Pattern = "cudnn*64_9.dll" }
      )) {
      if ([string]::IsNullOrWhiteSpace($directoryAndPattern.Directory) -or
          -not (Test-Path -LiteralPath $directoryAndPattern.Directory -PathType Container)) {
        continue
      }

      foreach ($file in Get-ChildItem -LiteralPath $directoryAndPattern.Directory -Filter $directoryAndPattern.Pattern -File) {
        $candidates.Add($file.FullName)
      }
    }
  }

  foreach ($path in @($candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      continue
    }

    $file = Get-Item -LiteralPath $path
    $version = $file.VersionInfo.FileVersion
    $isBridgeAsset = [string]::Equals($file.FullName, [System.IO.Path]::GetFullPath($BridgeOutputPath), [System.StringComparison]::OrdinalIgnoreCase)
    $sha256 = if ($isBridgeAsset -or -not $SkipVendorHashing.IsPresent) {
      (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
    else {
      ""
    }
    $assets.Add([pscustomobject]@{
        name = $file.Name
        path = $file.FullName
        length = $file.Length
        fileVersion = if ($null -eq $version) { "" } else { [string]$version }
        sha256 = $sha256
        sha256Captured = -not [string]::IsNullOrWhiteSpace($sha256)
        hashSkipReason = if ([string]::IsNullOrWhiteSpace($sha256)) { "diagnostic mode skipped installed vendor asset hashing" } else { "" }
      })
  }

  return $assets.ToArray()
}

function Get-NvidiaHostMetadata {
  $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $command) {
    return [pscustomobject]@{
      available = $false
      gpuName = ""
      driverVersion = ""
      diagnostic = "nvidia-smi was not found."
    }
  }

  $output = & $command.Source --query-gpu=name,driver_version --format=csv,noheader 2>&1
  if ($LASTEXITCODE -ne 0 -or @($output).Count -eq 0) {
    return [pscustomobject]@{
      available = $false
      gpuName = ""
      driverVersion = ""
      diagnostic = (@($output) -join " ")
    }
  }

  $parts = ([string]@($output)[0]).Split(',', 2)
  return [pscustomobject]@{
    available = $true
    gpuName = $parts[0].Trim()
    driverVersion = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }
    diagnostic = ""
  }
}

function Get-FirstMarkerValue {
  param(
    [string[]]$Lines,
    [Parameter(Mandatory = $true)][string]$Prefix
  )

  foreach ($line in @($Lines)) {
    $text = [string]$line
    if ($text.StartsWith($Prefix, [System.StringComparison]::Ordinal)) {
      return $text.Substring($Prefix.Length)
    }
  }

  return ""
}

function New-NuGetConfigContent {
  param(
    [Parameter(Mandatory = $true)][string]$ManagedSource,
    [Parameter(Mandatory = $true)][string]$BridgeSource
  )

  $sources = New-Object System.Collections.Generic.List[string]
  $sources.Add('    <clear />')
  $sources.Add('    <add key="jyppx-managed-local" value="' + (ConvertTo-XmlAttributeValue -Value $ManagedSource) + '" />')
  $sources.Add('    <add key="jyppx-bridge-local" value="' + (ConvertTo-XmlAttributeValue -Value $BridgeSource) + '" />')
  $index = 1
  foreach ($source in @(Expand-KeyList -Values $AdditionalPackageSource)) {
    $resolvedSource = Resolve-PackageSourceValue -Source $source
    $sources.Add('    <add key="additional-' + $index + '" value="' + (ConvertTo-XmlAttributeValue -Value $resolvedSource) + '" />')
    $index++
  }
  $sources.Add('    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />')

  return @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
$($sources -join "`r`n")
  </packageSources>
</configuration>
"@
}

function Get-PackageSourceEvidence {
  param(
    [Parameter(Mandatory = $true)][object]$Package,
    [Parameter(Mandatory = $true)][string]$SourceDirectory,
    [Parameter(Mandatory = $true)][string]$SourceName
  )

  $resolvedSourceDirectory = Resolve-PackageSourceValue -Source $SourceDirectory
  $resolvedPackagePath = [System.IO.Path]::GetFullPath([string]$Package.Path)
  $sourceIsLocalDirectory = Test-Path -LiteralPath $resolvedSourceDirectory -PathType Container
  $packageIsFromSourceDirectory = $false
  if ($sourceIsLocalDirectory) {
    $packageIsFromSourceDirectory = Test-PathWithin -Path $resolvedPackagePath -Parent $resolvedSourceDirectory
  }

  return [ordered]@{
    sourceName = $SourceName
    sourceDirectory = $resolvedSourceDirectory
    sourceIsLocalDirectory = $sourceIsLocalDirectory
    packageIsFromSourceDirectory = $packageIsFromSourceDirectory
    isLocalPackageFeed = $sourceIsLocalDirectory -and $packageIsFromSourceDirectory
    publicFeedProof = $false
    proofBoundary = "This package was resolved from a local package directory for compatible-host consumer validation. Local feed evidence is not public clean package-consumer or post-publish proof."
  }
}

function Write-Reports {
  param(
    [Parameter(Mandatory = $true)][object]$Result,
    [Parameter(Mandatory = $true)][string]$Directory
  )

  New-Item -ItemType Directory -Path $Directory -Force | Out-Null
  $jsonPath = Join-Path $Directory "bridge-package-runtime-consumer-proof.json"
  $markdownPath = Join-Path $Directory "bridge-package-runtime-consumer-proof.md"
  $Result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Bridge Package Runtime Consumer Proof")
  $lines.Add("")
  $lines.Add("| Runtime key | Classification | Smoke | Exit code | Identity output | DebugListener callback | Consumer outside repo |")
  $lines.Add("| --- | --- | --- | ---: | --- | --- | --- |")
  $lines.Add("| $($Result.sourceRuntimeKey) | $($Result.proofClassification) | $($Result.smokeStatus) | $($Result.exitCode) | $($Result.identityOutputMatch) | $($Result.debugListenerCallback.status) | $($Result.consumerRootOutsideRepository) |")
  $lines.Add("")
  $lines.Add("- managed package: ``$($Result.packages.managed.id) $($Result.packages.managed.version)``; SHA256=``$($Result.packages.managed.sha256)``")
  $lines.Add("  - source: ``$($Result.packages.managed.source.sourceDirectory)``; local feed=$($Result.packages.managed.source.isLocalPackageFeed); public feed proof=$($Result.packages.managed.source.publicFeedProof)")
  $lines.Add("- bridge package: ``$($Result.packages.bridge.id) $($Result.packages.bridge.version)``; SHA256=``$($Result.packages.bridge.sha256)``")
  $lines.Add("  - source: ``$($Result.packages.bridge.source.sourceDirectory)``; local feed=$($Result.packages.bridge.source.isLocalPackageFeed); public feed proof=$($Result.packages.bridge.source.publicFeedProof)")
  $lines.Add("- consumer project: ``$($Result.consumer.projectPath)``; SHA256=``$($Result.consumer.projectSha256)``")
  $lines.Add("- stdout: ``$($Result.logs.stdoutPath)``; SHA256=``$($Result.logs.stdoutSha256)``")
  $lines.Add("- stderr: ``$($Result.logs.stderrPath)``; SHA256=``$($Result.logs.stderrSha256)``")
  $lines.Add("- combined log SHA256: ``$($Result.logs.combinedSha256)``")
  $lines.Add("- engine serialized bytes: $($Result.engineSerializedBytes)")
  $lines.Add("- enqueue completed: $($Result.enqueueCompleted)")
  $lines.Add("- runtime execution proof: $($Result.isRuntimeExecutionProof)")
  $lines.Add("- package-consumer runtime proof: $($Result.isPackageConsumerRuntimeProof)")
  $lines.Add("- DebugListener evidence scope: ``$($Result.debugListenerCallback.evidenceScope)``")
  $lines.Add("- DebugListener invocation/failure/in-flight: $($Result.debugListenerCallback.invocationCount)/$($Result.debugListenerCallback.failureCount)/$($Result.debugListenerCallback.inFlightCallbackCount)")
  $lines.Add("- DebugListener metadata copied / borrowed pointer exposed: $($Result.debugListenerCallback.metadataCopied)/$($Result.debugListenerCallback.borrowedPointerExposed)")
  $lines.Add("- DebugListener detach count: $($Result.debugListenerCallback.detachCount)")
  $lines.Add("- local-package DebugListener callback runtime proof: $($Result.debugListenerCallback.isLocalPackageCallbackRuntimeProof)")
  $lines.Add("- callback-state snapshot observed/complete/coherent: $($Result.callbackStateSnapshot.observed)/$($Result.callbackStateSnapshot.complete)/$($Result.callbackStateSnapshot.coherent)")
  $lines.Add("- callback-state last status/operation: ``$($Result.callbackStateSnapshot.lastStatus)`` / ``$($Result.callbackStateSnapshot.lastOperation)``")
  $lines.Add("- DebugListener negative control: ``$($Result.negativeControl.scenario)``; requested=$($Result.negativeControl.requested); passed=$($Result.negativeControl.passed)")
  $lines.Add("- source-tree/public-package/post-publish proof in this report: $($Result.proofScopes.sourceTree.isProof)/$($Result.proofScopes.publicPackage.isProof)/$($Result.proofScopes.postPublish.isProof)")
  $lines.Add("- compatible-host runtime promotion: $($Result.canPromoteCompatibleHostRuntimeProof)")
  $lines.Add("- public publish: $($Result.canPublishPublicly)")
  $lines.Add("- release close: $($Result.canCloseReleaseIssue)")
  $lines.Add("- boundary: $($Result.runtimeProofBoundary)")
  $lines.Add("")
  $lines.Add("Generated by ``eng/Test-BridgePackageRuntimeConsumer.ps1``.")
  Set-Content -LiteralPath $markdownPath -Value $lines -Encoding utf8

  Write-Host "Bridge package runtime proof written to $jsonPath"
  Write-Host "Bridge package runtime proof written to $markdownPath"
}

$bridgeDefinition = Resolve-BridgePackage -RuntimeKey $SourceRuntimeKey
$rid = [string]$bridgeDefinition.rid
$bridgeFileName = [string]@($bridgeDefinition.assets)[0]
$tensorRtLineExpression = switch ([string]$bridgeDefinition.tensorRtLine) {
  "8" { "TensorRtApiLine.TensorRt8" }
  "10" { "TensorRtApiLine.TensorRt10" }
  "11" { "TensorRtApiLine.TensorRt11" }
  default { throw "Unsupported TensorRT line '$($bridgeDefinition.tensorRtLine)'." }
}
if ($DebugListenerScenario -ne "success" -and @("10", "11") -notcontains [string]$bridgeDefinition.tensorRtLine) {
  throw "DebugListener negative controls require a TensorRT 10 or 11 runtime key."
}

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
else {
  $ManagedPackageDirectory = Resolve-PackageSourceValue -Source $ManagedPackageDirectory
}

if ([string]::IsNullOrWhiteSpace($BridgePackageDirectory)) {
  $BridgePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$SourceRuntimeKey"
}
else {
  $BridgePackageDirectory = Resolve-PackageSourceValue -Source $BridgePackageDirectory
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\package-consumer\bridge-runtime\$SourceRuntimeKey"
}
elseif (-not [System.IO.Path]::IsPathRooted($ReportDirectory)) {
  $ReportDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ReportDirectory))
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $consumerRoot = Get-ShortRuntimeConsumerRoot -RuntimeKey $SourceRuntimeKey -Rid $rid
  $OutputRoot = [System.IO.Path]::GetDirectoryName($consumerRoot)
}
else {
  $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
  $safeKey = Get-SafePathName -Value $SourceRuntimeKey
  $consumerRoot = [System.IO.Path]::GetFullPath((Join-Path $OutputRoot $safeKey))
}

$consumerRootOutsideRepository = -not (Test-PathWithin -Path $consumerRoot -Parent $RepositoryRoot)
if (-not $consumerRootOutsideRepository) {
  throw "Runtime proof consumer root must be outside the source repository: $consumerRoot"
}

$runtimeRoots = Resolve-RuntimeRootSet
Assert-ExistingDirectory -Name "TensorRtRoot" -Path $runtimeRoots.TensorRtRoot
Assert-ExistingDirectory -Name "CudaRoot" -Path $runtimeRoots.CudaRoot
Assert-ExistingDirectory -Name "CudnnRoot" -Path $runtimeRoots.CudnnRoot -Optional
$runtimeSearchDirectories = @(Get-RuntimeSearchDirectories -Roots $runtimeRoots)
if ($runtimeSearchDirectories.Count -lt 2) {
  throw "TensorRT and CUDA runtime search directories were not resolved."
}

$managedPackage = Find-Package -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
$bridgePackage = Find-Package -Directory $BridgePackageDirectory -PackageId ([string]$bridgeDefinition.packageId)

if (-not $SkipBaselineValidation.IsPresent) {
  $baselineArguments = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepositoryRoot "eng\Test-BridgePackageConsumer.ps1"),
    "-SourceRuntimeKey",
    $SourceRuntimeKey,
    "-ManagedPackageDirectory",
    $ManagedPackageDirectory,
    "-BridgePackageDirectory",
    $BridgePackageDirectory,
    "-OutputRoot",
    (Join-Path $OutputRoot "baseline"),
    "-ReportDirectory",
    (Join-Path $ReportDirectory "baseline"),
    "-SkipProbe"
  )
  $expandedAdditionalSources = @(Expand-KeyList -Values $AdditionalPackageSource)
  if ($expandedAdditionalSources.Count -gt 0) {
    $baselineArguments += @("-AdditionalPackageSource", ($expandedAdditionalSources -join ","))
  }

  & pwsh @baselineArguments
  if ($LASTEXITCODE -ne 0) {
    throw "Bridge package baseline validation failed with exit code $LASTEXITCODE."
  }
}

Remove-ConsumerDirectory -Path $consumerRoot -AllowedRoot $OutputRoot
New-Item -ItemType Directory -Path $consumerRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null

$nugetConfigPath = Join-Path $consumerRoot "NuGet.config"
$projectPath = Join-Path $consumerRoot "BridgePackageRuntimeConsumer.csproj"
$programPath = Join-Path $consumerRoot "Program.cs"
$restorePackagesPath = Join-Path $consumerRoot ".nuget\packages"

$nugetConfig = New-NuGetConfigContent -ManagedSource $ManagedPackageDirectory -BridgeSource $BridgePackageDirectory
$project = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$TargetFramework</TargetFramework>
    <RuntimeIdentifier>$rid</RuntimeIdentifier>
    <RestorePackagesPath>$restorePackagesPath</RestorePackagesPath>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="$($managedPackage.Id)" Version="$($managedPackage.Version)" />
    <PackageReference Include="$($bridgePackage.Id)" Version="$($bridgePackage.Version)" />
  </ItemGroup>
</Project>
"@

$program = @'
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

static void WriteRuntimeCreateDiagnostic(TensorRtApiLine line)
{
    MethodInfo? method = typeof(TensorRtEnvironmentProbe).GetMethod(
        "GetRuntimeCreateDiagnostic",
        BindingFlags.Public | BindingFlags.Static,
        binder: null,
        types: new[] { typeof(TensorRtApiLine) },
        modifiers: null);

    if (method is null)
    {
        Console.WriteLine("NativeCreateRuntimeDiagnosticAvailable=False");
        Console.WriteLine("NativeCreateRuntimeAttempted=False");
        Console.WriteLine("NativeCreateRuntimeReturnedNull=False");
        Console.WriteLine("NativeCreateRuntimeReturnedNonNull=False");
        Console.WriteLine("NativeCreateRuntimeLastStatus=NotSupported");
        Console.WriteLine("NativeCreateRuntimeDetectedVersion=");
        Console.WriteLine("NativeCreateRuntimeLoggerCallbackAvailable=False");
        Console.WriteLine("NativeCreateRuntimeLoggerMessageCount=0");
        Console.WriteLine("NativeCreateRuntimeLastLoggerSeverity=0");
        Console.WriteLine("NativeCreateRuntimeLastLoggerMessage=");
        Console.WriteLine("NativeCreateRuntimePhase=not-supported");
        Console.WriteLine("NativeCreateRuntimeNativeDetail=Runtime create diagnostic API is not available in this managed package.");
        Console.WriteLine("NativeCreateRuntimeDiagnosticMessage=Runtime create diagnostic API is not available in this managed package.");
        return;
    }

    try
    {
        object? snapshot = method.Invoke(null, new object[] { line });
        if (snapshot is null)
        {
            Console.WriteLine("NativeCreateRuntimeDiagnosticAvailable=False");
            Console.WriteLine("NativeCreateRuntimeAttempted=False");
            Console.WriteLine("NativeCreateRuntimeReturnedNull=False");
            Console.WriteLine("NativeCreateRuntimeReturnedNonNull=False");
            Console.WriteLine("NativeCreateRuntimeLastStatus=RuntimeError");
            Console.WriteLine("NativeCreateRuntimeDetectedVersion=");
            Console.WriteLine("NativeCreateRuntimeLoggerCallbackAvailable=False");
            Console.WriteLine("NativeCreateRuntimeLoggerMessageCount=0");
            Console.WriteLine("NativeCreateRuntimeLastLoggerSeverity=0");
            Console.WriteLine("NativeCreateRuntimeLastLoggerMessage=");
            Console.WriteLine("NativeCreateRuntimePhase=managed-null-snapshot");
            Console.WriteLine("NativeCreateRuntimeNativeDetail=Runtime create diagnostic returned a null managed snapshot.");
            Console.WriteLine("NativeCreateRuntimeDiagnosticMessage=Runtime create diagnostic returned a null managed snapshot.");
            return;
        }

        Console.WriteLine("NativeCreateRuntimeDiagnosticAvailable=" + ReadSnapshotProperty(snapshot, "DiagnosticAvailable", "False"));
        Console.WriteLine("NativeCreateRuntimeAttempted=" + ReadSnapshotProperty(snapshot, "Attempted", "False"));
        Console.WriteLine("NativeCreateRuntimeReturnedNull=" + ReadSnapshotProperty(snapshot, "CreateInferRuntimeReturnedNull", "False"));
        Console.WriteLine("NativeCreateRuntimeReturnedNonNull=" + ReadSnapshotProperty(snapshot, "CreateInferRuntimeReturnedNonNull", "False"));
        Console.WriteLine("NativeCreateRuntimeLastStatus=" + ReadSnapshotProperty(snapshot, "LastStatus", ""));
        Console.WriteLine("NativeCreateRuntimeDetectedVersion=" + ReadSnapshotProperty(snapshot, "DetectedVersion", ""));
        Console.WriteLine("NativeCreateRuntimeLoggerCallbackAvailable=" + ReadSnapshotProperty(snapshot, "LoggerCallbackAvailable", "False"));
        Console.WriteLine("NativeCreateRuntimeLoggerMessageCount=" + ReadSnapshotProperty(snapshot, "LoggerMessageCount", "0"));
        Console.WriteLine("NativeCreateRuntimeLastLoggerSeverity=" + ReadSnapshotProperty(snapshot, "LastLoggerSeverity", "0"));
        Console.WriteLine("NativeCreateRuntimeLastLoggerMessage=" + ReadSnapshotProperty(snapshot, "LastLoggerMessage", ""));
        Console.WriteLine("NativeCreateRuntimePhase=" + ReadSnapshotProperty(snapshot, "CreateRuntimePhase", ""));
        Console.WriteLine("NativeCreateRuntimeNativeDetail=" + ReadSnapshotProperty(snapshot, "NativeDetail", ""));
        Console.WriteLine("NativeCreateRuntimeDiagnosticMessage=" + ReadSnapshotProperty(snapshot, "Diagnostic", ""));
    }
    catch (TargetInvocationException exception)
    {
        Console.WriteLine("NativeCreateRuntimeDiagnosticAvailable=False");
        Console.WriteLine("NativeCreateRuntimeAttempted=False");
        Console.WriteLine("NativeCreateRuntimeReturnedNull=False");
        Console.WriteLine("NativeCreateRuntimeReturnedNonNull=False");
        Console.WriteLine("NativeCreateRuntimeLastStatus=RuntimeError");
        Console.WriteLine("NativeCreateRuntimeDetectedVersion=");
        Console.WriteLine("NativeCreateRuntimeLoggerCallbackAvailable=False");
        Console.WriteLine("NativeCreateRuntimeLoggerMessageCount=0");
        Console.WriteLine("NativeCreateRuntimeLastLoggerSeverity=0");
        Console.WriteLine("NativeCreateRuntimeLastLoggerMessage=");
        Console.WriteLine("NativeCreateRuntimePhase=managed-invocation-exception");
        Console.WriteLine("NativeCreateRuntimeNativeDetail=" + (exception.InnerException?.Message ?? exception.Message));
        Console.WriteLine("NativeCreateRuntimeDiagnosticMessage=" + (exception.InnerException?.Message ?? exception.Message));
    }
}

static string ReadSnapshotProperty(object snapshot, string propertyName, string fallback)
{
    PropertyInfo? property = snapshot.GetType().GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance);
    object? value = property?.GetValue(snapshot);
    return value?.ToString() ?? fallback;
}

static void WriteCudaPreflight()
{
    bool available = false;
    bool attempted = true;
    string driverVersion = "";
    string runtimeVersion = "";
    string deviceCount = "";
    string selectedDevice = "";
    string deviceName = "";
    string getDeviceCountStatus = "NotStarted";
    string initStatus = "NotAttempted";
    string lastErrorName = "";
    string lastErrorMessage = "";
    bool canAttemptTensorRtRuntimeCreate = false;

    try
    {
        driverVersion = CudaDevice.DriverVersion.ToString();
        runtimeVersion = CudaDevice.RuntimeVersion.ToString();
        int count = CudaDevice.Count;
        deviceCount = count.ToString();
        getDeviceCountStatus = "Ok";
        available = true;

        if (count > 0)
        {
            try
            {
                selectedDevice = CudaDevice.Current.ToString();
                CudaDeviceInfo info = CudaDevice.GetInfo(CudaDevice.Current);
                deviceName = info.Name;
                initStatus = "DeviceInfoOk";
                canAttemptTensorRtRuntimeCreate = true;
            }
            catch (Exception exception)
            {
                initStatus = "DeviceInfoFailed";
                lastErrorName = exception.GetType().Name;
                lastErrorMessage = exception.Message;
            }
        }
        else
        {
            initStatus = "NoDevice";
            canAttemptTensorRtRuntimeCreate = false;
        }
    }
    catch (Exception exception)
    {
        available = false;
        getDeviceCountStatus = "Failed";
        initStatus = "CudaPreflightFailed";
        lastErrorName = exception.GetType().Name;
        lastErrorMessage = exception.Message;
    }

    Console.WriteLine("CudaPreflightAvailable=" + available);
    Console.WriteLine("CudaPreflightAttempted=" + attempted);
    Console.WriteLine("CudaPreflightDriverVersion=" + driverVersion);
    Console.WriteLine("CudaPreflightRuntimeVersion=" + runtimeVersion);
    Console.WriteLine("CudaPreflightDeviceCount=" + deviceCount);
    Console.WriteLine("CudaPreflightSelectedDevice=" + selectedDevice);
    Console.WriteLine("CudaPreflightDeviceName=" + deviceName);
    Console.WriteLine("CudaPreflightGetDeviceCountStatus=" + getDeviceCountStatus);
    Console.WriteLine("CudaPreflightInitStatus=" + initStatus);
    Console.WriteLine("CudaPreflightLastErrorName=" + lastErrorName);
    Console.WriteLine("CudaPreflightLastErrorMessage=" + lastErrorMessage);
    Console.WriteLine("CudaPreflightCanAttemptTensorRtRuntimeCreate=" + canAttemptTensorRtRuntimeCreate);
}

try
{
    TensorRtApiLine line = __TENSORRT_LINE__;
    string debugListenerScenario = "__DEBUG_LISTENER_SCENARIO__";
    bool negativeControlRequested = !string.Equals(debugListenerScenario, "success", StringComparison.Ordinal);
    Console.WriteLine("RuntimeSmokeRequested=True");
    Console.WriteLine("DebugListenerCallbackScenario=" + debugListenerScenario);
    Console.WriteLine("ConsumerProjectRoot=" + Directory.GetCurrentDirectory());
    Console.WriteLine("BridgeFileName=" + NativeBridgePathResolver.GetBridgeFileName());

    TensorRtEnvironmentSnapshot environment = TensorRtEnvironmentProbe.GetCurrent();
    Console.WriteLine("RuntimeEnvironment TRT=" + environment.BuildInfo.TensorRtVersion +
        " CUDA=" + environment.BuildInfo.CudaToolkitVersion +
        " TensorRtAvailable=" + environment.RuntimeInfo.TensorRtAvailable +
        " CudaAvailable=" + environment.RuntimeInfo.CudaToolkitAvailable);

    WriteCudaPreflight();
    WriteRuntimeCreateDiagnostic(line);

    using TensorRtLogger logger = new TensorRtLogger(line);
    using TensorRtRuntime runtime = new TensorRtRuntime(logger);
    using TensorRtBuilder builder = new TensorRtBuilder(logger);
    using TensorRtBuilderConfig config = builder.CreateBuilderConfig();
    using TensorRtNetworkDefinition network = builder.CreateNetwork(TensorRtNetworkDefinitionCreationFlags.ExplicitBatch);
    using CudaStream stream = new CudaStream();

    config.SetMemoryPoolLimit(TensorRtMemoryPoolType.Workspace, 32UL * 1024UL * 1024UL);
    TensorRtDims shape = new TensorRtDims(new[] { 1, 4 });
    using TensorRtTensor inputTensor = network.AddInput("input", TensorRtDataType.Float, shape);
    using TensorRtLayer identity = network.AddIdentity(inputTensor);
    using TensorRtTensor outputTensor = identity.GetOutput(0);
    outputTensor.Name = "output";
    network.MarkOutput(outputTensor);
    bool debugListenerCallbackRequired = line == TensorRtApiLine.TensorRt10 || line == TensorRtApiLine.TensorRt11;
    bool debugListenerCallbackEnabled = !string.Equals(debugListenerScenario, "attempted-no-invocation", StringComparison.Ordinal);
    bool debugTensorMarked = false;
    if (debugListenerCallbackRequired && debugListenerCallbackEnabled)
    {
        debugTensorMarked = network.MarkDebugTensor(outputTensor) && network.IsDebugTensor(outputTensor);
        if (!debugTensorMarked)
        {
            throw new InvalidOperationException("TensorRT did not retain the local-package DebugListener output mark.");
        }
    }
    Console.WriteLine("DebugListenerCallbackRequired=" + debugListenerCallbackRequired);
    Console.WriteLine("DebugListenerDebugTensorMarked=" + debugTensorMarked);

    using TensorRtHostMemory hostMemory = builder.BuildSerializedNetwork(network, config);
    byte[] serializedEngine = hostMemory.ToArray();
    Console.WriteLine("EngineSerializedBytes=" + serializedEngine.Length);
    if (serializedEngine.Length == 0 || hostMemory.SizeInBytes == 0)
    {
        throw new InvalidOperationException("TensorRT returned an empty serialized engine.");
    }

    using TensorRtEngine engine = runtime.Deserialize(serializedEngine);
    using TensorRtExecutionContext context = engine.CreateExecutionContext();
    float[] inputValues = new[] { 1.25f, -2.5f, 3.75f, 9.5f };
    using TensorRtInferenceBindings bindings = new TensorRtInferenceBindings(engine, context);
    bindings.CopyInputFromHost("input", inputValues, shape);
    bindings.AllocateDeviceBuffer("output", shape);
    bindings.BindAll();

    TensorRtExecutionContextReadiness readiness = bindings.GetReadiness(runShapeInference: true);
    Console.WriteLine("BindingReadiness=" + readiness);
    if (!readiness.IsReadyForEnqueue)
    {
        throw new InvalidOperationException("Identity network bindings are not ready for enqueue: " + readiness);
    }

    bool callbackStateComplete = context.TryGetCallbackStateSnapshot(
        "output",
        out TensorRtExecutionContextCallbackStateSnapshot callbackStateSnapshot,
        out string callbackStateDiagnostic);
    bool callbackStateCoherent =
        callbackStateComplete == callbackStateSnapshot.IsComplete &&
        (callbackStateComplete
            ? callbackStateSnapshot.LastStatus == BridgeStatusCode.Ok
            : callbackStateSnapshot.LastStatus != BridgeStatusCode.Ok &&
              !string.IsNullOrWhiteSpace(callbackStateSnapshot.LastOperation));
    Console.WriteLine("CallbackStateSnapshotComplete=" + callbackStateComplete);
    Console.WriteLine("CallbackStateSnapshotIsComplete=" + callbackStateSnapshot.IsComplete);
    Console.WriteLine("CallbackStateSnapshotCoherent=" + callbackStateCoherent);
    Console.WriteLine("CallbackStateSnapshotLastStatus=" + callbackStateSnapshot.LastStatus);
    Console.WriteLine("CallbackStateSnapshotLastOperation=" + callbackStateSnapshot.LastOperation);
    Console.WriteLine("CallbackStateSnapshotHasOutputAllocator=" + callbackStateSnapshot.HasOutputAllocator);
    Console.WriteLine("CallbackStateSnapshotHasTemporaryStorageAllocator=" + callbackStateSnapshot.HasTemporaryStorageAllocator);
    Console.WriteLine("CallbackStateSnapshotHasDebugListener=" + callbackStateSnapshot.HasDebugListener);
    Console.WriteLine("CallbackStateSnapshotDiagnostic=" + callbackStateDiagnostic);
    if (!callbackStateCoherent)
    {
        throw new InvalidOperationException("Callback-state snapshot completeness and partial-state diagnostics are inconsistent.");
    }

    TensorRtDebugTensorMetadataSnapshot callbackMetadata = default;
    string managedHandlerOutcome = "not-invoked";
    TensorRtDebugListenerCallbackOwner? debugListenerOwner = null;
    try
    {
        if (debugListenerCallbackRequired)
        {
            debugListenerOwner = new TensorRtDebugListenerCallbackOwner(
                line,
                metadata =>
                {
                    callbackMetadata = metadata;
                    if (string.Equals(debugListenerScenario, "callback-return-false", StringComparison.Ordinal))
                    {
                        managedHandlerOutcome = "returned-false";
                        return false;
                    }
                    if (string.Equals(debugListenerScenario, "callback-throw", StringComparison.Ordinal))
                    {
                        managedHandlerOutcome = "threw-invalid-operation";
                        throw new InvalidOperationException("Controlled DebugListener callback throw negative control.");
                    }
                    managedHandlerOutcome = "returned-true";
                    return true;
                });
            context.SetDebugListener(debugListenerOwner);
            if (debugListenerCallbackEnabled)
            {
                context.SetTensorDebugState("output", true);
            }
        }

        TensorRtInferenceExecutionSummary? execution = null;
        Exception? executionException = null;
        bool enqueueCompleted = false;
        bool outputMatch = false;
        try
        {
            execution = bindings.EnqueueAsync(stream, synchronize: true, runShapeInference: false);
            enqueueCompleted = true;
            float[] outputValues = bindings.ReadOutputSingles("output", inputValues.Length);
            outputMatch = inputValues.Zip(outputValues, static (expected, actual) => Math.Abs(expected - actual) <= 0.0001f).All(static match => match);
        }
        catch (Exception exception)
        {
            executionException = exception;
        }
        Console.WriteLine("EnqueueCompleted=" + enqueueCompleted);
        Console.WriteLine("ExecutionSummary=" + execution);
        Console.WriteLine("IdentityOutputMatch=" + outputMatch);
        Console.WriteLine("DebugListenerCallbackExecutionExceptionType=" + (executionException?.GetType().Name ?? ""));
        Console.WriteLine("DebugListenerCallbackExecutionExceptionMessage=" + (executionException?.Message ?? ""));

        if (debugListenerOwner is not null)
        {
            TensorRtDebugListenerRuntimeSnapshot attached = debugListenerOwner.GetRuntimeSnapshot();
            bool clearReturned = context.ClearDebugListener();
            TensorRtDebugListenerRuntimeSnapshot detached = debugListenerOwner.GetRuntimeSnapshot();
            bool detachedFromContext =
                clearReturned &&
                !detached.IsAttached &&
                !context.HasManagedDebugListener &&
                !context.HasDebugListener;
            bool callbackPassed =
                attached.IsAttached &&
                attached.AttachCount > 0 &&
                attached.IsRealCallbackRuntimeProof &&
                attached.InvocationCount > 0 &&
                attached.FailureCount == 0 &&
                attached.InFlightCallbackCount == 0 &&
                string.Equals(attached.TensorName, "output", StringComparison.Ordinal) &&
                attached.ShapeDimensions.SequenceEqual(new long[] { 1, 4 }) &&
                callbackMetadata.MetadataCopied &&
                string.Equals(callbackMetadata.TensorName, "output", StringComparison.Ordinal) &&
                !attached.BorrowedPointerExposed &&
                detached.DetachCount > 0 &&
                detachedFromContext;
            bool negativeCommonPassed =
                attached.IsAttached &&
                attached.AttachCount > 0 &&
                attached.InFlightCallbackCount == 0 &&
                !attached.BorrowedPointerExposed &&
                !attached.IsRealCallbackRuntimeProof &&
                detached.DetachCount > 0 &&
                detachedFromContext;
            bool negativeControlPassed = debugListenerScenario switch
            {
                "callback-return-false" =>
                    negativeCommonPassed &&
                    attached.InvocationCount > 0 &&
                    attached.FailureCount > 0 &&
                    !attached.LastCallbackSucceeded &&
                    callbackMetadata.MetadataCopied &&
                    string.Equals(managedHandlerOutcome, "returned-false", StringComparison.Ordinal),
                "callback-throw" =>
                    negativeCommonPassed &&
                    attached.InvocationCount > 0 &&
                    attached.FailureCount > 0 &&
                    !attached.LastCallbackSucceeded &&
                    callbackMetadata.MetadataCopied &&
                    string.Equals(managedHandlerOutcome, "threw-invalid-operation", StringComparison.Ordinal),
                "attempted-no-invocation" =>
                    negativeCommonPassed &&
                    attached.InvocationCount == 0 &&
                    attached.FailureCount == 0 &&
                    !callbackMetadata.MetadataCopied &&
                    !debugTensorMarked &&
                    enqueueCompleted &&
                    outputMatch &&
                    string.Equals(managedHandlerOutcome, "not-invoked", StringComparison.Ordinal),
                _ => false,
            };

            Console.WriteLine("DebugListenerCallbackRuntime=" + (callbackPassed ? "Passed" : "Failed"));
            Console.WriteLine("DebugListenerCallbackEvidenceScope=local-package");
            Console.WriteLine("DebugListenerCallbackAttached=" + attached.IsAttached);
            Console.WriteLine("DebugListenerCallbackNativeVTableInstalled=" + (attached.AttachCount > 0));
            Console.WriteLine("DebugListenerCallbackInvocationCount=" + attached.InvocationCount);
            Console.WriteLine("DebugListenerCallbackFailureCount=" + attached.FailureCount);
            Console.WriteLine("DebugListenerCallbackInFlightCallbackCount=" + attached.InFlightCallbackCount);
            Console.WriteLine("DebugListenerCallbackTensorName=" + attached.TensorName);
            Console.WriteLine("DebugListenerCallbackShape=[" + string.Join(",", attached.ShapeDimensions) + "]");
            Console.WriteLine("DebugListenerCallbackMetadataCopied=" + callbackMetadata.MetadataCopied);
            Console.WriteLine("DebugListenerCallbackBorrowedPointerExposed=" + attached.BorrowedPointerExposed);
            Console.WriteLine("DebugListenerCallbackClearReturned=" + clearReturned);
            Console.WriteLine("DebugListenerCallbackDetached=" + detachedFromContext);
            Console.WriteLine("DebugListenerCallbackDetachCount=" + detached.DetachCount);
            Console.WriteLine("DebugListenerCallbackIsRealRuntimeProof=" + attached.IsRealCallbackRuntimeProof);
            Console.WriteLine("DebugListenerCallbackLastCallbackSucceeded=" + attached.LastCallbackSucceeded);
            Console.WriteLine("DebugListenerCallbackLastStatus=" + attached.LastStatus);
            Console.WriteLine("DebugListenerCallbackManagedHandlerOutcome=" + managedHandlerOutcome);
            Console.WriteLine("DebugListenerNegativeControlPassed=" + negativeControlPassed);
            if (!negativeControlRequested && executionException is not null)
            {
                throw new InvalidOperationException("Local-package identity enqueue failed.", executionException);
            }
            if (!negativeControlRequested && !outputMatch)
            {
                throw new InvalidOperationException("Identity output mismatch in local-package runtime smoke.");
            }
            if (!negativeControlRequested && !callbackPassed)
            {
                throw new InvalidOperationException(
                    "Local-package DebugListener callback runtime did not satisfy attach/invoke/copy/detach invariants. Attached=" +
                    attached + " Detached=" + detached);
            }
            if (negativeControlRequested && !negativeControlPassed)
            {
                throw new InvalidOperationException(
                    "Controlled local-package DebugListener negative scenario did not satisfy fail-closed invariants. Scenario=" +
                    debugListenerScenario + " Attached=" + attached + " Detached=" + detached);
            }
        }
        else
        {
            Console.WriteLine("DebugListenerCallbackRuntime=SkippedUnsupportedTensorRtLine");
            Console.WriteLine("DebugListenerCallbackEvidenceScope=local-package");
        }

        Console.WriteLine(negativeControlRequested ? "RuntimeSmoke=ExpectedFailureVerified" : "RuntimeSmoke=Passed");
        return 0;
    }
    finally
    {
        if (debugListenerOwner is not null)
        {
            if (context.HasManagedDebugListener)
            {
                context.ClearDebugListener();
            }
            debugListenerOwner.Dispose();
        }
    }
}
catch (Exception exception)
{
    Console.Error.WriteLine("RuntimeSmoke=Failed");
    Console.Error.WriteLine(exception.ToString());
    return 1;
}
'@
$program = $program.Replace("__TENSORRT_LINE__", $tensorRtLineExpression)
$program = $program.Replace("__DEBUG_LISTENER_SCENARIO__", $DebugListenerScenario)

Set-Content -LiteralPath $nugetConfigPath -Value $nugetConfig -Encoding utf8
Set-Content -LiteralPath $projectPath -Value $project -Encoding utf8
Set-Content -LiteralPath $programPath -Value $program -Encoding utf8

Write-Host "Restoring clean external bridge package runtime consumer: $consumerRoot"
& dotnet restore $projectPath --configfile $nugetConfigPath
if ($LASTEXITCODE -ne 0) {
  throw "dotnet restore failed with exit code $LASTEXITCODE."
}

& dotnet build $projectPath -c Release --no-restore
if ($LASTEXITCODE -ne 0) {
  throw "dotnet build failed with exit code $LASTEXITCODE."
}

$outputDirectory = Join-Path $consumerRoot "bin\Release\$TargetFramework\$rid"
$bridgeOutputPath = Join-Path $outputDirectory $bridgeFileName
if (-not (Test-Path -LiteralPath $bridgeOutputPath -PathType Leaf)) {
  $bridgeOutputPath = Get-ChildItem -LiteralPath $outputDirectory -Recurse -Filter $bridgeFileName -File |
    Select-Object -First 1 -ExpandProperty FullName
}
if ([string]::IsNullOrWhiteSpace($bridgeOutputPath) -or -not (Test-Path -LiteralPath $bridgeOutputPath -PathType Leaf)) {
  throw "Bridge asset was not copied to the runtime consumer output: $bridgeFileName"
}

$stdoutPath = Join-Path $ReportDirectory "runtime-smoke.stdout.log"
$stderrPath = Join-Path $ReportDirectory "runtime-smoke.stderr.log"
$combinedPath = Join-Path $ReportDirectory "runtime-smoke.combined.log"
Remove-Item -LiteralPath $stdoutPath, $stderrPath, $combinedPath -Force -ErrorAction SilentlyContinue

$previousPath = $env:PATH
$previousBridgePath = $env:JYPPX_NATIVE_BRIDGE_PATH
$previousDevelopmentProbing = $env:JYPPX_ENABLE_DEVELOPMENT_PROBING
$dotnetCommand = (Get-Command dotnet -ErrorAction Stop).Source
$missingVendorDependencyIsolationApplied = $DebugListenerScenario -eq "missing-vendor-dependency"
try {
  if ($missingVendorDependencyIsolationApplied) {
    $minimalPathEntries = @(@(
      [System.IO.Path]::GetDirectoryName($dotnetCommand),
      (Join-Path $env:SystemRoot "System32"),
      $env:SystemRoot
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
    $env:PATH = $minimalPathEntries -join [System.IO.Path]::PathSeparator
  }
  else {
    $env:PATH = (@($runtimeSearchDirectories) + @($previousPath)) -join [System.IO.Path]::PathSeparator
  }
  $env:JYPPX_NATIVE_BRIDGE_PATH = $null
  $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $null
  Push-Location $consumerRoot
  try {
    & $dotnetCommand run --project $projectPath -c Release --no-build 1> $stdoutPath 2> $stderrPath
    $exitCode = $LASTEXITCODE
  }
  finally {
    Pop-Location
  }
}
finally {
  $env:PATH = $previousPath
  $env:JYPPX_NATIVE_BRIDGE_PATH = $previousBridgePath
  $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $previousDevelopmentProbing
}

$stdoutLines = if (Test-Path -LiteralPath $stdoutPath) { @(Get-Content -LiteralPath $stdoutPath) } else { @() }
$stderrLines = if (Test-Path -LiteralPath $stderrPath) { @(Get-Content -LiteralPath $stderrPath) } else { @() }
foreach ($line in $stdoutLines) {
  Write-Host $line
}
foreach ($line in $stderrLines) {
  Write-Warning $line
}
@($stdoutLines + $stderrLines) | Set-Content -LiteralPath $combinedPath -Encoding utf8

$debugListenerCallbackRequired = @("10", "11") -contains [string]$bridgeDefinition.tensorRtLine
$negativeControlRequested = $DebugListenerScenario -ne "success"
$debugListenerScenarioReported = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackScenario="
$debugListenerCallbackProgramRequired = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackRequired=") -eq "True"
$debugListenerCallbackStatus = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackRuntime="
$debugListenerCallbackEvidenceScope = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackEvidenceScope="
$debugListenerAttachedText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackAttached="
$debugListenerNativeVTableInstalledText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackNativeVTableInstalled="
$debugListenerInvocationCountText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackInvocationCount="
$debugListenerFailureCountText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackFailureCount="
$debugListenerInFlightCallbackCountText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackInFlightCallbackCount="
$debugListenerDetachCountText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackDetachCount="
$debugListenerInvocationCount = 0L
$debugListenerFailureCount = 0L
$debugListenerInFlightCallbackCount = 0L
$debugListenerDetachCount = 0L
$debugListenerInvocationCountParsed = [long]::TryParse($debugListenerInvocationCountText, [ref]$debugListenerInvocationCount)
$debugListenerFailureCountParsed = [long]::TryParse($debugListenerFailureCountText, [ref]$debugListenerFailureCount)
$debugListenerInFlightCallbackCountParsed = [long]::TryParse($debugListenerInFlightCallbackCountText, [ref]$debugListenerInFlightCallbackCount)
$debugListenerDetachCountParsed = [long]::TryParse($debugListenerDetachCountText, [ref]$debugListenerDetachCount)
$callbackStateSnapshotCompleteText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotComplete="
$callbackStateSnapshotIsCompleteText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotIsComplete="
$callbackStateSnapshotLastStatus = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotLastStatus="
$callbackStateSnapshotLastOperation = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotLastOperation="
$callbackStateSnapshotDiagnostic = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotDiagnostic="
$callbackStateSnapshotObserved = -not [string]::IsNullOrWhiteSpace($callbackStateSnapshotCompleteText) -and
  -not [string]::IsNullOrWhiteSpace($callbackStateSnapshotIsCompleteText) -and
  -not [string]::IsNullOrWhiteSpace($callbackStateSnapshotLastStatus)
$callbackStateSnapshotCoherent = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotCoherent=") -eq "True"
$debugListenerCallbackRuntimePassed =
  $debugListenerCallbackRequired -and
  $debugListenerCallbackProgramRequired -and
  $debugListenerCallbackStatus -eq "Passed" -and
  $debugListenerCallbackEvidenceScope -eq "local-package" -and
  $debugListenerAttachedText -eq "True" -and
  $debugListenerNativeVTableInstalledText -eq "True" -and
  $debugListenerInvocationCountParsed -and $debugListenerInvocationCount -gt 0 -and
  $debugListenerFailureCountParsed -and $debugListenerFailureCount -eq 0 -and
  $debugListenerInFlightCallbackCountParsed -and $debugListenerInFlightCallbackCount -eq 0 -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackTensorName=") -eq "output" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackShape=") -eq "[1,4]" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackMetadataCopied=") -eq "True" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackBorrowedPointerExposed=") -eq "False" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackClearReturned=") -eq "True" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackDetached=") -eq "True" -and
  $debugListenerDetachCountParsed -and $debugListenerDetachCount -gt 0 -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackIsRealRuntimeProof=") -eq "True"
$debugListenerNegativeCommonPassed =
  $debugListenerCallbackRequired -and
  $debugListenerCallbackProgramRequired -and
  $debugListenerScenarioReported -eq $DebugListenerScenario -and
  $debugListenerCallbackStatus -eq "Failed" -and
  $debugListenerCallbackEvidenceScope -eq "local-package" -and
  $callbackStateSnapshotCoherent -and
  $debugListenerAttachedText -eq "True" -and
  $debugListenerNativeVTableInstalledText -eq "True" -and
  $debugListenerInFlightCallbackCountParsed -and $debugListenerInFlightCallbackCount -eq 0 -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackBorrowedPointerExposed=") -eq "False" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackClearReturned=") -eq "True" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackDetached=") -eq "True" -and
  $debugListenerDetachCountParsed -and $debugListenerDetachCount -gt 0 -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackIsRealRuntimeProof=") -eq "False" -and
  (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerNegativeControlPassed=") -eq "True"
$combinedRuntimeOutput = @($stdoutLines + $stderrLines) -join "`n"
$missingVendorDependencyObserved =
  $missingVendorDependencyIsolationApplied -and
  $exitCode -ne 0 -and
  $combinedRuntimeOutput -match 'DllNotFoundException|BadImageFormatException|Unable to load DLL|specified module could not be found|找不到指定的模块|structured exception with code\s+3228369022'
$negativeControlPassed = switch ($DebugListenerScenario) {
  "callback-return-false" {
    $debugListenerNegativeCommonPassed -and
    $debugListenerInvocationCountParsed -and $debugListenerInvocationCount -gt 0 -and
    $debugListenerFailureCountParsed -and $debugListenerFailureCount -gt 0 -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackMetadataCopied=") -eq "True" -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackLastCallbackSucceeded=") -eq "False" -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackManagedHandlerOutcome=") -eq "returned-false"
    break
  }
  "callback-throw" {
    $debugListenerNegativeCommonPassed -and
    $debugListenerInvocationCountParsed -and $debugListenerInvocationCount -gt 0 -and
    $debugListenerFailureCountParsed -and $debugListenerFailureCount -gt 0 -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackMetadataCopied=") -eq "True" -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackLastCallbackSucceeded=") -eq "False" -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackManagedHandlerOutcome=") -eq "threw-invalid-operation"
    break
  }
  "attempted-no-invocation" {
    $debugListenerNegativeCommonPassed -and
    $debugListenerInvocationCountParsed -and $debugListenerInvocationCount -eq 0 -and
    $debugListenerFailureCountParsed -and $debugListenerFailureCount -eq 0 -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackMetadataCopied=") -eq "False" -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerDebugTensorMarked=") -eq "False" -and
    (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackManagedHandlerOutcome=") -eq "not-invoked" -and
    $exitCode -eq 0 -and
    $combinedRuntimeOutput -match 'RuntimeSmoke=ExpectedFailureVerified'
    break
  }
  "missing-vendor-dependency" { $missingVendorDependencyObserved; break }
  default { $false; break }
}
$runtimeSmokePassed = -not $negativeControlRequested -and $exitCode -eq 0 -and
  (($stdoutLines -join "`n") -match "RuntimeSmoke=Passed") -and
  (($stdoutLines -join "`n") -match "EnqueueCompleted=True") -and
  (($stdoutLines -join "`n") -match "IdentityOutputMatch=True") -and
  $callbackStateSnapshotCoherent -and
  (-not $debugListenerCallbackRequired -or $debugListenerCallbackRuntimePassed)
$identityOutputMatch = (($stdoutLines -join "`n") -match "IdentityOutputMatch=True")
$enqueueCompleted = (($stdoutLines -join "`n") -match "EnqueueCompleted=True")
$serializedBytesText = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "EngineSerializedBytes="
$engineSerializedBytes = 0L
[void][long]::TryParse($serializedBytesText, [ref]$engineSerializedBytes)
Write-Host "PostRuntimeCollection=host-metadata"
$nvidia = Get-NvidiaHostMetadata
Write-Host "PostRuntimeCollection=native-assets"
$nativeAssets = if ($SkipInstalledVendorAssetHashing.IsPresent) {
  @()
}
else {
  @(Get-NativeAssetEvidence -BridgeOutputPath $bridgeOutputPath -Roots $runtimeRoots)
}
Write-Host "PostRuntimeCollection=package-sources"
$managedPackageSourceEvidence = Get-PackageSourceEvidence -Package $managedPackage -SourceDirectory $ManagedPackageDirectory -SourceName "jyppx-managed-local"
$bridgePackageSourceEvidence = Get-PackageSourceEvidence -Package $bridgePackage -SourceDirectory $BridgePackageDirectory -SourceName "jyppx-bridge-local"
$usesProjectReference = $project -match '<ProjectReference(?:\s|>)'
$usesDirectAssemblyReference = $project -match '<Reference(?:\s|>)'
$packageReferenceCount = ([regex]::Matches($project, '<PackageReference(?:\s|>)')).Count
$usesRepositorySourceProbe =
  $program.Contains($RepositoryRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
  $program -match 'RepositoryPaths|RepositoryRoot|src[\\/]JYPPX'
$usesPackageReferenceOnly =
  $packageReferenceCount -eq 2 -and
  -not $usesProjectReference -and
  -not $usesDirectAssemblyReference -and
  -not $usesRepositorySourceProbe
$isLocalPackageDebugListenerCallbackRuntimeProof =
  -not $negativeControlRequested -and
  $debugListenerCallbackRuntimePassed -and
  $consumerRootOutsideRepository -and
  $usesPackageReferenceOnly -and
  $managedPackageSourceEvidence.isLocalPackageFeed -and
  $bridgePackageSourceEvidence.isLocalPackageFeed -and
  -not $SkipInstalledVendorAssetHashing.IsPresent
$additionalPackageSources = @(
  @(Expand-KeyList -Values $AdditionalPackageSource) |
    ForEach-Object { Resolve-PackageSourceValue -Source $_ }
)
$runtimeCreateDiagnostic = [ordered]@{
  available = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeDiagnosticAvailable=") -eq "True"
  attempted = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeAttempted=") -eq "True"
  returnedNull = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeReturnedNull=") -eq "True"
  returnedNonNull = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeReturnedNonNull=") -eq "True"
  lastStatus = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeLastStatus="
  detectedVersion = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeDetectedVersion="
  loggerCallbackAvailable = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeLoggerCallbackAvailable=") -eq "True"
  loggerMessageCount = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeLoggerMessageCount="
  lastLoggerSeverity = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeLastLoggerSeverity="
  lastLoggerMessage = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeLastLoggerMessage="
  phase = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimePhase="
  nativeDetail = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeNativeDetail="
  diagnosticMessage = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "NativeCreateRuntimeDiagnosticMessage="
}

$cudaPreflight = [ordered]@{
  available = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightAvailable=") -eq "True"
  attempted = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightAttempted=") -eq "True"
  driverVersion = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightDriverVersion="
  runtimeVersion = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightRuntimeVersion="
  deviceCount = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightDeviceCount="
  selectedDevice = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightSelectedDevice="
  deviceName = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightDeviceName="
  getDeviceCountStatus = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightGetDeviceCountStatus="
  initStatus = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightInitStatus="
  lastErrorName = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightLastErrorName="
  lastErrorMessage = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightLastErrorMessage="
  canAttemptTensorRtRuntimeCreate = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CudaPreflightCanAttemptTensorRtRuntimeCreate=") -eq "True"
  proofBoundary = "CUDA preflight is compatible-host diagnostic evidence captured before TensorRT createInferRuntime. It is not public clean package-consumer proof and cannot promote runtime proof by itself."
}

$result = [ordered]@{
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  sourceRuntimeKey = $SourceRuntimeKey
  bridgePackageKey = [string]$bridgeDefinition.key
  proofClassification = if ($negativeControlPassed) { "local-package-debug-listener-negative-control" } elseif ($SkipInstalledVendorAssetHashing.IsPresent) { "compatible-host-bridge-package-runtime-diagnostic" } elseif ($runtimeSmokePassed) { "compatible-host-bridge-package-runtime" } else { "compatible-host-bridge-package-runtime-failed" }
  smokeStatus = if ($negativeControlPassed) { "expected-failure-verified" } elseif ($runtimeSmokePassed) { "passed" } else { "failed" }
  exitCode = $exitCode
  engineSerializedBytes = $engineSerializedBytes
  enqueueCompleted = $enqueueCompleted
  identityOutputMatch = $identityOutputMatch
  isRuntimeExecutionProof = $runtimeSmokePassed
  isPackageConsumerRuntimeProof = $false
  isLocalPackageDebugListenerCallbackRuntimeProof = $isLocalPackageDebugListenerCallbackRuntimeProof
  canPromoteCompatibleHostRuntimeProof = $runtimeSmokePassed -and -not $SkipInstalledVendorAssetHashing.IsPresent
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  installedVendorAssetHashingSkipped = $SkipInstalledVendorAssetHashing.IsPresent
  installedVendorAssetInventorySkipped = $SkipInstalledVendorAssetHashing.IsPresent
  nativeAssetHashesComplete = -not $SkipInstalledVendorAssetHashing.IsPresent
  cudaPreflight = $cudaPreflight
  runtimeCreateDiagnostic = $runtimeCreateDiagnostic
  callbackStateSnapshot = [ordered]@{
    observed = $callbackStateSnapshotObserved
    complete = $callbackStateSnapshotCompleteText -eq "True"
    snapshotIsComplete = $callbackStateSnapshotIsCompleteText -eq "True"
    coherent = $callbackStateSnapshotCoherent
    lastStatus = $callbackStateSnapshotLastStatus
    lastOperation = $callbackStateSnapshotLastOperation
    hasOutputAllocator = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotHasOutputAllocator=") -eq "True"
    hasTemporaryStorageAllocator = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotHasTemporaryStorageAllocator=") -eq "True"
    hasDebugListener = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "CallbackStateSnapshotHasDebugListener=") -eq "True"
    diagnostic = $callbackStateSnapshotDiagnostic
    pointerFree = $callbackStateSnapshotObserved -and $callbackStateSnapshotCoherent
    proofBoundary = "A coherent complete or partial callback-state snapshot is package runtime diagnostic evidence. It is not callback invocation, public-package, or post-publish proof."
  }
  negativeControl = [ordered]@{
    scenario = $DebugListenerScenario
    requested = $negativeControlRequested
    passed = $negativeControlPassed
    programReportedScenario = $debugListenerScenarioReported
    programReportedPassed = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerNegativeControlPassed=") -eq "True"
    missingVendorDependencyIsolationApplied = $missingVendorDependencyIsolationApplied
    missingVendorDependencyObserved = $missingVendorDependencyObserved
    knownModuleNotFoundStructuredExceptionCode = 3228369022
    proofBoundary = "A passing negative control proves only that the selected failure was observed and rejected. It is not runtime, local-package, public-package, or post-publish proof."
  }
  debugListenerCallback = [ordered]@{
    status = if ($negativeControlPassed) { "expected-failure-verified" } elseif ($debugListenerCallbackRuntimePassed) { "passed" } elseif ($debugListenerCallbackRequired) { "failed" } else { "unsupported-tensorrt-line" }
    evidenceKind = if ($negativeControlRequested) { "debug-listener-callback-negative-control" } else { "debug-listener-real-callback-runtime" }
    evidenceScope = "local-package"
    scenario = $DebugListenerScenario
    requiredForTensorRtLine = $debugListenerCallbackRequired
    programReportedRequired = $debugListenerCallbackProgramRequired
    runtimeInvariantsSatisfied = $debugListenerCallbackRuntimePassed
    attached = $debugListenerAttachedText -eq "True"
    nativeVTableInstalled = $debugListenerNativeVTableInstalledText -eq "True"
    invocationCount = $debugListenerInvocationCount
    failureCount = $debugListenerFailureCount
    inFlightCallbackCount = $debugListenerInFlightCallbackCount
    tensorName = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackTensorName="
    shape = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackShape="
    metadataCopied = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackMetadataCopied=") -eq "True"
    borrowedPointerExposed = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackBorrowedPointerExposed=") -eq "True"
    clearReturned = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackClearReturned=") -eq "True"
    detached = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackDetached=") -eq "True"
    detachCount = $debugListenerDetachCount
    isRealCallbackRuntimeProof = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackIsRealRuntimeProof=") -eq "True"
    lastCallbackSucceeded = (Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackLastCallbackSucceeded=") -eq "True"
    lastStatus = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackLastStatus="
    managedHandlerOutcome = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackManagedHandlerOutcome="
    executionExceptionType = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "DebugListenerCallbackExecutionExceptionType="
    isLocalPackageCallbackRuntimeProof = $isLocalPackageDebugListenerCallbackRuntimeProof
    proofBoundary = "This callback proof is scoped to local managed and bridge packages restored into an external PackageReference-only consumer. It is not source-tree, public-package, or post-publish proof."
  }
  proofScopes = [ordered]@{
    sourceTree = [ordered]@{
      evidenceScope = "source-tree"
      isProof = $false
      boundary = "This report executes packaged assemblies and does not claim source-tree callback proof."
    }
    localPackage = [ordered]@{
      evidenceScope = "local-package"
      isProof = $isLocalPackageDebugListenerCallbackRuntimeProof
      boundary = "Local package callback proof requires an external PackageReference-only consumer plus real invocation, copied metadata, pointer-free reporting, zero failure/in-flight callbacks, and successful detach."
    }
    publicPackage = [ordered]@{
      evidenceScope = "public-package"
      isProof = $false
      boundary = "Local package directories cannot prove a public package channel."
    }
    postPublish = [ordered]@{
      evidenceScope = "post-publish"
      isProof = $false
      boundary = "This pre-publish local package run cannot prove post-publish installation or execution."
    }
  }
  runtimeProofBoundary = if ($negativeControlRequested) { "This record is a controlled fail-closed negative test. Even when the expected failure is verified, runtime, local-package, public-package, and post-publish proof remain false." } elseif ($SkipInstalledVendorAssetHashing.IsPresent) { "Diagnostic execution may record runtime and DebugListener callback behavior but skips installed vendor asset SHA256 values and cannot promote compatible-host or local-package callback runtime proof. Local package feeds are not public clean package-consumer proof." } else { "This record proves local managed plus bridge packages can build, serialize, deserialize, enqueue, verify an identity network, and complete a real DebugListener attach/invoke/detach cycle with system-installed vendor dependencies on this compatible host. The callback result is local-package proof only; it is not source-tree, public-package, or post-publish proof and does not clear other runtime lines." }
  packages = [ordered]@{
    managed = [ordered]@{
      id = $managedPackage.Id
      version = $managedPackage.Version
      path = $managedPackage.Path
      sha256 = $managedPackage.Sha256
      source = $managedPackageSourceEvidence
    }
    bridge = [ordered]@{
      id = $bridgePackage.Id
      version = $bridgePackage.Version
      path = $bridgePackage.Path
      sha256 = $bridgePackage.Sha256
      source = $bridgePackageSourceEvidence
    }
  }
  packageSources = [ordered]@{
    managed = $managedPackageSourceEvidence.sourceDirectory
    bridge = $bridgePackageSourceEvidence.sourceDirectory
    additional = @($additionalPackageSources)
    nugetOrgEnabledForTransitiveDependencies = $true
    usesLocalPackageFeed = $managedPackageSourceEvidence.isLocalPackageFeed -or $bridgePackageSourceEvidence.isLocalPackageFeed
    isPublicCleanPackageConsumerProof = $false
    isPostPublishCleanConsumerProof = $false
  }
  consumer = [ordered]@{
    root = $consumerRoot
    projectPath = $projectPath
    projectSha256 = (Get-FileHash -LiteralPath $projectPath -Algorithm SHA256).Hash
    targetFramework = $TargetFramework
    runtimeIdentifier = $rid
    consumerRootOutsideRepository = $consumerRootOutsideRepository
    packageReferenceCount = $packageReferenceCount
    usesPackageReferenceOnly = $usesPackageReferenceOnly
    usesProjectReference = $usesProjectReference
    usesDirectAssemblyReference = $usesDirectAssemblyReference
    usesRepositorySourceProbe = $usesRepositorySourceProbe
    outputDirectory = $outputDirectory
    outputPreserved = $KeepConsumerOutput.IsPresent
  }
  consumerRootOutsideRepository = $consumerRootOutsideRepository
  roots = [ordered]@{
    tensorRtRoot = $runtimeRoots.TensorRtRoot
    cudaRoot = $runtimeRoots.CudaRoot
    cudnnRoot = $runtimeRoots.CudnnRoot
    searchDirectories = @($runtimeSearchDirectories)
  }
  host = [ordered]@{
    machineName = [Environment]::MachineName
    osDescription = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    processArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
    gpuName = $nvidia.gpuName
    driverVersion = $nvidia.driverVersion
    nvidiaSmiAvailable = $nvidia.available
    nvidiaSmiDiagnostic = $nvidia.diagnostic
    runtimeEnvironmentLine = Get-FirstMarkerValue -Lines $stdoutLines -Prefix "RuntimeEnvironment "
  }
  logs = [ordered]@{
    stdoutPath = $stdoutPath
    stdoutSha256 = (Get-FileHash -LiteralPath $stdoutPath -Algorithm SHA256).Hash
    stderrPath = $stderrPath
    stderrSha256 = (Get-FileHash -LiteralPath $stderrPath -Algorithm SHA256).Hash
    combinedPath = $combinedPath
    combinedSha256 = (Get-FileHash -LiteralPath $combinedPath -Algorithm SHA256).Hash
    stdoutSummary = @($stdoutLines)
    stderrSummary = @($stderrLines)
  }
  nativeAssets = @($nativeAssets)
}

Write-Host "PostRuntimeCollection=serialize-report"
Write-Reports -Result ([pscustomobject]$result) -Directory $ReportDirectory
Write-Host "PostRuntimeCollection=report-written"

if (-not $KeepConsumerOutput.IsPresent) {
  Remove-ConsumerDirectory -Path $consumerRoot -AllowedRoot $OutputRoot
  Write-Host "Removed clean external runtime consumer output: $consumerRoot"
}

if ($negativeControlRequested -and -not $negativeControlPassed) {
  throw "Bridge package DebugListener negative control failed closed validation. Scenario=$DebugListenerScenario. See $ReportDirectory."
}
if (-not $negativeControlRequested -and -not $runtimeSmokePassed -and -not $AllowRuntimeSmokeFailure.IsPresent) {
  throw "Bridge package runtime smoke failed. See $ReportDirectory."
}

Write-Host "Bridge package runtime consumer completed: $($result.proofClassification)"
