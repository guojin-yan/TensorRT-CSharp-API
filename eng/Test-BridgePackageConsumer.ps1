[CmdletBinding()]
param(
  [string]$SourceRuntimeKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$BridgePackageKey,
  [string]$TargetFramework = "net8.0",
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string[]]$AdditionalPackageSource = @(),
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [switch]$SkipProbe,
  [switch]$KeepConsumerOutput,
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
  param(
    [string[]]$Values
  )

  $keys = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $keys.Add($trimmed)
      }
    }
  }

  return @($keys | Select-Object -Unique)
}

function ConvertTo-XmlAttributeValue {
  param(
    [string]$Value
  )

  return [System.Security.SecurityElement]::Escape($Value)
}

function Resolve-PackageSourceValue {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Source
  )

  if ($Source -match '^[a-zA-Z][a-zA-Z0-9+.-]*://') {
    return $Source
  }

  if ([System.IO.Path]::IsPathRooted($Source)) {
    return [System.IO.Path]::GetFullPath($Source)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Source))
}

function Join-PathMany {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Parts
  )

  if ($Parts.Count -eq 0) {
    throw "At least one path part is required."
  }

  $path = $Parts[0]
  for ($i = 1; $i -lt $Parts.Count; $i++) {
    $path = Join-Path $path $Parts[$i]
  }

  return $path
}

function New-NuGetConfigContent {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ManagedSource,
    [Parameter(Mandatory = $true)]
    [string]$BridgeSource
  )

  $packageSources = New-Object System.Collections.Generic.List[string]
  $packageSources.Add('    <clear />')
  $packageSources.Add('    <add key="jyppx-managed" value="' + (ConvertTo-XmlAttributeValue -Value $ManagedSource) + '" />')
  $packageSources.Add('    <add key="jyppx-bridge" value="' + (ConvertTo-XmlAttributeValue -Value $BridgeSource) + '" />')

  $sourceIndex = 1
  foreach ($source in @(Expand-KeyList -Values $AdditionalPackageSource)) {
    $resolvedSource = Resolve-PackageSourceValue -Source $source
    $packageSources.Add('    <add key="additional-' + $sourceIndex + '" value="' + (ConvertTo-XmlAttributeValue -Value $resolvedSource) + '" />')
    $sourceIndex++
  }

  return @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
$($packageSources -join "`r`n")
  </packageSources>
</configuration>
"@
}

function Get-NupkgMetadata {
  param(
    [string]$Path
  )

  $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
  try {
    $nuspec = $zip.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
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
    $id = $xml.SelectSingleNode("//n:metadata/n:id", $namespaceManager).InnerText
    $version = $xml.SelectSingleNode("//n:metadata/n:version", $namespaceManager).InnerText
    return [pscustomobject]@{
      Path = $Path
      Id = $id
      Version = $version
      LastWriteTime = (Get-Item -LiteralPath $Path).LastWriteTime
    }
  }
  finally {
    $zip.Dispose()
  }
}

function Find-Package {
  param(
    [string]$Directory,
    [string]$PackageId
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    throw "Package directory does not exist: $Directory"
  }

  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($package in Get-ChildItem -LiteralPath $Directory -Filter *.nupkg) {
    $metadata = Get-NupkgMetadata -Path $package.FullName
    if ($metadata.Id -eq $PackageId) {
      $matches.Add($metadata)
    }
  }

  if ($matches.Count -eq 0) {
    throw "Package '$PackageId' was not found under $Directory."
  }

  return @($matches | Sort-Object LastWriteTime, Version -Descending)[0]
}

$script:ManagedPackageFreshnessPackCommand = "dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj -c Debug -o .\artifacts\managed -p:JYPPXPackageVersion=4.0.0 /p:UseSharedCompilation=false"
$script:ManagedPackageFreshnessRequiredMarkers = @(
  "TensorRtAuxiliaryStreamAssignmentSnapshot",
  "GetAuxiliaryStreamAssignmentSnapshot",
  "SetAuxStreams",
  "ClearAuxStreams",
  "TensorRtPluginV2LayerMetadata",
  "GetPluginV2Metadata",
  "TryGetPluginV2Metadata",
  "OutputCount",
  "HasExtCapability",
  "HasIoExtCapability",
  "HasDynamicExtCapability",
  "GetPluginV2LegacyOutputDimensions",
  "GetPluginV2LegacyWorkspaceSize",
  "SupportsPluginV2LegacyFormat",
  "GetPluginV2OutputDataType",
  "CanPluginV2BroadcastInputAcrossBatch",
  "IsPluginV2OutputBroadcastAcrossBatch",
  "TensorRtPluginFormatSupportSnapshot",
  "GetPluginV2DynamicFormatSupportSnapshot",
  "GetPluginV2IoExtFormatSupportSnapshot",
  "TensorRtPluginV3BuildIoSnapshot",
  "GetPluginV3BuildIoSnapshot",
  "TensorRtPluginV3SerializationFieldInventory",
  "GetPluginV3RuntimeSerializationFields",
  "TensorRtPluginV3LayerMetadata",
  "GetPluginV3Metadata",
  "TryGetPluginV3Metadata",
  "TensorRtDebugListenerNativeAttachBridgeShapeGate",
  "TensorRtDebugListenerExceptionStatusMappingGate",
  "TensorRtDebugListenerInFlightAccountingGate",
  "TensorRtDebugListenerNativeNoThrowVTableScaffoldGate",
  "TensorRtDebugListenerNoThrowVTableCallbackStub",
  "TensorRtDebugListenerNoThrowVTableCallbackStubResult",
  "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate",
  "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult",
  "TensorRtDebugListenerNativeVTableInstallPreflight",
  "TensorRtDebugListenerNativeVTableInstallPreflightResult",
  "TensorRtDebugListenerNativeOwnerVTableInstallExperiment",
  "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult",
  "TensorRtRuntimeDeserializationBoundaryPrecheck",
  "TensorRtRuntimeDeserializationBoundaryPrecheckResult",
  "TensorRtOnnxParserDiagnosticSnapshot",
  "TensorRtOnnxParserRefitterDiagnosticSnapshot",
  "TensorRtOnnxParserDiagnosticSummary",
  "TensorRtOnnxParserRefitterDiagnosticSummary",
  "TensorRtRuntimeDiagnosticSummary",
  "TensorRtEngineDeploymentSummary",
  "TensorRtBuilderConfigDeploymentSummary",
  "TensorRtExecutionContextDeploymentSummary",
  "TensorRtSerializationConfigSummary",
  "TensorRtRuntimeConfigSummary",
  "TensorRtRefitterDiagnosticSummary",
  "TensorRtExecutionContextRuntimeDiagnosticSummary",
  "TensorRtExecutionContextCallbackAllocatorSafeControlSummary",
  "GetCallbackAllocatorSafeControlSummary",
  "CopiedInterfaceInfoCount",
  "CallbackInvocationAttempted",
  "CudaGraphDiagnosticSummary",
  "CudaGraphExecDiagnosticSummary",
  "CudaDevice.GetGraphMemorySummary",
  "CudaDevice.CurrentGraphMemorySummary",
  "CudaDeviceGraphMemorySummary",
  "CudaMemoryRangeDiagnosticSummary",
  "ManagedByteArrayDeserializeReady",
  "ManagedStreamDeserializeReady",
  "HostMemoryDeserializeReady",
  "SerializedBufferCopiedBeforeInterop",
  "EngineHandleOwnedByWrapper",
  "LoadRuntimeDeferred",
  "CallbackStubGateReady",
  "CallbackStubNoThrowReady",
  "CallbackMetadataCopyReady",
  "MetadataGateReady",
  "TensorNameCopied",
  "TensorNameLength",
  "TensorTypeCopied",
  "TensorLocationCopied",
  "TensorShapeCopied",
  "TensorFlagsCopied",
  "BorrowedDebugTensorDataPointerEscapeBlocked",
  "BorrowedDebugTensorMetadataGateReady",
  "NativeVTableInstallPreflightReady",
  "ExperimentShapeReady",
  "InstallAttemptGuardReady",
  "NonNullAttachEnabled",
  "RuntimeProofEnabled",
  "NativeVTableInstallAttempted",
  "RollbackReady",
  "DetachBeforeReleaseReady",
  "FailureStatusMappingReady",
  "PointerFree",
  "ReasonNativeOwnerVTableInstallStillBlocked",
  "VTableInstallShapeReady",
  "VTableInstallVersionGuardReady",
  "VTableInstallNoThrowBoundaryReady",
  "VTableInstallOwnershipDiagnosticsReady",
  "VTableInstallPointerFree",
  "NativeVTableInstallRuntimeReady",
  "ReasonNativeVTableInstallStillBlocked",
  "BorrowedDebugTensorLifetimeReady",
  "BorrowedDebugTensorDataLifetimeReady",
  "ReasonMetadataRuntimeStillBlocked",
  "CallbackExceptionCaptureReady",
  "CallbackStatusMappingReady",
  "CallbackInFlightPairingReady",
  "DebugTensorDataPointerExposed",
  "NativeVTableInstalled",
  "ReasonCallbackRuntimeStillBlocked",
  "TensorRtDebugListenerNativeAttachEntryMinimalSafety",
  "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult",
  "MinimalSafetyReady",
  "RuntimeScaffoldReady",
  "LifecycleGateReady",
  "NativeAttachEntryLocated",
  "SetDebugListenerNonNullEnabled",
  "NonNullAttachStillDisabled",
  "NativeAttachWouldBeBlocked",
  "ReasonNativeAttachStillBlocked",
  "TensorRtDebugListenerRuntimeProofPrecheck",
  "TensorRtDebugListenerRuntimeProofAttemptPreflight",
  "TensorRtDebugListenerRuntimeProofAttemptPreflightResult",
  "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke",
  "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult",
  "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline",
  "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult",
  "TensorRtDebugListenerRealCallbackRuntimeProof",
  "TensorRtDebugListenerRealCallbackRuntimeProofResult",
  "TensorRtDebugListenerCallbackProofGapReport",
  "TensorRtDebugListenerCallbackProofGapReportResult",
  "TensorRtCallbackOwnerClosureMatrix",
  "TensorRtCallbackOwnerClosureMatrixResult",
  "TensorRtCallbackOwnerClosureMatrixRow",
  "DesignGateReadyFamilyCount",
  "ClosureReadyFamilyCount",
  "RuntimeProofAttemptReadyFamilyCount",
  "PackageConsumerRuntimeProofReadyFamilyCount",
  "TensorRtDebugTensorMetadataSnapshot",
  "CanEnableSetDebugListenerNonNull",
  "CanInstallNativeVTable",
  "CanCallProcessDebugTensorRuntime",
  "CanPromoteRealCallbackRuntime",
  "ReasonNonNullAttachStillBlocked",
  "ReasonNativeVTableStillBlocked",
  "ReasonRuntimeProofStillBlocked",
  "OptInEnabled",
  "AttachGuardReady",
  "AttachAttempted",
  "AttachSucceeded",
  "DetachAttempted",
  "DetachSucceeded",
  "RollbackAttempted",
  "RollbackSucceeded",
  "ProcessDebugTensorInvoked",
  "InvocationCount",
  "AllocationCount",
  "ReleaseCount",
  "InFlightCallbackCount",
  "NonNullAttachStillDisabled",
  "NativeAttachEntryReady",
  "NativeVTableInstallBlocked",
  "NoThrowCallbackEntryReady",
  "ExceptionStatusMappingReady",
  "InFlightAccountingReady",
  "BorrowedDebugTensorMetadataCopied",
  "DetachRollbackReady",
  "FullPackageConsumerRuntimeProofReady",
  "GapReasonCount",
  "PrimaryGapReason",
  "RuntimeProofBlockerCategory",
  "PackageConsumerRuntimeProofRequired",
  "RuntimeInvocationRequired",
  "EvidenceSource",
  "NextOwnerAction",
  "LastDiagnostic",
  "FullPackageConsumerReport",
  "ReportPointerFree",
  "TrampolineShapeReady",
  "NativeCallbackEntryLocated",
  "NoThrowCallbackEntryReady",
  "ExceptionCaptureReady",
  "InFlightAccountingReady",
  "PointerFreeSurfaceReady",
  "CallbackStubEntryCount",
  "CallbackStubLeaveCount"
)

function Get-ManagedPackageXmlSurface {
  param(
    [Parameter(Mandatory = $true)]
    [object]$ManagedPackage
  )

  $zip = [System.IO.Compression.ZipFile]::OpenRead($ManagedPackage.Path)
  try {
    $xmlEntries = @($zip.Entries | Where-Object {
        $_.FullName.EndsWith(".xml", [System.StringComparison]::OrdinalIgnoreCase) -and
        $_.FullName.Contains("/JYPPX.", [System.StringComparison]::OrdinalIgnoreCase) -and
        $_.FullName.StartsWith("lib/", [System.StringComparison]::OrdinalIgnoreCase)
      })
    if ($xmlEntries.Count -eq 0) {
      throw "Managed package appears stale: '$($ManagedPackage.Id)' $($ManagedPackage.Version) at '$($ManagedPackage.Path)' does not contain lib/*/JYPPX.*.xml documentation surfaces, for example JYPPX.TensorRtSharp.xml and JYPPX.CudaSharp.xml. Repack the managed package with: $script:ManagedPackageFreshnessPackCommand"
    }

    $builder = [System.Text.StringBuilder]::new()
    foreach ($entry in @($xmlEntries)) {
      $stream = $entry.Open()
      try {
        $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
        try {
          [void]$builder.AppendLine($entry.FullName)
          [void]$builder.AppendLine($reader.ReadToEnd())
        }
        finally {
          $reader.Dispose()
        }
      }
      finally {
        $stream.Dispose()
      }
    }

    return $builder.ToString()
  }
  finally {
    $zip.Dispose()
  }
}

function Assert-ManagedPackageFreshness {
  param(
    [Parameter(Mandatory = $true)]
    [object]$ManagedPackage,
    [string[]]$RequiredMarkers = $script:ManagedPackageFreshnessRequiredMarkers
  )

  $surface = Get-ManagedPackageXmlSurface -ManagedPackage $ManagedPackage
  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($RequiredMarkers)) {
    if ($surface.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  if ($missingMarkers.Count -gt 0) {
    throw "Managed package appears stale: '$($ManagedPackage.Id)' $($ManagedPackage.Version) at '$($ManagedPackage.Path)' is missing public API marker(s): $($missingMarkers -join ', '). Repack the managed package with: $script:ManagedPackageFreshnessPackCommand"
  }

  Write-Host "Managed package freshness validated: $($ManagedPackage.Id) $($ManagedPackage.Version)"
}

function Test-NupkgContainsBridgeAsset {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackagePath,
    [Parameter(Mandatory = $true)]
    [string]$Rid,
    [Parameter(Mandatory = $true)]
    [string]$BridgeFileName
  )

  $expected = "runtimes/$Rid/native/$BridgeFileName"
  $zip = [System.IO.Compression.ZipFile]::OpenRead($PackagePath)
  try {
    foreach ($entry in $zip.Entries) {
      $entryName = $entry.FullName.Replace('\', '/')
      if ([string]::Equals($entryName, $expected, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
      }
    }
  }
  finally {
    $zip.Dispose()
  }

  return $false
}

function Invoke-DotNetCommand {
  param(
    [string[]]$Arguments
  )

  $dotnetOutput = & dotnet @Arguments 2>&1
  foreach ($line in @($dotnetOutput)) {
    Write-Host $line
  }

  return [pscustomobject]@{
    ExitCode = $LASTEXITCODE
    OutputLines = @($dotnetOutput)
  }
}

function Invoke-CheckedDotNet {
  param(
    [string[]]$Arguments
  )

  $result = Invoke-DotNetCommand -Arguments $Arguments
  if ($result.ExitCode -ne 0) {
    throw "dotnet $($Arguments -join ' ') failed with exit code $($result.ExitCode)."
  }
}

function Test-ApplicationControlPolicyBlock {
  param(
    [string[]]$OutputLines
  )

  $text = ($OutputLines -join "`n")
  return (
    $text -match '0x800711C7' -or
    $text -match 'application control policy' -or
    $text -match '应用程序控制策略'
  )
}

function Get-FirstProbeOutputLine {
  param(
    [string[]]$OutputLines,
    [Parameter(Mandatory = $true)]
    [string]$Prefix
  )

  foreach ($line in @($OutputLines)) {
    $text = [string]$line
    if ($text.StartsWith($Prefix, [System.StringComparison]::Ordinal)) {
      return $text
    }
  }

  return ""
}

function Get-NativeDependencyProbeClassification {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ProbeResult,
    [string[]]$OutputLines
  )

  $probeText = (@($OutputLines) -join "`n")
  $dependencyProbeLine = Get-FirstProbeOutputLine -OutputLines $OutputLines -Prefix "DependencyProbe "
  $skippedLine = Get-FirstProbeOutputLine -OutputLines $OutputLines -Prefix "Skipped=True Reason="
  $skippedReason = if ([string]::IsNullOrWhiteSpace($skippedLine)) { "" } else { $skippedLine.Substring("Skipped=True Reason=".Length) }
  $vendorExceptionCode = ""
  if ($probeText -match 'structured exception with code\s+([0-9]+)') {
    $vendorExceptionCode = $Matches[1]
  }

  if ($ProbeResult -eq "not-requested") {
    return [pscustomobject]@{
      status = "not-requested"
      diagnostic = "native dependency probe was not requested."
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $false
      isVendorStructuredException = $false
    }
  }

  if ($ProbeResult -eq "blocked-by-application-control") {
    return [pscustomobject]@{
      status = "blocked-by-application-control"
      diagnostic = "probe execution was blocked by the Windows application control policy."
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $false
      isVendorStructuredException = $false
    }
  }

  if ($probeText -match "EnvironmentProbe=Succeeded") {
    return [pscustomobject]@{
      status = "ready"
      diagnostic = "environment probe succeeded."
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $false
      isVendorStructuredException = $false
    }
  }

  if (-not [string]::IsNullOrWhiteSpace($vendorExceptionCode) -or $probeText -match "SEHException" -or $probeText -match "structured exception") {
    $diagnostic = if ([string]::IsNullOrWhiteSpace($skippedReason)) { "vendor structured exception during environment probe." } else { $skippedReason }
    return [pscustomobject]@{
      status = "vendor-structured-exception"
      diagnostic = $diagnostic
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $false
      isVendorStructuredException = $true
    }
  }

  if ($probeText -match "DllNotFoundException" -or $probeText -match "Unable to load DLL" -or $probeText -match "could not be found") {
    $diagnostic = if ([string]::IsNullOrWhiteSpace($skippedReason)) { "native bridge, TensorRT, or CUDA dependency could not be loaded." } else { $skippedReason }
    return [pscustomobject]@{
      status = "missing-native-dependency"
      diagnostic = $diagnostic
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $true
      isVendorStructuredException = $false
    }
  }

  if ($probeText -match "BadImageFormatException") {
    $diagnostic = if ([string]::IsNullOrWhiteSpace($skippedReason)) { "native bridge or vendor dependency architecture did not match the process." } else { $skippedReason }
    return [pscustomobject]@{
      status = "bad-native-image"
      diagnostic = $diagnostic
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $false
      isVendorStructuredException = $false
    }
  }

  if ($ProbeResult -eq "dependency-probe-only") {
    return [pscustomobject]@{
      status = "dependency-probe-only"
      diagnostic = "bridge layout and wrapper surface were validated, but the environment probe did not produce a runtime success marker."
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $false
      isVendorStructuredException = $false
    }
  }

  if (-not [string]::IsNullOrWhiteSpace($skippedReason)) {
    return [pscustomobject]@{
      status = "skipped-with-diagnostic"
      diagnostic = $skippedReason
      dependencyProbeLine = $dependencyProbeLine
      skippedReason = $skippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = $false
      isVendorStructuredException = $false
    }
  }

  return [pscustomobject]@{
    status = "unknown"
    diagnostic = "native dependency probe did not emit a recognized success or skip marker."
    dependencyProbeLine = $dependencyProbeLine
    skippedReason = $skippedReason
    vendorExceptionCode = $vendorExceptionCode
    isNativeDependencyMissing = $false
    isVendorStructuredException = $false
  }
}

function Get-SafePathName {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  return ($Value -replace '[^A-Za-z0-9._-]', '-')
}

function Get-RestorePackagesPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeKey
  )

  $safeKey = Get-SafePathName -Value $RuntimeKey
  if ($env:OS -eq "Windows_NT") {
    $root = Join-Path $env:SystemDrive "jyppx-pkgcache"
    return Join-Path $root "bridge-$safeKey"
  }

  return [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "jyppx-pkgcache", "bridge-$safeKey")
}

function Remove-ConsumerDirectory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  $lastError = $null
  $fullPath = [System.IO.Path]::GetFullPath($Path)
  if ($fullPath.StartsWith("\\", [System.StringComparison]::Ordinal)) {
    $extendedPath = "\\?\UNC\" + $fullPath.Substring(2)
  }
  else {
    $extendedPath = "\\?\" + $fullPath
  }

  for ($attempt = 1; $attempt -le 6; $attempt++) {
    try {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
      return
    }
    catch {
      $lastError = $_
    }

    try {
      [System.GC]::Collect()
      [System.GC]::WaitForPendingFinalizers()

      if ($attempt -eq 1) {
        try {
          & dotnet build-server shutdown *> $null
        }
        catch {
        }
      }

      [System.IO.Directory]::Delete($extendedPath, $true)
      return
    }
    catch {
      $lastError = $_
      Start-Sleep -Milliseconds (250 * $attempt)
    }

    if (-not (Test-Path -LiteralPath $Path)) {
      return
    }
  }

  if (Test-Path -LiteralPath $Path) {
    $reason = if ($lastError) { $lastError.Exception.Message } else { "unknown error" }
    throw "Unable to remove package consumer directory: $Path. Last error: $reason"
  }
}

function Resolve-BridgeSplitPackage {
  param(
    [string]$SourceKey,
    [string]$SplitKey
  )

  $runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
  $splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
  $runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json

  $splitPackage = $null
  if (-not [string]::IsNullOrWhiteSpace($SplitKey)) {
    $splitPackage = $splitManifest.packages | Where-Object { $_.key -eq $SplitKey } | Select-Object -First 1
    if (-not $splitPackage) {
      throw "Bridge split package key '$SplitKey' was not found."
    }

    if ([string]$splitPackage.role -ne "bridge") {
      throw "Split package '$SplitKey' has role '$($splitPackage.role)', not 'bridge'."
    }

    if ([string]::IsNullOrWhiteSpace($SourceKey)) {
      $SourceKey = [string]$splitPackage.sourceRuntimeKey
    }
  }

  if ([string]::IsNullOrWhiteSpace($SourceKey)) {
    throw "SourceRuntimeKey or BridgePackageKey is required."
  }

  $sourcePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $SourceKey } | Select-Object -First 1
  if (-not $sourcePackage) {
    throw "Source runtime key '$SourceKey' was not found."
  }

  if (-not $splitPackage) {
    $bridgeMatches = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $SourceKey -and $_.role -eq "bridge" })
    if ($bridgeMatches.Count -ne 1) {
      throw "Expected one bridge split package for '$SourceKey', found $($bridgeMatches.Count)."
    }

    $splitPackage = $bridgeMatches[0]
  }

  if ([string]$sourcePackage.rid -ne [string]$splitPackage.rid) {
    throw "Bridge split package '$($splitPackage.key)' rid '$($splitPackage.rid)' does not match source rid '$($sourcePackage.rid)'."
  }

  return [pscustomobject]@{
    SourceRuntimeKey = [string]$SourceKey
    SourcePackage = $sourcePackage
    SplitPackage = $splitPackage
  }
}

function Invoke-BridgePackageConsumerValidation {
  param(
    [Parameter(Mandatory = $true)]
    [object]$ResolvedPackage,
    [Parameter(Mandatory = $true)]
    [object]$ManagedPackage,
    [Parameter(Mandatory = $true)]
    [object]$BridgePackage
  )

  $sourcePackage = $ResolvedPackage.SourcePackage
  $splitPackage = $ResolvedPackage.SplitPackage
  $key = [string]$splitPackage.key
  $rid = [string]$splitPackage.rid
  $bridgeFileName = [string]($splitPackage.assets | Where-Object { [string]$_ -like "jyppxtrtbridge*" -or [string]$_ -like "libjyppxtrtbridge*" } | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($bridgeFileName)) {
    throw "Bridge split package '$key' does not declare a jyppxtrtbridge asset."
  }

  if (-not (Test-NupkgContainsBridgeAsset -PackagePath $BridgePackage.Path -Rid $rid -BridgeFileName $bridgeFileName)) {
    throw "Bridge package '$($BridgePackage.Path)' does not contain runtimes/$rid/native/$bridgeFileName."
  }

  $timer = [System.Diagnostics.Stopwatch]::StartNew()
  $safeKey = Get-SafePathName -Value $key
  $consumerRoot = Join-Path $OutputRoot $safeKey
  $resolvedConsumerRoot = [System.IO.Path]::GetFullPath($consumerRoot)
  $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
  if (-not $resolvedConsumerRoot.StartsWith($resolvedOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean consumer path outside output root: $resolvedConsumerRoot"
  }

  $restorePackagesPath = $null
  try {
    if (Test-Path -LiteralPath $resolvedConsumerRoot) {
      Remove-ConsumerDirectory -Path $resolvedConsumerRoot
    }

    New-Item -ItemType Directory -Path $resolvedConsumerRoot -Force | Out-Null

    $nugetConfig = New-NuGetConfigContent -ManagedSource $ManagedPackageDirectory -BridgeSource $BridgePackageDirectory
    $project = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$TargetFramework</TargetFramework>
    <RuntimeIdentifier>$rid</RuntimeIdentifier>
    <RestorePackagesPath>`$(MSBuildProjectDirectory)\.nuget\packages</RestorePackagesPath>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="$($ManagedPackage.Id)" Version="$($ManagedPackage.Version)" />
    <PackageReference Include="$($BridgePackage.Id)" Version="$($BridgePackage.Version)" />
  </ItemGroup>
</Project>
"@

    $tensorRtLineExpression = switch ([string]$splitPackage.tensorRtLine) {
      "8" { "TensorRtApiLine.TensorRt8" }
      "10" { "TensorRtApiLine.TensorRt10" }
      "11" { "TensorRtApiLine.TensorRt11" }
      default { "TensorRtApiLine.TensorRt11" }
    }

    $program = @'
using System;
using System.IO;
using JYPPX.CudaSharp;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp;

const string Rid = "__RID__";
TensorRtApiLine probeLine = __TENSORRT_LINE__;

string bridgeFileName = NativeBridgePathResolver.GetBridgeFileName();
string outputRoot = AppContext.BaseDirectory;
string rootCandidate = Path.Combine(outputRoot, bridgeFileName);
string runtimeCandidate = Path.Combine(outputRoot, "runtimes", Rid, "native", bridgeFileName);
bool rootCopied = File.Exists(rootCandidate);
bool runtimeCopied = File.Exists(runtimeCandidate);

Console.WriteLine("TensorRtAssemblyBridge=" + TensorRtSharpInfo.NativeBridgeLibraryName);
Console.WriteLine("CudaAssemblyBridge=" + CudaSharpInfo.NativeBridgeLibraryName);
Console.WriteLine("BridgeFileName=" + bridgeFileName);
Console.WriteLine("BridgeRootCopied=" + rootCopied);
Console.WriteLine("BridgeRuntimeLayoutCopied=" + runtimeCopied);
Console.WriteLine("HighLevelWrapperSurface=" + HighLevelWrapperSurfaceProbe.CreateSummary());

if (!rootCopied && !runtimeCopied)
{
    Console.WriteLine("Failed=True Reason=Bridge asset was not copied to the consumer output.");
    return 2;
}

TensorRtDependencyProbeReport dependencyProbe = TensorRtEnvironmentProbe.ProbeNativeDependencies(probeLine);
Console.WriteLine("DependencyProbe BridgeInitialized=" + dependencyProbe.BridgeInitialized +
    " Candidates=" + dependencyProbe.NativeBridgeCandidates.Count +
    " Loaded=" + dependencyProbe.LoadedModuleCount +
    " SearchPathCandidates=" + dependencyProbe.SearchPathCandidateCount +
    " Diagnostics=" + dependencyProbe.Diagnostics.Count +
    " Message=" + dependencyProbe.BridgeDiagnostic);

try
{
    TensorRtEnvironmentSnapshot snapshot = TensorRtEnvironmentProbe.GetCurrent();
    Console.WriteLine("EnvironmentProbe=Succeeded Bridge=" + snapshot.BuildInfo.BridgeName +
        " TRT=" + snapshot.BuildInfo.TensorRtVersion +
        " CUDA=" + snapshot.BuildInfo.CudaToolkitVersion +
        " RuntimeTensorRtAvailable=" + snapshot.RuntimeInfo.TensorRtAvailable +
        " RuntimeCudaAvailable=" + snapshot.RuntimeInfo.CudaToolkitAvailable);
}
catch (Exception exception)
{
    Console.WriteLine("EnvironmentProbe=Skipped");
    Console.WriteLine("Skipped=True Reason=" + exception.GetType().FullName + ": " + exception.Message.Replace(Environment.NewLine, " "));
}

return 0;

static class HighLevelWrapperSurfaceProbe
{
    public static string CreateSummary()
    {
        Func<TensorRtBuilder, TensorRtPluginRegistryInventory> pluginInventory =
            static builder => builder.GetPluginRegistryInventory();
        Func<TensorRtBuilder, string, string, string, bool> pluginCreatorLookup =
            static (builder, name, version, pluginNamespace) => builder.IsPluginCreatorRegistered(name, version, pluginNamespace);
        Func<TensorRtBuilder, string, string, string, (bool success, bool found, string diagnostic)> safePluginCreatorLookup =
            static (builder, name, version, pluginNamespace) =>
            {
                bool success = builder.TryIsPluginCreatorRegistered(name, version, pluginNamespace, out bool found, out string diagnostic);
                return (success, found, diagnostic);
            };
        Func<TensorRtApiLine, bool> globalPluginRegistryAvailable =
            static line => TensorRtEnvironmentProbe.IsGlobalPluginRegistryAvailable(line);
        Func<TensorRtApiLine, (bool success, bool available, string diagnostic)> safeGlobalPluginRegistryAvailable =
            static line =>
            {
                bool success = TensorRtEnvironmentProbe.TryIsGlobalPluginRegistryAvailable(line, out bool available, out string diagnostic);
                return (success, available, diagnostic);
            };
        Func<TensorRtApiLine, TensorRtPluginRegistryInventory> globalPluginRegistryInventory =
            static line => TensorRtEnvironmentProbe.GetGlobalPluginRegistryInventory(line);
        Func<TensorRtApiLine, bool> globalPluginRegistryParentSearch =
            static line => TensorRtEnvironmentProbe.IsGlobalPluginRegistryParentSearchEnabled(line);
        Action<TensorRtApiLine, bool> setGlobalPluginRegistryParentSearch =
            static (line, enabled) => TensorRtEnvironmentProbe.SetGlobalPluginRegistryParentSearchEnabled(line, enabled);
        Func<TensorRtApiLine, bool, (bool success, bool actual, string diagnostic)> trySetGlobalPluginRegistryParentSearch =
            static (line, enabled) =>
            {
                bool success = TensorRtEnvironmentProbe.TrySetGlobalPluginRegistryParentSearchEnabled(line, enabled, out bool actual, out string diagnostic);
                return (success, actual, diagnostic);
            };
        Func<TensorRtPluginRegistryInventory, int> creatorCount = static inventory => inventory.Creators.Count;
        Func<TensorRtPluginRegistryInventory, string, string, string, TensorRtPluginCreatorInfo?> snapshotFindCreator =
            static (inventory, name, version, pluginNamespace) => inventory.FindCreator(name, version, pluginNamespace);
        Func<TensorRtPluginRegistryInventory, string, string, string, bool> snapshotTryFindCreator =
            static (inventory, name, version, pluginNamespace) => inventory.TryFindCreator(name, version, pluginNamespace, out _);
        Func<TensorRtPluginRegistryInventory, IReadOnlyList<TensorRtPluginCreatorSummary>> creatorSummaries =
            static inventory => inventory.GetCreatorSummaries();
        Func<TensorRtPluginRegistryInventory, int, IReadOnlyList<TensorRtPluginCreatorSummary>> limitedCreatorSummaries =
            static (inventory, maxCreators) => inventory.GetCreatorSummaries(maxCreators);
        Func<TensorRtPluginRegistryInventory, IReadOnlyList<TensorRtPluginFieldSummary>> fieldSummaries =
            static inventory => inventory.GetFieldSummaries();
        Func<TensorRtPluginRegistryInventory, int, int, IReadOnlyList<TensorRtPluginFieldSummary>> limitedFieldSummaries =
            static (inventory, maxCreators, maxFieldsPerCreator) => inventory.GetFieldSummaries(maxCreators, maxFieldsPerCreator);
        Func<TensorRtPluginRegistryInventory, TensorRtPluginRegistryInventoryDiagnostics> pluginInventoryDiagnostics =
            static inventory => inventory.GetDiagnostics();
        Func<TensorRtPluginRegistryInventoryDiagnostics, string> pluginInventoryDiagnosticsText =
            static diagnostics => diagnostics.IsConsistent + ":" +
                diagnostics.CreatorCount + ":" +
                diagnostics.SummaryCount + ":" +
                diagnostics.TotalFieldCount + ":" +
                diagnostics.EmptyNameCount + ":" +
                diagnostics.NegativeFieldLengthCount;
        Func<TensorRtPluginCreatorSummary, string> creatorSummaryIdentity =
            static summary => summary.Name + ":" + summary.Version + ":" + summary.Namespace + ":" + summary.FieldCount;
        Func<TensorRtPluginFieldSummary, string> fieldSummaryIdentity =
            static summary => summary.CreatorName + ":" + summary.CreatorVersion + ":" + summary.CreatorNamespace + ":" +
                summary.FieldIndex + ":" + summary.FieldName + ":" + summary.FieldType + ":" + summary.Length + ":" + summary.HasData;
        Func<TensorRtPluginCreatorInfo, string> creatorIdentity =
            static creator => creator.Name + ":" + creator.Version + ":" + creator.Namespace;
        Func<TensorRtPluginCreatorInfo, int?> creatorTensorRtVersion = static creator => creator.TensorRtVersion;
        Func<TensorRtPluginCreatorSummary, int?> creatorSummaryTensorRtVersion = static summary => summary.TensorRtVersion;
        Func<TensorRtPluginCreatorInfo, int> creatorFieldCount = static creator => creator.Fields.Count;
        Func<TensorRtPluginFieldInfo, string> fieldMetadata =
            static field => field.Name + ":" + field.FieldType + ":" + field.Length + ":" + field.HasData;
        Func<TensorRtLayer, TensorRtPluginV2LayerMetadata> pluginV2LayerMetadata =
            static layer => layer.GetPluginV2Metadata();
        Func<TensorRtLayer, (bool success, TensorRtPluginV2LayerMetadata? metadata, string diagnostic)> safePluginV2LayerMetadata =
            static layer =>
            {
                bool success = layer.TryGetPluginV2Metadata(out TensorRtPluginV2LayerMetadata? metadata, out string diagnostic);
                return (success, metadata, diagnostic);
            };
        Func<TensorRtPluginV2LayerMetadata, string> pluginV2LayerMetadataSummary =
            static metadata => metadata.PluginType + ":" + metadata.PluginVersion + ":" + metadata.PluginNamespace + ":" +
                metadata.SerializationSize + ":" + metadata.PackedTensorRtVersion + ":" + metadata.PluginApiVersionTag + ":" +
                metadata.TensorRtVersion + ":" + metadata.TensorRtMajor + ":" + metadata.TensorRtMinor + ":" +
                metadata.TensorRtPatch + ":" + metadata.OutputCount + ":" + metadata.HasExtCapability + ":" +
                metadata.HasIoExtCapability + ":" + metadata.HasDynamicExtCapability + ":" + metadata.IsConsistent;
        Func<TensorRtLayer, int, TensorRtDims> pluginV2LegacyOutputDimensions =
            static (layer, outputIndex) => layer.GetPluginV2LegacyOutputDimensions(outputIndex);
        Func<TensorRtLayer, int, ulong> pluginV2LegacyWorkspaceSize =
            static (layer, maxBatchSize) => layer.GetPluginV2LegacyWorkspaceSize(maxBatchSize);
        Func<TensorRtLayer, TensorRtDataType, TensorRtTensorFormat, bool> pluginV2LegacyFormat =
            static (layer, dataType, tensorFormat) => layer.SupportsPluginV2LegacyFormat(dataType, tensorFormat);
        Func<TensorRtLayer, int, TensorRtDataType> pluginV2OutputDataType =
            static (layer, outputIndex) => layer.GetPluginV2OutputDataType(outputIndex);
        Func<TensorRtLayer, int, bool> pluginV2InputBroadcast =
            static (layer, inputIndex) => layer.CanPluginV2BroadcastInputAcrossBatch(inputIndex);
        Func<TensorRtLayer, int, bool[], bool> pluginV2OutputBroadcast =
            static (layer, outputIndex, inputFlags) => layer.IsPluginV2OutputBroadcastAcrossBatch(outputIndex, inputFlags);
        Func<TensorRtLayer, TensorRtPluginFormatSupportSnapshot> pluginV2DynamicFormatSupport =
            static layer => layer.GetPluginV2DynamicFormatSupportSnapshot();
        Func<TensorRtLayer, TensorRtPluginFormatSupportSnapshot> pluginV2IoExtFormatSupport =
            static layer => layer.GetPluginV2IoExtFormatSupportSnapshot();
        Func<TensorRtPluginFormatSupportSnapshot, string> pluginFormatSupportSummary =
            static snapshot => snapshot.Capability + ":" + snapshot.InputCount + ":" + snapshot.OutputCount + ":" +
                snapshot.Support.Count + ":" + snapshot.AllSupported + ":" + snapshot.IsConsistent;
        Func<TensorRtLayer, TensorRtPluginV3LayerMetadata> pluginV3LayerMetadata =
            static layer => layer.GetPluginV3Metadata();
        Func<TensorRtLayer, (bool success, TensorRtPluginV3LayerMetadata? metadata, string diagnostic)> safePluginV3LayerMetadata =
            static layer =>
            {
                bool success = layer.TryGetPluginV3Metadata(out TensorRtPluginV3LayerMetadata? metadata, out string diagnostic);
                return (success, metadata, diagnostic);
            };
        Func<TensorRtPluginV3LayerMetadata, string> pluginV3LayerMetadataSummary =
            static metadata => metadata.Line + ":" + metadata.PluginInterface + ":" + metadata.HasCoreCapability + ":" +
                metadata.Core.PluginName + ":" + metadata.Core.PluginVersion + ":" + metadata.Core.PluginNamespace + ":" +
                metadata.HasBuildCapability + ":" + metadata.Build?.OutputCount + ":" + metadata.Build?.TacticCount + ":" +
                metadata.Build?.FormatCombinationLimit + ":" + metadata.Build?.TimingCacheId + ":" + metadata.Build?.MetadataString + ":" +
                metadata.HasRuntimeCapability + ":" + metadata.Runtime?.InterfaceMetadata + ":" + metadata.IsConsistent;
        Func<TensorRtLayer, TensorRtPluginV3BuildIoSnapshot> pluginV3BuildIoSnapshot =
            static layer => layer.GetPluginV3BuildIoSnapshot();
        Func<TensorRtPluginV3BuildIoSnapshot, string> pluginV3BuildIoSummary =
            static snapshot => snapshot.InputCount + ":" + snapshot.OutputCount + ":" + snapshot.OutputDataTypes.Count + ":" +
                snapshot.AliasedInputIndices.Count + ":" + snapshot.AliasMetadataAvailable + ":" + snapshot.FormatSupport + ":" + snapshot.IsConsistent;
        Func<TensorRtLayer, TensorRtPluginV3SerializationFieldInventory> pluginV3SerializationFields =
            static layer => layer.GetPluginV3RuntimeSerializationFields();
        Func<TensorRtPluginV3SerializationFieldInventory, string> pluginV3SerializationFieldSummary =
            static inventory => inventory.Line + ":" + inventory.Fields.Count + ":" + inventory.PointerFreeCopiedInventory + ":" + inventory.IsConsistent;
        Func<TensorRtEngine, bool> engineImplicitBatchCompatibility =
            static engine => engine.HasImplicitBatchDimensionCompatibility;
        Action<TensorRtBuilder, int> setBuilderMaxBatchCompatibility =
            static (builder, value) => builder.SetMaxBatchSizeCompatibility(value);
        Func<TensorRtBuilderConfig, int> serializedPluginPathCountCompatibility =
            static config => config.SerializedPluginPathCountCompatibility;
        Action<TensorRtBuilderConfig, ulong> setMaxWorkspaceSizeCompatibility =
            static (config, value) => config.SetMaxWorkspaceSizeCompatibility(value);
        Action<TensorRtBuilderConfig, int> setMinTimingIterationsCompatibility =
            static (config, value) => config.SetMinTimingIterationsCompatibility(value);
        Func<TensorRtInferenceBindings, TensorRtInferenceExecutionSummary> executeV2 =
            static bindings => bindings.ExecuteV2(runShapeInference: false);
        Func<TensorRtInferenceBindings, int, TensorRtInferenceExecutionSummary> executeLegacy =
            static (bindings, batchSize) => bindings.ExecuteLegacy(batchSize, runShapeInference: false);
        Func<TensorRtInferenceBindings, CudaStream, TensorRtInferenceExecutionSummary> enqueueV2AndSynchronize =
            static (bindings, stream) => bindings.EnqueueV2AndSynchronize(stream, runShapeInference: false);
        Func<TensorRtLayer, int> rnnV2LayerCount = static layer => layer.GetRnnV2LayerCount();
        Func<TensorRtLayer, int> rnnV2HiddenSize = static layer => layer.GetRnnV2HiddenSize();
        Func<TensorRtLayer, int> rnnV2DataLength = static layer => layer.GetRnnV2DataLength();
        Func<TensorRtLayer, int> rnnV2MaxSequenceLength = static layer => layer.GetRnnV2MaxSequenceLength();
        Func<TensorRtLayer, TensorRtRnnOperation> rnnV2Operation = static layer => layer.GetRnnV2Operation();
        Func<TensorRtLayer, TensorRtRnnDirection> rnnV2Direction = static layer => layer.GetRnnV2Direction();
        Func<TensorRtLayer, TensorRtRnnInputMode> rnnV2InputMode = static layer => layer.GetRnnV2InputMode();
        Func<TensorRtLayer, TensorRtTensor?> rnnV2CellState = static layer => layer.GetRnnV2CellState();
        Func<TensorRtLayer, TensorRtTensor?> rnnV2HiddenState = static layer => layer.GetRnnV2HiddenState();
        Func<TensorRtLayer, TensorRtTensor?> rnnV2SequenceLengths = static layer => layer.GetRnnV2SequenceLengths();
        Action<TensorRtLayer, TensorRtRnnOperation> setRnnV2Operation = static (layer, value) => layer.SetRnnV2Operation(value);
        Action<TensorRtLayer, TensorRtRnnDirection> setRnnV2Direction = static (layer, value) => layer.SetRnnV2Direction(value);
        Action<TensorRtLayer, TensorRtRnnInputMode> setRnnV2InputMode = static (layer, value) => layer.SetRnnV2InputMode(value);
        Action<TensorRtLayer, TensorRtTensor> setRnnV2CellState = static (layer, tensor) => layer.SetRnnV2CellState(tensor);
        Action<TensorRtLayer, TensorRtTensor> setRnnV2HiddenState = static (layer, tensor) => layer.SetRnnV2HiddenState(tensor);
        Action<TensorRtLayer, TensorRtTensor> setRnnV2SequenceLengths = static (layer, tensor) => layer.SetRnnV2SequenceLengths(tensor);
        Func<TensorRtLayer, int, TensorRtRnnGateType, bool, TensorRtRnnV2GateWeightsSnapshot> rnnV2WeightsForGate =
            static (layer, layerIndex, gate, inputWeights) => layer.GetRnnV2WeightsForGate(layerIndex, gate, inputWeights);
        Func<TensorRtLayer, int, TensorRtRnnGateType, bool, TensorRtRnnV2GateWeightsSnapshot> rnnV2BiasForGate =
            static (layer, layerIndex, gate, inputWeights) => layer.GetRnnV2BiasForGate(layerIndex, gate, inputWeights);
        Func<TensorRtTensor, bool> rnnV2OwnerLifetimeBound = static tensor => tensor.IsOwnerLifetimeBound;
        Func<TensorRtRnnV2GateWeightsSnapshot, byte[]> rnnV2CopiedWeightBytes = static snapshot => snapshot.ToArray();
        Func<TensorRtRnnV2BorrowedStateDesignGateResult> rnnV2BorrowedStateDesignGate =
            static () => TensorRtRnnV2BorrowedStateDesignGate.EvaluateKnownSurface();
        Func<TensorRtRnnV2BorrowedStateDesignGateResult, string> rnnV2BorrowedStateSummary =
            static gate => gate.EvidenceKind + ":" + gate.DataLengthScalarPromoted + ":" +
                gate.RemainingDeferredTriageRowCount + ":" + gate.CanPromoteRuntimeProof;
        Func<TensorRtApiLine, TensorRtRuntimeDeserializationBoundaryPrecheckResult> runtimeDeserializationBoundaryPrecheck =
            static line => TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface(line);
        Func<TensorRtRuntimeDeserializationBoundaryPrecheckResult, string> runtimeDeserializationBoundaryPrecheckSummary =
            static precheck => precheck.EvidenceKind + ":" +
                precheck.RuntimeEvidenceKind + ":" +
                precheck.IsRuntimeExecutionEvidence + ":" +
                precheck.IsRuntimeExecutionProof + ":" +
                precheck.ManagedByteArrayDeserializeReady + ":" +
                precheck.ManagedStreamDeserializeReady + ":" +
                precheck.HostMemoryDeserializeReady + ":" +
                precheck.SerializedBufferCopiedBeforeInterop + ":" +
                precheck.PinnedBufferScopedToInteropCall + ":" +
                precheck.BorrowedSerializedBufferEscaped + ":" +
                precheck.EngineHandleOwnedByWrapper + ":" +
                precheck.EnginePointerExposed + ":" +
                precheck.DirectDeserializeCudaEngineRowsDeferred + ":" +
                precheck.DirectDeserializeCudaEngineV2RowsDeferred + ":" +
                precheck.LoadRuntimeDeferred + ":" +
                precheck.SafeDeserializeBridgeReady + ":" +
                precheck.CanAttemptRuntimeProof + ":" +
                precheck.RuntimeProofBlocked + ":" +
                precheck.BlockedPrerequisiteCount;

        TensorRtLogHandler logHandler = static (severity, message) => _ = severity.ToString() + message;
        TensorRtProfilerHandler profilerHandler = static (layerName, milliseconds) => _ = layerName.Length + milliseconds;
        TensorRtProgressMonitorHandler progressHandler =
            static progressEvent => progressEvent.Kind != TensorRtProgressMonitorEventKind.StepComplete || progressEvent.Step <= progressEvent.StepCount;
        TensorRtAllocatorDryRunHandler allocatorDryRunHandler =
            static request => TensorRtAllocatorDryRunResult.Success("package-consumer:" + request.Reason + ":" + request.Size + ":" + request.Alignment);
        Func<TensorRtApiLine, TensorRtLogHandler, TensorRtLogger> loggerFactory =
            static (line, handler) => new TensorRtLogger(line, handler);
        Func<TensorRtApiLine, TensorRtProfilerHandler, TensorRtProfiler> profilerFactory =
            static (line, handler) => new TensorRtProfiler(line, handler);
        Func<TensorRtApiLine, TensorRtProgressMonitorHandler, TensorRtProgressMonitor> progressFactory =
            static (line, handler) => new TensorRtProgressMonitor(line, handler);
        Func<TensorRtAllocatorDryRunHandler, TensorRtAllocatorCallbackOwner> allocatorOwnerFactory =
            static handler => new TensorRtAllocatorCallbackOwner(handler);
        Func<TensorRtApiLine, int> errorCodeExclusiveUpperBound =
            static line => TensorRtErrorCodeMetadata.GetExclusiveUpperBound(line);
        Func<TensorRtBuilder, TensorRtNetworkDefinition, TensorRtBuilderConfig, TensorRtEngine> directEngineBuild =
            static (builder, network, config) => builder.BuildEngineWithConfig(network, config);
        Func<TensorRtBuilderConfig, IReadOnlyList<string>, bool> serializedPluginPathSet =
            static (config, paths) => config.SetPluginsToSerialize(paths);
        Func<TensorRtBuilderConfig, IReadOnlyList<string>> serializedPluginPathCopy =
            static config => config.GetPluginsToSerialize();
        Func<TensorRtExecutionContext, int, IReadOnlyList<int>, bool> legacyShapeBindingSet =
            static (context, bindingIndex, values) => context.SetInputShapeBinding(bindingIndex, values);
        Func<TensorRtRefitter, bool> refitterHasLogger = static refitter => refitter.HasLogger;
        Func<TensorRtRuntime, string> runtimeErrorRecorderSnapshot =
            static runtime => runtime.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)
                ? snapshot.ErrorCount + ":" + snapshot.Records.Count + ":" + snapshot.HasOverflowed
                : "none";
        Func<TensorRtRuntime, string> runtimeDiagnosticSnapshot =
            static runtime =>
            {
                TensorRtRuntimeDiagnosticSnapshot snapshot = runtime.GetDiagnosticSnapshot();
                TensorRtRuntimeDiagnosticSummary summary = snapshot.ToSummary();
                return snapshot.HasLogger + ":" +
                    snapshot.HasErrorRecorder + ":" +
                    snapshot.ErrorRecorder.ErrorCount + ":" +
                    snapshot.ErrorRecorder.Records.Count + ":" +
                    snapshot.Diagnostics.Count + ":" +
                    summary.ErrorCount + ":" +
                    summary.CopiedErrorRecordCount + ":" +
                    summary.DiagnosticCount;
            };
        Func<TensorRtRuntimeDiagnosticSnapshot, TensorRtRuntimeDiagnosticSummary> runtimeDiagnosticSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtRuntimeDiagnosticSummary, string> runtimeDiagnosticSummaryText =
            static summary => summary.HasLogger + ":" + summary.HasErrorRecorder + ":" + summary.ErrorCount + ":" + summary.CopiedErrorRecordCount + ":" + summary.DiagnosticCount + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtEngineDeploymentSnapshot, TensorRtEngineDeploymentSummary> engineDeploymentSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtEngineDeploymentSummary, string> engineDeploymentSummaryText =
            static summary => summary.CopiedTensorCount + ":" + summary.CopiedProfileTensorValueCount + ":" + summary.DiagnosticCount + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtBuilderConfigDeploymentSnapshot, TensorRtBuilderConfigDeploymentSummary> builderConfigDeploymentSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtBuilderConfigDeploymentSummary, string> builderConfigDeploymentSummaryText =
            static summary => summary.OptimizationProfileCount + ":" + summary.PluginToSerializeCount + ":" + summary.DiagnosticCount + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtExecutionContextDeploymentSnapshot, TensorRtExecutionContextDeploymentSummary> executionContextDeploymentSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtExecutionContextDeploymentSummary, string> executionContextDeploymentSummaryText =
            static summary => summary.CopiedTensorStateCount + ":" + summary.CopiedRuntimeDiagnosticCount + ":" + summary.DiagnosticCount + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtSerializationConfig, TensorRtSerializationConfigSummary> serializationConfigSummary =
            static config => config.ToSummary();
        Func<TensorRtSerializationConfigSummary, string> serializationConfigSummaryText =
            static summary => summary.Line + ":" + summary.Flags + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtRuntimeConfig, TensorRtRuntimeConfigSummary> runtimeConfigSummary =
            static config => config.ToSummary();
        Func<TensorRtRuntimeConfigSummary, string> runtimeConfigSummaryText =
            static summary => summary.Line + ":" + summary.AllocationStrategy + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtRuntime, bool> runtimeHasLogger = static runtime => runtime.HasLogger;
        Action<TensorRtRuntime> runtimeClearErrorRecorder = static runtime => runtime.ClearErrorRecorder();
        Action<TensorRtRuntime> runtimeClearGpuAllocator = static runtime => runtime.ClearGpuAllocator();
        Func<TensorRtRefitter, string> refitterErrorRecorderSnapshot =
            static refitter => refitter.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)
                ? snapshot.ErrorCount + ":" + snapshot.Records.Count + ":" + snapshot.HasOverflowed
                : "none";
        Func<TensorRtRefitter, string> refitterDiagnosticSnapshot =
            static refitter =>
            {
                TensorRtRefitterDiagnosticSnapshot snapshot = refitter.GetDiagnosticSnapshot();
                TensorRtRefitterDiagnosticSummary summary = snapshot.ToSummary();
                return snapshot.HasLogger + ":" +
                    snapshot.HasErrorRecorder + ":" +
                    snapshot.ErrorRecorder.ErrorCount + ":" +
                    snapshot.MissingNamedWeightCount + ":" +
                    snapshot.AllNamedWeightCount + ":" +
                    snapshot.DynamicRangeTensorNames.Count + ":" +
                    snapshot.MissingNamedWeights.Count + ":" +
                    snapshot.AllNamedWeights.Count + ":" +
                    snapshot.Diagnostics.Count + ":" +
                    summary.CopiedMissingNamedWeightCount + ":" +
                    summary.CopiedAllNamedWeightCount + ":" +
                    summary.DiagnosticCount;
            };
        Func<TensorRtRefitterDiagnosticSnapshot, TensorRtRefitterDiagnosticSummary> refitterDiagnosticSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtRefitterDiagnosticSummary, string> refitterDiagnosticSummaryText =
            static summary => summary.HasLogger + ":" + summary.HasErrorRecorder + ":" + summary.MissingNamedWeightCount + ":" + summary.CopiedMissingNamedWeightCount + ":" + summary.AllNamedWeightCount + ":" + summary.CopiedAllNamedWeightCount + ":" + summary.DiagnosticCount;
        Func<TensorRtOnnxParser, string> onnxParserDiagnosticSnapshot =
            static parser =>
            {
                TensorRtOnnxParserDiagnosticSnapshot snapshot = parser.GetDiagnosticSnapshot();
                TensorRtOnnxParserDiagnosticSummary summary = snapshot.ToSummary();
                return snapshot.Line + ":" +
                    snapshot.ErrorCount + ":" +
                    snapshot.Diagnostics.Count + ":" +
                    snapshot.UsedVCPluginLibraries.Count + ":" +
                    snapshot.IdentityOperatorSupported + ":" +
                    snapshot.DiagnosticSummary.Length + ":" +
                    summary.CopiedDiagnosticCount + ":" +
                    summary.UsedVCPluginLibraryCount + ":" +
                    summary.DiagnosticSummaryLength;
            };
        Func<TensorRtOnnxParserDiagnosticSnapshot, TensorRtOnnxParserDiagnosticSummary> onnxParserDiagnosticSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtOnnxParserDiagnosticSummary, string> onnxParserDiagnosticSummaryText =
            static summary => summary.Line + ":" + summary.ErrorCount + ":" + summary.CopiedDiagnosticCount + ":" + summary.UsedVCPluginLibraryCount + ":" + summary.DiagnosticSummaryLength + ":" + summary.IdentityOperatorSupported + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtOnnxModelSupportReport, TensorRtOnnxModelSupportSummary> onnxModelSupportSummary =
            static report => report.ToSummary();
        Func<TensorRtOnnxModelSupportSummary, string> onnxModelSupportSummaryText =
            static summary => summary.IsSupported + ":" + summary.ReportedSupportedSubgraphCount + ":" + summary.ReportedUnsupportedSubgraphCount + ":" + summary.CopiedSubgraphCount + ":" + summary.CopiedSupportedSubgraphCount + ":" + summary.CopiedUnsupportedSubgraphCount + ":" + summary.CopiedNodeCount + ":" + summary.CopiedSubgraphCountsMatchReportedCounts + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Func<TensorRtOnnxParserRefitter, string> onnxParserRefitterDiagnosticSnapshot =
            static parserRefitter =>
            {
                TensorRtOnnxParserRefitterDiagnosticSnapshot snapshot = parserRefitter.GetDiagnosticSnapshot();
                TensorRtOnnxParserRefitterDiagnosticSummary summary = snapshot.ToSummary();
                return snapshot.Line + ":" +
                    snapshot.ErrorCount + ":" +
                    snapshot.Diagnostics.Count + ":" +
                    snapshot.DiagnosticSummary.Length + ":" +
                    summary.CopiedDiagnosticCount + ":" +
                    summary.DiagnosticSummaryLength;
            };
        Func<TensorRtOnnxParserRefitterDiagnosticSnapshot, TensorRtOnnxParserRefitterDiagnosticSummary> onnxParserRefitterDiagnosticSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtOnnxParserRefitterDiagnosticSummary, string> onnxParserRefitterDiagnosticSummaryText =
            static summary => summary.Line + ":" + summary.ErrorCount + ":" + summary.CopiedDiagnosticCount + ":" + summary.DiagnosticSummaryLength + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Action<TensorRtRefitter> refitterClearErrorRecorder = static refitter => refitter.ClearErrorRecorder();
        Func<TensorRtBuilder, bool> builderHasErrorRecorder = static builder => builder.HasErrorRecorder;
        Func<TensorRtBuilder, bool> builderHasLogger = static builder => builder.HasLogger;
        Action<TensorRtBuilder> builderClearErrorRecorder = static builder => builder.ClearErrorRecorder();
        Action<TensorRtBuilder> builderClearGpuAllocator = static builder => builder.ClearGpuAllocator();
        Func<TensorRtExecutionContext, bool> contextHasErrorRecorder = static context => context.HasErrorRecorder;
        Action<TensorRtExecutionContext> contextClearErrorRecorder = static context => context.ClearErrorRecorder();
        Func<TensorRtExecutionContext, string, bool> contextHasOutputAllocator =
            static (context, tensorName) => context.HasOutputAllocator(tensorName);
        Action<TensorRtExecutionContext, string> contextClearOutputAllocator =
            static (context, tensorName) => context.ClearOutputAllocator(tensorName);
        Func<TensorRtExecutionContext, bool> contextHasTemporaryStorageAllocator =
            static context => context.HasTemporaryStorageAllocator;
        Action<TensorRtExecutionContext> contextClearTemporaryStorageAllocator =
            static context => context.ClearTemporaryStorageAllocator();
        Func<TensorRtExecutionContext, bool> contextHasDebugListener = static context => context.HasDebugListener;
        Action<TensorRtExecutionContext> contextClearDebugListener = static context => context.ClearDebugListener();
        Func<TensorRtExecutionContext, string, (bool success, TensorRtInterfaceInfo info, string diagnostic)> contextOutputAllocatorInterfaceInfo =
            static (context, tensorName) =>
            {
                bool success = context.TryGetOutputAllocatorInterfaceInfo(tensorName, out TensorRtInterfaceInfo info, out string diagnostic);
                return (success, info, diagnostic);
            };
        Func<TensorRtExecutionContext, (bool success, TensorRtInterfaceInfo info, string diagnostic)> contextTemporaryStorageAllocatorInterfaceInfo =
            static context =>
            {
                bool success = context.TryGetTemporaryStorageAllocatorInterfaceInfo(out TensorRtInterfaceInfo info, out string diagnostic);
                return (success, info, diagnostic);
            };
        Func<TensorRtExecutionContext, (bool success, TensorRtInterfaceInfo info, string diagnostic)> contextDebugListenerInterfaceInfo =
            static context =>
            {
                bool success = context.TryGetDebugListenerInterfaceInfo(out TensorRtInterfaceInfo info, out string diagnostic);
                return (success, info, diagnostic);
            };
        Func<TensorRtExecutionContext, string, TensorRtExecutionContextCallbackStateSnapshot> contextCallbackStateSnapshot =
            static (context, tensorName) => context.GetCallbackStateSnapshot(tensorName);
        Func<TensorRtExecutionContext, string, TensorRtExecutionContextCallbackStateSnapshot> contextClearCallbackState =
            static (context, tensorName) => context.ClearCallbackState(tensorName);
        Func<TensorRtExecutionContextCallbackStateSnapshot, string> callbackStateSummary =
            static snapshot =>
                snapshot.HasOutputAllocator + ":" +
                snapshot.HasTemporaryStorageAllocator + ":" +
                snapshot.HasDebugListener + ":" +
                snapshot.OutputAllocatorInterfaceInfoAvailable + ":" +
                snapshot.TemporaryStorageAllocatorInterfaceInfoAvailable + ":" +
                snapshot.DebugListenerInterfaceInfoAvailable + ":" +
                snapshot.OutputAllocatorClearSupported + ":" +
                snapshot.TemporaryStorageAllocatorClearSupported + ":" +
                snapshot.DebugListenerClearSupported + ":" +
                snapshot.OutputAllocatorCleared + ":" +
                snapshot.TemporaryStorageAllocatorCleared + ":" +
                snapshot.DebugListenerCleared + ":" +
                snapshot.LastStatus + ":" +
                snapshot.LastOperation + ":" +
                snapshot.Diagnostic;
        Func<TensorRtExecutionContext, string, TensorRtExecutionContextRuntimeDiagnosticSnapshot> contextRuntimeDiagnosticSnapshot =
            static (context, tensorName) => context.GetRuntimeDiagnosticSnapshot(tensorName);
        Func<TensorRtExecutionContextRuntimeDiagnosticSnapshot, TensorRtExecutionContextRuntimeDiagnosticSummary> contextRuntimeDiagnosticSummary =
            static snapshot => snapshot.ToSummary();
        Func<TensorRtExecutionContextRuntimeDiagnosticSummary, string> contextRuntimeDiagnosticSummaryText =
            static summary =>
                summary.HasOutputTensorName + ":" +
                summary.HasErrorRecorder + ":" +
                summary.HasOutputAllocator + ":" +
                summary.IsOutputTensorAddressSet + ":" +
                summary.HasTemporaryStorageAllocator + ":" +
                summary.HasDebugListener + ":" +
                summary.HasManagedProfiler + ":" +
                summary.HasNativeProfiler + ":" +
                summary.CallbackStateLastStatus + ":" +
                summary.DiagnosticCount;
        Action<TensorRtExecutionContext, CudaStream[]> setExecutionContextAuxiliaryStreams =
            static (context, streams) => context.SetAuxStreams(streams);
        Action<TensorRtExecutionContext> clearExecutionContextAuxiliaryStreams =
            static context => context.ClearAuxStreams();
        Func<TensorRtExecutionContext, TensorRtAuxiliaryStreamAssignmentSnapshot> contextAuxiliaryStreamAssignmentSnapshot =
            static context => context.GetAuxiliaryStreamAssignmentSnapshot();
        Func<TensorRtAuxiliaryStreamAssignmentSnapshot, string> contextAuxiliaryStreamAssignmentSummary =
            static snapshot =>
                snapshot.Line + ":" +
                snapshot.AssignedStreamCount + ":" +
                snapshot.IsCleared + ":" +
                snapshot.ManagedHandleLeaseActive + ":" +
                snapshot.NativeStreamPointerExposed + ":" +
                snapshot.BorrowedHandleEscaped + ":" +
                snapshot.Diagnostic;
        Func<TensorRtExecutionContext, string, TensorRtExecutionContextCallbackAllocatorSafeControlSummary> contextCallbackAllocatorSafeControlSummary =
            static (context, tensorName) => context.GetCallbackAllocatorSafeControlSummary(tensorName);
        Func<TensorRtExecutionContextCallbackAllocatorSafeControlSummary, string> callbackAllocatorSafeControlSummaryText =
            static summary =>
                summary.EvidenceKind + ":" +
                summary.RuntimeEvidenceKind + ":" +
                summary.RealCallbackRuntime + ":" +
                summary.IsRealCallbackRuntimeProof + ":" +
                summary.HasOutputAllocator + ":" +
                summary.HasTemporaryStorageAllocator + ":" +
                summary.HasDebugListener + ":" +
                summary.OutputAllocatorInterfaceInfoAvailable + ":" +
                summary.TemporaryStorageAllocatorInterfaceInfoAvailable + ":" +
                summary.DebugListenerInterfaceInfoAvailable + ":" +
                summary.CopiedInterfaceInfoCount + ":" +
                summary.DiagnosticCount + ":" +
                summary.PointerFreeSurfaceReady + ":" +
                summary.CallbackInvocationAttempted + ":" +
                summary.IsRuntimeInvocationProofComplete;
        Func<TensorRtLogger, bool> loggerDiagnostic =
            static logger => logger.EmitDiagnostic(TensorRtLogSeverity.Info, "package-consumer");
        Func<TensorRtLogger, string> loggerCallbackState =
            static logger => logger.CallbackInvocationCount + ":" + logger.CallbackFailureCount + ":" + (logger.LastCallbackException?.GetType().Name ?? "none") + ":" + logger.IsAttached;
        Func<TensorRtLogger, TensorRtInterfaceInfo> loggerInterfaceInfo = static logger => logger.InterfaceInfo;
        Func<TensorRtLogger, (bool success, TensorRtInterfaceInfo info, string diagnostic)> tryLoggerInterfaceInfo =
            static logger =>
            {
                bool success = logger.TryGetInterfaceInfo(out TensorRtInterfaceInfo info, out string diagnostic);
                return (success, info, diagnostic);
            };
        Func<TensorRtLogger, TensorRtApiLanguage> loggerApiLanguage = static logger => logger.ApiLanguage;
        Func<TensorRtLogger, (bool success, TensorRtApiLanguage apiLanguage, string diagnostic)> tryLoggerApiLanguage =
            static logger =>
            {
                bool success = logger.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic);
                return (success, apiLanguage, diagnostic);
            };
        Func<TensorRtProfiler, bool> profilerDiagnostic =
            static profiler => profiler.EmitDiagnostic("package-consumer", 0.125f);
        Func<TensorRtProfiler, string> profilerCallbackState =
            static profiler => profiler.CallbackInvocationCount + ":" + profiler.CallbackFailureCount + ":" + (profiler.LastCallbackException?.GetType().Name ?? "none") + ":" + profiler.IsAttached;
        Func<TensorRtProfiler, TensorRtInterfaceInfo> profilerInterfaceInfo = static profiler => profiler.InterfaceInfo;
        Func<TensorRtProfiler, (bool success, TensorRtInterfaceInfo info, string diagnostic)> tryProfilerInterfaceInfo =
            static profiler =>
            {
                bool success = profiler.TryGetInterfaceInfo(out TensorRtInterfaceInfo info, out string diagnostic);
                return (success, info, diagnostic);
            };
        Func<TensorRtProfiler, TensorRtApiLanguage> profilerApiLanguage = static profiler => profiler.ApiLanguage;
        Func<TensorRtProfiler, (bool success, TensorRtApiLanguage apiLanguage, string diagnostic)> tryProfilerApiLanguage =
            static profiler =>
            {
                bool success = profiler.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic);
                return (success, apiLanguage, diagnostic);
            };
        Action<TensorRtExecutionContext, TensorRtProfiler> attachProfiler =
            static (context, profiler) => context.SetProfiler(profiler);
        Action<TensorRtExecutionContext> clearProfiler = static context => context.ClearProfiler();
        Func<TensorRtExecutionContext, bool> hasManagedProfiler = static context => context.HasProfiler;
        Func<TensorRtExecutionContext, bool> hasNativeProfiler = static context => context.HasNativeProfiler;
        Func<TensorRtProgressMonitor, bool> progressDiagnostic =
            static monitor => monitor.EmitDiagnostic(
                TensorRtProgressMonitorEventKind.StepComplete,
                "package-consumer",
                step: 1,
                stepCount: 1).CallbackAccepted;
        Func<TensorRtProgressMonitor, string> progressCallbackState =
            static monitor => monitor.CallbackInvocationCount + ":" + monitor.CallbackFailureCount + ":" + (monitor.LastCallbackException?.GetType().Name ?? "none") + ":" + monitor.IsAttached;
        Func<TensorRtProgressMonitor, TensorRtInterfaceInfo> progressInterfaceInfo = static monitor => monitor.InterfaceInfo;
        Func<TensorRtProgressMonitor, (bool success, TensorRtInterfaceInfo info, string diagnostic)> tryProgressInterfaceInfo =
            static monitor =>
            {
                bool success = monitor.TryGetInterfaceInfo(out TensorRtInterfaceInfo info, out string diagnostic);
                return (success, info, diagnostic);
            };
        Func<TensorRtProgressMonitor, TensorRtApiLanguage> progressApiLanguage = static monitor => monitor.ApiLanguage;
        Func<TensorRtProgressMonitor, (bool success, TensorRtApiLanguage apiLanguage, string diagnostic)> tryProgressApiLanguage =
            static monitor =>
            {
                bool success = monitor.TryGetApiLanguage(out TensorRtApiLanguage apiLanguage, out string diagnostic);
                return (success, apiLanguage, diagnostic);
            };
        Action<TensorRtBuilderConfig, TensorRtProgressMonitor> attachProgressMonitor =
            static (config, monitor) => config.SetProgressMonitor(monitor);
        Action<TensorRtBuilderConfig> clearProgressMonitor = static config => config.ClearProgressMonitor();
        Func<TensorRtBuilderConfig, bool> hasProgressMonitor = static config => config.HasProgressMonitor;
        Func<TensorRtAllocatorCallbackOwner, TensorRtAllocatorDryRunResult> allocatorOwnerDryRunDiagnostic =
            static owner => owner.RunDryRunDiagnostic(new TensorRtAllocatorDryRunRequest(4096, 256, "package-consumer"));
        Func<TensorRtAllocatorCallbackOwner, TensorRtAllocatorNativeDryRunResult> allocatorOwnerNativeDryRunDiagnostic =
            static owner => owner.RunNativeDryRunDiagnostic(TensorRtApiLine.TensorRt11, new TensorRtAllocatorDryRunRequest(8192, 512, "package-consumer-native"));
        Func<TensorRtAllocatorCallbackOwner, TensorRtAllocatorOwnerStateDryRunResult> allocatorOwnerStateLedgerDryRunDiagnostic =
            static owner => owner.RunNativeStateLedgerDryRunDiagnostic(TensorRtApiLine.TensorRt11, new TensorRtAllocatorDryRunRequest(16384, 1024, "package-consumer-ledger"), "IGpuAllocator", 0);
        Func<TensorRtAllocatorCallbackOwner, TensorRtAllocatorCallbackOwnerSnapshot> allocatorOwnerLifecycleDiagnostic =
            static owner => owner.RunLifecycleDiagnostic(new TensorRtAllocatorDryRunRequest(4096, 256, "package-consumer-lifecycle"));
        Func<TensorRtAllocatorCallbackOwner, TensorRtAllocatorCallbackOwnerSnapshot> allocatorOwnerLifecycleSnapshot =
            static owner => owner.GetSnapshot("package-consumer-snapshot");
        Func<TensorRtAllocatorCallbackOwnerSnapshot, string> allocatorOwnerLifecycleSummary =
            static snapshot => snapshot.EvidenceKind + ":" +
                snapshot.RuntimeEvidenceKind + ":" +
                snapshot.RealCallbackRuntime + ":" +
                snapshot.IsRealCallbackRuntimeProof + ":" +
                snapshot.DevicePointerExposed + ":" +
                snapshot.DevicePointerProduced + ":" +
                snapshot.BorrowedPointerEscaped + ":" +
                snapshot.PointerFreeSurfaceReady + ":" +
                snapshot.ManagedKeepAliveReady + ":" +
                snapshot.DisposeReleaseReady;
        Func<TensorRtAllocatorCallbackOwner, TensorRtAllocatorLedgerSafetyGateResult> allocatorOwnerLedgerSafetyGate =
            static owner => TensorRtAllocatorLedgerSafetyGate.Evaluate(
                owner,
                TensorRtApiLine.TensorRt11,
                new TensorRtAllocatorDryRunRequest(16384, 1024, "package-consumer-ledger-safety"),
                "IGpuAllocator",
                0);
        Func<TensorRtAllocatorLedgerSafetyGateResult, string> allocatorOwnerLedgerSafetyGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.ManagedKeepAliveReady + ":" +
                gate.NativeLedgerDesignReady + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtAllocatorCallbackOwner, string> allocatorOwnerCallbackState =
            static owner => owner.CallbackInvocationCount + ":" + owner.CallbackFailureCount + ":" + (owner.LastCallbackException?.GetType().Name ?? "none") + ":" + owner.IsAttached + ":" + owner.IsDisposed + ":" + owner.LastDiagnostic;
        Func<TensorRtOutputAllocatorCallbackOwner> outputAllocatorOwnerFactory =
            static () => new TensorRtOutputAllocatorCallbackOwner();
        Func<TensorRtOutputAllocatorCallbackOwner, TensorRtOutputAllocatorCallbackOwnerSnapshot> outputAllocatorOwnerDesignDiagnostic =
            static owner => owner.RunDesignDiagnostic(
                TensorRtApiLine.TensorRt11,
                new TensorRtOutputAllocatorCallbackRequest(
                    "package_consumer_output_tensor",
                    4096UL,
                    256UL,
                    new long[] { 1, 1000 },
                    "package-consumer-output-allocator-owner-design",
                    hasCurrentMemory: true),
                0UL);
        Func<TensorRtOutputAllocatorCallbackOwnerSnapshot, TensorRtOutputAllocatorRuntimeProofPrecheckResult> outputAllocatorRuntimeProofPrecheck =
            static snapshot => TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate(snapshot);
        Func<TensorRtOutputAllocatorCallbackOwnerSnapshot, TensorRtOutputAllocatorAttachDetachDesignGateResult> outputAllocatorAttachDetachDesignGate =
            static snapshot => TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate(snapshot);
        Func<TensorRtOutputAllocatorAttachDetachDesignGateResult, string> outputAllocatorAttachDetachDesignGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.LineSupportsOutputAllocator + ":" +
                gate.AttachControlAvailable + ":" +
                gate.DetachClearControlAvailable + ":" +
                gate.LineSpecificAttachDetachReady + ":" +
                gate.NativeVTableReady + ":" +
                gate.OutputBufferOwnershipRuntimeReady + ":" +
                gate.DesignGateReady + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtOutputAllocatorCallbackOwnerSnapshot, TensorRtOutputBufferOwnershipSafetyGateResult> outputBufferOwnershipSafetyGate =
            static snapshot => TensorRtOutputBufferOwnershipSafetyGate.Evaluate(snapshot);
        Func<TensorRtOutputBufferOwnershipSafetyGateResult, string> outputBufferOwnershipSafetyGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.AttachDetachDesignGateReady + ":" +
                gate.CopiedCurrentMemoryMetadataReady + ":" +
                gate.CopiedShapeMetadataReady + ":" +
                gate.CopiedRequestMetadataReady + ":" +
                gate.SafetyGateReady + ":" +
                gate.OutputBufferOwnershipRuntimeReady + ":" +
                gate.CurrentMemoryReusePolicyReady + ":" +
                gate.BorrowedPointerEscapeBlocked + ":" +
                gate.OwnedDevicePointerReleasePolicyReady + ":" +
                gate.ShapeNotificationOrderingReady + ":" +
                gate.ReallocateOutputRuntimeReady + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtOutputAllocatorRuntimeProofPrecheckResult, string> outputAllocatorRuntimeProofPrecheckSummary =
            static precheck => precheck.EvidenceKind + ":" +
                precheck.RuntimeEvidenceKind + ":" +
                precheck.RealCallbackRuntime + ":" +
                precheck.IsRealCallbackRuntimeProof + ":" +
                precheck.LineSupportsOutputAllocator + ":" +
                precheck.AttachDetachDesignGateReady + ":" +
                precheck.AttachControlAvailable + ":" +
                precheck.DetachClearControlAvailable + ":" +
                precheck.ManagedOwnerStateMachineReady + ":" +
                precheck.NativeVTableReady + ":" +
                precheck.OutputBufferOwnershipSafetyGateReady + ":" +
                precheck.OutputBufferOwnershipRuntimeReady + ":" +
                precheck.CurrentMemoryReusePolicyReady + ":" +
                precheck.BorrowedPointerEscapeBlocked + ":" +
                precheck.OwnedDevicePointerReleasePolicyReady + ":" +
                precheck.ShapeNotificationOrderingReady + ":" +
                precheck.ReallocateOutputRuntimeReady + ":" +
                precheck.CanAttemptRuntimeProof + ":" +
                precheck.RuntimeProofBlocked + ":" +
                precheck.BlockedPrerequisiteCount;
        Func<TensorRtAllocatorLedgerSafetyGateResult, TensorRtOutputAllocatorRuntimeProofPrecheckResult, TensorRtDebugListenerRuntimeProofPrecheckResult, TensorRtCallbackAllocatorReadinessSnapshot> callbackAllocatorReadinessSnapshot =
            static (allocatorGate, outputPrecheck, debugPrecheck) => TensorRtCallbackAllocatorReadiness.Evaluate(
                allocatorGate,
                outputPrecheck,
                debugPrecheck);
        Func<TensorRtCallbackAllocatorReadinessSnapshot, string> callbackAllocatorReadinessSummary =
            static readiness => readiness.EvidenceKind + ":" +
                readiness.RuntimeEvidenceKind + ":" +
                readiness.RealCallbackRuntime + ":" +
                readiness.IsRealCallbackRuntimeProof + ":" +
                readiness.LoggerCallbackReady + ":" +
                readiness.ProfilerCallbackReady + ":" +
                readiness.ProgressMonitorCallbackReady + ":" +
                readiness.AllocatorOwnerDryRunReady + ":" +
                readiness.AllocatorLedgerSafetyGateReady + ":" +
                readiness.OutputAllocatorOwnerDesignReady + ":" +
                readiness.OutputAllocatorRuntimeGateReady + ":" +
                readiness.DebugListenerOwnerDesignReady + ":" +
                readiness.DebugListenerNoThrowVTableGateReady + ":" +
                readiness.DebugListenerRuntimeProofPrecheckReady + ":" +
                readiness.RealCallbackInvocationProofReady + ":" +
                readiness.IsPublishSafeForManagedCallbacks + ":" +
                readiness.IsRuntimeInvocationProofComplete + ":" +
                readiness.RuntimeProofBlocked + ":" +
                readiness.BlockedReasonCount;
        Func<TensorRtStreamIoInterfaceInfoDesignGateResult> streamIoInterfaceInfoDesignGate =
            static () => TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface(TensorRtApiLine.TensorRt11);
        Func<TensorRtAllocatorLedgerSafetyGateResult, TensorRtOutputAllocatorRuntimeProofPrecheckResult, TensorRtDebugListenerRuntimeProofPrecheckResult, TensorRtStreamIoInterfaceInfoDesignGateResult, TensorRtCallbackOwnerClosureMatrixResult> callbackOwnerClosureMatrix =
            static (allocatorGate, outputPrecheck, debugPrecheck, streamGate) => TensorRtCallbackOwnerClosureMatrix.Evaluate(
                allocatorGate,
                outputPrecheck,
                debugPrecheck,
                streamGate);
        Func<TensorRtCallbackOwnerClosureMatrixResult, string> callbackOwnerClosureMatrixSummary =
            static matrix => matrix.EvidenceKind + ":" +
                matrix.RuntimeEvidenceKind + ":" +
                matrix.RealCallbackRuntime + ":" +
                matrix.IsRealCallbackRuntimeProof + ":" +
                matrix.FamilyCount + ":" +
                matrix.DesignGateReadyFamilyCount + ":" +
                matrix.ClosureReadyFamilyCount + ":" +
                matrix.RuntimeProofAttemptReadyFamilyCount + ":" +
                matrix.PackageConsumerRuntimeProofReadyFamilyCount + ":" +
                matrix.RuntimeProofBlocked + ":" +
                matrix.DeferredRowsStillRequired + ":" +
                matrix.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwner> debugListenerOwnerFactory =
            static () => new TensorRtDebugListenerCallbackOwner();
        Func<TensorRtDebugListenerCallbackOwner, TensorRtDebugListenerCallbackOwnerSnapshot> debugListenerOwnerDesignDiagnostic =
            static owner => owner.RunDesignDiagnostic(
                TensorRtApiLine.TensorRt11,
                new TensorRtDebugListenerCallbackRequest(
                    "package_consumer_debug_tensor",
                    TensorRtDataType.Float,
                    TensorRtTensorLocation.Device,
                    new long[] { 1, 3, 224, 224 },
                    "package-consumer-debug-listener-owner-design",
                    isInput: true,
                    isExecutionTensor: true));
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, string> debugListenerOwnerDesignSummary =
            static snapshot => snapshot.EvidenceKind + ":" +
                snapshot.RuntimeEvidenceKind + ":" +
                snapshot.RealCallbackRuntime + ":" +
                snapshot.IsRealCallbackRuntimeProof + ":" +
                snapshot.ProcessDebugTensorCount + ":" +
                snapshot.DebugTensorPointerExposed + ":" +
                snapshot.DebugTensorPointerProduced + ":" +
                snapshot.BorrowedDebugTensorPointerEscaped;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerAttachDetachDesignGateResult> debugListenerAttachDetachDesignGate =
            static snapshot => TensorRtDebugListenerAttachDetachDesignGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerAttachDetachDesignGateResult, string> debugListenerAttachDetachDesignGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.AttachControlAvailable + ":" +
                gate.DetachClearControlAvailable + ":" +
                gate.LineSpecificAttachDetachReady + ":" +
                gate.NativeVTableReady + ":" +
                gate.BorrowedDebugTensorLifetimeReady + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerBorrowedTensorSafetyGateResult> debugListenerBorrowedTensorSafetyGate =
            static snapshot => TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerBorrowedTensorSafetyGateResult, string> debugListenerBorrowedTensorSafetyGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.AttachDetachDesignGateReady + ":" +
                gate.DebugTensorMetadataCopied + ":" +
                gate.PointerFreeSurfaceReady + ":" +
                gate.BorrowedDebugTensorPointerEscapeBlocked + ":" +
                gate.BorrowedDebugTensorLifetimeReady + ":" +
                gate.BorrowedDebugTensorDataLifetimeReady + ":" +
                gate.ProcessDebugTensorRuntimeReady + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerAttachVTableSafetyGateResult> debugListenerAttachVTableSafetyGate =
            static snapshot => TensorRtDebugListenerAttachVTableSafetyGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerAttachVTableSafetyGateResult, string> debugListenerAttachVTableSafetyGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.SafetyGateReady + ":" +
                gate.AttachControlAvailable + ":" +
                gate.StableNativeOwnerAddressReady + ":" +
                gate.NoThrowNativeVTableReady + ":" +
                gate.ExceptionToStatusMappingReady + ":" +
                gate.ProcessDebugTensorRuntimeReady + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeAttachNoThrowPreflightResult> debugListenerNativeAttachNoThrowPreflight =
            static snapshot => TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeAttachNoThrowPreflightResult, string> debugListenerNativeAttachNoThrowPreflightSummary =
            static preflight => preflight.EvidenceKind + ":" +
                preflight.RuntimeEvidenceKind + ":" +
                preflight.RealCallbackRuntime + ":" +
                preflight.IsRealCallbackRuntimeProof + ":" +
                preflight.PreflightReady + ":" +
                preflight.NativeAttachEntryLocated + ":" +
                preflight.NativeDetachEntryLocated + ":" +
                preflight.StableNativeOwnerAddressDesignReady + ":" +
                preflight.ManagedCallbackKeepAliveDesignReady + ":" +
                preflight.NoThrowVTableDesignReady + ":" +
                preflight.ExceptionToStatusMappingDesignReady + ":" +
                preflight.CanImplementNativeAttach + ":" +
                preflight.CanAttemptRuntimeProof + ":" +
                preflight.RuntimeProofBlocked + ":" +
                preflight.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeOwnerAddressDesignGateResult> debugListenerNativeOwnerAddressDesignGate =
            static snapshot => TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeOwnerAddressDesignGateResult, string> debugListenerNativeOwnerAddressDesignGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.DesignGateReady + ":" +
                gate.NativeAttachNoThrowPreflightReady + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.NativeDetachEntryLocated + ":" +
                gate.StableNativeOwnerAddressReady + ":" +
                gate.StableNativeOwnerAddressDesignReady + ":" +
                gate.NativeOwnerNonCopyableReady + ":" +
                gate.NativeOwnerDisposeOrderReady + ":" +
                gate.NativeOwnerReleaseHookReady + ":" +
                gate.NativeOwnerInFlightDrainReady + ":" +
                gate.NoThrowNativeDestructorReady + ":" +
                gate.NativeOwnerLifecycleReady + ":" +
                gate.NativeVTableDesignReady + ":" +
                gate.CanImplementNativeAttach + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeNoThrowVTableDesignGateResult> debugListenerNativeNoThrowVTableDesignGate =
            static snapshot => TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeNoThrowVTableDesignGateResult, string> debugListenerNativeNoThrowVTableDesignGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.DesignGateReady + ":" +
                gate.NativeOwnerAddressDesignGateReady + ":" +
                gate.NativeAttachNoThrowPreflightReady + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.NativeOwnerLifecycleReady + ":" +
                gate.NoThrowNativeDestructorReady + ":" +
                gate.NoThrowVTableDesignReady + ":" +
                gate.ExceptionToStatusMappingDesignReady + ":" +
                gate.NativeVTableTrampolineReady + ":" +
                gate.CallbackExceptionCaptureReady + ":" +
                gate.CallbackStatusMappingReady + ":" +
                gate.CallbackInFlightAccountingReady + ":" +
                gate.NativeVTableDesignReady + ":" +
                gate.CanImplementNativeAttach + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeAttachEntryDesignGateResult> debugListenerNativeAttachEntryDesignGate =
            static snapshot => TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeAttachEntryDesignGateResult, string> debugListenerNativeAttachEntryDesignGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.DesignGateReady + ":" +
                gate.NativeNoThrowVTableDesignGateReady + ":" +
                gate.NativeOwnerAddressDesignGateReady + ":" +
                gate.NativeAttachNoThrowPreflightReady + ":" +
                gate.NativeDetachEntryLocated + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.LineSpecificAttachEntryDesignReady + ":" +
                gate.AttachEntryNoThrowReady + ":" +
                gate.AttachEntryVersionGuardReady + ":" +
                gate.AttachEntryOwnershipReady + ":" +
                gate.DetachBeforeReleaseReady + ":" +
                gate.NativeOwnerLifecycleReady + ":" +
                gate.NativeVTableDesignReady + ":" +
                gate.CanImplementNativeAttach + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult> debugListenerNativeDetachBeforeReleaseDesignGate =
            static snapshot => TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult, string> debugListenerNativeDetachBeforeReleaseDesignGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.DesignGateReady + ":" +
                gate.NativeAttachEntryDesignGateReady + ":" +
                gate.NativeNoThrowVTableDesignGateReady + ":" +
                gate.NativeOwnerAddressDesignGateReady + ":" +
                gate.NativeDetachEntryLocated + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.LineSpecificAttachEntryDesignReady + ":" +
                gate.AttachEntryNoThrowReady + ":" +
                gate.AttachEntryVersionGuardReady + ":" +
                gate.AttachEntryOwnershipReady + ":" +
                gate.DetachBeforeReleaseReady + ":" +
                gate.ReleaseHookOrderingReady + ":" +
                gate.DisposeIdempotencyReady + ":" +
                gate.InFlightDrainBeforeReleaseReady + ":" +
                gate.CallbackStateUnpinAfterDetachReady + ":" +
                gate.DelegateUnpinAfterDetachReady + ":" +
                gate.NativeOwnerLifecycleReady + ":" +
                gate.NativeVTableDesignReady + ":" +
                gate.CanImplementNativeAttach + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeOwnerLifecycleDryRunResult> debugListenerNativeOwnerLifecycleDryRun =
            static snapshot => TensorRtDebugListenerNativeOwnerLifecycleDryRun.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeOwnerLifecycleDryRunResult, string> debugListenerNativeOwnerLifecycleDryRunSummary =
            static dryRun => dryRun.EvidenceKind + ":" +
                dryRun.CallbackKind + ":" +
                dryRun.RuntimeEvidenceKind + ":" +
                dryRun.RealCallbackRuntime + ":" +
                dryRun.IsRealCallbackRuntimeProof + ":" +
                dryRun.Line + ":" +
                dryRun.OwnerId + ":" +
                dryRun.LastStatus + ":" +
                dryRun.ReleaseHookCount + ":" +
                dryRun.InFlightCallbackCount + ":" +
                dryRun.CallbackStatePinned + ":" +
                dryRun.DelegatePinned + ":" +
                dryRun.DisposeRequested + ":" +
                dryRun.LastDiagnostic + ":" +
                dryRun.ReleaseDiagnostic + ":" +
                dryRun.NativeDetachBeforeReleaseDesignGateReady + ":" +
                dryRun.NativeAttachEntryDesignGateReady + ":" +
                dryRun.NativeNoThrowVTableDesignGateReady + ":" +
                dryRun.NativeOwnerAddressDesignGateReady + ":" +
                dryRun.NativeDetachEntryLocated + ":" +
                dryRun.NativeAttachEntryLocated + ":" +
                dryRun.StableNativeOwnerIdentityReady + ":" +
                dryRun.NativeOwnerNonCopyableReady + ":" +
                dryRun.NativeOwnerDisposeOrderReady + ":" +
                dryRun.NativeOwnerReleaseHookReady + ":" +
                dryRun.NativeOwnerInFlightDrainReady + ":" +
                dryRun.DetachBeforeReleaseReady + ":" +
                dryRun.ReleaseHookOrderingReady + ":" +
                dryRun.DisposeIdempotencyReady + ":" +
                dryRun.InFlightDrainBeforeReleaseReady + ":" +
                dryRun.CallbackStateUnpinAfterDetachReady + ":" +
                dryRun.DelegateUnpinAfterDetachReady + ":" +
                dryRun.NoThrowNativeDestructorReady + ":" +
                dryRun.NativeOwnerLifecycleReady + ":" +
                dryRun.NativeVTableDesignReady + ":" +
                dryRun.ManagedCallbackKeepAliveDesignReady + ":" +
                dryRun.BorrowedDebugTensorMetadataCopyDesignReady + ":" +
                dryRun.BorrowedDebugTensorPointerEscapeBlocked + ":" +
                dryRun.DryRunReady + ":" +
                dryRun.BorrowedDebugTensorLifetimeRuntimeReady + ":" +
                dryRun.BorrowedDebugTensorDataLifetimeRuntimeReady + ":" +
                dryRun.ProcessDebugTensorRuntimeReady + ":" +
                dryRun.FullPackageConsumerRuntimeEvidenceReady + ":" +
                dryRun.CanImplementNativeAttach + ":" +
                dryRun.CanAttemptRuntimeProof + ":" +
                dryRun.RuntimeProofBlocked + ":" +
                dryRun.DeferredRowsStillRequired + ":" +
                dryRun.BlockedPrerequisiteCount + ":" +
                dryRun.Status + ":" +
                dryRun.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult> debugListenerNativeAttachEntryRuntimeScaffold =
            static snapshot => TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult, string> debugListenerNativeAttachEntryRuntimeScaffoldSummary =
            static scaffold => scaffold.EvidenceKind + ":" +
                scaffold.CallbackKind + ":" +
                scaffold.RuntimeEvidenceKind + ":" +
                scaffold.RealCallbackRuntime + ":" +
                scaffold.IsRealCallbackRuntimeProof + ":" +
                scaffold.Line + ":" +
                scaffold.OwnerId + ":" +
                scaffold.LastStatus + ":" +
                scaffold.NativeOwnerLifecycleDryRunReady + ":" +
                scaffold.NativeAttachEntryLocated + ":" +
                scaffold.NativeDetachEntryLocated + ":" +
                scaffold.AttachEntryParameterShapeReady + ":" +
                scaffold.AttachEntryVersionGuardReady + ":" +
                scaffold.AttachEntryNoThrowBoundaryReady + ":" +
                scaffold.AttachEntryOwnershipDiagnosticsReady + ":" +
                scaffold.StableNativeOwnerIdentityReady + ":" +
                scaffold.NativeOwnerNonCopyableReady + ":" +
                scaffold.NoThrowNativeDestructorReady + ":" +
                scaffold.NativeOwnerLifecycleReady + ":" +
                scaffold.RuntimeScaffoldReady + ":" +
                scaffold.ProcessDebugTensorRuntimeReady + ":" +
                scaffold.FullPackageConsumerRuntimeEvidenceReady + ":" +
                scaffold.CanImplementNativeAttach + ":" +
                scaffold.CanAttemptRuntimeProof + ":" +
                scaffold.RuntimeProofBlocked + ":" +
                scaffold.DeferredRowsStillRequired + ":" +
                scaffold.BlockedPrerequisiteCount + ":" +
                scaffold.Status + ":" +
                scaffold.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult> debugListenerNativeAttachEntryMinimalSafety =
            static snapshot => TensorRtDebugListenerNativeAttachEntryMinimalSafety.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult, string> debugListenerNativeAttachEntryMinimalSafetySummary =
            static safety => safety.EvidenceKind + ":" +
                safety.CallbackKind + ":" +
                safety.RuntimeEvidenceKind + ":" +
                safety.RealCallbackRuntime + ":" +
                safety.IsRealCallbackRuntimeProof + ":" +
                safety.Line + ":" +
                safety.OwnerId + ":" +
                safety.LastStatus + ":" +
                safety.RuntimeScaffoldReady + ":" +
                safety.LifecycleGateReady + ":" +
                safety.LifecyclePointerFree + ":" +
                safety.NativeAttachEntryLocated + ":" +
                safety.NativeDetachEntryLocated + ":" +
                safety.AttachEntryParameterShapeReady + ":" +
                safety.AttachEntryNoThrowReady + ":" +
                safety.AttachEntryVersionGuardReady + ":" +
                safety.AttachEntryOwnershipDiagnosticsReady + ":" +
                safety.SetDebugListenerNonNullEnabled + ":" +
                safety.NonNullAttachStillDisabled + ":" +
                safety.NativeAttachWouldBeBlocked + ":" +
                safety.MinimalSafetyReady + ":" +
                safety.ProcessDebugTensorRuntimeReady + ":" +
                safety.FullPackageConsumerRuntimeEvidenceReady + ":" +
                safety.CanImplementNativeAttach + ":" +
                safety.CanAttemptRuntimeProof + ":" +
                safety.RuntimeProofBlocked + ":" +
                safety.DeferredRowsStillRequired + ":" +
                safety.ReasonNativeAttachStillBlocked + ":" +
                safety.BlockedPrerequisiteCount + ":" +
                safety.Status + ":" +
                safety.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeOwnerStableIdentityResult> debugListenerNativeOwnerStableIdentity =
            static snapshot => TensorRtDebugListenerNativeOwnerStableIdentity.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeOwnerStableIdentityResult, string> debugListenerNativeOwnerStableIdentitySummary =
            static identity => identity.EvidenceKind + ":" +
                identity.CallbackKind + ":" +
                identity.RuntimeEvidenceKind + ":" +
                identity.RealCallbackRuntime + ":" +
                identity.IsRealCallbackRuntimeProof + ":" +
                identity.Line + ":" +
                identity.OwnerId + ":" +
                identity.LastStatus + ":" +
                identity.LastDiagnostic + ":" +
                identity.ReleaseDiagnostic + ":" +
                identity.NativeAttachEntryRuntimeScaffoldReady + ":" +
                identity.StableNativeOwnerIdentityReady + ":" +
                identity.NativeOwnerNonCopyableReady + ":" +
                identity.OwnerIdentityDiagnosticsReady + ":" +
                identity.OwnerIdentityPointerFree + ":" +
                identity.NativeAttachEntryLocated + ":" +
                identity.NativeDetachEntryLocated + ":" +
                identity.NoThrowNativeDestructorReady + ":" +
                identity.NativeOwnerLifecycleReady + ":" +
                identity.ProcessDebugTensorRuntimeReady + ":" +
                identity.FullPackageConsumerRuntimeEvidenceReady + ":" +
                identity.CanImplementNativeAttach + ":" +
                identity.CanAttemptRuntimeProof + ":" +
                identity.RuntimeProofBlocked + ":" +
                identity.DeferredRowsStillRequired + ":" +
                identity.BlockedPrerequisiteCount + ":" +
                identity.Status + ":" +
                identity.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeOwnerNonCopyableStorageResult> debugListenerNativeOwnerNonCopyableStorage =
            static snapshot => TensorRtDebugListenerNativeOwnerNonCopyableStorage.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeOwnerNonCopyableStorageResult, string> debugListenerNativeOwnerNonCopyableStorageSummary =
            static storage => storage.EvidenceKind + ":" +
                storage.CallbackKind + ":" +
                storage.RuntimeEvidenceKind + ":" +
                storage.RealCallbackRuntime + ":" +
                storage.IsRealCallbackRuntimeProof + ":" +
                storage.Line + ":" +
                storage.OwnerId + ":" +
                storage.LastStatus + ":" +
                storage.LastDiagnostic + ":" +
                storage.ReleaseDiagnostic + ":" +
                storage.NativeOwnerStableIdentityReady + ":" +
                storage.OwnerIdentityDiagnosticsReady + ":" +
                storage.OwnerIdentityPointerFree + ":" +
                storage.NativeOwnerNonCopyableReady + ":" +
                storage.NativeOwnerCopyBlocked + ":" +
                storage.NativeOwnerMoveBlocked + ":" +
                storage.NativeOwnerAddressExposed + ":" +
                storage.NativeOwnerPointerProduced + ":" +
                storage.NativeAttachEntryLocated + ":" +
                storage.NativeDetachEntryLocated + ":" +
                storage.NoThrowNativeDestructorReady + ":" +
                storage.NativeOwnerLifecycleReady + ":" +
                storage.ProcessDebugTensorRuntimeReady + ":" +
                storage.FullPackageConsumerRuntimeEvidenceReady + ":" +
                storage.CanImplementNativeAttach + ":" +
                storage.CanAttemptRuntimeProof + ":" +
                storage.RuntimeProofBlocked + ":" +
                storage.DeferredRowsStillRequired + ":" +
                storage.BlockedPrerequisiteCount + ":" +
                storage.Status + ":" +
                storage.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeNoThrowDestructorResult> debugListenerNativeNoThrowDestructor =
            static snapshot => TensorRtDebugListenerNativeNoThrowDestructor.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeNoThrowDestructorResult, string> debugListenerNativeNoThrowDestructorSummary =
            static destructor => destructor.EvidenceKind + ":" +
                destructor.CallbackKind + ":" +
                destructor.RuntimeEvidenceKind + ":" +
                destructor.RealCallbackRuntime + ":" +
                destructor.IsRealCallbackRuntimeProof + ":" +
                destructor.Line + ":" +
                destructor.OwnerId + ":" +
                destructor.LastStatus + ":" +
                destructor.LastDiagnostic + ":" +
                destructor.ReleaseDiagnostic + ":" +
                destructor.NativeOwnerNonCopyableStorageReady + ":" +
                destructor.NativeOwnerNonCopyableReady + ":" +
                destructor.NativeOwnerCopyBlocked + ":" +
                destructor.NativeOwnerMoveBlocked + ":" +
                destructor.NativeOwnerAddressExposed + ":" +
                destructor.NativeOwnerPointerProduced + ":" +
                destructor.DestructorNoThrowScaffoldReady + ":" +
                destructor.DestructorExceptionEscapeBlocked + ":" +
                destructor.DestructorAddressExposed + ":" +
                destructor.DestructorPointerProduced + ":" +
                destructor.NativeAttachEntryLocated + ":" +
                destructor.NativeDetachEntryLocated + ":" +
                destructor.NoThrowNativeDestructorReady + ":" +
                destructor.NativeOwnerLifecycleReady + ":" +
                destructor.ProcessDebugTensorRuntimeReady + ":" +
                destructor.FullPackageConsumerRuntimeEvidenceReady + ":" +
                destructor.CanImplementNativeAttach + ":" +
                destructor.CanAttemptRuntimeProof + ":" +
                destructor.RuntimeProofBlocked + ":" +
                destructor.DeferredRowsStillRequired + ":" +
                destructor.BlockedPrerequisiteCount + ":" +
                destructor.Status + ":" +
                destructor.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeOwnerLifecycleGateResult> debugListenerNativeOwnerLifecycleGate =
            static snapshot => TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeOwnerLifecycleGateResult, string> debugListenerNativeOwnerLifecycleGateSummary =
            static lifecycle => lifecycle.EvidenceKind + ":" +
                lifecycle.CallbackKind + ":" +
                lifecycle.RuntimeEvidenceKind + ":" +
                lifecycle.RealCallbackRuntime + ":" +
                lifecycle.IsRealCallbackRuntimeProof + ":" +
                lifecycle.Line + ":" +
                lifecycle.OwnerId + ":" +
                lifecycle.LastStatus + ":" +
                lifecycle.NativeNoThrowDestructorGateReady + ":" +
                lifecycle.ManagedDisposeSnapshotReady + ":" +
                lifecycle.LifecycleScaffoldReady + ":" +
                lifecycle.ReleaseHookOrderingGateReady + ":" +
                lifecycle.DisposeIdempotencyGateReady + ":" +
                lifecycle.InFlightDrainGateReady + ":" +
                lifecycle.CallbackStateUnpinAfterDetachGateReady + ":" +
                lifecycle.DelegateUnpinAfterDetachGateReady + ":" +
                lifecycle.LifecycleAddressExposed + ":" +
                lifecycle.LifecyclePointerProduced + ":" +
                lifecycle.NativeAttachEntryLocated + ":" +
                lifecycle.NativeDetachEntryLocated + ":" +
                lifecycle.NoThrowNativeDestructorReady + ":" +
                lifecycle.ReleaseHookOrderingReady + ":" +
                lifecycle.DisposeIdempotencyReady + ":" +
                lifecycle.InFlightDrainBeforeReleaseReady + ":" +
                lifecycle.CallbackStateUnpinAfterDetachReady + ":" +
                lifecycle.DelegateUnpinAfterDetachReady + ":" +
                lifecycle.LifecycleGateReady + ":" +
                lifecycle.NativeOwnerLifecycleReady + ":" +
                lifecycle.NativeVTableDesignReady + ":" +
                lifecycle.ProcessDebugTensorRuntimeReady + ":" +
                lifecycle.FullPackageConsumerRuntimeEvidenceReady + ":" +
                lifecycle.CanImplementNativeAttach + ":" +
                lifecycle.CanAttemptRuntimeProof + ":" +
                lifecycle.RuntimeProofBlocked + ":" +
                lifecycle.DeferredRowsStillRequired + ":" +
                lifecycle.BlockedPrerequisiteCount + ":" +
                lifecycle.Status + ":" +
                lifecycle.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeAttachBridgeShapeGateResult> debugListenerNativeAttachBridgeShapeGate =
            static snapshot => TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeAttachBridgeShapeGateResult, string> debugListenerNativeAttachBridgeShapeGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.NativeOwnerLifecycleGateReady + ":" +
                gate.AttachBridgeShapeReady + ":" +
                gate.AttachBridgeNoThrowBoundaryReady + ":" +
                gate.AttachBridgeVersionGuardReady + ":" +
                gate.AttachBridgeOwnershipDiagnosticsReady + ":" +
                gate.AttachBridgePointerFree + ":" +
                gate.SetDebugListenerNonNullEnabled + ":" +
                gate.NonNullAttachStillDisabled + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.NativeOwnerLifecycleReady + ":" +
                gate.NativeVTableDesignReady + ":" +
                gate.AttachBridgeShapeGateReady + ":" +
                gate.CanImplementNativeAttach + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.DeferredRowsStillRequired + ":" +
                gate.BlockedPrerequisiteCount + ":" +
                gate.Status + ":" +
                gate.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerExceptionStatusMappingGateResult> debugListenerExceptionStatusMappingGate =
            static snapshot => TensorRtDebugListenerExceptionStatusMappingGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerExceptionStatusMappingGateResult, string> debugListenerExceptionStatusMappingGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.AttachBridgeShapeGateReady + ":" +
                gate.ManagedCallbackExceptionCaptureReady + ":" +
                gate.NativeCallbackExceptionCaptureReady + ":" +
                gate.CallbackStatusMappingGateReady + ":" +
                gate.ExceptionEscapeBlocked + ":" +
                gate.DiagnosticCopyReady + ":" +
                gate.MappingAddressExposed + ":" +
                gate.MappingPointerProduced + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.ExceptionStatusMappingGateReady + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.DeferredRowsStillRequired + ":" +
                gate.BlockedPrerequisiteCount + ":" +
                gate.Status + ":" +
                gate.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerInFlightAccountingGateResult> debugListenerInFlightAccountingGate =
            static snapshot => TensorRtDebugListenerInFlightAccountingGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerInFlightAccountingGateResult, string> debugListenerInFlightAccountingGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.ProcessDebugTensorCount + ":" +
                gate.InFlightCallbackCount + ":" +
                gate.MaxInFlightCallbackCount + ":" +
                gate.ReleaseHookCount + ":" +
                gate.ExceptionStatusMappingGateReady + ":" +
                gate.CallbackEnterAccountingGateReady + ":" +
                gate.CallbackLeaveAccountingGateReady + ":" +
                gate.CallbackInFlightNeverNegativeReady + ":" +
                gate.ReleaseAfterDrainGateReady + ":" +
                gate.CallbackStateUnpinAfterDrainGateReady + ":" +
                gate.AccountingAddressExposed + ":" +
                gate.AccountingPointerProduced + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.InFlightAccountingGateReady + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.DeferredRowsStillRequired + ":" +
                gate.BlockedPrerequisiteCount + ":" +
                gate.Status + ":" +
                gate.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult> debugListenerNativeNoThrowVTableScaffoldGate =
            static snapshot => TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult, string> debugListenerNativeNoThrowVTableScaffoldGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.NativeAttachBridgeShapeGateReady + ":" +
                gate.ExceptionStatusMappingGateReady + ":" +
                gate.InFlightAccountingGateReady + ":" +
                gate.NoThrowVTableScaffoldReady + ":" +
                gate.VTableDestructorNoThrowReady + ":" +
                gate.ProcessDebugTensorCallbackStubNoThrowReady + ":" +
                gate.ExceptionEscapeBlocked + ":" +
                gate.CallbackExceptionCaptureGateReady + ":" +
                gate.CallbackStatusMappingGateReady + ":" +
                gate.CallbackInFlightAccountingGateReady + ":" +
                gate.BorrowedDebugTensorPointerEscapeBlocked + ":" +
                gate.VTableAddressExposed + ":" +
                gate.VTablePointerProduced + ":" +
                gate.NativeAttachEntryLocated + ":" +
                gate.NativeVTableDesignReady + ":" +
                gate.VTableScaffoldGateReady + ":" +
                gate.CanImplementNativeAttach + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.DeferredRowsStillRequired + ":" +
                gate.BlockedPrerequisiteCount + ":" +
                gate.Status + ":" +
                gate.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNoThrowVTableCallbackStubResult> debugListenerNoThrowVTableCallbackStub =
            static snapshot => TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(snapshot);
        Func<TensorRtDebugListenerNoThrowVTableCallbackStubResult, string> debugListenerNoThrowVTableCallbackStubSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.CallbackStubGateReady + ":" +
                gate.CallbackStubShapeReady + ":" +
                gate.CallbackStubNoThrowReady + ":" +
                gate.CallbackMetadataCopyReady + ":" +
                gate.CallbackExceptionCaptureReady + ":" +
                gate.CallbackStatusMappingReady + ":" +
                gate.CallbackInFlightEnterReady + ":" +
                gate.CallbackInFlightLeaveReady + ":" +
                gate.CallbackInFlightPairingReady + ":" +
                gate.CallbackInFlightNeverNegativeReady + ":" +
                gate.BorrowedDebugTensorMetadataCopyReady + ":" +
                gate.BorrowedDebugTensorPointerEscapeBlocked + ":" +
                gate.DebugTensorPointerExposed + ":" +
                gate.DebugTensorDataPointerExposed + ":" +
                gate.SetDebugListenerNonNullEnabled + ":" +
                gate.NativeAttachWouldBeBlocked + ":" +
                gate.NativeVTableInstalled + ":" +
                gate.ProcessDebugTensorRuntimeReady + ":" +
                gate.CanInstallNativeVTable + ":" +
                gate.CanCallProcessDebugTensorRuntime + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.DeferredRowsStillRequired + ":" +
                gate.BlockedPrerequisiteCount + ":" +
                gate.ReasonCallbackRuntimeStillBlocked + ":" +
                gate.Status + ":" +
                gate.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult> debugListenerBorrowedDebugTensorMetadataRuntimeGate =
            static snapshot => TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(snapshot);
        Func<TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult, string> debugListenerBorrowedDebugTensorMetadataRuntimeGateSummary =
            static gate => gate.EvidenceKind + ":" +
                gate.RuntimeEvidenceKind + ":" +
                gate.RealCallbackRuntime + ":" +
                gate.IsRealCallbackRuntimeProof + ":" +
                gate.BorrowedTensorSafetyGateReady + ":" +
                gate.CallbackStubGateReady + ":" +
                gate.MetadataGateReady + ":" +
                gate.TensorNameCopied + ":" +
                gate.TensorNameLength + ":" +
                gate.TensorTypeCopied + ":" +
                gate.TensorLocationCopied + ":" +
                gate.TensorShapeCopied + ":" +
                gate.TensorShapeRank + ":" +
                gate.TensorFlagsCopied + ":" +
                gate.BorrowedDebugTensorMetadataCopyReady + ":" +
                gate.BorrowedDebugTensorPointerEscapeBlocked + ":" +
                gate.BorrowedDebugTensorDataPointerEscapeBlocked + ":" +
                gate.DebugTensorPointerExposed + ":" +
                gate.DebugTensorDataPointerExposed + ":" +
                gate.BorrowedDebugTensorLifetimeReady + ":" +
                gate.BorrowedDebugTensorDataLifetimeReady + ":" +
                gate.SetDebugListenerNonNullEnabled + ":" +
                gate.NativeVTableInstalled + ":" +
                gate.ProcessDebugTensorRuntimeReady + ":" +
                gate.CanCallProcessDebugTensorRuntime + ":" +
                gate.CanAttemptRuntimeProof + ":" +
                gate.RuntimeProofBlocked + ":" +
                gate.DeferredRowsStillRequired + ":" +
                gate.BlockedPrerequisiteCount + ":" +
                gate.ReasonMetadataRuntimeStillBlocked + ":" +
                gate.Status + ":" +
                gate.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeVTableInstallPreflightResult> debugListenerNativeVTableInstallPreflight =
            static snapshot => TensorRtDebugListenerNativeVTableInstallPreflight.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeVTableInstallPreflightResult, string> debugListenerNativeVTableInstallPreflightSummary =
            static preflight => preflight.EvidenceKind + ":" +
                preflight.RuntimeEvidenceKind + ":" +
                preflight.RealCallbackRuntime + ":" +
                preflight.IsRealCallbackRuntimeProof + ":" +
                preflight.NativeOwnerLifecycleGateReady + ":" +
                preflight.NativeAttachBridgeShapeGateReady + ":" +
                preflight.NativeNoThrowVTableScaffoldGateReady + ":" +
                preflight.BorrowedDebugTensorMetadataGateReady + ":" +
                preflight.VTableInstallShapeReady + ":" +
                preflight.VTableInstallVersionGuardReady + ":" +
                preflight.VTableInstallNoThrowBoundaryReady + ":" +
                preflight.VTableInstallOwnershipDiagnosticsReady + ":" +
                preflight.VTableInstallPointerFree + ":" +
                preflight.AttachBridgeSetDebugListenerNonNullEnabled + ":" +
                preflight.SetDebugListenerNonNullEnabled + ":" +
                preflight.NonNullAttachStillDisabled + ":" +
                preflight.VTableAddressExposed + ":" +
                preflight.VTablePointerProduced + ":" +
                preflight.DebugTensorPointerExposed + ":" +
                preflight.DebugTensorDataPointerExposed + ":" +
                preflight.BorrowedDebugTensorPointerEscapeBlocked + ":" +
                preflight.BorrowedDebugTensorDataPointerEscapeBlocked + ":" +
                preflight.BorrowedDebugTensorLifetimeReady + ":" +
                preflight.BorrowedDebugTensorDataLifetimeReady + ":" +
                preflight.NativeVTableInstallPreflightReady + ":" +
                preflight.NativeVTableInstalled + ":" +
                preflight.NativeVTableInstallRuntimeReady + ":" +
                preflight.CanEnableSetDebugListenerNonNull + ":" +
                preflight.CanInstallNativeVTable + ":" +
                preflight.ProcessDebugTensorRuntimeReady + ":" +
                preflight.CanCallProcessDebugTensorRuntime + ":" +
                preflight.FullPackageConsumerRuntimeEvidenceReady + ":" +
                preflight.CanAttemptRuntimeProof + ":" +
                preflight.RuntimeProofBlocked + ":" +
                preflight.DeferredRowsStillRequired + ":" +
                preflight.BlockedPrerequisiteCount + ":" +
                preflight.ReasonNativeVTableInstallStillBlocked + ":" +
                preflight.Status + ":" +
                preflight.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult> debugListenerNativeOwnerVTableInstallExperiment =
            static snapshot => TensorRtDebugListenerNativeOwnerVTableInstallExperiment.Evaluate(snapshot);
        Func<TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult, string> debugListenerNativeOwnerVTableInstallExperimentSummary =
            static experiment => experiment.EvidenceKind + ":" +
                experiment.RuntimeEvidenceKind + ":" +
                experiment.RealCallbackRuntime + ":" +
                experiment.IsRealCallbackRuntimeProof + ":" +
                experiment.NativeOwnerLifecycleGateReady + ":" +
                experiment.NativeAttachBridgeShapeGateReady + ":" +
                experiment.NativeNoThrowVTableScaffoldGateReady + ":" +
                experiment.BorrowedDebugTensorMetadataGateReady + ":" +
                experiment.NativeVTableInstallPreflightReady + ":" +
                experiment.ExperimentShapeReady + ":" +
                experiment.InstallAttemptGuardReady + ":" +
                experiment.NonNullAttachEnabled + ":" +
                experiment.RuntimeProofEnabled + ":" +
                experiment.NativeVTableInstallAttempted + ":" +
                experiment.NativeVTableInstalled + ":" +
                experiment.RollbackReady + ":" +
                experiment.DetachBeforeReleaseReady + ":" +
                experiment.FailureStatusMappingReady + ":" +
                experiment.PointerFree + ":" +
                experiment.VTableAddressExposed + ":" +
                experiment.VTablePointerProduced + ":" +
                experiment.DebugTensorPointerExposed + ":" +
                experiment.DebugTensorDataPointerExposed + ":" +
                experiment.CanEnableSetDebugListenerNonNull + ":" +
                experiment.CanInstallNativeVTable + ":" +
                experiment.ProcessDebugTensorRuntimeReady + ":" +
                experiment.CanCallProcessDebugTensorRuntime + ":" +
                experiment.CanAttemptRuntimeProof + ":" +
                experiment.RuntimeProofBlocked + ":" +
                experiment.DeferredRowsStillRequired + ":" +
                experiment.BlockedPrerequisiteCount + ":" +
                experiment.ReasonNativeOwnerVTableInstallStillBlocked + ":" +
                experiment.Status + ":" +
                experiment.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerRuntimeProofPrecheckResult> debugListenerRuntimeProofPrecheck =
            static snapshot => TensorRtDebugListenerRuntimeProofPrecheck.Evaluate(snapshot);
        Func<TensorRtDebugListenerRuntimeProofPrecheckResult, string> debugListenerRuntimeProofPrecheckSummary =
            static precheck => precheck.EvidenceKind + ":" +
                precheck.RuntimeEvidenceKind + ":" +
                precheck.RealCallbackRuntime + ":" +
                precheck.IsRealCallbackRuntimeProof + ":" +
                precheck.NativeNoThrowDestructorGateReady + ":" +
                precheck.DestructorNoThrowScaffoldReady + ":" +
                precheck.DestructorExceptionEscapeBlocked + ":" +
                precheck.DestructorAddressExposed + ":" +
                precheck.DestructorPointerProduced + ":" +
                precheck.NoThrowNativeDestructorReady + ":" +
                precheck.NativeOwnerLifecycleGateReady + ":" +
                precheck.ManagedDisposeSnapshotReady + ":" +
                precheck.LifecycleScaffoldReady + ":" +
                precheck.ReleaseHookOrderingGateReady + ":" +
                precheck.DisposeIdempotencyGateReady + ":" +
                precheck.InFlightDrainGateReady + ":" +
                precheck.CallbackStateUnpinAfterDetachGateReady + ":" +
                precheck.DelegateUnpinAfterDetachGateReady + ":" +
                precheck.LifecycleAddressExposed + ":" +
                precheck.LifecyclePointerProduced + ":" +
                precheck.NativeAttachBridgeShapeGateReady + ":" +
                precheck.AttachBridgeShapeReady + ":" +
                precheck.AttachBridgeNoThrowBoundaryReady + ":" +
                precheck.AttachBridgeVersionGuardReady + ":" +
                precheck.AttachBridgeOwnershipDiagnosticsReady + ":" +
                precheck.AttachBridgePointerFree + ":" +
                precheck.NonNullAttachStillDisabled + ":" +
                precheck.ExceptionStatusMappingGateReady + ":" +
                precheck.NativeCallbackExceptionCaptureReady + ":" +
                precheck.CallbackStatusMappingGateReady + ":" +
                precheck.ExceptionEscapeBlocked + ":" +
                precheck.DiagnosticCopyReady + ":" +
                precheck.InFlightAccountingGateReady + ":" +
                precheck.CallbackEnterAccountingGateReady + ":" +
                precheck.CallbackLeaveAccountingGateReady + ":" +
                precheck.CallbackInFlightNeverNegativeReady + ":" +
                precheck.ReleaseAfterDrainGateReady + ":" +
                precheck.CallbackStateUnpinAfterDrainGateReady + ":" +
                precheck.NativeNoThrowVTableScaffoldGateReady + ":" +
                precheck.NoThrowVTableScaffoldReady + ":" +
                precheck.VTableDestructorNoThrowReady + ":" +
                precheck.ProcessDebugTensorCallbackStubNoThrowReady + ":" +
                precheck.VTableAddressExposed + ":" +
                precheck.VTablePointerProduced + ":" +
                precheck.BorrowedTensorSafetyGateReady + ":" +
                precheck.NativeOwnerLifecycleDryRunReady + ":" +
                precheck.NativeAttachEntryRuntimeScaffoldReady + ":" +
                precheck.AttachEntryParameterShapeReady + ":" +
                precheck.AttachEntryNoThrowBoundaryReady + ":" +
                precheck.AttachEntryOwnershipDiagnosticsReady + ":" +
                precheck.BorrowedDebugTensorPointerEscapeBlocked + ":" +
                precheck.BorrowedDebugTensorDataLifetimeReady + ":" +
                precheck.ProcessDebugTensorRuntimeReady + ":" +
                precheck.CanAttemptRuntimeProof + ":" +
                precheck.RuntimeProofBlocked + ":" +
                precheck.BlockedPrerequisiteCount;
        Func<TensorRtDebugListenerRuntimeProofPrecheckResult, TensorRtDebugListenerRuntimeProofAttemptPreflightResult> debugListenerRuntimeProofAttemptPreflight =
            static precheck => TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(precheck);
        Func<TensorRtDebugListenerRuntimeProofAttemptPreflightResult, string> debugListenerRuntimeProofAttemptPreflightSummary =
            static preflight => preflight.EvidenceKind + ":" +
                preflight.RuntimeEvidenceKind + ":" +
                preflight.RealCallbackRuntime + ":" +
                preflight.IsRealCallbackRuntimeProof + ":" +
                preflight.CanEnableSetDebugListenerNonNull + ":" +
                preflight.CanInstallNativeVTable + ":" +
                preflight.CanCallProcessDebugTensorRuntime + ":" +
                preflight.CanPromoteRealCallbackRuntime + ":" +
                preflight.RuntimeProofBlocked + ":" +
                preflight.BlockedPrerequisiteCount + ":" +
                preflight.ReasonNonNullAttachStillBlocked + ":" +
                preflight.ReasonNativeVTableStillBlocked + ":" +
                preflight.ReasonRuntimeProofStillBlocked;
        Func<TensorRtDebugListenerRuntimeProofAttemptPreflightResult, TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult> debugListenerRealNonNullAttachRuntimeSmoke =
            static preflight => TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(preflight, "", optInEnabled: false, fullPackageConsumerReport: false);
        Func<TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult, string> debugListenerRealNonNullAttachRuntimeSmokeSummary =
            static smoke => smoke.EvidenceKind + ":" +
                smoke.RuntimeEvidenceKind + ":" +
                smoke.RealCallbackRuntime + ":" +
                smoke.IsRealCallbackRuntimeProof + ":" +
                smoke.CallbackKind + ":" +
                smoke.TensorRtLine + ":" +
                smoke.RuntimePackageKey + ":" +
                smoke.OptInEnabled + ":" +
                smoke.FullPackageConsumerReport + ":" +
                smoke.AttachGuardReady + ":" +
                smoke.NativeVTableReady + ":" +
                smoke.BorrowedDebugTensorRuntimeReady + ":" +
                smoke.CallbackInvocationReady + ":" +
                smoke.AttachAttempted + ":" +
                smoke.AttachSucceeded + ":" +
                smoke.DetachAttempted + ":" +
                smoke.DetachSucceeded + ":" +
                smoke.RollbackAttempted + ":" +
                smoke.RollbackSucceeded + ":" +
                smoke.NativeVTableInstalled + ":" +
                smoke.ProcessDebugTensorInvoked + ":" +
                smoke.InvocationCount + ":" +
                smoke.AllocationCount + ":" +
                smoke.ReleaseCount + ":" +
                smoke.FailureCount + ":" +
                smoke.InFlightCallbackCount + ":" +
                smoke.LastStatus + ":" +
                smoke.LastDiagnostic + ":" +
                smoke.ReportPointerFree + ":" +
                smoke.CanAttemptRuntimeProof + ":" +
                smoke.CanPromoteRealCallbackRuntime + ":" +
                smoke.RuntimeProofBlocked + ":" +
                smoke.DeferredRowsStillRequired + ":" +
                smoke.BlockedPrerequisiteCount + ":" +
                smoke.ReasonRuntimeProofStillBlocked + ":" +
                smoke.Status + ":" +
                smoke.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult> debugListenerProcessDebugTensorCallbackTrampoline =
            static snapshot => TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(
                TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(snapshot),
                TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(snapshot),
                TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(snapshot, "", optInEnabled: false, fullPackageConsumerReport: false));
        Func<TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult, string> debugListenerProcessDebugTensorCallbackTrampolineSummary =
            static trampoline => trampoline.EvidenceKind + ":" +
                trampoline.RuntimeEvidenceKind + ":" +
                trampoline.RealCallbackRuntime + ":" +
                trampoline.IsRealCallbackRuntimeProof + ":" +
                trampoline.CallbackKind + ":" +
                trampoline.TensorRtLine + ":" +
                trampoline.RuntimePackageKey + ":" +
                trampoline.TrampolineShapeReady + ":" +
                trampoline.NativeCallbackEntryLocated + ":" +
                trampoline.NoThrowCallbackEntryReady + ":" +
                trampoline.ExceptionCaptureReady + ":" +
                trampoline.CallbackStatusMappingReady + ":" +
                trampoline.InFlightAccountingReady + ":" +
                trampoline.DetachBeforeReleaseReady + ":" +
                trampoline.BorrowedDebugTensorMetadataCopyReady + ":" +
                trampoline.BorrowedDebugTensorPointerExposed + ":" +
                trampoline.BorrowedDebugTensorDataPointerExposed + ":" +
                trampoline.PointerFreeSurfaceReady + ":" +
                trampoline.ProcessDebugTensorRuntimeReady + ":" +
                trampoline.OptInEnabled + ":" +
                trampoline.FullPackageConsumerReport + ":" +
                trampoline.AttachAttempted + ":" +
                trampoline.AttachSucceeded + ":" +
                trampoline.NativeVTableInstalled + ":" +
                trampoline.ProcessDebugTensorInvoked + ":" +
                trampoline.InvocationCount + ":" +
                trampoline.CallbackStubEntryCount + ":" +
                trampoline.CallbackStubLeaveCount + ":" +
                trampoline.FailureCount + ":" +
                trampoline.InFlightCallbackCount + ":" +
                trampoline.LastStatus + ":" +
                trampoline.LastDiagnostic + ":" +
                trampoline.CanAttemptRuntimeProof + ":" +
                trampoline.CanPromoteRealCallbackRuntime + ":" +
                trampoline.RuntimeProofBlocked + ":" +
                trampoline.DeferredRowsStillRequired + ":" +
                trampoline.BlockedPrerequisiteCount + ":" +
                trampoline.Metadata.TensorName + ":" +
                trampoline.Metadata.TensorNameLength + ":" +
                trampoline.Metadata.DataType + ":" +
                trampoline.Metadata.Location + ":" +
                trampoline.Metadata.TensorShapeRank + ":" +
                trampoline.Metadata.MetadataCopied + ":" +
                trampoline.ReasonRuntimeProofStillBlocked + ":" +
                trampoline.Status + ":" +
                trampoline.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerRealCallbackRuntimeProofResult> debugListenerRealCallbackRuntimeProof =
            static snapshot => TensorRtDebugListenerRealCallbackRuntimeProof.Evaluate(
                TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(snapshot, "", optInEnabled: false, fullPackageConsumerReport: false),
                TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(snapshot, "", optInEnabled: false, fullPackageConsumerReport: false));
        Func<TensorRtDebugListenerRealCallbackRuntimeProofResult, string> debugListenerRealCallbackRuntimeProofSummary =
            static proof => proof.EvidenceKind + ":" +
                proof.RuntimeEvidenceKind + ":" +
                proof.RealCallbackRuntime + ":" +
                proof.IsRealCallbackRuntimeProof + ":" +
                proof.CallbackKind + ":" +
                proof.TensorRtLine + ":" +
                proof.RuntimePackageKey + ":" +
                proof.OptInEnabled + ":" +
                proof.FullPackageConsumerReport + ":" +
                proof.RuntimeSmokeReady + ":" +
                proof.TrampolineShapeReady + ":" +
                proof.AttachAttempted + ":" +
                proof.AttachSucceeded + ":" +
                proof.DetachAttempted + ":" +
                proof.DetachSucceeded + ":" +
                proof.RollbackAttempted + ":" +
                proof.RollbackSucceeded + ":" +
                proof.NativeVTableInstalled + ":" +
                proof.ProcessDebugTensorInvoked + ":" +
                proof.InvocationCount + ":" +
                proof.FailureCount + ":" +
                proof.InFlightCallbackCount + ":" +
                proof.BorrowedDebugTensorMetadataCopied + ":" +
                proof.PointerFreeSurfaceReady + ":" +
                proof.ProcessDebugTensorRuntimeReady + ":" +
                proof.AttemptedNoInvocation + ":" +
                proof.LastStatus + ":" +
                proof.LastDiagnostic + ":" +
                proof.CanAttemptRuntimeProof + ":" +
                proof.CanPromoteRealCallbackRuntime + ":" +
                proof.RuntimeProofBlocked + ":" +
                proof.DeferredRowsStillRequired + ":" +
                proof.BlockedPrerequisiteCount + ":" +
                proof.ReasonRuntimeProofStillBlocked + ":" +
                proof.Status + ":" +
                proof.Diagnostic;
        Func<TensorRtDebugListenerCallbackOwnerSnapshot, TensorRtDebugListenerCallbackProofGapReportResult> debugListenerCallbackProofGapReport =
            static snapshot =>
            {
                TensorRtDebugListenerRuntimeProofAttemptPreflightResult attempt =
                    TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(snapshot);
                TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult smoke =
                    TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(attempt, "", optInEnabled: false, fullPackageConsumerReport: false);
                TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult trampoline =
                    TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(snapshot, "", optInEnabled: false, fullPackageConsumerReport: false);
                TensorRtDebugListenerRealCallbackRuntimeProofResult proof =
                    TensorRtDebugListenerRealCallbackRuntimeProof.Evaluate(smoke, trampoline);
                return TensorRtDebugListenerCallbackProofGapReport.Evaluate(attempt, smoke, trampoline, proof);
            };
        Func<TensorRtDebugListenerCallbackProofGapReportResult, string> debugListenerCallbackProofGapReportSummary =
            static gap => gap.EvidenceKind + ":" +
                gap.RuntimeEvidenceKind + ":" +
                gap.RealCallbackRuntime + ":" +
                gap.IsRealCallbackRuntimeProof + ":" +
                gap.CallbackKind + ":" +
                gap.TensorRtLine + ":" +
                gap.RuntimePackageKey + ":" +
                gap.NonNullAttachStillDisabled + ":" +
                gap.NativeAttachEntryReady + ":" +
                gap.NativeVTableInstallBlocked + ":" +
                gap.NoThrowCallbackEntryReady + ":" +
                gap.ExceptionStatusMappingReady + ":" +
                gap.InFlightAccountingReady + ":" +
                gap.BorrowedDebugTensorMetadataCopied + ":" +
                gap.DetachRollbackReady + ":" +
                gap.ProcessDebugTensorRuntimeInvoked + ":" +
                gap.FullPackageConsumerRuntimeProofReady + ":" +
                gap.PointerFreeSurfaceReady + ":" +
                gap.AttemptedNoInvocation + ":" +
                gap.InvocationCount + ":" +
                gap.FailureCount + ":" +
                gap.InFlightCallbackCount + ":" +
                gap.CanAttemptRuntimeProof + ":" +
                gap.CanPromoteRealCallbackRuntime + ":" +
                gap.RuntimeProofBlocked + ":" +
                gap.DeferredRowsStillRequired + ":" +
                gap.GapReasonCount + ":" +
                gap.PrimaryGapReason + ":" +
                gap.RuntimeProofBlockerCategory + ":" +
                gap.PackageConsumerRuntimeProofRequired + ":" +
                gap.RuntimeInvocationRequired + ":" +
                gap.EvidenceSource + ":" +
                gap.NextOwnerAction + ":" +
                gap.Status + ":" +
                gap.Diagnostic;

        Func<CudaMemory, CudaMemoryRangeAttributeValue> rangeAttribute =
            static memory => memory.GetRangeAttribute(CudaMemoryRangeAttribute.PreferredLocation);
        Func<CudaMemory, CudaMemoryRangeAttributeValue[]> rangeAttributes =
            static memory => memory.GetRangeAttributes(CudaMemoryRangeAttribute.ReadMostly, CudaMemoryRangeAttribute.PreferredLocation);
        Func<CudaMemory, int[]> accessedByDevices = static memory => memory.GetRangeAccessedByDevices();
        Func<CudaMemory, CudaMemoryRangeDiagnosticSummary> memoryRangeDiagnosticSummary =
            static memory => memory.GetRangeDiagnosticSummary(CudaMemoryRangeAttribute.ReadMostly, CudaMemoryRangeAttribute.PreferredLocation);
        Func<CudaMemoryRangeDiagnosticSummary, string> memoryRangeDiagnosticSummaryText =
            static summary => summary.RangeSizeInBytes + ":" + summary.CopiedScalarAttributeCount + ":" + summary.CopiedAccessedByDeviceCount + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Func<CudaGraphDiagnosticSnapshot, CudaGraphDiagnosticSummary> graphDiagnosticSummary =
            static snapshot => snapshot.ToSummary();
        Func<CudaGraphDiagnosticSummary, string> graphDiagnosticSummaryText =
            static summary => summary.CopiedNodeSnapshotCount + ":" + summary.CopiedEdgeSnapshotCount + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Func<CudaGraphExecDiagnosticSnapshot, CudaGraphExecDiagnosticSummary> graphExecDiagnosticSummary =
            static snapshot => snapshot.ToSummary();
        Func<CudaGraphExecDiagnosticSummary, string> graphExecDiagnosticSummaryText =
            static summary => summary.CopiedNodeStateCount + ":" + summary.SnapshotsWithNodeTokenCount + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Func<CudaDeviceGraphMemoryInfo, CudaDeviceGraphMemorySummary> deviceGraphMemorySummary =
            static info => info.ToSummary();
        Func<CudaDeviceGraphMemorySummary, string> deviceGraphMemorySummaryText =
            static summary => summary.DeviceOrdinal + ":" + summary.CopiedScalarCounterCount + ":" + summary.CurrentCountersWithinHighWatermarks + ":" + summary.RuntimeEvidenceKind + ":" + summary.IsRuntimeExecutionEvidence + ":" + summary.IsRuntimeExecutionProof + ":" + summary.PointerFreeCopiedSummary + ":" + summary.CanPromoteRuntimeProof + ":" + summary.CanPromoteReleaseProof + ":" + summary.CanDeleteDeferredRecord;
        Func<int, CudaDeviceGraphMemorySummary> getDeviceGraphMemorySummary =
            static device => CudaDevice.GetGraphMemorySummary(device);
        Func<CudaDeviceGraphMemorySummary> currentDeviceGraphMemorySummary =
            static () => CudaDevice.CurrentGraphMemorySummary;
        Action<CudaMemory, int, int, CudaMemoryAdvice, int> adviseRange =
            static (memory, offset, count, advice, device) => memory.Advise(offset, count, advice, device);
        Action<CudaMemory, int, int, int, CudaStream> prefetchRange =
            static (memory, offset, count, destinationDevice, stream) => memory.PrefetchAsync(offset, count, destinationDevice, stream);

        _ = pluginInventory;
        _ = pluginCreatorLookup;
        _ = safePluginCreatorLookup;
        _ = globalPluginRegistryAvailable;
        _ = safeGlobalPluginRegistryAvailable;
        _ = globalPluginRegistryInventory;
        _ = globalPluginRegistryParentSearch;
        _ = setGlobalPluginRegistryParentSearch;
        _ = trySetGlobalPluginRegistryParentSearch;
        _ = creatorCount;
        _ = snapshotFindCreator;
        _ = snapshotTryFindCreator;
        _ = creatorSummaries;
        _ = limitedCreatorSummaries;
        _ = fieldSummaries;
        _ = limitedFieldSummaries;
        _ = creatorSummaryIdentity;
        _ = fieldSummaryIdentity;
        _ = creatorIdentity;
        _ = creatorTensorRtVersion;
        _ = creatorSummaryTensorRtVersion;
        _ = creatorFieldCount;
        _ = fieldMetadata;
        _ = pluginV2LayerMetadata;
        _ = safePluginV2LayerMetadata;
        _ = pluginV2LayerMetadataSummary;
        _ = pluginV2LegacyOutputDimensions;
        _ = pluginV2LegacyWorkspaceSize;
        _ = pluginV2LegacyFormat;
        _ = pluginV2OutputDataType;
        _ = pluginV2InputBroadcast;
        _ = pluginV2OutputBroadcast;
        _ = pluginV3LayerMetadata;
        _ = safePluginV3LayerMetadata;
        _ = pluginV3LayerMetadataSummary;
        _ = logHandler;
        _ = profilerHandler;
        _ = progressHandler;
        _ = allocatorDryRunHandler;
        _ = loggerFactory;
        _ = profilerFactory;
        _ = progressFactory;
        _ = allocatorOwnerFactory;
        _ = errorCodeExclusiveUpperBound;
        _ = directEngineBuild;
        _ = serializedPluginPathSet;
        _ = serializedPluginPathCopy;
        _ = legacyShapeBindingSet;
        _ = refitterHasLogger;
        _ = runtimeErrorRecorderSnapshot;
        _ = runtimeDiagnosticSnapshot;
        _ = runtimeDiagnosticSummary;
        _ = runtimeDiagnosticSummaryText;
        _ = engineDeploymentSummary;
        _ = engineDeploymentSummaryText;
        _ = builderConfigDeploymentSummary;
        _ = builderConfigDeploymentSummaryText;
        _ = executionContextDeploymentSummary;
        _ = executionContextDeploymentSummaryText;
        _ = serializationConfigSummary;
        _ = serializationConfigSummaryText;
        _ = runtimeConfigSummary;
        _ = runtimeConfigSummaryText;
        _ = runtimeHasLogger;
        _ = runtimeClearErrorRecorder;
        _ = runtimeClearGpuAllocator;
        _ = refitterErrorRecorderSnapshot;
        _ = refitterDiagnosticSnapshot;
        _ = refitterDiagnosticSummary;
        _ = refitterDiagnosticSummaryText;
        _ = onnxParserDiagnosticSnapshot;
        _ = onnxParserDiagnosticSummary;
        _ = onnxParserDiagnosticSummaryText;
        _ = onnxModelSupportSummary;
        _ = onnxModelSupportSummaryText;
        _ = onnxParserRefitterDiagnosticSnapshot;
        _ = onnxParserRefitterDiagnosticSummary;
        _ = onnxParserRefitterDiagnosticSummaryText;
        _ = refitterClearErrorRecorder;
        _ = builderHasErrorRecorder;
        _ = builderHasLogger;
        _ = builderClearErrorRecorder;
        _ = builderClearGpuAllocator;
        _ = contextHasErrorRecorder;
        _ = contextClearErrorRecorder;
        _ = contextHasOutputAllocator;
        _ = contextClearOutputAllocator;
        _ = contextHasTemporaryStorageAllocator;
        _ = contextClearTemporaryStorageAllocator;
        _ = contextHasDebugListener;
        _ = contextClearDebugListener;
        _ = contextOutputAllocatorInterfaceInfo;
        _ = contextTemporaryStorageAllocatorInterfaceInfo;
        _ = contextDebugListenerInterfaceInfo;
        _ = contextCallbackStateSnapshot;
        _ = contextClearCallbackState;
        _ = callbackStateSummary;
        _ = contextRuntimeDiagnosticSnapshot;
        _ = contextRuntimeDiagnosticSummary;
        _ = contextRuntimeDiagnosticSummaryText;
        _ = contextCallbackAllocatorSafeControlSummary;
        _ = callbackAllocatorSafeControlSummaryText;
        _ = loggerDiagnostic;
        _ = loggerCallbackState;
        _ = loggerInterfaceInfo;
        _ = tryLoggerInterfaceInfo;
        _ = loggerApiLanguage;
        _ = tryLoggerApiLanguage;
        _ = profilerDiagnostic;
        _ = profilerCallbackState;
        _ = profilerInterfaceInfo;
        _ = tryProfilerInterfaceInfo;
        _ = profilerApiLanguage;
        _ = tryProfilerApiLanguage;
        _ = attachProfiler;
        _ = clearProfiler;
        _ = hasManagedProfiler;
        _ = hasNativeProfiler;
        _ = progressDiagnostic;
        _ = progressCallbackState;
        _ = progressInterfaceInfo;
        _ = tryProgressInterfaceInfo;
        _ = progressApiLanguage;
        _ = tryProgressApiLanguage;
        _ = attachProgressMonitor;
        _ = clearProgressMonitor;
        _ = hasProgressMonitor;
        _ = allocatorOwnerDryRunDiagnostic;
        _ = allocatorOwnerNativeDryRunDiagnostic;
        _ = allocatorOwnerStateLedgerDryRunDiagnostic;
        _ = allocatorOwnerLedgerSafetyGate;
        _ = allocatorOwnerLedgerSafetyGateSummary;
        _ = allocatorOwnerCallbackState;
        _ = outputAllocatorOwnerFactory;
        _ = outputAllocatorOwnerDesignDiagnostic;
        _ = outputAllocatorAttachDetachDesignGate;
        _ = outputAllocatorAttachDetachDesignGateSummary;
        _ = outputBufferOwnershipSafetyGate;
        _ = outputBufferOwnershipSafetyGateSummary;
        _ = outputAllocatorRuntimeProofPrecheck;
        _ = outputAllocatorRuntimeProofPrecheckSummary;
        _ = callbackAllocatorReadinessSnapshot;
        _ = callbackAllocatorReadinessSummary;
        _ = streamIoInterfaceInfoDesignGate;
        _ = callbackOwnerClosureMatrix;
        _ = callbackOwnerClosureMatrixSummary;
        _ = debugListenerOwnerFactory;
        _ = debugListenerOwnerDesignDiagnostic;
        _ = debugListenerOwnerDesignSummary;
        _ = debugListenerAttachDetachDesignGate;
        _ = debugListenerAttachDetachDesignGateSummary;
        _ = debugListenerBorrowedTensorSafetyGate;
        _ = debugListenerBorrowedTensorSafetyGateSummary;
        _ = debugListenerAttachVTableSafetyGate;
        _ = debugListenerAttachVTableSafetyGateSummary;
        _ = debugListenerNativeAttachNoThrowPreflight;
        _ = debugListenerNativeAttachNoThrowPreflightSummary;
        _ = debugListenerNativeOwnerAddressDesignGate;
        _ = debugListenerNativeOwnerAddressDesignGateSummary;
        _ = debugListenerNativeNoThrowVTableDesignGate;
        _ = debugListenerNativeNoThrowVTableDesignGateSummary;
        _ = debugListenerNativeAttachEntryDesignGate;
        _ = debugListenerNativeAttachEntryDesignGateSummary;
        _ = debugListenerNativeDetachBeforeReleaseDesignGate;
        _ = debugListenerNativeDetachBeforeReleaseDesignGateSummary;
        _ = debugListenerNativeOwnerLifecycleDryRun;
        _ = debugListenerNativeOwnerLifecycleDryRunSummary;
        _ = debugListenerNativeAttachEntryRuntimeScaffold;
        _ = debugListenerNativeAttachEntryRuntimeScaffoldSummary;
        _ = debugListenerNativeAttachEntryMinimalSafety;
        _ = debugListenerNativeAttachEntryMinimalSafetySummary;
        _ = debugListenerNativeOwnerStableIdentity;
        _ = debugListenerNativeOwnerStableIdentitySummary;
        _ = debugListenerNativeOwnerNonCopyableStorage;
        _ = debugListenerNativeOwnerNonCopyableStorageSummary;
        _ = debugListenerNativeNoThrowDestructor;
        _ = debugListenerNativeNoThrowDestructorSummary;
        _ = debugListenerNativeOwnerLifecycleGate;
        _ = debugListenerNativeOwnerLifecycleGateSummary;
        _ = debugListenerRuntimeProofPrecheck;
        _ = debugListenerRuntimeProofPrecheckSummary;
        _ = debugListenerNativeAttachBridgeShapeGate;
        _ = debugListenerNativeAttachBridgeShapeGateSummary;
        _ = debugListenerExceptionStatusMappingGate;
        _ = debugListenerExceptionStatusMappingGateSummary;
        _ = debugListenerInFlightAccountingGate;
        _ = debugListenerInFlightAccountingGateSummary;
        _ = debugListenerNativeNoThrowVTableScaffoldGate;
        _ = debugListenerNativeNoThrowVTableScaffoldGateSummary;
        _ = debugListenerNoThrowVTableCallbackStub;
        _ = debugListenerNoThrowVTableCallbackStubSummary;
        _ = debugListenerBorrowedDebugTensorMetadataRuntimeGate;
        _ = debugListenerBorrowedDebugTensorMetadataRuntimeGateSummary;
        _ = debugListenerNativeVTableInstallPreflight;
        _ = debugListenerNativeVTableInstallPreflightSummary;
        _ = debugListenerNativeOwnerVTableInstallExperiment;
        _ = debugListenerNativeOwnerVTableInstallExperimentSummary;
        _ = debugListenerRuntimeProofAttemptPreflight;
        _ = debugListenerRuntimeProofAttemptPreflightSummary;
        _ = debugListenerRealNonNullAttachRuntimeSmoke;
        _ = debugListenerRealNonNullAttachRuntimeSmokeSummary;
        _ = debugListenerProcessDebugTensorCallbackTrampoline;
        _ = debugListenerProcessDebugTensorCallbackTrampolineSummary;
        _ = debugListenerRealCallbackRuntimeProof;
        _ = debugListenerRealCallbackRuntimeProofSummary;
        _ = debugListenerCallbackProofGapReport;
        _ = debugListenerCallbackProofGapReportSummary;
        _ = rangeAttribute;
        _ = rangeAttributes;
        _ = accessedByDevices;
        _ = memoryRangeDiagnosticSummary;
        _ = memoryRangeDiagnosticSummaryText;
        _ = graphDiagnosticSummary;
        _ = graphDiagnosticSummaryText;
        _ = graphExecDiagnosticSummary;
        _ = graphExecDiagnosticSummaryText;
        _ = deviceGraphMemorySummary;
        _ = deviceGraphMemorySummaryText;
        _ = getDeviceGraphMemorySummary;
        _ = currentDeviceGraphMemorySummary;
        _ = adviseRange;
        _ = prefetchRange;
        _ = executeV2;
        _ = executeLegacy;
        _ = enqueueV2AndSynchronize;
        _ = engineImplicitBatchCompatibility;
        _ = serializedPluginPathCountCompatibility;
        _ = pluginInventoryDiagnostics;
        _ = pluginInventoryDiagnosticsText;
        _ = rnnV2LayerCount;
        _ = rnnV2HiddenSize;
        _ = rnnV2MaxSequenceLength;
        _ = rnnV2Operation;
        _ = rnnV2Direction;
        _ = rnnV2InputMode;
        _ = runtimeDeserializationBoundaryPrecheck;
        _ = runtimeDeserializationBoundaryPrecheckSummary;
        _ = setExecutionContextAuxiliaryStreams;
        _ = clearExecutionContextAuxiliaryStreams;
        _ = contextAuxiliaryStreamAssignmentSnapshot;
        _ = contextAuxiliaryStreamAssignmentSummary;

        return string.Join(";",
            "compiled:plugin-inventory",
            "plugin-inventory-field-metadata",
            "engine-rnn-readonly-diagnostics",
            "execution-context-auxiliary-stream-lifetime",
            "managed-callbacks",
            "callback-diagnostics",
            "callback-api-language-safe-controls",
            "error-recorder-snapshot",
            "error-recorder-diagnostics-design-gate",
            "dimension-expression-snapshot-design-gate",
            "runtime-deserialization-boundary-precheck",
            "onnx-parser-diagnostic-snapshot",
            "onnx-parser-diagnostic-summary",
            "onnx-parser-refitter-diagnostic-snapshot",
            "onnx-parser-refitter-diagnostic-summary",
            "logger-presence-safe-controls",
            "allocator-debug-listener-safe-controls",
            "callback-interface-info-safe-controls",
            "execution-context-callback-state-snapshot",
            "execution-context-callback-allocator-safe-control-summary",
            "allocator-owner-dry-run-diagnostics",
            "allocator-owner-native-dry-run-controls",
            "allocator-owner-state-ledger-dry-run-controls",
            "allocator-owner-ledger-safety-gate",
            "output-allocator-callback-owner-design",
            "output-allocator-attach-detach-design-gate",
            "output-buffer-ownership-safety-gate",
            "output-allocator-runtime-proof-precheck",
            "callback-allocator-readiness-snapshot",
            "callback-owner-closure-matrix",
            "debug-listener-callback-owner-design",
            "debug-listener-attach-detach-design-gate",
            "debug-listener-borrowed-tensor-safety-gate",
            "debug-listener-attach-vtable-safety-gate",
            "debug-listener-native-attach-nothrow-preflight",
            "debug-listener-native-owner-address-design-gate",
            "debug-listener-native-nothrow-vtable-design-gate",
            "debug-listener-native-attach-entry-design-gate",
            "debug-listener-native-detach-before-release-design-gate",
            "debug-listener-native-owner-lifecycle-dry-run",
            "debug-listener-native-attach-entry-runtime-scaffold",
            "debug-listener-native-attach-entry-minimal-safety",
            "debug-listener-native-owner-stable-identity",
            "debug-listener-native-owner-noncopyable-storage",
            "debug-listener-native-nothrow-destructor",
            "debug-listener-native-owner-lifecycle-gate",
            "debug-listener-native-attach-bridge-shape-gate",
            "debug-listener-exception-status-mapping-gate",
            "debug-listener-inflight-accounting-gate",
            "debug-listener-native-nothrow-vtable-scaffold-gate",
            "debug-listener-nothrow-vtable-callback-stub",
            "callback-stub-gate",
            "debug-listener-borrowed-debug-tensor-metadata-runtime-gate",
            "borrowed-debug-tensor-metadata-gate",
            "debug-listener-native-vtable-install-preflight",
            "native-vtable-install-preflight",
            "debug-listener-native-owner-vtable-install-experiment",
            "native-owner-vtable-install-experiment",
            "debug-listener-runtime-proof-precheck",
            "debug-listener-runtime-proof-attempt-preflight",
            "debug-listener-real-non-null-attach-runtime-smoke",
            "runtime-smoke-skipped",
            "runtime-smoke-blocked",
            "runtime-smoke-attempted",
            "debug-listener-process-debug-tensor-callback-trampoline",
            "callback-trampoline-shape",
            "debug-listener-real-callback-runtime-proof",
            "debug-listener-callback-proof-gap-report",
            "real-callback-runtime-blocked",
            "attempted-no-invocation",
            "profiler-safe-controls",
            "progress-monitor-safe-controls",
            "cuda-memory-range",
            nameof(TensorRtEngine.HasImplicitBatchDimensionCompatibility),
            nameof(TensorRtBuilderConfig.SerializedPluginPathCountCompatibility),
            nameof(TensorRtBuilder.SetMaxBatchSizeCompatibility),
            nameof(TensorRtBuilderConfig.SetMaxWorkspaceSizeCompatibility),
            nameof(TensorRtBuilderConfig.SetMinTimingIterationsCompatibility),
            nameof(TensorRtCallbackAllocatorReadiness),
            nameof(TensorRtCallbackAllocatorReadiness.Evaluate),
            nameof(TensorRtCallbackAllocatorReadinessSnapshot),
            nameof(TensorRtCallbackAllocatorReadinessSnapshot.IsPublishSafeForManagedCallbacks),
            nameof(TensorRtCallbackAllocatorReadinessSnapshot.IsRuntimeInvocationProofComplete),
            nameof(TensorRtCallbackAllocatorReadinessSnapshot.BlockedReasonCount),
            nameof(TensorRtCallbackOwnerClosureMatrix),
            nameof(TensorRtCallbackOwnerClosureMatrix.Evaluate),
            nameof(TensorRtCallbackOwnerClosureMatrixResult),
            nameof(TensorRtCallbackOwnerClosureMatrixRow),
            nameof(TensorRtCallbackOwnerClosureMatrixResult.FamilyCount),
            nameof(TensorRtCallbackOwnerClosureMatrixResult.DesignGateReadyFamilyCount),
            nameof(TensorRtCallbackOwnerClosureMatrixResult.ClosureReadyFamilyCount),
            nameof(TensorRtCallbackOwnerClosureMatrixResult.RuntimeProofAttemptReadyFamilyCount),
            nameof(TensorRtCallbackOwnerClosureMatrixResult.PackageConsumerRuntimeProofReadyFamilyCount),
            nameof(TensorRtCallbackOwnerClosureMatrixResult.RuntimeProofBlocked),
            nameof(TensorRtCallbackOwnerClosureMatrixResult.DeferredRowsStillRequired),
            nameof(TensorRtLayer.GetRnnV2LayerCount),
            nameof(TensorRtLayer.GetRnnV2HiddenSize),
            nameof(TensorRtLayer.GetRnnV2DataLength),
            nameof(TensorRtLayer.GetRnnV2MaxSequenceLength),
            nameof(TensorRtLayer.GetRnnV2Operation),
            nameof(TensorRtLayer.GetRnnV2Direction),
            nameof(TensorRtLayer.GetRnnV2InputMode),
            nameof(TensorRtLayer.GetRnnV2CellState),
            nameof(TensorRtLayer.GetRnnV2HiddenState),
            nameof(TensorRtLayer.GetRnnV2SequenceLengths),
            nameof(TensorRtLayer.SetRnnV2Operation),
            nameof(TensorRtLayer.SetRnnV2Direction),
            nameof(TensorRtLayer.SetRnnV2InputMode),
            nameof(TensorRtLayer.SetRnnV2CellState),
            nameof(TensorRtLayer.SetRnnV2HiddenState),
            nameof(TensorRtLayer.SetRnnV2SequenceLengths),
            nameof(TensorRtLayer.GetRnnV2WeightsForGate),
            nameof(TensorRtLayer.GetRnnV2BiasForGate),
            nameof(TensorRtTensor.IsOwnerLifetimeBound),
            nameof(TensorRtRnnV2GateWeightsSnapshot),
            nameof(TensorRtRnnV2GateWeightsSnapshot.ToArray),
            nameof(TensorRtRnnV2BorrowedStateDesignGate),
            nameof(TensorRtRnnV2BorrowedStateDesignGate.EvaluateKnownSurface),
            nameof(TensorRtRnnV2BorrowedStateDesignGateResult),
            nameof(TensorRtRnnV2BorrowedStateDesignGateResult.DataLengthScalarPromoted),
            nameof(TensorRtRnnV2BorrowedStateDesignGateResult.RemainingDeferredTriageRowCount),
            nameof(TensorRtRnnV2BorrowedStateDesignGateResult.CanPromoteRuntimeProof),
            nameof(TensorRtRnnOperation),
            nameof(TensorRtRnnDirection),
            nameof(TensorRtRnnInputMode),
            nameof(TensorRtRnnGateType),
            nameof(TensorRtPluginRegistryInventory),
            nameof(TensorRtPluginRegistryInventory.FindCreator),
            nameof(TensorRtPluginRegistryInventory.TryFindCreator),
            nameof(TensorRtPluginRegistryInventory.GetCreatorSummaries),
            nameof(TensorRtPluginRegistryInventory.GetFieldSummaries),
            nameof(TensorRtPluginRegistryInventory.GetDiagnostics),
            nameof(TensorRtPluginRegistryInventoryDiagnostics),
            nameof(TensorRtPluginRegistryInventoryDiagnostics.IsConsistent),
            nameof(TensorRtPluginRegistryInventoryDiagnostics.TotalFieldCount),
            nameof(TensorRtPluginRegistryInventoryDiagnostics.EmptyFieldNameCount),
            nameof(TensorRtPluginRegistryInventoryDiagnostics.NegativeFieldLengthCount),
            nameof(TensorRtPluginCreatorSummary),
            nameof(TensorRtPluginCreatorSummary.FieldCount),
            nameof(TensorRtPluginCreatorSummary.TensorRtVersion),
            nameof(TensorRtPluginCreatorInfo.TensorRtVersion),
            nameof(TensorRtPluginFieldSummary),
            nameof(TensorRtPluginFieldSummary.FieldName),
            nameof(TensorRtPluginFieldSummary.FieldType),
            nameof(TensorRtPluginFieldSummary.HasData),
            nameof(TensorRtEnvironmentProbe.IsGlobalPluginRegistryAvailable),
            nameof(TensorRtEnvironmentProbe.TryIsGlobalPluginRegistryAvailable),
            nameof(TensorRtEnvironmentProbe.GetGlobalPluginRegistryInventory),
            nameof(TensorRtEnvironmentProbe.IsGlobalPluginRegistryParentSearchEnabled),
            nameof(TensorRtEnvironmentProbe.SetGlobalPluginRegistryParentSearchEnabled),
            nameof(TensorRtEnvironmentProbe.TrySetGlobalPluginRegistryParentSearchEnabled),
            nameof(TensorRtInferenceBindings.ExecuteV2),
            nameof(TensorRtInferenceBindings.ExecuteLegacy),
            nameof(TensorRtInferenceBindings.EnqueueV2AndSynchronize),
            nameof(TensorRtInferenceExecutionSummary.Synchronized),
            nameof(TensorRtBuilder.GetPluginRegistryInventory),
            nameof(TensorRtBuilder.IsPluginCreatorRegistered),
            nameof(TensorRtBuilder.TryIsPluginCreatorRegistered),
            nameof(TensorRtPluginV2LayerMetadata),
            nameof(TensorRtPluginV2LayerMetadata.PluginType),
            nameof(TensorRtPluginV2LayerMetadata.PluginVersion),
            nameof(TensorRtPluginV2LayerMetadata.PluginNamespace),
            nameof(TensorRtPluginV2LayerMetadata.SerializationSize),
            nameof(TensorRtPluginV2LayerMetadata.PackedTensorRtVersion),
            nameof(TensorRtPluginV2LayerMetadata.PluginApiVersionTag),
            nameof(TensorRtPluginV2LayerMetadata.TensorRtVersion),
            nameof(TensorRtPluginV2LayerMetadata.OutputCount),
            nameof(TensorRtPluginV2LayerMetadata.HasExtCapability),
            nameof(TensorRtPluginV2LayerMetadata.HasIoExtCapability),
            nameof(TensorRtPluginV2LayerMetadata.HasDynamicExtCapability),
            nameof(TensorRtPluginV2LayerMetadata.IsConsistent),
            nameof(TensorRtLayer.GetPluginV2Metadata),
            nameof(TensorRtLayer.TryGetPluginV2Metadata),
            nameof(TensorRtLayer.GetPluginV2LegacyOutputDimensions),
            nameof(TensorRtLayer.GetPluginV2LegacyWorkspaceSize),
            nameof(TensorRtLayer.SupportsPluginV2LegacyFormat),
            nameof(TensorRtLayer.GetPluginV2OutputDataType),
            nameof(TensorRtLayer.CanPluginV2BroadcastInputAcrossBatch),
            nameof(TensorRtLayer.IsPluginV2OutputBroadcastAcrossBatch),
            nameof(TensorRtPluginFormatCapabilityKind),
            nameof(TensorRtPluginFormatSupportSnapshot),
            nameof(TensorRtPluginFormatSupportSnapshot.Support),
            nameof(TensorRtPluginFormatSupportSnapshot.AllSupported),
            nameof(TensorRtPluginFormatSupportSnapshot.IsInputSupported),
            nameof(TensorRtPluginFormatSupportSnapshot.IsOutputSupported),
            nameof(TensorRtLayer.GetPluginV2DynamicFormatSupportSnapshot),
            nameof(TensorRtLayer.GetPluginV2IoExtFormatSupportSnapshot),
            nameof(TensorRtLayer.TryGetPluginV2DynamicFormatSupportSnapshot),
            nameof(TensorRtLayer.TryGetPluginV2IoExtFormatSupportSnapshot),
            nameof(TensorRtPluginV3InterfaceMetadata),
            nameof(TensorRtPluginV3InterfaceMetadata.InterfaceInfo),
            nameof(TensorRtPluginV3InterfaceMetadata.ApiLanguage),
            nameof(TensorRtPluginV3CoreMetadata),
            nameof(TensorRtPluginV3CoreMetadata.PluginName),
            nameof(TensorRtPluginV3CoreMetadata.PluginVersion),
            nameof(TensorRtPluginV3CoreMetadata.PluginNamespace),
            nameof(TensorRtPluginV3BuildMetadata),
            nameof(TensorRtPluginV3BuildMetadata.OutputCount),
            nameof(TensorRtPluginV3BuildMetadata.TacticCount),
            nameof(TensorRtPluginV3BuildMetadata.FormatCombinationLimit),
            nameof(TensorRtPluginV3BuildMetadata.TimingCacheId),
            nameof(TensorRtPluginV3BuildMetadata.MetadataString),
            nameof(TensorRtPluginV3RuntimeMetadata),
            nameof(TensorRtPluginV3LayerMetadata),
            nameof(TensorRtPluginV3LayerMetadata.PluginInterface),
            nameof(TensorRtPluginV3LayerMetadata.Core),
            nameof(TensorRtPluginV3LayerMetadata.Build),
            nameof(TensorRtPluginV3LayerMetadata.Runtime),
            nameof(TensorRtPluginV3LayerMetadata.IsConsistent),
            nameof(TensorRtLayer.GetPluginV3Metadata),
            nameof(TensorRtLayer.TryGetPluginV3Metadata),
            nameof(TensorRtPluginV3BuildIoSnapshot),
            nameof(TensorRtPluginV3BuildIoSnapshot.OutputDataTypes),
            nameof(TensorRtPluginV3BuildIoSnapshot.AliasedInputIndices),
            nameof(TensorRtPluginV3BuildIoSnapshot.FormatSupport),
            nameof(TensorRtPluginV3SerializationFieldInventory),
            nameof(TensorRtPluginV3SerializationFieldInventory.Fields),
            nameof(TensorRtPluginV3SerializationFieldInventory.PointerFreeCopiedInventory),
            nameof(TensorRtLayer.GetPluginV3BuildIoSnapshot),
            nameof(TensorRtLayer.TryGetPluginV3BuildIoSnapshot),
            nameof(TensorRtLayer.GetPluginV3RuntimeSerializationFields),
            nameof(TensorRtLayer.TryGetPluginV3RuntimeSerializationFields),
            nameof(TensorRtLogger),
            nameof(TensorRtLogger.EmitDiagnostic),
            nameof(TensorRtLogger.CallbackFailureCount),
            nameof(TensorRtLogger.LastCallbackException),
            nameof(TensorRtLogger.InterfaceInfo),
            nameof(TensorRtLogger.TryGetInterfaceInfo),
            nameof(TensorRtLogger.ApiLanguage),
            nameof(TensorRtLogger.TryGetApiLanguage),
            nameof(TensorRtApiLanguage),
            nameof(TensorRtProfiler),
            nameof(TensorRtProfiler.EmitDiagnostic),
            nameof(TensorRtProfiler.CallbackFailureCount),
            nameof(TensorRtProfiler.LastCallbackException),
            nameof(TensorRtProfiler.InterfaceInfo),
            nameof(TensorRtProfiler.TryGetInterfaceInfo),
            nameof(TensorRtProfiler.ApiLanguage),
            nameof(TensorRtProfiler.TryGetApiLanguage),
            nameof(TensorRtProgressMonitor.ApiLanguage),
            nameof(TensorRtProgressMonitor.TryGetApiLanguage),
            nameof(TensorRtVersionedInterfaceMetadata),
            nameof(TensorRtVersionedInterfaceMetadata.InterfaceInfo),
            nameof(TensorRtVersionedInterfaceMetadata.ApiLanguage),
            nameof(TensorRtVersionedInterfaceMetadata.PointerFreeCopiedMetadata),
            nameof(TensorRtVersionedInterfaceMetadata.RetainsNativeInterface),
            nameof(TensorRtRuntime.TryGetErrorRecorderVersionedMetadata),
            nameof(TensorRtRefitter.TryGetErrorRecorderVersionedMetadata),
            nameof(TensorRtEngine.TryGetErrorRecorderVersionedMetadata),
            nameof(TensorRtExecutionContext.TryGetErrorRecorderVersionedMetadata),
            nameof(TensorRtBuilder.TryGetErrorRecorderVersionedMetadata),
            nameof(TensorRtNetworkDefinition.TryGetErrorRecorderVersionedMetadata),
            nameof(TensorRtEngineInspector.TryGetErrorRecorderVersionedMetadata),
            nameof(TensorRtBuilderConfig.TryGetProgressMonitorVersionedMetadata),
            nameof(TensorRtExecutionContext.TryGetOutputAllocatorVersionedMetadata),
            nameof(TensorRtExecutionContext.TryGetTemporaryStorageAllocatorVersionedMetadata),
            nameof(TensorRtExecutionContext.TryGetDebugListenerVersionedMetadata),
            nameof(TensorRtErrorRecorderSnapshot),
            nameof(TensorRtErrorRecord),
            nameof(TensorRtRuntimeDiagnosticSnapshot),
            nameof(TensorRtRuntimeDiagnosticSnapshot.ToSummary),
            nameof(TensorRtRuntimeDiagnosticSummary),
            nameof(TensorRtRuntimeDiagnosticSummary.CopiedErrorRecordCount),
            nameof(TensorRtRuntimeDiagnosticSummary.DiagnosticCount),
            nameof(TensorRtRuntimeDiagnosticSummary.RuntimeEvidenceKind),
            nameof(TensorRtRuntimeDiagnosticSummary.IsRuntimeExecutionEvidence),
            nameof(TensorRtRuntimeDiagnosticSummary.IsRuntimeExecutionProof),
            nameof(TensorRtRuntimeDiagnosticSummary.CanPromoteReleaseProof),
            nameof(TensorRtRuntimeDiagnosticSummary.CanPromoteRuntimeProof),
            nameof(TensorRtRuntimeDiagnosticSummary.CanDeleteDeferredRecord),
            nameof(TensorRtEngineDeploymentSnapshot),
            nameof(TensorRtEngineDeploymentSnapshot.ToSummary),
            nameof(TensorRtEngineDeploymentSummary),
            nameof(TensorRtEngineDeploymentSummary.CopiedTensorCount),
            nameof(TensorRtEngineDeploymentSummary.CopiedProfileTensorValueCount),
            nameof(TensorRtEngineDeploymentSummary.CanPromoteRuntimeProof),
            nameof(TensorRtEngineDeploymentSummary.CanDeleteDeferredRecord),
            nameof(TensorRtBuilderConfigDeploymentSnapshot),
            nameof(TensorRtBuilderConfigDeploymentSnapshot.ToSummary),
            nameof(TensorRtBuilderConfigDeploymentSummary),
            nameof(TensorRtBuilderConfigDeploymentSummary.OptimizationProfileCount),
            nameof(TensorRtBuilderConfigDeploymentSummary.PluginToSerializeCount),
            nameof(TensorRtBuilderConfigDeploymentSummary.CanPromoteRuntimeProof),
            nameof(TensorRtBuilderConfigDeploymentSummary.CanDeleteDeferredRecord),
            nameof(TensorRtExecutionContextDeploymentSnapshot),
            nameof(TensorRtExecutionContextDeploymentSnapshot.ToSummary),
            nameof(TensorRtExecutionContextDeploymentSummary),
            nameof(TensorRtExecutionContextDeploymentSummary.CopiedTensorStateCount),
            nameof(TensorRtExecutionContextDeploymentSummary.CopiedRuntimeDiagnosticCount),
            nameof(TensorRtExecutionContextDeploymentSummary.CanPromoteRuntimeProof),
            nameof(TensorRtExecutionContextDeploymentSummary.CanDeleteDeferredRecord),
            nameof(TensorRtSerializationConfig),
            nameof(TensorRtSerializationConfig.ToSummary),
            nameof(TensorRtSerializationConfigSummary),
            nameof(TensorRtSerializationConfigSummary.CanPromoteRuntimeProof),
            nameof(TensorRtSerializationConfigSummary.CanDeleteDeferredRecord),
            nameof(TensorRtRuntimeConfig),
            nameof(TensorRtRuntimeConfig.ToSummary),
            nameof(TensorRtRuntimeConfigSummary),
            nameof(TensorRtRuntimeConfigSummary.CanPromoteRuntimeProof),
            nameof(TensorRtRuntimeConfigSummary.CanDeleteDeferredRecord),
            "TensorRtRuntime.HasLogger",
            nameof(TensorRtRuntime.HasErrorRecorder),
            nameof(TensorRtRuntime.TryGetErrorRecorderSnapshot),
            nameof(TensorRtRuntime.GetDiagnosticSnapshot),
            nameof(TensorRtRuntime.ClearErrorRecorder),
            nameof(TensorRtRuntime.ClearGpuAllocator),
            nameof(TensorRtOnnxParserDiagnosticSnapshot),
            nameof(TensorRtOnnxParserDiagnosticSnapshot.Line),
            nameof(TensorRtOnnxParserDiagnosticSnapshot.ErrorCount),
            nameof(TensorRtOnnxParserDiagnosticSnapshot.Diagnostics),
            nameof(TensorRtOnnxParserDiagnosticSnapshot.DiagnosticSummary),
            nameof(TensorRtOnnxParserDiagnosticSnapshot.UsedVCPluginLibraries),
            nameof(TensorRtOnnxParserDiagnosticSnapshot.IdentityOperatorSupported),
            nameof(TensorRtOnnxParserDiagnosticSnapshot.ToSummary),
            nameof(TensorRtOnnxParserDiagnosticSummary),
            nameof(TensorRtOnnxParserDiagnosticSummary.Line),
            nameof(TensorRtOnnxParserDiagnosticSummary.ErrorCount),
            nameof(TensorRtOnnxParserDiagnosticSummary.CopiedDiagnosticCount),
            nameof(TensorRtOnnxParserDiagnosticSummary.DiagnosticSummaryLength),
            nameof(TensorRtOnnxParserDiagnosticSummary.UsedVCPluginLibraryCount),
            nameof(TensorRtOnnxParserDiagnosticSummary.IdentityOperatorSupported),
            nameof(TensorRtOnnxParserDiagnosticSummary.RuntimeEvidenceKind),
            nameof(TensorRtOnnxParserDiagnosticSummary.IsRuntimeExecutionEvidence),
            nameof(TensorRtOnnxParserDiagnosticSummary.IsRuntimeExecutionProof),
            nameof(TensorRtOnnxParserDiagnosticSummary.PointerFreeCopiedSummary),
            nameof(TensorRtOnnxParserDiagnosticSummary.CanPromoteRuntimeProof),
            nameof(TensorRtOnnxParserDiagnosticSummary.CanPromoteReleaseProof),
            nameof(TensorRtOnnxParserDiagnosticSummary.CanDeleteDeferredRecord),
            nameof(TensorRtOnnxModelSupportReport),
            nameof(TensorRtOnnxModelSupportReport.ToSummary),
            nameof(TensorRtOnnxModelSupportSummary),
            nameof(TensorRtOnnxModelSupportSummary.ReportedSupportedSubgraphCount),
            nameof(TensorRtOnnxModelSupportSummary.ReportedUnsupportedSubgraphCount),
            nameof(TensorRtOnnxModelSupportSummary.CopiedSubgraphCount),
            nameof(TensorRtOnnxModelSupportSummary.CopiedSupportedSubgraphCount),
            nameof(TensorRtOnnxModelSupportSummary.CopiedUnsupportedSubgraphCount),
            nameof(TensorRtOnnxModelSupportSummary.CopiedNodeCount),
            nameof(TensorRtOnnxModelSupportSummary.CopiedSubgraphCountsMatchReportedCounts),
            nameof(TensorRtOnnxModelSupportSummary.RuntimeEvidenceKind),
            nameof(TensorRtOnnxModelSupportSummary.IsRuntimeExecutionEvidence),
            nameof(TensorRtOnnxModelSupportSummary.IsRuntimeExecutionProof),
            nameof(TensorRtOnnxModelSupportSummary.PointerFreeCopiedSummary),
            nameof(TensorRtOnnxModelSupportSummary.CanPromoteRuntimeProof),
            nameof(TensorRtOnnxModelSupportSummary.CanPromoteReleaseProof),
            nameof(TensorRtOnnxModelSupportSummary.CanDeleteDeferredRecord),
            nameof(TensorRtOnnxParser.GetDiagnosticSnapshot),
            nameof(TensorRtOnnxParserRefitterDiagnosticSnapshot),
            nameof(TensorRtOnnxParserRefitterDiagnosticSnapshot.Line),
            nameof(TensorRtOnnxParserRefitterDiagnosticSnapshot.ErrorCount),
            nameof(TensorRtOnnxParserRefitterDiagnosticSnapshot.Diagnostics),
            nameof(TensorRtOnnxParserRefitterDiagnosticSnapshot.DiagnosticSummary),
            nameof(TensorRtOnnxParserRefitterDiagnosticSnapshot.ToSummary),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.Line),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.ErrorCount),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.CopiedDiagnosticCount),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.DiagnosticSummaryLength),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.RuntimeEvidenceKind),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.IsRuntimeExecutionEvidence),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.IsRuntimeExecutionProof),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.PointerFreeCopiedSummary),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.CanPromoteRuntimeProof),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.CanPromoteReleaseProof),
            nameof(TensorRtOnnxParserRefitterDiagnosticSummary.CanDeleteDeferredRecord),
            nameof(TensorRtOnnxParserRefitter.GetDiagnosticSnapshot),
            nameof(TensorRtRefitterDiagnosticSnapshot),
            nameof(TensorRtRefitterDiagnosticSnapshot.ToSummary),
            nameof(TensorRtRefitterDiagnosticSummary),
            nameof(TensorRtRefitterDiagnosticSummary.CopiedMissingNamedWeightCount),
            nameof(TensorRtRefitterDiagnosticSummary.CopiedAllNamedWeightCount),
            nameof(TensorRtRefitterDiagnosticSummary.DiagnosticCount),
            nameof(TensorRtRefitter.HasErrorRecorder),
            nameof(TensorRtRefitter.TryGetErrorRecorderSnapshot),
            nameof(TensorRtRefitter.GetDiagnosticSnapshot),
            nameof(TensorRtRefitter.ClearErrorRecorder),
            nameof(TensorRtRefitter.HasLogger),
            nameof(TensorRtErrorCodeMetadata),
            nameof(TensorRtErrorCodeMetadata.GetExclusiveUpperBound),
            nameof(TensorRtErrorCodeMetadata.IsDefinedRangeValue),
            nameof(TensorRtBuilder.BuildEngineWithConfig),
            nameof(TensorRtBuilderConfig.SetPluginsToSerialize),
            nameof(TensorRtBuilderConfig.GetPluginToSerialize),
            nameof(TensorRtBuilderConfig.GetPluginsToSerialize),
            "TensorRtBuilder.HasLogger",
            nameof(TensorRtBuilder.HasErrorRecorder),
            nameof(TensorRtBuilder.ClearErrorRecorder),
            nameof(TensorRtBuilder.ClearGpuAllocator),
            nameof(TensorRtExecutionContext.SetProfiler),
            nameof(TensorRtExecutionContext.ClearProfiler),
            nameof(TensorRtExecutionContext.HasNativeProfiler),
            nameof(TensorRtExecutionContext.HasProfiler),
            nameof(TensorRtExecutionContext.HasErrorRecorder),
            nameof(TensorRtExecutionContext.ClearErrorRecorder),
            nameof(TensorRtExecutionContext.SetInputShapeBinding),
            nameof(TensorRtExecutionContext.SetAuxStreams),
            nameof(TensorRtExecutionContext.ClearAuxStreams),
            nameof(TensorRtExecutionContext.GetAuxiliaryStreamAssignmentSnapshot),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.Line),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.AssignedStreamCount),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.IsCleared),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.ManagedHandleLeaseActive),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.NativeStreamPointerExposed),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.BorrowedHandleEscaped),
            nameof(TensorRtAuxiliaryStreamAssignmentSnapshot.Diagnostic),
            nameof(TensorRtExecutionContext.HasOutputAllocator),
            nameof(TensorRtExecutionContext.ClearOutputAllocator),
            nameof(TensorRtExecutionContext.HasTemporaryStorageAllocator),
            nameof(TensorRtExecutionContext.ClearTemporaryStorageAllocator),
            nameof(TensorRtExecutionContext.HasDebugListener),
            nameof(TensorRtExecutionContext.ClearDebugListener),
            nameof(TensorRtExecutionContext.TryGetOutputAllocatorInterfaceInfo),
            nameof(TensorRtExecutionContext.TryGetTemporaryStorageAllocatorInterfaceInfo),
            nameof(TensorRtExecutionContext.TryGetDebugListenerInterfaceInfo),
            nameof(TensorRtExecutionContext.GetCallbackStateSnapshot),
            nameof(TensorRtExecutionContext.ClearCallbackState),
            nameof(TensorRtExecutionContext.GetRuntimeDiagnosticSnapshot),
            nameof(TensorRtExecutionContext.GetCallbackAllocatorSafeControlSummary),
            nameof(TensorRtExecutionContextCallbackStateSnapshot),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.HasOutputAllocator),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.HasTemporaryStorageAllocator),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.HasDebugListener),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.OutputAllocatorInterfaceInfoAvailable),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.TemporaryStorageAllocatorInterfaceInfoAvailable),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.DebugListenerInterfaceInfoAvailable),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.OutputAllocatorClearSupported),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.TemporaryStorageAllocatorClearSupported),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.DebugListenerClearSupported),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.OutputAllocatorCleared),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.TemporaryStorageAllocatorCleared),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.DebugListenerCleared),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.LastStatus),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.LastOperation),
            nameof(TensorRtExecutionContextCallbackStateSnapshot.Diagnostic),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSnapshot),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSnapshot.ToSummary),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSummary),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSummary.HasOutputTensorName),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSummary.HasOutputAllocator),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSummary.IsOutputTensorAddressSet),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSummary.CallbackStateLastStatus),
            nameof(TensorRtExecutionContextRuntimeDiagnosticSummary.DiagnosticCount),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.EvidenceKind),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.RuntimeEvidenceKind),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.RealCallbackRuntime),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.IsRealCallbackRuntimeProof),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.CopiedInterfaceInfoCount),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.DiagnosticCount),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.PointerFreeSurfaceReady),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.CallbackInvocationAttempted),
            nameof(TensorRtExecutionContextCallbackAllocatorSafeControlSummary.IsRuntimeInvocationProofComplete),
            nameof(TensorRtProgressMonitor),
            nameof(TensorRtProgressMonitor.EmitDiagnostic),
            nameof(TensorRtProgressMonitor.CallbackFailureCount),
            nameof(TensorRtProgressMonitor.LastCallbackException),
            nameof(TensorRtProgressMonitor.InterfaceInfo),
            nameof(TensorRtProgressMonitor.TryGetInterfaceInfo),
            nameof(TensorRtBuilderConfig.SetProgressMonitor),
            nameof(TensorRtBuilderConfig.ClearProgressMonitor),
            nameof(TensorRtBuilderConfig.HasProgressMonitor),
            nameof(TensorRtAllocatorCallbackOwner),
            nameof(TensorRtAllocatorCallbackOwner.RunDryRunDiagnostic),
            nameof(TensorRtAllocatorCallbackOwner.CallbackInvocationCount),
            nameof(TensorRtAllocatorCallbackOwner.CallbackFailureCount),
            nameof(TensorRtAllocatorCallbackOwner.LastCallbackException),
            nameof(TensorRtAllocatorCallbackOwner.IsAttached),
            nameof(TensorRtAllocatorCallbackOwner.IsDisposed),
            nameof(TensorRtAllocatorCallbackOwner.LastDiagnostic),
            nameof(TensorRtAllocatorCallbackOwner.RunNativeDryRunDiagnostic),
            nameof(TensorRtAllocatorNativeDryRunResult),
            nameof(TensorRtAllocatorNativeDryRunResult.Line),
            nameof(TensorRtAllocatorNativeDryRunResult.InvocationCount),
            nameof(TensorRtAllocatorNativeDryRunResult.FailureCount),
            nameof(TensorRtAllocatorNativeDryRunResult.LastStatus),
            nameof(TensorRtAllocatorNativeDryRunResult.IsAttached),
            nameof(TensorRtAllocatorNativeDryRunResult.LastSize),
            nameof(TensorRtAllocatorNativeDryRunResult.LastAlignment),
            nameof(TensorRtAllocatorNativeDryRunResult.Diagnostic),
            nameof(TensorRtAllocatorNativeDryRunResult.Succeeded),
            nameof(TensorRtAllocatorCallbackOwner.RunNativeStateLedgerDryRunDiagnostic),
            nameof(TensorRtAllocatorOwnerStateDryRunResult),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.Line),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.OwnerId),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.StateTransitionCount),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LedgerAllocationCount),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LedgerReleaseCount),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LedgerFailureCount),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LastAllocationId),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LastReleaseAllocationId),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LastSize),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LastAlignment),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LastStreamValue),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.AttachState),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LastStatus),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.IsAttached),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.HasLiveAllocation),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.LastOperation),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.Diagnostic),
            nameof(TensorRtAllocatorOwnerStateDryRunResult.Succeeded),
            nameof(TensorRtAllocatorCallbackOwner.RunLifecycleDiagnostic),
            nameof(TensorRtAllocatorCallbackOwner.GetSnapshot),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.EvidenceKind),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.RuntimeEvidenceKind),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.RealCallbackRuntime),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.IsRealCallbackRuntimeProof),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.DevicePointerExposed),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.DevicePointerProduced),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.BorrowedPointerEscaped),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.ManagedKeepAliveReady),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.DisposeReleaseReady),
            nameof(TensorRtAllocatorCallbackOwnerSnapshot.PointerFreeSurfaceReady),
            nameof(TensorRtAllocatorLedgerSafetyGate),
            nameof(TensorRtAllocatorLedgerSafetyGate.Evaluate),
            nameof(TensorRtAllocatorLedgerSafetyGate.GetSnapshot),
            nameof(TensorRtAllocatorLedgerSafetyGateResult),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.EvidenceKind),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.RuntimeEvidenceKind),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.RealCallbackRuntime),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.ManagedKeepAliveReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.DisposeReleaseReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.NativeLedgerDesignReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.PointerFreeSurfaceReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.LineSpecificAttachDetachReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.DevicePointerLedgerRuntimeReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.StreamLifetimeReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.RuntimeProofBlocked),
            nameof(TensorRtAllocatorLedgerSafetyGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtAllocatorDryRunRequest),
            nameof(TensorRtAllocatorDryRunRequest.Size),
            nameof(TensorRtAllocatorDryRunRequest.Alignment),
            nameof(TensorRtAllocatorDryRunRequest.Reason),
            nameof(TensorRtAllocatorDryRunResult),
            nameof(TensorRtAllocatorDryRunResult.Succeeded),
            nameof(TensorRtAllocatorDryRunResult.Diagnostic),
            nameof(TensorRtAllocatorDryRunResult.Success),
            nameof(TensorRtAllocatorDryRunResult.Failure),
            nameof(TensorRtOutputAllocatorCallbackOwner),
            nameof(TensorRtOutputAllocatorCallbackRequest),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot),
            nameof(TensorRtOutputAllocatorCallbackOwner.RunDesignDiagnostic),
            nameof(TensorRtOutputAllocatorCallbackOwner.GetSnapshot),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.EvidenceKind),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.RuntimeEvidenceKind),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.RealCallbackRuntime),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.IsRealCallbackRuntimeProof),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.NativeLedgerAvailable),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.StateTransitionCount),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.LedgerAllocationCount),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.LedgerReleaseCount),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.OutputBufferPointerExposed),
            nameof(TensorRtOutputAllocatorCallbackOwnerSnapshot.OutputBufferPointerProduced),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGate),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.EvidenceKind),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.RuntimeEvidenceKind),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.RealCallbackRuntime),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.LineSupportsOutputAllocator),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.AttachControlAvailable),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.DetachClearControlAvailable),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.LineSpecificAttachDetachReady),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.StableNativeOwnerAddressReady),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.NoThrowNativeVTableReady),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.NativeVTableReady),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.OutputBufferOwnershipRuntimeReady),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.DesignGateReady),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.RuntimeProofBlocked),
            nameof(TensorRtOutputAllocatorAttachDetachDesignGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtOutputBufferOwnershipSafetyGate),
            nameof(TensorRtOutputBufferOwnershipSafetyGate.Evaluate),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.EvidenceKind),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.RuntimeEvidenceKind),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.RealCallbackRuntime),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.AttachDetachDesignGateReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.CopiedCurrentMemoryMetadataReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.CopiedShapeMetadataReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.CopiedRequestMetadataReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.SafetyGateReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.OutputBufferOwnershipRuntimeReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.CurrentMemoryReusePolicyReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.BorrowedPointerEscapeBlocked),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.OwnedDevicePointerReleasePolicyReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.ShapeNotificationOrderingReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.ReallocateOutputRuntimeReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.RuntimeProofBlocked),
            nameof(TensorRtOutputBufferOwnershipSafetyGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheck),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.EvidenceKind),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.RuntimeEvidenceKind),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.RealCallbackRuntime),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.LineSupportsOutputAllocator),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.NativeLedgerDesignReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.AttachDetachDesignGateReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.AttachControlAvailable),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.DetachClearControlAvailable),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.ManagedOwnerStateMachineReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.LineSpecificAttachDetachReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.StableNativeOwnerAddressReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.NoThrowNativeVTableReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.NativeVTableReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.DevicePointerLedgerRuntimeReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.StreamLifetimeReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.OutputBufferOwnershipSafetyGateReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.OutputBufferOwnershipRuntimeReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.CurrentMemoryReusePolicyReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.BorrowedPointerEscapeBlocked),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.OwnedDevicePointerReleasePolicyReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.ShapeNotificationOrderingReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.ReallocateOutputRuntimeReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.CanAttemptRuntimeProof),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.RuntimeProofBlocked),
            nameof(TensorRtOutputAllocatorRuntimeProofPrecheckResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerCallbackOwner),
            nameof(TensorRtDebugListenerCallbackRequest),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot),
            nameof(TensorRtDebugListenerCallbackOwner.RunDesignDiagnostic),
            nameof(TensorRtDebugListenerCallbackOwner.GetSnapshot),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.EvidenceKind),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.RealCallbackRuntime),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.ProcessDebugTensorCount),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.DebugTensorPointerExposed),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.DebugTensorPointerProduced),
            nameof(TensorRtDebugListenerCallbackOwnerSnapshot.BorrowedDebugTensorPointerEscaped),
            nameof(TensorRtDebugListenerAttachDetachDesignGate),
            nameof(TensorRtDebugListenerAttachDetachDesignGate.Evaluate),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.AttachControlAvailable),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.DetachClearControlAvailable),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.LineSpecificAttachDetachReady),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.StableNativeOwnerAddressReady),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.NoThrowNativeVTableReady),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.NativeVTableReady),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.BorrowedDebugTensorLifetimeReady),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.DesignGateReady),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerAttachDetachDesignGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGate),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGate.Evaluate),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.AttachDetachDesignGateReady),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.DebugTensorMetadataCopied),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.PointerFreeSurfaceReady),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.SafetyGateReady),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.BorrowedDebugTensorPointerEscapeBlocked),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.BorrowedDebugTensorLifetimeReady),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.BorrowedDebugTensorDataLifetimeReady),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerBorrowedTensorSafetyGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerAttachVTableSafetyGate),
            nameof(TensorRtDebugListenerAttachVTableSafetyGate.Evaluate),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.SafetyGateReady),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.AttachControlAvailable),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.StableNativeOwnerAddressReady),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.NoThrowNativeVTableReady),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.NativeVTableReady),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.ExceptionToStatusMappingReady),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerAttachVTableSafetyGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflight),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflight.Evaluate),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.PreflightReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.StableNativeOwnerAddressDesignReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.ManagedCallbackKeepAliveDesignReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.NoThrowVTableDesignReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.ExceptionToStatusMappingDesignReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.BorrowedDebugTensorMetadataCopyDesignReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.BorrowedDebugTensorPointerEscapeBlocked),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.BorrowedDebugTensorLifetimeRuntimeReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.BorrowedDebugTensorDataLifetimeRuntimeReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeAttachNoThrowPreflightResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGate),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGate.Evaluate),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.DesignGateReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeAttachNoThrowPreflightReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.StableNativeOwnerAddressReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.StableNativeOwnerAddressDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.ManagedCallbackKeepAliveDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeOwnerDisposeOrderReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeOwnerReleaseHookReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeOwnerInFlightDrainReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NoThrowVTableDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.ExceptionToStatusMappingDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeOwnerAddressDesignGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGate),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGate.Evaluate),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.DesignGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NativeOwnerAddressDesignGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NativeAttachNoThrowPreflightReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.ManagedCallbackKeepAliveDesignReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NoThrowVTableDesignReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.ExceptionToStatusMappingDesignReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.BorrowedDebugTensorMetadataCopyDesignReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.BorrowedDebugTensorPointerEscapeBlocked),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NativeVTableTrampolineReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.CallbackExceptionCaptureReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.CallbackStatusMappingReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.CallbackInFlightAccountingReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeNoThrowVTableDesignGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGate),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGate.Evaluate),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.DesignGateReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.NativeNoThrowVTableDesignGateReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.NativeOwnerAddressDesignGateReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.NativeAttachNoThrowPreflightReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.LineSpecificAttachEntryDesignReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.AttachEntryNoThrowReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.AttachEntryVersionGuardReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.AttachEntryOwnershipReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.DetachBeforeReleaseReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeAttachEntryDesignGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.Evaluate),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.DesignGateReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.NativeAttachEntryDesignGateReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.NativeNoThrowVTableDesignGateReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.NativeOwnerAddressDesignGateReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.LineSpecificAttachEntryDesignReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.AttachEntryNoThrowReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.AttachEntryVersionGuardReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.AttachEntryOwnershipReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.DetachBeforeReleaseReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.ReleaseHookOrderingReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.DisposeIdempotencyReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.InFlightDrainBeforeReleaseReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.CallbackStateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.DelegateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRun),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRun.Evaluate),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.CallbackKind),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.OwnerId),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.LastStatus),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.ReleaseHookCount),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.InFlightCallbackCount),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.CallbackStatePinned),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.DelegatePinned),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.DisposeRequested),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeDetachBeforeReleaseDesignGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeAttachEntryDesignGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeNoThrowVTableDesignGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeOwnerAddressDesignGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.StableNativeOwnerIdentityReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.ReleaseHookOrderingReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.DisposeIdempotencyReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.InFlightDrainBeforeReleaseReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.CallbackStateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.DelegateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.ManagedCallbackKeepAliveDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.BorrowedDebugTensorMetadataCopyDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.BorrowedDebugTensorPointerEscapeBlocked),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.DryRunReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleDryRunResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffold),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.Evaluate),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.CallbackKind),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.NativeOwnerLifecycleDryRunReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.AttachEntryParameterShapeReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.AttachEntryVersionGuardReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.AttachEntryNoThrowBoundaryReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.AttachEntryOwnershipDiagnosticsReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.StableNativeOwnerIdentityReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.RuntimeScaffoldReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafety),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafety.Evaluate),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.CallbackKind),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.RuntimeScaffoldReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.LifecycleGateReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.LifecyclePointerFree),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.AttachEntryParameterShapeReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.AttachEntryNoThrowReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.AttachEntryVersionGuardReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.AttachEntryOwnershipDiagnosticsReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.SetDebugListenerNonNullEnabled),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.NonNullAttachStillDisabled),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.NativeAttachWouldBeBlocked),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.MinimalSafetyReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult.ReasonNativeAttachStillBlocked),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentity),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentity.Evaluate),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.CallbackKind),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.NativeAttachEntryRuntimeScaffoldReady),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.StableNativeOwnerIdentityReady),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.OwnerIdentityDiagnosticsReady),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.OwnerIdentityPointerFree),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeOwnerStableIdentityResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorage),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorage.Evaluate),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.CallbackKind),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeOwnerStableIdentityReady),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.OwnerIdentityDiagnosticsReady),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.OwnerIdentityPointerFree),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeOwnerCopyBlocked),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeOwnerMoveBlocked),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeOwnerAddressExposed),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeOwnerPointerProduced),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeOwnerNonCopyableStorageResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerNativeNoThrowDestructor),
            nameof(TensorRtDebugListenerNativeNoThrowDestructor.Evaluate),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.CallbackKind),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeOwnerNonCopyableStorageReady),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeOwnerCopyBlocked),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeOwnerMoveBlocked),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeOwnerAddressExposed),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeOwnerPointerProduced),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.DestructorNoThrowScaffoldReady),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.DestructorExceptionEscapeBlocked),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.DestructorAddressExposed),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.DestructorPointerProduced),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeNoThrowDestructorResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGate),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGate.Evaluate),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.CallbackKind),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeNoThrowDestructorGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeOwnerCopyBlocked),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeOwnerMoveBlocked),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeOwnerAddressExposed),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeOwnerPointerProduced),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DestructorNoThrowScaffoldReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DestructorExceptionEscapeBlocked),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DestructorAddressExposed),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DestructorPointerProduced),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.ManagedDisposeSnapshotReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.LifecycleScaffoldReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.ReleaseHookOrderingGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DisposeIdempotencyGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.InFlightDrainGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.CallbackStateUnpinAfterDetachGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DelegateUnpinAfterDetachGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.LifecycleAddressExposed),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.LifecyclePointerProduced),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeDetachEntryLocated),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.ReleaseHookOrderingReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DisposeIdempotencyReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.InFlightDrainBeforeReleaseReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.CallbackStateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DelegateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.LifecycleGateReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerNativeOwnerLifecycleGateResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGate),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGate.Evaluate),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.NativeOwnerLifecycleGateReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.AttachBridgeShapeReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.AttachBridgeNoThrowBoundaryReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.AttachBridgeVersionGuardReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.AttachBridgeOwnershipDiagnosticsReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.AttachBridgePointerFree),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.SetDebugListenerNonNullEnabled),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.NonNullAttachStillDisabled),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.NativeAttachEntryLocated),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.NativeOwnerLifecycleReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.NativeVTableDesignReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.AttachBridgeShapeGateReady),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.CanImplementNativeAttach),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerNativeAttachBridgeShapeGateResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerExceptionStatusMappingGate),
            nameof(TensorRtDebugListenerExceptionStatusMappingGate.Evaluate),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.ManagedCallbackExceptionCaptureReady),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.NativeCallbackExceptionCaptureReady),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.CallbackStatusMappingGateReady),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.ExceptionEscapeBlocked),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.DiagnosticCopyReady),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.MappingAddressExposed),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.MappingPointerProduced),
            nameof(TensorRtDebugListenerExceptionStatusMappingGateResult.ExceptionStatusMappingGateReady),
            nameof(TensorRtDebugListenerInFlightAccountingGate),
            nameof(TensorRtDebugListenerInFlightAccountingGate.Evaluate),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.CallbackEnterAccountingGateReady),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.CallbackLeaveAccountingGateReady),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.CallbackInFlightNeverNegativeReady),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.ReleaseAfterDrainGateReady),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.CallbackStateUnpinAfterDrainGateReady),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.AccountingAddressExposed),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.AccountingPointerProduced),
            nameof(TensorRtDebugListenerInFlightAccountingGateResult.InFlightAccountingGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGate),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.Evaluate),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.NativeAttachBridgeShapeGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.ExceptionStatusMappingGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.InFlightAccountingGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.NoThrowVTableScaffoldReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.VTableDestructorNoThrowReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.ProcessDebugTensorCallbackStubNoThrowReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.ExceptionEscapeBlocked),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.CallbackExceptionCaptureGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.CallbackStatusMappingGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.CallbackInFlightAccountingGateReady),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.VTableAddressExposed),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.VTablePointerProduced),
            nameof(TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult.VTableScaffoldGateReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStub),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.EvidenceKind),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackStubGateReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackStubShapeReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackStubNoThrowReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackMetadataCopyReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackExceptionCaptureReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackStatusMappingReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackInFlightPairingReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CallbackInFlightNeverNegativeReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.BorrowedDebugTensorMetadataCopyReady),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.BorrowedDebugTensorPointerEscapeBlocked),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.DebugTensorPointerExposed),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.DebugTensorDataPointerExposed),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.SetDebugListenerNonNullEnabled),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.NativeAttachWouldBeBlocked),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.NativeVTableInstalled),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CanInstallNativeVTable),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.CanCallProcessDebugTensorRuntime),
            nameof(TensorRtDebugListenerNoThrowVTableCallbackStubResult.ReasonCallbackRuntimeStillBlocked),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.EvidenceKind),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.MetadataGateReady),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.TensorNameCopied),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.TensorNameLength),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.TensorTypeCopied),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.TensorLocationCopied),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.TensorShapeCopied),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.TensorShapeRank),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.TensorFlagsCopied),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.BorrowedDebugTensorMetadataCopyReady),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.BorrowedDebugTensorPointerEscapeBlocked),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.BorrowedDebugTensorDataPointerEscapeBlocked),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.DebugTensorPointerExposed),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.DebugTensorDataPointerExposed),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.BorrowedDebugTensorLifetimeReady),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.BorrowedDebugTensorDataLifetimeReady),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.CanCallProcessDebugTensorRuntime),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult.ReasonMetadataRuntimeStillBlocked),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflight),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflight.Evaluate),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.NativeOwnerLifecycleGateReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.NativeAttachBridgeShapeGateReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.NativeNoThrowVTableScaffoldGateReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.BorrowedDebugTensorMetadataGateReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.VTableInstallShapeReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.VTableInstallVersionGuardReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.VTableInstallNoThrowBoundaryReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.VTableInstallOwnershipDiagnosticsReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.VTableInstallPointerFree),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.SetDebugListenerNonNullEnabled),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.NativeVTableInstallPreflightReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.NativeVTableInstalled),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.NativeVTableInstallRuntimeReady),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.CanEnableSetDebugListenerNonNull),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.CanInstallNativeVTable),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.CanCallProcessDebugTensorRuntime),
            nameof(TensorRtDebugListenerNativeVTableInstallPreflightResult.ReasonNativeVTableInstallStillBlocked),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperiment),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperiment.Evaluate),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.EvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.ExperimentShapeReady),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.InstallAttemptGuardReady),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.NonNullAttachEnabled),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.RuntimeProofEnabled),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.NativeVTableInstallAttempted),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.NativeVTableInstalled),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.RollbackReady),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.DetachBeforeReleaseReady),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.FailureStatusMappingReady),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.PointerFree),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.CanEnableSetDebugListenerNonNull),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.CanInstallNativeVTable),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.CanCallProcessDebugTensorRuntime),
            nameof(TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.ReasonNativeOwnerVTableInstallStillBlocked),
            nameof(TensorRtDebugListenerRuntimeProofPrecheck),
            nameof(TensorRtDebugListenerRuntimeProofPrecheck.Evaluate),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.EvidenceKind),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachDetachDesignGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachControlAvailable),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DetachClearControlAvailable),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.ManagedOwnerStateMachineReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.LineSpecificAttachDetachReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.StableNativeOwnerAddressReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NoThrowNativeVTableReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeVTableReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.ExceptionToStatusMappingReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.BorrowedTensorSafetyGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachVTableSafetyGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeAttachEntryDesignGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeDetachBeforeReleaseDesignGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerLifecycleDryRunReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeAttachEntryRuntimeScaffoldReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerStableIdentityReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.OwnerIdentityDiagnosticsReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.OwnerIdentityPointerFree),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerNonCopyableStorageReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerNonCopyableReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerCopyBlocked),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerMoveBlocked),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerAddressExposed),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerPointerProduced),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeNoThrowDestructorGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DestructorNoThrowScaffoldReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DestructorExceptionEscapeBlocked),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DestructorAddressExposed),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DestructorPointerProduced),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NoThrowNativeDestructorReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.NativeOwnerLifecycleGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.ManagedDisposeSnapshotReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.LifecycleScaffoldReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.ReleaseHookOrderingGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DisposeIdempotencyGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.InFlightDrainGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.CallbackStateUnpinAfterDetachGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DelegateUnpinAfterDetachGateReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.LifecycleAddressExposed),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.LifecyclePointerProduced),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachEntryParameterShapeReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachEntryNoThrowBoundaryReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachEntryOwnershipDiagnosticsReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.LineSpecificAttachEntryDesignReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachEntryNoThrowReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachEntryVersionGuardReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.AttachEntryOwnershipReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DetachBeforeReleaseReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.ReleaseHookOrderingReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DisposeIdempotencyReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.InFlightDrainBeforeReleaseReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.CallbackStateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.DelegateUnpinAfterDetachReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.BorrowedDebugTensorPointerEscapeBlocked),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.BorrowedDebugTensorLifetimeReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.BorrowedDebugTensorDataLifetimeReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.FullPackageConsumerRuntimeEvidenceReady),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerRuntimeProofPrecheckResult.BlockedPrerequisiteCount),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflight),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.EvidenceKind),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.CanEnableSetDebugListenerNonNull),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.CanInstallNativeVTable),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.CanCallProcessDebugTensorRuntime),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.CanPromoteRealCallbackRuntime),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.ReasonNonNullAttachStillBlocked),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.ReasonNativeVTableStillBlocked),
            nameof(TensorRtDebugListenerRuntimeProofAttemptPreflightResult.ReasonRuntimeProofStillBlocked),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmoke),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.EvidenceKind),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.CallbackKind),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.TensorRtLine),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.RuntimePackageKey),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.OptInEnabled),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.FullPackageConsumerReport),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.AttachGuardReady),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.NativeVTableReady),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.BorrowedDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.CallbackInvocationReady),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.AttachAttempted),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.AttachSucceeded),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.DetachAttempted),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.DetachSucceeded),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.RollbackAttempted),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.RollbackSucceeded),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.NativeVTableInstalled),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.ProcessDebugTensorInvoked),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.InvocationCount),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.AllocationCount),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.ReleaseCount),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.FailureCount),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.InFlightCallbackCount),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.LastStatus),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.LastDiagnostic),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.ReportPointerFree),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.CanPromoteRealCallbackRuntime),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult.ReasonRuntimeProofStillBlocked),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampoline),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.EvidenceKind),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.CallbackKind),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.TensorRtLine),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.RuntimePackageKey),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.TrampolineShapeReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.NativeCallbackEntryLocated),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.NoThrowCallbackEntryReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.ExceptionCaptureReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.CallbackStatusMappingReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.InFlightAccountingReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.DetachBeforeReleaseReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.BorrowedDebugTensorMetadataCopyReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.BorrowedDebugTensorPointerExposed),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.BorrowedDebugTensorDataPointerExposed),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.PointerFreeSurfaceReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.OptInEnabled),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.FullPackageConsumerReport),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.AttachAttempted),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.AttachSucceeded),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.NativeVTableInstalled),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.ProcessDebugTensorInvoked),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.InvocationCount),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.CallbackStubEntryCount),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.CallbackStubLeaveCount),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.FailureCount),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.InFlightCallbackCount),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.LastStatus),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.LastDiagnostic),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.CanPromoteRealCallbackRuntime),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult.ReasonRuntimeProofStillBlocked),
            nameof(TensorRtDebugTensorMetadataSnapshot),
            nameof(TensorRtDebugTensorMetadataSnapshot.TensorName),
            nameof(TensorRtDebugTensorMetadataSnapshot.TensorNameLength),
            nameof(TensorRtDebugTensorMetadataSnapshot.TensorShapeRank),
            nameof(TensorRtDebugTensorMetadataSnapshot.MetadataCopied),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProof.Evaluate),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.EvidenceKind),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.CallbackKind),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.TensorRtLine),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.RuntimePackageKey),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.OptInEnabled),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.FullPackageConsumerReport),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.RuntimeSmokeReady),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.TrampolineShapeReady),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.AttachAttempted),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.AttachSucceeded),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.DetachAttempted),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.DetachSucceeded),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.RollbackAttempted),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.RollbackSucceeded),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.NativeVTableInstalled),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.ProcessDebugTensorInvoked),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.InvocationCount),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.FailureCount),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.InFlightCallbackCount),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.BorrowedDebugTensorMetadataCopied),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.PointerFreeSurfaceReady),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.ProcessDebugTensorRuntimeReady),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.AttemptedNoInvocation),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.LastStatus),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.LastDiagnostic),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.CanPromoteRealCallbackRuntime),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerRealCallbackRuntimeProofResult.ReasonRuntimeProofStillBlocked),
            nameof(TensorRtDebugListenerCallbackProofGapReport),
            nameof(TensorRtDebugListenerCallbackProofGapReport.Evaluate),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.EvidenceKind),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.RuntimeEvidenceKind),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.RealCallbackRuntime),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.IsRealCallbackRuntimeProof),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.CallbackKind),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.TensorRtLine),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.RuntimePackageKey),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.NonNullAttachStillDisabled),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.NativeAttachEntryReady),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.NativeVTableInstallBlocked),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.NoThrowCallbackEntryReady),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.ExceptionStatusMappingReady),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.InFlightAccountingReady),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.BorrowedDebugTensorMetadataCopied),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.DetachRollbackReady),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.ProcessDebugTensorRuntimeInvoked),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.FullPackageConsumerRuntimeProofReady),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.PointerFreeSurfaceReady),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.AttemptedNoInvocation),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.InvocationCount),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.FailureCount),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.InFlightCallbackCount),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.CanAttemptRuntimeProof),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.CanPromoteRealCallbackRuntime),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.RuntimeProofBlocked),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.DeferredRowsStillRequired),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.GapReasonCount),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.PrimaryGapReason),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.RuntimeProofBlockerCategory),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.PackageConsumerRuntimeProofRequired),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.RuntimeInvocationRequired),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.EvidenceSource),
            nameof(TensorRtDebugListenerCallbackProofGapReportResult.NextOwnerAction),
            nameof(CudaMemory.GetRangeAttribute),
            nameof(CudaMemory.GetRangeAttributes),
            nameof(CudaMemory.GetRangeAccessedByDevices),
            nameof(CudaMemory.GetRangeDiagnosticSummary),
            nameof(CudaMemoryRangeDiagnosticSummary),
            nameof(CudaMemoryRangeDiagnosticSummary.CopiedScalarAttributeCount),
            nameof(CudaMemoryRangeDiagnosticSummary.RuntimeEvidenceKind),
            nameof(CudaMemoryRangeDiagnosticSummary.IsRuntimeExecutionEvidence),
            nameof(CudaMemoryRangeDiagnosticSummary.IsRuntimeExecutionProof),
            nameof(CudaMemoryRangeDiagnosticSummary.CanPromoteReleaseProof),
            nameof(CudaMemoryRangeDiagnosticSummary.CanPromoteRuntimeProof),
            nameof(CudaMemoryRangeDiagnosticSummary.CanDeleteDeferredRecord),
            nameof(CudaGraphDiagnosticSnapshot),
            nameof(CudaGraphDiagnosticSnapshot.ToSummary),
            nameof(CudaGraphDiagnosticSummary),
            nameof(CudaGraphDiagnosticSummary.CopiedNodeSnapshotCount),
            nameof(CudaGraphDiagnosticSummary.CopiedEdgeSnapshotCount),
            nameof(CudaGraphDiagnosticSummary.RuntimeEvidenceKind),
            nameof(CudaGraphDiagnosticSummary.IsRuntimeExecutionEvidence),
            nameof(CudaGraphDiagnosticSummary.IsRuntimeExecutionProof),
            nameof(CudaGraphDiagnosticSummary.CanPromoteReleaseProof),
            nameof(CudaGraphDiagnosticSummary.CanPromoteRuntimeProof),
            nameof(CudaGraphDiagnosticSummary.CanDeleteDeferredRecord),
            nameof(CudaGraphExecDiagnosticSnapshot),
            nameof(CudaGraphExecDiagnosticSnapshot.ToSummary),
            nameof(CudaGraphExecDiagnosticSummary),
            nameof(CudaGraphExecDiagnosticSummary.CopiedNodeStateCount),
            nameof(CudaGraphExecDiagnosticSummary.SnapshotsWithNodeTokenCount),
            nameof(CudaGraphExecDiagnosticSummary.RuntimeEvidenceKind),
            nameof(CudaGraphExecDiagnosticSummary.IsRuntimeExecutionEvidence),
            nameof(CudaGraphExecDiagnosticSummary.IsRuntimeExecutionProof),
            nameof(CudaGraphExecDiagnosticSummary.CanPromoteReleaseProof),
            nameof(CudaGraphExecDiagnosticSummary.CanPromoteRuntimeProof),
            nameof(CudaGraphExecDiagnosticSummary.CanDeleteDeferredRecord),
            nameof(CudaDevice.GetGraphMemorySummary),
            nameof(CudaDevice.CurrentGraphMemorySummary),
            nameof(CudaDeviceGraphMemoryInfo),
            nameof(CudaDeviceGraphMemoryInfo.ToSummary),
            nameof(CudaDeviceGraphMemorySummary),
            nameof(CudaDeviceGraphMemorySummary.CopiedScalarCounterCount),
            nameof(CudaDeviceGraphMemorySummary.CurrentCountersWithinHighWatermarks),
            nameof(CudaDeviceGraphMemorySummary.RuntimeEvidenceKind),
            nameof(CudaDeviceGraphMemorySummary.IsRuntimeExecutionEvidence),
            nameof(CudaDeviceGraphMemorySummary.IsRuntimeExecutionProof),
            nameof(CudaDeviceGraphMemorySummary.CanPromoteReleaseProof),
            nameof(CudaDeviceGraphMemorySummary.CanPromoteRuntimeProof),
            nameof(CudaDeviceGraphMemorySummary.CanDeleteDeferredRecord),
            nameof(CudaMemory.Advise),
            nameof(CudaMemory.PrefetchAsync));
    }
}
'@
    $program = $program.Replace("__RID__", $rid)
    $program = $program.Replace("__TENSORRT_LINE__", $tensorRtLineExpression)

    Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "NuGet.config") -Value $nugetConfig -Encoding utf8
    $restorePackagesPath = Get-RestorePackagesPath -RuntimeKey $key
    New-Item -ItemType Directory -Path $restorePackagesPath -Force | Out-Null

    $project = $project.Replace('$(MSBuildProjectDirectory)\.nuget\packages', $restorePackagesPath)
    Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "BridgePackageConsumer.csproj") -Value $project -Encoding utf8
    Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "Program.cs") -Value $program -Encoding utf8

    Invoke-CheckedDotNet -Arguments @("restore", (Join-Path $resolvedConsumerRoot "BridgePackageConsumer.csproj"), "--configfile", (Join-Path $resolvedConsumerRoot "NuGet.config"))
    Invoke-CheckedDotNet -Arguments @("build", (Join-Path $resolvedConsumerRoot "BridgePackageConsumer.csproj"), "-c", "Release", "--no-restore")

    $outputDirectory = Join-PathMany -Parts @($resolvedConsumerRoot, "bin", "Release", $TargetFramework, $rid)
    if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
      throw "Consumer output directory was not found: $outputDirectory"
    }

    $expectedManagedAssemblies = @("JYPPX.Shared.dll", "JYPPX.TensorRtSharp.dll", "JYPPX.CudaSharp.dll")
    foreach ($assembly in $expectedManagedAssemblies) {
      $matches = @(Get-ChildItem -LiteralPath $outputDirectory -Recurse -Filter $assembly)
      if ($matches.Count -eq 0) {
        throw "Expected managed assembly was not copied to the consumer output: $assembly"
      }
    }

    $bridgeMatches = @(Get-ChildItem -LiteralPath $outputDirectory -Recurse -Filter $bridgeFileName)
    if ($bridgeMatches.Count -eq 0) {
      throw "Expected bridge asset was not copied to the consumer output: $bridgeFileName"
    }

    $runtimeLayoutBridgePath = Join-PathMany -Parts @($outputDirectory, "runtimes", $rid, "native", $bridgeFileName)
    $rootBridgePath = Join-Path $outputDirectory $bridgeFileName
    $runtimeLayoutCopied = Test-Path -LiteralPath $runtimeLayoutBridgePath -PathType Leaf
    $rootCopied = Test-Path -LiteralPath $rootBridgePath -PathType Leaf

    $probeResult = "not-requested"
    $probeOutput = @()
    if (-not $SkipProbe.IsPresent) {
      $probeArguments = @("run", "--project", (Join-Path $resolvedConsumerRoot "BridgePackageConsumer.csproj"), "-c", "Release", "--no-build")
      $previousBridgePath = $env:JYPPX_NATIVE_BRIDGE_PATH
      $previousDevelopmentProbing = $env:JYPPX_ENABLE_DEVELOPMENT_PROBING
      try {
        $env:JYPPX_NATIVE_BRIDGE_PATH = $null
        $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $null
        $probeRun = Invoke-DotNetCommand -Arguments $probeArguments
      }
      finally {
        $env:JYPPX_NATIVE_BRIDGE_PATH = $previousBridgePath
        $env:JYPPX_ENABLE_DEVELOPMENT_PROBING = $previousDevelopmentProbing
      }

      $probeOutput = @($probeRun.OutputLines)
      if ($probeRun.ExitCode -eq 0) {
        $probeText = $probeOutput -join "`n"
        if ($probeText -match "EnvironmentProbe=Succeeded") {
          $probeResult = "environment-probe-succeeded"
        }
        elseif ($probeText -match "Skipped=True") {
          $probeResult = "skipped-with-diagnostic"
        }
        else {
          $probeResult = "dependency-probe-only"
        }
      }
      elseif (Test-ApplicationControlPolicyBlock -OutputLines $probeRun.OutputLines) {
        $probeResult = "blocked-by-application-control"
        Write-Warning "Bridge package consumer probe was blocked by the Windows application control policy on this runner. Restore/build/native asset validation passed, so packaging will continue."
      }
      else {
        throw "dotnet $($probeArguments -join ' ') failed with exit code $($probeRun.ExitCode)."
      }
    }

    $probeClassification = Get-NativeDependencyProbeClassification -ProbeResult $probeResult -OutputLines $probeOutput

    $timer.Stop()
    $elapsedSeconds = [Math]::Round($timer.Elapsed.TotalSeconds, 2)
    $wrapperSurfaceProbe = "compiled:plugin-inventory;plugin-inventory-field-metadata;engine-rnn-readonly-diagnostics;rnnv2-borrowed-state-design-gate;rnnv2-owner-bound-tensors;rnnv2-copied-gate-weights;managed-callbacks;callback-diagnostics;callback-api-language-safe-controls;error-recorder-snapshot;logger-presence-safe-controls;allocator-debug-listener-safe-controls;callback-interface-info-safe-controls;execution-context-callback-state-snapshot;execution-context-callback-allocator-safe-control-summary;allocator-owner-dry-run-diagnostics;allocator-owner-native-dry-run-controls;allocator-owner-state-ledger-dry-run-controls;allocator-owner-ledger-safety-gate;output-allocator-callback-owner-design;output-allocator-attach-detach-design-gate;output-allocator-runtime-proof-precheck;debug-listener-callback-owner-design;debug-listener-attach-detach-design-gate;debug-listener-borrowed-tensor-safety-gate;debug-listener-attach-vtable-safety-gate;debug-listener-native-attach-nothrow-preflight;debug-listener-native-owner-address-design-gate;debug-listener-native-nothrow-vtable-design-gate;debug-listener-native-attach-entry-design-gate;debug-listener-native-detach-before-release-design-gate;debug-listener-native-owner-lifecycle-dry-run;debug-listener-native-attach-entry-runtime-scaffold;debug-listener-native-attach-entry-minimal-safety;debug-listener-native-owner-stable-identity;debug-listener-native-owner-noncopyable-storage;debug-listener-native-nothrow-destructor;debug-listener-native-owner-lifecycle-gate;debug-listener-native-attach-bridge-shape-gate;debug-listener-exception-status-mapping-gate;debug-listener-inflight-accounting-gate;debug-listener-native-nothrow-vtable-scaffold-gate;debug-listener-nothrow-vtable-callback-stub;callback-stub-gate;debug-listener-borrowed-debug-tensor-metadata-runtime-gate;borrowed-debug-tensor-metadata-gate;debug-listener-native-vtable-install-preflight;native-vtable-install-preflight;debug-listener-native-owner-vtable-install-experiment;native-owner-vtable-install-experiment;debug-listener-runtime-proof-precheck;debug-listener-runtime-proof-attempt-preflight;debug-listener-real-non-null-attach-runtime-smoke;runtime-smoke-skipped;runtime-smoke-blocked;runtime-smoke-attempted;debug-listener-process-debug-tensor-callback-trampoline;callback-trampoline-shape;onnx-parser-diagnostic-snapshot;onnx-parser-diagnostic-summary;onnx-parser-refitter-diagnostic-snapshot;onnx-parser-refitter-diagnostic-summary;profiler-safe-controls;progress-monitor-safe-controls;cuda-memory-range;HasImplicitBatchDimensionCompatibility;SerializedPluginPathCountCompatibility;GetRnnV2LayerCount;GetRnnV2HiddenSize;GetRnnV2DataLength;GetRnnV2MaxSequenceLength;GetRnnV2Operation;GetRnnV2Direction;GetRnnV2InputMode;GetRnnV2CellState;GetRnnV2HiddenState;GetRnnV2SequenceLengths;GetRnnV2WeightsForGate;GetRnnV2BiasForGate;TensorRtRnnV2GateWeightsSnapshot;TensorRtRnnOperation;TensorRtRnnDirection;TensorRtRnnInputMode;TensorRtRnnGateType"
    $wrapperSurfaceProbe = "$wrapperSurfaceProbe;execution-context-auxiliary-stream-lifetime"
    $wrapperSurfaceEvidenceKind = "compile-surface-proof"
    $isRuntimeExecutionProof = $false
    $runtimeProofBoundary = "bridge-only consumer validates package layout, high-level wrapper compile surface, and dependency diagnostics; it is not clean package-consumer runtime proof."
    $parserSnapshotBoundary = "Parser/ParserRefitter diagnostic snapshots and summaries are copied managed surface proof only."
    foreach ($line in @($probeOutput)) {
      if ($line -like "HighLevelWrapperSurface=*") {
        $wrapperSurfaceProbe = $line.Substring("HighLevelWrapperSurface=".Length)
        break
      }
    }

    Write-Host "Bridge package consumer validation passed for $key."
    Write-Host "  Managed package: $($ManagedPackage.Id) $($ManagedPackage.Version)"
    Write-Host "  Bridge package: $($BridgePackage.Id) $($BridgePackage.Version)"
    Write-Host "  Bridge nupkg layout: runtimes/$rid/native/$bridgeFileName"
    Write-Host "  Bridge output root copy: $rootCopied"
    Write-Host "  Bridge output runtime layout copy: $runtimeLayoutCopied"
    Write-Host "  High-level wrapper surface: $wrapperSurfaceProbe"
    Write-Host "  Wrapper surface evidence kind: $wrapperSurfaceEvidenceKind"
    Write-Host "  Runtime execution proof: $isRuntimeExecutionProof"
    Write-Host "  Runtime proof boundary: $runtimeProofBoundary"
    Write-Host "  Probe: $probeResult"
    Write-Host "  Native dependency status: $($probeClassification.status)"
    Write-Host "  Native dependency diagnostic: $($probeClassification.diagnostic)"
    Write-Host "  Elapsed: ${elapsedSeconds}s"
    Write-Host "  Consumer output: $outputDirectory"

    return [pscustomobject]@{
      SourceRuntimeKey = $ResolvedPackage.SourceRuntimeKey
      BridgePackageKey = $key
      BridgePackageId = $BridgePackage.Id
      BridgePackageVersion = $BridgePackage.Version
      BridgePackagePath = $BridgePackage.Path
      ManagedPackageId = $ManagedPackage.Id
      ManagedPackageVersion = $ManagedPackage.Version
      TargetFramework = $TargetFramework
      RuntimeIdentifier = $rid
      TensorRtLine = [string]$splitPackage.tensorRtLine
      CudaLine = [string]$splitPackage.cudaLine
      BridgeFileName = $bridgeFileName
      BridgeNupkgRuntimeLayout = "runtimes/$rid/native/$bridgeFileName"
      BridgeOutputRootCopied = $rootCopied
      BridgeOutputRuntimeLayoutCopied = $runtimeLayoutCopied
      BridgeOutputPath = @($bridgeMatches | Select-Object -First 1).FullName
      ConsumerBuildConfiguration = "Release"
      RestoreSucceeded = $true
      BuildSucceeded = $true
      PackageConsumerValidationSucceeded = $true
      HighLevelWrapperSurface = $wrapperSurfaceProbe
      WrapperSurfaceEvidenceKind = $wrapperSurfaceEvidenceKind
      IsRuntimeExecutionProof = $isRuntimeExecutionProof
      IsPackageConsumerRuntimeProof = $false
      CanPromoteRuntimeProof = $false
      CanPublishPublicly = $false
      CanCloseReleaseIssue = $false
      RuntimeProofBoundary = $runtimeProofBoundary
      ParserSnapshotBoundary = $parserSnapshotBoundary
      ProbeResult = $probeResult
      ProbeDiagnostic = [string]$probeClassification.diagnostic
      NativeDependencyStatus = [string]$probeClassification.status
      NativeDependencyDiagnostic = [string]$probeClassification.diagnostic
      NativeDependencyProbeLine = [string]$probeClassification.dependencyProbeLine
      NativeDependencySkippedReason = [string]$probeClassification.skippedReason
      VendorExceptionCode = [string]$probeClassification.vendorExceptionCode
      IsNativeDependencyMissing = [bool]$probeClassification.isNativeDependencyMissing
      IsVendorStructuredException = [bool]$probeClassification.isVendorStructuredException
      ProbeOutput = @($probeOutput)
      ElapsedSeconds = $elapsedSeconds
      ConsumerOutput = $outputDirectory
      ConsumerOutputPreserved = $KeepConsumerOutput.IsPresent
    }
  }
  finally {
    if (-not [string]::IsNullOrWhiteSpace($restorePackagesPath) -and (Test-Path -LiteralPath $restorePackagesPath)) {
      try {
        Remove-ConsumerDirectory -Path $restorePackagesPath
        Write-Host "  Removed restore package cache: $restorePackagesPath"
      }
      catch {
        Write-Warning "Unable to remove restore package cache '$restorePackagesPath': $($_.Exception.Message)"
      }
    }

    if (-not $KeepConsumerOutput.IsPresent -and (Test-Path -LiteralPath $resolvedConsumerRoot)) {
      try {
        Remove-ConsumerDirectory -Path $resolvedConsumerRoot
        Write-Host "  Removed consumer output: $resolvedConsumerRoot"
      }
      catch {
        Write-Warning "Unable to remove consumer output '$resolvedConsumerRoot': $($_.Exception.Message)"
      }
    }
  }
}

function Write-ValidationReports {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Result,
    [Parameter(Mandatory = $true)]
    [string]$Directory
  )

  New-Item -ItemType Directory -Path $Directory -Force | Out-Null
  $jsonPath = Join-Path $Directory "bridge-package-consumer-validation-summary.json"
  $markdownPath = Join-Path $Directory "bridge-package-consumer-validation-summary.md"

  $Result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Bridge Package Consumer Validation Summary")
  $lines.Add("")
  $lines.Add("| Source runtime | Bridge package | Bridge asset | Output root | Output runtime layout | Wrapper surface | Probe | Native dependency | Vendor SEH | Elapsed |")
  $lines.Add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |")
  $package = '`' + $Result.BridgePackageId + ' ' + $Result.BridgePackageVersion + '`'
  $asset = '`' + $Result.BridgeNupkgRuntimeLayout + '`'
  $surface = '`' + $Result.HighLevelWrapperSurface + '`'
  $vendorSeh = if ([string]::IsNullOrWhiteSpace([string]$Result.VendorExceptionCode)) { "" } else { [string]$Result.VendorExceptionCode }
  $lines.Add("| $($Result.SourceRuntimeKey) | $package | $asset | $($Result.BridgeOutputRootCopied) | $($Result.BridgeOutputRuntimeLayoutCopied) | $surface | $($Result.ProbeResult) | $($Result.NativeDependencyStatus) | $vendorSeh | $($Result.ElapsedSeconds)s |")
  $lines.Add("")
  $lines.Add("- probe diagnostic: $($Result.ProbeDiagnostic)")
  $lines.Add("- consumer build configuration: $($Result.ConsumerBuildConfiguration)")
  $lines.Add("- restore succeeded: $($Result.RestoreSucceeded)")
  $lines.Add("- build succeeded: $($Result.BuildSucceeded)")
  $lines.Add("- package consumer validation succeeded: $($Result.PackageConsumerValidationSucceeded)")
  $lines.Add("- wrapper surface evidence kind: $($Result.WrapperSurfaceEvidenceKind)")
  $lines.Add("- runtime proof: $($Result.IsRuntimeExecutionProof)")
  $lines.Add("- package consumer runtime proof: $($Result.IsPackageConsumerRuntimeProof)")
  $lines.Add("- can promote runtime proof: $($Result.CanPromoteRuntimeProof)")
  $lines.Add("- can publish publicly: $($Result.CanPublishPublicly)")
  $lines.Add("- can close release issue: $($Result.CanCloseReleaseIssue)")
  $lines.Add("- runtime proof boundary: $($Result.RuntimeProofBoundary)")
  $lines.Add("- Parser/ParserRefitter diagnostic snapshots and summaries are copied managed surface proof only.")
  $lines.Add("- Skipped=True/dependency-probe-only is not runtime proof.")
  $lines.Add("- native dependency diagnostic: $($Result.NativeDependencyDiagnostic)")
  $lines.Add("- native dependency probe line: ``$($Result.NativeDependencyProbeLine)``")
  $lines.Add("")
  $lines.Add('Generated by `eng/Test-BridgePackageConsumer.ps1`.')
  Set-Content -LiteralPath $markdownPath -Value $lines -Encoding utf8

  Write-Host "Bridge package consumer summary written to $jsonPath"
  Write-Host "Bridge package consumer summary written to $markdownPath"
}

if ([string]::IsNullOrWhiteSpace($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"
}
elseif (-not [System.IO.Path]::IsPathRooted($ManagedPackageDirectory)) {
  $ManagedPackageDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ManagedPackageDirectory))
}

$resolvedPackage = Resolve-BridgeSplitPackage -SourceKey $SourceRuntimeKey -SplitKey $BridgePackageKey

if ([string]::IsNullOrWhiteSpace($BridgePackageDirectory)) {
  $BridgePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$($resolvedPackage.SourceRuntimeKey)"
}
elseif (-not [System.IO.Path]::IsPathRooted($BridgePackageDirectory)) {
  $BridgePackageDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $BridgePackageDirectory))
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "build-out\bridge-package-consumer"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $OutputRoot))
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\package-consumer"
}
elseif (-not [System.IO.Path]::IsPathRooted($ReportDirectory)) {
  $ReportDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ReportDirectory))
}

$managedPackage = Find-Package -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
Assert-ManagedPackageFreshness -ManagedPackage $managedPackage
$bridgePackage = Find-Package -Directory $BridgePackageDirectory -PackageId $resolvedPackage.SplitPackage.packageId
$result = Invoke-BridgePackageConsumerValidation -ResolvedPackage $resolvedPackage -ManagedPackage $managedPackage -BridgePackage $bridgePackage
Write-ValidationReports -Result $result -Directory $ReportDirectory
