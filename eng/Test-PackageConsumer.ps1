[CmdletBinding()]
param(
  [string[]]$RuntimePackageKey = @("win-x64-trt8.6-cuda11.8-cudnn8.9"),
  [string]$TargetFramework = "net8.0",
  [string]$ManagedPackageDirectory,
  [string]$RuntimePackageDirectory,
  [string[]]$AdditionalPackageSource = @(),
  [string]$AdditionalPackageSourceUsername,
  [string]$AdditionalPackageSourcePassword,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [switch]$RunSmoke,
  [switch]$AllowSmokeFailure,
  [string[]]$SmokeRuntimePackageKey = @(),
  [switch]$KeepConsumerOutput,
  [switch]$SignConsumerOutput,
  [switch]$TrustSigningCertificate,
  [switch]$TrustSigningCertificateRoot,
  [string]$CertificateThumbprint,
  [string]$CertificateSubject = "CN=JYPPX TensorRtSharp Local Dev Code Signing",
  [string]$SigntoolPath,
  [string]$RepositoryRoot
)

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
  $OutputRoot = Join-Path $RepositoryRoot "build-out\package-consumer"
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

function New-NuGetConfigContent {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ManagedSource,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeSource
  )

  $packageSources = New-Object System.Collections.Generic.List[string]
  $packageSources.Add('    <clear />')
  $packageSources.Add('    <add key="jyppx-managed" value="' + (ConvertTo-XmlAttributeValue -Value $ManagedSource) + '" />')
  $packageSources.Add('    <add key="jyppx-runtime" value="' + (ConvertTo-XmlAttributeValue -Value $RuntimeSource) + '" />')

  $sourceIndex = 1
  foreach ($source in @(Expand-KeyList -Values $AdditionalPackageSource)) {
    $resolvedSource = Resolve-PackageSourceValue -Source $source
    $packageSources.Add('    <add key="additional-' + $sourceIndex + '" value="' + (ConvertTo-XmlAttributeValue -Value $resolvedSource) + '" />')
    $sourceIndex++
  }

  $packageSourceCredentials = New-Object System.Collections.Generic.List[string]
  if (-not [string]::IsNullOrWhiteSpace($AdditionalPackageSourcePassword)) {
    $credentialUserName = if ([string]::IsNullOrWhiteSpace($AdditionalPackageSourceUsername)) { "github" } else { $AdditionalPackageSourceUsername }
    $additionalSourceCount = $sourceIndex - 1
    for ($credentialIndex = 1; $credentialIndex -le $additionalSourceCount; $credentialIndex++) {
      $packageSourceCredentials.Add('    <additional-' + $credentialIndex + '>')
      $packageSourceCredentials.Add('      <add key="Username" value="' + (ConvertTo-XmlAttributeValue -Value $credentialUserName) + '" />')
      $packageSourceCredentials.Add('      <add key="ClearTextPassword" value="' + (ConvertTo-XmlAttributeValue -Value $AdditionalPackageSourcePassword) + '" />')
      $packageSourceCredentials.Add('    </additional-' + $credentialIndex + '>')
    }
  }

  $packageSourceCredentialBlock = if ($packageSourceCredentials.Count -gt 0) {
    "  <packageSourceCredentials>`r`n$($packageSourceCredentials -join "`r`n")`r`n  </packageSourceCredentials>"
  }
  else {
    ""
  }

  return @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
$($packageSources -join "`r`n")
  </packageSources>
$packageSourceCredentialBlock
</configuration>
"@
}

function Get-RestorePackagesPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeKey
  )

  $safeKey = ($RuntimeKey -replace '[^A-Za-z0-9\.-]', '-')
  $isWindowsHost = $false
  try {
    $isWindowsHost = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
  }
  catch {
    $isWindowsHost = $env:OS -eq "Windows_NT"
  }

  if ($isWindowsHost) {
    $root = Join-Path $env:SystemDrive "jyppx-pkgcache"
    return Join-Path $root $safeKey
  }

  return [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "jyppx-pkgcache", $safeKey)
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
  "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke",
  "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult",
  "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline",
  "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult",
  "TensorRtDebugTensorMetadataSnapshot",
  "TensorRtDebugListenerRealCallbackRuntimeProof",
  "TensorRtDebugListenerRealCallbackRuntimeProofResult",
  "TensorRtDebugListenerCallbackProofGapReport",
  "TensorRtDebugListenerCallbackProofGapReportResult",
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
  "BorrowedDebugTensorMetadataCopyReady",
  "BorrowedDebugTensorPointerEscapeBlocked",
  "BorrowedDebugTensorDataPointerEscapeBlocked",
  "DebugTensorPointerExposed",
  "DebugTensorDataPointerExposed",
  "BorrowedDebugTensorMetadataGateReady",
  "NativeVTableInstallPreflightReady",
  "VTableInstallShapeReady",
  "VTableInstallVersionGuardReady",
  "VTableInstallNoThrowBoundaryReady",
  "VTableInstallOwnershipDiagnosticsReady",
  "VTableInstallPointerFree",
  "NativeVTableInstallRuntimeReady",
  "ReasonNativeVTableInstallStillBlocked",
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
  "LastDiagnostic",
  "FullPackageConsumerReport",
  "ReportPointerFree",
  "BorrowedDebugTensorLifetimeReady",
  "BorrowedDebugTensorDataLifetimeReady",
  "ReasonMetadataRuntimeStillBlocked",
  "RuntimeProofBlocked",
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
  "CanEnableSetDebugListenerNonNull",
  "CanInstallNativeVTable",
  "CanCallProcessDebugTensorRuntime",
  "CanPromoteRealCallbackRuntime",
  "ReasonNonNullAttachStillBlocked",
  "ReasonNativeVTableStillBlocked",
  "ReasonRuntimeProofStillBlocked",
  "TrampolineShapeReady",
  "NativeCallbackEntryLocated",
  "NoThrowCallbackEntryReady",
  "ExceptionCaptureReady",
  "InFlightAccountingReady",
  "PointerFreeSurfaceReady",
  "CallbackStubEntryCount",
  "CallbackStubLeaveCount",
  "TensorRtDebugTensorMetadataSnapshot",
  "RuntimeSmokeReady",
  "TrampolineShapeReady",
  "AttemptedNoInvocation",
  "BorrowedDebugTensorMetadataCopied"
)

function Get-ManagedPackageXmlSurface {
  param(
    [Parameter(Mandatory = $true)]
    [object]$ManagedPackage
  )

  $zip = [System.IO.Compression.ZipFile]::OpenRead($ManagedPackage.Path)
  try {
    $xmlEntries = @($zip.Entries | Where-Object {
        $_.FullName.EndsWith("JYPPX.TensorRtSharp.xml", [System.StringComparison]::OrdinalIgnoreCase) -and
        $_.FullName.StartsWith("lib/", [System.StringComparison]::OrdinalIgnoreCase)
      })
    if ($xmlEntries.Count -eq 0) {
      throw "Managed package appears stale: '$($ManagedPackage.Id)' $($ManagedPackage.Version) at '$($ManagedPackage.Path)' does not contain lib/*/JYPPX.TensorRtSharp.xml. Repack the managed package with: $script:ManagedPackageFreshnessPackCommand"
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

function Test-CudaDriverRuntimeCompatibilityBlock {
  param(
    [string[]]$OutputLines
  )

  $text = ($OutputLines -join "`n")
  return (
    $text -match 'CUDA error 35' -or
    $text -match 'cudaErrorInsufficientDriver' -or
    $text -match 'driver version is insufficient' -or
    $text -match 'cudaRuntimeGetVersion failed'
  )
}

$script:RealCallbackRuntimeRequiredSmokeMarkers = @(
  "EvidenceKind=real-callback-runtime",
  "RuntimeEvidenceKind=real-callback-runtime",
  "RealCallbackRuntime=True",
  "IsRealCallbackRuntimeProof=True",
  "CallbackKind",
  "TensorRtLine",
  "CudaLine",
  "RuntimePackageKey",
  "OwnerId",
  "InvocationCount",
  "AllocationCount",
  "ReleaseCount",
  "FailureCount",
  "InFlightCallbackCount",
  "LastStatus",
  "LastDiagnostic",
  "FullPackageConsumerReport"
)

$script:RealCallbackRuntimeNonProofSmokeMarkers = @(
  "allocator-owner-internal-runtime-prototype",
  "allocator-owner-ledger-safety-gate",
  "output-allocator-internal-runtime-gate",
  "output-allocator-callback-owner-design",
  "output-allocator-attach-detach-design-gate",
  "output-buffer-ownership-safety-gate",
  "output-allocator-runtime-proof-precheck",
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
  "debug-listener-borrowed-debug-tensor-metadata-runtime-gate",
  "debug-listener-native-vtable-install-preflight",
  "debug-listener-native-owner-vtable-install-experiment",
  "debug-listener-real-non-null-attach-runtime-smoke",
  "debug-listener-process-debug-tensor-callback-trampoline",
  "debug-listener-real-callback-runtime-proof",
  "debug-listener-callback-proof-gap-report",
  "callback-owner-closure-matrix",
  "debug-listener-runtime-proof-precheck",
  "debug-listener-runtime-proof-attempt-preflight",
  "attach-bridge-shape-gate",
  "exception-status-gate",
  "inflight-accounting-gate",
  "vtable-scaffold-gate",
  "callback-stub-gate",
  "borrowed-debug-tensor-metadata-gate",
  "native-vtable-install-preflight",
  "native-owner-vtable-install-experiment",
  "callback-trampoline-shape",
  "real-callback-runtime-blocked",
  "attempted-no-invocation",
  "runtime-smoke-skipped",
  "runtime-smoke-blocked",
  "runtime-smoke-attempted",
  "runtime-smoke-failed",
  "RealCallbackRuntime=False",
  "IsRealCallbackRuntimeProof=False",
  "CanPromoteRealCallbackRuntime=False",
  "not proof",
  "minimal-safety",
  "runtime-gate",
  "runtime-proof-attempt-preflight",
  "dependency-probe-only",
  "copied-state",
  "dry-run"
)

function Test-SmokeOutputContainsAnyMarker {
  param(
    [AllowNull()]
    [AllowEmptyString()]
    [string]$Text = "",
    [Parameter(Mandatory = $true)]
    [string[]]$Markers
  )

  if ([string]::IsNullOrEmpty($Text)) {
    return $false
  }

  foreach ($marker in @($Markers)) {
    if ($Text.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }

  return $false
}

function New-RealCallbackRuntimeEvidenceFromSmoke {
  param(
    [bool]$SmokeRequested,
    [string]$SmokeResult,
    $SmokeExitCode = $null,
    [string]$SmokeDiagnostic = "",
    [string[]]$SmokeOutputLines = @()
  )

  $requiredSmokeMarkers = @($script:RealCallbackRuntimeRequiredSmokeMarkers)
  $smokeLines = @($SmokeOutputLines | ForEach-Object { [string]$_ })
  $combinedSmokeOutput = $smokeLines -join "`n"
  $hasRuntimeMarker = $combinedSmokeOutput.IndexOf("EvidenceKind=real-callback-runtime", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
  $hasNonProofCallbackEvidence = Test-SmokeOutputContainsAnyMarker -Text $combinedSmokeOutput -Markers $script:RealCallbackRuntimeNonProofSmokeMarkers
  $runtimeSmokeLines = @($smokeLines | Where-Object {
      $_.IndexOf("DebugListenerRealNonNullAttachRuntimeSmoke=", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("debug-listener-real-non-null-attach-runtime-smoke", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("DebugListenerProcessDebugTensorCallbackTrampoline=", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("debug-listener-process-debug-tensor-callback-trampoline", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("DebugListenerRealCallbackRuntimeProof=", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("debug-listener-real-callback-runtime-proof", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("DebugListenerCallbackProofGapReport=", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("debug-listener-callback-proof-gap-report", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("callback-owner-closure-matrix", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    })

  if (-not $SmokeRequested) {
    return [pscustomobject]@{
      Status = "not-present"
      Marker = "real-callback-runtime"
      EvidenceKind = "not-present"
      RuntimeEvidenceKind = "not-present"
      RequiredSmokeMarkers = @($requiredSmokeMarkers)
      MissingSmokeMarkers = @()
      MatchedSmokeLines = @($runtimeSmokeLines)
      IsRealCallbackRuntimeProof = $false
      Diagnostic = "package consumer smoke was not requested; real-callback-runtime evidence is not present."
    }
  }

  if (-not $hasRuntimeMarker) {
    $status = switch ($SmokeResult) {
      "blocked-by-cuda-driver" { "blocked-by-cuda-driver"; break }
      "blocked-by-application-control" { "blocked-by-application-control"; break }
      "failed" { "blocked"; break }
      default {
        if ($runtimeSmokeLines.Count -gt 0) {
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-skipped", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "skipped"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-blocked", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "blocked"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-attempted", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "attempted"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-failed", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "failed"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=real-callback-runtime-blocked", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "blocked"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=attempted-no-invocation", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "attempted"; break }
          "blocked"
          break
        }

        "not-present"
        break
      }
    }

    $diagnostic = switch ($status) {
      "blocked-by-cuda-driver" { "package consumer smoke reached the packaged runtime, but CUDA driver/runtime compatibility blocked callback runtime evidence collection."; break }
      "blocked-by-application-control" { "package consumer smoke was blocked by application control before callback runtime evidence could be collected."; break }
      "blocked" { "package consumer smoke failed without reporting real-callback-runtime evidence: $SmokeDiagnostic"; break }
      "skipped" { "package consumer smoke reported debug-listener runtime smoke skipped evidence only; real-callback-runtime evidence is not present."; break }
      "attempted" { "package consumer smoke reported debug-listener runtime smoke attempted evidence only; real-callback-runtime proof is not present."; break }
      "failed" { "package consumer smoke reported debug-listener runtime smoke failed evidence only; real-callback-runtime proof is not present."; break }
      default {
        if ($hasNonProofCallbackEvidence) {
          "package consumer smoke reported dry-run/copied-state/internal-runtime-gate/precheck/dependency-probe evidence only; real-callback-runtime evidence is not present."
        }
        else {
          "package consumer smoke did not report real-callback-runtime evidence."
        }
        break
      }
    }

    return [pscustomobject]@{
      Status = $status
      Marker = "real-callback-runtime"
      EvidenceKind = "not-present"
      RuntimeEvidenceKind = "not-present"
      RequiredSmokeMarkers = @($requiredSmokeMarkers)
      MissingSmokeMarkers = @()
      MatchedSmokeLines = @($runtimeSmokeLines)
      IsRealCallbackRuntimeProof = $false
      Diagnostic = $diagnostic
    }
  }

  $missingSmokeMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredSmokeMarkers)) {
    if ($combinedSmokeOutput.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingSmokeMarkers.Add($marker)
    }
  }

  $matchedSmokeLines = @($smokeLines | Where-Object {
      $_.IndexOf("real-callback-runtime", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("RealCallbackRuntime", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("CallbackKind", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("RuntimeEvidenceKind", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("IsRealCallbackRuntimeProof", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("InvocationCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("AllocationCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("ReleaseCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("FailureCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("FullPackageConsumerReport", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    })

  $invocationCount = 0
  $invocationMatches = [regex]::Matches($combinedSmokeOutput, "InvocationCount=(?<count>\d+)")
  foreach ($match in @($invocationMatches)) {
    $parsedCount = 0
    if ([int]::TryParse($match.Groups["count"].Value, [ref]$parsedCount) -and $parsedCount -gt $invocationCount) {
      $invocationCount = $parsedCount
    }
  }

  if ($invocationCount -le 0) {
    $missingSmokeMarkers.Add("InvocationCount>0")
  }

  $hasBlockingNonProofRuntimeKind =
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=proof-gap-report", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=closure-matrix", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-skipped", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-blocked", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-attempted", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-failed", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=callback-trampoline-shape", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=real-callback-runtime-blocked", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=attempted-no-invocation", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
    $combinedSmokeOutput.IndexOf("IsRealCallbackRuntimeProof=False", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
  if ($hasBlockingNonProofRuntimeKind) {
    $missingSmokeMarkers.Add("NoNonProofCallbackRuntimeMarker")
  }

  $isReady = [string]$SmokeResult -eq "passed" -and $missingSmokeMarkers.Count -eq 0 -and $invocationCount -gt 0 -and -not $hasBlockingNonProofRuntimeKind
  $status = if ($isReady) { "ready" } else { "incomplete" }
  $diagnostic = if ($isReady) {
    "package consumer smoke reported complete real-callback-runtime evidence."
  }
  elseif ([string]$SmokeResult -ne "passed") {
    "real-callback-runtime markers were found, but package consumer smoke did not pass."
  }
  else {
    "real-callback-runtime markers were found, but required smoke fields are missing, InvocationCount is zero, or non-proof callback markers are present."
  }

  return [pscustomobject]@{
    Status = $status
    Marker = "real-callback-runtime"
    EvidenceKind = if ($isReady) { "real-callback-runtime" } else { "incomplete-real-callback-runtime" }
    RuntimeEvidenceKind = if ($isReady) { "real-callback-runtime" } else { "incomplete-real-callback-runtime" }
    RequiredSmokeMarkers = @($requiredSmokeMarkers)
    MissingSmokeMarkers = @($missingSmokeMarkers.ToArray())
    MatchedSmokeLines = @($matchedSmokeLines)
    InvocationCount = $invocationCount
    IsRealCallbackRuntimeProof = $isReady
    Diagnostic = $diagnostic
  }
}

function ConvertTo-MarkdownCell {
  param(
    [string]$Value
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return ""
  }

  return (($Value -replace '\|', '\|') -replace "(`r`n|`n|`r)", "<br>")
}

function Get-PackageConsumerRuntimeProofPreflightMatrix {
  $matrixPath = Join-Path $RepositoryRoot "artifacts\final-release\package-consumer-runtime-proof-preflight-matrix.json"
  if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $matrixPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Find-PackageConsumerRuntimeProofPreflightEntry {
  param(
    [object]$Matrix,
    [Parameter(Mandatory = $true)]
    [string]$RuntimePackageKey
  )

  if ($null -eq $Matrix -or $Matrix.PSObject.Properties.Name -notcontains "entries") {
    return $null
  }

  return @($Matrix.entries | Where-Object { $_.runtimePackageKey -eq $RuntimePackageKey } | Select-Object -First 1)[0]
}

function Find-Signtool {
  param(
    [string]$PreferredPath
  )

  if (-not [string]::IsNullOrWhiteSpace($PreferredPath)) {
    if (-not (Test-Path -LiteralPath $PreferredPath -PathType Leaf)) {
      throw "signtool.exe was not found at the specified path: $PreferredPath"
    }

    return (Resolve-Path -LiteralPath $PreferredPath).Path
  }

  $command = Get-Command signtool.exe -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }

  $kitsRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
  if (Test-Path -LiteralPath $kitsRoot -PathType Container) {
    $matches = @(Get-ChildItem -LiteralPath $kitsRoot -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
      Sort-Object FullName -Descending)
    if ($matches.Count -gt 0) {
      return $matches[0].FullName
    }
  }

  throw "signtool.exe was not found. Install Windows SDK or pass -SigntoolPath."
}

function Get-OrCreate-CodeSigningCertificate {
  param(
    [string]$Thumbprint,
    [string]$Subject
  )

  if (-not [string]::IsNullOrWhiteSpace($Thumbprint)) {
    $normalizedThumbprint = ($Thumbprint -replace "\s", "").ToUpperInvariant()
    $certificate = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue |
      Where-Object { $_.Thumbprint -eq $normalizedThumbprint } |
      Select-Object -First 1
    if (-not $certificate) {
      throw "Code signing certificate was not found in Cert:\CurrentUser\My: $normalizedThumbprint"
    }

    return $certificate
  }

  $certificate = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert -ErrorAction SilentlyContinue |
    Where-Object { $_.Subject -eq $Subject } |
    Sort-Object NotAfter -Descending |
    Select-Object -First 1
  if ($certificate) {
    return $certificate
  }

  $newSelfSignedCertificate = Get-Command New-SelfSignedCertificate -ErrorAction SilentlyContinue
  if (-not $newSelfSignedCertificate) {
    throw "No matching code signing certificate was found and New-SelfSignedCertificate is unavailable."
  }

  Write-Host "Creating local development code signing certificate: $Subject"
  return New-SelfSignedCertificate `
    -Type CodeSigningCert `
    -Subject $Subject `
    -CertStoreLocation Cert:\CurrentUser\My `
    -KeyExportPolicy Exportable `
    -KeyUsage DigitalSignature `
    -NotAfter (Get-Date).AddYears(5)
}

function Add-CertificateToStore {
  param(
    [Parameter(Mandatory = $true)]
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
    [Parameter(Mandatory = $true)]
    [System.Security.Cryptography.X509Certificates.StoreName]$StoreName
  )

  $store = [System.Security.Cryptography.X509Certificates.X509Store]::new(
    $StoreName,
    [System.Security.Cryptography.X509Certificates.StoreLocation]::CurrentUser)
  try {
    $store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadWrite)
    $existing = $store.Certificates |
      Where-Object { $_.Thumbprint -eq $Certificate.Thumbprint } |
      Select-Object -First 1
    if ($existing) {
      return "already-present"
    }

    $store.Add($Certificate)
    return "added"
  }
  finally {
    $store.Close()
  }
}

function Ensure-ConsumerSigningCertificateTrust {
  param(
    [Parameter(Mandatory = $true)]
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate
  )

  $publisherStatus = Add-CertificateToStore -Certificate $Certificate -StoreName ([System.Security.Cryptography.X509Certificates.StoreName]::TrustedPublisher)
  $rootStatus = "not-requested"
  if ($TrustSigningCertificateRoot.IsPresent) {
    $rootStatus = Add-CertificateToStore -Certificate $Certificate -StoreName ([System.Security.Cryptography.X509Certificates.StoreName]::Root)
  }

  return [pscustomobject]@{
    currentUserTrustedPublisher = $publisherStatus
    currentUserRoot = $rootStatus
  }
}

function Sign-ConsumerOutput {
  param(
    [string]$OutputDirectory
  )

  $resolvedSigntoolPath = Find-Signtool -PreferredPath $SigntoolPath
  $certificate = Get-OrCreate-CodeSigningCertificate -Thumbprint $CertificateThumbprint -Subject $CertificateSubject
  $trustStatus = [pscustomobject]@{
    currentUserTrustedPublisher = "not-requested"
    currentUserRoot = "not-requested"
  }
  if ($TrustSigningCertificate.IsPresent) {
    $trustStatus = Ensure-ConsumerSigningCertificateTrust -Certificate $certificate
  }
  $candidates = @(Get-ChildItem -LiteralPath $OutputDirectory -File -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -like "JYPPX*.dll" -or
      $_.Name -eq "PackageConsumerSmoke.dll" -or
      $_.Name -eq "PackageConsumerSmoke.exe" -or
      $_.Name -eq "jyppxtrtbridge.dll"
    })

  if ($candidates.Count -eq 0) {
    throw "No consumer output files were found for signing under $OutputDirectory."
  }

  foreach ($file in $candidates) {
    Write-Host "Signing consumer output: $($file.Name)"
    $signOutput = & $resolvedSigntoolPath sign /fd SHA256 /sha1 $certificate.Thumbprint /tr http://timestamp.digicert.com /td SHA256 $file.FullName 2>&1
    foreach ($line in @($signOutput)) {
      Write-Host $line
    }

    if ($LASTEXITCODE -ne 0) {
      throw "signtool failed for '$($file.FullName)' with exit code $LASTEXITCODE."
    }
  }

  return [pscustomobject]@{
    count = [int]$candidates.Count
    certificateThumbprint = $certificate.Thumbprint
    currentUserTrustedPublisher = $trustStatus.currentUserTrustedPublisher
    currentUserRoot = $trustStatus.currentUserRoot
  }
}

function Unblock-ConsumerRuntimeAssets {
  param(
    [string]$OutputDirectory,
    [string[]]$FileNames
  )

  $unblockCommand = Get-Command Unblock-File -ErrorAction SilentlyContinue
  if (-not $unblockCommand) {
    return
  }

  foreach ($fileName in @($FileNames)) {
    if ([string]::IsNullOrWhiteSpace($fileName)) {
      continue
    }

    foreach ($match in @(Get-ChildItem -LiteralPath $OutputDirectory -Recurse -Filter $fileName -File -ErrorAction SilentlyContinue)) {
      try {
        Unblock-File -LiteralPath $match.FullName -ErrorAction Stop
      }
      catch {
        Write-Warning "Unable to unblock consumer runtime asset '$($match.FullName)': $($_.Exception.Message)"
      }
    }
  }
}

function Get-ExpectedNativeFileNames {
  param(
    [object]$RuntimePackage
  )

  $expectedNativeFiles = @($RuntimePackage.bridgeFile)
  foreach ($relativePath in @($RuntimePackage.tensorRtFiles + $RuntimePackage.cudaFiles + $RuntimePackage.cudnnFiles)) {
    if ([string]::IsNullOrWhiteSpace([string]$relativePath)) {
      continue
    }

    $expectedNativeFiles += [System.IO.Path]::GetFileName($relativePath)
  }

  return @($expectedNativeFiles | Sort-Object -Unique)
}

function New-RuntimePackageReferenceItems {
  param(
    [object]$RuntimeNupkg
  )

  $items = New-Object System.Collections.Generic.List[string]
  $items.Add("    <PackageReference Include=""$($RuntimeNupkg.Id)"" Version=""$($RuntimeNupkg.Version)"" />")

  if ([string]::IsNullOrWhiteSpace([string]$RuntimeNupkg.Path) -or -not (Test-Path -LiteralPath $RuntimeNupkg.Path -PathType Leaf)) {
    return @($items.ToArray())
  }

  $zip = [System.IO.Compression.ZipFile]::OpenRead($RuntimeNupkg.Path)
  try {
    $nuspec = $zip.Entries | Where-Object { $_.FullName.EndsWith(".nuspec", [System.StringComparison]::OrdinalIgnoreCase) } | Select-Object -First 1
    if (-not $nuspec) {
      return @($items.ToArray())
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
    foreach ($dependency in @($xml.SelectNodes("//n:metadata/n:dependencies//n:dependency", $namespaceManager))) {
      $id = [string]$dependency.id
      $version = [string]$dependency.version
      if ([string]::IsNullOrWhiteSpace($id) -or [string]::IsNullOrWhiteSpace($version)) {
        continue
      }

      $items.Add("    <PackageReference Include=""$id"" Version=""$version"" />")
    }
  }
  finally {
    $zip.Dispose()
  }

  return @($items.ToArray())
}

function Write-ValidationReports {
  param(
    [object[]]$Results,
    [string]$Directory,
    [object]$PreflightMatrix
  )

  New-Item -ItemType Directory -Path $Directory -Force | Out-Null
  $jsonPath = Join-Path $Directory "package-consumer-validation-summary.json"
  $markdownPath = Join-Path $Directory "package-consumer-validation-summary.md"

  $Results | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Package Consumer Validation Summary")
  $lines.Add("")
  $lines.Add("| Runtime key | Package | Configuration | Restore | Build | Validation | Native assets | Missing native assets | Smoke | Smoke exit code | Callback runtime evidence | Runtime proof | Smoke diagnostic | Signed consumer output | Elapsed |")
  $lines.Add("| --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | ---: |")
  foreach ($result in $Results) {
    $missing = if ($result.MissingNativeAssets.Count -eq 0) { "none" } else { ($result.MissingNativeAssets -join ", ") }
    $package = '`' + $result.RuntimePackageId + ' ' + $result.RuntimePackageVersion + '`'
    $smokeExitCode = if ($null -eq $result.SmokeExitCode) { "" } else { [string]$result.SmokeExitCode }
    $callbackEvidenceStatus = if ($result.PSObject.Properties.Name -contains "RealCallbackRuntimeEvidence") { [string]$result.RealCallbackRuntimeEvidence.Status } else { "not-present" }
    $callbackEvidenceProof = if ($result.PSObject.Properties.Name -contains "RealCallbackRuntimeEvidence") { [bool]$result.RealCallbackRuntimeEvidence.IsRealCallbackRuntimeProof } else { $false }
    $lines.Add("| $($result.RuntimePackageKey) | $package | $($result.ConsumerBuildConfiguration) | $($result.RestoreSucceeded) | $($result.BuildSucceeded) | $($result.PackageConsumerValidationSucceeded) | $($result.NativeAssetsFound)/$($result.NativeAssetsExpected) | $missing | $($result.SmokeResult) | $smokeExitCode | $callbackEvidenceStatus | $callbackEvidenceProof | $(ConvertTo-MarkdownCell -Value ([string]$result.SmokeDiagnostic)) | $($result.ConsumerOutputSigned) | $($result.ElapsedSeconds)s |")
  }

  $lines.Add("")
  $lines.Add("## Evidence Classification")
  $lines.Add("")
  $lines.Add("| Runtime key | Evidence kind | Runtime smoke classification | Runtime execution evidence | Package consumer runtime proof | Dependency probe only | Readonly summary evidence | Wrapper surface evidence | Real callback proof |")
  $lines.Add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  foreach ($result in $Results) {
    $evidenceKind = if ($result.PSObject.Properties.Name -contains "EvidenceKind") { [string]$result.EvidenceKind } else { "legacy-package-consumer-evidence" }
    $runtimeSmokeClassification = if ($result.PSObject.Properties.Name -contains "RuntimeSmokeClassification") { [string]$result.RuntimeSmokeClassification } else { [string]$result.SmokeResult }
    $isRuntimeExecutionEvidence = if ($result.PSObject.Properties.Name -contains "IsRuntimeExecutionEvidence") { [bool]$result.IsRuntimeExecutionEvidence } else { $false }
    $isPackageConsumerRuntimeProof = if ($result.PSObject.Properties.Name -contains "IsPackageConsumerRuntimeProof") { [bool]$result.IsPackageConsumerRuntimeProof } else { $false }
    $isDependencyProbeOnly = if ($result.PSObject.Properties.Name -contains "IsDependencyProbeOnly") { [bool]$result.IsDependencyProbeOnly } else { $true }
    $readonlySummaryEvidenceKind = if ($result.PSObject.Properties.Name -contains "ReadonlySummaryEvidenceKind") { [string]$result.ReadonlySummaryEvidenceKind } else { "not-recorded" }
    $wrapperSurfaceEvidenceKind = if ($result.PSObject.Properties.Name -contains "WrapperSurfaceEvidenceKind") { [string]$result.WrapperSurfaceEvidenceKind } else { "not-recorded" }
    $isRealCallbackRuntimeProof = if ($result.PSObject.Properties.Name -contains "IsRealCallbackRuntimeProof") { [bool]$result.IsRealCallbackRuntimeProof } else { $false }
    $lines.Add("| $($result.RuntimePackageKey) | $evidenceKind | $runtimeSmokeClassification | $isRuntimeExecutionEvidence | $isPackageConsumerRuntimeProof | $isDependencyProbeOnly | $readonlySummaryEvidenceKind | $wrapperSurfaceEvidenceKind | $isRealCallbackRuntimeProof |")
  }
  $lines.Add("")
  $lines.Add("`IsRuntimeExecutionEvidence=True` only means the package consumer smoke exited successfully. It still does not imply real callback runtime proof unless `IsRealCallbackRuntimeProof=True` and the callback markers are present.")
  $lines.Add("`IsPackageConsumerRuntimeProof=True` is reserved for a clean consumer restore/build/native-copy/runtime-smoke record with package hashes, host metadata, stdout/stderr summaries, real log hashes, no ProjectReference, and strict validator success. This script does not infer that state from readonly summary markers, bridge-only evidence, local feed restore, dependency probe, or `SmokeResult=passed` alone.")
  $lines.Add("`IsDependencyProbeOnly=True` means the report must be treated as packaging/native-copy/dependency evidence, not as runtime execution proof.")
  $lines.Add("Readonly summary markers such as `EngineDeploymentSummary=`, `BuilderConfigDeploymentSummary=`, `ExecutionContextDeploymentSummary=`, `SerializationConfigSummary=`, `RuntimeConfigSummary=`, `GraphDiagnosticSummary=`, `GraphExecDiagnosticSummary=`, and `MemoryRangeSummary=` are API/wrapper diagnostics only and are not package-consumer runtime proof.")
  $lines.Add("")
  $lines.Add("## Runtime Proof Preflight Boundary")
  $lines.Add("")
  if ($null -eq $PreflightMatrix) {
    $lines.Add("`artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json` was not found. Treat this report as package-consumer diagnostics only; it cannot promote package-consumer runtime proof.")
    $lines.Add("")
  }
  else {
    $boundary = $PreflightMatrix.proofBoundary
    $lines.Add("Preflight matrix schema: ``$($PreflightMatrix.schemaVersion)``.")
    $lines.Add("Promotion rule: $(ConvertTo-MarkdownCell -Value ([string]$boundary.promotionRule))")
    $lines.Add("This script copies preflight boundary fields into each result, but it keeps `IsPackageConsumerRuntimeProof=False` until a strict external proof validator promotes a real owner runtime smoke record.")
    $lines.Add("")
    $lines.Add("| Runtime key | Preflight entry | Owner action required | Restore source mode | Native assets expected | Can promote from this report | Blocked reason |")
    $lines.Add("| --- | --- | --- | --- | ---: | --- | --- |")
    foreach ($result in $Results) {
      if ($result.PSObject.Properties.Name -contains "RuntimeProofPreflight") {
        $preflight = $result.RuntimeProofPreflight
        $hasEntry = [bool]$preflight.EntryFound
        $ownerAction = [bool]$preflight.OwnerActionRequired
        $restoreSourceMode = [string]$preflight.RestoreSourceMode
        $expected = [string]$preflight.NativeAssetCopyExpected
        $canPromote = [bool]$preflight.CanPromotePackageConsumerRuntimeProof
        $blockedReason = [string]$preflight.BlockedReason
      }
      else {
        $hasEntry = $false
        $ownerAction = $true
        $restoreSourceMode = "not-recorded"
        $expected = ""
        $canPromote = $false
        $blockedReason = "preflight-entry-not-recorded"
      }

      $lines.Add("| $($result.RuntimePackageKey) | $hasEntry | $ownerAction | $restoreSourceMode | $expected | $canPromote | $(ConvertTo-MarkdownCell -Value $blockedReason) |")
    }
    $lines.Add("")
  }

  $lines.Add("Real callback runtime evidence is not inferred from `SmokeResult=passed`. Future TensorRT callback smoke must emit `EvidenceKind=real-callback-runtime`, `RealCallbackRuntime=True`, `CallbackKind`, `TensorRtLine`, `CudaLine`, `RuntimePackageKey`, `OwnerId`, `InvocationCount`, `AllocationCount`, `ReleaseCount`, `FailureCount`, `InFlightCallbackCount`, `LastStatus`, `LastDiagnostic`, and `FullPackageConsumerReport` before readiness can set `isRealCallbackRuntimeProof=true`.")
  $lines.Add('`RealCallbackRuntimeEvidence.Status` is `not-present`, `blocked-by-cuda-driver`, `blocked-by-application-control`, `blocked`, `incomplete`, or `ready`; only `ready` with `IsRealCallbackRuntimeProof=True` can be promoted by readiness.')
  $lines.Add("")
  $lines.Add("| Runtime key | Callback runtime status | Evidence kind | Missing markers | Diagnostic |")
  $lines.Add("| --- | --- | --- | --- | --- |")
  foreach ($result in $Results) {
    if ($result.PSObject.Properties.Name -notcontains "RealCallbackRuntimeEvidence") {
      continue
    }

    $evidence = $result.RealCallbackRuntimeEvidence
    $missingMarkers = if (@($evidence.MissingSmokeMarkers).Count -eq 0) { "none" } else { @($evidence.MissingSmokeMarkers) -join ", " }
    $lines.Add("| $($result.RuntimePackageKey) | $($evidence.Status) | $($evidence.EvidenceKind) | $(ConvertTo-MarkdownCell -Value $missingMarkers) | $(ConvertTo-MarkdownCell -Value ([string]$evidence.Diagnostic)) |")
  }
  $lines.Add("")
  $lines.Add('Generated by `eng/Test-PackageConsumer.ps1`.')
  Set-Content -LiteralPath $markdownPath -Value $lines -Encoding utf8

  Write-Host "Package consumer summary written to $jsonPath"
  Write-Host "Package consumer summary written to $markdownPath"
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
      # Long runtime package ids can push the local NuGet cache path near the
      # legacy MAX_PATH boundary on Windows PowerShell. Fall back to an extended
      # path delete after releasing handles from the previous restore/build.
      [System.GC]::Collect()
      [System.GC]::WaitForPendingFinalizers()

      if ($attempt -eq 1) {
        try {
          & dotnet build-server shutdown *> $null
        }
        catch {
          # Build server shutdown is best-effort only; deletion retries below are authoritative.
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

function Invoke-PackageConsumerValidation {
  param(
    [string]$Key,
    [object]$RuntimePackage,
    [object]$ManagedPackage,
    [object]$RuntimeNupkg,
    [bool]$ShouldRunSmoke
  )

  $timer = [System.Diagnostics.Stopwatch]::StartNew()
  $consumerRoot = Join-Path $OutputRoot $Key
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

  $nugetConfig = New-NuGetConfigContent -ManagedSource $ManagedPackageDirectory -RuntimeSource $RuntimePackageDirectory

  $runtimePackageReferences = New-RuntimePackageReferenceItems -RuntimeNupkg $RuntimeNupkg
  $project = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$TargetFramework</TargetFramework>
    <RuntimeIdentifier>$($RuntimePackage.rid)</RuntimeIdentifier>
    <RestorePackagesPath>`$(MSBuildProjectDirectory)\.nuget\packages</RestorePackagesPath>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="$($ManagedPackage.Id)" Version="$($ManagedPackage.Version)" />
$($runtimePackageReferences -join "`r`n")
  </ItemGroup>
</Project>
"@

  $program = @"
using JYPPX.CudaSharp;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp;

static string GetStringArgument(string[] args, string name, string defaultValue)
{
    for (int index = 0; index < args.Length - 1; index++)
    {
        if (string.Equals(args[index], name, StringComparison.OrdinalIgnoreCase))
        {
            return args[index + 1];
        }
    }

    return defaultValue;
}

static bool HasSwitch(string[] args, string name)
{
    return args.Any(argument => string.Equals(argument, name, StringComparison.OrdinalIgnoreCase));
}

static string SanitizeSmokeValue(string value)
{
    return value.Replace(Environment.NewLine, " ").Replace(';', ',');
}

static string FormatDebugListenerRealNonNullAttachRuntimeSmoke(TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult result)
{
    return "debug-listener-real-non-null-attach-runtime-smoke" +
        $";EvidenceKind={result.EvidenceKind}" +
        $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
        $";RealCallbackRuntime={result.RealCallbackRuntime}" +
        $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
        $";CallbackKind={result.CallbackKind}" +
        $";TensorRtLine={result.TensorRtLine}" +
        $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
        $";Status={result.Status}" +
        $";OptInEnabled={result.OptInEnabled}" +
        $";FullPackageConsumerReport={result.FullPackageConsumerReport}" +
        $";AttachGuardReady={result.AttachGuardReady}" +
        $";NativeVTableReady={result.NativeVTableReady}" +
        $";BorrowedDebugTensorRuntimeReady={result.BorrowedDebugTensorRuntimeReady}" +
        $";CallbackInvocationReady={result.CallbackInvocationReady}" +
        $";AttachAttempted={result.AttachAttempted}" +
        $";AttachSucceeded={result.AttachSucceeded}" +
        $";DetachAttempted={result.DetachAttempted}" +
        $";DetachSucceeded={result.DetachSucceeded}" +
        $";RollbackAttempted={result.RollbackAttempted}" +
        $";RollbackSucceeded={result.RollbackSucceeded}" +
        $";NativeVTableInstalled={result.NativeVTableInstalled}" +
        $";ProcessDebugTensorInvoked={result.ProcessDebugTensorInvoked}" +
        $";InvocationCount={result.InvocationCount}" +
        $";AllocationCount={result.AllocationCount}" +
        $";ReleaseCount={result.ReleaseCount}" +
        $";FailureCount={result.FailureCount}" +
        $";InFlightCallbackCount={result.InFlightCallbackCount}" +
        $";LastStatus={result.LastStatus}" +
        $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
        $";ReportPointerFree={result.ReportPointerFree}" +
        $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
        $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
        $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
        $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
        $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
        $";ReasonRuntimeProofStillBlocked={SanitizeSmokeValue(result.ReasonRuntimeProofStillBlocked)}" +
        $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
        $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
}

static string FormatDebugListenerProcessDebugTensorCallbackTrampoline(TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult result)
{
    return "debug-listener-process-debug-tensor-callback-trampoline" +
        $";EvidenceKind={result.EvidenceKind}" +
        $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
        $";RealCallbackRuntime={result.RealCallbackRuntime}" +
        $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
        $";CallbackKind={result.CallbackKind}" +
        $";TensorRtLine={result.TensorRtLine}" +
        $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
        $";Status={result.Status}" +
        $";TrampolineShapeReady={result.TrampolineShapeReady}" +
        $";NativeCallbackEntryLocated={result.NativeCallbackEntryLocated}" +
        $";NoThrowCallbackEntryReady={result.NoThrowCallbackEntryReady}" +
        $";ExceptionCaptureReady={result.ExceptionCaptureReady}" +
        $";CallbackStatusMappingReady={result.CallbackStatusMappingReady}" +
        $";InFlightAccountingReady={result.InFlightAccountingReady}" +
        $";DetachBeforeReleaseReady={result.DetachBeforeReleaseReady}" +
        $";BorrowedDebugTensorMetadataCopyReady={result.BorrowedDebugTensorMetadataCopyReady}" +
        $";BorrowedDebugTensorPointerExposed={result.BorrowedDebugTensorPointerExposed}" +
        $";BorrowedDebugTensorDataPointerExposed={result.BorrowedDebugTensorDataPointerExposed}" +
        $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
        $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
        $";OptInEnabled={result.OptInEnabled}" +
        $";FullPackageConsumerReport={result.FullPackageConsumerReport}" +
        $";AttachAttempted={result.AttachAttempted}" +
        $";AttachSucceeded={result.AttachSucceeded}" +
        $";NativeVTableInstalled={result.NativeVTableInstalled}" +
        $";ProcessDebugTensorInvoked={result.ProcessDebugTensorInvoked}" +
        $";InvocationCount={result.InvocationCount}" +
        $";CallbackStubEntryCount={result.CallbackStubEntryCount}" +
        $";CallbackStubLeaveCount={result.CallbackStubLeaveCount}" +
        $";FailureCount={result.FailureCount}" +
        $";InFlightCallbackCount={result.InFlightCallbackCount}" +
        $";LastStatus={result.LastStatus}" +
        $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
        $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
        $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
        $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
        $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
        $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
        $";TensorName={SanitizeSmokeValue(result.Metadata.TensorName)}" +
        $";TensorNameLength={result.Metadata.TensorNameLength}" +
        $";DataType={result.Metadata.DataType}" +
        $";Location={result.Metadata.Location}" +
        $";TensorShapeRank={result.Metadata.TensorShapeRank}" +
        $";ShapeSummary={SanitizeSmokeValue(result.Metadata.ShapeSummary)}" +
        $";MetadataCopied={result.Metadata.MetadataCopied}" +
        $";ReasonRuntimeProofStillBlocked={SanitizeSmokeValue(result.ReasonRuntimeProofStillBlocked)}" +
        $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
        $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
}

static string FormatDebugListenerRealCallbackRuntimeProof(TensorRtDebugListenerRealCallbackRuntimeProofResult result)
{
    return "debug-listener-real-callback-runtime-proof" +
        $";EvidenceKind={result.EvidenceKind}" +
        $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
        $";RealCallbackRuntime={result.RealCallbackRuntime}" +
        $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
        $";CallbackKind={result.CallbackKind}" +
        $";TensorRtLine={result.TensorRtLine}" +
        $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
        $";Status={result.Status}" +
        $";OptInEnabled={result.OptInEnabled}" +
        $";FullPackageConsumerReport={result.FullPackageConsumerReport}" +
        $";RuntimeSmokeReady={result.RuntimeSmokeReady}" +
        $";TrampolineShapeReady={result.TrampolineShapeReady}" +
        $";AttachAttempted={result.AttachAttempted}" +
        $";AttachSucceeded={result.AttachSucceeded}" +
        $";DetachAttempted={result.DetachAttempted}" +
        $";DetachSucceeded={result.DetachSucceeded}" +
        $";RollbackAttempted={result.RollbackAttempted}" +
        $";RollbackSucceeded={result.RollbackSucceeded}" +
        $";NativeVTableInstalled={result.NativeVTableInstalled}" +
        $";ProcessDebugTensorInvoked={result.ProcessDebugTensorInvoked}" +
        $";InvocationCount={result.InvocationCount}" +
        $";FailureCount={result.FailureCount}" +
        $";InFlightCallbackCount={result.InFlightCallbackCount}" +
        $";BorrowedDebugTensorMetadataCopied={result.BorrowedDebugTensorMetadataCopied}" +
        $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
        $";ProcessDebugTensorRuntimeReady={result.ProcessDebugTensorRuntimeReady}" +
        $";AttemptedNoInvocation={result.AttemptedNoInvocation}" +
        $";LastStatus={result.LastStatus}" +
        $";LastDiagnostic={SanitizeSmokeValue(result.LastDiagnostic)}" +
        $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
        $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
        $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
        $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
        $";BlockedPrerequisiteCount={result.BlockedPrerequisiteCount}" +
        $";ReasonRuntimeProofStillBlocked={SanitizeSmokeValue(result.ReasonRuntimeProofStillBlocked)}" +
        $";BlockedPrerequisites={SanitizeSmokeValue(string.Join(",", result.BlockedPrerequisites))}" +
        $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
}

static string FormatDebugListenerCallbackProofGapReport(TensorRtDebugListenerCallbackProofGapReportResult result)
{
    return "debug-listener-callback-proof-gap-report" +
        $";EvidenceKind={result.EvidenceKind}" +
        $";RuntimeEvidenceKind={result.RuntimeEvidenceKind}" +
        $";RealCallbackRuntime={result.RealCallbackRuntime}" +
        $";IsRealCallbackRuntimeProof={result.IsRealCallbackRuntimeProof}" +
        $";CallbackKind={result.CallbackKind}" +
        $";TensorRtLine={result.TensorRtLine}" +
        $";RuntimePackageKey={SanitizeSmokeValue(result.RuntimePackageKey)}" +
        $";Status={result.Status}" +
        $";NonNullAttachStillDisabled={result.NonNullAttachStillDisabled}" +
        $";NativeAttachEntryReady={result.NativeAttachEntryReady}" +
        $";NativeVTableInstallBlocked={result.NativeVTableInstallBlocked}" +
        $";NoThrowCallbackEntryReady={result.NoThrowCallbackEntryReady}" +
        $";ExceptionStatusMappingReady={result.ExceptionStatusMappingReady}" +
        $";InFlightAccountingReady={result.InFlightAccountingReady}" +
        $";BorrowedDebugTensorMetadataCopied={result.BorrowedDebugTensorMetadataCopied}" +
        $";DetachRollbackReady={result.DetachRollbackReady}" +
        $";ProcessDebugTensorRuntimeInvoked={result.ProcessDebugTensorRuntimeInvoked}" +
        $";FullPackageConsumerRuntimeProofReady={result.FullPackageConsumerRuntimeProofReady}" +
        $";PointerFreeSurfaceReady={result.PointerFreeSurfaceReady}" +
        $";AttemptedNoInvocation={result.AttemptedNoInvocation}" +
        $";InvocationCount={result.InvocationCount}" +
        $";FailureCount={result.FailureCount}" +
        $";InFlightCallbackCount={result.InFlightCallbackCount}" +
        $";CanAttemptRuntimeProof={result.CanAttemptRuntimeProof}" +
        $";CanPromoteRealCallbackRuntime={result.CanPromoteRealCallbackRuntime}" +
        $";RuntimeProofBlocked={result.RuntimeProofBlocked}" +
        $";DeferredRowsStillRequired={result.DeferredRowsStillRequired}" +
        $";GapReasonCount={result.GapReasonCount}" +
        $";GapReasons={SanitizeSmokeValue(string.Join(",", result.GapReasons))}" +
        $";Diagnostic={SanitizeSmokeValue(result.Diagnostic)}";
}

Console.WriteLine("TensorRtAssemblyBridge=" + TensorRtSharpInfo.NativeBridgeLibraryName);
Console.WriteLine("CudaAssemblyBridge=" + CudaSharpInfo.NativeBridgeLibraryName);

var tensorRt = TensorRtEnvironmentProbe.GetCurrent();
Console.WriteLine("Bridge=" + tensorRt.BuildInfo.BridgeName + " TRT=" + tensorRt.BuildInfo.TensorRtVersion + " CUDA=" + tensorRt.BuildInfo.CudaToolkitVersion);

var cuda = CudaEnvironmentProbe.GetCurrent();
Console.WriteLine("CudaDevices=" + cuda.CudaRuntimeInfo.DeviceCount + " Vendor=" + cuda.CudaRuntimeInfo.VendorDependencyAvailable);

string runtimePackageKey = GetStringArgument(args, "--runtime-package-key", string.Empty);
bool enableDebugListenerRuntimeSmoke =
    HasSwitch(args, "--enable-debug-listener-runtime-smoke") ||
    string.Equals(Environment.GetEnvironmentVariable("JYPPX_ENABLE_DEBUG_LISTENER_RUNTIME_SMOKE"), "1", StringComparison.OrdinalIgnoreCase);
using TensorRtDebugListenerCallbackOwner debugListenerOwner = new TensorRtDebugListenerCallbackOwner();
TensorRtDebugListenerCallbackRequest debugListenerRequest = new TensorRtDebugListenerCallbackRequest(
    "package_consumer_debug_tensor",
    TensorRtDataType.Float,
    TensorRtTensorLocation.Device,
    new long[] { 1, 3, 16, 16 },
    "package-consumer-debug-listener-runtime-smoke",
    isInput: true,
    isExecutionTensor: true);
debugListenerOwner.RunDesignDiagnostic(TensorRtApiLine.TensorRt11, debugListenerRequest);
debugListenerOwner.Dispose();
TensorRtDebugListenerCallbackOwnerSnapshot debugListenerSnapshot = debugListenerOwner.GetSnapshot("post-dispose");
TensorRtDebugListenerRuntimeProofAttemptPreflightResult debugListenerAttemptPreflight =
    TensorRtDebugListenerRuntimeProofAttemptPreflight.Evaluate(debugListenerSnapshot);
TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult debugListenerRuntimeSmoke =
    TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.Evaluate(
        debugListenerAttemptPreflight,
        runtimePackageKey,
        enableDebugListenerRuntimeSmoke,
        fullPackageConsumerReport: true);
Console.WriteLine("DebugListenerRealNonNullAttachRuntimeSmoke=" + FormatDebugListenerRealNonNullAttachRuntimeSmoke(debugListenerRuntimeSmoke));
TensorRtDebugListenerNoThrowVTableCallbackStubResult debugListenerCallbackStub =
    TensorRtDebugListenerNoThrowVTableCallbackStub.Evaluate(debugListenerSnapshot);
TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult debugListenerMetadataGate =
    TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.Evaluate(debugListenerSnapshot);
TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult debugListenerCallbackTrampoline =
    TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.Evaluate(
        debugListenerCallbackStub,
        debugListenerMetadataGate,
        debugListenerRuntimeSmoke);
Console.WriteLine("DebugListenerProcessDebugTensorCallbackTrampoline=" + FormatDebugListenerProcessDebugTensorCallbackTrampoline(debugListenerCallbackTrampoline));
TensorRtDebugListenerRealCallbackRuntimeProofResult debugListenerRealCallbackRuntimeProof =
    TensorRtDebugListenerRealCallbackRuntimeProof.Evaluate(
        debugListenerRuntimeSmoke,
        debugListenerCallbackTrampoline);
Console.WriteLine("DebugListenerRealCallbackRuntimeProof=" + FormatDebugListenerRealCallbackRuntimeProof(debugListenerRealCallbackRuntimeProof));
TensorRtDebugListenerCallbackProofGapReportResult debugListenerCallbackProofGapReport =
    TensorRtDebugListenerCallbackProofGapReport.Evaluate(
        debugListenerAttemptPreflight,
        debugListenerRuntimeSmoke,
        debugListenerCallbackTrampoline,
        debugListenerRealCallbackRuntimeProof);
Console.WriteLine("DebugListenerCallbackProofGapReport=" + FormatDebugListenerCallbackProofGapReport(debugListenerCallbackProofGapReport));
"@

  Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "NuGet.config") -Value $nugetConfig -Encoding utf8
  $restorePackagesPath = Get-RestorePackagesPath -RuntimeKey $Key
  New-Item -ItemType Directory -Path $restorePackagesPath -Force | Out-Null

  $project = $project.Replace('$(MSBuildProjectDirectory)\.nuget\packages', $restorePackagesPath)
  Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj") -Value $project -Encoding utf8
  Set-Content -LiteralPath (Join-Path $resolvedConsumerRoot "Program.cs") -Value $program -Encoding utf8

  Invoke-CheckedDotNet -Arguments @("restore", (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj"), "--configfile", (Join-Path $resolvedConsumerRoot "NuGet.config"))
  Invoke-CheckedDotNet -Arguments @("build", (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj"), "-c", "Release", "--no-restore")

  $outputDirectory = Join-PathMany -Parts @($resolvedConsumerRoot, "bin", "Release", $TargetFramework, $RuntimePackage.rid)
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

  $expectedNativeFiles = @(Get-ExpectedNativeFileNames -RuntimePackage $RuntimePackage)
  $missingNativeFiles = New-Object System.Collections.Generic.List[string]
  $foundNativeFileCount = 0
  foreach ($fileName in $expectedNativeFiles) {
    $matches = @(Get-ChildItem -LiteralPath $outputDirectory -Recurse -Filter $fileName)
    if ($matches.Count -eq 0) {
      $missingNativeFiles.Add($fileName)
    }
    else {
      $foundNativeFileCount++
    }
  }

  if ($missingNativeFiles.Count -gt 0) {
    throw "Expected native runtime assets were not copied to the consumer output for ${Key}: $($missingNativeFiles -join ', ')"
  }

  Unblock-ConsumerRuntimeAssets -OutputDirectory $outputDirectory -FileNames $expectedNativeFiles
  $signedConsumerOutputCount = 0
  $consumerSigningThumbprint = $null
  $consumerSigningTrustedPublisher = "not-requested"
  $consumerSigningRoot = "not-requested"
  if ($SignConsumerOutput.IsPresent) {
    $consumerSigningResult = Sign-ConsumerOutput -OutputDirectory $outputDirectory
    $signedConsumerOutputCount = $consumerSigningResult.count
    $consumerSigningThumbprint = $consumerSigningResult.certificateThumbprint
    $consumerSigningTrustedPublisher = $consumerSigningResult.currentUserTrustedPublisher
    $consumerSigningRoot = $consumerSigningResult.currentUserRoot
  }
  Unblock-ConsumerRuntimeAssets -OutputDirectory $outputDirectory -FileNames @(
    $expectedNativeFiles +
    $expectedManagedAssemblies +
    @("PackageConsumerSmoke.dll", "PackageConsumerSmoke.exe")
  )

  $smokeRequested = [bool]$ShouldRunSmoke
  $smokeResult = "not-requested"
  $smokeExitCode = $null
  $smokeCommand = ""
  $smokeDiagnostic = "smoke was not requested."
  $smokeOutputLines = @()
  if ($ShouldRunSmoke) {
    $smokeArguments = @("run", "--project", (Join-Path $resolvedConsumerRoot "PackageConsumerSmoke.csproj"), "-c", "Release", "--no-build", "--", "--runtime-package-key", $Key)
    $smokeCommand = "dotnet $($smokeArguments -join ' ')"
    $smokeRun = Invoke-DotNetCommand -Arguments $smokeArguments
    $smokeExitCode = [int]$smokeRun.ExitCode
    $smokeOutputLines = @($smokeRun.OutputLines | ForEach-Object { [string]$_ })
    if ($smokeRun.ExitCode -eq 0) {
      $smokeResult = "passed"
      $smokeDiagnostic = "smoke completed successfully."
    }
    elseif (Test-ApplicationControlPolicyBlock -OutputLines $smokeRun.OutputLines) {
      $smokeResult = "blocked-by-application-control"
      $smokeDiagnostic = "smoke execution was blocked by the Windows application control policy."
      Write-Warning "Package consumer smoke was blocked by the Windows application control policy on this runner (0x800711C7). Restore/build/native asset validation passed, so packaging will continue."
    }
    elseif (Test-CudaDriverRuntimeCompatibilityBlock -OutputLines $smokeRun.OutputLines) {
      $smokeResult = "blocked-by-cuda-driver"
      $smokeDiagnostic = "smoke execution reached the packaged runtime, but CUDA driver/runtime compatibility blocked execution; cudaRuntimeGetVersion reported CUDA error 35."
      Write-Warning "Package consumer smoke reached the packaged runtime but was blocked by CUDA driver/runtime compatibility (CUDA error 35). Restore/build/native asset validation passed, so packaging will continue."
    }
    else {
      $smokeResult = "failed"
      $smokeDiagnostic = "$smokeCommand failed with exit code $($smokeRun.ExitCode)."
      if (-not $AllowSmokeFailure.IsPresent) {
        throw "$smokeDiagnostic`n$($smokeOutputLines -join [Environment]::NewLine)"
      }

      Write-Warning "Package consumer smoke failed but -AllowSmokeFailure was provided. Restore/build/native asset validation passed, so a diagnostic report will be written."
    }
  }

  $timer.Stop()
  $elapsedSeconds = [Math]::Round($timer.Elapsed.TotalSeconds, 2)
  $realCallbackRuntimeEvidence = New-RealCallbackRuntimeEvidenceFromSmoke `
    -SmokeRequested $smokeRequested `
    -SmokeResult $smokeResult `
    -SmokeExitCode $smokeExitCode `
    -SmokeDiagnostic $smokeDiagnostic `
    -SmokeOutputLines $smokeOutputLines

  if (-not $smokeRequested) {
    $runtimeSmokeClassification = "not-requested"
    $packageConsumerEvidenceKind = "package-consumer-native-copy"
  }
  elseif ($smokeResult -eq "passed") {
    $runtimeSmokeClassification = "runtime-smoke-passed"
    $packageConsumerEvidenceKind = "full-runtime-package-consumer-smoke"
  }
  elseif ($smokeResult -eq "blocked-by-cuda-driver") {
    $runtimeSmokeClassification = "runtime-smoke-driver-blocked"
    $packageConsumerEvidenceKind = "full-runtime-package-consumer-smoke-driver-blocked"
  }
  elseif ($smokeResult -eq "blocked-by-application-control") {
    $runtimeSmokeClassification = "runtime-smoke-application-control-blocked"
    $packageConsumerEvidenceKind = "full-runtime-package-consumer-smoke-application-control-blocked"
  }
  else {
    $runtimeSmokeClassification = "runtime-smoke-failed"
    $packageConsumerEvidenceKind = "full-runtime-package-consumer-smoke-failed"
  }

  $isRuntimeExecutionEvidence = $smokeRequested -and $smokeResult -eq "passed"
  $isDependencyProbeOnly = -not $smokeRequested -or $smokeResult -ne "passed"
  $isRealCallbackRuntimeProof = [bool]$realCallbackRuntimeEvidence.IsRealCallbackRuntimeProof
  $isPackageConsumerRuntimeProof = $false
  $readonlySummaryEvidenceKind = "readonly-summary-diagnostics-not-runtime-proof"
  $wrapperSurfaceEvidenceKind = "package-consumer-wrapper-surface-diagnostics"
  $runtimeProofPreflightEntry = Find-PackageConsumerRuntimeProofPreflightEntry -Matrix $packageConsumerRuntimeProofPreflightMatrix -RuntimePackageKey $Key
  $runtimeProofPreflight = if ($null -eq $runtimeProofPreflightEntry) {
    [pscustomobject]@{
      MatrixSchemaVersion = if ($null -eq $packageConsumerRuntimeProofPreflightMatrix) { "not-found" } else { [string]$packageConsumerRuntimeProofPreflightMatrix.schemaVersion }
      EntryFound = $false
      OwnerActionRequired = $true
      RestoreSourceMode = "not-recorded"
      UsesProjectReference = $false
      NativeAssetCopyExpected = $expectedNativeFiles.Count
      NativeAssetCopyActual = $null
      RuntimeSmokeRequired = $true
      CanPromotePackageConsumerRuntimeProof = $false
      BlockedReason = "preflight-entry-missing-or-matrix-not-found"
      ValidatorCommand = ""
    }
  }
  else {
    [pscustomobject]@{
      MatrixSchemaVersion = [string]$packageConsumerRuntimeProofPreflightMatrix.schemaVersion
      EntryFound = $true
      OwnerActionRequired = [bool]$runtimeProofPreflightEntry.ownerActionRequired
      RestoreSourceMode = [string]$runtimeProofPreflightEntry.restoreSourceMode
      UsesProjectReference = [bool]$runtimeProofPreflightEntry.usesProjectReference
      NativeAssetCopyExpected = [int]$runtimeProofPreflightEntry.nativeAssetCopyExpected
      NativeAssetCopyActual = $runtimeProofPreflightEntry.nativeAssetCopyActual
      RuntimeSmokeRequired = [bool]$runtimeProofPreflightEntry.runtimeSmokeRequired
      CanPromotePackageConsumerRuntimeProof = [bool]$runtimeProofPreflightEntry.canPromotePackageConsumerRuntimeProof
      BlockedReason = [string]$runtimeProofPreflightEntry.blockedReason
      ValidatorCommand = [string]$runtimeProofPreflightEntry.validatorCommand
    }
  }
  $forbiddenProofSubstitutes = @(
    "readonly summary",
    "readonly diagnostics",
    "TensorRtExec report",
    "OnnxToEngine report",
    "YoloVision matrix",
    "bridge-only",
    "dependency probe",
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "build-only",
    "dry-run",
    "template",
    "blocked-by-cuda-driver",
    "SmokeResult=passed without strict validator"
  )
  $requiredRuntimeProofFields = @(
    "cleanConsumerRoot",
    "noProjectReference",
    "packageRestoreSource",
    "managedPackageSha256",
    "runtimePackageSha256",
    "nativeAssetListingSha256",
    "runtimeSmokeLogPath",
    "runtimeSmokeLogSha256",
    "stdoutSummary",
    "stderrSummary",
    "hostMetadata",
    "strictValidatorPassed"
  )

  Write-Host "Package consumer validation passed for $Key."
  Write-Host "  Managed package: $($ManagedPackage.Id) $($ManagedPackage.Version)"
  Write-Host "  Runtime package: $($RuntimeNupkg.Id) $($RuntimeNupkg.Version)"
  Write-Host "  Native assets: $foundNativeFileCount/$($expectedNativeFiles.Count)"
  Write-Host "  Smoke: $smokeResult"
  Write-Host "  Smoke diagnostic: $smokeDiagnostic"
  Write-Host "  Real callback runtime evidence: $($realCallbackRuntimeEvidence.Status) proof=$($realCallbackRuntimeEvidence.IsRealCallbackRuntimeProof)"
  Write-Host "  Signed consumer output: $signedConsumerOutputCount"
  Write-Host "  Consumer signing certificate: $consumerSigningThumbprint"
  Write-Host "  Consumer signing trust: publisher=$consumerSigningTrustedPublisher root=$consumerSigningRoot"
  Write-Host "  Elapsed: ${elapsedSeconds}s"
  Write-Host "  Consumer output: $outputDirectory"

  return [pscustomobject]@{
    RuntimePackageKey = $Key
    RuntimePackageId = $RuntimeNupkg.Id
    RuntimePackageVersion = $RuntimeNupkg.Version
    RuntimePackagePath = $RuntimeNupkg.Path
    ManagedPackageId = $ManagedPackage.Id
    ManagedPackageVersion = $ManagedPackage.Version
    TargetFramework = $TargetFramework
    RuntimeIdentifier = $RuntimePackage.rid
    DistributionTier = $RuntimePackage.distributionTier
    ValidationState = $RuntimePackage.validationState
    ConsumerBuildConfiguration = "Release"
    RestoreSucceeded = $true
    BuildSucceeded = $true
    PackageConsumerValidationSucceeded = $true
    NativeAssetsExpected = $expectedNativeFiles.Count
    NativeAssetsFound = $foundNativeFileCount
    MissingNativeAssets = @($missingNativeFiles.ToArray())
    SmokeRequested = $smokeRequested
    SmokeResult = $smokeResult
    SmokeExitCode = $smokeExitCode
    SmokeCommand = $smokeCommand
    SmokeDiagnostic = $smokeDiagnostic
    SmokeOutputLines = @($smokeOutputLines)
    SmokeFailureAllowed = $AllowSmokeFailure.IsPresent
    EvidenceKind = $packageConsumerEvidenceKind
    IsRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
    IsPackageConsumerRuntimeProof = $isPackageConsumerRuntimeProof
    CanPromoteRuntimeProof = $false
    CanPublishPublicly = $false
    CanCloseReleaseIssue = $false
    RuntimeSmokeClassification = $runtimeSmokeClassification
    IsDependencyProbeOnly = $isDependencyProbeOnly
    IsRealCallbackRuntimeProof = $isRealCallbackRuntimeProof
    ReadonlySummaryEvidenceKind = $readonlySummaryEvidenceKind
    WrapperSurfaceEvidenceKind = $wrapperSurfaceEvidenceKind
    RuntimeProofPreflight = $runtimeProofPreflight
    ForbiddenProofSubstitutes = @($forbiddenProofSubstitutes)
    RequiredRuntimeProofFields = @($requiredRuntimeProofFields)
    RealCallbackRuntimeEvidence = $realCallbackRuntimeEvidence
    ConsumerOutputSigned = $SignConsumerOutput.IsPresent
    ConsumerOutputSignedFileCount = $signedConsumerOutputCount
    ConsumerOutputSigningCertificateThumbprint = $consumerSigningThumbprint
    ConsumerOutputSigningTrustedPublisher = $consumerSigningTrustedPublisher
    ConsumerOutputSigningRoot = $consumerSigningRoot
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

$keys = @(Expand-KeyList -Values $RuntimePackageKey)
if ($keys.Count -eq 0) {
  throw "At least one runtime package key is required."
}

$smokeKeys = @(Expand-KeyList -Values $SmokeRuntimePackageKey)
$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$packageConsumerRuntimeProofPreflightMatrix = Get-PackageConsumerRuntimeProofPreflightMatrix
$managedPackage = Find-Package -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
Assert-ManagedPackageFreshness -ManagedPackage $managedPackage
$results = New-Object System.Collections.Generic.List[object]

foreach ($key in $keys) {
  $runtimePackage = $manifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $runtimePackage) {
    throw "Runtime package key '$key' was not found."
  }

  $runtimeNupkg = Find-Package -Directory $RuntimePackageDirectory -PackageId $runtimePackage.packageId
  $shouldRunSmoke = $RunSmoke.IsPresent -and ($smokeKeys.Count -eq 0 -or $smokeKeys -contains $key)
  $results.Add((Invoke-PackageConsumerValidation -Key $key -RuntimePackage $runtimePackage -ManagedPackage $managedPackage -RuntimeNupkg $runtimeNupkg -ShouldRunSmoke $shouldRunSmoke))
}

Write-ValidationReports -Results @($results.ToArray()) -Directory $ReportDirectory -PreflightMatrix $packageConsumerRuntimeProofPreflightMatrix
