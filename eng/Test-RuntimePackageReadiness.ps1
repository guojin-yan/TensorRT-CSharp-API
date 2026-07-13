[CmdletBinding()]
param(
  [string[]]$RuntimePackageKey = @("win-x64-trt11.0-cuda13.2-cudnn9.22"),
  [string]$ManagedPackageDirectory,
  [string]$RuntimePackageDirectory,
  [string]$SplitPackageRoot,
  [string]$PackageConsumerReportDirectory,
  [string]$SplitCollectionConsumerReportDirectory,
  [string]$ReportDirectory,
  [switch]$FailOnBlocked,
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

function Get-SafePathName {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  return ($Value -replace '[^A-Za-z0-9._-]', '-')
}

function Get-NupkgMetadata {
  param(
    [Parameter(Mandatory = $true)]
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
      SizeBytes = (Get-Item -LiteralPath $Path).Length
      LastWriteTime = (Get-Item -LiteralPath $Path).LastWriteTime
    }
  }
  finally {
    $zip.Dispose()
  }
}

function Find-LocalPackage {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Directory,
    [Parameter(Mandatory = $true)]
    [string]$PackageId
  )

  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
    return $null
  }

  $matches = New-Object System.Collections.Generic.List[object]
  foreach ($package in @(Get-ChildItem -LiteralPath $Directory -Filter *.nupkg -ErrorAction SilentlyContinue)) {
    try {
      $metadata = Get-NupkgMetadata -Path $package.FullName
      if ($metadata.Id -eq $PackageId) {
        $matches.Add($metadata)
      }
    }
    catch {
      Write-Warning "Unable to read nupkg metadata '$($package.FullName)': $($_.Exception.Message)"
    }
  }

  if ($matches.Count -eq 0) {
    return $null
  }

  return @($matches | Sort-Object LastWriteTime -Descending)[0]
}

function Get-RelativeAssetDisplayPath {
  param(
    [string]$BaseRoot,
    [string]$RelativePath
  )

  if ([string]::IsNullOrWhiteSpace($BaseRoot)) {
    return $RelativePath
  }

  return Join-Path $BaseRoot ($RelativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar)
}

function Test-RelativeAsset {
  param(
    [string]$BaseRoot,
    [string]$RelativePath,
    [string]$Kind
  )

  $displayPath = Get-RelativeAssetDisplayPath -BaseRoot $BaseRoot -RelativePath $RelativePath
  if ([string]::IsNullOrWhiteSpace($BaseRoot) -or -not (Test-Path -LiteralPath $BaseRoot -PathType Container)) {
    return [pscustomobject]@{
      kind = $Kind
      relativePath = $RelativePath
      path = $displayPath
      status = "missing-root"
      matchCount = 0
      matches = @()
    }
  }

  $normalized = $RelativePath -replace '/', [System.IO.Path]::DirectorySeparatorChar
  $path = Join-Path $BaseRoot $normalized
  if ($normalized.IndexOfAny(@('*', '?')) -ge 0) {
    $matches = @(Resolve-Path -Path $path -ErrorAction SilentlyContinue | ForEach-Object { $_.Path })
    return [pscustomobject]@{
      kind = $Kind
      relativePath = $RelativePath
      path = $path
      status = if ($matches.Count -gt 0) { "present" } else { "missing-asset" }
      matchCount = $matches.Count
      matches = @($matches)
    }
  }

  $exists = Test-Path -LiteralPath $path -PathType Leaf
  return [pscustomobject]@{
    kind = $Kind
    relativePath = $RelativePath
    path = $path
    status = if ($exists) { "present" } else { "missing-asset" }
    matchCount = if ($exists) { 1 } else { 0 }
    matches = if ($exists) { @($path) } else { @() }
  }
}

function Resolve-RuntimeRootsForKey {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Key
  )

  try {
    return & (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $Key -RepositoryRoot $RepositoryRoot | ConvertFrom-Json
  }
  catch {
    return [pscustomobject]@{
      runtimeKey = $Key
      platform = ""
      tensorRtRoot = $null
      cudaRoot = $null
      cudnnRoot = $null
      error = $_.Exception.Message
    }
  }
}

function New-WrapperSurfaceCapabilityEvidence {
  param(
    [string]$Surface
  )

  $surfaceText = if ([string]::IsNullOrWhiteSpace($Surface)) { "" } else { [string]$Surface }
  $tokens = New-Object System.Collections.Generic.List[string]
  $tokenSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

  foreach ($part in ($surfaceText -split ";")) {
    $trimmed = $part.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
      continue
    }

    $tokens.Add($trimmed)
    [void]$tokenSet.Add($trimmed)
  }

  function Test-WrapperSurfaceMarker {
    param(
      [string[]]$Markers
    )

    foreach ($marker in @($Markers)) {
      if ([string]::IsNullOrWhiteSpace($marker)) {
        continue
      }

      if ($tokenSet.Contains($marker)) {
        return $true
      }

      if ($surfaceText.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        return $true
      }
    }

    return $false
  }

  function Test-WrapperSurfaceRequirements {
    param(
      [string[]]$RequiredMarkers
    )

    foreach ($marker in @($RequiredMarkers)) {
      if (-not (Test-WrapperSurfaceMarker -Markers @($marker))) {
        return $false
      }
    }

    return $true
  }

  $definitions = @(
    [pscustomobject]@{
      name = "plugin-inventory"
      categoryMarkers = @("plugin-inventory", "compiled:plugin-inventory")
      requiredMarkers = @("TensorRtPluginRegistryInventory", "FindCreator", "TryFindCreator", "GetPluginRegistryInventory", "IsPluginCreatorRegistered", "TryIsPluginCreatorRegistered")
    },
    [pscustomobject]@{
      name = "plugin-inventory-field-metadata"
      categoryMarkers = @("plugin-inventory-field-metadata")
      requiredMarkers = @(
        "TensorRtPluginRegistryInventory.GetFieldSummaries",
        "TensorRtPluginFieldSummary",
        "TensorRtPluginFieldSummary.FieldName",
        "TensorRtPluginFieldSummary.FieldType",
        "TensorRtPluginFieldSummary.HasData",
        "TensorRtPluginRegistryInventoryDiagnostics.EmptyFieldNameCount",
        "TensorRtPluginRegistryInventoryDiagnostics.NegativeFieldLengthCount"
      )
    },
    [pscustomobject]@{
      name = "onnx-parser-diagnostic-readiness"
      categoryMarkers = @("onnx-parser-diagnostic-snapshot", "onnx-parser-diagnostic-summary")
      requiredMarkers = @(
        "TensorRtOnnxParserDiagnosticSnapshot",
        "TensorRtOnnxParserDiagnosticSnapshot.ToSummary",
        "TensorRtOnnxParserDiagnosticSummary",
        "TensorRtOnnxParserDiagnosticSnapshot.ErrorCount",
        "TensorRtOnnxParserDiagnosticSnapshot.Diagnostics",
        "TensorRtOnnxParserDiagnosticSnapshot.DiagnosticSummary",
        "TensorRtOnnxParserDiagnosticSnapshot.UsedVCPluginLibraries",
        "TensorRtOnnxParserDiagnosticSnapshot.IdentityOperatorSupported",
        "TensorRtOnnxParserDiagnosticSummary.CopiedDiagnosticCount",
        "TensorRtOnnxParserDiagnosticSummary.UsedVCPluginLibraryCount",
        "TensorRtOnnxParserDiagnosticSummary.DiagnosticSummaryLength",
        "TensorRtOnnxParserDiagnosticSummary.RuntimeEvidenceKind",
        "TensorRtOnnxParserDiagnosticSummary.IsRuntimeExecutionProof",
        "TensorRtOnnxParserDiagnosticSummary.CanPromoteReleaseProof",
        "TensorRtOnnxParser.GetDiagnosticSnapshot"
      )
    },
    [pscustomobject]@{
      name = "onnx-parser-refitter-diagnostic-readiness"
      categoryMarkers = @("onnx-parser-refitter-diagnostic-snapshot", "onnx-parser-refitter-diagnostic-summary")
      requiredMarkers = @(
        "TensorRtOnnxParserRefitterDiagnosticSnapshot",
        "TensorRtOnnxParserRefitterDiagnosticSnapshot.ToSummary",
        "TensorRtOnnxParserRefitterDiagnosticSummary",
        "TensorRtOnnxParserRefitterDiagnosticSnapshot.ErrorCount",
        "TensorRtOnnxParserRefitterDiagnosticSnapshot.Diagnostics",
        "TensorRtOnnxParserRefitterDiagnosticSnapshot.DiagnosticSummary",
        "TensorRtOnnxParserRefitterDiagnosticSummary.CopiedDiagnosticCount",
        "TensorRtOnnxParserRefitterDiagnosticSummary.DiagnosticSummaryLength",
        "TensorRtOnnxParserRefitterDiagnosticSummary.RuntimeEvidenceKind",
        "TensorRtOnnxParserRefitterDiagnosticSummary.IsRuntimeExecutionProof",
        "TensorRtOnnxParserRefitterDiagnosticSummary.CanPromoteReleaseProof",
        "TensorRtOnnxParserRefitter.GetDiagnosticSnapshot"
      )
    },
    [pscustomobject]@{
      name = "engine-rnn-readonly-diagnostics"
      categoryMarkers = @("engine-rnn-readonly-diagnostics")
      requiredMarkers = @(
        "HasImplicitBatchDimensionCompatibility",
        "SerializedPluginPathCountCompatibility",
        "GetRnnV2LayerCount",
        "GetRnnV2HiddenSize",
        "GetRnnV2DataLength",
        "GetRnnV2MaxSequenceLength",
        "GetRnnV2Operation",
        "GetRnnV2Direction",
        "GetRnnV2InputMode",
        "GetRnnV2CellState",
        "GetRnnV2HiddenState",
        "GetRnnV2SequenceLengths",
        "GetRnnV2WeightsForGate",
        "GetRnnV2BiasForGate",
        "TensorRtRnnV2GateWeightsSnapshot",
        "TensorRtRnnGateType",
        "TensorRtRnnOperation",
        "TensorRtRnnDirection",
        "TensorRtRnnInputMode"
      )
    },
    [pscustomobject]@{
      name = "rnnv2-borrowed-state-design-gate"
      categoryMarkers = @("rnnv2-borrowed-state-design-gate")
      requiredMarkers = @(
        "TensorRtRnnV2BorrowedStateDesignGate",
        "TensorRtRnnV2BorrowedStateDesignGateResult",
        "DataLengthScalarPromoted",
        "NetworkOwnedTensorReferencePolicyReady",
        "GateWeightSnapshotCopyReady",
        "OwnerLifetimeKnown",
        "BorrowedSnapshotPromotionReady",
        "DeferredBorrowedRowsStillRequired",
        "RemainingDeferredTriageRowCount",
        "CanPromoteRuntimeProof"
      )
    },
    [pscustomobject]@{
      name = "managed-callbacks"
      categoryMarkers = @("managed-callbacks")
      requiredMarkers = @("TensorRtLogger", "TensorRtProfiler", "TensorRtProgressMonitor")
    },
    [pscustomobject]@{
      name = "callback-diagnostics"
      categoryMarkers = @("callback-diagnostics")
      requiredMarkers = @("EmitDiagnostic", "CallbackFailureCount", "LastCallbackException", "TryGetInterfaceInfo")
    },
    [pscustomobject]@{
      name = "callback-api-language-safe-controls"
      categoryMarkers = @("callback-api-language-safe-controls")
      requiredMarkers = @(
        "TensorRtApiLanguage",
        "TensorRtLogger.ApiLanguage",
        "TensorRtLogger.TryGetApiLanguage",
        "TensorRtProfiler.ApiLanguage",
        "TensorRtProfiler.TryGetApiLanguage",
        "TensorRtProgressMonitor.ApiLanguage",
        "TensorRtProgressMonitor.TryGetApiLanguage"
      )
    },
    [pscustomobject]@{
      name = "error-recorder-snapshot"
      categoryMarkers = @("error-recorder-snapshot", "error-recorder-snapshots")
      requiredMarkers = @("TensorRtErrorRecorderSnapshot", "TensorRtErrorRecord", "TryGetErrorRecorderSnapshot", "HasErrorRecorder", "ClearErrorRecorder")
    },
    [pscustomobject]@{
      name = "error-recorder-diagnostics-design-gate"
      categoryMarkers = @("error-recorder-diagnostics-design-gate")
      requiredMarkers = @("TensorRtErrorRecorderDiagnosticsDesignGate", "TensorRtErrorRecorderDiagnosticsDesignGateResult", "RuntimeEvidenceKind", "CopiedDiagnosticsReady", "PointerFreeSurfaceReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "dimension-expression-snapshot-design-gate"
      categoryMarkers = @("dimension-expression-snapshot-design-gate")
      requiredMarkers = @("TensorRtDimensionExpressionSnapshotDesignGate", "TensorRtDimensionExpressionSnapshotDesignGateResult", "RuntimeEvidenceKind", "SnapshotTypeReady", "OwnerLifetimeKnown", "ExpressionPointerExposed", "ExprBuilderCreationEnabled", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "calibrator-metadata-design-gate"
      categoryMarkers = @("calibrator-metadata-design-gate")
      requiredMarkers = @("TensorRtCalibratorMetadataDesignGate", "TensorRtCalibratorMetadataDesignGateResult", "RuntimeEvidenceKind", "PresenceProbeAvailable", "CalibratorPointerExposed", "CallbackInvocationEnabled", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "runtime-deserialization-boundary-precheck"
      categoryMarkers = @("runtime-deserialization-boundary-precheck")
      requiredMarkers = @("TensorRtRuntimeDeserializationBoundaryPrecheck", "TensorRtRuntimeDeserializationBoundaryPrecheckResult", "RuntimeEvidenceKind", "ManagedByteArrayDeserializeReady", "ManagedStreamDeserializeReady", "HostMemoryDeserializeReady", "SerializedBufferCopiedBeforeInterop", "EngineHandleOwnedByWrapper", "LoadRuntimeDeferred", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "logger-presence-safe-controls"
      categoryMarkers = @("logger-presence-safe-controls")
      requiredMarkers = @("TensorRtBuilder.HasLogger", "TensorRtRuntime.HasLogger")
    },
    [pscustomobject]@{
      name = "allocator-debug-listener-safe-controls"
      categoryMarkers = @("allocator-debug-listener-safe-controls")
      requiredMarkers = @("ClearGpuAllocator", "HasOutputAllocator", "ClearOutputAllocator", "HasTemporaryStorageAllocator", "ClearTemporaryStorageAllocator", "HasDebugListener", "ClearDebugListener")
    },
    [pscustomobject]@{
      name = "callback-interface-info-safe-controls"
      categoryMarkers = @("callback-interface-info-safe-controls")
      requiredMarkers = @("TryGetOutputAllocatorInterfaceInfo", "TryGetTemporaryStorageAllocatorInterfaceInfo", "TryGetDebugListenerInterfaceInfo")
    },
    [pscustomobject]@{
      name = "execution-context-callback-state-snapshot"
      categoryMarkers = @("execution-context-callback-state-snapshot")
      requiredMarkers = @(
        "TensorRtExecutionContext.GetCallbackStateSnapshot",
        "TensorRtExecutionContext.ClearCallbackState",
        "TensorRtExecutionContextCallbackStateSnapshot",
        "HasOutputAllocator",
        "HasTemporaryStorageAllocator",
        "HasDebugListener",
        "OutputAllocatorInterfaceInfoAvailable",
        "TemporaryStorageAllocatorInterfaceInfoAvailable",
        "DebugListenerInterfaceInfoAvailable",
        "OutputAllocatorClearSupported",
        "TemporaryStorageAllocatorClearSupported",
        "DebugListenerClearSupported",
        "OutputAllocatorCleared",
        "TemporaryStorageAllocatorCleared",
        "DebugListenerCleared",
        "LastStatus",
        "LastOperation",
        "Diagnostic"
      )
    },
    [pscustomobject]@{
      name = "execution-context-callback-allocator-safe-control-summary"
      categoryMarkers = @("execution-context-callback-allocator-safe-control-summary")
      requiredMarkers = @(
        "TensorRtExecutionContext.GetCallbackAllocatorSafeControlSummary",
        "TensorRtExecutionContextCallbackAllocatorSafeControlSummary",
        "EvidenceKind",
        "RuntimeEvidenceKind",
        "RealCallbackRuntime",
        "IsRealCallbackRuntimeProof",
        "CopiedInterfaceInfoCount",
        "DiagnosticCount",
        "PointerFreeSurfaceReady",
        "CallbackInvocationAttempted",
        "IsRuntimeInvocationProofComplete"
      )
    },
    [pscustomobject]@{
      name = "allocator-owner-dry-run-diagnostics"
      categoryMarkers = @("allocator-owner-dry-run-diagnostics")
      requiredMarkers = @("TensorRtAllocatorCallbackOwner", "TensorRtAllocatorDryRunRequest", "TensorRtAllocatorDryRunResult", "RunDryRunDiagnostic", "CallbackInvocationCount", "CallbackFailureCount", "LastDiagnostic")
    },
    [pscustomobject]@{
      name = "allocator-owner-native-dry-run-controls"
      categoryMarkers = @("allocator-owner-native-dry-run-controls")
      requiredMarkers = @("RunNativeDryRunDiagnostic", "TensorRtAllocatorNativeDryRunResult", "LastStatus", "LastSize", "LastAlignment", "InvocationCount")
    },
    [pscustomobject]@{
      name = "allocator-owner-state-ledger-dry-run-controls"
      categoryMarkers = @("allocator-owner-state-ledger-dry-run-controls")
      requiredMarkers = @("RunNativeStateLedgerDryRunDiagnostic", "TensorRtAllocatorOwnerStateDryRunResult", "OwnerId", "LedgerAllocationCount", "LedgerReleaseCount", "LastOperation")
    },
    [pscustomobject]@{
      name = "allocator-owner-lifecycle-snapshot"
      categoryMarkers = @("allocator-owner-internal-runtime-prototype")
      requiredMarkers = @("TensorRtAllocatorCallbackOwnerSnapshot", "RunLifecycleDiagnostic", "GetSnapshot", "RuntimeEvidenceKind", "DevicePointerExposed", "BorrowedPointerEscaped", "PointerFreeSurfaceReady")
    },
    [pscustomobject]@{
      name = "allocator-owner-ledger-safety-gate"
      categoryMarkers = @("allocator-owner-ledger-safety-gate")
      requiredMarkers = @("TensorRtAllocatorLedgerSafetyGate", "TensorRtAllocatorLedgerSafetyGateResult", "Evaluate", "GetSnapshot", "ManagedKeepAliveReady", "NativeLedgerDesignReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "output-allocator-callback-owner-design"
      categoryMarkers = @("output-allocator-callback-owner-design")
      requiredMarkers = @("TensorRtOutputAllocatorCallbackOwner", "TensorRtOutputAllocatorCallbackRequest", "TensorRtOutputAllocatorCallbackOwnerSnapshot", "RunDesignDiagnostic", "NativeLedgerAvailable", "OutputBufferPointerExposed")
    },
    [pscustomobject]@{
      name = "output-allocator-attach-detach-design-gate"
      categoryMarkers = @("output-allocator-attach-detach-design-gate")
      requiredMarkers = @("TensorRtOutputAllocatorAttachDetachDesignGate", "TensorRtOutputAllocatorAttachDetachDesignGateResult", "Evaluate", "AttachControlAvailable", "DetachClearControlAvailable", "LineSpecificAttachDetachReady", "NativeVTableReady", "OutputBufferOwnershipRuntimeReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "output-buffer-ownership-safety-gate"
      categoryMarkers = @("output-buffer-ownership-safety-gate")
      requiredMarkers = @("TensorRtOutputBufferOwnershipSafetyGate", "TensorRtOutputBufferOwnershipSafetyGateResult", "Evaluate", "SafetyGateReady", "CurrentMemoryReusePolicyReady", "BorrowedPointerEscapeBlocked", "OwnedDevicePointerReleasePolicyReady", "ShapeNotificationOrderingReady", "ReallocateOutputRuntimeReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "output-allocator-runtime-proof-precheck"
      categoryMarkers = @("output-allocator-runtime-proof-precheck")
      requiredMarkers = @("TensorRtOutputAllocatorRuntimeProofPrecheck", "TensorRtOutputAllocatorRuntimeProofPrecheckResult", "Evaluate", "NativeLedgerDesignReady", "AttachDetachDesignGateReady", "AttachControlAvailable", "DetachClearControlAvailable", "NativeVTableReady", "DevicePointerLedgerRuntimeReady", "OutputBufferOwnershipSafetyGateReady", "OutputBufferOwnershipRuntimeReady", "CurrentMemoryReusePolicyReady", "ReallocateOutputRuntimeReady", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "callback-owner-closure-matrix"
      categoryMarkers = @("callback-owner-closure-matrix")
      requiredMarkers = @("TensorRtCallbackOwnerClosureMatrix", "TensorRtCallbackOwnerClosureMatrixResult", "TensorRtCallbackOwnerClosureMatrixRow", "Evaluate", "FamilyCount", "DesignGateReadyFamilyCount", "ClosureReadyFamilyCount", "RuntimeProofAttemptReadyFamilyCount", "PackageConsumerRuntimeProofReadyFamilyCount", "RuntimeProofBlocked", "DeferredRowsStillRequired")
    },
    [pscustomobject]@{
      name = "debug-listener-callback-owner-design"
      categoryMarkers = @("debug-listener-callback-owner-design")
      requiredMarkers = @("TensorRtDebugListenerCallbackOwner", "TensorRtDebugListenerCallbackRequest", "TensorRtDebugListenerCallbackOwnerSnapshot", "RunDesignDiagnostic", "ProcessDebugTensorCount", "DebugTensorPointerExposed")
    },
    [pscustomobject]@{
      name = "debug-listener-attach-detach-design-gate"
      categoryMarkers = @("debug-listener-attach-detach-design-gate")
      requiredMarkers = @("TensorRtDebugListenerAttachDetachDesignGate", "TensorRtDebugListenerAttachDetachDesignGateResult", "Evaluate", "AttachControlAvailable", "DetachClearControlAvailable", "LineSpecificAttachDetachReady", "NativeVTableReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-borrowed-tensor-safety-gate"
      categoryMarkers = @("debug-listener-borrowed-tensor-safety-gate")
      requiredMarkers = @("TensorRtDebugListenerBorrowedTensorSafetyGate", "TensorRtDebugListenerBorrowedTensorSafetyGateResult", "Evaluate", "SafetyGateReady", "BorrowedDebugTensorPointerEscapeBlocked", "BorrowedDebugTensorLifetimeReady", "BorrowedDebugTensorDataLifetimeReady", "ProcessDebugTensorRuntimeReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-attach-vtable-safety-gate"
      categoryMarkers = @("debug-listener-attach-vtable-safety-gate")
      requiredMarkers = @("TensorRtDebugListenerAttachVTableSafetyGate", "TensorRtDebugListenerAttachVTableSafetyGateResult", "Evaluate", "SafetyGateReady", "AttachControlAvailable", "StableNativeOwnerAddressReady", "NoThrowNativeVTableReady", "ExceptionToStatusMappingReady", "ProcessDebugTensorRuntimeReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-attach-nothrow-preflight"
      categoryMarkers = @("debug-listener-native-attach-nothrow-preflight")
      requiredMarkers = @("TensorRtDebugListenerNativeAttachNoThrowPreflight", "TensorRtDebugListenerNativeAttachNoThrowPreflightResult", "Evaluate", "PreflightReady", "NativeAttachEntryLocated", "NativeDetachEntryLocated", "NoThrowVTableDesignReady", "ExceptionToStatusMappingDesignReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-owner-address-design-gate"
      categoryMarkers = @("debug-listener-native-owner-address-design-gate")
      requiredMarkers = @("TensorRtDebugListenerNativeOwnerAddressDesignGate", "TensorRtDebugListenerNativeOwnerAddressDesignGateResult", "Evaluate", "DesignGateReady", "NativeAttachNoThrowPreflightReady", "StableNativeOwnerAddressDesignReady", "NativeOwnerNonCopyableReady", "NoThrowNativeDestructorReady", "NativeOwnerLifecycleReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-nothrow-vtable-design-gate"
      categoryMarkers = @("debug-listener-native-nothrow-vtable-design-gate")
      requiredMarkers = @("TensorRtDebugListenerNativeNoThrowVTableDesignGate", "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult", "Evaluate", "DesignGateReady", "NativeOwnerAddressDesignGateReady", "NoThrowVTableDesignReady", "ExceptionToStatusMappingDesignReady", "NativeVTableTrampolineReady", "CallbackExceptionCaptureReady", "CallbackStatusMappingReady", "CallbackInFlightAccountingReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-attach-entry-design-gate"
      categoryMarkers = @("debug-listener-native-attach-entry-design-gate")
      requiredMarkers = @("TensorRtDebugListenerNativeAttachEntryDesignGate", "TensorRtDebugListenerNativeAttachEntryDesignGateResult", "Evaluate", "DesignGateReady", "NativeNoThrowVTableDesignGateReady", "NativeAttachEntryLocated", "LineSpecificAttachEntryDesignReady", "AttachEntryNoThrowReady", "AttachEntryVersionGuardReady", "AttachEntryOwnershipReady", "DetachBeforeReleaseReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-detach-before-release-design-gate"
      categoryMarkers = @("debug-listener-native-detach-before-release-design-gate")
      requiredMarkers = @("TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate", "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult", "Evaluate", "DesignGateReady", "NativeAttachEntryDesignGateReady", "NativeDetachEntryLocated", "DetachBeforeReleaseReady", "ReleaseHookOrderingReady", "DisposeIdempotencyReady", "InFlightDrainBeforeReleaseReady", "CallbackStateUnpinAfterDetachReady", "DelegateUnpinAfterDetachReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-owner-lifecycle-dry-run"
      categoryMarkers = @("debug-listener-native-owner-lifecycle-dry-run")
      requiredMarkers = @("TensorRtDebugListenerNativeOwnerLifecycleDryRun", "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult", "Evaluate", "DryRunReady", "NativeDetachBeforeReleaseDesignGateReady", "StableNativeOwnerIdentityReady", "NativeOwnerNonCopyableReady", "ReleaseHookOrderingReady", "DisposeIdempotencyReady", "InFlightDrainBeforeReleaseReady", "CallbackStateUnpinAfterDetachReady", "DelegateUnpinAfterDetachReady", "NoThrowNativeDestructorReady", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-attach-entry-runtime-scaffold"
      categoryMarkers = @("debug-listener-native-attach-entry-runtime-scaffold")
      requiredMarkers = @("TensorRtDebugListenerNativeAttachEntryRuntimeScaffold", "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult", "Evaluate", "RuntimeScaffoldReady", "NativeOwnerLifecycleDryRunReady", "NativeAttachEntryLocated", "NativeDetachEntryLocated", "AttachEntryParameterShapeReady", "AttachEntryVersionGuardReady", "AttachEntryNoThrowBoundaryReady", "AttachEntryOwnershipDiagnosticsReady", "StableNativeOwnerIdentityReady", "NativeOwnerNonCopyableReady", "NoThrowNativeDestructorReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-attach-entry-minimal-safety"
      categoryMarkers = @("debug-listener-native-attach-entry-minimal-safety")
      requiredMarkers = @("TensorRtDebugListenerNativeAttachEntryMinimalSafety", "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult", "Evaluate", "RuntimeEvidenceKind", "MinimalSafetyReady", "RuntimeScaffoldReady", "LifecycleGateReady", "LifecyclePointerFree", "NativeAttachEntryLocated", "AttachEntryParameterShapeReady", "AttachEntryNoThrowReady", "AttachEntryVersionGuardReady", "AttachEntryOwnershipDiagnosticsReady", "SetDebugListenerNonNullEnabled", "NonNullAttachStillDisabled", "NativeAttachWouldBeBlocked", "ReasonNativeAttachStillBlocked", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-owner-stable-identity"
      categoryMarkers = @("debug-listener-native-owner-stable-identity")
      requiredMarkers = @("TensorRtDebugListenerNativeOwnerStableIdentity", "TensorRtDebugListenerNativeOwnerStableIdentityResult", "Evaluate", "RuntimeEvidenceKind", "NativeAttachEntryRuntimeScaffoldReady", "StableNativeOwnerIdentityReady", "OwnerIdentityDiagnosticsReady", "OwnerIdentityPointerFree", "NativeAttachEntryLocated", "NativeDetachEntryLocated", "NativeOwnerNonCopyableReady", "NoThrowNativeDestructorReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-owner-noncopyable-storage"
      categoryMarkers = @("debug-listener-native-owner-noncopyable-storage")
      requiredMarkers = @("TensorRtDebugListenerNativeOwnerNonCopyableStorage", "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult", "Evaluate", "RuntimeEvidenceKind", "NativeOwnerStableIdentityReady", "OwnerIdentityDiagnosticsReady", "OwnerIdentityPointerFree", "NativeOwnerNonCopyableReady", "NativeOwnerCopyBlocked", "NativeOwnerMoveBlocked", "NativeOwnerAddressExposed", "NativeOwnerPointerProduced", "NativeAttachEntryLocated", "NativeDetachEntryLocated", "NoThrowNativeDestructorReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-nothrow-destructor"
      categoryMarkers = @("debug-listener-native-nothrow-destructor")
      requiredMarkers = @("TensorRtDebugListenerNativeNoThrowDestructor", "TensorRtDebugListenerNativeNoThrowDestructorResult", "Evaluate", "RuntimeEvidenceKind", "NativeOwnerNonCopyableStorageReady", "NativeOwnerNonCopyableReady", "NativeOwnerCopyBlocked", "NativeOwnerMoveBlocked", "DestructorNoThrowScaffoldReady", "DestructorExceptionEscapeBlocked", "DestructorAddressExposed", "DestructorPointerProduced", "NoThrowNativeDestructorReady", "NativeOwnerLifecycleReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-owner-lifecycle-gate"
      categoryMarkers = @("debug-listener-native-owner-lifecycle-gate")
      requiredMarkers = @("TensorRtDebugListenerNativeOwnerLifecycleGate", "TensorRtDebugListenerNativeOwnerLifecycleGateResult", "Evaluate", "RuntimeEvidenceKind", "NativeNoThrowDestructorGateReady", "ManagedDisposeSnapshotReady", "LifecycleScaffoldReady", "ReleaseHookOrderingGateReady", "DisposeIdempotencyGateReady", "InFlightDrainGateReady", "CallbackStateUnpinAfterDetachGateReady", "DelegateUnpinAfterDetachGateReady", "LifecycleAddressExposed", "LifecyclePointerProduced", "LifecycleGateReady", "NativeOwnerLifecycleReady", "CanImplementNativeAttach", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-attach-bridge-shape-gate"
      categoryMarkers = @("debug-listener-native-attach-bridge-shape-gate")
      requiredMarkers = @("TensorRtDebugListenerNativeAttachBridgeShapeGate", "TensorRtDebugListenerNativeAttachBridgeShapeGateResult", "Evaluate", "RuntimeEvidenceKind", "NativeOwnerLifecycleGateReady", "AttachBridgeShapeReady", "AttachBridgeNoThrowBoundaryReady", "AttachBridgeVersionGuardReady", "AttachBridgeOwnershipDiagnosticsReady", "AttachBridgePointerFree", "SetDebugListenerNonNullEnabled", "NonNullAttachStillDisabled", "NativeAttachEntryLocated", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-exception-status-mapping-gate"
      categoryMarkers = @("debug-listener-exception-status-mapping-gate")
      requiredMarkers = @("TensorRtDebugListenerExceptionStatusMappingGate", "TensorRtDebugListenerExceptionStatusMappingGateResult", "Evaluate", "RuntimeEvidenceKind", "AttachBridgeShapeGateReady", "ManagedCallbackExceptionCaptureReady", "NativeCallbackExceptionCaptureReady", "CallbackStatusMappingGateReady", "ExceptionEscapeBlocked", "DiagnosticCopyReady", "MappingAddressExposed", "MappingPointerProduced", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-inflight-accounting-gate"
      categoryMarkers = @("debug-listener-inflight-accounting-gate")
      requiredMarkers = @("TensorRtDebugListenerInFlightAccountingGate", "TensorRtDebugListenerInFlightAccountingGateResult", "Evaluate", "RuntimeEvidenceKind", "ExceptionStatusMappingGateReady", "CallbackEnterAccountingGateReady", "CallbackLeaveAccountingGateReady", "CallbackInFlightNeverNegativeReady", "ReleaseAfterDrainGateReady", "CallbackStateUnpinAfterDrainGateReady", "AccountingAddressExposed", "AccountingPointerProduced", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-nothrow-vtable-scaffold-gate"
      categoryMarkers = @("debug-listener-native-nothrow-vtable-scaffold-gate")
      requiredMarkers = @("TensorRtDebugListenerNativeNoThrowVTableScaffoldGate", "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult", "Evaluate", "RuntimeEvidenceKind", "NativeAttachBridgeShapeGateReady", "ExceptionStatusMappingGateReady", "InFlightAccountingGateReady", "NoThrowVTableScaffoldReady", "VTableDestructorNoThrowReady", "ProcessDebugTensorCallbackStubNoThrowReady", "ExceptionEscapeBlocked", "CallbackExceptionCaptureGateReady", "CallbackStatusMappingGateReady", "CallbackInFlightAccountingGateReady", "VTableAddressExposed", "VTablePointerProduced", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-nothrow-vtable-callback-stub"
      categoryMarkers = @("debug-listener-nothrow-vtable-callback-stub")
      requiredMarkers = @("TensorRtDebugListenerNoThrowVTableCallbackStub", "TensorRtDebugListenerNoThrowVTableCallbackStubResult", "Evaluate", "RuntimeEvidenceKind", "CallbackStubGateReady", "CallbackStubShapeReady", "CallbackStubNoThrowReady", "CallbackMetadataCopyReady", "CallbackExceptionCaptureReady", "CallbackStatusMappingReady", "CallbackInFlightEnterReady", "CallbackInFlightLeaveReady", "CallbackInFlightPairingReady", "CallbackInFlightNeverNegativeReady", "BorrowedDebugTensorMetadataCopyReady", "BorrowedDebugTensorPointerEscapeBlocked", "DebugTensorPointerExposed", "DebugTensorDataPointerExposed", "SetDebugListenerNonNullEnabled", "NativeAttachWouldBeBlocked", "NativeVTableInstalled", "CanInstallNativeVTable", "CanCallProcessDebugTensorRuntime", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-borrowed-debug-tensor-metadata-runtime-gate"
      categoryMarkers = @("debug-listener-borrowed-debug-tensor-metadata-runtime-gate")
      requiredMarkers = @("TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate", "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult", "Evaluate", "RuntimeEvidenceKind", "MetadataGateReady", "TensorNameCopied", "TensorNameLength", "TensorTypeCopied", "TensorLocationCopied", "TensorShapeCopied", "TensorShapeRank", "TensorFlagsCopied", "BorrowedDebugTensorMetadataCopyReady", "BorrowedDebugTensorPointerEscapeBlocked", "BorrowedDebugTensorDataPointerEscapeBlocked", "DebugTensorPointerExposed", "DebugTensorDataPointerExposed", "BorrowedDebugTensorLifetimeReady", "BorrowedDebugTensorDataLifetimeReady", "CanCallProcessDebugTensorRuntime", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-vtable-install-preflight"
      categoryMarkers = @("debug-listener-native-vtable-install-preflight")
      requiredMarkers = @("TensorRtDebugListenerNativeVTableInstallPreflight", "TensorRtDebugListenerNativeVTableInstallPreflightResult", "Evaluate", "RuntimeEvidenceKind", "NativeVTableInstallPreflightReady", "VTableInstallShapeReady", "VTableInstallVersionGuardReady", "VTableInstallNoThrowBoundaryReady", "VTableInstallOwnershipDiagnosticsReady", "VTableInstallPointerFree", "SetDebugListenerNonNullEnabled", "NativeVTableInstalled", "NativeVTableInstallRuntimeReady", "CanEnableSetDebugListenerNonNull", "CanInstallNativeVTable", "CanCallProcessDebugTensorRuntime", "RuntimeProofBlocked", "ReasonNativeVTableInstallStillBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-native-owner-vtable-install-experiment"
      categoryMarkers = @("debug-listener-native-owner-vtable-install-experiment")
      requiredMarkers = @("TensorRtDebugListenerNativeOwnerVTableInstallExperiment", "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult", "Evaluate", "RuntimeEvidenceKind", "ExperimentShapeReady", "InstallAttemptGuardReady", "NonNullAttachEnabled", "RuntimeProofEnabled", "NativeVTableInstallAttempted", "NativeVTableInstalled", "RollbackReady", "DetachBeforeReleaseReady", "FailureStatusMappingReady", "PointerFree", "CanEnableSetDebugListenerNonNull", "CanInstallNativeVTable", "CanCallProcessDebugTensorRuntime", "RuntimeProofBlocked", "ReasonNativeOwnerVTableInstallStillBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-runtime-proof-precheck"
      categoryMarkers = @("debug-listener-runtime-proof-precheck")
      requiredMarkers = @("TensorRtDebugListenerRuntimeProofPrecheck", "TensorRtDebugListenerRuntimeProofPrecheckResult", "Evaluate", "AttachDetachDesignGateReady", "LineSpecificAttachDetachReady", "NativeVTableReady", "BorrowedTensorSafetyGateReady", "AttachVTableSafetyGateReady", "NativeAttachNoThrowPreflightReady", "NativeOwnerAddressDesignGateReady", "NativeNoThrowVTableDesignGateReady", "NativeAttachEntryDesignGateReady", "NativeDetachBeforeReleaseDesignGateReady", "NativeOwnerLifecycleDryRunReady", "NativeAttachEntryRuntimeScaffoldReady", "NativeOwnerStableIdentityReady", "OwnerIdentityDiagnosticsReady", "OwnerIdentityPointerFree", "NativeOwnerNonCopyableStorageReady", "NativeOwnerNonCopyableReady", "NativeOwnerCopyBlocked", "NativeOwnerMoveBlocked", "NativeOwnerAddressExposed", "NativeOwnerPointerProduced", "NativeNoThrowDestructorGateReady", "DestructorNoThrowScaffoldReady", "DestructorExceptionEscapeBlocked", "DestructorAddressExposed", "DestructorPointerProduced", "NoThrowNativeDestructorReady", "NativeOwnerLifecycleGateReady", "ManagedDisposeSnapshotReady", "LifecycleScaffoldReady", "ReleaseHookOrderingGateReady", "DisposeIdempotencyGateReady", "InFlightDrainGateReady", "CallbackStateUnpinAfterDetachGateReady", "DelegateUnpinAfterDetachGateReady", "LifecycleAddressExposed", "LifecyclePointerProduced", "AttachEntryParameterShapeReady", "AttachEntryNoThrowBoundaryReady", "AttachEntryOwnershipDiagnosticsReady", "LineSpecificAttachEntryDesignReady", "AttachEntryNoThrowReady", "AttachEntryVersionGuardReady", "AttachEntryOwnershipReady", "DetachBeforeReleaseReady", "ReleaseHookOrderingReady", "DisposeIdempotencyReady", "InFlightDrainBeforeReleaseReady", "CallbackStateUnpinAfterDetachReady", "DelegateUnpinAfterDetachReady", "BorrowedDebugTensorPointerEscapeBlocked", "ProcessDebugTensorRuntimeReady", "CanAttemptRuntimeProof", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-runtime-proof-attempt-preflight"
      categoryMarkers = @("debug-listener-runtime-proof-attempt-preflight")
      requiredMarkers = @("TensorRtDebugListenerRuntimeProofAttemptPreflight", "TensorRtDebugListenerRuntimeProofAttemptPreflightResult", "Evaluate", "RuntimeEvidenceKind", "CanEnableSetDebugListenerNonNull", "CanInstallNativeVTable", "CanCallProcessDebugTensorRuntime", "CanPromoteRealCallbackRuntime", "ReasonNonNullAttachStillBlocked", "ReasonNativeVTableStillBlocked", "ReasonRuntimeProofStillBlocked", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-real-non-null-attach-runtime-smoke"
      categoryMarkers = @("debug-listener-real-non-null-attach-runtime-smoke", "runtime-smoke-skipped", "runtime-smoke-blocked", "runtime-smoke-attempted")
      requiredMarkers = @("TensorRtDebugListenerRealNonNullAttachRuntimeSmoke", "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult", "Evaluate", "RuntimeEvidenceKind", "OptInEnabled", "FullPackageConsumerReport", "AttachGuardReady", "NativeVTableReady", "BorrowedDebugTensorRuntimeReady", "CallbackInvocationReady", "AttachAttempted", "AttachSucceeded", "DetachAttempted", "DetachSucceeded", "RollbackAttempted", "RollbackSucceeded", "NativeVTableInstalled", "ProcessDebugTensorInvoked", "InvocationCount", "AllocationCount", "ReleaseCount", "FailureCount", "InFlightCallbackCount", "LastStatus", "LastDiagnostic", "ReportPointerFree", "CanAttemptRuntimeProof", "CanPromoteRealCallbackRuntime", "ReasonRuntimeProofStillBlocked", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-process-debug-tensor-callback-trampoline"
      categoryMarkers = @("debug-listener-process-debug-tensor-callback-trampoline", "callback-trampoline-shape")
      requiredMarkers = @("TensorRtDebugListenerProcessDebugTensorCallbackTrampoline", "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult", "TensorRtDebugTensorMetadataSnapshot", "Evaluate", "RuntimeEvidenceKind", "TrampolineShapeReady", "NativeCallbackEntryLocated", "NoThrowCallbackEntryReady", "ExceptionCaptureReady", "CallbackStatusMappingReady", "InFlightAccountingReady", "DetachBeforeReleaseReady", "BorrowedDebugTensorMetadataCopyReady", "BorrowedDebugTensorPointerExposed", "BorrowedDebugTensorDataPointerExposed", "PointerFreeSurfaceReady", "ProcessDebugTensorRuntimeReady", "CallbackStubEntryCount", "CallbackStubLeaveCount", "CanPromoteRealCallbackRuntime", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-real-callback-runtime-proof"
      categoryMarkers = @("debug-listener-real-callback-runtime-proof", "real-callback-runtime-blocked", "attempted-no-invocation")
      requiredMarkers = @("TensorRtDebugListenerRealCallbackRuntimeProof", "TensorRtDebugListenerRealCallbackRuntimeProofResult", "Evaluate", "RuntimeEvidenceKind", "OptInEnabled", "FullPackageConsumerReport", "RuntimeSmokeReady", "TrampolineShapeReady", "AttachAttempted", "AttachSucceeded", "DetachAttempted", "DetachSucceeded", "RollbackAttempted", "RollbackSucceeded", "NativeVTableInstalled", "ProcessDebugTensorInvoked", "InvocationCount", "FailureCount", "InFlightCallbackCount", "BorrowedDebugTensorMetadataCopied", "PointerFreeSurfaceReady", "ProcessDebugTensorRuntimeReady", "AttemptedNoInvocation", "CanPromoteRealCallbackRuntime", "RuntimeProofBlocked")
    },
    [pscustomobject]@{
      name = "debug-listener-callback-proof-gap-report"
      categoryMarkers = @("debug-listener-callback-proof-gap-report", "proof-gap-report")
      requiredMarkers = @("TensorRtDebugListenerCallbackProofGapReport", "TensorRtDebugListenerCallbackProofGapReportResult", "Evaluate", "RuntimeEvidenceKind", "NonNullAttachStillDisabled", "NativeAttachEntryReady", "NativeVTableInstallBlocked", "NoThrowCallbackEntryReady", "ExceptionStatusMappingReady", "InFlightAccountingReady", "BorrowedDebugTensorMetadataCopied", "DetachRollbackReady", "ProcessDebugTensorRuntimeInvoked", "FullPackageConsumerRuntimeProofReady", "PointerFreeSurfaceReady", "AttemptedNoInvocation", "InvocationCount", "FailureCount", "InFlightCallbackCount", "CanPromoteRealCallbackRuntime", "RuntimeProofBlocked", "GapReasonCount", "PrimaryGapReason", "RuntimeProofBlockerCategory", "PackageConsumerRuntimeProofRequired", "RuntimeInvocationRequired", "EvidenceSource", "NextOwnerAction")
    },
    [pscustomobject]@{
      name = "profiler-safe-controls"
      categoryMarkers = @("profiler-safe-controls")
      requiredMarkers = @("SetProfiler", "ClearProfiler", "HasNativeProfiler", "HasProfiler")
    },
    [pscustomobject]@{
      name = "progress-monitor-safe-controls"
      categoryMarkers = @("progress-monitor-safe-controls")
      requiredMarkers = @("SetProgressMonitor", "ClearProgressMonitor", "HasProgressMonitor")
    },
    [pscustomobject]@{
      name = "cuda-memory-range"
      categoryMarkers = @("cuda-memory-range")
      requiredMarkers = @("GetRangeAttribute", "GetRangeAttributes", "GetRangeAccessedByDevices", "Advise", "PrefetchAsync")
    }
  )

  $groups = @(
    foreach ($definition in $definitions) {
      $present = (Test-WrapperSurfaceMarker -Markers @($definition.categoryMarkers)) -or
        (Test-WrapperSurfaceRequirements -RequiredMarkers @($definition.requiredMarkers))

      [pscustomobject]@{
        name = [string]$definition.name
        status = if ($present) { "ready" } else { "missing" }
      }
    }
  )

  $missingGroups = @($groups | Where-Object { [string]$_.status -ne "ready" } | ForEach-Object { [string]$_.name })
  $status = if ([string]::IsNullOrWhiteSpace($surfaceText)) {
    "missing"
  }
  elseif ($missingGroups.Count -eq 0) {
    "ready"
  }
  else {
    "partial"
  }

  $groupStatus = @{}
  foreach ($group in @($groups)) {
    $groupStatus[[string]$group.name] = [string]$group.status -eq "ready"
  }

  return [pscustomobject]@{
    status = $status
    itemCount = $tokens.Count
    items = @($tokens.ToArray())
    groups = @($groups)
    missingGroups = @($missingGroups)
    hasPluginInventory = [bool]$groupStatus["plugin-inventory"]
    hasPluginInventoryFieldMetadata = [bool]$groupStatus["plugin-inventory-field-metadata"]
    hasOnnxParserDiagnosticReadiness = [bool]$groupStatus["onnx-parser-diagnostic-readiness"]
    hasOnnxParserRefitterDiagnosticReadiness = [bool]$groupStatus["onnx-parser-refitter-diagnostic-readiness"]
    hasEngineRnnReadonlyDiagnostics = [bool]$groupStatus["engine-rnn-readonly-diagnostics"]
    hasManagedCallbacks = [bool]$groupStatus["managed-callbacks"]
    hasCallbackDiagnostics = [bool]$groupStatus["callback-diagnostics"]
    hasCallbackApiLanguageSafeControls = [bool]$groupStatus["callback-api-language-safe-controls"]
    hasErrorRecorderSnapshots = [bool]$groupStatus["error-recorder-snapshot"]
    hasErrorRecorderDiagnosticsDesignGate = [bool]$groupStatus["error-recorder-diagnostics-design-gate"]
    hasDimensionExpressionSnapshotDesignGate = [bool]$groupStatus["dimension-expression-snapshot-design-gate"]
    hasCalibratorMetadataDesignGate = [bool]$groupStatus["calibrator-metadata-design-gate"]
    hasRuntimeDeserializationBoundaryPrecheck = [bool]$groupStatus["runtime-deserialization-boundary-precheck"]
    hasLoggerPresenceSafeControls = [bool]$groupStatus["logger-presence-safe-controls"]
    hasAllocatorDebugListenerSafeControls = [bool]$groupStatus["allocator-debug-listener-safe-controls"]
    hasCallbackInterfaceInfoSafeControls = [bool]$groupStatus["callback-interface-info-safe-controls"]
    hasExecutionContextCallbackStateSnapshot = [bool]$groupStatus["execution-context-callback-state-snapshot"]
    hasExecutionContextCallbackAllocatorSafeControlSummary = [bool]$groupStatus["execution-context-callback-allocator-safe-control-summary"]
    hasAllocatorOwnerDryRunDiagnostics = [bool]$groupStatus["allocator-owner-dry-run-diagnostics"]
    hasAllocatorOwnerNativeDryRunControls = [bool]$groupStatus["allocator-owner-native-dry-run-controls"]
    hasAllocatorOwnerStateLedgerDryRunControls = [bool]$groupStatus["allocator-owner-state-ledger-dry-run-controls"]
    hasAllocatorOwnerLedgerSafetyGate = [bool]$groupStatus["allocator-owner-ledger-safety-gate"]
    hasOutputAllocatorCallbackOwnerDesign = [bool]$groupStatus["output-allocator-callback-owner-design"]
    hasOutputAllocatorAttachDetachDesignGate = [bool]$groupStatus["output-allocator-attach-detach-design-gate"]
    hasOutputBufferOwnershipSafetyGate = [bool]$groupStatus["output-buffer-ownership-safety-gate"]
    hasOutputAllocatorRuntimeProofPrecheck = [bool]$groupStatus["output-allocator-runtime-proof-precheck"]
    hasCallbackOwnerClosureMatrix = [bool]$groupStatus["callback-owner-closure-matrix"]
    hasDebugListenerCallbackOwnerDesign = [bool]$groupStatus["debug-listener-callback-owner-design"]
    hasDebugListenerAttachDetachDesignGate = [bool]$groupStatus["debug-listener-attach-detach-design-gate"]
    hasDebugListenerBorrowedTensorSafetyGate = [bool]$groupStatus["debug-listener-borrowed-tensor-safety-gate"]
    hasDebugListenerAttachVTableSafetyGate = [bool]$groupStatus["debug-listener-attach-vtable-safety-gate"]
    hasDebugListenerNativeAttachNoThrowPreflight = [bool]$groupStatus["debug-listener-native-attach-nothrow-preflight"]
    hasDebugListenerNativeOwnerAddressDesignGate = [bool]$groupStatus["debug-listener-native-owner-address-design-gate"]
    hasDebugListenerNativeNoThrowVTableDesignGate = [bool]$groupStatus["debug-listener-native-nothrow-vtable-design-gate"]
    hasDebugListenerNativeAttachEntryDesignGate = [bool]$groupStatus["debug-listener-native-attach-entry-design-gate"]
    hasDebugListenerNativeDetachBeforeReleaseDesignGate = [bool]$groupStatus["debug-listener-native-detach-before-release-design-gate"]
    hasDebugListenerNativeOwnerLifecycleDryRun = [bool]$groupStatus["debug-listener-native-owner-lifecycle-dry-run"]
    hasDebugListenerNativeAttachEntryRuntimeScaffold = [bool]$groupStatus["debug-listener-native-attach-entry-runtime-scaffold"]
    hasDebugListenerNativeAttachEntryMinimalSafety = [bool]$groupStatus["debug-listener-native-attach-entry-minimal-safety"]
    hasDebugListenerNativeOwnerStableIdentity = [bool]$groupStatus["debug-listener-native-owner-stable-identity"]
    hasDebugListenerNativeNoThrowDestructor = [bool]$groupStatus["debug-listener-native-nothrow-destructor"]
    hasDebugListenerNativeOwnerLifecycleGate = [bool]$groupStatus["debug-listener-native-owner-lifecycle-gate"]
    hasDebugListenerNativeAttachBridgeShapeGate = [bool]$groupStatus["debug-listener-native-attach-bridge-shape-gate"]
    hasDebugListenerExceptionStatusMappingGate = [bool]$groupStatus["debug-listener-exception-status-mapping-gate"]
    hasDebugListenerInFlightAccountingGate = [bool]$groupStatus["debug-listener-inflight-accounting-gate"]
    hasDebugListenerNativeNoThrowVTableScaffoldGate = [bool]$groupStatus["debug-listener-native-nothrow-vtable-scaffold-gate"]
    hasDebugListenerNoThrowVTableCallbackStub = [bool]$groupStatus["debug-listener-nothrow-vtable-callback-stub"]
    hasDebugListenerBorrowedDebugTensorMetadataRuntimeGate = [bool]$groupStatus["debug-listener-borrowed-debug-tensor-metadata-runtime-gate"]
    hasDebugListenerNativeVTableInstallPreflight = [bool]$groupStatus["debug-listener-native-vtable-install-preflight"]
    hasDebugListenerNativeOwnerVTableInstallExperiment = [bool]$groupStatus["debug-listener-native-owner-vtable-install-experiment"]
    hasDebugListenerRuntimeProofPrecheck = [bool]$groupStatus["debug-listener-runtime-proof-precheck"]
    hasDebugListenerRuntimeProofAttemptPreflight = [bool]$groupStatus["debug-listener-runtime-proof-attempt-preflight"]
    hasDebugListenerRealNonNullAttachRuntimeSmoke = [bool]$groupStatus["debug-listener-real-non-null-attach-runtime-smoke"]
    hasDebugListenerProcessDebugTensorCallbackTrampoline = [bool]$groupStatus["debug-listener-process-debug-tensor-callback-trampoline"]
    hasDebugListenerRealCallbackRuntimeProof = [bool]$groupStatus["debug-listener-real-callback-runtime-proof"]
    hasDebugListenerCallbackProofGapReport = [bool]$groupStatus["debug-listener-callback-proof-gap-report"]
    hasProfilerSafeControls = [bool]$groupStatus["profiler-safe-controls"]
    hasProgressMonitorSafeControls = [bool]$groupStatus["progress-monitor-safe-controls"]
    hasCudaMemoryRange = [bool]$groupStatus["cuda-memory-range"]
  }
}

function Get-BridgeConsumerEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Key
  )

  $reportPath = Join-Path $PackageConsumerReportDirectory "bridge-package-consumer-validation-summary.json"
  if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    return [pscustomobject]@{
      status = "missing"
      reportPath = $reportPath
      probeResult = ""
      highLevelWrapperSurface = ""
      wrapperSurfaceCapabilities = (New-WrapperSurfaceCapabilityEvidence -Surface "")
      evidenceKind = "missing"
      isBridgeOnlyEvidence = $false
      isFullRuntimeEvidence = $false
      isRuntimeExecutionEvidence = $false
      diagnostic = "bridge package consumer report was not found."
    }
  }

  try {
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $probeResult = [string]$report.ProbeResult
    $probeDiagnostic = [string]$report.ProbeDiagnostic
    $nativeDependencyStatus = [string]$report.NativeDependencyStatus
    $nativeDependencyDiagnostic = [string]$report.NativeDependencyDiagnostic
    $nativeDependencyProbeLine = [string]$report.NativeDependencyProbeLine
    $nativeDependencySkippedReason = [string]$report.NativeDependencySkippedReason
    $vendorExceptionCode = [string]$report.VendorExceptionCode
    if ([string]::IsNullOrWhiteSpace($nativeDependencyStatus)) {
      switch ($probeResult) {
        "not-requested" {
          $nativeDependencyStatus = "not-requested"
          if ([string]::IsNullOrWhiteSpace($nativeDependencyDiagnostic)) {
            $nativeDependencyDiagnostic = "native dependency probe was not requested, or the bridge consumer report was produced before native dependency fields were added."
          }
          break
        }
        "environment-probe-succeeded" {
          $nativeDependencyStatus = "ready"
          if ([string]::IsNullOrWhiteSpace($nativeDependencyDiagnostic)) {
            $nativeDependencyDiagnostic = "environment probe succeeded."
          }
          break
        }
        "skipped-with-diagnostic" {
          $nativeDependencyStatus = "skipped-with-diagnostic"
          if ([string]::IsNullOrWhiteSpace($nativeDependencyDiagnostic)) {
            $nativeDependencyDiagnostic = "environment probe emitted Skipped=True, but the bridge consumer report did not include a parsed native dependency diagnostic."
          }
          break
        }
        "blocked-by-application-control" {
          $nativeDependencyStatus = "blocked-by-application-control"
          if ([string]::IsNullOrWhiteSpace($nativeDependencyDiagnostic)) {
            $nativeDependencyDiagnostic = "probe execution was blocked by the Windows application control policy."
          }
          break
        }
        default {
          $nativeDependencyStatus = "not-reported"
          if ([string]::IsNullOrWhiteSpace($nativeDependencyDiagnostic)) {
            $nativeDependencyDiagnostic = "bridge consumer report does not include native dependency fields."
          }
          break
        }
      }
    }
    if ([string]::IsNullOrWhiteSpace($probeDiagnostic)) {
      $probeDiagnostic = $nativeDependencyDiagnostic
    }
    $wrapperSurface = [string]$report.HighLevelWrapperSurface
    $wrapperCapabilities = New-WrapperSurfaceCapabilityEvidence -Surface $wrapperSurface
    $evidenceKind = switch ($probeResult) {
      "environment-probe-succeeded" { "bridge-layout-wrapper-surface-dependency-probe"; break }
      "skipped-with-diagnostic" { "bridge-layout-wrapper-surface-skipped-diagnostic"; break }
      "dependency-probe-only" { "bridge-layout-wrapper-surface-dependency-probe-only"; break }
      "blocked-by-application-control" { "bridge-layout-wrapper-surface-application-control-blocked"; break }
      "" { "unknown"; break }
      default { "bridge-layout-wrapper-surface-diagnostic"; break }
    }

    if ([string]$report.SourceRuntimeKey -ne $Key) {
      return [pscustomobject]@{
        status = "mismatch"
        reportPath = $reportPath
        probeResult = $probeResult
        probeDiagnostic = $probeDiagnostic
        nativeDependencyStatus = $nativeDependencyStatus
        nativeDependencyDiagnostic = $nativeDependencyDiagnostic
        nativeDependencyProbeLine = $nativeDependencyProbeLine
        nativeDependencySkippedReason = $nativeDependencySkippedReason
        vendorExceptionCode = $vendorExceptionCode
        isNativeDependencyMissing = [bool]$report.IsNativeDependencyMissing
        isVendorStructuredException = [bool]$report.IsVendorStructuredException
        highLevelWrapperSurface = $wrapperSurface
        wrapperSurfaceCapabilities = $wrapperCapabilities
        evidenceKind = "mismatch"
        isBridgeOnlyEvidence = $false
        isFullRuntimeEvidence = $false
        isRuntimeExecutionEvidence = $false
        diagnostic = "bridge package consumer report is for '$($report.SourceRuntimeKey)', not '$Key'."
      }
    }

    return [pscustomobject]@{
      status = if ([string]::IsNullOrWhiteSpace($probeResult)) { "unknown" } else { "ready" }
      reportPath = $reportPath
      probeResult = $probeResult
      probeDiagnostic = $probeDiagnostic
      nativeDependencyStatus = $nativeDependencyStatus
      nativeDependencyDiagnostic = $nativeDependencyDiagnostic
      nativeDependencyProbeLine = $nativeDependencyProbeLine
      nativeDependencySkippedReason = $nativeDependencySkippedReason
      vendorExceptionCode = $vendorExceptionCode
      isNativeDependencyMissing = [bool]$report.IsNativeDependencyMissing
      isVendorStructuredException = [bool]$report.IsVendorStructuredException
      highLevelWrapperSurface = $wrapperSurface
      wrapperSurfaceCapabilities = $wrapperCapabilities
      evidenceKind = $evidenceKind
      isBridgeOnlyEvidence = $true
      isFullRuntimeEvidence = $false
      isRuntimeExecutionEvidence = $false
      bridgePackageId = [string]$report.BridgePackageId
      bridgePackageVersion = [string]$report.BridgePackageVersion
      bridgeNupkgRuntimeLayout = [string]$report.BridgeNupkgRuntimeLayout
      outputRootCopied = [bool]$report.BridgeOutputRootCopied
      outputRuntimeLayoutCopied = [bool]$report.BridgeOutputRuntimeLayoutCopied
      diagnostic = "bridge package consumer evidence found."
    }
  }
  catch {
    return [pscustomobject]@{
      status = "invalid-report"
      reportPath = $reportPath
      probeResult = ""
      highLevelWrapperSurface = ""
      wrapperSurfaceCapabilities = (New-WrapperSurfaceCapabilityEvidence -Surface "")
      evidenceKind = "invalid-report"
      isBridgeOnlyEvidence = $false
      isFullRuntimeEvidence = $false
      isRuntimeExecutionEvidence = $false
      diagnostic = $_.Exception.Message
    }
  }
}

function Get-FullPackageConsumerEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Key
  )

  $reportPath = Join-Path $PackageConsumerReportDirectory "package-consumer-validation-summary.json"
  if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    return [pscustomobject]@{
      status = "missing"
      reportPath = $reportPath
      smokeResult = ""
      smokeRequested = $false
      smokeExitCode = $null
      smokeDiagnostic = ""
      smokeCommand = ""
      smokeOutputLines = @()
      evidenceKind = "missing"
      runtimeSmokeClassification = "missing"
      isDependencyProbeOnly = $true
      isRealCallbackRuntimeProof = $false
      isFullRuntimeEvidence = $false
      isRuntimeExecutionEvidence = $false
      diagnostic = "full runtime package consumer report was not found."
    }
  }

  try {
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $rows = @($report)
    $row = $rows | Where-Object { [string]$_.RuntimePackageKey -eq $Key } | Select-Object -First 1
    if (-not $row) {
      return [pscustomobject]@{
        status = "missing-key"
        reportPath = $reportPath
        smokeResult = ""
        smokeRequested = $false
        smokeExitCode = $null
        smokeDiagnostic = ""
        smokeCommand = ""
        smokeOutputLines = @()
        evidenceKind = "missing-key"
        runtimeSmokeClassification = "missing-key"
        isDependencyProbeOnly = $true
        isRealCallbackRuntimeProof = $false
        isFullRuntimeEvidence = $false
        isRuntimeExecutionEvidence = $false
        diagnostic = "full runtime package consumer report does not contain '$Key'."
      }
    }

    $smokeResult = [string]$row.SmokeResult
    $smokeRequested = if ($row.PSObject.Properties.Name -contains "SmokeRequested") { [bool]$row.SmokeRequested } else { $smokeResult -ne "not-requested" -and -not [string]::IsNullOrWhiteSpace($smokeResult) }
    $smokeExitCode = if ($row.PSObject.Properties.Name -contains "SmokeExitCode" -and $null -ne $row.SmokeExitCode) { [int]$row.SmokeExitCode } else { $null }
    $smokeDiagnostic = if ($row.PSObject.Properties.Name -contains "SmokeDiagnostic") { [string]$row.SmokeDiagnostic } else { "" }
    $smokeCommand = if ($row.PSObject.Properties.Name -contains "SmokeCommand") { [string]$row.SmokeCommand } else { "" }
    $smokeOutputLines = if ($row.PSObject.Properties.Name -contains "SmokeOutputLines") { @($row.SmokeOutputLines | ForEach-Object { [string]$_ }) } else { @() }
    $fallbackEvidenceKind = switch ($smokeResult) {
      "passed" { "full-runtime-package-consumer-smoke"; break }
      "not-requested" { "full-runtime-package-consumer-compile-only"; break }
      "blocked-by-application-control" { "full-runtime-package-consumer-smoke-application-control-blocked"; break }
      "blocked-by-cuda-driver" { "full-runtime-package-consumer-smoke-driver-blocked"; break }
      "failed" { "full-runtime-package-consumer-smoke-failed"; break }
      "" { "full-runtime-package-consumer-unknown"; break }
      default { "full-runtime-package-consumer-diagnostic"; break }
    }
    $fallbackRuntimeSmokeClassification = switch ($smokeResult) {
      "passed" { "runtime-smoke-passed"; break }
      "not-requested" { "not-requested"; break }
      "blocked-by-application-control" { "runtime-smoke-application-control-blocked"; break }
      "blocked-by-cuda-driver" { "runtime-smoke-driver-blocked"; break }
      "failed" { "runtime-smoke-failed"; break }
      "" { "runtime-smoke-unknown"; break }
      default { "runtime-smoke-diagnostic"; break }
    }
    $packageConsumerEvidenceKind = if ($row.PSObject.Properties.Name -contains "EvidenceKind" -and -not [string]::IsNullOrWhiteSpace([string]$row.EvidenceKind)) { [string]$row.EvidenceKind } else { $fallbackEvidenceKind }
    $runtimeSmokeClassification = if ($row.PSObject.Properties.Name -contains "RuntimeSmokeClassification" -and -not [string]::IsNullOrWhiteSpace([string]$row.RuntimeSmokeClassification)) { [string]$row.RuntimeSmokeClassification } else { $fallbackRuntimeSmokeClassification }
    $isRuntimeExecutionEvidence = if ($row.PSObject.Properties.Name -contains "IsRuntimeExecutionEvidence") { [bool]$row.IsRuntimeExecutionEvidence } else { $smokeResult -eq "passed" }
    $isDependencyProbeOnly = if ($row.PSObject.Properties.Name -contains "IsDependencyProbeOnly") { [bool]$row.IsDependencyProbeOnly } else { -not $isRuntimeExecutionEvidence }
    $isRealCallbackRuntimeProof = if ($row.PSObject.Properties.Name -contains "IsRealCallbackRuntimeProof") { [bool]$row.IsRealCallbackRuntimeProof } else { $false }
    $callbackRuntimeEvidence = if ($row.PSObject.Properties.Name -contains "RealCallbackRuntimeEvidence") { $row.RealCallbackRuntimeEvidence } else { $null }
    $callbackRuntimeEvidenceStatus = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "Status") { [string]$callbackRuntimeEvidence.Status } else { "" }
    $callbackRuntimeEvidenceKind = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "EvidenceKind") { [string]$callbackRuntimeEvidence.EvidenceKind } else { "" }
    $callbackRuntimeRuntimeEvidenceKind = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "RuntimeEvidenceKind") { [string]$callbackRuntimeEvidence.RuntimeEvidenceKind } else { "" }
    $callbackRuntimeIsProof = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "IsRealCallbackRuntimeProof") { [bool]$callbackRuntimeEvidence.IsRealCallbackRuntimeProof } else { $false }
    $callbackRuntimeRequiredMarkers = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "RequiredSmokeMarkers") { @($callbackRuntimeEvidence.RequiredSmokeMarkers | ForEach-Object { [string]$_ }) } else { @() }
    $callbackRuntimeMissingMarkers = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "MissingSmokeMarkers") { @($callbackRuntimeEvidence.MissingSmokeMarkers | ForEach-Object { [string]$_ }) } else { @() }
    $callbackRuntimeMatchedLines = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "MatchedSmokeLines") { @($callbackRuntimeEvidence.MatchedSmokeLines | ForEach-Object { [string]$_ }) } else { @() }
    $callbackRuntimeDiagnostic = if ($null -ne $callbackRuntimeEvidence -and $callbackRuntimeEvidence.PSObject.Properties.Name -contains "Diagnostic") { [string]$callbackRuntimeEvidence.Diagnostic } else { "" }

    return [pscustomobject]@{
      status = "ready"
      reportPath = $reportPath
      smokeResult = $smokeResult
      smokeRequested = $smokeRequested
      smokeExitCode = $smokeExitCode
      smokeDiagnostic = $smokeDiagnostic
      smokeCommand = $smokeCommand
      smokeOutputLines = @($smokeOutputLines)
      callbackRuntimeEvidenceStatus = $callbackRuntimeEvidenceStatus
      callbackRuntimeEvidenceKind = $callbackRuntimeEvidenceKind
      callbackRuntimeRuntimeEvidenceKind = $callbackRuntimeRuntimeEvidenceKind
      callbackRuntimeIsProof = $callbackRuntimeIsProof
      callbackRuntimeRequiredMarkers = @($callbackRuntimeRequiredMarkers)
      callbackRuntimeMissingMarkers = @($callbackRuntimeMissingMarkers)
      callbackRuntimeMatchedLines = @($callbackRuntimeMatchedLines)
      callbackRuntimeDiagnostic = $callbackRuntimeDiagnostic
      evidenceKind = $packageConsumerEvidenceKind
      packageConsumerEvidenceKind = $packageConsumerEvidenceKind
      runtimeSmokeClassification = $runtimeSmokeClassification
      isDependencyProbeOnly = $isDependencyProbeOnly
      isRealCallbackRuntimeProof = $isRealCallbackRuntimeProof
      isFullRuntimeEvidence = $true
      isRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
      runtimePackageId = [string]$row.RuntimePackageId
      runtimePackageVersion = [string]$row.RuntimePackageVersion
      nativeAssetsExpected = [int]$row.NativeAssetsExpected
      nativeAssetsFound = [int]$row.NativeAssetsFound
      missingNativeAssets = @($row.MissingNativeAssets)
      diagnostic = "full runtime package consumer evidence found."
    }
  }
  catch {
    return [pscustomobject]@{
      status = "invalid-report"
      reportPath = $reportPath
      smokeResult = ""
      smokeRequested = $false
      smokeExitCode = $null
      smokeDiagnostic = ""
      smokeCommand = ""
      smokeOutputLines = @()
      evidenceKind = "invalid-report"
      runtimeSmokeClassification = "invalid-report"
      isDependencyProbeOnly = $true
      isRealCallbackRuntimeProof = $false
      isFullRuntimeEvidence = $false
      isRuntimeExecutionEvidence = $false
      diagnostic = $_.Exception.Message
    }
  }
}

function Get-SplitCollectionConsumerEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Key,
    [Parameter(Mandatory = $true)]
    [string]$PackageId
  )

  $reportPath = Join-Path $SplitCollectionConsumerReportDirectory "package-consumer-validation-summary.json"
  if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    return [pscustomobject]@{
      status = "missing"
      reportPath = $reportPath
      smokeResult = ""
      smokeRequested = $false
      smokeExitCode = $null
      smokeDiagnostic = ""
      smokeCommand = ""
      smokeOutputLines = @()
      diagnostic = "split collection package consumer report was not found."
    }
  }

  try {
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $rows = @($report)
    $row = $rows | Where-Object { [string]$_.RuntimePackageKey -eq $Key -and [string]$_.RuntimePackageId -eq $PackageId } | Select-Object -First 1
    if (-not $row) {
      return [pscustomobject]@{
        status = "missing-key"
        reportPath = $reportPath
        smokeResult = ""
        smokeRequested = $false
        smokeExitCode = $null
        smokeDiagnostic = ""
        smokeCommand = ""
        smokeOutputLines = @()
        diagnostic = "split collection package consumer report does not contain '$Key' for '$PackageId'."
      }
    }

    return [pscustomobject]@{
      status = "ready"
      reportPath = $reportPath
      smokeResult = [string]$row.SmokeResult
      smokeRequested = if ($row.PSObject.Properties.Name -contains "SmokeRequested") { [bool]$row.SmokeRequested } else { [string]$row.SmokeResult -ne "not-requested" -and -not [string]::IsNullOrWhiteSpace([string]$row.SmokeResult) }
      smokeExitCode = if ($row.PSObject.Properties.Name -contains "SmokeExitCode" -and $null -ne $row.SmokeExitCode) { [int]$row.SmokeExitCode } else { $null }
      smokeDiagnostic = if ($row.PSObject.Properties.Name -contains "SmokeDiagnostic") { [string]$row.SmokeDiagnostic } else { "" }
      smokeCommand = if ($row.PSObject.Properties.Name -contains "SmokeCommand") { [string]$row.SmokeCommand } else { "" }
      smokeOutputLines = if ($row.PSObject.Properties.Name -contains "SmokeOutputLines") { @($row.SmokeOutputLines | ForEach-Object { [string]$_ }) } else { @() }
      evidenceKind = switch ([string]$row.SmokeResult) {
        "passed" { "split-collection-package-consumer-smoke"; break }
        "not-requested" { "split-collection-package-consumer-compile-only"; break }
        "blocked-by-application-control" { "split-collection-package-consumer-smoke-blocked"; break }
        "blocked-by-cuda-driver" { "split-collection-package-consumer-smoke-driver-blocked"; break }
        "failed" { "split-collection-package-consumer-smoke-failed"; break }
        "" { "split-collection-package-consumer-unknown"; break }
        default { "split-collection-package-consumer-diagnostic"; break }
      }
      isRuntimeExecutionEvidence = [string]$row.SmokeResult -eq "passed"
      runtimePackageId = [string]$row.RuntimePackageId
      runtimePackageVersion = [string]$row.RuntimePackageVersion
      nativeAssetsExpected = [int]$row.NativeAssetsExpected
      nativeAssetsFound = [int]$row.NativeAssetsFound
      missingNativeAssets = @($row.MissingNativeAssets)
      diagnostic = "split collection package consumer evidence found."
    }
  }
  catch {
    return [pscustomobject]@{
      status = "invalid-report"
      reportPath = $reportPath
      smokeResult = ""
      smokeRequested = $false
      smokeExitCode = $null
      smokeDiagnostic = ""
      smokeCommand = ""
      smokeOutputLines = @()
      diagnostic = $_.Exception.Message
    }
  }
}

function New-RuntimeExecutionEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SplitCollectionConsumer,
    [Parameter(Mandatory = $true)]
    [object]$FullPackageConsumer
  )

  $consumers = @(
    [pscustomobject]@{ scope = "split-collection-consumer"; evidence = $SplitCollectionConsumer },
    [pscustomobject]@{ scope = "full-package-consumer"; evidence = $FullPackageConsumer }
  )

  $rows = @(
    foreach ($consumer in $consumers) {
      $evidence = $consumer.evidence
      [pscustomobject]@{
        scope = [string]$consumer.scope
        status = [string]$evidence.status
        smokeRequested = [bool]$evidence.smokeRequested
        smokeResult = [string]$evidence.smokeResult
        smokeExitCode = $evidence.smokeExitCode
        smokeDiagnostic = [string]$evidence.smokeDiagnostic
        smokeCommand = [string]$evidence.smokeCommand
        evidenceKind = [string]$evidence.evidenceKind
        runtimeSmokeClassification = if ($evidence.PSObject.Properties.Name -contains "runtimeSmokeClassification") { [string]$evidence.runtimeSmokeClassification } else { [string]$evidence.smokeResult }
        isRuntimeExecutionEvidence = [bool]$evidence.isRuntimeExecutionEvidence
        isDependencyProbeOnly = if ($evidence.PSObject.Properties.Name -contains "isDependencyProbeOnly") { [bool]$evidence.isDependencyProbeOnly } else { -not [bool]$evidence.isRuntimeExecutionEvidence }
        isRealCallbackRuntimeProof = if ($evidence.PSObject.Properties.Name -contains "isRealCallbackRuntimeProof") { [bool]$evidence.isRealCallbackRuntimeProof } else { $false }
        reportPath = [string]$evidence.reportPath
      }
    }
  )

  $passedRows = @($rows | Where-Object { [bool]$_.isRuntimeExecutionEvidence })
  $failedRows = @($rows | Where-Object { [string]$_.smokeResult -eq "failed" })
  $blockedRows = @($rows | Where-Object { [string]$_.smokeResult -eq "blocked-by-application-control" })
  $cudaDriverBlockedRows = @($rows | Where-Object { [string]$_.smokeResult -eq "blocked-by-cuda-driver" })
  $requestedRows = @($rows | Where-Object { [bool]$_.smokeRequested })

  $status = if ($passedRows.Count -gt 0) {
    "ready"
  }
  elseif ($failedRows.Count -gt 0) {
    "blocked"
  }
  elseif ($cudaDriverBlockedRows.Count -gt 0) {
    "blocked-by-cuda-driver"
  }
  elseif ($blockedRows.Count -gt 0) {
    "blocked-by-application-control"
  }
  elseif ($requestedRows.Count -eq 0) {
    "not-requested"
  }
  else {
    "unknown"
  }

  $diagnostic = switch ($status) {
    "ready" { "at least one package consumer smoke executed successfully."; break }
    "blocked" { "at least one package consumer smoke was requested and failed; package restore/build/native-copy evidence remains separate."; break }
    "blocked-by-cuda-driver" { "smoke execution reached the packaged runtime, but CUDA driver/runtime compatibility blocked execution."; break }
    "blocked-by-application-control" { "smoke execution was requested but blocked by application control policy."; break }
    "not-requested" { "package consumer smoke has not been requested; package readiness is not runtime execution proof."; break }
    default { "runtime execution smoke evidence is inconclusive."; break }
  }

  return [pscustomobject]@{
    status = $status
    diagnostic = $diagnostic
    consumers = @($rows)
  }
}

function New-RuntimeProofBlockerOwnerAction {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Key,
    [Parameter(Mandatory = $true)]
    [object]$RuntimeExecutionEvidence,
    [Parameter(Mandatory = $true)]
    [object]$FullPackageConsumer,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeProofStatus,
    [Parameter(Mandatory = $true)]
    [bool]$RuntimeProofRequiredForRelease,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeProofDiagnostic
  )

  $classification = if ($FullPackageConsumer.PSObject.Properties.Name -contains "runtimeSmokeClassification") { [string]$FullPackageConsumer.runtimeSmokeClassification } else { [string]$FullPackageConsumer.smokeResult }
  $blockerCategory = switch ($RuntimeProofStatus) {
    "ready" { "none"; break }
    "blocked-by-cuda-driver" { "cuda-driver-runtime-compatibility"; break }
    "blocked-by-application-control" { "application-control-policy"; break }
    "not-requested" { "runtime-smoke-not-requested"; break }
    "blocked" { "runtime-smoke-failed"; break }
    default { "runtime-proof-incomplete"; break }
  }

  $summary = switch ($RuntimeProofStatus) {
    "ready" { "runtime proof is ready from package consumer smoke." ; break }
    "blocked-by-cuda-driver" { "package consumer smoke reached the packaged runtime but CUDA driver/runtime compatibility blocked execution; this is owner-action-required and is not smoke passed." ; break }
    "blocked-by-application-control" { "package consumer smoke was blocked by Windows application control policy; this is owner-action-required and is not smoke passed." ; break }
    "not-requested" { "full package consumer runtime smoke has not been requested; build/native-copy evidence is not runtime proof." ; break }
    default { "runtime proof is incomplete; inspect package consumer smoke and runtime readiness evidence." ; break }
  }

  $externalInputRequired = [string]$RuntimeProofStatus -ne "ready"
  $suggestedCommands = @(
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 -RuntimePackageKey $Key",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Materialize-WindowsVendorRuntimeAssets.ps1 -RuntimePackageKey $Key -DryRun",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalRuntimePackage.ps1 -RuntimePackageKey $Key",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $Key -RunSmoke -AllowSmokeFailure",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordTemplate.ps1 -RuntimePackageKey $Key",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1"
  )

  $externalInputs = @(
    "compatible NVIDIA driver for the selected CUDA runtime",
    "CUDA runtime assets matching the runtime package key",
    "TensorRT runtime assets matching the runtime package key",
    "cuDNN runtime assets when the package line requires cuDNN",
    "GPU host allowed to execute package-consumer runtime smoke"
  )

  $evidencePaths = @(
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/package-readiness/runtime-package-readiness-summary.json",
    "artifacts/final-release/external-runtime-proof-record-template.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )

  return [pscustomobject]@{
    status = if ([string]$RuntimeProofStatus -eq "ready") { "resolved" } else { "owner-action-required" }
    blockerCategory = $blockerCategory
    runtimeProofStatus = $RuntimeProofStatus
    runtimeProofRequiredForRelease = $RuntimeProofRequiredForRelease
    runtimeProofDiagnostic = $RuntimeProofDiagnostic
    packageConsumerStatus = [string]$FullPackageConsumer.status
    packageConsumerSmokeResult = [string]$FullPackageConsumer.smokeResult
    packageConsumerSmokeRequested = [bool]$FullPackageConsumer.smokeRequested
    packageConsumerRuntimeSmokeClassification = $classification
    packageConsumerEvidenceKind = [string]$FullPackageConsumer.evidenceKind
    packageConsumerIsRuntimeExecutionEvidence = [bool]$FullPackageConsumer.isRuntimeExecutionEvidence
    packageConsumerIsDependencyProbeOnly = [bool]$FullPackageConsumer.isDependencyProbeOnly
    packageConsumerIsRealCallbackRuntimeProof = [bool]$FullPackageConsumer.isRealCallbackRuntimeProof
    packageConsumerReportPath = [string]$FullPackageConsumer.reportPath
    externalInputRequired = $externalInputRequired
    externalInputs = @($externalInputs)
    suggestedCommands = @($suggestedCommands)
    evidencePaths = @($evidencePaths)
    summary = $summary
    whyNotSmokePassed = "blocked-by-cuda-driver, dependency-probe-only, build-only, precheck, design-gate, and allowRuntimeSmokeBlocked are not runtime execution proof."
    nextAction = "Run the suggested command sequence on a CUDA-compatible host, attach a promotable package-consumer-runtime external proof record, then refresh release evidence."
    runtimeExecutionConsumers = @($RuntimeExecutionEvidence.consumers)
  }
}

function New-DeferredReadOnlyDesignGateEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Marker,
    [Parameter(Mandatory = $true)]
    [string]$ReadyStatus,
    [Parameter(Mandatory = $true)]
    [string[]]$EvidenceRelativePaths,
    [Parameter(Mandatory = $true)]
    [string[]]$RequiredMarkers,
    [Parameter(Mandatory = $true)]
    [string[]]$RequiredDeferredRows,
    [Parameter(Mandatory = $true)]
    [string]$ReadyDiagnostic,
    [Parameter(Mandatory = $true)]
    [string]$IncompleteDiagnostic,
    [Parameter(Mandatory = $true)]
    [hashtable]$ReadyProperties,
    [Parameter(Mandatory = $true)]
    [hashtable]$MissingProperties,
    [string]$RuntimeEvidenceKind = "design-gate"
  )

  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $relativePaths = @($EvidenceRelativePaths) + @($comparisonRelativePath)
  $evidencePaths = @($relativePaths | ForEach-Object { Join-Path $RepositoryRoot $_ })
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    $missingResult = [ordered]@{
      status = "missing"
      marker = $Marker
      evidenceKind = $Marker
      runtimeEvidenceKind = $RuntimeEvidenceKind
      source = "source-smoke-docs-coverage"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($RequiredMarkers)
      missingDeferredRows = @($RequiredDeferredRows)
      hasDeferredRowEvidence = $false
      isRuntimeExecutionEvidence = $false
      isRuntimeExecutionProof = $false
      canPromoteRuntimeProof = $false
      runtimeProofBlocked = $true
      deferredRowsStillRequired = $true
      diagnostic = "$Marker evidence files are missing."
    }
    foreach ($key in @($MissingProperties.Keys)) {
      $missingResult[$key] = $MissingProperties[$key]
    }

    return [pscustomobject]$missingResult
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($requiredMarker in @($RequiredMarkers)) {
    if ($combined.IndexOf($requiredMarker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($requiredMarker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($RequiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { $ReadyStatus } else { "incomplete" }
  $diagnostic = if ($status -eq $ReadyStatus) { $ReadyDiagnostic } else { $IncompleteDiagnostic }

  $result = [ordered]@{
    status = $status
    marker = $Marker
    evidenceKind = $Marker
    runtimeEvidenceKind = $RuntimeEvidenceKind
    source = "source-smoke-docs-coverage"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRuntimeExecutionEvidence = $false
    isRuntimeExecutionProof = $false
    canPromoteRuntimeProof = $false
    runtimeProofBlocked = $true
    deferredRowsStillRequired = $true
    diagnostic = $diagnostic
  }
  foreach ($key in @($ReadyProperties.Keys)) {
    $result[$key] = $ReadyProperties[$key]
  }

  return [pscustomobject]$result
}

function New-ErrorRecorderDiagnosticsDesignGateEvidence {
  $requiredMarkers = @(
    "error-recorder-diagnostics-design-gate",
    "TensorRtErrorRecorderDiagnosticsDesignGate",
    "TensorRtErrorRecorderDiagnosticsDesignGateResult",
    "RuntimeEvidenceKind=design-gate",
    "IsRuntimeExecutionEvidence=False",
    "IsRuntimeExecutionProof=False",
    "CopiedDiagnosticsReady=True",
    "PointerFreeSurfaceReady=True",
    "RecorderPointerExposed=False",
    "RefCountPublicOwnershipControl=False",
    "RuntimeProofBlocked=True",
    "DeferredRowsStillRequired=True",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IErrorRecorder","getNbErrors","IErrorRecorder::getNbErrors","diagnostics","implemented-with-deferred-history"',
    '"IErrorRecorder","getErrorCode","IErrorRecorder::getErrorCode","diagnostics","implemented-with-deferred-history"',
    '"IErrorRecorder","getErrorDesc","IErrorRecorder::getErrorDesc","diagnostics","implemented-with-deferred-history"',
    "error-recorder-get-nb-errors-deferred",
    "error-recorder-get-error-code-deferred",
    "error-recorder-get-error-desc-deferred"
  )

  return New-DeferredReadOnlyDesignGateEvidence `
    -Marker "error-recorder-diagnostics-design-gate" `
    -ReadyStatus "design-gate-ready" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtErrorRecorderDiagnosticsDesignGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtErrorRecorderSnapshot.cs",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\error-recorder-diagnostics-design-gate.md",
      "docs\articles\zh-cn\error-recorder-snapshot-guide.md",
      "docs\articles\zh-cn\deferred-manual-design-groups.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml"
    ) `
    -RequiredMarkers $requiredMarkers `
    -RequiredDeferredRows $requiredDeferredRows `
    -ReadyDiagnostic "error recorder diagnostics design gate is documented, pointer-free, and explicitly not runtime execution proof." `
    -IncompleteDiagnostic "error recorder diagnostics design gate evidence is incomplete; inspect source/smoke/doc markers and deferred row evidence." `
    -ReadyProperties @{
      copiedDiagnosticsReady = $true
      pointerFreeSurfaceReady = $true
      recorderPointerExposed = $false
      refCountPublicOwnershipControl = $false
    } `
    -MissingProperties @{
      copiedDiagnosticsReady = $false
      pointerFreeSurfaceReady = $false
      recorderPointerExposed = $false
      refCountPublicOwnershipControl = $false
    }
}

function New-DimensionExpressionSnapshotDesignGateEvidence {
  $requiredMarkers = @(
    "dimension-expression-snapshot-design-gate",
    "TensorRtDimensionExpressionSnapshotDesignGate",
    "TensorRtDimensionExpressionSnapshotDesignGateResult",
    "RuntimeEvidenceKind=design-gate",
    "IsRuntimeExecutionEvidence=False",
    "IsRuntimeExecutionProof=False",
    "SnapshotTypeReady=True",
    "OwnerLifetimeKnown=False",
    "ExpressionPointerExposed=False",
    "BorrowedExpressionPointerEscaped=False",
    "ExprBuilderCreationEnabled=False",
    "DirectDimensionExpressionRowsDeferred=True",
    "DirectExpressionBuilderRowsDeferred=True",
    "CanPromoteWithoutRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DeferredRowsStillRequired=True",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDimensionExpr","getConstantValue","IDimensionExpr::getConstantValue","other","deferred-only"',
    '"IDimensionExpr","isConstant","IDimensionExpr::isConstant","other","deferred-only"',
    '"IDimensionExpr","isSizeTensor","IDimensionExpr::isSizeTensor","other","deferred-only"',
    '"IExprBuilder","constant","IExprBuilder::constant","builder","deferred-only"',
    '"IExprBuilder","declareSizeTensor","IExprBuilder::declareSizeTensor","builder","deferred-only"',
    '"IExprBuilder","operation","IExprBuilder::operation","builder","deferred-only"'
  )

  return New-DeferredReadOnlyDesignGateEvidence `
    -Marker "dimension-expression-snapshot-design-gate" `
    -ReadyStatus "design-gate-ready" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDimensionExpressionSnapshotDesignGate.cs",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\dimension-expression-snapshot-design-gate.md",
      "docs\articles\zh-cn\deferred-manual-design-groups.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml"
    ) `
    -RequiredMarkers $requiredMarkers `
    -RequiredDeferredRows $requiredDeferredRows `
    -ReadyDiagnostic "dimension expression snapshot design gate is documented, pointer-free, and explicitly not runtime execution proof." `
    -IncompleteDiagnostic "dimension expression snapshot design gate evidence is incomplete; inspect source/smoke/doc markers and deferred row evidence." `
    -ReadyProperties @{
      snapshotTypeReady = $true
      ownerLifetimeKnown = $false
      expressionPointerExposed = $false
      exprBuilderCreationEnabled = $false
      pointerFreeSurfaceReady = $true
    } `
    -MissingProperties @{
      snapshotTypeReady = $false
      ownerLifetimeKnown = $false
      expressionPointerExposed = $false
      exprBuilderCreationEnabled = $false
      pointerFreeSurfaceReady = $false
    }
}

function New-CalibratorMetadataDesignGateEvidence {
  $requiredMarkers = @(
    "calibrator-metadata-design-gate",
    "TensorRtCalibratorMetadataDesignGate",
    "TensorRtCalibratorMetadataDesignGateResult",
    "RuntimeEvidenceKind=design-gate",
    "IsRuntimeExecutionEvidence=False",
    "IsRuntimeExecutionProof=False",
    "PresenceProbeAvailable=True",
    "CopiedMetadataShapeReady=True",
    "PointerFreeSurfaceReady=True",
    "CalibratorPointerExposed=False",
    "BorrowedCalibratorPointerEscaped=False",
    "CallbackInvocationEnabled=False",
    "BatchBufferAccessEnabled=False",
    "CacheBufferAccessEnabled=False",
    "DirectCalibratorCallbackRowsDeferred=True",
    "DirectCalibratorCacheRowsDeferred=True",
    "CanPromoteWithoutRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DeferredRowsStillRequired=True",
    "not proof"
  )

  $requiredDeferredRows = @(
    "IInt8Calibrator::getAlgorithm",
    "IInt8Calibrator::getBatch",
    "IInt8Calibrator::getBatchSize",
    "IInt8EntropyCalibrator::getAlgorithm",
    "IInt8EntropyCalibrator2::getAlgorithm",
    "IInt8LegacyCalibrator::getQuantile",
    "IInt8LegacyCalibrator::getRegressionCutoff",
    "int8-calibrator-read-calibration-cache-deferred",
    "int8-calibrator-write-calibration-cache-deferred"
  )

  return New-DeferredReadOnlyDesignGateEvidence `
    -Marker "calibrator-metadata-design-gate" `
    -ReadyStatus "design-gate-ready" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtCalibratorMetadataDesignGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtBuilderConfig.cs",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\calibrator-metadata-design-gate.md",
      "docs\articles\zh-cn\deferred-manual-design-groups.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml"
    ) `
    -RequiredMarkers $requiredMarkers `
    -RequiredDeferredRows $requiredDeferredRows `
    -ReadyDiagnostic "calibrator metadata design gate is documented, pointer-free, presence-only, and explicitly not runtime execution proof." `
    -IncompleteDiagnostic "calibrator metadata design gate evidence is incomplete; inspect source/smoke/doc markers and deferred row evidence." `
    -ReadyProperties @{
      presenceProbeAvailable = $true
      copiedMetadataShapeReady = $true
      pointerFreeSurfaceReady = $true
      calibratorPointerExposed = $false
      callbackInvocationEnabled = $false
      batchBufferAccessEnabled = $false
      cacheBufferAccessEnabled = $false
    } `
    -MissingProperties @{
      presenceProbeAvailable = $false
      copiedMetadataShapeReady = $false
      pointerFreeSurfaceReady = $false
      calibratorPointerExposed = $false
      callbackInvocationEnabled = $false
      batchBufferAccessEnabled = $false
      cacheBufferAccessEnabled = $false
    }
}

function New-RuntimeDeserializationBoundaryPrecheckEvidence {
  $requiredMarkers = @(
    "runtime-deserialization-boundary-precheck",
    "TensorRtRuntimeDeserializationBoundaryPrecheck",
    "TensorRtRuntimeDeserializationBoundaryPrecheckResult",
    "RuntimeEvidenceKind=runtime-precheck",
    "IsRuntimeExecutionEvidence=False",
    "IsRuntimeExecutionProof=False",
    "ManagedByteArrayDeserializeReady=True",
    "ManagedStreamDeserializeReady=True",
    "HostMemoryDeserializeReady=True",
    "SerializedBufferCopiedBeforeInterop=True",
    "PinnedBufferScopedToInteropCall=True",
    "BorrowedSerializedBufferEscaped=False",
    "EngineHandleOwnedByWrapper=True",
    "EnginePointerExposed=False",
    "DirectDeserializeCudaEngineRowsDeferred=True",
    "DirectDeserializeCudaEngineV2RowsDeferred=True",
    "LoadRuntimeDeferred=True",
    "CanAttemptRuntimeProof=False",
    "CanPromoteWithoutRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DeferredRowsStillRequired=True",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IRuntime","deserializeCudaEngineV2","IRuntime::deserializeCudaEngineV2","runtime-serialization","deferred-only"',
    '"IRuntime","loadRuntime","IRuntime::loadRuntime","runtime-serialization","deferred-only"',
    "trt10-runtime-deserialize-cuda-engine-v2-deferred",
    "trt11-runtime-deserialize-cuda-engine-v2-deferred",
    "trt8-runtime-load-runtime-deferred",
    "trt10-runtime-load-runtime-deferred",
    "trt11-runtime-load-runtime-deferred"
  )

  return New-DeferredReadOnlyDesignGateEvidence `
    -Marker "runtime-deserialization-boundary-precheck" `
    -ReadyStatus "precheck-ready" `
    -RuntimeEvidenceKind "runtime-precheck" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtRuntimeDeserializationBoundaryPrecheck.cs",
      "src\JYPPX.TensorRtSharp\TensorRtRuntime.cs",
      "src\JYPPX.TensorRtSharp\Internal\Interop\NativeBridgeApi.cs",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\runtime-deserialization-boundary-precheck.md",
      "docs\articles\zh-cn\deferred-manual-design-groups.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "native\manifests\tensorrt\v8\trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json",
      "native\manifests\tensorrt\v10\trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json",
      "native\manifests\tensorrt\v11\trt11-twenty-third-batch-deferred-coverage.manifest.json",
      "native\src\tensorrt\v10\modules\deferred\cross_version_runtime_serialization_deferred.inc",
      "native\src\tensorrt\v11\modules\deferred\twenty_third_batch_deferred.inc"
    ) `
    -RequiredMarkers $requiredMarkers `
    -RequiredDeferredRows $requiredDeferredRows `
    -ReadyDiagnostic "runtime deserialization boundary precheck documents the safe managed Deserialize surface, keeps direct runtime serialization rows deferred, and is explicitly not runtime execution proof." `
    -IncompleteDiagnostic "runtime deserialization boundary precheck evidence is incomplete; inspect source/smoke/doc markers and deferred row evidence." `
    -ReadyProperties @{
      managedByteArrayDeserializeReady = $true
      managedStreamDeserializeReady = $true
      hostMemoryDeserializeReady = $true
      serializedBufferCopiedBeforeInterop = $true
      borrowedSerializedBufferEscaped = $false
      engineHandleOwnedByWrapper = $true
      enginePointerExposed = $false
      loadRuntimeDeferred = $true
      pointerFreeSurfaceReady = $true
      safeDeserializeBridgeReady = $true
    } `
    -MissingProperties @{
      managedByteArrayDeserializeReady = $false
      managedStreamDeserializeReady = $false
      hostMemoryDeserializeReady = $false
      serializedBufferCopiedBeforeInterop = $false
      borrowedSerializedBufferEscaped = $false
      engineHandleOwnedByWrapper = $false
      enginePointerExposed = $false
      loadRuntimeDeferred = $true
      pointerFreeSurfaceReady = $false
      safeDeserializeBridgeReady = $false
    }
}

function New-RuntimeDeserializationDependencyDiagnosticsEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [object]$BoundaryPrecheck,
    [Parameter(Mandatory = $true)]
    [object]$FullPackageConsumer
  )

  $requiredMarkers = @(
    "runtime-deserialization-dependency-diagnostics",
    "TensorRtRuntimeDeserializationDependencyDiagnostics",
    "TensorRtRuntimeDeserializationDependencyDiagnosticsResult",
    "RuntimeEvidenceKind=dependency-diagnostics",
    "IsRuntimeExecutionEvidence=False",
    "IsRuntimeExecutionProof=False",
    "PrecheckReady=True",
    "ManagedDeserializeSurfaceReady=True",
    "FullPackageConsumerReportPresent",
    "FullPackageConsumerSmokeRequested",
    "FullPackageConsumerSmokeResult",
    "DependencyProbeOnly",
    "BlockedByCudaDriver",
    "PluginLibraryDependencyDiagnosticsComplete=False",
    "LoadRuntimeOwnershipModeled=False",
    "CanAttemptRuntimeProof=False",
    "CanPromoteRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DeferredRowsStillRequired=True",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IRuntime","deserializeCudaEngineV2","IRuntime::deserializeCudaEngineV2","runtime-serialization","deferred-only"',
    '"IRuntime","loadRuntime","IRuntime::loadRuntime","runtime-serialization","deferred-only"',
    "trt10-runtime-deserialize-cuda-engine-v2-deferred",
    "trt11-runtime-deserialize-cuda-engine-v2-deferred",
    "trt8-runtime-load-runtime-deferred",
    "trt10-runtime-load-runtime-deferred",
    "trt11-runtime-load-runtime-deferred"
  )

  $fullPackageConsumerReportPresent = [string]$FullPackageConsumer.status -eq "ready"
  $fullPackageConsumerSmokeRequested = [bool]$FullPackageConsumer.smokeRequested
  $fullPackageConsumerSmokeResult = if ([string]::IsNullOrWhiteSpace([string]$FullPackageConsumer.smokeResult)) { "not-present" } else { [string]$FullPackageConsumer.smokeResult }
  $dependencyProbeOnly = if ($FullPackageConsumer.PSObject.Properties.Name -contains "isDependencyProbeOnly") { [bool]$FullPackageConsumer.isDependencyProbeOnly } else { $true }
  $blockedByCudaDriver = [string]::Equals($fullPackageConsumerSmokeResult, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -or
    [string]::Equals([string]$FullPackageConsumer.runtimeSmokeClassification, "runtime-smoke-driver-blocked", [System.StringComparison]::OrdinalIgnoreCase)
  $driverRuntimeMismatchClassified = $blockedByCudaDriver
  $packageConsumerEvidenceClassification = if ($blockedByCudaDriver) {
    "runtime-smoke-driver-blocked"
  }
  elseif ($dependencyProbeOnly) {
    "dependency-probe-only"
  }
  elseif (-not $fullPackageConsumerSmokeRequested) {
    "runtime-smoke-not-requested"
  }
  else {
    "runtime-proof-incomplete"
  }

  $runtimeProofBlockerCategory = if (-not [bool]$BoundaryPrecheck.safeDeserializeBridgeReady) {
    "runtime-deserialization-precheck-incomplete"
  }
  elseif (-not $fullPackageConsumerReportPresent) {
    "full-package-consumer-report-missing"
  }
  elseif (-not $fullPackageConsumerSmokeRequested) {
    "runtime-smoke-not-requested"
  }
  elseif ($blockedByCudaDriver) {
    "cuda-driver-runtime-compatibility"
  }
  elseif ($dependencyProbeOnly) {
    "dependency-probe-only"
  }
  else {
    "plugin-library-dependency-diagnostics-incomplete"
  }

  $nextOwnerAction = switch ($runtimeProofBlockerCategory) {
    "cuda-driver-runtime-compatibility" { "Run full package consumer smoke on a host with a compatible NVIDIA driver, then attach a promotable package-consumer-runtime external proof record."; break }
    "runtime-smoke-not-requested" { "Run Test-PackageConsumer.ps1 with -RunSmoke for the selected runtime package key and refresh runtime readiness evidence."; break }
    "full-package-consumer-report-missing" { "Generate the full package consumer report before evaluating runtime proof."; break }
    "dependency-probe-only" { "Replace dependency-probe-only evidence with successful full package consumer runtime smoke evidence."; break }
    "plugin-library-dependency-diagnostics-incomplete" { "Complete plugin library dependency diagnostics before attempting runtime proof promotion."; break }
    default { "Inspect blocked prerequisites, refresh package consumer evidence, and provide a promotable package-consumer-runtime proof record."; break }
  }

  $whyNotRuntimeProof = "runtime deserialization dependency diagnostics, dependency-probe-only output, blocked-by-cuda-driver, precheck, design-gate, build-only, and deferred loadRuntime ownership are not runtime execution proof."

  $evidence = New-DeferredReadOnlyDesignGateEvidence `
    -Marker "runtime-deserialization-dependency-diagnostics" `
    -ReadyStatus "dependency-diagnostics-ready" `
    -RuntimeEvidenceKind "dependency-diagnostics" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtRuntimeDeserializationDependencyDiagnostics.cs",
      "src\JYPPX.TensorRtSharp\TensorRtRuntimeDeserializationBoundaryPrecheck.cs",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\runtime-deserialization-dependency-diagnostics.md",
      "docs\articles\zh-cn\runtime-deserialization-boundary-precheck.md",
      "docs\articles\zh-cn\deferred-manual-design-groups.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "eng\Test-RuntimePackageReadiness.ps1",
      "eng\Export-ReleaseEvidenceBundle.ps1",
      "native\manifests\tensorrt\v8\trt8-cross-version-seventh-batch-deferred-boundaries.manifest.json",
      "native\manifests\tensorrt\v10\trt10-cross-version-fifth-batch-runtime-serialization-deferred.manifest.json",
      "native\manifests\tensorrt\v11\trt11-twenty-third-batch-deferred-coverage.manifest.json"
    ) `
    -RequiredMarkers $requiredMarkers `
    -RequiredDeferredRows $requiredDeferredRows `
    -ReadyDiagnostic "runtime deserialization dependency diagnostics distinguish dependency-probe-only, blocked-by-cuda-driver, and package-consumer runtime proof states while keeping direct deserializeCudaEngineV2/loadRuntime rows deferred." `
    -IncompleteDiagnostic "runtime deserialization dependency diagnostics evidence is incomplete; inspect source/smoke/doc/release markers and deferred row evidence." `
    -ReadyProperties @{
      precheckReady = [bool]$BoundaryPrecheck.safeDeserializeBridgeReady
      managedDeserializeSurfaceReady = [bool]$BoundaryPrecheck.managedByteArrayDeserializeReady -and [bool]$BoundaryPrecheck.managedStreamDeserializeReady
      fullPackageConsumerReportPresent = $fullPackageConsumerReportPresent
      fullPackageConsumerSmokeRequested = $fullPackageConsumerSmokeRequested
      fullPackageConsumerSmokeResult = $fullPackageConsumerSmokeResult
      dependencyProbeOnly = $dependencyProbeOnly
      blockedByCudaDriver = $blockedByCudaDriver
      driverRuntimeMismatchClassified = $driverRuntimeMismatchClassified
      packageConsumerEvidenceClassification = $packageConsumerEvidenceClassification
      runtimeProofBlockerCategory = $runtimeProofBlockerCategory
      runtimeProofOwnerActionRequired = $true
      externalRuntimeProofRequired = $true
      packageConsumerRuntimeProofPresent = $false
      whyNotRuntimeProof = $whyNotRuntimeProof
      nextOwnerAction = $nextOwnerAction
      pluginLibraryDependencyDiagnosticsComplete = $false
      loadRuntimeOwnershipModeled = $false
      canAttemptRuntimeProof = $false
      canPromoteRuntimeProof = $false
      pointerFreeSurfaceReady = $true
    } `
    -MissingProperties @{
      precheckReady = $false
      managedDeserializeSurfaceReady = $false
      fullPackageConsumerReportPresent = $false
      fullPackageConsumerSmokeRequested = $false
      fullPackageConsumerSmokeResult = "not-present"
      dependencyProbeOnly = $true
      blockedByCudaDriver = $false
      driverRuntimeMismatchClassified = $false
      packageConsumerEvidenceClassification = "dependency-probe-only"
      runtimeProofBlockerCategory = "runtime-deserialization-precheck-incomplete"
      runtimeProofOwnerActionRequired = $true
      externalRuntimeProofRequired = $true
      packageConsumerRuntimeProofPresent = $false
      whyNotRuntimeProof = $whyNotRuntimeProof
      nextOwnerAction = "Inspect blocked prerequisites, refresh package consumer evidence, and provide a promotable package-consumer-runtime proof record."
      pluginLibraryDependencyDiagnosticsComplete = $false
      loadRuntimeOwnershipModeled = $false
      canAttemptRuntimeProof = $false
      canPromoteRuntimeProof = $false
      pointerFreeSurfaceReady = $false
    }

  $evidence | Add-Member -NotePropertyName "fullPackageConsumerReportPresent" -NotePropertyValue $fullPackageConsumerReportPresent -Force
  $evidence | Add-Member -NotePropertyName "fullPackageConsumerSmokeRequested" -NotePropertyValue $fullPackageConsumerSmokeRequested -Force
  $evidence | Add-Member -NotePropertyName "fullPackageConsumerSmokeResult" -NotePropertyValue $fullPackageConsumerSmokeResult -Force
  $evidence | Add-Member -NotePropertyName "dependencyProbeOnly" -NotePropertyValue $dependencyProbeOnly -Force
  $evidence | Add-Member -NotePropertyName "blockedByCudaDriver" -NotePropertyValue $blockedByCudaDriver -Force
  $evidence | Add-Member -NotePropertyName "driverRuntimeMismatchClassified" -NotePropertyValue $driverRuntimeMismatchClassified -Force
  $evidence | Add-Member -NotePropertyName "packageConsumerEvidenceClassification" -NotePropertyValue $packageConsumerEvidenceClassification -Force
  $evidence | Add-Member -NotePropertyName "runtimeProofBlockerCategory" -NotePropertyValue $runtimeProofBlockerCategory -Force
  $evidence | Add-Member -NotePropertyName "runtimeProofOwnerActionRequired" -NotePropertyValue $true -Force
  $evidence | Add-Member -NotePropertyName "externalRuntimeProofRequired" -NotePropertyValue $true -Force
  $evidence | Add-Member -NotePropertyName "packageConsumerRuntimeProofPresent" -NotePropertyValue $false -Force
  $evidence | Add-Member -NotePropertyName "pluginLibraryDependencyDiagnosticsComplete" -NotePropertyValue $false -Force
  $evidence | Add-Member -NotePropertyName "loadRuntimeOwnershipModeled" -NotePropertyValue $false -Force
  $evidence | Add-Member -NotePropertyName "canAttemptRuntimeProof" -NotePropertyValue $false -Force
  $evidence | Add-Member -NotePropertyName "canPromoteRuntimeProof" -NotePropertyValue $false -Force
  $evidence | Add-Member -NotePropertyName "whyNotRuntimeProof" -NotePropertyValue $whyNotRuntimeProof -Force
  $evidence | Add-Member -NotePropertyName "nextOwnerAction" -NotePropertyValue $nextOwnerAction -Force
  return $evidence
}

function New-RealCallbackRuntimeEvidenceSchema {
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $gateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $tocRelativePath = "docs\toc.yml"
  $indexRelativePath = "docs\index.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $gatePath = Join-Path $RepositoryRoot $gateRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath

  $requiredSmokeMarkers = @(
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

  $requiredMarkers = @(
    "real-callback-runtime-evidence-schema",
    "schema-only",
    "not-present",
    "realCallbackRuntimeEvidenceSchema",
    "realCallbackRuntimeEvidence",
    "RealCallbackRuntimeEvidence",
    "RealCallbackRuntimeEvidence.Status",
    "callbackRuntimeEvidenceStatus",
    "blocked-by-cuda-driver",
    "debug-listener-real-non-null-attach-runtime-smoke",
    "debug-listener-process-debug-tensor-callback-trampoline",
    "debug-listener-callback-proof-gap-report",
    "callback-trampoline-shape",
    "proof-gap-report",
    "callback-owner-closure-matrix",
    "runtime-smoke-skipped",
    "runtime-smoke-blocked",
    "runtime-smoke-attempted",
    "runtime-smoke-failed",
    "isRealCallbackRuntimeProof=false",
    "SmokeResult=passed",
    "dry-run",
    "copied-state",
    "bridge-only wrapper surface",
    "dependency probe"
  ) + $requiredSmokeMarkers

  $evidencePaths = @($schemaPath, $gatePath, $tocPath, $indexPath, $latestPath, $runtimeSplitReadmePath, $packageConsumerPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "real-callback-runtime-evidence-schema"
      evidenceKind = "schema-only"
      runtimeEvidenceKind = "not-present"
      schemaPath = $schemaPath
      evidencePaths = @($evidencePaths)
      requiredSmokeMarkers = @($requiredSmokeMarkers)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      hasSchemaDocument = $false
      hasDocfxLinks = $false
      hasPackageConsumerNote = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "real callback runtime evidence schema files are missing."
    }
  }

  $schema = Get-Content -LiteralPath $schemaPath -Raw -Encoding utf8
  $gate = Get-Content -LiteralPath $gatePath -Raw -Encoding utf8
  $toc = Get-Content -LiteralPath $tocPath -Raw -Encoding utf8
  $index = Get-Content -LiteralPath $indexPath -Raw -Encoding utf8
  $latest = Get-Content -LiteralPath $latestPath -Raw -Encoding utf8
  $runtimeSplitReadme = Get-Content -LiteralPath $runtimeSplitReadmePath -Raw -Encoding utf8
  $packageConsumer = Get-Content -LiteralPath $packageConsumerPath -Raw -Encoding utf8
  $combined = $schema + "`n" + $gate + "`n" + $toc + "`n" + $index + "`n" + $latest + "`n" + $runtimeSplitReadme + "`n" + $packageConsumer

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $hasDocfxLinks = $toc.Contains("real-callback-runtime-evidence-schema.md") -and
    $index.Contains("real-callback-runtime-evidence-schema.md") -and
    $latest.Contains("real-callback-runtime-evidence-schema.md") -and
    $gate.Contains("real-callback-runtime-evidence-schema.md")
  $hasPackageConsumerNote = $packageConsumer.Contains("EvidenceKind=real-callback-runtime") -and
    $packageConsumer.Contains("isRealCallbackRuntimeProof=true") -and
    $runtimeSplitReadme.Contains("realCallbackRuntimeEvidenceSchema") -and
    $runtimeSplitReadme.Contains("realCallbackRuntimeEvidence")
  $hasSchemaDocument = $missingMarkers.Count -eq 0
  $status = if ($hasSchemaDocument -and $hasDocfxLinks -and $hasPackageConsumerNote) { "schema-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "schema-ready") {
    "real callback runtime evidence schema is documented, linked, and package-consumer/readiness markers are in place."
  }
  else {
    "real callback runtime evidence schema is incomplete; inspect missing markers, DocFX links, or package-consumer notes."
  }

  return [pscustomobject]@{
    status = $status
    marker = "real-callback-runtime-evidence-schema"
    evidenceKind = "schema-only"
    runtimeEvidenceKind = "not-present"
    schemaPath = $schemaPath
    evidencePaths = @($evidencePaths)
    requiredSmokeMarkers = @($requiredSmokeMarkers)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    hasSchemaDocument = $hasSchemaDocument
    hasDocfxLinks = $hasDocfxLinks
    hasPackageConsumerNote = $hasPackageConsumerNote
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-RealCallbackRuntimeEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [object]$FullPackageConsumer,
    [Parameter(Mandatory = $true)]
    [object]$Schema
  )

  $requiredSmokeMarkers = @($Schema.requiredSmokeMarkers)
  $smokeOutputLines = @($FullPackageConsumer.smokeOutputLines | ForEach-Object { [string]$_ })
  $combinedSmokeOutput = $smokeOutputLines -join "`n"
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
  if ($FullPackageConsumer.PSObject.Properties.Name -contains "callbackRuntimeEvidenceStatus" -and
    -not [string]::IsNullOrWhiteSpace([string]$FullPackageConsumer.callbackRuntimeEvidenceStatus)) {
    $explicitStatus = [string]$FullPackageConsumer.callbackRuntimeEvidenceStatus
    $explicitEvidenceKind = if ([string]::IsNullOrWhiteSpace([string]$FullPackageConsumer.callbackRuntimeEvidenceKind)) { "not-present" } else { [string]$FullPackageConsumer.callbackRuntimeEvidenceKind }
    $explicitRuntimeEvidenceKind = if ([string]::IsNullOrWhiteSpace([string]$FullPackageConsumer.callbackRuntimeRuntimeEvidenceKind)) { $explicitEvidenceKind } else { [string]$FullPackageConsumer.callbackRuntimeRuntimeEvidenceKind }
    $explicitRequiredMarkers = if (@($FullPackageConsumer.callbackRuntimeRequiredMarkers).Count -gt 0) { @($FullPackageConsumer.callbackRuntimeRequiredMarkers | ForEach-Object { [string]$_ }) } else { @($requiredSmokeMarkers) }
    $explicitMissingMarkers = @($FullPackageConsumer.callbackRuntimeMissingMarkers | ForEach-Object { [string]$_ })
    $explicitMatchedLines = @($FullPackageConsumer.callbackRuntimeMatchedLines | ForEach-Object { [string]$_ })
    $explicitProof = [bool]$FullPackageConsumer.callbackRuntimeIsProof
    $isSmokePassed = [string]$FullPackageConsumer.smokeResult -eq "passed"
    $hasExplicitRealCallbackRuntimeKind =
      [string]::Equals($explicitEvidenceKind, "real-callback-runtime", [System.StringComparison]::OrdinalIgnoreCase) -and
      [string]::Equals($explicitRuntimeEvidenceKind, "real-callback-runtime", [System.StringComparison]::OrdinalIgnoreCase)
    $explicitHasBlockingNonProofRuntimeKind =
      $hasBlockingNonProofRuntimeKind -or
      $explicitRuntimeEvidenceKind -eq "proof-gap-report" -or
      $explicitRuntimeEvidenceKind -eq "closure-matrix" -or
      $explicitRuntimeEvidenceKind -eq "runtime-smoke-skipped" -or
      $explicitRuntimeEvidenceKind -eq "runtime-smoke-blocked" -or
      $explicitRuntimeEvidenceKind -eq "runtime-smoke-attempted" -or
      $explicitRuntimeEvidenceKind -eq "runtime-smoke-failed" -or
      $explicitRuntimeEvidenceKind -eq "callback-trampoline-shape" -or
      $explicitRuntimeEvidenceKind -eq "real-callback-runtime-blocked" -or
      $explicitRuntimeEvidenceKind -eq "attempted-no-invocation"
    if ($explicitHasBlockingNonProofRuntimeKind -and -not ($explicitMissingMarkers -contains "NoNonProofCallbackRuntimeMarker")) {
      $explicitMissingMarkers += "NoNonProofCallbackRuntimeMarker"
    }
    $isReady = $explicitStatus -eq "ready" -and $explicitProof -and $isSmokePassed -and $explicitMissingMarkers.Count -eq 0 -and $hasExplicitRealCallbackRuntimeKind -and -not $explicitHasBlockingNonProofRuntimeKind
    $status = if ($explicitStatus -eq "ready" -and -not $isReady) { "incomplete" } else { $explicitStatus }
    $evidenceKind = if ($isReady) { "real-callback-runtime" } elseif ($explicitStatus -eq "ready") { "incomplete-real-callback-runtime" } else { $explicitEvidenceKind }
    $runtimeEvidenceKind = if ($isReady) { "real-callback-runtime" } elseif ($explicitStatus -eq "ready") { "incomplete-real-callback-runtime" } else { $explicitRuntimeEvidenceKind }
    $diagnostic = if (-not [string]::IsNullOrWhiteSpace([string]$FullPackageConsumer.callbackRuntimeDiagnostic)) {
      [string]$FullPackageConsumer.callbackRuntimeDiagnostic
    }
    elseif ($isReady) {
      "full package consumer report carried complete real-callback-runtime evidence."
    }
    else {
      "full package consumer report carried callback runtime evidence status '$status'."
    }

    return [pscustomobject]@{
      status = $status
      marker = "real-callback-runtime"
      evidenceKind = $evidenceKind
      runtimeEvidenceKind = $runtimeEvidenceKind
      source = "full-package-consumer-report"
      smokeResult = [string]$FullPackageConsumer.smokeResult
      smokeRequested = [bool]$FullPackageConsumer.smokeRequested
      reportPath = [string]$FullPackageConsumer.reportPath
      requiredSmokeMarkers = @($explicitRequiredMarkers)
      missingSmokeMarkers = @($explicitMissingMarkers)
      matchedSmokeLines = @($explicitMatchedLines)
      isRealCallbackRuntimeProof = $isReady
      diagnostic = $diagnostic
    }
  }

  $hasRuntimeMarker = $combinedSmokeOutput.IndexOf("EvidenceKind=real-callback-runtime", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
  $runtimeSmokeLines = @($smokeOutputLines | Where-Object {
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

  if (-not $hasRuntimeMarker) {
    $status = switch ([string]$FullPackageConsumer.smokeResult) {
      "blocked-by-cuda-driver" { "blocked-by-cuda-driver"; break }
      "blocked-by-application-control" { "blocked-by-application-control"; break }
      "failed" { "blocked"; break }
      default {
        if ($runtimeSmokeLines.Count -gt 0) {
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-skipped", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "skipped"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-blocked", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "blocked"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-attempted", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "attempted"; break }
          if ($combinedSmokeOutput.IndexOf("RuntimeEvidenceKind=runtime-smoke-failed", [System.StringComparison]::OrdinalIgnoreCase) -ge 0) { "failed"; break }
          "blocked"
          break
        }

        "not-present"
        break
      }
    }
    $diagnostic = switch ($status) {
      "blocked-by-cuda-driver" { "full package consumer smoke reached the packaged runtime, but CUDA driver/runtime compatibility blocked real-callback-runtime evidence collection."; break }
      "blocked-by-application-control" { "full package consumer smoke was blocked by application control before real-callback-runtime evidence could be collected."; break }
      "blocked" { "full package consumer smoke failed without reporting real-callback-runtime evidence."; break }
      "skipped" { "full package consumer smoke reported debug-listener runtime smoke skipped evidence only; real-callback-runtime evidence is not present."; break }
      "attempted" { "full package consumer smoke reported debug-listener runtime smoke attempted evidence only; real-callback-runtime proof is not present."; break }
      "failed" { "full package consumer smoke reported debug-listener runtime smoke failed evidence only; real-callback-runtime proof is not present."; break }
      default { "full package consumer smoke did not report real-callback-runtime evidence."; break }
    }

    return [pscustomobject]@{
      status = $status
      marker = "real-callback-runtime"
      evidenceKind = "not-present"
      runtimeEvidenceKind = "not-present"
      source = "full-package-consumer"
      smokeResult = [string]$FullPackageConsumer.smokeResult
      smokeRequested = [bool]$FullPackageConsumer.smokeRequested
      reportPath = [string]$FullPackageConsumer.reportPath
      requiredSmokeMarkers = @($requiredSmokeMarkers)
      missingSmokeMarkers = @()
      matchedSmokeLines = @($runtimeSmokeLines)
      isRealCallbackRuntimeProof = $false
      diagnostic = $diagnostic
    }
  }

  $missingSmokeMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredSmokeMarkers)) {
    if ($combinedSmokeOutput.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingSmokeMarkers.Add($marker)
    }
  }
  if ($hasBlockingNonProofRuntimeKind) {
    $missingSmokeMarkers.Add("NoNonProofCallbackRuntimeMarker")
  }

  $matchedSmokeLines = @($smokeOutputLines | Where-Object {
      $_.IndexOf("real-callback-runtime", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("RealCallbackRuntime", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("CallbackKind", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("InvocationCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("AllocationCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("ReleaseCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $_.IndexOf("FailureCount", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    })

  $isSmokePassed = [string]$FullPackageConsumer.smokeResult -eq "passed"
  $isReady = $isSmokePassed -and $missingSmokeMarkers.Count -eq 0 -and -not $hasBlockingNonProofRuntimeKind
  $status = if ($isReady) { "ready" } else { "incomplete" }
  $diagnostic = if ($isReady) {
    "full package consumer smoke reported complete real-callback-runtime evidence."
  }
  elseif (-not $isSmokePassed) {
    "real-callback-runtime markers were found, but full package consumer smoke did not pass."
  }
  else {
    "real-callback-runtime markers were found, but required smoke fields are missing or non-proof callback runtime markers are present."
  }

  return [pscustomobject]@{
    status = $status
    marker = "real-callback-runtime"
    evidenceKind = if ($isReady) { "real-callback-runtime" } else { "incomplete-real-callback-runtime" }
    runtimeEvidenceKind = if ($isReady) { "real-callback-runtime" } else { "incomplete-real-callback-runtime" }
    source = "full-package-consumer"
    smokeResult = [string]$FullPackageConsumer.smokeResult
    smokeRequested = [bool]$FullPackageConsumer.smokeRequested
    reportPath = [string]$FullPackageConsumer.reportPath
    requiredSmokeMarkers = @($requiredSmokeMarkers)
    missingSmokeMarkers = @($missingSmokeMarkers.ToArray())
    matchedSmokeLines = @($matchedSmokeLines)
    isRealCallbackRuntimeProof = $isReady
    diagnostic = $diagnostic
  }
}

function New-AllocatorOwnerInternalRuntimePrototypeEvidence {
  $ownerRelativePath = "src\JYPPX.TensorRtSharp\TensorRtAllocatorCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $callbackDesignRelativePath = "docs\articles\zh-cn\allocator-callback-owner-design.md"
  $gateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $ownerPath = Join-Path $RepositoryRoot $ownerRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $callbackDesignPath = Join-Path $RepositoryRoot $callbackDesignRelativePath
  $gatePath = Join-Path $RepositoryRoot $gateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "allocator-owner-internal-runtime-prototype",
    "TensorRtAllocatorCallbackOwnerSnapshot",
    "RunLifecycleDiagnostic",
    "GetSnapshot",
    "RunInternalSyncAllocatorRuntimePrototype",
    "GetInternalRuntimePrototypeSnapshot",
    "TensorRtAllocatorInternalRuntimePrototypeResult",
    "TensorRtAllocatorInternalRuntimePrototypeCallback",
    "GCHandle.Alloc(_runtimePrototypeCallback)",
    "RuntimeEvidenceKind",
    "IsRealCallbackRuntimeProof",
    "DevicePointerExposed",
    "DevicePointerProduced",
    "BorrowedPointerEscaped",
    "ManagedKeepAliveReady",
    "DisposeReleaseReady",
    "PointerFreeSurfaceReady",
    "InFlightCallbackCount",
    "ReleaseHookCount",
    "CallbackStatePinned",
    "DelegatePinned",
    "DisposeRequested",
    "LastStatus",
    "LastDiagnostic",
    "RealCallbackRuntime=False",
    "EvidenceKind=allocator-owner-internal-runtime-prototype",
    "internal-runtime-prototype",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"'
  )

  $evidencePaths = @($ownerPath, $smokePath, $callbackDesignPath, $gatePath, $schemaPath, $runtimeSplitReadmePath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "allocator-owner-internal-runtime-prototype"
      evidenceKind = "internal-runtime-prototype"
      runtimeEvidenceKind = "not-present"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "allocator owner internal runtime prototype evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "prototype-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "prototype-ready") {
    "allocator owner lifecycle snapshot and internal runtime prototype are present and documented as not proof of real TensorRT callback runtime."
  }
  else {
    "allocator owner lifecycle snapshot evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "allocator-owner-internal-runtime-prototype"
    evidenceKind = "internal-runtime-prototype"
    runtimeEvidenceKind = "not-present"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-AllocatorOwnerLedgerSafetyGateEvidence {
  $gateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtAllocatorLedgerSafetyGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtAllocatorCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $gateDocRelativePath = "docs\articles\zh-cn\allocator-owner-ledger-safety-gate.md"
  $callbackDesignRelativePath = "docs\articles\zh-cn\allocator-callback-owner-design.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $gateSourcePath = Join-Path $RepositoryRoot $gateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $gateDocPath = Join-Path $RepositoryRoot $gateDocRelativePath
  $callbackDesignPath = Join-Path $RepositoryRoot $callbackDesignRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "allocator-owner-ledger-safety-gate",
    "TensorRtAllocatorLedgerSafetyGate",
    "TensorRtAllocatorLedgerSafetyGateResult",
    "Evaluate",
    "GetSnapshot",
    "EvidenceKind=allocator-owner-ledger-safety-gate",
    "RuntimeEvidenceKind=ledger-safety-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "ManagedKeepAliveReady",
    "DisposeReleaseReady",
    "NativeLedgerDesignReady",
    "PointerFreeSurfaceReady",
    "LineSpecificAttachDetachReady",
    "DevicePointerLedgerRuntimeReady",
    "StreamLifetimeReady",
    "FullPackageConsumerRuntimeEvidenceReady",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "BlockedPrerequisiteCount",
    "NativeLedgerAvailable",
    "StateTransitionCount",
    "LedgerAllocationCount",
    "LedgerReleaseCount",
    "InFlightCallbackCount",
    "ReleaseHookCount",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"'
  )

  $evidencePaths = @($gateSourcePath, $ownerSourcePath, $smokePath, $gateDocPath, $callbackDesignPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "allocator-owner-ledger-safety-gate"
      evidenceKind = "ledger-safety-gate"
      runtimeEvidenceKind = "ledger-safety-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "allocator owner ledger safety gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "safety-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "safety-gate-ready") {
    "allocator owner ledger safety gate is present, documented, pointer-free, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "allocator owner ledger safety gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "allocator-owner-ledger-safety-gate"
    evidenceKind = "ledger-safety-gate"
    runtimeEvidenceKind = "ledger-safety-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-OutputAllocatorInternalRuntimeGateEvidence {
  $gateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorRuntimeGate.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $gateDocRelativePath = "docs\articles\zh-cn\output-allocator-runtime-gate.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $gateSourcePath = Join-Path $RepositoryRoot $gateSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $gateDocPath = Join-Path $RepositoryRoot $gateDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "output-allocator-internal-runtime-gate",
    "TensorRtOutputAllocatorRuntimeGate",
    "TensorRtOutputAllocatorRuntimeGateRequest",
    "TensorRtOutputAllocatorRuntimeGateResult",
    "RunInternalNotifyShapeRuntimeGate",
    "RunInternalReallocateOutputRuntimeGate",
    "GetInternalRuntimeGateSnapshot",
    "TensorRtOutputAllocatorInternalRuntimeGateCallback",
    "GCHandle.Alloc(_runtimeGateCallback)",
    "NotifyShapeCount",
    "ReallocateOutputCount",
    "ShapeRank",
    "ShapeSummary",
    "OutputBufferPointerExposed",
    "OutputBufferPointerProduced",
    "RealCallbackRuntime=False",
    "EvidenceKind=output-allocator-internal-runtime-gate",
    "internal-runtime-gate",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"'
  )

  $evidencePaths = @($gateSourcePath, $smokePath, $gateDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "output-allocator-internal-runtime-gate"
      evidenceKind = "internal-runtime-gate"
      runtimeEvidenceKind = "not-present"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "output allocator internal runtime gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "gate-ready") {
    "output allocator internal runtime gate is present and documented as not proof of real TensorRT callback runtime."
  }
  else {
    "output allocator internal runtime gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "output-allocator-internal-runtime-gate"
    evidenceKind = "internal-runtime-gate"
    runtimeEvidenceKind = "not-present"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-OutputAllocatorCallbackOwnerDesignEvidence {
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorCallbackOwner.cs"
  $gateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorRuntimeGate.cs"
  $allocatorOwnerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtAllocatorCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $designDocRelativePath = "docs\articles\zh-cn\output-allocator-callback-owner-design.md"
  $gateDocRelativePath = "docs\articles\zh-cn\output-allocator-runtime-gate.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $gateSourcePath = Join-Path $RepositoryRoot $gateSourceRelativePath
  $allocatorOwnerSourcePath = Join-Path $RepositoryRoot $allocatorOwnerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $designDocPath = Join-Path $RepositoryRoot $designDocRelativePath
  $gateDocPath = Join-Path $RepositoryRoot $gateDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "output-allocator-callback-owner-design",
    "TensorRtOutputAllocatorCallbackOwner",
    "TensorRtOutputAllocatorCallbackRequest",
    "TensorRtOutputAllocatorCallbackOwnerSnapshot",
    "RunDesignDiagnostic",
    "NativeLedgerAvailable",
    "StateTransitionCount",
    "LedgerAllocationCount",
    "LedgerReleaseCount",
    "OutputBufferPointerExposed",
    "OutputBufferPointerProduced",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "RuntimeEvidenceKind=not-present",
    "design gate",
    "not proof",
    "setOutputAllocator",
    "IOutputAllocator::notifyShape",
    "IOutputAllocator::reallocateOutput"
  )

  $requiredDeferredRows = @(
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($ownerSourcePath, $gateSourcePath, $allocatorOwnerSourcePath, $smokePath, $designDocPath, $gateDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "output-allocator-callback-owner-design"
      evidenceKind = "owner-design-gate"
      runtimeEvidenceKind = "not-present"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "output allocator callback owner design evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-ready") {
    "output allocator callback owner design gate is present, documented, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "output allocator callback owner design evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "output-allocator-callback-owner-design"
    evidenceKind = "owner-design-gate"
    runtimeEvidenceKind = "not-present"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-OutputAllocatorAttachDetachDesignGateEvidence {
  $gateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorAttachDetachDesignGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorCallbackOwner.cs"
  $contextSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $designDocRelativePath = "docs\articles\zh-cn\output-allocator-attach-detach-design-gate.md"
  $callbackDesignRelativePath = "docs\articles\zh-cn\output-allocator-callback-owner-design.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\output-allocator-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $gateSourcePath = Join-Path $RepositoryRoot $gateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $contextSourcePath = Join-Path $RepositoryRoot $contextSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $designDocPath = Join-Path $RepositoryRoot $designDocRelativePath
  $callbackDesignPath = Join-Path $RepositoryRoot $callbackDesignRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "output-allocator-attach-detach-design-gate",
    "TensorRtOutputAllocatorAttachDetachDesignGate",
    "TensorRtOutputAllocatorAttachDetachDesignGateResult",
    "Evaluate",
    "EvidenceKind=output-allocator-attach-detach-design-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "LineSupportsOutputAllocator",
    "AttachControlAvailable=False",
    "DetachClearControlAvailable=True",
    "LineSpecificAttachDetachReady=False",
    "NativeVTableReady=False",
    "OutputBufferOwnershipRuntimeReady=False",
    "DesignGateReady",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setOutputAllocator",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($gateSourcePath, $ownerSourcePath, $contextSourcePath, $smokePath, $designDocPath, $callbackDesignPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "output-allocator-attach-detach-design-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "output allocator attach/detach design gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-gate-ready") {
    "output allocator attach/detach design gate is documented, pointer-free, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "output allocator attach/detach design gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "output-allocator-attach-detach-design-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-OutputBufferOwnershipSafetyGateEvidence {
  $ownershipGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputBufferOwnershipSafetyGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorRuntimeProofPrecheck.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorAttachDetachDesignGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $ownershipGateDocRelativePath = "docs\articles\zh-cn\output-buffer-ownership-safety-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\output-allocator-runtime-proof-precheck.md"
  $attachDetachGateDocRelativePath = "docs\articles\zh-cn\output-allocator-attach-detach-design-gate.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $ownershipGateSourcePath = Join-Path $RepositoryRoot $ownershipGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $ownershipGateDocPath = Join-Path $RepositoryRoot $ownershipGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $attachDetachGateDocPath = Join-Path $RepositoryRoot $attachDetachGateDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "output-buffer-ownership-safety-gate",
    "TensorRtOutputBufferOwnershipSafetyGate",
    "TensorRtOutputBufferOwnershipSafetyGateResult",
    "Evaluate",
    "EvidenceKind=output-buffer-ownership-safety-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "SafetyGateReady",
    "CopiedCurrentMemoryMetadataReady",
    "CopiedShapeMetadataReady",
    "CopiedRequestMetadataReady",
    "OutputBufferOwnershipRuntimeReady=False",
    "CurrentMemoryReusePolicyReady=False",
    "BorrowedPointerEscapeBlocked=True",
    "OwnedDevicePointerReleasePolicyReady=False",
    "ShapeNotificationOrderingReady=False",
    "ReallocateOutputRuntimeReady=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "currentMemory",
    "IOutputAllocator::reallocateOutput",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($ownershipGateSourcePath, $precheckSourcePath, $attachDetachGateSourcePath, $ownerSourcePath, $smokePath, $ownershipGateDocPath, $precheckDocPath, $attachDetachGateDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "output-buffer-ownership-safety-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "output buffer ownership safety gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "safety-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "safety-gate-ready") {
    "output buffer ownership safety gate is documented, pointer-free, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "output buffer ownership safety gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "output-buffer-ownership-safety-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-OutputAllocatorRuntimeProofPrecheckEvidence {
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorRuntimeProofPrecheck.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorAttachDetachDesignGate.cs"
  $ownershipGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputBufferOwnershipSafetyGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorCallbackOwner.cs"
  $gateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtOutputAllocatorRuntimeGate.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $precheckDocRelativePath = "docs\articles\zh-cn\output-allocator-runtime-proof-precheck.md"
  $attachDetachGateDocRelativePath = "docs\articles\zh-cn\output-allocator-attach-detach-design-gate.md"
  $ownershipGateDocRelativePath = "docs\articles\zh-cn\output-buffer-ownership-safety-gate.md"
  $designDocRelativePath = "docs\articles\zh-cn\output-allocator-callback-owner-design.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $ownershipGateSourcePath = Join-Path $RepositoryRoot $ownershipGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $gateSourcePath = Join-Path $RepositoryRoot $gateSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $attachDetachGateDocPath = Join-Path $RepositoryRoot $attachDetachGateDocRelativePath
  $ownershipGateDocPath = Join-Path $RepositoryRoot $ownershipGateDocRelativePath
  $designDocPath = Join-Path $RepositoryRoot $designDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "output-allocator-runtime-proof-precheck",
    "TensorRtOutputAllocatorRuntimeProofPrecheck",
    "TensorRtOutputAllocatorRuntimeProofPrecheckResult",
    "Evaluate",
    "RuntimeEvidenceKind=runtime-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "output-allocator-attach-detach-design-gate",
    "output-buffer-ownership-safety-gate",
    "AttachDetachDesignGateReady",
    "NativeLedgerDesignReady",
    "LineSupportsOutputAllocator",
    "AttachControlAvailable=False",
    "DetachClearControlAvailable",
    "ManagedOwnerStateMachineReady",
    "LineSpecificAttachDetachReady",
    "StableNativeOwnerAddressReady",
    "NoThrowNativeVTableReady",
    "NativeVTableReady",
    "DevicePointerLedgerRuntimeReady",
    "StreamLifetimeReady",
    "OutputBufferOwnershipSafetyGateReady",
    "OutputBufferOwnershipRuntimeReady",
    "CurrentMemoryReusePolicyReady",
    "BorrowedPointerEscapeBlocked",
    "OwnedDevicePointerReleasePolicyReady",
    "ShapeNotificationOrderingReady",
    "ReallocateOutputRuntimeReady",
    "FullPackageConsumerRuntimeEvidenceReady",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "BlockedPrerequisiteCount",
    "IOutputAllocator::notifyShape",
    "IOutputAllocator::reallocateOutput",
    "setOutputAllocator",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($precheckSourcePath, $attachDetachGateSourcePath, $ownershipGateSourcePath, $ownerSourcePath, $gateSourcePath, $smokePath, $precheckDocPath, $attachDetachGateDocPath, $ownershipGateDocPath, $designDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "output-allocator-runtime-proof-precheck"
      evidenceKind = "runtime-gate-precheck"
      runtimeEvidenceKind = "runtime-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "output allocator runtime proof precheck evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "precheck-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "precheck-ready") {
    "output allocator runtime proof precheck is present and explicitly blocked from real-callback-runtime promotion until non-null attach, native owner/vtable, runtime device pointer ledger, stream/output-buffer ownership, and full package consumer runtime evidence exist."
  }
  else {
    "output allocator runtime proof precheck evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "output-allocator-runtime-proof-precheck"
    evidenceKind = "runtime-gate-precheck"
    runtimeEvidenceKind = "runtime-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerCallbackOwnerDesignEvidence {
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $designDocRelativePath = "docs\articles\zh-cn\debug-listener-callback-owner-design.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $designDocPath = Join-Path $RepositoryRoot $designDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-callback-owner-design",
    "TensorRtDebugListenerCallbackOwner",
    "TensorRtDebugListenerCallbackRequest",
    "TensorRtDebugListenerCallbackOwnerSnapshot",
    "RunDesignDiagnostic",
    "TensorRtDataType",
    "TensorRtTensorLocation",
    "ProcessDebugTensorCount",
    "DebugTensorMetadataCopied",
    "DebugTensorPointerExposed",
    "DebugTensorPointerProduced",
    "BorrowedDebugTensorPointerEscaped",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "RuntimeEvidenceKind=not-present",
    "owner-design-gate",
    "setDebugListener",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($ownerSourcePath, $smokePath, $designDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-callback-owner-design"
      evidenceKind = "owner-design-gate"
      runtimeEvidenceKind = "not-present"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener callback owner design evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-ready") {
    "debug listener callback owner design gate is present, documented, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "debug listener callback owner design evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-callback-owner-design"
    evidenceKind = "owner-design-gate"
    runtimeEvidenceKind = "not-present"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerAttachDetachDesignGateEvidence {
  $gateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $contextSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtExecutionContext.Trt11RuntimeDiagnostics.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $designDocRelativePath = "docs\articles\zh-cn\debug-listener-attach-detach-design-gate.md"
  $callbackDesignRelativePath = "docs\articles\zh-cn\debug-listener-callback-owner-design.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $gateSourcePath = Join-Path $RepositoryRoot $gateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $contextSourcePath = Join-Path $RepositoryRoot $contextSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $designDocPath = Join-Path $RepositoryRoot $designDocRelativePath
  $callbackDesignPath = Join-Path $RepositoryRoot $callbackDesignRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-attach-detach-design-gate",
    "TensorRtDebugListenerAttachDetachDesignGate",
    "TensorRtDebugListenerAttachDetachDesignGateResult",
    "Evaluate",
    "EvidenceKind=debug-listener-attach-detach-design-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "AttachControlAvailable=False",
    "DetachClearControlAvailable=True",
    "LineSpecificAttachDetachReady=False",
    "NativeVTableReady=False",
    "BorrowedDebugTensorLifetimeReady=False",
    "DesignGateReady",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setDebugListener",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($gateSourcePath, $ownerSourcePath, $contextSourcePath, $smokePath, $designDocPath, $callbackDesignPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-attach-detach-design-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener attach/detach design gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-gate-ready") {
    "debug listener attach/detach design gate is documented, pointer-free, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "debug listener attach/detach design gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-attach-detach-design-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerBorrowedTensorSafetyGateEvidence {
  $borrowedTensorSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $borrowedTensorSafetyGateDocRelativePath = "docs\articles\zh-cn\debug-listener-borrowed-tensor-safety-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $attachDetachGateDocRelativePath = "docs\articles\zh-cn\debug-listener-attach-detach-design-gate.md"
  $callbackDesignDocRelativePath = "docs\articles\zh-cn\debug-listener-callback-owner-design.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $borrowedTensorSafetyGateSourcePath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $borrowedTensorSafetyGateDocPath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $attachDetachGateDocPath = Join-Path $RepositoryRoot $attachDetachGateDocRelativePath
  $callbackDesignDocPath = Join-Path $RepositoryRoot $callbackDesignDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-borrowed-tensor-safety-gate",
    "TensorRtDebugListenerBorrowedTensorSafetyGate",
    "TensorRtDebugListenerBorrowedTensorSafetyGateResult",
    "Evaluate",
    "EvidenceKind=debug-listener-borrowed-tensor-safety-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "SafetyGateReady",
    "DebugTensorMetadataCopied",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "BorrowedDebugTensorLifetimeReady=False",
    "BorrowedDebugTensorDataLifetimeReady=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($borrowedTensorSafetyGateSourcePath, $precheckSourcePath, $attachDetachGateSourcePath, $ownerSourcePath, $smokePath, $borrowedTensorSafetyGateDocPath, $precheckDocPath, $attachDetachGateDocPath, $callbackDesignDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-borrowed-tensor-safety-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener borrowed tensor safety gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "safety-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "safety-gate-ready") {
    "debug listener borrowed tensor safety gate is documented, pointer-free, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "debug listener borrowed tensor safety gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-borrowed-tensor-safety-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerRuntimeProofPrecheckEvidence {
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $borrowedTensorSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs"
  $attachVTableSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachVTableSafetyGate.cs"
  $nativeAttachNoThrowPreflightSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachNoThrowPreflight.cs"
  $nativeOwnerAddressDesignGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerAddressDesignGate.cs"
  $nativeNoThrowVTableDesignGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs"
  $nativeAttachEntryDesignGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryDesignGate.cs"
  $nativeDetachBeforeReleaseDesignGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs"
  $nativeOwnerStableIdentitySourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerStableIdentity.cs"
  $nativeOwnerNonCopyableStorageSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs"
  $nativeNoThrowDestructorSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowDestructor.cs"
  $nativeOwnerLifecycleGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleGate.cs"
  $nativeAttachBridgeShapeGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachBridgeShapeGate.cs"
  $exceptionStatusMappingGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerExceptionStatusMappingGate.cs"
  $inFlightAccountingGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerInFlightAccountingGate.cs"
  $nativeNoThrowVTableScaffoldGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs"
  $nativeOwnerNonCopyableStorageScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_owner_noncopyable_storage.inc"
  $nativeNoThrowDestructorScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_nothrow_destructor.inc"
  $nativeOwnerLifecycleGateScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_owner_lifecycle_gate.inc"
  $nativeAttachBridgeShapeGateScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_attach_bridge_shape_gate.inc"
  $exceptionStatusMappingGateScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_exception_status_mapping_gate.inc"
  $inFlightAccountingGateScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_inflight_accounting_gate.inc"
  $nativeNoThrowVTableScaffoldGateScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_nothrow_vtable_scaffold_gate.inc"
  $nativeTrt8RelativePath = "native\src\tensorrt\v8\api.cpp"
  $nativeTrt10RelativePath = "native\src\tensorrt\v10\api.cpp"
  $nativeTrt11RelativePath = "native\src\tensorrt\v11\api.cpp"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $attachDetachGateDocRelativePath = "docs\articles\zh-cn\debug-listener-attach-detach-design-gate.md"
  $borrowedTensorSafetyGateDocRelativePath = "docs\articles\zh-cn\debug-listener-borrowed-tensor-safety-gate.md"
  $attachVTableSafetyGateDocRelativePath = "docs\articles\zh-cn\debug-listener-attach-vtable-safety-gate.md"
  $nativeAttachNoThrowPreflightDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-nothrow-preflight.md"
  $nativeOwnerAddressDesignGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-address-design-gate.md"
  $nativeNoThrowVTableDesignGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-nothrow-vtable-design-gate.md"
  $nativeAttachEntryDesignGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-entry-design-gate.md"
  $nativeDetachBeforeReleaseDesignGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-detach-before-release-design-gate.md"
  $nativeOwnerStableIdentityDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-stable-identity.md"
  $nativeOwnerNonCopyableStorageDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-noncopyable-storage.md"
  $nativeNoThrowDestructorDocRelativePath = "docs\articles\zh-cn\debug-listener-native-nothrow-destructor.md"
  $nativeOwnerLifecycleGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-lifecycle-gate.md"
  $nativeAttachBridgeShapeGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-bridge-shape-gate.md"
  $exceptionStatusMappingGateDocRelativePath = "docs\articles\zh-cn\debug-listener-exception-status-mapping-gate.md"
  $inFlightAccountingGateDocRelativePath = "docs\articles\zh-cn\debug-listener-inflight-accounting-gate.md"
  $nativeNoThrowVTableScaffoldGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-nothrow-vtable-scaffold-gate.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $borrowedTensorSafetyGateSourcePath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateSourceRelativePath
  $attachVTableSafetyGateSourcePath = Join-Path $RepositoryRoot $attachVTableSafetyGateSourceRelativePath
  $nativeAttachNoThrowPreflightSourcePath = Join-Path $RepositoryRoot $nativeAttachNoThrowPreflightSourceRelativePath
  $nativeOwnerAddressDesignGateSourcePath = Join-Path $RepositoryRoot $nativeOwnerAddressDesignGateSourceRelativePath
  $nativeNoThrowVTableDesignGateSourcePath = Join-Path $RepositoryRoot $nativeNoThrowVTableDesignGateSourceRelativePath
  $nativeAttachEntryDesignGateSourcePath = Join-Path $RepositoryRoot $nativeAttachEntryDesignGateSourceRelativePath
  $nativeDetachBeforeReleaseDesignGateSourcePath = Join-Path $RepositoryRoot $nativeDetachBeforeReleaseDesignGateSourceRelativePath
  $nativeOwnerStableIdentitySourcePath = Join-Path $RepositoryRoot $nativeOwnerStableIdentitySourceRelativePath
  $nativeOwnerNonCopyableStorageSourcePath = Join-Path $RepositoryRoot $nativeOwnerNonCopyableStorageSourceRelativePath
  $nativeNoThrowDestructorSourcePath = Join-Path $RepositoryRoot $nativeNoThrowDestructorSourceRelativePath
  $nativeOwnerLifecycleGateSourcePath = Join-Path $RepositoryRoot $nativeOwnerLifecycleGateSourceRelativePath
  $nativeAttachBridgeShapeGateSourcePath = Join-Path $RepositoryRoot $nativeAttachBridgeShapeGateSourceRelativePath
  $exceptionStatusMappingGateSourcePath = Join-Path $RepositoryRoot $exceptionStatusMappingGateSourceRelativePath
  $inFlightAccountingGateSourcePath = Join-Path $RepositoryRoot $inFlightAccountingGateSourceRelativePath
  $nativeNoThrowVTableScaffoldGateSourcePath = Join-Path $RepositoryRoot $nativeNoThrowVTableScaffoldGateSourceRelativePath
  $nativeOwnerNonCopyableStorageScaffoldPath = Join-Path $RepositoryRoot $nativeOwnerNonCopyableStorageScaffoldRelativePath
  $nativeNoThrowDestructorScaffoldPath = Join-Path $RepositoryRoot $nativeNoThrowDestructorScaffoldRelativePath
  $nativeOwnerLifecycleGateScaffoldPath = Join-Path $RepositoryRoot $nativeOwnerLifecycleGateScaffoldRelativePath
  $nativeAttachBridgeShapeGateScaffoldPath = Join-Path $RepositoryRoot $nativeAttachBridgeShapeGateScaffoldRelativePath
  $exceptionStatusMappingGateScaffoldPath = Join-Path $RepositoryRoot $exceptionStatusMappingGateScaffoldRelativePath
  $inFlightAccountingGateScaffoldPath = Join-Path $RepositoryRoot $inFlightAccountingGateScaffoldRelativePath
  $nativeNoThrowVTableScaffoldGateScaffoldPath = Join-Path $RepositoryRoot $nativeNoThrowVTableScaffoldGateScaffoldRelativePath
  $nativeTrt8Path = Join-Path $RepositoryRoot $nativeTrt8RelativePath
  $nativeTrt10Path = Join-Path $RepositoryRoot $nativeTrt10RelativePath
  $nativeTrt11Path = Join-Path $RepositoryRoot $nativeTrt11RelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $attachDetachGateDocPath = Join-Path $RepositoryRoot $attachDetachGateDocRelativePath
  $borrowedTensorSafetyGateDocPath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateDocRelativePath
  $attachVTableSafetyGateDocPath = Join-Path $RepositoryRoot $attachVTableSafetyGateDocRelativePath
  $nativeAttachNoThrowPreflightDocPath = Join-Path $RepositoryRoot $nativeAttachNoThrowPreflightDocRelativePath
  $nativeOwnerAddressDesignGateDocPath = Join-Path $RepositoryRoot $nativeOwnerAddressDesignGateDocRelativePath
  $nativeNoThrowVTableDesignGateDocPath = Join-Path $RepositoryRoot $nativeNoThrowVTableDesignGateDocRelativePath
  $nativeAttachEntryDesignGateDocPath = Join-Path $RepositoryRoot $nativeAttachEntryDesignGateDocRelativePath
  $nativeDetachBeforeReleaseDesignGateDocPath = Join-Path $RepositoryRoot $nativeDetachBeforeReleaseDesignGateDocRelativePath
  $nativeOwnerStableIdentityDocPath = Join-Path $RepositoryRoot $nativeOwnerStableIdentityDocRelativePath
  $nativeOwnerNonCopyableStorageDocPath = Join-Path $RepositoryRoot $nativeOwnerNonCopyableStorageDocRelativePath
  $nativeNoThrowDestructorDocPath = Join-Path $RepositoryRoot $nativeNoThrowDestructorDocRelativePath
  $nativeOwnerLifecycleGateDocPath = Join-Path $RepositoryRoot $nativeOwnerLifecycleGateDocRelativePath
  $nativeAttachBridgeShapeGateDocPath = Join-Path $RepositoryRoot $nativeAttachBridgeShapeGateDocRelativePath
  $exceptionStatusMappingGateDocPath = Join-Path $RepositoryRoot $exceptionStatusMappingGateDocRelativePath
  $inFlightAccountingGateDocPath = Join-Path $RepositoryRoot $inFlightAccountingGateDocRelativePath
  $nativeNoThrowVTableScaffoldGateDocPath = Join-Path $RepositoryRoot $nativeNoThrowVTableScaffoldGateDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-runtime-proof-precheck",
    "TensorRtDebugListenerRuntimeProofPrecheck",
    "TensorRtDebugListenerRuntimeProofPrecheckResult",
    "Evaluate",
    "RuntimeEvidenceKind=runtime-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
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
    "debug-listener-native-owner-stable-identity",
    "debug-listener-native-owner-noncopyable-storage",
    "debug-listener-native-nothrow-destructor",
    "debug-listener-native-owner-lifecycle-gate",
    "debug-listener-native-attach-bridge-shape-gate",
    "debug-listener-exception-status-mapping-gate",
    "debug-listener-inflight-accounting-gate",
    "debug-listener-native-nothrow-vtable-scaffold-gate",
    "AttachDetachDesignGateReady",
    "AttachControlAvailable=False",
    "DetachClearControlAvailable=True",
    "ManagedOwnerStateMachineReady",
    "LineSpecificAttachDetachReady",
    "StableNativeOwnerAddressReady",
    "NoThrowNativeVTableReady",
    "NativeVTableReady",
    "ExceptionToStatusMappingReady",
    "BorrowedTensorSafetyGateReady",
    "AttachVTableSafetyGateReady",
    "NativeAttachNoThrowPreflightReady",
    "NativeOwnerAddressDesignGateReady",
    "NativeNoThrowVTableDesignGateReady",
    "NativeAttachEntryDesignGateReady",
    "NativeDetachBeforeReleaseDesignGateReady",
    "NativeOwnerLifecycleDryRunReady",
    "NativeAttachEntryRuntimeScaffoldReady",
    "NativeOwnerStableIdentityReady",
    "OwnerIdentityDiagnosticsReady",
    "OwnerIdentityPointerFree",
    "NativeOwnerNonCopyableStorageReady",
    "NativeOwnerCopyBlocked",
    "NativeOwnerMoveBlocked",
    "NativeOwnerAddressExposed",
    "NativeOwnerPointerProduced",
    "NativeNoThrowDestructorGateReady",
    "DestructorNoThrowScaffoldReady",
    "DestructorExceptionEscapeBlocked",
    "DestructorAddressExposed=False",
    "DestructorPointerProduced=False",
    "NativeOwnerLifecycleGateReady",
    "ManagedDisposeSnapshotReady",
    "LifecycleScaffoldReady",
    "ReleaseHookOrderingGateReady",
    "DisposeIdempotencyGateReady",
    "InFlightDrainGateReady",
    "CallbackStateUnpinAfterDetachGateReady",
    "DelegateUnpinAfterDetachGateReady",
    "LifecycleAddressExposed=False",
    "LifecyclePointerProduced=False",
    "NativeAttachBridgeShapeGateReady=True",
    "AttachBridgeShapeReady=True",
    "AttachBridgeNoThrowBoundaryReady=True",
    "AttachBridgeVersionGuardReady=True",
    "AttachBridgeOwnershipDiagnosticsReady=True",
    "AttachBridgePointerFree=True",
    "NonNullAttachStillDisabled=True",
    "ExceptionStatusMappingGateReady=True",
    "NativeCallbackExceptionCaptureReady=True",
    "CallbackStatusMappingGateReady=True",
    "ExceptionEscapeBlocked=True",
    "DiagnosticCopyReady=True",
    "InFlightAccountingGateReady=True",
    "CallbackEnterAccountingGateReady=True",
    "CallbackLeaveAccountingGateReady=True",
    "CallbackInFlightNeverNegativeReady=True",
    "ReleaseAfterDrainGateReady=True",
    "CallbackStateUnpinAfterDrainGateReady=True",
    "NativeNoThrowVTableScaffoldGateReady=True",
    "NoThrowVTableScaffoldReady=True",
    "VTableDestructorNoThrowReady=True",
    "ProcessDebugTensorCallbackStubNoThrowReady=True",
    "VTableAddressExposed=False",
    "VTablePointerProduced=False",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "LineSpecificAttachEntryDesignReady=False",
    "AttachEntryNoThrowReady=False",
    "AttachEntryVersionGuardReady=False",
    "AttachEntryOwnershipReady=False",
    "DetachBeforeReleaseReady=False",
    "ReleaseHookOrderingReady=False",
    "DisposeIdempotencyReady=False",
    "InFlightDrainBeforeReleaseReady=False",
    "CallbackStateUnpinAfterDetachReady=False",
    "DelegateUnpinAfterDetachReady=False",
    "StableNativeOwnerAddressDesignReady=False",
    "NativeOwnerNonCopyableReady=True",
    "NativeOwnerDisposeOrderReady=False",
    "NativeOwnerReleaseHookReady=False",
    "NativeOwnerInFlightDrainReady=False",
    "NoThrowNativeDestructorReady=True",
    "NativeOwnerLifecycleReady=False",
    "NoThrowVTableDesignReady=False",
    "ExceptionToStatusMappingDesignReady=False",
    "NativeVTableTrampolineReady=False",
    "CallbackExceptionCaptureReady=False",
    "CallbackStatusMappingReady=False",
    "CallbackInFlightAccountingReady=False",
    "CanImplementNativeAttach=False",
    "BorrowedDebugTensorPointerEscapeBlocked",
    "BorrowedDebugTensorLifetimeReady",
    "BorrowedDebugTensorDataLifetimeReady",
    "ProcessDebugTensorRuntimeReady",
    "FullPackageConsumerRuntimeEvidenceReady",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "BlockedPrerequisiteCount",
    "IDebugListener::processDebugTensor",
    "setDebugListener",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($precheckSourcePath, $attachDetachGateSourcePath, $borrowedTensorSafetyGateSourcePath, $attachVTableSafetyGateSourcePath, $nativeAttachNoThrowPreflightSourcePath, $nativeOwnerAddressDesignGateSourcePath, $nativeNoThrowVTableDesignGateSourcePath, $nativeAttachEntryDesignGateSourcePath, $nativeDetachBeforeReleaseDesignGateSourcePath, $nativeOwnerStableIdentitySourcePath, $nativeOwnerNonCopyableStorageSourcePath, $nativeNoThrowDestructorSourcePath, $nativeOwnerLifecycleGateSourcePath, $nativeAttachBridgeShapeGateSourcePath, $exceptionStatusMappingGateSourcePath, $inFlightAccountingGateSourcePath, $nativeNoThrowVTableScaffoldGateSourcePath, $nativeOwnerNonCopyableStorageScaffoldPath, $nativeNoThrowDestructorScaffoldPath, $nativeOwnerLifecycleGateScaffoldPath, $nativeAttachBridgeShapeGateScaffoldPath, $exceptionStatusMappingGateScaffoldPath, $inFlightAccountingGateScaffoldPath, $nativeNoThrowVTableScaffoldGateScaffoldPath, $nativeTrt8Path, $nativeTrt10Path, $nativeTrt11Path, $ownerSourcePath, $smokePath, $precheckDocPath, $attachDetachGateDocPath, $borrowedTensorSafetyGateDocPath, $attachVTableSafetyGateDocPath, $nativeAttachNoThrowPreflightDocPath, $nativeOwnerAddressDesignGateDocPath, $nativeNoThrowVTableDesignGateDocPath, $nativeAttachEntryDesignGateDocPath, $nativeDetachBeforeReleaseDesignGateDocPath, $nativeOwnerStableIdentityDocPath, $nativeOwnerNonCopyableStorageDocPath, $nativeNoThrowDestructorDocPath, $nativeOwnerLifecycleGateDocPath, $nativeAttachBridgeShapeGateDocPath, $exceptionStatusMappingGateDocPath, $inFlightAccountingGateDocPath, $nativeNoThrowVTableScaffoldGateDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-runtime-proof-precheck"
      evidenceKind = "runtime-gate-precheck"
      runtimeEvidenceKind = "runtime-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener runtime proof precheck evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "precheck-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "precheck-ready") {
    "debug listener runtime proof precheck is present, consumes attach bridge shape, exception/status mapping, in-flight accounting, and no-throw vtable scaffold gates as source-visible non-proof evidence, and remains blocked from real-callback-runtime promotion until non-null attach, complete native vtable, processDebugTensor runtime, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener runtime proof precheck evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-runtime-proof-precheck"
    evidenceKind = "runtime-gate-precheck"
    runtimeEvidenceKind = "runtime-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerRuntimeProofAttemptPreflightEvidence {
  $preflightSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofAttemptPreflight.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $preflightDocRelativePath = "docs\articles\zh-cn\debug-listener-real-callback-runtime-proof-preflight.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $tocRelativePath = "docs\toc.yml"
  $indexRelativePath = "docs\index.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $preflightSourcePath = Join-Path $RepositoryRoot $preflightSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $preflightDocPath = Join-Path $RepositoryRoot $preflightDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-runtime-proof-attempt-preflight",
    "TensorRtDebugListenerRuntimeProofAttemptPreflight",
    "TensorRtDebugListenerRuntimeProofAttemptPreflightResult",
    "Evaluate",
    "RuntimeEvidenceKind=runtime-proof-attempt-preflight",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "CanEnableSetDebugListenerNonNull",
    "CanInstallNativeVTable",
    "CanCallProcessDebugTensorRuntime",
    "CanPromoteRealCallbackRuntime",
    "ReasonNonNullAttachStillBlocked",
    "ReasonNativeVTableStillBlocked",
    "ReasonRuntimeProofStillBlocked",
    "CanEnableSetDebugListenerNonNull=False",
    "CanInstallNativeVTable=False",
    "CanCallProcessDebugTensorRuntime=False",
    "CanPromoteRealCallbackRuntime=False",
    "NativeAttachEntryLocated=False",
    "NonNullAttachStillDisabled=True",
    "NativeVTableReady=False",
    "NativeVTableTrampolineReady=False",
    "ProcessDebugTensorRuntimeReady=False",
    "FullPackageConsumerRuntimeEvidenceReady=False",
    "RuntimeProofBlocked=True",
    "not proof",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "real-callback-runtime"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"'
  )

  $evidencePaths = @($preflightSourcePath, $precheckSourcePath, $smokePath, $preflightDocPath, $precheckDocPath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $tocPath, $indexPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-runtime-proof-attempt-preflight"
      evidenceKind = "runtime-proof-attempt-preflight"
      runtimeEvidenceKind = "runtime-proof-attempt-preflight"
      source = "source-smoke-docs-consumer"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canEnableSetDebugListenerNonNull = $false
      canInstallNativeVTable = $false
      canCallProcessDebugTensorRuntime = $false
      canPromoteRealCallbackRuntime = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener runtime proof attempt preflight evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "preflight-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "preflight-ready") {
    "debug listener runtime proof attempt preflight is documented, smoke-visible, package-consumer-visible, pointer-free, and remains blocked from real-callback-runtime promotion until non-null attach, native vtable installation, processDebugTensor runtime invocation, and full package consumer proof exist."
  }
  else {
    "debug listener runtime proof attempt preflight evidence is incomplete; inspect source/smoke/docs/consumer markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-runtime-proof-attempt-preflight"
    evidenceKind = "runtime-proof-attempt-preflight"
    runtimeEvidenceKind = "runtime-proof-attempt-preflight"
    source = "source-smoke-docs-consumer"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canEnableSetDebugListenerNonNull = $false
    canInstallNativeVTable = $false
    canCallProcessDebugTensorRuntime = $false
    canPromoteRealCallbackRuntime = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerAttachVTableSafetyGateEvidence {
  $attachVTableSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachVTableSafetyGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $borrowedTensorSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $attachVTableSafetyGateDocRelativePath = "docs\articles\zh-cn\debug-listener-attach-vtable-safety-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $attachDetachGateDocRelativePath = "docs\articles\zh-cn\debug-listener-attach-detach-design-gate.md"
  $borrowedTensorSafetyGateDocRelativePath = "docs\articles\zh-cn\debug-listener-borrowed-tensor-safety-gate.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $attachVTableSafetyGateSourcePath = Join-Path $RepositoryRoot $attachVTableSafetyGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $borrowedTensorSafetyGateSourcePath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $attachVTableSafetyGateDocPath = Join-Path $RepositoryRoot $attachVTableSafetyGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $attachDetachGateDocPath = Join-Path $RepositoryRoot $attachDetachGateDocRelativePath
  $borrowedTensorSafetyGateDocPath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-attach-vtable-safety-gate",
    "TensorRtDebugListenerAttachVTableSafetyGate",
    "TensorRtDebugListenerAttachVTableSafetyGateResult",
    "Evaluate",
    "EvidenceKind=debug-listener-attach-vtable-safety-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "SafetyGateReady",
    "AttachControlAvailable=False",
    "StableNativeOwnerAddressReady=False",
    "NoThrowNativeVTableReady=False",
    "ExceptionToStatusMappingReady=False",
    "ProcessDebugTensorRuntimeReady=False",
    "FullPackageConsumerRuntimeEvidenceReady",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($attachVTableSafetyGateSourcePath, $precheckSourcePath, $attachDetachGateSourcePath, $borrowedTensorSafetyGateSourcePath, $ownerSourcePath, $smokePath, $attachVTableSafetyGateDocPath, $precheckDocPath, $attachDetachGateDocPath, $borrowedTensorSafetyGateDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-attach-vtable-safety-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener attach/vtable safety gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "safety-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "safety-gate-ready") {
    "debug listener attach/vtable safety gate is documented, pointer-free, and explicitly not real TensorRT callback runtime proof."
  }
  else {
    "debug listener attach/vtable safety gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-attach-vtable-safety-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeAttachNoThrowPreflightEvidence {
  $preflightSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachNoThrowPreflight.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $attachVTableSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachVTableSafetyGate.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $borrowedTensorSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $preflightDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-nothrow-preflight.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $attachVTableSafetyGateDocRelativePath = "docs\articles\zh-cn\debug-listener-attach-vtable-safety-gate.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $preflightSourcePath = Join-Path $RepositoryRoot $preflightSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $attachVTableSafetyGateSourcePath = Join-Path $RepositoryRoot $attachVTableSafetyGateSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $borrowedTensorSafetyGateSourcePath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $preflightDocPath = Join-Path $RepositoryRoot $preflightDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $attachVTableSafetyGateDocPath = Join-Path $RepositoryRoot $attachVTableSafetyGateDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-attach-nothrow-preflight",
    "TensorRtDebugListenerNativeAttachNoThrowPreflight",
    "TensorRtDebugListenerNativeAttachNoThrowPreflightResult",
    "Evaluate",
    "EvidenceKind=debug-listener-native-attach-nothrow-preflight",
    "RuntimeEvidenceKind=preflight",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "PreflightReady",
    "AttachVTableSafetyGateReady",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "StableNativeOwnerAddressDesignReady=False",
    "ManagedCallbackKeepAliveDesignReady",
    "NoThrowVTableDesignReady=False",
    "ExceptionToStatusMappingDesignReady=False",
    "BorrowedDebugTensorMetadataCopyDesignReady",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "NativeVTableDesignReady",
    "BorrowedDebugTensorLifetimeRuntimeReady=False",
    "BorrowedDebugTensorDataLifetimeRuntimeReady=False",
    "ProcessDebugTensorRuntimeReady=False",
    "FullPackageConsumerRuntimeEvidenceReady",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($preflightSourcePath, $precheckSourcePath, $attachVTableSafetyGateSourcePath, $attachDetachGateSourcePath, $borrowedTensorSafetyGateSourcePath, $ownerSourcePath, $smokePath, $preflightDocPath, $precheckDocPath, $attachVTableSafetyGateDocPath, $callbackGatePath, $schemaPath, $latestPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-attach-nothrow-preflight"
      evidenceKind = "preflight"
      runtimeEvidenceKind = "preflight"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native attach/no-throw preflight evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "preflight-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "preflight-ready") {
    "debug listener native attach/no-throw preflight is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until native attach, stable owner address, no-throw vtable, exception mapping, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native attach/no-throw preflight evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-attach-nothrow-preflight"
    evidenceKind = "preflight"
    runtimeEvidenceKind = "preflight"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeOwnerAddressDesignGateEvidence {
  $ownerAddressGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerAddressDesignGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $preflightSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachNoThrowPreflight.cs"
  $attachVTableSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachVTableSafetyGate.cs"
  $borrowedTensorSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $ownerAddressGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-address-design-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $preflightDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-nothrow-preflight.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $ownerAddressGateSourcePath = Join-Path $RepositoryRoot $ownerAddressGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $preflightSourcePath = Join-Path $RepositoryRoot $preflightSourceRelativePath
  $attachVTableSafetyGateSourcePath = Join-Path $RepositoryRoot $attachVTableSafetyGateSourceRelativePath
  $borrowedTensorSafetyGateSourcePath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $ownerAddressGateDocPath = Join-Path $RepositoryRoot $ownerAddressGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $preflightDocPath = Join-Path $RepositoryRoot $preflightDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-owner-address-design-gate",
    "TensorRtDebugListenerNativeOwnerAddressDesignGate",
    "TensorRtDebugListenerNativeOwnerAddressDesignGateResult",
    "Evaluate",
    "EvidenceKind=debug-listener-native-owner-address-design-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "DesignGateReady",
    "NativeAttachNoThrowPreflightReady",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "StableNativeOwnerAddressReady=False",
    "StableNativeOwnerAddressDesignReady=False",
    "ManagedCallbackKeepAliveDesignReady",
    "NativeOwnerNonCopyableReady=False",
    "NativeOwnerDisposeOrderReady=False",
    "NativeOwnerReleaseHookReady=False",
    "NativeOwnerInFlightDrainReady=False",
    "NoThrowNativeDestructorReady=False",
    "NoThrowVTableDesignReady=False",
    "ExceptionToStatusMappingDesignReady=False",
    "BorrowedDebugTensorMetadataCopyDesignReady",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "NativeOwnerLifecycleReady=False",
    "NativeVTableDesignReady",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($ownerAddressGateSourcePath, $precheckSourcePath, $preflightSourcePath, $attachVTableSafetyGateSourcePath, $borrowedTensorSafetyGateSourcePath, $attachDetachGateSourcePath, $ownerSourcePath, $smokePath, $ownerAddressGateDocPath, $precheckDocPath, $preflightDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-owner-address-design-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native owner address design gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-gate-ready") {
    "debug listener native owner address design gate is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until stable native owner lifecycle, no-throw destructor, no-throw vtable, exception mapping, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native owner address design gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-owner-address-design-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeNoThrowVTableDesignGateEvidence {
  $noThrowVTableGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs"
  $ownerAddressGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerAddressDesignGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $preflightSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachNoThrowPreflight.cs"
  $attachVTableSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachVTableSafetyGate.cs"
  $borrowedTensorSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $noThrowVTableGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-nothrow-vtable-design-gate.md"
  $ownerAddressGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-address-design-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $noThrowVTableGateSourcePath = Join-Path $RepositoryRoot $noThrowVTableGateSourceRelativePath
  $ownerAddressGateSourcePath = Join-Path $RepositoryRoot $ownerAddressGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $preflightSourcePath = Join-Path $RepositoryRoot $preflightSourceRelativePath
  $attachVTableSafetyGateSourcePath = Join-Path $RepositoryRoot $attachVTableSafetyGateSourceRelativePath
  $borrowedTensorSafetyGateSourcePath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $noThrowVTableGateDocPath = Join-Path $RepositoryRoot $noThrowVTableGateDocRelativePath
  $ownerAddressGateDocPath = Join-Path $RepositoryRoot $ownerAddressGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-nothrow-vtable-design-gate",
    "TensorRtDebugListenerNativeNoThrowVTableDesignGate",
    "TensorRtDebugListenerNativeNoThrowVTableDesignGateResult",
    "Evaluate",
    "EvidenceKind=debug-listener-native-nothrow-vtable-design-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "DesignGateReady",
    "NativeOwnerAddressDesignGateReady",
    "NativeAttachNoThrowPreflightReady",
    "NativeAttachEntryLocated=False",
    "NativeOwnerLifecycleReady=False",
    "ManagedCallbackKeepAliveDesignReady",
    "NoThrowNativeDestructorReady=False",
    "NoThrowVTableDesignReady=False",
    "ExceptionToStatusMappingDesignReady=False",
    "BorrowedDebugTensorMetadataCopyDesignReady",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "NativeVTableTrampolineReady=False",
    "CallbackExceptionCaptureReady=False",
    "CallbackStatusMappingReady=False",
    "CallbackInFlightAccountingReady=False",
    "NativeVTableDesignReady",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($noThrowVTableGateSourcePath, $ownerAddressGateSourcePath, $precheckSourcePath, $preflightSourcePath, $attachVTableSafetyGateSourcePath, $borrowedTensorSafetyGateSourcePath, $attachDetachGateSourcePath, $ownerSourcePath, $smokePath, $noThrowVTableGateDocPath, $ownerAddressGateDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-nothrow-vtable-design-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native no-throw vtable design gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-gate-ready") {
    "debug listener native no-throw vtable design gate is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until native vtable trampoline, exception capture/status mapping, in-flight accounting, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native no-throw vtable design gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-nothrow-vtable-design-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeAttachEntryDesignGateEvidence {
  $attachEntryGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryDesignGate.cs"
  $noThrowVTableGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs"
  $ownerAddressGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerAddressDesignGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $preflightSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachNoThrowPreflight.cs"
  $attachVTableSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachVTableSafetyGate.cs"
  $borrowedTensorSafetyGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs"
  $attachDetachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerAttachDetachDesignGate.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $attachEntryGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-entry-design-gate.md"
  $noThrowVTableGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-nothrow-vtable-design-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $attachEntryGateSourcePath = Join-Path $RepositoryRoot $attachEntryGateSourceRelativePath
  $noThrowVTableGateSourcePath = Join-Path $RepositoryRoot $noThrowVTableGateSourceRelativePath
  $ownerAddressGateSourcePath = Join-Path $RepositoryRoot $ownerAddressGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $preflightSourcePath = Join-Path $RepositoryRoot $preflightSourceRelativePath
  $attachVTableSafetyGateSourcePath = Join-Path $RepositoryRoot $attachVTableSafetyGateSourceRelativePath
  $borrowedTensorSafetyGateSourcePath = Join-Path $RepositoryRoot $borrowedTensorSafetyGateSourceRelativePath
  $attachDetachGateSourcePath = Join-Path $RepositoryRoot $attachDetachGateSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $attachEntryGateDocPath = Join-Path $RepositoryRoot $attachEntryGateDocRelativePath
  $noThrowVTableGateDocPath = Join-Path $RepositoryRoot $noThrowVTableGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-attach-entry-design-gate",
    "TensorRtDebugListenerNativeAttachEntryDesignGate",
    "TensorRtDebugListenerNativeAttachEntryDesignGateResult",
    "Evaluate",
    "EvidenceKind=debug-listener-native-attach-entry-design-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "DesignGateReady",
    "NativeNoThrowVTableDesignGateReady",
    "NativeOwnerAddressDesignGateReady",
    "NativeAttachNoThrowPreflightReady",
    "NativeDetachEntryLocated=True",
    "NativeAttachEntryLocated=False",
    "LineSpecificAttachEntryDesignReady=False",
    "AttachEntryNoThrowReady=False",
    "AttachEntryVersionGuardReady=False",
    "AttachEntryOwnershipReady=False",
    "DetachBeforeReleaseReady=False",
    "NativeOwnerLifecycleReady=False",
    "NativeVTableDesignReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($attachEntryGateSourcePath, $noThrowVTableGateSourcePath, $ownerAddressGateSourcePath, $precheckSourcePath, $preflightSourcePath, $attachVTableSafetyGateSourcePath, $borrowedTensorSafetyGateSourcePath, $attachDetachGateSourcePath, $ownerSourcePath, $smokePath, $attachEntryGateDocPath, $noThrowVTableGateDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-attach-entry-design-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native attach entry design gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-gate-ready") {
    "debug listener native attach entry design gate is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until line-specific attach entry, no-throw boundary, version guard, ownership contract, detach-before-release ordering, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native attach entry design gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-attach-entry-design-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeDetachBeforeReleaseDesignGateEvidence {
  $detachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs"
  $attachEntryGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryDesignGate.cs"
  $noThrowVTableGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableDesignGate.cs"
  $ownerAddressGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerAddressDesignGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $detachGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-detach-before-release-design-gate.md"
  $attachEntryGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-entry-design-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $detachGateSourcePath = Join-Path $RepositoryRoot $detachGateSourceRelativePath
  $attachEntryGateSourcePath = Join-Path $RepositoryRoot $attachEntryGateSourceRelativePath
  $noThrowVTableGateSourcePath = Join-Path $RepositoryRoot $noThrowVTableGateSourceRelativePath
  $ownerAddressGateSourcePath = Join-Path $RepositoryRoot $ownerAddressGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $detachGateDocPath = Join-Path $RepositoryRoot $detachGateDocRelativePath
  $attachEntryGateDocPath = Join-Path $RepositoryRoot $attachEntryGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-detach-before-release-design-gate",
    "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate",
    "TensorRtDebugListenerNativeDetachBeforeReleaseDesignGateResult",
    "Evaluate",
    "EvidenceKind=debug-listener-native-detach-before-release-design-gate",
    "RuntimeEvidenceKind=design-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "DesignGateReady",
    "NativeAttachEntryDesignGateReady",
    "NativeNoThrowVTableDesignGateReady",
    "NativeOwnerAddressDesignGateReady",
    "NativeDetachEntryLocated=True",
    "NativeAttachEntryLocated=False",
    "LineSpecificAttachEntryDesignReady=False",
    "AttachEntryNoThrowReady=False",
    "AttachEntryVersionGuardReady=False",
    "AttachEntryOwnershipReady=False",
    "DetachBeforeReleaseReady=False",
    "ReleaseHookOrderingReady=False",
    "DisposeIdempotencyReady=False",
    "InFlightDrainBeforeReleaseReady=False",
    "CallbackStateUnpinAfterDetachReady=False",
    "DelegateUnpinAfterDetachReady=False",
    "NativeOwnerLifecycleReady=False",
    "NativeVTableDesignReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($detachGateSourcePath, $attachEntryGateSourcePath, $noThrowVTableGateSourcePath, $ownerAddressGateSourcePath, $precheckSourcePath, $ownerSourcePath, $smokePath, $detachGateDocPath, $attachEntryGateDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-detach-before-release-design-gate"
      evidenceKind = "design-gate"
      runtimeEvidenceKind = "design-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native detach-before-release design gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "design-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "design-gate-ready") {
    "debug listener native detach-before-release design gate is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until release hook ordering, dispose idempotency, in-flight drain, post-detach unpinning, native owner lifecycle, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native detach-before-release design gate evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-detach-before-release-design-gate"
    evidenceKind = "design-gate"
    runtimeEvidenceKind = "design-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeOwnerLifecycleDryRunEvidence {
  $dryRunSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs"
  $detachGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeDetachBeforeReleaseDesignGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $dryRunDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-lifecycle-dry-run.md"
  $detachGateDocRelativePath = "docs\articles\zh-cn\debug-listener-native-detach-before-release-design-gate.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $dryRunSourcePath = Join-Path $RepositoryRoot $dryRunSourceRelativePath
  $detachGateSourcePath = Join-Path $RepositoryRoot $detachGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $dryRunDocPath = Join-Path $RepositoryRoot $dryRunDocRelativePath
  $detachGateDocPath = Join-Path $RepositoryRoot $detachGateDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-owner-lifecycle-dry-run",
    "TensorRtDebugListenerNativeOwnerLifecycleDryRun",
    "TensorRtDebugListenerNativeOwnerLifecycleDryRunResult",
    "Evaluate",
    "RuntimeEvidenceKind=dry-run",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "DryRunReady",
    "NativeDetachBeforeReleaseDesignGateReady",
    "NativeAttachEntryDesignGateReady",
    "NativeNoThrowVTableDesignGateReady",
    "NativeOwnerAddressDesignGateReady",
    "NativeDetachEntryLocated=True",
    "NativeAttachEntryLocated=False",
    "OwnerId",
    "StableNativeOwnerIdentityReady=False",
    "NativeOwnerNonCopyableReady=False",
    "NativeOwnerDisposeOrderReady=False",
    "NativeOwnerReleaseHookReady=False",
    "NativeOwnerInFlightDrainReady=False",
    "DetachBeforeReleaseReady=False",
    "ReleaseHookOrderingReady=False",
    "DisposeIdempotencyReady=False",
    "InFlightDrainBeforeReleaseReady=False",
    "CallbackStateUnpinAfterDetachReady=False",
    "DelegateUnpinAfterDetachReady=False",
    "NoThrowNativeDestructorReady=False",
    "NativeOwnerLifecycleReady=False",
    "NativeVTableDesignReady=False",
    "ManagedCallbackKeepAliveDesignReady=True",
    "BorrowedDebugTensorMetadataCopyDesignReady=True",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "ProcessDebugTensorRuntimeReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "NativeOwnerLifecycleDryRunReady",
    "IDebugListener::processDebugTensor",
    "setDebugListener(non-null)",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($dryRunSourcePath, $detachGateSourcePath, $precheckSourcePath, $ownerSourcePath, $smokePath, $dryRunDocPath, $detachGateDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-owner-lifecycle-dry-run"
      evidenceKind = "dry-run"
      runtimeEvidenceKind = "dry-run"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native owner lifecycle dry-run evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "dry-run-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "dry-run-ready") {
    "debug listener native owner lifecycle dry-run is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until stable owner identity, non-copyable storage, release hook ordering, idempotency, in-flight drain, post-detach unpinning, no-throw destructor, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native owner lifecycle dry-run evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-owner-lifecycle-dry-run"
    evidenceKind = "dry-run"
    runtimeEvidenceKind = "dry-run"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeAttachEntryRuntimeScaffoldEvidence {
  $scaffoldSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs"
  $dryRunSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleDryRun.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $scaffoldDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-entry-runtime-scaffold.md"
  $dryRunDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-lifecycle-dry-run.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $scaffoldSourcePath = Join-Path $RepositoryRoot $scaffoldSourceRelativePath
  $dryRunSourcePath = Join-Path $RepositoryRoot $dryRunSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $scaffoldDocPath = Join-Path $RepositoryRoot $scaffoldDocRelativePath
  $dryRunDocPath = Join-Path $RepositoryRoot $dryRunDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-attach-entry-runtime-scaffold",
    "TensorRtDebugListenerNativeAttachEntryRuntimeScaffold",
    "TensorRtDebugListenerNativeAttachEntryRuntimeScaffoldResult",
    "Evaluate",
    "RuntimeEvidenceKind=scaffold",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "RuntimeScaffoldReady",
    "NativeOwnerLifecycleDryRunReady",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "AttachEntryParameterShapeReady=True",
    "AttachEntryVersionGuardReady=True",
    "AttachEntryNoThrowBoundaryReady=True",
    "AttachEntryOwnershipDiagnosticsReady=True",
    "StableNativeOwnerIdentityReady=False",
    "NativeOwnerNonCopyableReady=False",
    "NoThrowNativeDestructorReady=False",
    "NativeOwnerLifecycleReady=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "NativeAttachEntryRuntimeScaffoldReady",
    "IDebugListener::processDebugTensor",
    "setDebugListener(non-null)",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($scaffoldSourcePath, $dryRunSourcePath, $precheckSourcePath, $ownerSourcePath, $smokePath, $scaffoldDocPath, $dryRunDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-attach-entry-runtime-scaffold"
      evidenceKind = "scaffold"
      runtimeEvidenceKind = "scaffold"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native attach entry runtime scaffold evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "scaffold-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "scaffold-ready") {
    "debug listener native attach entry runtime scaffold is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until real non-null attach entry, stable owner identity, non-copyable storage, no-throw destructor, processDebugTensor runtime, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native attach entry runtime scaffold evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-attach-entry-runtime-scaffold"
    evidenceKind = "scaffold"
    runtimeEvidenceKind = "scaffold"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeAttachEntryMinimalSafetyEvidence {
  $minimalSafetySourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs"
  $scaffoldSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs"
  $lifecycleGateSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleGate.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $nativeMinimalSafetyRelativePath = "native\src\tensorrt\common\debug_listener_native_attach_entry_minimal_safety.inc"
  $nativeTrt8RelativePath = "native\src\tensorrt\v8\api.cpp"
  $nativeTrt10RelativePath = "native\src\tensorrt\v10\api.cpp"
  $nativeTrt11RelativePath = "native\src\tensorrt\v11\api.cpp"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $minimalSafetyDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-entry-minimal-safety.md"
  $scaffoldDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-entry-runtime-scaffold.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $minimalSafetySourcePath = Join-Path $RepositoryRoot $minimalSafetySourceRelativePath
  $scaffoldSourcePath = Join-Path $RepositoryRoot $scaffoldSourceRelativePath
  $lifecycleGateSourcePath = Join-Path $RepositoryRoot $lifecycleGateSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $nativeMinimalSafetyPath = Join-Path $RepositoryRoot $nativeMinimalSafetyRelativePath
  $nativeTrt8Path = Join-Path $RepositoryRoot $nativeTrt8RelativePath
  $nativeTrt10Path = Join-Path $RepositoryRoot $nativeTrt10RelativePath
  $nativeTrt11Path = Join-Path $RepositoryRoot $nativeTrt11RelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $minimalSafetyDocPath = Join-Path $RepositoryRoot $minimalSafetyDocRelativePath
  $scaffoldDocPath = Join-Path $RepositoryRoot $scaffoldDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-attach-entry-minimal-safety",
    "TensorRtDebugListenerNativeAttachEntryMinimalSafety",
    "TensorRtDebugListenerNativeAttachEntryMinimalSafetyResult",
    "DebugListenerNativeAttachEntryMinimalSafety",
    "Evaluate",
    "RuntimeEvidenceKind=minimal-safety",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "MinimalSafetyReady=True",
    "RuntimeScaffoldReady=True",
    "LifecycleGateReady=True",
    "LifecyclePointerFree=True",
    "NativeAttachEntryLocated=True",
    "NativeDetachEntryLocated=True",
    "AttachEntryParameterShapeReady=True",
    "AttachEntryNoThrowReady=True",
    "AttachEntryVersionGuardReady=True",
    "AttachEntryOwnershipDiagnosticsReady=True",
    "SetDebugListenerNonNullEnabled=False",
    "NonNullAttachStillDisabled=True",
    "NativeAttachWouldBeBlocked=True",
    "ProcessDebugTensorRuntimeReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "ReasonNativeAttachStillBlocked",
    "debug_listener_native_attach_entry_minimal_safety.inc",
    "DebugListenerNativeAttachEntryMinimalSafety final",
    "can_call_set_debug_listener_non_null",
    "setDebugListener(non-null)",
    "IDebugListener::processDebugTensor",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($minimalSafetySourcePath, $scaffoldSourcePath, $lifecycleGateSourcePath, $precheckSourcePath, $nativeMinimalSafetyPath, $nativeTrt8Path, $nativeTrt10Path, $nativeTrt11Path, $smokePath, $minimalSafetyDocPath, $scaffoldDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-attach-entry-minimal-safety"
      evidenceKind = "minimal-safety"
      runtimeEvidenceKind = "minimal-safety"
      source = "source-native-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      minimalSafetyReady = $false
      nativeAttachEntryLocated = $false
      setDebugListenerNonNullEnabled = $false
      nativeAttachWouldBeBlocked = $true
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native attach entry minimal-safety evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "minimal-safety-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "minimal-safety-ready") {
    "debug listener native attach entry minimal-safety is source-visible, pointer-free, version-guarded, no-throw, and still blocks setDebugListener(non-null), processDebugTensor runtime, and real-callback-runtime promotion."
  }
  else {
    "debug listener native attach entry minimal-safety evidence is incomplete; inspect source/native/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-attach-entry-minimal-safety"
    evidenceKind = "minimal-safety"
    runtimeEvidenceKind = "minimal-safety"
    source = "source-native-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    minimalSafetyReady = $true
    nativeAttachEntryLocated = $true
    setDebugListenerNonNullEnabled = $false
    nativeAttachWouldBeBlocked = $true
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeOwnerStableIdentityEvidence {
  $identitySourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerStableIdentity.cs"
  $scaffoldSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryRuntimeScaffold.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $ownerSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackOwner.cs"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $identityDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-stable-identity.md"
  $scaffoldDocRelativePath = "docs\articles\zh-cn\debug-listener-native-attach-entry-runtime-scaffold.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $identitySourcePath = Join-Path $RepositoryRoot $identitySourceRelativePath
  $scaffoldSourcePath = Join-Path $RepositoryRoot $scaffoldSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $ownerSourcePath = Join-Path $RepositoryRoot $ownerSourceRelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $identityDocPath = Join-Path $RepositoryRoot $identityDocRelativePath
  $scaffoldDocPath = Join-Path $RepositoryRoot $scaffoldDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-owner-stable-identity",
    "TensorRtDebugListenerNativeOwnerStableIdentity",
    "TensorRtDebugListenerNativeOwnerStableIdentityResult",
    "Evaluate",
    "RuntimeEvidenceKind=identity-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeAttachEntryRuntimeScaffoldReady=True",
    "StableNativeOwnerIdentityReady=True",
    "OwnerIdentityDiagnosticsReady=True",
    "OwnerIdentityPointerFree=True",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "NativeOwnerNonCopyableReady=False",
    "NoThrowNativeDestructorReady=False",
    "NativeOwnerLifecycleReady=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "NativeOwnerStableIdentityReady",
    "IDebugListener::processDebugTensor",
    "setDebugListener(non-null)",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($identitySourcePath, $scaffoldSourcePath, $precheckSourcePath, $ownerSourcePath, $smokePath, $identityDocPath, $scaffoldDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-owner-stable-identity"
      evidenceKind = "identity-gate"
      runtimeEvidenceKind = "identity-gate"
      source = "source-smoke-docs"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      stableNativeOwnerIdentityReady = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native owner stable identity evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "identity-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "identity-gate-ready") {
    "debug listener native owner stable identity is documented, pointer-free, and explicitly blocked from native attach/runtime proof promotion until non-copyable storage, no-throw destructor, non-null attach entry, processDebugTensor runtime, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native owner stable identity evidence is incomplete; inspect source/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-owner-stable-identity"
    evidenceKind = "identity-gate"
    runtimeEvidenceKind = "identity-gate"
    source = "source-smoke-docs"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    stableNativeOwnerIdentityReady = $true
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeOwnerNonCopyableStorageEvidence {
  $storageSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs"
  $stableIdentitySourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerStableIdentity.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $nativeStorageScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_owner_noncopyable_storage.inc"
  $nativeTrt8RelativePath = "native\src\tensorrt\v8\api.cpp"
  $nativeTrt10RelativePath = "native\src\tensorrt\v10\api.cpp"
  $nativeTrt11RelativePath = "native\src\tensorrt\v11\api.cpp"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $storageDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-noncopyable-storage.md"
  $stableIdentityDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-stable-identity.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"

  $storageSourcePath = Join-Path $RepositoryRoot $storageSourceRelativePath
  $stableIdentitySourcePath = Join-Path $RepositoryRoot $stableIdentitySourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $nativeStorageScaffoldPath = Join-Path $RepositoryRoot $nativeStorageScaffoldRelativePath
  $nativeTrt8Path = Join-Path $RepositoryRoot $nativeTrt8RelativePath
  $nativeTrt10Path = Join-Path $RepositoryRoot $nativeTrt10RelativePath
  $nativeTrt11Path = Join-Path $RepositoryRoot $nativeTrt11RelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $storageDocPath = Join-Path $RepositoryRoot $storageDocRelativePath
  $stableIdentityDocPath = Join-Path $RepositoryRoot $stableIdentityDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-owner-noncopyable-storage",
    "TensorRtDebugListenerNativeOwnerNonCopyableStorage",
    "TensorRtDebugListenerNativeOwnerNonCopyableStorageResult",
    "Evaluate",
    "RuntimeEvidenceKind=storage-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeOwnerStableIdentityReady=True",
    "OwnerIdentityDiagnosticsReady=True",
    "OwnerIdentityPointerFree=True",
    "NativeOwnerNonCopyableReady=True",
    "NativeOwnerCopyBlocked=True",
    "NativeOwnerMoveBlocked=True",
    "NativeOwnerAddressExposed=False",
    "NativeOwnerPointerProduced=False",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "NoThrowNativeDestructorReady=False",
    "NativeOwnerLifecycleReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DebugListenerNativeOwnerNonCopyableStorage=",
    "DebugListenerNativeOwnerNonCopyableStorage",
    "DebugListenerNativeOwnerNonCopyableStorageResult",
    "DebugListenerNativeOwnerNonCopyableStorage(const DebugListenerNativeOwnerNonCopyableStorage&) = delete",
    "operator=(const DebugListenerNativeOwnerNonCopyableStorage&) = delete",
    "DebugListenerNativeOwnerNonCopyableStorage(DebugListenerNativeOwnerNonCopyableStorage&&) = delete",
    "operator=(DebugListenerNativeOwnerNonCopyableStorage&&) = delete",
    "std::is_nothrow_destructible",
    "debug_listener_native_owner_noncopyable_storage.inc",
    "IDebugListener::processDebugTensor",
    "setDebugListener(non-null)",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($storageSourcePath, $stableIdentitySourcePath, $precheckSourcePath, $nativeStorageScaffoldPath, $nativeTrt8Path, $nativeTrt10Path, $nativeTrt11Path, $smokePath, $storageDocPath, $stableIdentityDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-owner-noncopyable-storage"
      evidenceKind = "storage-gate"
      runtimeEvidenceKind = "storage-gate"
      source = "source-smoke-docs-native-scaffold"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      nativeOwnerNonCopyableReady = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native owner non-copyable storage evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "storage-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "storage-gate-ready") {
    "debug listener native owner non-copyable storage is documented, source-visible, pointer-free, and explicitly blocked from native attach/runtime proof promotion until non-null attach entry, no-throw destructor, owner lifecycle, processDebugTensor runtime, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native owner non-copyable storage evidence is incomplete; inspect source/native scaffold/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-owner-noncopyable-storage"
    evidenceKind = "storage-gate"
    runtimeEvidenceKind = "storage-gate"
    source = "source-smoke-docs-native-scaffold"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    nativeOwnerNonCopyableReady = $true
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeNoThrowDestructorEvidence {
  $destructorSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowDestructor.cs"
  $storageSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $nativeDestructorScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_nothrow_destructor.inc"
  $nativeStorageScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_owner_noncopyable_storage.inc"
  $nativeTrt8RelativePath = "native\src\tensorrt\v8\api.cpp"
  $nativeTrt10RelativePath = "native\src\tensorrt\v10\api.cpp"
  $nativeTrt11RelativePath = "native\src\tensorrt\v11\api.cpp"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $destructorDocRelativePath = "docs\articles\zh-cn\debug-listener-native-nothrow-destructor.md"
  $storageDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-noncopyable-storage.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"

  $destructorSourcePath = Join-Path $RepositoryRoot $destructorSourceRelativePath
  $storageSourcePath = Join-Path $RepositoryRoot $storageSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $nativeDestructorScaffoldPath = Join-Path $RepositoryRoot $nativeDestructorScaffoldRelativePath
  $nativeStorageScaffoldPath = Join-Path $RepositoryRoot $nativeStorageScaffoldRelativePath
  $nativeTrt8Path = Join-Path $RepositoryRoot $nativeTrt8RelativePath
  $nativeTrt10Path = Join-Path $RepositoryRoot $nativeTrt10RelativePath
  $nativeTrt11Path = Join-Path $RepositoryRoot $nativeTrt11RelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $destructorDocPath = Join-Path $RepositoryRoot $destructorDocRelativePath
  $storageDocPath = Join-Path $RepositoryRoot $storageDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-nothrow-destructor",
    "TensorRtDebugListenerNativeNoThrowDestructor",
    "TensorRtDebugListenerNativeNoThrowDestructorResult",
    "Evaluate",
    "RuntimeEvidenceKind=destructor-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeOwnerNonCopyableStorageReady=True",
    "NativeOwnerNonCopyableReady=True",
    "NativeOwnerCopyBlocked=True",
    "NativeOwnerMoveBlocked=True",
    "NativeOwnerAddressExposed=False",
    "NativeOwnerPointerProduced=False",
    "DestructorNoThrowScaffoldReady=True",
    "DestructorExceptionEscapeBlocked=True",
    "DestructorAddressExposed=False",
    "DestructorPointerProduced=False",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "NoThrowNativeDestructorReady=True",
    "NativeOwnerLifecycleReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DebugListenerNativeNoThrowDestructor=",
    "DebugListenerNativeNoThrowDestructor",
    "DebugListenerNativeNoThrowDestructorResult",
    "NativeNoThrowDestructorGateReady",
    "DebugListenerNativeNoThrowDestructor(const DebugListenerNativeNoThrowDestructor&) = delete",
    "operator=(const DebugListenerNativeNoThrowDestructor&) = delete",
    "DebugListenerNativeNoThrowDestructor(DebugListenerNativeNoThrowDestructor&&) = delete",
    "operator=(DebugListenerNativeNoThrowDestructor&&) = delete",
    "std::is_nothrow_destructible",
    "debug_listener_native_nothrow_destructor.inc",
    "IDebugListener::processDebugTensor",
    "setDebugListener(non-null)",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($destructorSourcePath, $storageSourcePath, $precheckSourcePath, $nativeDestructorScaffoldPath, $nativeStorageScaffoldPath, $nativeTrt8Path, $nativeTrt10Path, $nativeTrt11Path, $smokePath, $destructorDocPath, $storageDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-nothrow-destructor"
      evidenceKind = "destructor-gate"
      runtimeEvidenceKind = "destructor-gate"
      source = "source-smoke-docs-native-scaffold"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      noThrowNativeDestructorReady = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native no-throw destructor evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "destructor-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "destructor-gate-ready") {
    "debug listener native no-throw destructor scaffold is documented, source-visible, pointer-free, and explicitly blocked from native attach/runtime proof promotion until non-null attach entry, native owner lifecycle, processDebugTensor runtime, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native no-throw destructor evidence is incomplete; inspect source/native scaffold/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-nothrow-destructor"
    evidenceKind = "destructor-gate"
    runtimeEvidenceKind = "destructor-gate"
    source = "source-smoke-docs-native-scaffold"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    noThrowNativeDestructorReady = $true
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-DebugListenerNativeOwnerLifecycleGateEvidence {
  $lifecycleSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleGate.cs"
  $destructorSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowDestructor.cs"
  $storageSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerNonCopyableStorage.cs"
  $precheckSourceRelativePath = "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs"
  $nativeLifecycleScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_owner_lifecycle_gate.inc"
  $nativeDestructorScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_nothrow_destructor.inc"
  $nativeStorageScaffoldRelativePath = "native\src\tensorrt\common\debug_listener_native_owner_noncopyable_storage.inc"
  $nativeTrt8RelativePath = "native\src\tensorrt\v8\api.cpp"
  $nativeTrt10RelativePath = "native\src\tensorrt\v10\api.cpp"
  $nativeTrt11RelativePath = "native\src\tensorrt\v11\api.cpp"
  $smokeRelativePath = "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs"
  $lifecycleDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-lifecycle-gate.md"
  $destructorDocRelativePath = "docs\articles\zh-cn\debug-listener-native-nothrow-destructor.md"
  $storageDocRelativePath = "docs\articles\zh-cn\debug-listener-native-owner-noncopyable-storage.md"
  $precheckDocRelativePath = "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md"
  $callbackGateRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $schemaRelativePath = "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $indexRelativePath = "docs\index.md"
  $tocRelativePath = "docs\toc.yml"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $smokeReadmeRelativePath = "smoke\README.md"
  $bridgeConsumerRelativePath = "eng\Test-BridgePackageConsumer.ps1"
  $packageConsumerRelativePath = "eng\Test-PackageConsumer.ps1"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"

  $lifecycleSourcePath = Join-Path $RepositoryRoot $lifecycleSourceRelativePath
  $destructorSourcePath = Join-Path $RepositoryRoot $destructorSourceRelativePath
  $storageSourcePath = Join-Path $RepositoryRoot $storageSourceRelativePath
  $precheckSourcePath = Join-Path $RepositoryRoot $precheckSourceRelativePath
  $nativeLifecycleScaffoldPath = Join-Path $RepositoryRoot $nativeLifecycleScaffoldRelativePath
  $nativeDestructorScaffoldPath = Join-Path $RepositoryRoot $nativeDestructorScaffoldRelativePath
  $nativeStorageScaffoldPath = Join-Path $RepositoryRoot $nativeStorageScaffoldRelativePath
  $nativeTrt8Path = Join-Path $RepositoryRoot $nativeTrt8RelativePath
  $nativeTrt10Path = Join-Path $RepositoryRoot $nativeTrt10RelativePath
  $nativeTrt11Path = Join-Path $RepositoryRoot $nativeTrt11RelativePath
  $smokePath = Join-Path $RepositoryRoot $smokeRelativePath
  $lifecycleDocPath = Join-Path $RepositoryRoot $lifecycleDocRelativePath
  $destructorDocPath = Join-Path $RepositoryRoot $destructorDocRelativePath
  $storageDocPath = Join-Path $RepositoryRoot $storageDocRelativePath
  $precheckDocPath = Join-Path $RepositoryRoot $precheckDocRelativePath
  $callbackGatePath = Join-Path $RepositoryRoot $callbackGateRelativePath
  $schemaPath = Join-Path $RepositoryRoot $schemaRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $smokeReadmePath = Join-Path $RepositoryRoot $smokeReadmeRelativePath
  $bridgeConsumerPath = Join-Path $RepositoryRoot $bridgeConsumerRelativePath
  $packageConsumerPath = Join-Path $RepositoryRoot $packageConsumerRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "debug-listener-native-owner-lifecycle-gate",
    "TensorRtDebugListenerNativeOwnerLifecycleGate",
    "TensorRtDebugListenerNativeOwnerLifecycleGateResult",
    "Evaluate",
    "RuntimeEvidenceKind=lifecycle-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeNoThrowDestructorGateReady=True",
    "ManagedDisposeSnapshotReady=True",
    "LifecycleScaffoldReady=True",
    "ReleaseHookOrderingGateReady=True",
    "DisposeIdempotencyGateReady=True",
    "InFlightDrainGateReady=True",
    "CallbackStateUnpinAfterDetachGateReady=True",
    "DelegateUnpinAfterDetachGateReady=True",
    "LifecycleAddressExposed=False",
    "LifecyclePointerProduced=False",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "NoThrowNativeDestructorReady=True",
    "ReleaseHookOrderingReady=False",
    "DisposeIdempotencyReady=False",
    "InFlightDrainBeforeReleaseReady=False",
    "CallbackStateUnpinAfterDetachReady=False",
    "DelegateUnpinAfterDetachReady=False",
    "LifecycleGateReady=True",
    "NativeOwnerLifecycleReady=False",
    "NativeVTableDesignReady=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DebugListenerNativeOwnerLifecycleGate=",
    "DebugListenerNativeOwnerLifecycleGate",
    "DebugListenerNativeOwnerLifecycleGateResult",
    "DebugListenerNativeOwnerLifecycleGate(const DebugListenerNativeOwnerLifecycleGate&) = delete",
    "operator=(const DebugListenerNativeOwnerLifecycleGate&) = delete",
    "DebugListenerNativeOwnerLifecycleGate(DebugListenerNativeOwnerLifecycleGate&&) = delete",
    "operator=(DebugListenerNativeOwnerLifecycleGate&&) = delete",
    "request_detach_before_release",
    "request_release",
    "can_unpin_after_detach",
    "std::is_nothrow_destructible",
    "debug_listener_native_owner_lifecycle_gate.inc",
    "IDebugListener::processDebugTensor",
    "setDebugListener(non-null)",
    "not proof"
  )

  $requiredDeferredRows = @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )

  $evidencePaths = @($lifecycleSourcePath, $destructorSourcePath, $storageSourcePath, $precheckSourcePath, $nativeLifecycleScaffoldPath, $nativeDestructorScaffoldPath, $nativeStorageScaffoldPath, $nativeTrt8Path, $nativeTrt10Path, $nativeTrt11Path, $smokePath, $lifecycleDocPath, $destructorDocPath, $storageDocPath, $precheckDocPath, $callbackGatePath, $schemaPath, $latestPath, $indexPath, $tocPath, $runtimeSplitReadmePath, $smokeReadmePath, $bridgeConsumerPath, $packageConsumerPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "debug-listener-native-owner-lifecycle-gate"
      evidenceKind = "lifecycle-gate"
      runtimeEvidenceKind = "lifecycle-gate"
      source = "source-smoke-docs-native-scaffold"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      lifecycleGateReady = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "debug listener native owner lifecycle gate evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "lifecycle-gate-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "lifecycle-gate-ready") {
    "debug listener native owner lifecycle gate is documented, source-visible, pointer-free, and explicitly blocked from native attach/runtime proof promotion until non-null attach entry, complete native vtable, processDebugTensor runtime, and full package consumer runtime evidence exist."
  }
  else {
    "debug listener native owner lifecycle gate evidence is incomplete; inspect source/native scaffold/smoke/doc markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "debug-listener-native-owner-lifecycle-gate"
    evidenceKind = "lifecycle-gate"
    runtimeEvidenceKind = "lifecycle-gate"
    source = "source-smoke-docs-native-scaffold"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    lifecycleGateReady = $true
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function Get-CallbackDeferredEvidenceRows {
  return @(
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"'
  )
}

function New-CallbackOwnerClosureMatrixEvidence {
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $relativePaths = @(
    "src\JYPPX.TensorRtSharp\TensorRtCallbackOwnerClosureMatrix.cs",
    "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
    "smoke\README.md",
    "docs\articles\zh-cn\callback-owner-closure-matrix.md",
    "docs\articles\zh-cn\callback-allocator-boundary-guide.md",
    "docs\articles\zh-cn\callback-allocator-safety-bridge-roadmap.md",
    "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
    "docs\toc.yml",
    "docs\index.md",
    "eng\Test-BridgePackageConsumer.ps1",
    "eng\Test-RuntimePackageReadiness.ps1",
    $comparisonRelativePath
  )
  $evidencePaths = @($relativePaths | ForEach-Object { Join-Path $RepositoryRoot $_ })
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath
  $requiredDeferredRows = @(Get-CallbackDeferredEvidenceRows) + @(
    '"IStreamReader","read","IStreamReader::read","other","deferred-only"',
    '"IStreamReaderV2","read","IStreamReaderV2::read","other","deferred-only"',
    '"IStreamReaderV2","seek","IStreamReaderV2::seek","other","deferred-only"',
    '"IStreamWriter","write","IStreamWriter::write","other","deferred-only"'
  )
  $requiredMarkers = @(
    "callback-owner-closure-matrix",
    "TensorRtCallbackOwnerClosureMatrix",
    "TensorRtCallbackOwnerClosureMatrix.Evaluate",
    "TensorRtCallbackOwnerClosureMatrixResult",
    "TensorRtCallbackOwnerClosureMatrixRow",
    "TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface",
    "CallbackOwnerClosureMatrix=",
    "RuntimeEvidenceKind=closure-matrix",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "FamilyCount=5",
    "DesignGateReadyFamilyCount",
    "ClosureReadyFamilyCount",
    "RuntimeProofAttemptReadyFamilyCount",
    "PackageConsumerRuntimeProofReadyFamilyCount",
    "PackageConsumerRuntimeProofRequired",
    "PackageConsumerRuntimeProofReady",
    "RuntimeProofBlocked=True",
    "DeferredRowsStillRequired=True",
    "GpuAllocator",
    "GpuAsyncAllocator",
    "OutputAllocator",
    "DebugListener",
    "StreamReaderWriter",
    "not proof",
    "不能作为真实 TensorRT callback runtime proof"
  )

  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "callback-owner-closure-matrix"
      evidenceKind = "closure-matrix"
      runtimeEvidenceKind = "closure-matrix"
      source = "source-smoke-docs-bridge-readiness"
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      familyCount = 5
      designGateReadyFamilyCount = 0
      closureReadyFamilyCount = 0
      runtimeProofAttemptReadyFamilyCount = 0
      packageConsumerRuntimeProofReadyFamilyCount = 0
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      deferredRowsStillRequired = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "callback owner closure matrix evidence files are missing."
    }
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($marker in @($requiredMarkers)) {
    if ($combined.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { "closure-matrix-ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "closure-matrix-ready") {
    "callback owner closure matrix is connected to source, smoke, docs, bridge consumer surface, runtime readiness, and deferred-row evidence; it remains closure-matrix / non-proof."
  }
  else {
    "callback owner closure matrix evidence is incomplete; inspect source/smoke/docs/bridge/readiness markers and callback deferred rows."
  }

  return [pscustomobject]@{
    status = $status
    marker = "callback-owner-closure-matrix"
    evidenceKind = "closure-matrix"
    runtimeEvidenceKind = "closure-matrix"
    source = "source-smoke-docs-bridge-readiness"
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    familyCount = 5
    designGateReadyFamilyCount = 3
    closureReadyFamilyCount = 0
    runtimeProofAttemptReadyFamilyCount = 0
    packageConsumerRuntimeProofReadyFamilyCount = 0
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    deferredRowsStillRequired = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-SourceVisibleDebugListenerGateEvidence {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Marker,
    [Parameter(Mandatory = $true)]
    [string]$EvidenceKind,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeEvidenceKind,
    [Parameter(Mandatory = $true)]
    [string]$ReadyStatus,
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [Parameter(Mandatory = $true)]
    [string[]]$EvidenceRelativePaths,
    [Parameter(Mandatory = $true)]
    [string[]]$RequiredMarkers,
    [Parameter(Mandatory = $true)]
    [string]$ReadyDiagnostic,
    [Parameter(Mandatory = $true)]
    [string]$IncompleteDiagnostic,
    [Parameter(Mandatory = $true)]
    [hashtable]$ReadyProperties,
    [Parameter(Mandatory = $true)]
    [hashtable]$MissingProperties
  )

  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $relativePaths = @($EvidenceRelativePaths) + @($comparisonRelativePath)
  $evidencePaths = @($relativePaths | ForEach-Object { Join-Path $RepositoryRoot $_ })
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath
  $requiredDeferredRows = @(Get-CallbackDeferredEvidenceRows)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    $missingResult = [ordered]@{
      status = "missing"
      marker = $Marker
      evidenceKind = $EvidenceKind
      runtimeEvidenceKind = $RuntimeEvidenceKind
      source = $Source
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($RequiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDeferredRowEvidence = $false
      canImplementNativeAttach = $false
      canAttemptRuntimeProof = $false
      runtimeProofBlocked = $true
      isRealCallbackRuntimeProof = $false
      diagnostic = "$Marker evidence files are missing."
    }
    foreach ($key in @($MissingProperties.Keys)) {
      $missingResult[$key] = $MissingProperties[$key]
    }

    return [pscustomobject]$missingResult
  }

  $combined = ""
  foreach ($path in @($evidencePaths)) {
    $combined += (Get-Content -LiteralPath $path -Raw -Encoding utf8) + "`n"
  }

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  foreach ($requiredMarker in @($RequiredMarkers)) {
    if ($combined.IndexOf($requiredMarker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($requiredMarker)
    }
  }

  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($missingMarkers.Count -eq 0 -and $hasDeferredRowEvidence) { $ReadyStatus } else { "incomplete" }
  $diagnostic = if ($status -eq $ReadyStatus) { $ReadyDiagnostic } else { $IncompleteDiagnostic }

  $result = [ordered]@{
    status = $status
    marker = $Marker
    evidenceKind = $EvidenceKind
    runtimeEvidenceKind = $RuntimeEvidenceKind
    source = $Source
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    canImplementNativeAttach = $false
    canAttemptRuntimeProof = $false
    runtimeProofBlocked = $true
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
  foreach ($key in @($ReadyProperties.Keys)) {
    $result[$key] = $ReadyProperties[$key]
  }

  return [pscustomobject]$result
}

function New-DebugListenerNativeAttachBridgeShapeGateEvidence {
  $requiredMarkers = @(
    "debug-listener-native-attach-bridge-shape-gate",
    "TensorRtDebugListenerNativeAttachBridgeShapeGate",
    "TensorRtDebugListenerNativeAttachBridgeShapeGateResult",
    "Evaluate",
    "RuntimeEvidenceKind=attach-bridge-shape-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeOwnerLifecycleGateReady=True",
    "AttachBridgeShapeReady=True",
    "AttachBridgeNoThrowBoundaryReady=True",
    "AttachBridgeVersionGuardReady=True",
    "AttachBridgeOwnershipDiagnosticsReady=True",
    "AttachBridgePointerFree=True",
    "SetDebugListenerNonNullEnabled=False",
    "NonNullAttachStillDisabled=True",
    "NativeAttachEntryLocated=False",
    "NativeDetachEntryLocated=True",
    "NativeOwnerLifecycleReady=False",
    "NativeVTableDesignReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DebugListenerNativeAttachBridgeShapeGate=",
    "DebugListenerNativeAttachBridgeShapeGate",
    "DebugListenerNativeAttachBridgeShapeGateResult",
    "debug_listener_native_attach_bridge_shape_gate.inc",
    "DebugListenerNativeAttachBridgeShapeGate(const DebugListenerNativeAttachBridgeShapeGate&) = delete",
    "setDebugListener(non-null)",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-native-attach-bridge-shape-gate" `
    -EvidenceKind "attach-bridge-shape-gate" `
    -RuntimeEvidenceKind "attach-bridge-shape-gate" `
    -ReadyStatus "attach-bridge-shape-gate-ready" `
    -Source "source-smoke-docs-native-scaffold" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs",
      "native\src\tensorrt\common\debug_listener_native_attach_bridge_shape_gate.inc",
      "native\src\tensorrt\common\debug_listener_native_owner_lifecycle_gate.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-native-attach-bridge-shape-gate.md",
      "docs\articles\zh-cn\debug-listener-native-owner-lifecycle-gate.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener native attach bridge shape gate is source-visible, version-guarded, pointer-free, and still blocks setDebugListener(non-null), native vtable, processDebugTensor runtime, and real-callback-runtime promotion." `
    -IncompleteDiagnostic "debug listener native attach bridge shape gate evidence is incomplete; inspect source/native scaffold/smoke/doc markers and callback deferred rows." `
    -ReadyProperties @{
      attachBridgeShapeGateReady = $true
      nativeAttachEntryLocated = $false
    } `
    -MissingProperties @{
      attachBridgeShapeGateReady = $false
      nativeAttachEntryLocated = $false
    }
}

function New-DebugListenerExceptionStatusMappingGateEvidence {
  $requiredMarkers = @(
    "debug-listener-exception-status-mapping-gate",
    "TensorRtDebugListenerExceptionStatusMappingGate",
    "TensorRtDebugListenerExceptionStatusMappingGateResult",
    "Evaluate",
    "RuntimeEvidenceKind=exception-status-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "AttachBridgeShapeGateReady=True",
    "ManagedCallbackExceptionCaptureReady=True",
    "NativeCallbackExceptionCaptureReady=True",
    "CallbackStatusMappingGateReady=True",
    "ExceptionEscapeBlocked=True",
    "DiagnosticCopyReady=True",
    "MappingAddressExposed=False",
    "MappingPointerProduced=False",
    "NativeAttachEntryLocated=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DebugListenerExceptionStatusMappingGate=",
    "DebugListenerExceptionStatusMappingGate",
    "DebugListenerExceptionStatusMappingGateResult",
    "debug_listener_exception_status_mapping_gate.inc",
    "map_exception_to_status",
    "setDebugListener(non-null)",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-exception-status-mapping-gate" `
    -EvidenceKind "exception-status-gate" `
    -RuntimeEvidenceKind "exception-status-gate" `
    -ReadyStatus "exception-status-gate-ready" `
    -Source "source-smoke-docs-native-scaffold" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerExceptionStatusMappingGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs",
      "native\src\tensorrt\common\debug_listener_exception_status_mapping_gate.inc",
      "native\src\tensorrt\common\debug_listener_native_attach_bridge_shape_gate.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-exception-status-mapping-gate.md",
      "docs\articles\zh-cn\debug-listener-native-attach-bridge-shape-gate.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener exception/status mapping gate is source-visible, no-throw mapped, pointer-free, and still blocks real runtime proof until non-null attach and processDebugTensor runtime evidence exist." `
    -IncompleteDiagnostic "debug listener exception/status mapping gate evidence is incomplete; inspect source/native scaffold/smoke/doc markers and callback deferred rows." `
    -ReadyProperties @{
      exceptionStatusMappingGateReady = $true
      nativeAttachEntryLocated = $false
    } `
    -MissingProperties @{
      exceptionStatusMappingGateReady = $false
      nativeAttachEntryLocated = $false
    }
}

function New-DebugListenerInFlightAccountingGateEvidence {
  $requiredMarkers = @(
    "debug-listener-inflight-accounting-gate",
    "TensorRtDebugListenerInFlightAccountingGate",
    "TensorRtDebugListenerInFlightAccountingGateResult",
    "Evaluate",
    "RuntimeEvidenceKind=inflight-accounting-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "ExceptionStatusMappingGateReady=True",
    "CallbackEnterAccountingGateReady=True",
    "CallbackLeaveAccountingGateReady=True",
    "CallbackInFlightNeverNegativeReady=True",
    "ReleaseAfterDrainGateReady=True",
    "CallbackStateUnpinAfterDrainGateReady=True",
    "AccountingAddressExposed=False",
    "AccountingPointerProduced=False",
    "NativeAttachEntryLocated=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DebugListenerInFlightAccountingGate=",
    "DebugListenerInFlightAccountingGate",
    "DebugListenerInFlightAccountingGateResult",
    "debug_listener_inflight_accounting_gate.inc",
    "DebugListenerInFlightAccountingGate",
    "setDebugListener(non-null)",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-inflight-accounting-gate" `
    -EvidenceKind "inflight-accounting-gate" `
    -RuntimeEvidenceKind "inflight-accounting-gate" `
    -ReadyStatus "inflight-accounting-gate-ready" `
    -Source "source-smoke-docs-native-scaffold" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerInFlightAccountingGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerExceptionStatusMappingGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs",
      "native\src\tensorrt\common\debug_listener_inflight_accounting_gate.inc",
      "native\src\tensorrt\common\debug_listener_exception_status_mapping_gate.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-inflight-accounting-gate.md",
      "docs\articles\zh-cn\debug-listener-exception-status-mapping-gate.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener in-flight accounting gate is source-visible, pointer-free, and keeps callback drain/unpin evidence separate from real TensorRT callback execution proof." `
    -IncompleteDiagnostic "debug listener in-flight accounting gate evidence is incomplete; inspect source/native scaffold/smoke/doc markers and callback deferred rows." `
    -ReadyProperties @{
      inFlightAccountingGateReady = $true
      nativeAttachEntryLocated = $false
    } `
    -MissingProperties @{
      inFlightAccountingGateReady = $false
      nativeAttachEntryLocated = $false
    }
}

function New-DebugListenerNativeNoThrowVTableScaffoldGateEvidence {
  $requiredMarkers = @(
    "debug-listener-native-nothrow-vtable-scaffold-gate",
    "TensorRtDebugListenerNativeNoThrowVTableScaffoldGate",
    "TensorRtDebugListenerNativeNoThrowVTableScaffoldGateResult",
    "Evaluate",
    "RuntimeEvidenceKind=vtable-scaffold-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeAttachBridgeShapeGateReady=True",
    "ExceptionStatusMappingGateReady=True",
    "InFlightAccountingGateReady=True",
    "NoThrowVTableScaffoldReady=True",
    "VTableDestructorNoThrowReady=True",
    "ProcessDebugTensorCallbackStubNoThrowReady=True",
    "ExceptionEscapeBlocked=True",
    "CallbackExceptionCaptureGateReady=True",
    "CallbackStatusMappingGateReady=True",
    "CallbackInFlightAccountingGateReady=True",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "VTableAddressExposed=False",
    "VTablePointerProduced=False",
    "NativeAttachEntryLocated=False",
    "NativeVTableDesignReady=False",
    "CanImplementNativeAttach=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "DebugListenerNativeNoThrowVTableScaffoldGate=",
    "DebugListenerNativeNoThrowVTableScaffoldGate",
    "DebugListenerNativeNoThrowVTableScaffoldGateResult",
    "debug_listener_native_nothrow_vtable_scaffold_gate.inc",
    "DebugListenerNativeNoThrowVTableScaffoldGate(const DebugListenerNativeNoThrowVTableScaffoldGate&) = delete",
    "process_debug_tensor_stub",
    "setDebugListener(non-null)",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-native-nothrow-vtable-scaffold-gate" `
    -EvidenceKind "vtable-scaffold-gate" `
    -RuntimeEvidenceKind "vtable-scaffold-gate" `
    -ReadyStatus "vtable-scaffold-gate-ready" `
    -Source "source-smoke-docs-native-scaffold" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerExceptionStatusMappingGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerInFlightAccountingGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofPrecheck.cs",
      "native\src\tensorrt\common\debug_listener_native_nothrow_vtable_scaffold_gate.inc",
      "native\src\tensorrt\common\debug_listener_native_attach_bridge_shape_gate.inc",
      "native\src\tensorrt\common\debug_listener_exception_status_mapping_gate.inc",
      "native\src\tensorrt\common\debug_listener_inflight_accounting_gate.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-native-nothrow-vtable-scaffold-gate.md",
      "docs\articles\zh-cn\debug-listener-native-attach-bridge-shape-gate.md",
      "docs\articles\zh-cn\debug-listener-exception-status-mapping-gate.md",
      "docs\articles\zh-cn\debug-listener-inflight-accounting-gate.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener native no-throw vtable scaffold gate is source-visible, exception-safe, pointer-free, and still blocks native vtable/runtime proof promotion until non-null attach and real processDebugTensor execution evidence exist." `
    -IncompleteDiagnostic "debug listener native no-throw vtable scaffold gate evidence is incomplete; inspect source/native scaffold/smoke/doc markers and callback deferred rows." `
    -ReadyProperties @{
      vTableScaffoldGateReady = $true
      nativeVTableDesignReady = $false
      nativeAttachEntryLocated = $false
    } `
    -MissingProperties @{
      vTableScaffoldGateReady = $false
      nativeVTableDesignReady = $false
      nativeAttachEntryLocated = $false
    }
}

function New-DebugListenerNoThrowVTableCallbackStubEvidence {
  $requiredMarkers = @(
    "debug-listener-nothrow-vtable-callback-stub",
    "TensorRtDebugListenerNoThrowVTableCallbackStub",
    "TensorRtDebugListenerNoThrowVTableCallbackStubResult",
    "Evaluate",
    "RuntimeEvidenceKind=callback-stub-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "MinimalSafetyReady=True",
    "NoThrowVTableScaffoldGateReady=True",
    "NoThrowVTableScaffoldReady=True",
    "CallbackStubGateReady=True",
    "CallbackStubShapeReady=True",
    "CallbackStubNoThrowReady=True",
    "CallbackMetadataCopyReady=True",
    "CallbackExceptionCaptureReady=True",
    "CallbackStatusMappingReady=True",
    "CallbackInFlightEnterReady=True",
    "CallbackInFlightLeaveReady=True",
    "CallbackInFlightPairingReady=True",
    "CallbackInFlightNeverNegativeReady=True",
    "BorrowedDebugTensorMetadataCopyReady=True",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "DebugTensorPointerExposed=False",
    "DebugTensorDataPointerExposed=False",
    "SetDebugListenerNonNullEnabled=False",
    "NativeAttachWouldBeBlocked=True",
    "NativeVTableInstalled=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanInstallNativeVTable=False",
    "CanCallProcessDebugTensorRuntime=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "ReasonCallbackRuntimeStillBlocked",
    "DebugListenerNoThrowVTableCallbackStub=",
    "DebugListenerNoThrowVTableCallbackStub",
    "DebugListenerNoThrowVTableCallbackStubResult",
    "debug_listener_nothrow_vtable_callback_stub.inc",
    "DebugListenerNoThrowVTableCallbackStub final",
    "begin_callback",
    "complete_callback_success",
    "complete_callback_failure",
    "can_return_status_without_throwing",
    "configure_api_line",
    "line_supports_debug_listener",
    "callback-stub-gate is non-proof evidence",
    "setDebugListener(non-null)",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-nothrow-vtable-callback-stub" `
    -EvidenceKind "callback-stub-gate" `
    -RuntimeEvidenceKind "callback-stub-gate" `
    -ReadyStatus "callback-stub-gate-ready" `
    -Source "source-smoke-docs-native-stub" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNoThrowVTableCallbackStub.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachEntryMinimalSafety.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs",
      "native\src\tensorrt\common\debug_listener_nothrow_vtable_callback_stub.inc",
      "native\src\tensorrt\common\debug_listener_native_attach_entry_minimal_safety.inc",
      "native\src\tensorrt\common\debug_listener_native_nothrow_vtable_scaffold_gate.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-nothrow-vtable-callback-stub.md",
      "docs\articles\zh-cn\debug-listener-native-attach-entry-minimal-safety.md",
      "docs\articles\zh-cn\debug-listener-native-nothrow-vtable-scaffold-gate.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener no-throw vtable callback stub is source-visible, metadata-copy-only, pointer-free, version-guarded for TRT10/TRT11, and still blocks native vtable/runtime proof promotion." `
    -IncompleteDiagnostic "debug listener no-throw vtable callback stub evidence is incomplete; inspect source/native stub/smoke/doc/package markers and callback deferred rows." `
    -ReadyProperties @{
      callbackStubGateReady = $true
      callbackMetadataCopyReady = $true
      nativeVTableInstalled = $false
      setDebugListenerNonNullEnabled = $false
    } `
    -MissingProperties @{
      callbackStubGateReady = $false
      callbackMetadataCopyReady = $false
      nativeVTableInstalled = $false
      setDebugListenerNonNullEnabled = $false
    }
}

function New-DebugListenerBorrowedDebugTensorMetadataRuntimeGateEvidence {
  $requiredMarkers = @(
    "debug-listener-borrowed-debug-tensor-metadata-runtime-gate",
    "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate",
    "TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult",
    "Evaluate",
    "RuntimeEvidenceKind=borrowed-debug-tensor-metadata-gate",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "BorrowedTensorSafetyGateReady=True",
    "CallbackStubGateReady=True",
    "MetadataGateReady=True",
    "TensorNameCopied=True",
    "TensorNameLength",
    "TensorTypeCopied=True",
    "TensorLocationCopied=True",
    "TensorShapeCopied=True",
    "TensorShapeRank",
    "TensorFlagsCopied=True",
    "BorrowedDebugTensorMetadataCopyReady=True",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "BorrowedDebugTensorDataPointerEscapeBlocked=True",
    "DebugTensorPointerExposed=False",
    "DebugTensorDataPointerExposed=False",
    "BorrowedDebugTensorLifetimeReady=False",
    "BorrowedDebugTensorDataLifetimeReady=False",
    "SetDebugListenerNonNullEnabled=False",
    "NativeVTableInstalled=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanCallProcessDebugTensorRuntime=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "ReasonMetadataRuntimeStillBlocked",
    "DebugListenerBorrowedDebugTensorMetadataRuntimeGate=",
    "DebugListenerBorrowedDebugTensorMetadataRuntimeGate",
    "DebugListenerBorrowedDebugTensorMetadataRuntimeGateResult",
    "debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc",
    "DebugListenerBorrowedDebugTensorMetadataRuntimeGate final",
    "copy_metadata",
    "metadata_copy_ready",
    "borrowed_tensor_pointer_escape_blocked",
    "borrowed_tensor_data_pointer_escape_blocked",
    "borrowed_tensor_lifetime_runtime_ready",
    "borrowed_tensor_data_lifetime_runtime_ready",
    "process_debug_tensor_runtime_ready",
    "borrowed-debug-tensor-metadata-gate is non-proof evidence",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-borrowed-debug-tensor-metadata-runtime-gate" `
    -EvidenceKind "borrowed-debug-tensor-metadata-gate" `
    -RuntimeEvidenceKind "borrowed-debug-tensor-metadata-gate" `
    -ReadyStatus "borrowed-debug-tensor-metadata-gate-ready" `
    -Source "source-smoke-docs-native-metadata-gate" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedTensorSafetyGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNoThrowVTableCallbackStub.cs",
      "native\src\tensorrt\common\debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc",
      "native\src\tensorrt\common\debug_listener_nothrow_vtable_callback_stub.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-borrowed-debug-tensor-metadata-runtime-gate.md",
      "docs\articles\zh-cn\debug-listener-borrowed-tensor-safety-gate.md",
      "docs\articles\zh-cn\debug-listener-nothrow-vtable-callback-stub.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener borrowed debug tensor metadata runtime gate is source-visible, copied-metadata-only, pointer-free, version-guarded for TRT10/TRT11, and still blocks borrowed lifetime/runtime proof promotion." `
    -IncompleteDiagnostic "debug listener borrowed debug tensor metadata runtime gate evidence is incomplete; inspect source/native metadata gate/smoke/doc/package markers and callback deferred rows." `
    -ReadyProperties @{
      metadataGateReady = $true
      borrowedDebugTensorMetadataCopyReady = $true
      borrowedDebugTensorPointerEscapeBlocked = $true
      borrowedDebugTensorDataPointerEscapeBlocked = $true
      borrowedDebugTensorLifetimeReady = $false
      borrowedDebugTensorDataLifetimeReady = $false
      processDebugTensorRuntimeReady = $false
    } `
    -MissingProperties @{
      metadataGateReady = $false
      borrowedDebugTensorMetadataCopyReady = $false
      borrowedDebugTensorPointerEscapeBlocked = $false
      borrowedDebugTensorDataPointerEscapeBlocked = $false
      borrowedDebugTensorLifetimeReady = $false
      borrowedDebugTensorDataLifetimeReady = $false
      processDebugTensorRuntimeReady = $false
    }
}

function New-DebugListenerNativeVTableInstallPreflightEvidence {
  $requiredMarkers = @(
    "debug-listener-native-vtable-install-preflight",
    "TensorRtDebugListenerNativeVTableInstallPreflight",
    "TensorRtDebugListenerNativeVTableInstallPreflightResult",
    "Evaluate",
    "RuntimeEvidenceKind=native-vtable-install-preflight",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeOwnerLifecycleGateReady=True",
    "NativeAttachBridgeShapeGateReady=True",
    "NativeNoThrowVTableScaffoldGateReady=True",
    "BorrowedDebugTensorMetadataGateReady=True",
    "VTableInstallShapeReady=True",
    "VTableInstallVersionGuardReady=True",
    "VTableInstallNoThrowBoundaryReady=True",
    "VTableInstallOwnershipDiagnosticsReady=True",
    "VTableInstallPointerFree=True",
    "AttachBridgeSetDebugListenerNonNullEnabled=False",
    "SetDebugListenerNonNullEnabled=False",
    "NonNullAttachStillDisabled=True",
    "VTableAddressExposed=False",
    "VTablePointerProduced=False",
    "DebugTensorPointerExposed=False",
    "DebugTensorDataPointerExposed=False",
    "BorrowedDebugTensorPointerEscapeBlocked=True",
    "BorrowedDebugTensorDataPointerEscapeBlocked=True",
    "BorrowedDebugTensorLifetimeReady=False",
    "BorrowedDebugTensorDataLifetimeReady=False",
    "NativeVTableInstallPreflightReady=True",
    "NativeVTableInstalled=False",
    "NativeVTableInstallRuntimeReady=False",
    "CanEnableSetDebugListenerNonNull=False",
    "CanInstallNativeVTable=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanCallProcessDebugTensorRuntime=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "ReasonNativeVTableInstallStillBlocked",
    "DebugListenerNativeVTableInstallPreflight=",
    "DebugListenerNativeVTableInstallPreflight",
    "DebugListenerNativeVTableInstallPreflightResult",
    "debug_listener_native_vtable_install_preflight.inc",
    "DebugListenerNativeVTableInstallPreflight final",
    "configure_api_line",
    "configure_preflight",
    "preflight_shape_ready",
    "version_guard_ready",
    "no_throw_install_boundary_ready",
    "ownership_diagnostics_ready",
    "pointer_free",
    "can_enable_set_debug_listener_non_null",
    "can_install_native_vtable",
    "native_vtable_installed",
    "native-vtable-install-preflight is non-proof evidence",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-native-vtable-install-preflight" `
    -EvidenceKind "native-vtable-install-preflight" `
    -RuntimeEvidenceKind "native-vtable-install-preflight" `
    -ReadyStatus "native-vtable-install-preflight-ready" `
    -Source "source-smoke-docs-native-vtable-install-preflight" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeVTableInstallPreflight.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
      "native\src\tensorrt\common\debug_listener_native_vtable_install_preflight.inc",
      "native\src\tensorrt\common\debug_listener_native_owner_lifecycle_gate.inc",
      "native\src\tensorrt\common\debug_listener_native_attach_bridge_shape_gate.inc",
      "native\src\tensorrt\common\debug_listener_native_nothrow_vtable_scaffold_gate.inc",
      "native\src\tensorrt\common\debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-native-vtable-install-preflight.md",
      "docs\articles\zh-cn\debug-listener-borrowed-debug-tensor-metadata-runtime-gate.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener native vtable install preflight is source-visible, pointer-free, version-guarded for TRT10/TRT11, no-throw scaffolded, and still blocks non-null attach, native vtable installation, and runtime proof promotion." `
    -IncompleteDiagnostic "debug listener native vtable install preflight evidence is incomplete; inspect source/native scaffold/smoke/doc/package markers and callback deferred rows." `
    -ReadyProperties @{
      nativeVTableInstallPreflightReady = $true
      vTableInstallShapeReady = $true
      vTableInstallPointerFree = $true
      nativeVTableInstalled = $false
      nativeVTableInstallRuntimeReady = $false
      canEnableSetDebugListenerNonNull = $false
      canInstallNativeVTable = $false
      canCallProcessDebugTensorRuntime = $false
      processDebugTensorRuntimeReady = $false
    } `
    -MissingProperties @{
      nativeVTableInstallPreflightReady = $false
      vTableInstallShapeReady = $false
      vTableInstallPointerFree = $false
      nativeVTableInstalled = $false
      nativeVTableInstallRuntimeReady = $false
      canEnableSetDebugListenerNonNull = $false
      canInstallNativeVTable = $false
      canCallProcessDebugTensorRuntime = $false
      processDebugTensorRuntimeReady = $false
    }
}

function New-DebugListenerNativeOwnerVTableInstallExperimentEvidence {
  $requiredMarkers = @(
    "debug-listener-native-owner-vtable-install-experiment",
    "TensorRtDebugListenerNativeOwnerVTableInstallExperiment",
    "TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult",
    "Evaluate",
    "RuntimeEvidenceKind=native-owner-vtable-install-experiment",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NativeOwnerLifecycleGateReady=True",
    "NativeAttachBridgeShapeGateReady=True",
    "NativeNoThrowVTableScaffoldGateReady=True",
    "BorrowedDebugTensorMetadataGateReady=True",
    "NativeVTableInstallPreflightReady=True",
    "ExperimentShapeReady=True",
    "InstallAttemptGuardReady=True",
    "NonNullAttachEnabled=False",
    "RuntimeProofEnabled=False",
    "NativeVTableInstallAttempted=False",
    "NativeVTableInstalled=False",
    "RollbackReady=True",
    "DetachBeforeReleaseReady=True",
    "FailureStatusMappingReady=True",
    "PointerFree=True",
    "VTableAddressExposed=False",
    "VTablePointerProduced=False",
    "DebugTensorPointerExposed=False",
    "DebugTensorDataPointerExposed=False",
    "CanEnableSetDebugListenerNonNull=False",
    "CanInstallNativeVTable=False",
    "ProcessDebugTensorRuntimeReady=False",
    "CanCallProcessDebugTensorRuntime=False",
    "CanAttemptRuntimeProof=False",
    "RuntimeProofBlocked=True",
    "ReasonNativeOwnerVTableInstallStillBlocked",
    "DebugListenerNativeOwnerVTableInstallExperiment=",
    "DebugListenerNativeOwnerVTableInstallExperiment",
    "DebugListenerNativeOwnerVTableInstallExperimentResult",
    "debug_listener_native_owner_vtable_install_experiment.inc",
    "DebugListenerNativeOwnerVTableInstallExperiment final",
    "configure_api_line",
    "configure_prerequisites",
    "configure_disabled_reason",
    "experiment_shape_ready",
    "install_attempt_guard_ready",
    "non_null_attach_enabled",
    "native_vtable_install_attempted",
    "native_vtable_installed",
    "rollback_ready",
    "detach_before_release_ready",
    "failure_status_mapping_ready",
    "pointer_free",
    "process_debug_tensor_runtime_ready",
    "can_attempt_runtime_proof",
    "native-owner-vtable-install-experiment is non-proof evidence",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-native-owner-vtable-install-experiment" `
    -EvidenceKind "native-owner-vtable-install-experiment" `
    -RuntimeEvidenceKind "native-owner-vtable-install-experiment" `
    -ReadyStatus "native-owner-vtable-install-experiment-ready" `
    -Source "source-smoke-docs-native-owner-vtable-install-experiment" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeVTableInstallPreflight.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerLifecycleGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeAttachBridgeShapeGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeNoThrowVTableScaffoldGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
      "native\src\tensorrt\common\debug_listener_native_owner_vtable_install_experiment.inc",
      "native\src\tensorrt\common\debug_listener_native_vtable_install_preflight.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-native-owner-vtable-install-experiment.md",
      "docs\articles\zh-cn\debug-listener-native-vtable-install-preflight.md",
      "docs\articles\zh-cn\debug-listener-runtime-proof-precheck.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener native owner/vtable install experiment is source-visible, disabled-by-default, pointer-free, version-guarded for TRT10/TRT11, rollback/detach/status scaffolded, and still blocks non-null attach, native vtable installation, and runtime proof promotion." `
    -IncompleteDiagnostic "debug listener native owner/vtable install experiment evidence is incomplete; inspect source/native scaffold/smoke/doc/package markers and callback deferred rows." `
    -ReadyProperties @{
      experimentShapeReady = $true
      installAttemptGuardReady = $true
      nonNullAttachEnabled = $false
      runtimeProofEnabled = $false
      nativeVTableInstallAttempted = $false
      nativeVTableInstalled = $false
      rollbackReady = $true
      detachBeforeReleaseReady = $true
      failureStatusMappingReady = $true
      pointerFree = $true
      canEnableSetDebugListenerNonNull = $false
      canInstallNativeVTable = $false
      canCallProcessDebugTensorRuntime = $false
      processDebugTensorRuntimeReady = $false
    } `
    -MissingProperties @{
      experimentShapeReady = $false
      installAttemptGuardReady = $false
      nonNullAttachEnabled = $false
      runtimeProofEnabled = $false
      nativeVTableInstallAttempted = $false
      nativeVTableInstalled = $false
      rollbackReady = $false
      detachBeforeReleaseReady = $false
      failureStatusMappingReady = $false
      pointerFree = $false
      canEnableSetDebugListenerNonNull = $false
      canInstallNativeVTable = $false
      canCallProcessDebugTensorRuntime = $false
      processDebugTensorRuntimeReady = $false
    }
}

function New-DebugListenerRealNonNullAttachRuntimeSmokeEvidence {
  $requiredMarkers = @(
    "debug-listener-real-non-null-attach-runtime-smoke",
    "TensorRtDebugListenerRealNonNullAttachRuntimeSmoke",
    "TensorRtDebugListenerRealNonNullAttachRuntimeSmokeResult",
    "Evaluate",
    "RuntimeEvidenceKind=runtime-smoke-skipped",
    "runtime-smoke-blocked",
    "runtime-smoke-attempted",
    "runtime-smoke-failed",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "OptInEnabled",
    "FullPackageConsumerReport",
    "AttachGuardReady",
    "NativeVTableReady",
    "BorrowedDebugTensorRuntimeReady",
    "CallbackInvocationReady",
    "AttachAttempted",
    "AttachSucceeded",
    "DetachAttempted",
    "DetachSucceeded",
    "RollbackAttempted",
    "RollbackSucceeded",
    "NativeVTableInstalled",
    "ProcessDebugTensorInvoked",
    "InvocationCount",
    "AllocationCount",
    "ReleaseCount",
    "FailureCount",
    "InFlightCallbackCount",
    "LastStatus",
    "LastDiagnostic",
    "ReportPointerFree",
    "CanAttemptRuntimeProof",
    "CanPromoteRealCallbackRuntime",
    "ReasonRuntimeProofStillBlocked",
    "DebugListenerRealNonNullAttachRuntimeSmoke=",
    "DebugListenerRealNonNullAttachRuntimeSmokeAttempt final",
    "debug_listener_real_non_null_attach_runtime_smoke.inc",
    "configure_api_line",
    "configure_opt_in",
    "configure_prerequisites",
    "can_attempt_attach",
    "attach_attempted",
    "attach_succeeded",
    "detach_attempted",
    "detach_succeeded",
    "rollback_attempted",
    "rollback_succeeded",
    "native_vtable_installed",
    "process_debug_tensor_invoked",
    "can_promote_real_callback_runtime",
    "ReportPointerFree=True",
    "CanPromoteRealCallbackRuntime=False",
    "RuntimeProofBlocked=True",
    "DeferredRowsStillRequired",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-real-non-null-attach-runtime-smoke" `
    -EvidenceKind "debug-listener-real-non-null-attach-runtime-smoke" `
    -RuntimeEvidenceKind "runtime-smoke-skipped" `
    -ReadyStatus "runtime-smoke-ready" `
    -Source "source-smoke-docs-real-non-null-attach-runtime-smoke" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofAttemptPreflight.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs",
      "native\src\tensorrt\common\debug_listener_real_non_null_attach_runtime_smoke.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-real-non-null-attach-runtime-smoke.md",
      "docs\articles\zh-cn\debug-listener-native-owner-vtable-install-experiment.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-RuntimePackageReadiness.ps1",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener real non-null attach runtime smoke is source-visible, opt-in, pointer-free, full-package-consumer-aware, and still blocks non-null attach, native vtable installation, processDebugTensor invocation, and real-callback-runtime promotion." `
    -IncompleteDiagnostic "debug listener real non-null attach runtime smoke evidence is incomplete; inspect source/native scaffold/smoke/doc/package/readiness markers and callback deferred rows." `
    -ReadyProperties @{
      runtimeSmokeShapeReady = $true
      defaultSkippedReady = $true
      optInGuardReady = $true
      fullPackageConsumerReportRequired = $true
      attachGuardReady = $false
      nativeVTableReady = $false
      borrowedDebugTensorRuntimeReady = $false
      callbackInvocationReady = $false
      attachAttempted = $false
      attachSucceeded = $false
      detachAttempted = $false
      detachSucceeded = $false
      rollbackReady = $true
      rollbackAttempted = $false
      rollbackSucceeded = $false
      nativeVTableInstalled = $false
      processDebugTensorInvoked = $false
      invocationCount = 0
      allocationCount = 0
      releaseCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      reportPointerFree = $true
      canEnableSetDebugListenerNonNull = $false
      canInstallNativeVTable = $false
      canCallProcessDebugTensorRuntime = $false
      canPromoteRealCallbackRuntime = $false
    } `
    -MissingProperties @{
      runtimeSmokeShapeReady = $false
      defaultSkippedReady = $false
      optInGuardReady = $false
      fullPackageConsumerReportRequired = $false
      attachGuardReady = $false
      nativeVTableReady = $false
      borrowedDebugTensorRuntimeReady = $false
      callbackInvocationReady = $false
      attachAttempted = $false
      attachSucceeded = $false
      detachAttempted = $false
      detachSucceeded = $false
      rollbackReady = $false
      rollbackAttempted = $false
      rollbackSucceeded = $false
      nativeVTableInstalled = $false
      processDebugTensorInvoked = $false
      invocationCount = 0
      allocationCount = 0
      releaseCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      reportPointerFree = $false
      canEnableSetDebugListenerNonNull = $false
      canInstallNativeVTable = $false
      canCallProcessDebugTensorRuntime = $false
      canPromoteRealCallbackRuntime = $false
    }
}

function New-DebugListenerProcessDebugTensorCallbackTrampolineEvidence {
  $requiredMarkers = @(
    "debug-listener-process-debug-tensor-callback-trampoline",
    "callback-trampoline-shape",
    "TensorRtDebugListenerProcessDebugTensorCallbackTrampoline",
    "TensorRtDebugListenerProcessDebugTensorCallbackTrampolineResult",
    "TensorRtDebugTensorMetadataSnapshot",
    "Evaluate",
    "RuntimeEvidenceKind=callback-trampoline-shape",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "CallbackKind",
    "TrampolineShapeReady",
    "NativeCallbackEntryLocated",
    "NoThrowCallbackEntryReady",
    "ExceptionCaptureReady",
    "CallbackStatusMappingReady",
    "InFlightAccountingReady",
    "DetachBeforeReleaseReady",
    "BorrowedDebugTensorMetadataCopyReady",
    "BorrowedDebugTensorPointerExposed",
    "BorrowedDebugTensorDataPointerExposed",
    "PointerFreeSurfaceReady",
    "ProcessDebugTensorRuntimeReady",
    "OptInEnabled",
    "FullPackageConsumerReport",
    "AttachAttempted",
    "AttachSucceeded",
    "NativeVTableInstalled",
    "ProcessDebugTensorInvoked",
    "InvocationCount",
    "CallbackStubEntryCount",
    "CallbackStubLeaveCount",
    "FailureCount",
    "InFlightCallbackCount",
    "LastStatus",
    "LastDiagnostic",
    "CanAttemptRuntimeProof",
    "CanPromoteRealCallbackRuntime",
    "RuntimeProofBlocked",
    "TensorNameLength",
    "TensorShapeRank",
    "MetadataCopied",
    "DebugListenerProcessDebugTensorCallbackTrampoline=",
    "DebugListenerProcessDebugTensorCallbackTrampoline final",
    "DebugListenerProcessDebugTensorCallbackReport",
    "DebugListenerBorrowedDebugTensorMetadataCopy",
    "DebugListenerCallbackInFlightScope",
    "debug_listener_process_debug_tensor_callback_trampoline.inc",
    "configure_api_line",
    "configure_shape",
    "begin_callback",
    "complete_callback_success",
    "complete_callback_failure",
    "can_return_status_without_throwing",
    "trampoline_shape_ready",
    "make_report",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-process-debug-tensor-callback-trampoline" `
    -EvidenceKind "debug-listener-process-debug-tensor-callback-trampoline" `
    -RuntimeEvidenceKind "callback-trampoline-shape" `
    -ReadyStatus "callback-trampoline-shape-ready" `
    -Source "source-smoke-docs-callback-trampoline-shape" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerNoThrowVTableCallbackStub.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs",
      "native\src\tensorrt\common\debug_listener_process_debug_tensor_callback_trampoline.inc",
      "native\src\tensorrt\common\debug_listener_nothrow_vtable_callback_stub.inc",
      "native\src\tensorrt\common\debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-process-debug-tensor-callback-trampoline.md",
      "docs\articles\zh-cn\debug-listener-real-non-null-attach-runtime-smoke.md",
      "docs\articles\zh-cn\debug-listener-borrowed-debug-tensor-metadata-runtime-gate.md",
      "docs\articles\zh-cn\debug-listener-exception-status-mapping-gate.md",
      "docs\articles\zh-cn\debug-listener-inflight-accounting-gate.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-RuntimePackageReadiness.ps1",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener processDebugTensor callback trampoline shape is source-visible, no-throw, pointer-free, metadata-copy aware, package-consumer-visible, and remains non-proof until TensorRT invokes processDebugTensor in a full package consumer runtime." `
    -IncompleteDiagnostic "debug listener processDebugTensor callback trampoline shape evidence is incomplete; inspect source/native scaffold/smoke/doc/package/readiness markers and callback deferred rows." `
    -ReadyProperties @{
      trampolineShapeReady = $true
      nativeCallbackEntryLocated = $true
      noThrowCallbackEntryReady = $true
      exceptionCaptureReady = $true
      callbackStatusMappingReady = $true
      inFlightAccountingReady = $true
      detachBeforeReleaseReady = $true
      borrowedDebugTensorMetadataCopyReady = $true
      borrowedDebugTensorPointerExposed = $false
      borrowedDebugTensorDataPointerExposed = $false
      pointerFreeSurfaceReady = $true
      processDebugTensorRuntimeReady = $false
      optInEnabled = $false
      fullPackageConsumerReport = $false
      attachAttempted = $false
      attachSucceeded = $false
      nativeVTableInstalled = $false
      processDebugTensorInvoked = $false
      invocationCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      canPromoteRealCallbackRuntime = $false
    } `
    -MissingProperties @{
      trampolineShapeReady = $false
      nativeCallbackEntryLocated = $false
      noThrowCallbackEntryReady = $false
      exceptionCaptureReady = $false
      callbackStatusMappingReady = $false
      inFlightAccountingReady = $false
      detachBeforeReleaseReady = $false
      borrowedDebugTensorMetadataCopyReady = $false
      borrowedDebugTensorPointerExposed = $false
      borrowedDebugTensorDataPointerExposed = $false
      pointerFreeSurfaceReady = $false
      processDebugTensorRuntimeReady = $false
      optInEnabled = $false
      fullPackageConsumerReport = $false
      attachAttempted = $false
      attachSucceeded = $false
      nativeVTableInstalled = $false
      processDebugTensorInvoked = $false
      invocationCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      canPromoteRealCallbackRuntime = $false
    }
}

function New-DebugListenerRealCallbackRuntimeProofEvidence {
  $requiredMarkers = @(
    "debug-listener-real-callback-runtime-proof",
    "TensorRtDebugListenerRealCallbackRuntimeProof",
    "TensorRtDebugListenerRealCallbackRuntimeProofResult",
    "Evaluate",
    "RuntimeEvidenceKind=runtime-smoke-skipped",
    "real-callback-runtime-blocked",
    "attempted-no-invocation",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "CallbackKind",
    "OptInEnabled",
    "FullPackageConsumerReport",
    "RuntimeSmokeReady",
    "TrampolineShapeReady",
    "AttachAttempted",
    "AttachSucceeded",
    "DetachAttempted",
    "DetachSucceeded",
    "RollbackAttempted",
    "RollbackSucceeded",
    "NativeVTableInstalled",
    "ProcessDebugTensorInvoked",
    "InvocationCount",
    "FailureCount",
    "InFlightCallbackCount",
    "BorrowedDebugTensorMetadataCopied",
    "PointerFreeSurfaceReady",
    "ProcessDebugTensorRuntimeReady",
    "AttemptedNoInvocation",
    "LastStatus",
    "LastDiagnostic",
    "CanAttemptRuntimeProof",
    "CanPromoteRealCallbackRuntime",
    "RuntimeProofBlocked",
    "DeferredRowsStillRequired",
    "DebugListenerRealCallbackRuntimeProof=",
    "DebugListenerRealCallbackRuntimeProofGate final",
    "DebugListenerRealCallbackRuntimeProofReport",
    "debug_listener_real_callback_runtime_proof.inc",
    "configure_api_line",
    "configure_prerequisites",
    "configure_attempt",
    "configure_invocation",
    "can_attempt_runtime_proof",
    "can_promote_real_callback_runtime",
    "InvocationCount>0",
    "not proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-real-callback-runtime-proof" `
    -EvidenceKind "debug-listener-real-callback-runtime-proof" `
    -RuntimeEvidenceKind "runtime-smoke-skipped" `
    -ReadyStatus "real-callback-runtime-proof-gate-ready" `
    -Source "source-smoke-docs-real-callback-runtime-proof-gate" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRealCallbackRuntimeProof.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs",
      "native\src\tensorrt\common\debug_listener_real_callback_runtime_proof.inc",
      "native\src\tensorrt\common\debug_listener_real_non_null_attach_runtime_smoke.inc",
      "native\src\tensorrt\common\debug_listener_process_debug_tensor_callback_trampoline.inc",
      "native\src\tensorrt\v8\api.cpp",
      "native\src\tensorrt\v10\api.cpp",
      "native\src\tensorrt\v11\api.cpp",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "docs\articles\zh-cn\debug-listener-real-callback-runtime-proof.md",
      "docs\articles\zh-cn\debug-listener-process-debug-tensor-callback-trampoline.md",
      "docs\articles\zh-cn\debug-listener-real-non-null-attach-runtime-smoke.md",
      "docs\articles\zh-cn\real-callback-trampoline-gate.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\windows-api-completion-latest.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "smoke\README.md",
      "eng\Test-RuntimePackageReadiness.ps1",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener real callback runtime proof gate is source-visible, pointer-free, package-consumer-visible, invocation-count guarded, and remains non-proof until a full package consumer reports InvocationCount>0 with real-callback-runtime markers." `
    -IncompleteDiagnostic "debug listener real callback runtime proof gate evidence is incomplete; inspect source/native scaffold/smoke/doc/package/readiness markers and callback deferred rows." `
    -ReadyProperties @{
      proofGateReady = $true
      optInEnabled = $false
      fullPackageConsumerReport = $false
      runtimeSmokeReady = $false
      trampolineShapeReady = $true
      attachAttempted = $false
      attachSucceeded = $false
      detachAttempted = $false
      detachSucceeded = $true
      rollbackAttempted = $false
      rollbackSucceeded = $true
      nativeVTableInstalled = $false
      processDebugTensorInvoked = $false
      invocationCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      borrowedDebugTensorMetadataCopied = $true
      pointerFreeSurfaceReady = $true
      processDebugTensorRuntimeReady = $false
      attemptedNoInvocation = $false
      canPromoteRealCallbackRuntime = $false
    } `
    -MissingProperties @{
      proofGateReady = $false
      optInEnabled = $false
      fullPackageConsumerReport = $false
      runtimeSmokeReady = $false
      trampolineShapeReady = $false
      attachAttempted = $false
      attachSucceeded = $false
      detachAttempted = $false
      detachSucceeded = $false
      rollbackAttempted = $false
      rollbackSucceeded = $false
      nativeVTableInstalled = $false
      processDebugTensorInvoked = $false
      invocationCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      borrowedDebugTensorMetadataCopied = $false
      pointerFreeSurfaceReady = $false
      processDebugTensorRuntimeReady = $false
      attemptedNoInvocation = $false
      canPromoteRealCallbackRuntime = $false
    }
}

function New-DebugListenerCallbackProofGapReportEvidence {
  $requiredMarkers = @(
    "debug-listener-callback-proof-gap-report",
    "TensorRtDebugListenerCallbackProofGapReport",
    "TensorRtDebugListenerCallbackProofGapReportResult",
    "Evaluate",
    "RuntimeEvidenceKind=proof-gap-report",
    "RealCallbackRuntime=False",
    "IsRealCallbackRuntimeProof=False",
    "NonNullAttachStillDisabled",
    "NativeAttachEntryReady",
    "NativeVTableInstallBlocked",
    "NoThrowCallbackEntryReady",
    "ExceptionStatusMappingReady",
    "InFlightAccountingReady",
    "BorrowedDebugTensorMetadataCopied",
    "DetachRollbackReady",
    "ProcessDebugTensorRuntimeInvoked",
    "FullPackageConsumerRuntimeProofReady",
    "PointerFreeSurfaceReady",
    "AttemptedNoInvocation",
    "InvocationCount",
    "FailureCount",
    "InFlightCallbackCount",
    "CanPromoteRealCallbackRuntime",
    "RuntimeProofBlocked",
    "DeferredRowsStillRequired",
    "GapReasonCount",
    "PrimaryGapReason",
    "RuntimeProofBlockerCategory",
    "PackageConsumerRuntimeProofRequired",
    "RuntimeInvocationRequired",
    "EvidenceSource",
    "NextOwnerAction",
    "DebugListenerCallbackProofGapReport=",
    "NoNonProofCallbackRuntimeMarker",
    "not proof",
    "不能替代真实 TensorRT callback runtime proof"
  )

  return New-SourceVisibleDebugListenerGateEvidence `
    -Marker "debug-listener-callback-proof-gap-report" `
    -EvidenceKind "debug-listener-callback-proof-gap-report" `
    -RuntimeEvidenceKind "proof-gap-report" `
    -ReadyStatus "proof-gap-report-ready" `
    -Source "source-smoke-docs-package-readiness-proof-gap-report" `
    -EvidenceRelativePaths @(
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackProofGapReport.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRuntimeProofAttemptPreflight.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRealNonNullAttachRuntimeSmoke.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerProcessDebugTensorCallbackTrampoline.cs",
      "src\JYPPX.TensorRtSharp\TensorRtDebugListenerRealCallbackRuntimeProof.cs",
      "smoke\CallbackAllocatorSafeControlsSmokeRunner\Program.cs",
      "smoke\README.md",
      "docs\articles\zh-cn\debug-listener-callback-proof-gap-report.md",
      "docs\articles\zh-cn\real-callback-runtime-evidence-schema.md",
      "docs\articles\zh-cn\callback-owner-closure-matrix.md",
      "docs\index.md",
      "docs\toc.yml",
      "pack\runtime-split\README.md",
      "eng\Test-RuntimePackageReadiness.ps1",
      "eng\Test-BridgePackageConsumer.ps1",
      "eng\Test-PackageConsumer.ps1") `
    -RequiredMarkers $requiredMarkers `
    -ReadyDiagnostic "debug listener callback proof gap report is source-visible, package-consumer-visible, pointer-free, and classified as proof-gap-report/non-proof until full package consumer real callback invocation evidence exists." `
    -IncompleteDiagnostic "debug listener callback proof gap report evidence is incomplete; inspect source/smoke/docs/package/readiness markers and callback deferred rows." `
    -ReadyProperties @{
      proofGapReportReady = $true
      nonNullAttachStillDisabled = $true
      nativeAttachEntryReady = $false
      nativeVTableInstallBlocked = $true
      noThrowCallbackEntryReady = $true
      exceptionStatusMappingReady = $true
      inFlightAccountingReady = $true
      borrowedDebugTensorMetadataCopied = $true
      detachRollbackReady = $true
      processDebugTensorRuntimeInvoked = $false
      fullPackageConsumerRuntimeProofReady = $false
      pointerFreeSurfaceReady = $true
      attemptedNoInvocation = $false
      invocationCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      canPromoteRealCallbackRuntime = $false
      gapReasonCount = 1
    } `
    -MissingProperties @{
      proofGapReportReady = $false
      nonNullAttachStillDisabled = $true
      nativeAttachEntryReady = $false
      nativeVTableInstallBlocked = $true
      noThrowCallbackEntryReady = $false
      exceptionStatusMappingReady = $false
      inFlightAccountingReady = $false
      borrowedDebugTensorMetadataCopied = $false
      detachRollbackReady = $false
      processDebugTensorRuntimeInvoked = $false
      fullPackageConsumerRuntimeProofReady = $false
      pointerFreeSurfaceReady = $false
      attemptedNoInvocation = $false
      invocationCount = 0
      failureCount = 0
      inFlightCallbackCount = 0
      canPromoteRealCallbackRuntime = $false
      gapReasonCount = 0
    }
}

function New-AllocatorOwnerLedgerDesignGateEvidence {
  $designRelativePath = "docs\articles\zh-cn\allocator-owner-ledger-design.md"
  $callbackDesignRelativePath = "docs\articles\zh-cn\allocator-callback-owner-design.md"
  $tocRelativePath = "docs\toc.yml"
  $indexRelativePath = "docs\index.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $designPath = Join-Path $RepositoryRoot $designRelativePath
  $callbackDesignPath = Join-Path $RepositoryRoot $callbackDesignRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "allocator-owner-ledger-design-gate",
    "Native Owner 形状",
    "Device Pointer Ledger",
    "失败与异常映射",
    "跨版本 Route",
    "Public C# 边界",
    "IGpuAllocator::allocate",
    "IGpuAllocator::free",
    "IGpuAsyncAllocator::allocateAsync",
    "IOutputAllocator::reallocateOutput",
    "IDebugListener::processDebugTensor",
    "no-throw",
    "TRT8",
    "TRT10",
    "TRT11",
    "不能作为真实 TensorRT allocator callback 已启用的证据"
  )

  $requiredDeferredRows = @(
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAllocator","reallocate","IGpuAllocator::reallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"'
  )

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $evidencePaths = @($designPath, $callbackDesignPath, $tocPath, $indexPath, $latestPath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "allocator-owner-ledger-design-gate"
      designPath = $designPath
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDesignDocument = $false
      hasDocfxLinks = $false
      hasDeferredRowEvidence = $false
      diagnostic = "allocator owner ledger design gate evidence files are missing."
    }
  }

  $design = Get-Content -LiteralPath $designPath -Raw -Encoding utf8
  $callbackDesign = Get-Content -LiteralPath $callbackDesignPath -Raw -Encoding utf8
  $toc = Get-Content -LiteralPath $tocPath -Raw -Encoding utf8
  $index = Get-Content -LiteralPath $indexPath -Raw -Encoding utf8
  $latest = Get-Content -LiteralPath $latestPath -Raw -Encoding utf8
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  $combinedDocs = $design + "`n" + $callbackDesign + "`n" + $toc + "`n" + $index + "`n" + $latest

  foreach ($marker in @($requiredMarkers)) {
    if ($combinedDocs.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDocfxLinks = $toc.Contains("allocator-owner-ledger-design.md") -and
    $index.Contains("allocator-owner-ledger-design.md") -and
    $callbackDesign.Contains("allocator-owner-ledger-design.md") -and
    $latest.Contains("allocator-owner-ledger-design.md")
  $hasDesignDocument = $missingMarkers.Count -eq 0
  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($hasDesignDocument -and $hasDocfxLinks -and $hasDeferredRowEvidence) { "ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "ready") {
    "allocator owner ledger design gate is documented, linked, and still preserves deferred callback rows."
  }
  else {
    "allocator owner ledger design gate is incomplete; inspect missing markers or deferred row evidence."
  }

  return [pscustomobject]@{
    status = $status
    marker = "allocator-owner-ledger-design-gate"
    designPath = $designPath
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDesignDocument = $hasDesignDocument
    hasDocfxLinks = $hasDocfxLinks
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    diagnostic = $diagnostic
  }
}

function New-RealCallbackTrampolineGateEvidence {
  $designRelativePath = "docs\articles\zh-cn\real-callback-trampoline-gate.md"
  $callbackDesignRelativePath = "docs\articles\zh-cn\allocator-callback-owner-design.md"
  $ledgerDesignRelativePath = "docs\articles\zh-cn\allocator-owner-ledger-design.md"
  $tocRelativePath = "docs\toc.yml"
  $indexRelativePath = "docs\index.md"
  $latestRelativePath = "docs\articles\zh-cn\windows-api-completion-latest.md"
  $runtimeSplitReadmeRelativePath = "pack\runtime-split\README.md"
  $comparisonRelativePath = "artifacts\interface-coverage\tensorrt-interface-comparison.csv"
  $designPath = Join-Path $RepositoryRoot $designRelativePath
  $callbackDesignPath = Join-Path $RepositoryRoot $callbackDesignRelativePath
  $ledgerDesignPath = Join-Path $RepositoryRoot $ledgerDesignRelativePath
  $tocPath = Join-Path $RepositoryRoot $tocRelativePath
  $indexPath = Join-Path $RepositoryRoot $indexRelativePath
  $latestPath = Join-Path $RepositoryRoot $latestRelativePath
  $runtimeSplitReadmePath = Join-Path $RepositoryRoot $runtimeSplitReadmeRelativePath
  $comparisonPath = Join-Path $RepositoryRoot $comparisonRelativePath

  $requiredMarkers = @(
    "真实 Callback Trampoline 门禁复审",
    "real-callback-trampoline-gate",
    "go/no-go checklist",
    "native owner 生命周期",
    "dispose 顺序",
    "GCHandle",
    "delegate pinning",
    "C ABI no-throw",
    "Windows SEH",
    "exception-to-status",
    "device pointer ledger",
    "stream/async",
    "real-callback-runtime",
    "dry-run",
    "copied-state",
    "TRT8",
    "TRT10",
    "TRT11",
    "不能作为真实 TensorRT callback 已启用的证据"
  )

  $requiredDeferredRows = @(
    '"IGpuAllocator","allocate","IGpuAllocator::allocate","other","deferred-only"',
    '"IGpuAllocator","free","IGpuAllocator::free","other","deferred-only"',
    '"IGpuAllocator","deallocate","IGpuAllocator::deallocate","other","deferred-only"',
    '"IGpuAsyncAllocator","allocateAsync","IGpuAsyncAllocator::allocateAsync","other","deferred-only"',
    '"IGpuAsyncAllocator","deallocateAsync","IGpuAsyncAllocator::deallocateAsync","other","deferred-only"',
    '"IOutputAllocator","notifyShape","IOutputAllocator::notifyShape","other","deferred-only"',
    '"IOutputAllocator","reallocateOutput","IOutputAllocator::reallocateOutput","other","deferred-only"',
    '"IDebugListener","processDebugTensor","IDebugListener::processDebugTensor","other","deferred-only"'
  )

  $missingMarkers = New-Object System.Collections.Generic.List[string]
  $missingDeferredRows = New-Object System.Collections.Generic.List[string]
  $evidencePaths = @($designPath, $callbackDesignPath, $ledgerDesignPath, $tocPath, $indexPath, $latestPath, $runtimeSplitReadmePath, $comparisonPath)
  $missingFiles = @($evidencePaths | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
  if ($missingFiles.Count -gt 0) {
    return [pscustomobject]@{
      status = "missing"
      marker = "real-callback-trampoline-gate"
      evidenceKind = "design-gate-only"
      realCallbackRuntimeEvidenceKind = "not-present"
      designPath = $designPath
      evidencePaths = @($evidencePaths)
      missingFiles = @($missingFiles)
      missingMarkers = @($requiredMarkers)
      missingDeferredRows = @($requiredDeferredRows)
      hasDesignDocument = $false
      hasDocfxLinks = $false
      hasRuntimeSplitReadmeNote = $false
      hasDeferredRowEvidence = $false
      isRealCallbackRuntimeProof = $false
      diagnostic = "real callback trampoline gate evidence files are missing."
    }
  }

  $design = Get-Content -LiteralPath $designPath -Raw -Encoding utf8
  $callbackDesign = Get-Content -LiteralPath $callbackDesignPath -Raw -Encoding utf8
  $ledgerDesign = Get-Content -LiteralPath $ledgerDesignPath -Raw -Encoding utf8
  $toc = Get-Content -LiteralPath $tocPath -Raw -Encoding utf8
  $index = Get-Content -LiteralPath $indexPath -Raw -Encoding utf8
  $latest = Get-Content -LiteralPath $latestPath -Raw -Encoding utf8
  $runtimeSplitReadme = Get-Content -LiteralPath $runtimeSplitReadmePath -Raw -Encoding utf8
  $comparison = Get-Content -LiteralPath $comparisonPath -Raw -Encoding utf8
  $combinedDocs = $design + "`n" + $callbackDesign + "`n" + $ledgerDesign + "`n" + $toc + "`n" + $index + "`n" + $latest + "`n" + $runtimeSplitReadme

  foreach ($marker in @($requiredMarkers)) {
    if ($combinedDocs.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingMarkers.Add($marker)
    }
  }

  foreach ($row in @($requiredDeferredRows)) {
    if ($comparison.IndexOf($row, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
      $missingDeferredRows.Add($row)
    }
  }

  $hasDocfxLinks = $toc.Contains("real-callback-trampoline-gate.md") -and
    $index.Contains("real-callback-trampoline-gate.md") -and
    $latest.Contains("real-callback-trampoline-gate.md") -and
    $callbackDesign.Contains("real-callback-trampoline-gate.md") -and
    $ledgerDesign.Contains("real-callback-trampoline-gate.md")
  $hasRuntimeSplitReadmeNote = $runtimeSplitReadme.Contains("realCallbackTrampolineGate") -and
    $runtimeSplitReadme.Contains("real-callback-trampoline-gate") -and
    $runtimeSplitReadme.Contains("design gate only") -and
    $runtimeSplitReadme.Contains("not proof that TensorRT callbacks are enabled")
  $hasDesignDocument = $missingMarkers.Count -eq 0
  $hasDeferredRowEvidence = $missingDeferredRows.Count -eq 0
  $status = if ($hasDesignDocument -and $hasDocfxLinks -and $hasRuntimeSplitReadmeNote -and $hasDeferredRowEvidence) { "ready" } else { "incomplete" }
  $diagnostic = if ($status -eq "ready") {
    "real callback trampoline gate is documented, linked, package-noted, and still separates dry-run/copied-state evidence from real-callback-runtime."
  }
  else {
    "real callback trampoline gate is incomplete; inspect missing markers, README notes, DocFX links, or deferred row evidence."
  }

  return [pscustomobject]@{
    status = $status
    marker = "real-callback-trampoline-gate"
    evidenceKind = "design-gate-only"
    realCallbackRuntimeEvidenceKind = "not-present"
    designPath = $designPath
    evidencePaths = @($evidencePaths)
    missingFiles = @()
    missingMarkers = @($missingMarkers.ToArray())
    missingDeferredRows = @($missingDeferredRows.ToArray())
    hasDesignDocument = $hasDesignDocument
    hasDocfxLinks = $hasDocfxLinks
    hasRuntimeSplitReadmeNote = $hasRuntimeSplitReadmeNote
    hasDeferredRowEvidence = $hasDeferredRowEvidence
    isRealCallbackRuntimeProof = $false
    diagnostic = $diagnostic
  }
}

function New-PackageEvidence {
  param(
    [string]$Directory,
    [string]$PackageId
  )

  $package = Find-LocalPackage -Directory $Directory -PackageId $PackageId
  if ($null -eq $package) {
    return [pscustomobject]@{
      status = "missing"
      packageId = $PackageId
      version = ""
      path = ""
      directory = $Directory
    }
  }

  return [pscustomobject]@{
    status = "ready"
    packageId = $PackageId
    version = [string]$package.Version
    path = [string]$package.Path
    sizeBytes = [Int64]$package.SizeBytes
    directory = $Directory
  }
}

function New-ReadinessBlocker {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Category,
    [Parameter(Mandatory = $true)]
    [string]$Status,
    [Parameter(Mandatory = $true)]
    [string]$Detail,
    [string]$EvidencePath = "",
    [string]$NextAction = "",
    [string]$SuggestedCommand = "",
    [bool]$IsExternalInputRequired = $false
  )

  return [pscustomobject]@{
    category = $Category
    status = $Status
    detail = $Detail
    nextAction = $NextAction
    suggestedCommand = $SuggestedCommand
    isExternalInputRequired = $IsExternalInputRequired
    evidencePath = $EvidencePath
  }
}

function New-MissingAssetSummaryByKind {
  param(
    [object[]]$MissingAssets,
    [string]$TensorRtRoot,
    [string]$CudaRoot,
    [string]$CudnnRoot
  )

  $rows = New-Object System.Collections.Generic.List[object]
  foreach ($group in @($MissingAssets | Group-Object kind | Sort-Object Name)) {
    $items = @($group.Group)
    $missingRootCount = @($items | Where-Object { [string]$_.status -eq "missing-root" }).Count
    $missingAssetCount = @($items | Where-Object { [string]$_.status -eq "missing-asset" }).Count
    $root = switch ([string]$group.Name) {
      "TensorRT" { $TensorRtRoot; break }
      "CUDA" { $CudaRoot; break }
      "cuDNN" { $CudnnRoot; break }
      default { "" }
    }

    $rows.Add([pscustomobject]@{
        kind = [string]$group.Name
        missingCount = $items.Count
        missingRootCount = $missingRootCount
        missingAssetCount = $missingAssetCount
        firstMissingRelativePath = [string]($items | Select-Object -First 1).relativePath
        root = [string]$root
      })
  }

  return @($rows.ToArray())
}

function Get-FileCountByExtension {
  param(
    [string]$Root,
    [Parameter(Mandatory = $true)]
    [string]$Extension
  )

  if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root -PathType Container)) {
    return 0
  }

  return @(Get-ChildItem -Path (Join-Path $Root "*$Extension") -Recurse -File -ErrorAction SilentlyContinue).Count
}

function New-VendorRootDiagnostics {
  param(
    [string]$TensorRtRoot,
    [string]$CudaRoot,
    [string]$CudnnRoot,
    [object[]]$MissingAssets
  )

  $definitions = @(
    [pscustomobject]@{ kind = "TensorRT"; root = [string]$TensorRtRoot },
    [pscustomobject]@{ kind = "CUDA"; root = [string]$CudaRoot },
    [pscustomobject]@{ kind = "cuDNN"; root = [string]$CudnnRoot }
  )

  $rows = New-Object System.Collections.Generic.List[object]
  foreach ($definition in @($definitions)) {
    $kind = [string]$definition.kind
    $root = [string]$definition.root
    $exists = -not [string]::IsNullOrWhiteSpace($root) -and (Test-Path -LiteralPath $root -PathType Container)
    $missingForKind = @($MissingAssets | Where-Object { [string]$_.kind -eq $kind })
    $dllCount = Get-FileCountByExtension -Root $root -Extension ".dll"
    $libCount = Get-FileCountByExtension -Root $root -Extension ".lib"
    $hasImportLibrariesOnly = $exists -and $missingForKind.Count -gt 0 -and $dllCount -eq 0 -and $libCount -gt 0
    $diagnostic = if (-not $exists) {
      "vendor root is missing."
    }
    elseif ($hasImportLibrariesOnly) {
      "root contains import/static libraries but no runtime DLLs for the expected assets; this looks like a development-library-only vendor layout."
    }
    elseif ($missingForKind.Count -gt 0) {
      "root exists but one or more expected runtime assets are missing."
    }
    else {
      "root contains all expected runtime assets for this package key."
    }

    $rows.Add([pscustomobject]@{
        kind = $kind
        root = $root
        rootExists = $exists
        dllCount = $dllCount
        libCount = $libCount
        expectedMissingCount = $missingForKind.Count
        firstExpectedMissingRelativePath = [string]($missingForKind | Select-Object -First 1).relativePath
        hasImportLibrariesOnly = $hasImportLibrariesOnly
        diagnostic = $diagnostic
      })
  }

  return @($rows.ToArray())
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

function Add-MissingEvidenceTable {
  param(
    [Parameter(Mandatory = $true)]
    [System.Collections.Generic.List[string]]$Lines,
    [Parameter(Mandatory = $true)]
    [string]$Title,
    [Parameter(Mandatory = $true)]
    [object]$Evidence
  )

  if (@($Evidence.missingMarkers).Count -eq 0 -and @($Evidence.missingDeferredRows).Count -eq 0) {
    return
  }

  $Lines.Add("")
  $Lines.Add("| $Title | Value |")
  $Lines.Add("| --- | --- |")
  foreach ($marker in @($Evidence.missingMarkers)) {
    $Lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
  }

  foreach ($row in @($Evidence.missingDeferredRows)) {
    $Lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
  }
}

function Write-ReadinessReports {
  param(
    [Parameter(Mandatory = $true)]
    [object[]]$Results
  )

  New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
  $jsonPath = Join-Path $ReportDirectory "runtime-package-readiness-summary.json"
  $markdownPath = Join-Path $ReportDirectory "runtime-package-readiness-summary.md"
  $Results | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Runtime Package Readiness Summary")
  $lines.Add("")
  $lines.Add("| Runtime key | Managed | Bridge package | Bridge consumer | Split components | Split collection | Split collection consumer | Full vendor inputs | Full runtime package | Full consumer | Overall | Runtime proof |")
  $lines.Add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  foreach ($result in $Results) {
    $splitComponentSummary = "$($result.splitPackages.status) $($result.splitPackages.foundCount)/$($result.splitPackages.expectedCount)"
    $lines.Add("| $($result.key) | $($result.managedPackage.status) | $($result.bridgePackage.status) | $($result.bridgeConsumer.status) | $splitComponentSummary | $($result.splitCollectionPackage.status) | $($result.splitCollectionConsumer.status) | $($result.fullVendorInputs.status) | $($result.fullRuntimePackage.status) | $($result.fullPackageConsumer.status) | $($result.overallStatus) | $($result.runtimeProofStatus) |")
  }

  $lines.Add("")
  foreach ($result in $Results) {
    $lines.Add("## $($result.key)")
    $lines.Add("")
    $lines.Add("- package: ``$($result.packageId)``")
    $lines.Add("- managed package: $($result.managedPackage.status) ``$($result.managedPackage.version)``")
    $lines.Add("- bridge package: $($result.bridgePackage.status) ``$($result.bridgePackage.version)``")
    $lines.Add("- bridge consumer: $($result.bridgeConsumer.status) $($result.bridgeConsumer.probeResult)")
    $lines.Add("- bridge consumer native dependency: $($result.bridgeConsumer.nativeDependencyStatus); missing=$($result.bridgeConsumer.isNativeDependencyMissing); vendor-structured-exception=$($result.bridgeConsumer.isVendorStructuredException); code=``$($result.bridgeConsumer.vendorExceptionCode)``")
    $lines.Add("- bridge consumer probe diagnostic: $($result.bridgeConsumer.probeDiagnostic)")
    $lines.Add("- bridge consumer native dependency diagnostic: $($result.bridgeConsumer.nativeDependencyDiagnostic)")
    $lines.Add("- bridge consumer evidence scope: $($result.bridgeConsumer.evidenceKind); bridge-only=$($result.bridgeConsumer.isBridgeOnlyEvidence); full-runtime=$($result.bridgeConsumer.isFullRuntimeEvidence); runtime-execution=$($result.bridgeConsumer.isRuntimeExecutionEvidence)")
    $lines.Add("- bridge consumer wrapper surface: ``$($result.bridgeConsumer.highLevelWrapperSurface)``")
    $readyWrapperGroups = @($result.bridgeConsumer.wrapperSurfaceCapabilities.groups | Where-Object { [string]$_.status -eq "ready" } | ForEach-Object { [string]$_.name }) -join ", "
    $missingWrapperGroups = @($result.bridgeConsumer.wrapperSurfaceCapabilities.missingGroups) -join ", "
    $lines.Add("- bridge consumer wrapper capability status: $($result.bridgeConsumer.wrapperSurfaceCapabilities.status); ready=``$readyWrapperGroups``; missing=``$missingWrapperGroups``")
    $lines.Add("- bridge consumer plugin inventory field metadata: $($result.bridgeConsumer.wrapperSurfaceCapabilities.hasPluginInventoryFieldMetadata); marker=``plugin-inventory-field-metadata``; evidence-kind=compile-surface-proof; runtime-evidence=copied-plugin-field-metadata; proof=false")
    $lines.Add("- bridge consumer onnx parser diagnostics: $($result.bridgeConsumer.wrapperSurfaceCapabilities.hasOnnxParserDiagnosticReadiness); marker=``onnx-parser-diagnostic-readiness``; evidence-kind=compile-surface-proof; runtime-evidence=copied-parser-diagnostics; proof=false")
    $lines.Add("- bridge consumer onnx parser-refitter diagnostics: $($result.bridgeConsumer.wrapperSurfaceCapabilities.hasOnnxParserRefitterDiagnosticReadiness); marker=``onnx-parser-refitter-diagnostic-readiness``; evidence-kind=compile-surface-proof; runtime-evidence=copied-parser-refitter-diagnostics; proof=false")
    $lines.Add("- bridge consumer callback api-language safe controls: $($result.bridgeConsumer.wrapperSurfaceCapabilities.hasCallbackApiLanguageSafeControls); marker=``callback-api-language-safe-controls``; evidence-kind=compile-surface-proof; runtime-evidence=scalar-copy-api-language; proof=false")
    $lines.Add("- bridge consumer callback allocator safe-control summary: $($result.bridgeConsumer.wrapperSurfaceCapabilities.hasExecutionContextCallbackAllocatorSafeControlSummary); marker=``execution-context-callback-allocator-safe-control-summary``; evidence-kind=compile-surface-proof; runtime-evidence=copied-interface-info-safe-controls; proof=false")
    $lines.Add("- error recorder diagnostics design gate: $($result.errorRecorderDiagnosticsDesignGate.status); marker=``$($result.errorRecorderDiagnosticsDesignGate.marker)``; evidence-kind=$($result.errorRecorderDiagnosticsDesignGate.evidenceKind); runtime-evidence=$($result.errorRecorderDiagnosticsDesignGate.runtimeEvidenceKind); runtime-execution=$($result.errorRecorderDiagnosticsDesignGate.isRuntimeExecutionEvidence); proof=$($result.errorRecorderDiagnosticsDesignGate.isRuntimeExecutionProof); runtime-blocked=$($result.errorRecorderDiagnosticsDesignGate.runtimeProofBlocked); deferred-rows=$($result.errorRecorderDiagnosticsDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- error recorder diagnostics design gate diagnostic: $($result.errorRecorderDiagnosticsDesignGate.diagnostic)")
    $lines.Add("- dimension expression snapshot design gate: $($result.dimensionExpressionSnapshotDesignGate.status); marker=``$($result.dimensionExpressionSnapshotDesignGate.marker)``; evidence-kind=$($result.dimensionExpressionSnapshotDesignGate.evidenceKind); runtime-evidence=$($result.dimensionExpressionSnapshotDesignGate.runtimeEvidenceKind); owner-lifetime=$($result.dimensionExpressionSnapshotDesignGate.ownerLifetimeKnown); expression-pointer=$($result.dimensionExpressionSnapshotDesignGate.expressionPointerExposed); expr-builder-create=$($result.dimensionExpressionSnapshotDesignGate.exprBuilderCreationEnabled); proof=$($result.dimensionExpressionSnapshotDesignGate.isRuntimeExecutionProof); runtime-blocked=$($result.dimensionExpressionSnapshotDesignGate.runtimeProofBlocked); deferred-rows=$($result.dimensionExpressionSnapshotDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- dimension expression snapshot design gate diagnostic: $($result.dimensionExpressionSnapshotDesignGate.diagnostic)")
    $lines.Add("- calibrator metadata design gate: $($result.calibratorMetadataDesignGate.status); marker=``$($result.calibratorMetadataDesignGate.marker)``; evidence-kind=$($result.calibratorMetadataDesignGate.evidenceKind); runtime-evidence=$($result.calibratorMetadataDesignGate.runtimeEvidenceKind); presence=$($result.calibratorMetadataDesignGate.presenceProbeAvailable); pointer-free=$($result.calibratorMetadataDesignGate.pointerFreeSurfaceReady); callback-invocation=$($result.calibratorMetadataDesignGate.callbackInvocationEnabled); proof=$($result.calibratorMetadataDesignGate.isRuntimeExecutionProof); runtime-blocked=$($result.calibratorMetadataDesignGate.runtimeProofBlocked); deferred-rows=$($result.calibratorMetadataDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- calibrator metadata design gate diagnostic: $($result.calibratorMetadataDesignGate.diagnostic)")
    $lines.Add("- runtime deserialization boundary precheck: $($result.runtimeDeserializationBoundaryPrecheck.status); marker=``$($result.runtimeDeserializationBoundaryPrecheck.marker)``; evidence-kind=$($result.runtimeDeserializationBoundaryPrecheck.evidenceKind); runtime-evidence=$($result.runtimeDeserializationBoundaryPrecheck.runtimeEvidenceKind); byte-array=$($result.runtimeDeserializationBoundaryPrecheck.managedByteArrayDeserializeReady); stream=$($result.runtimeDeserializationBoundaryPrecheck.managedStreamDeserializeReady); host-memory=$($result.runtimeDeserializationBoundaryPrecheck.hostMemoryDeserializeReady); engine-owned=$($result.runtimeDeserializationBoundaryPrecheck.engineHandleOwnedByWrapper); engine-pointer=$($result.runtimeDeserializationBoundaryPrecheck.enginePointerExposed); load-runtime-deferred=$($result.runtimeDeserializationBoundaryPrecheck.loadRuntimeDeferred); proof=$($result.runtimeDeserializationBoundaryPrecheck.isRuntimeExecutionProof); runtime-blocked=$($result.runtimeDeserializationBoundaryPrecheck.runtimeProofBlocked); deferred-rows=$($result.runtimeDeserializationBoundaryPrecheck.hasDeferredRowEvidence)")
    $lines.Add("- runtime deserialization boundary precheck diagnostic: $($result.runtimeDeserializationBoundaryPrecheck.diagnostic)")
    $lines.Add("- runtime deserialization dependency diagnostics: $($result.runtimeDeserializationDependencyDiagnostics.status); marker=``$($result.runtimeDeserializationDependencyDiagnostics.marker)``; evidence-kind=$($result.runtimeDeserializationDependencyDiagnostics.evidenceKind); runtime-evidence=$($result.runtimeDeserializationDependencyDiagnostics.runtimeEvidenceKind); precheck=$($result.runtimeDeserializationDependencyDiagnostics.precheckReady); managed-surface=$($result.runtimeDeserializationDependencyDiagnostics.managedDeserializeSurfaceReady); full-consumer-report=$($result.runtimeDeserializationDependencyDiagnostics.fullPackageConsumerReportPresent); smoke-requested=$($result.runtimeDeserializationDependencyDiagnostics.fullPackageConsumerSmokeRequested); smoke-result=$($result.runtimeDeserializationDependencyDiagnostics.fullPackageConsumerSmokeResult); dependency-probe-only=$($result.runtimeDeserializationDependencyDiagnostics.dependencyProbeOnly); blocked-by-cuda-driver=$($result.runtimeDeserializationDependencyDiagnostics.blockedByCudaDriver); driver-mismatch=$($result.runtimeDeserializationDependencyDiagnostics.driverRuntimeMismatchClassified); classification=$($result.runtimeDeserializationDependencyDiagnostics.packageConsumerEvidenceClassification); blocker-category=$($result.runtimeDeserializationDependencyDiagnostics.runtimeProofBlockerCategory); owner-action=$($result.runtimeDeserializationDependencyDiagnostics.runtimeProofOwnerActionRequired); external-proof=$($result.runtimeDeserializationDependencyDiagnostics.externalRuntimeProofRequired); plugin-diagnostics=$($result.runtimeDeserializationDependencyDiagnostics.pluginLibraryDependencyDiagnosticsComplete); load-runtime-ownership=$($result.runtimeDeserializationDependencyDiagnostics.loadRuntimeOwnershipModeled); proof=$($result.runtimeDeserializationDependencyDiagnostics.isRuntimeExecutionProof); promote=$($result.runtimeDeserializationDependencyDiagnostics.canPromoteRuntimeProof); deferred-rows=$($result.runtimeDeserializationDependencyDiagnostics.hasDeferredRowEvidence)")
    $lines.Add("- runtime deserialization dependency diagnostics diagnostic: $($result.runtimeDeserializationDependencyDiagnostics.diagnostic)")
    $lines.Add("- runtime deserialization dependency diagnostics next owner action: $($result.runtimeDeserializationDependencyDiagnostics.nextOwnerAction)")
    $lines.Add("- allocator owner ledger design gate: $($result.allocatorOwnerLedgerDesignGate.status); marker=``$($result.allocatorOwnerLedgerDesignGate.marker)``; docfx-links=$($result.allocatorOwnerLedgerDesignGate.hasDocfxLinks); deferred-rows=$($result.allocatorOwnerLedgerDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- allocator owner ledger diagnostic: $($result.allocatorOwnerLedgerDesignGate.diagnostic)")
    $lines.Add("- allocator owner internal runtime prototype: $($result.allocatorOwnerInternalRuntimePrototype.status); marker=``$($result.allocatorOwnerInternalRuntimePrototype.marker)``; evidence-kind=$($result.allocatorOwnerInternalRuntimePrototype.evidenceKind); proof=$($result.allocatorOwnerInternalRuntimePrototype.isRealCallbackRuntimeProof); deferred-rows=$($result.allocatorOwnerInternalRuntimePrototype.hasDeferredRowEvidence)")
    $lines.Add("- allocator owner internal runtime prototype diagnostic: $($result.allocatorOwnerInternalRuntimePrototype.diagnostic)")
    $lines.Add("- allocator owner ledger safety gate: $($result.allocatorOwnerLedgerSafetyGate.status); marker=``$($result.allocatorOwnerLedgerSafetyGate.marker)``; evidence-kind=$($result.allocatorOwnerLedgerSafetyGate.evidenceKind); can-attempt=$($result.allocatorOwnerLedgerSafetyGate.canAttemptRuntimeProof); runtime-blocked=$($result.allocatorOwnerLedgerSafetyGate.runtimeProofBlocked); proof=$($result.allocatorOwnerLedgerSafetyGate.isRealCallbackRuntimeProof); deferred-rows=$($result.allocatorOwnerLedgerSafetyGate.hasDeferredRowEvidence)")
    $lines.Add("- allocator owner ledger safety gate diagnostic: $($result.allocatorOwnerLedgerSafetyGate.diagnostic)")
    $lines.Add("- output allocator internal runtime gate: $($result.outputAllocatorInternalRuntimeGate.status); marker=``$($result.outputAllocatorInternalRuntimeGate.marker)``; evidence-kind=$($result.outputAllocatorInternalRuntimeGate.evidenceKind); proof=$($result.outputAllocatorInternalRuntimeGate.isRealCallbackRuntimeProof); deferred-rows=$($result.outputAllocatorInternalRuntimeGate.hasDeferredRowEvidence)")
    $lines.Add("- output allocator internal runtime gate diagnostic: $($result.outputAllocatorInternalRuntimeGate.diagnostic)")
    $lines.Add("- output allocator callback owner design: $($result.outputAllocatorCallbackOwnerDesign.status); marker=``$($result.outputAllocatorCallbackOwnerDesign.marker)``; evidence-kind=$($result.outputAllocatorCallbackOwnerDesign.evidenceKind); proof=$($result.outputAllocatorCallbackOwnerDesign.isRealCallbackRuntimeProof); deferred-rows=$($result.outputAllocatorCallbackOwnerDesign.hasDeferredRowEvidence)")
    $lines.Add("- output allocator callback owner design diagnostic: $($result.outputAllocatorCallbackOwnerDesign.diagnostic)")
    $lines.Add("- output allocator attach/detach design gate: $($result.outputAllocatorAttachDetachDesignGate.status); marker=``$($result.outputAllocatorAttachDetachDesignGate.marker)``; evidence-kind=$($result.outputAllocatorAttachDetachDesignGate.evidenceKind); proof=$($result.outputAllocatorAttachDetachDesignGate.isRealCallbackRuntimeProof); deferred-rows=$($result.outputAllocatorAttachDetachDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- output allocator attach/detach design gate diagnostic: $($result.outputAllocatorAttachDetachDesignGate.diagnostic)")
    $lines.Add("- output buffer ownership safety gate: $($result.outputBufferOwnershipSafetyGate.status); marker=``$($result.outputBufferOwnershipSafetyGate.marker)``; evidence-kind=$($result.outputBufferOwnershipSafetyGate.evidenceKind); can-attempt=$($result.outputBufferOwnershipSafetyGate.canAttemptRuntimeProof); runtime-blocked=$($result.outputBufferOwnershipSafetyGate.runtimeProofBlocked); proof=$($result.outputBufferOwnershipSafetyGate.isRealCallbackRuntimeProof); deferred-rows=$($result.outputBufferOwnershipSafetyGate.hasDeferredRowEvidence)")
    $lines.Add("- output buffer ownership safety gate diagnostic: $($result.outputBufferOwnershipSafetyGate.diagnostic)")
    $lines.Add("- output allocator runtime proof precheck: $($result.outputAllocatorRuntimeProofPrecheck.status); marker=``$($result.outputAllocatorRuntimeProofPrecheck.marker)``; evidence-kind=$($result.outputAllocatorRuntimeProofPrecheck.evidenceKind); can-attempt=$($result.outputAllocatorRuntimeProofPrecheck.canAttemptRuntimeProof); runtime-blocked=$($result.outputAllocatorRuntimeProofPrecheck.runtimeProofBlocked); proof=$($result.outputAllocatorRuntimeProofPrecheck.isRealCallbackRuntimeProof); deferred-rows=$($result.outputAllocatorRuntimeProofPrecheck.hasDeferredRowEvidence)")
    $lines.Add("- output allocator runtime proof precheck diagnostic: $($result.outputAllocatorRuntimeProofPrecheck.diagnostic)")
    $lines.Add("- callback owner closure matrix: $($result.callbackOwnerClosureMatrix.status); marker=``$($result.callbackOwnerClosureMatrix.marker)``; evidence-kind=$($result.callbackOwnerClosureMatrix.evidenceKind); runtime-evidence=$($result.callbackOwnerClosureMatrix.runtimeEvidenceKind); family-count=$($result.callbackOwnerClosureMatrix.familyCount); design-gate-ready=$($result.callbackOwnerClosureMatrix.designGateReadyFamilyCount); closure-ready=$($result.callbackOwnerClosureMatrix.closureReadyFamilyCount); runtime-proof-attempt-ready=$($result.callbackOwnerClosureMatrix.runtimeProofAttemptReadyFamilyCount); package-consumer-proof-ready=$($result.callbackOwnerClosureMatrix.packageConsumerRuntimeProofReadyFamilyCount); runtime-blocked=$($result.callbackOwnerClosureMatrix.runtimeProofBlocked); proof=$($result.callbackOwnerClosureMatrix.isRealCallbackRuntimeProof); deferred-rows=$($result.callbackOwnerClosureMatrix.hasDeferredRowEvidence)")
    $lines.Add("- callback owner closure matrix diagnostic: $($result.callbackOwnerClosureMatrix.diagnostic)")
    $lines.Add("- debug listener callback owner design: $($result.debugListenerCallbackOwnerDesign.status); marker=``$($result.debugListenerCallbackOwnerDesign.marker)``; evidence-kind=$($result.debugListenerCallbackOwnerDesign.evidenceKind); proof=$($result.debugListenerCallbackOwnerDesign.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerCallbackOwnerDesign.hasDeferredRowEvidence)")
    $lines.Add("- debug listener callback owner design diagnostic: $($result.debugListenerCallbackOwnerDesign.diagnostic)")
    $lines.Add("- debug listener attach/detach design gate: $($result.debugListenerAttachDetachDesignGate.status); marker=``$($result.debugListenerAttachDetachDesignGate.marker)``; evidence-kind=$($result.debugListenerAttachDetachDesignGate.evidenceKind); proof=$($result.debugListenerAttachDetachDesignGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerAttachDetachDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener attach/detach design gate diagnostic: $($result.debugListenerAttachDetachDesignGate.diagnostic)")
    $lines.Add("- debug listener borrowed tensor safety gate: $($result.debugListenerBorrowedTensorSafetyGate.status); marker=``$($result.debugListenerBorrowedTensorSafetyGate.marker)``; evidence-kind=$($result.debugListenerBorrowedTensorSafetyGate.evidenceKind); can-attempt=$($result.debugListenerBorrowedTensorSafetyGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerBorrowedTensorSafetyGate.runtimeProofBlocked); proof=$($result.debugListenerBorrowedTensorSafetyGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerBorrowedTensorSafetyGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener borrowed tensor safety gate diagnostic: $($result.debugListenerBorrowedTensorSafetyGate.diagnostic)")
    $lines.Add("- debug listener attach/vtable safety gate: $($result.debugListenerAttachVTableSafetyGate.status); marker=``$($result.debugListenerAttachVTableSafetyGate.marker)``; evidence-kind=$($result.debugListenerAttachVTableSafetyGate.evidenceKind); can-attempt=$($result.debugListenerAttachVTableSafetyGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerAttachVTableSafetyGate.runtimeProofBlocked); proof=$($result.debugListenerAttachVTableSafetyGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerAttachVTableSafetyGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener attach/vtable safety gate diagnostic: $($result.debugListenerAttachVTableSafetyGate.diagnostic)")
    $lines.Add("- debug listener native attach/no-throw preflight: $($result.debugListenerNativeAttachNoThrowPreflight.status); marker=``$($result.debugListenerNativeAttachNoThrowPreflight.marker)``; evidence-kind=$($result.debugListenerNativeAttachNoThrowPreflight.evidenceKind); can-implement-native-attach=$($result.debugListenerNativeAttachNoThrowPreflight.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeAttachNoThrowPreflight.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeAttachNoThrowPreflight.runtimeProofBlocked); proof=$($result.debugListenerNativeAttachNoThrowPreflight.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeAttachNoThrowPreflight.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native attach/no-throw preflight diagnostic: $($result.debugListenerNativeAttachNoThrowPreflight.diagnostic)")
    $lines.Add("- debug listener native owner address design gate: $($result.debugListenerNativeOwnerAddressDesignGate.status); marker=``$($result.debugListenerNativeOwnerAddressDesignGate.marker)``; evidence-kind=$($result.debugListenerNativeOwnerAddressDesignGate.evidenceKind); can-implement-native-attach=$($result.debugListenerNativeOwnerAddressDesignGate.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeOwnerAddressDesignGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeOwnerAddressDesignGate.runtimeProofBlocked); proof=$($result.debugListenerNativeOwnerAddressDesignGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeOwnerAddressDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native owner address design gate diagnostic: $($result.debugListenerNativeOwnerAddressDesignGate.diagnostic)")
    $lines.Add("- debug listener native no-throw vtable design gate: $($result.debugListenerNativeNoThrowVTableDesignGate.status); marker=``$($result.debugListenerNativeNoThrowVTableDesignGate.marker)``; evidence-kind=$($result.debugListenerNativeNoThrowVTableDesignGate.evidenceKind); can-implement-native-attach=$($result.debugListenerNativeNoThrowVTableDesignGate.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeNoThrowVTableDesignGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeNoThrowVTableDesignGate.runtimeProofBlocked); proof=$($result.debugListenerNativeNoThrowVTableDesignGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeNoThrowVTableDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native no-throw vtable design gate diagnostic: $($result.debugListenerNativeNoThrowVTableDesignGate.diagnostic)")
    $lines.Add("- debug listener native attach entry design gate: $($result.debugListenerNativeAttachEntryDesignGate.status); marker=``$($result.debugListenerNativeAttachEntryDesignGate.marker)``; evidence-kind=$($result.debugListenerNativeAttachEntryDesignGate.evidenceKind); can-implement-native-attach=$($result.debugListenerNativeAttachEntryDesignGate.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeAttachEntryDesignGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeAttachEntryDesignGate.runtimeProofBlocked); proof=$($result.debugListenerNativeAttachEntryDesignGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeAttachEntryDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native attach entry design gate diagnostic: $($result.debugListenerNativeAttachEntryDesignGate.diagnostic)")
    $lines.Add("- debug listener native detach-before-release design gate: $($result.debugListenerNativeDetachBeforeReleaseDesignGate.status); marker=``$($result.debugListenerNativeDetachBeforeReleaseDesignGate.marker)``; evidence-kind=$($result.debugListenerNativeDetachBeforeReleaseDesignGate.evidenceKind); can-implement-native-attach=$($result.debugListenerNativeDetachBeforeReleaseDesignGate.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeDetachBeforeReleaseDesignGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeDetachBeforeReleaseDesignGate.runtimeProofBlocked); proof=$($result.debugListenerNativeDetachBeforeReleaseDesignGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeDetachBeforeReleaseDesignGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native detach-before-release design gate diagnostic: $($result.debugListenerNativeDetachBeforeReleaseDesignGate.diagnostic)")
    $lines.Add("- debug listener native owner lifecycle dry-run: $($result.debugListenerNativeOwnerLifecycleDryRun.status); marker=``$($result.debugListenerNativeOwnerLifecycleDryRun.marker)``; evidence-kind=$($result.debugListenerNativeOwnerLifecycleDryRun.evidenceKind); can-implement-native-attach=$($result.debugListenerNativeOwnerLifecycleDryRun.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeOwnerLifecycleDryRun.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeOwnerLifecycleDryRun.runtimeProofBlocked); proof=$($result.debugListenerNativeOwnerLifecycleDryRun.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeOwnerLifecycleDryRun.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native owner lifecycle dry-run diagnostic: $($result.debugListenerNativeOwnerLifecycleDryRun.diagnostic)")
    $lines.Add("- debug listener native attach entry runtime scaffold: $($result.debugListenerNativeAttachEntryRuntimeScaffold.status); marker=``$($result.debugListenerNativeAttachEntryRuntimeScaffold.marker)``; evidence-kind=$($result.debugListenerNativeAttachEntryRuntimeScaffold.evidenceKind); can-implement-native-attach=$($result.debugListenerNativeAttachEntryRuntimeScaffold.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeAttachEntryRuntimeScaffold.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeAttachEntryRuntimeScaffold.runtimeProofBlocked); proof=$($result.debugListenerNativeAttachEntryRuntimeScaffold.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeAttachEntryRuntimeScaffold.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native attach entry runtime scaffold diagnostic: $($result.debugListenerNativeAttachEntryRuntimeScaffold.diagnostic)")
    $lines.Add("- debug listener native attach entry minimal safety: $($result.debugListenerNativeAttachEntryMinimalSafety.status); marker=``$($result.debugListenerNativeAttachEntryMinimalSafety.marker)``; evidence-kind=$($result.debugListenerNativeAttachEntryMinimalSafety.evidenceKind); minimal-safety=$($result.debugListenerNativeAttachEntryMinimalSafety.minimalSafetyReady); scoped-native-attach-entry=$($result.debugListenerNativeAttachEntryMinimalSafety.nativeAttachEntryLocated); set-non-null=$($result.debugListenerNativeAttachEntryMinimalSafety.setDebugListenerNonNullEnabled); attach-blocked=$($result.debugListenerNativeAttachEntryMinimalSafety.nativeAttachWouldBeBlocked); can-attempt=$($result.debugListenerNativeAttachEntryMinimalSafety.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeAttachEntryMinimalSafety.runtimeProofBlocked); proof=$($result.debugListenerNativeAttachEntryMinimalSafety.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeAttachEntryMinimalSafety.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native attach entry minimal safety diagnostic: $($result.debugListenerNativeAttachEntryMinimalSafety.diagnostic)")
    $lines.Add("- debug listener native owner stable identity: $($result.debugListenerNativeOwnerStableIdentity.status); marker=``$($result.debugListenerNativeOwnerStableIdentity.marker)``; evidence-kind=$($result.debugListenerNativeOwnerStableIdentity.evidenceKind); stable-identity=$($result.debugListenerNativeOwnerStableIdentity.stableNativeOwnerIdentityReady); can-implement-native-attach=$($result.debugListenerNativeOwnerStableIdentity.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeOwnerStableIdentity.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeOwnerStableIdentity.runtimeProofBlocked); proof=$($result.debugListenerNativeOwnerStableIdentity.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeOwnerStableIdentity.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native owner stable identity diagnostic: $($result.debugListenerNativeOwnerStableIdentity.diagnostic)")
    $lines.Add("- debug listener native owner non-copyable storage: $($result.debugListenerNativeOwnerNonCopyableStorage.status); marker=``$($result.debugListenerNativeOwnerNonCopyableStorage.marker)``; evidence-kind=$($result.debugListenerNativeOwnerNonCopyableStorage.evidenceKind); noncopyable=$($result.debugListenerNativeOwnerNonCopyableStorage.nativeOwnerNonCopyableReady); can-implement-native-attach=$($result.debugListenerNativeOwnerNonCopyableStorage.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeOwnerNonCopyableStorage.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeOwnerNonCopyableStorage.runtimeProofBlocked); proof=$($result.debugListenerNativeOwnerNonCopyableStorage.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeOwnerNonCopyableStorage.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native owner non-copyable storage diagnostic: $($result.debugListenerNativeOwnerNonCopyableStorage.diagnostic)")
    $lines.Add("- debug listener native no-throw destructor: $($result.debugListenerNativeNoThrowDestructor.status); marker=``$($result.debugListenerNativeNoThrowDestructor.marker)``; evidence-kind=$($result.debugListenerNativeNoThrowDestructor.evidenceKind); no-throw-destructor=$($result.debugListenerNativeNoThrowDestructor.noThrowNativeDestructorReady); can-implement-native-attach=$($result.debugListenerNativeNoThrowDestructor.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeNoThrowDestructor.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeNoThrowDestructor.runtimeProofBlocked); proof=$($result.debugListenerNativeNoThrowDestructor.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeNoThrowDestructor.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native no-throw destructor diagnostic: $($result.debugListenerNativeNoThrowDestructor.diagnostic)")
    $lines.Add("- debug listener native owner lifecycle gate: $($result.debugListenerNativeOwnerLifecycleGate.status); marker=``$($result.debugListenerNativeOwnerLifecycleGate.marker)``; evidence-kind=$($result.debugListenerNativeOwnerLifecycleGate.evidenceKind); lifecycle-gate=$($result.debugListenerNativeOwnerLifecycleGate.lifecycleGateReady); can-implement-native-attach=$($result.debugListenerNativeOwnerLifecycleGate.canImplementNativeAttach); can-attempt=$($result.debugListenerNativeOwnerLifecycleGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeOwnerLifecycleGate.runtimeProofBlocked); proof=$($result.debugListenerNativeOwnerLifecycleGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeOwnerLifecycleGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native owner lifecycle gate diagnostic: $($result.debugListenerNativeOwnerLifecycleGate.diagnostic)")
    $lines.Add("- debug listener native attach bridge shape gate: $($result.debugListenerNativeAttachBridgeShapeGate.status); marker=``$($result.debugListenerNativeAttachBridgeShapeGate.marker)``; evidence-kind=$($result.debugListenerNativeAttachBridgeShapeGate.evidenceKind); attach-bridge-shape=$($result.debugListenerNativeAttachBridgeShapeGate.attachBridgeShapeGateReady); native-attach-entry=$($result.debugListenerNativeAttachBridgeShapeGate.nativeAttachEntryLocated); can-attempt=$($result.debugListenerNativeAttachBridgeShapeGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeAttachBridgeShapeGate.runtimeProofBlocked); proof=$($result.debugListenerNativeAttachBridgeShapeGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeAttachBridgeShapeGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native attach bridge shape gate diagnostic: $($result.debugListenerNativeAttachBridgeShapeGate.diagnostic)")
    $lines.Add("- debug listener exception/status mapping gate: $($result.debugListenerExceptionStatusMappingGate.status); marker=``$($result.debugListenerExceptionStatusMappingGate.marker)``; evidence-kind=$($result.debugListenerExceptionStatusMappingGate.evidenceKind); exception-status=$($result.debugListenerExceptionStatusMappingGate.exceptionStatusMappingGateReady); native-attach-entry=$($result.debugListenerExceptionStatusMappingGate.nativeAttachEntryLocated); can-attempt=$($result.debugListenerExceptionStatusMappingGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerExceptionStatusMappingGate.runtimeProofBlocked); proof=$($result.debugListenerExceptionStatusMappingGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerExceptionStatusMappingGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener exception/status mapping gate diagnostic: $($result.debugListenerExceptionStatusMappingGate.diagnostic)")
    $lines.Add("- debug listener in-flight accounting gate: $($result.debugListenerInFlightAccountingGate.status); marker=``$($result.debugListenerInFlightAccountingGate.marker)``; evidence-kind=$($result.debugListenerInFlightAccountingGate.evidenceKind); inflight-accounting=$($result.debugListenerInFlightAccountingGate.inFlightAccountingGateReady); native-attach-entry=$($result.debugListenerInFlightAccountingGate.nativeAttachEntryLocated); can-attempt=$($result.debugListenerInFlightAccountingGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerInFlightAccountingGate.runtimeProofBlocked); proof=$($result.debugListenerInFlightAccountingGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerInFlightAccountingGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener in-flight accounting gate diagnostic: $($result.debugListenerInFlightAccountingGate.diagnostic)")
    $lines.Add("- debug listener native no-throw vtable scaffold gate: $($result.debugListenerNativeNoThrowVTableScaffoldGate.status); marker=``$($result.debugListenerNativeNoThrowVTableScaffoldGate.marker)``; evidence-kind=$($result.debugListenerNativeNoThrowVTableScaffoldGate.evidenceKind); vtable-scaffold=$($result.debugListenerNativeNoThrowVTableScaffoldGate.vTableScaffoldGateReady); native-vtable-design=$($result.debugListenerNativeNoThrowVTableScaffoldGate.nativeVTableDesignReady); native-attach-entry=$($result.debugListenerNativeNoThrowVTableScaffoldGate.nativeAttachEntryLocated); can-attempt=$($result.debugListenerNativeNoThrowVTableScaffoldGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeNoThrowVTableScaffoldGate.runtimeProofBlocked); proof=$($result.debugListenerNativeNoThrowVTableScaffoldGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeNoThrowVTableScaffoldGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native no-throw vtable scaffold gate diagnostic: $($result.debugListenerNativeNoThrowVTableScaffoldGate.diagnostic)")
    $lines.Add("- debug listener no-throw vtable callback stub: $($result.debugListenerNoThrowVTableCallbackStub.status); marker=``$($result.debugListenerNoThrowVTableCallbackStub.marker)``; evidence-kind=$($result.debugListenerNoThrowVTableCallbackStub.evidenceKind); callback-stub=$($result.debugListenerNoThrowVTableCallbackStub.callbackStubGateReady); metadata-copy=$($result.debugListenerNoThrowVTableCallbackStub.callbackMetadataCopyReady); native-vtable-installed=$($result.debugListenerNoThrowVTableCallbackStub.nativeVTableInstalled); set-non-null=$($result.debugListenerNoThrowVTableCallbackStub.setDebugListenerNonNullEnabled); can-attempt=$($result.debugListenerNoThrowVTableCallbackStub.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNoThrowVTableCallbackStub.runtimeProofBlocked); proof=$($result.debugListenerNoThrowVTableCallbackStub.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNoThrowVTableCallbackStub.hasDeferredRowEvidence)")
    $lines.Add("- debug listener no-throw vtable callback stub diagnostic: $($result.debugListenerNoThrowVTableCallbackStub.diagnostic)")
    $lines.Add("- debug listener borrowed debug tensor metadata runtime gate: $($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.status); marker=``$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.marker)``; evidence-kind=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.evidenceKind); metadata-gate=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.metadataGateReady); metadata-copy=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.borrowedDebugTensorMetadataCopyReady); pointer-escape-blocked=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.borrowedDebugTensorPointerEscapeBlocked); data-pointer-escape-blocked=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.borrowedDebugTensorDataPointerEscapeBlocked); lifetime-ready=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.borrowedDebugTensorLifetimeReady); data-lifetime-ready=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.borrowedDebugTensorDataLifetimeReady); process-runtime=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.processDebugTensorRuntimeReady); can-attempt=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.runtimeProofBlocked); proof=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.hasDeferredRowEvidence)")
    $lines.Add("- debug listener borrowed debug tensor metadata runtime gate diagnostic: $($result.debugListenerBorrowedDebugTensorMetadataRuntimeGate.diagnostic)")
    $lines.Add("- debug listener native vtable install preflight: $($result.debugListenerNativeVTableInstallPreflight.status); marker=``$($result.debugListenerNativeVTableInstallPreflight.marker)``; evidence-kind=$($result.debugListenerNativeVTableInstallPreflight.evidenceKind); preflight=$($result.debugListenerNativeVTableInstallPreflight.nativeVTableInstallPreflightReady); shape=$($result.debugListenerNativeVTableInstallPreflight.vTableInstallShapeReady); pointer-free=$($result.debugListenerNativeVTableInstallPreflight.vTableInstallPointerFree); native-vtable-installed=$($result.debugListenerNativeVTableInstallPreflight.nativeVTableInstalled); install-runtime=$($result.debugListenerNativeVTableInstallPreflight.nativeVTableInstallRuntimeReady); can-non-null=$($result.debugListenerNativeVTableInstallPreflight.canEnableSetDebugListenerNonNull); can-install=$($result.debugListenerNativeVTableInstallPreflight.canInstallNativeVTable); process-runtime=$($result.debugListenerNativeVTableInstallPreflight.processDebugTensorRuntimeReady); can-attempt=$($result.debugListenerNativeVTableInstallPreflight.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeVTableInstallPreflight.runtimeProofBlocked); proof=$($result.debugListenerNativeVTableInstallPreflight.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeVTableInstallPreflight.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native vtable install preflight diagnostic: $($result.debugListenerNativeVTableInstallPreflight.diagnostic)")
    $lines.Add("- debug listener native owner/vtable install experiment: $($result.debugListenerNativeOwnerVTableInstallExperiment.status); marker=``$($result.debugListenerNativeOwnerVTableInstallExperiment.marker)``; evidence-kind=$($result.debugListenerNativeOwnerVTableInstallExperiment.evidenceKind); experiment=$($result.debugListenerNativeOwnerVTableInstallExperiment.experimentShapeReady); guard=$($result.debugListenerNativeOwnerVTableInstallExperiment.installAttemptGuardReady); non-null=$($result.debugListenerNativeOwnerVTableInstallExperiment.nonNullAttachEnabled); runtime-proof-enabled=$($result.debugListenerNativeOwnerVTableInstallExperiment.runtimeProofEnabled); attempted=$($result.debugListenerNativeOwnerVTableInstallExperiment.nativeVTableInstallAttempted); installed=$($result.debugListenerNativeOwnerVTableInstallExperiment.nativeVTableInstalled); rollback=$($result.debugListenerNativeOwnerVTableInstallExperiment.rollbackReady); detach-before-release=$($result.debugListenerNativeOwnerVTableInstallExperiment.detachBeforeReleaseReady); status-mapping=$($result.debugListenerNativeOwnerVTableInstallExperiment.failureStatusMappingReady); pointer-free=$($result.debugListenerNativeOwnerVTableInstallExperiment.pointerFree); can-non-null=$($result.debugListenerNativeOwnerVTableInstallExperiment.canEnableSetDebugListenerNonNull); can-install=$($result.debugListenerNativeOwnerVTableInstallExperiment.canInstallNativeVTable); process-runtime=$($result.debugListenerNativeOwnerVTableInstallExperiment.processDebugTensorRuntimeReady); can-attempt=$($result.debugListenerNativeOwnerVTableInstallExperiment.canAttemptRuntimeProof); runtime-blocked=$($result.debugListenerNativeOwnerVTableInstallExperiment.runtimeProofBlocked); proof=$($result.debugListenerNativeOwnerVTableInstallExperiment.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerNativeOwnerVTableInstallExperiment.hasDeferredRowEvidence)")
    $lines.Add("- debug listener native owner/vtable install experiment diagnostic: $($result.debugListenerNativeOwnerVTableInstallExperiment.diagnostic)")
    $lines.Add("- debug listener runtime proof precheck: $($result.debugListenerRuntimeProofPrecheck.status); marker=``$($result.debugListenerRuntimeProofPrecheck.marker)``; evidence-kind=$($result.debugListenerRuntimeProofPrecheck.evidenceKind); can-attempt=$($result.debugListenerRuntimeProofPrecheck.canAttemptRuntimeProof); proof=$($result.debugListenerRuntimeProofPrecheck.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerRuntimeProofPrecheck.hasDeferredRowEvidence)")
    $lines.Add("- debug listener runtime proof precheck diagnostic: $($result.debugListenerRuntimeProofPrecheck.diagnostic)")
    $lines.Add("- debug listener runtime proof attempt preflight: $($result.debugListenerRuntimeProofAttemptPreflight.status); marker=``$($result.debugListenerRuntimeProofAttemptPreflight.marker)``; evidence-kind=$($result.debugListenerRuntimeProofAttemptPreflight.evidenceKind); non-null-attach=$($result.debugListenerRuntimeProofAttemptPreflight.canEnableSetDebugListenerNonNull); native-vtable=$($result.debugListenerRuntimeProofAttemptPreflight.canInstallNativeVTable); process-runtime=$($result.debugListenerRuntimeProofAttemptPreflight.canCallProcessDebugTensorRuntime); promote=$($result.debugListenerRuntimeProofAttemptPreflight.canPromoteRealCallbackRuntime); proof=$($result.debugListenerRuntimeProofAttemptPreflight.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerRuntimeProofAttemptPreflight.hasDeferredRowEvidence)")
    $lines.Add("- debug listener runtime proof attempt preflight diagnostic: $($result.debugListenerRuntimeProofAttemptPreflight.diagnostic)")
    $lines.Add("- debug listener real non-null attach runtime smoke: $($result.debugListenerRealNonNullAttachRuntimeSmoke.status); marker=``$($result.debugListenerRealNonNullAttachRuntimeSmoke.marker)``; evidence-kind=$($result.debugListenerRealNonNullAttachRuntimeSmoke.evidenceKind); runtime-evidence=$($result.debugListenerRealNonNullAttachRuntimeSmoke.runtimeEvidenceKind); default-skipped=$($result.debugListenerRealNonNullAttachRuntimeSmoke.defaultSkippedReady); opt-in-guard=$($result.debugListenerRealNonNullAttachRuntimeSmoke.optInGuardReady); full-package-report-required=$($result.debugListenerRealNonNullAttachRuntimeSmoke.fullPackageConsumerReportRequired); attach-guard=$($result.debugListenerRealNonNullAttachRuntimeSmoke.attachGuardReady); native-vtable=$($result.debugListenerRealNonNullAttachRuntimeSmoke.nativeVTableReady); callback-runtime=$($result.debugListenerRealNonNullAttachRuntimeSmoke.callbackInvocationReady); attempted=$($result.debugListenerRealNonNullAttachRuntimeSmoke.attachAttempted); attached=$($result.debugListenerRealNonNullAttachRuntimeSmoke.attachSucceeded); installed=$($result.debugListenerRealNonNullAttachRuntimeSmoke.nativeVTableInstalled); invoked=$($result.debugListenerRealNonNullAttachRuntimeSmoke.processDebugTensorInvoked); pointer-free=$($result.debugListenerRealNonNullAttachRuntimeSmoke.reportPointerFree); promote=$($result.debugListenerRealNonNullAttachRuntimeSmoke.canPromoteRealCallbackRuntime); proof=$($result.debugListenerRealNonNullAttachRuntimeSmoke.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerRealNonNullAttachRuntimeSmoke.hasDeferredRowEvidence)")
    $lines.Add("- debug listener real non-null attach runtime smoke diagnostic: $($result.debugListenerRealNonNullAttachRuntimeSmoke.diagnostic)")
    $lines.Add("- debug listener processDebugTensor callback trampoline: $($result.debugListenerProcessDebugTensorCallbackTrampoline.status); marker=``$($result.debugListenerProcessDebugTensorCallbackTrampoline.marker)``; evidence-kind=$($result.debugListenerProcessDebugTensorCallbackTrampoline.evidenceKind); runtime-evidence=$($result.debugListenerProcessDebugTensorCallbackTrampoline.runtimeEvidenceKind); trampoline=$($result.debugListenerProcessDebugTensorCallbackTrampoline.trampolineShapeReady); native-entry=$($result.debugListenerProcessDebugTensorCallbackTrampoline.nativeCallbackEntryLocated); no-throw=$($result.debugListenerProcessDebugTensorCallbackTrampoline.noThrowCallbackEntryReady); exception-capture=$($result.debugListenerProcessDebugTensorCallbackTrampoline.exceptionCaptureReady); status-mapping=$($result.debugListenerProcessDebugTensorCallbackTrampoline.callbackStatusMappingReady); inflight=$($result.debugListenerProcessDebugTensorCallbackTrampoline.inFlightAccountingReady); metadata-copy=$($result.debugListenerProcessDebugTensorCallbackTrampoline.borrowedDebugTensorMetadataCopyReady); pointer-free=$($result.debugListenerProcessDebugTensorCallbackTrampoline.pointerFreeSurfaceReady); process-runtime=$($result.debugListenerProcessDebugTensorCallbackTrampoline.processDebugTensorRuntimeReady); invoked=$($result.debugListenerProcessDebugTensorCallbackTrampoline.processDebugTensorInvoked); promote=$($result.debugListenerProcessDebugTensorCallbackTrampoline.canPromoteRealCallbackRuntime); proof=$($result.debugListenerProcessDebugTensorCallbackTrampoline.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerProcessDebugTensorCallbackTrampoline.hasDeferredRowEvidence)")
    $lines.Add("- debug listener processDebugTensor callback trampoline diagnostic: $($result.debugListenerProcessDebugTensorCallbackTrampoline.diagnostic)")
    $lines.Add("- debug listener real callback runtime proof gate: $($result.debugListenerRealCallbackRuntimeProof.status); marker=``$($result.debugListenerRealCallbackRuntimeProof.marker)``; evidence-kind=$($result.debugListenerRealCallbackRuntimeProof.evidenceKind); runtime-evidence=$($result.debugListenerRealCallbackRuntimeProof.runtimeEvidenceKind); opt-in=$($result.debugListenerRealCallbackRuntimeProof.optInEnabled); full-consumer=$($result.debugListenerRealCallbackRuntimeProof.fullPackageConsumerReport); runtime-smoke=$($result.debugListenerRealCallbackRuntimeProof.runtimeSmokeReady); trampoline=$($result.debugListenerRealCallbackRuntimeProof.trampolineShapeReady); attach=$($result.debugListenerRealCallbackRuntimeProof.attachSucceeded); detach=$($result.debugListenerRealCallbackRuntimeProof.detachSucceeded); rollback=$($result.debugListenerRealCallbackRuntimeProof.rollbackSucceeded); vtable=$($result.debugListenerRealCallbackRuntimeProof.nativeVTableInstalled); invoked=$($result.debugListenerRealCallbackRuntimeProof.processDebugTensorInvoked); invocation-count=$($result.debugListenerRealCallbackRuntimeProof.invocationCount); attempted-no-invocation=$($result.debugListenerRealCallbackRuntimeProof.attemptedNoInvocation); promote=$($result.debugListenerRealCallbackRuntimeProof.canPromoteRealCallbackRuntime); proof=$($result.debugListenerRealCallbackRuntimeProof.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerRealCallbackRuntimeProof.hasDeferredRowEvidence)")
    $lines.Add("- debug listener real callback runtime proof gate diagnostic: $($result.debugListenerRealCallbackRuntimeProof.diagnostic)")
    $lines.Add("- debug listener callback proof gap report: $($result.debugListenerCallbackProofGapReport.status); marker=``$($result.debugListenerCallbackProofGapReport.marker)``; evidence-kind=$($result.debugListenerCallbackProofGapReport.evidenceKind); runtime-evidence=$($result.debugListenerCallbackProofGapReport.runtimeEvidenceKind); non-null-disabled=$($result.debugListenerCallbackProofGapReport.nonNullAttachStillDisabled); native-attach-ready=$($result.debugListenerCallbackProofGapReport.nativeAttachEntryReady); vtable-blocked=$($result.debugListenerCallbackProofGapReport.nativeVTableInstallBlocked); no-throw=$($result.debugListenerCallbackProofGapReport.noThrowCallbackEntryReady); status-mapping=$($result.debugListenerCallbackProofGapReport.exceptionStatusMappingReady); inflight=$($result.debugListenerCallbackProofGapReport.inFlightAccountingReady); metadata-copy=$($result.debugListenerCallbackProofGapReport.borrowedDebugTensorMetadataCopied); detach-rollback=$($result.debugListenerCallbackProofGapReport.detachRollbackReady); invoked=$($result.debugListenerCallbackProofGapReport.processDebugTensorRuntimeInvoked); full-consumer-proof=$($result.debugListenerCallbackProofGapReport.fullPackageConsumerRuntimeProofReady); pointer-free=$($result.debugListenerCallbackProofGapReport.pointerFreeSurfaceReady); invocation-count=$($result.debugListenerCallbackProofGapReport.invocationCount); gap-count=$($result.debugListenerCallbackProofGapReport.gapReasonCount); promote=$($result.debugListenerCallbackProofGapReport.canPromoteRealCallbackRuntime); proof=$($result.debugListenerCallbackProofGapReport.isRealCallbackRuntimeProof); deferred-rows=$($result.debugListenerCallbackProofGapReport.hasDeferredRowEvidence)")
    $lines.Add("- debug listener callback proof gap report diagnostic: $($result.debugListenerCallbackProofGapReport.diagnostic)")
    $lines.Add("- real callback trampoline gate: $($result.realCallbackTrampolineGate.status); marker=``$($result.realCallbackTrampolineGate.marker)``; evidence-kind=$($result.realCallbackTrampolineGate.evidenceKind); real-callback-runtime=$($result.realCallbackTrampolineGate.isRealCallbackRuntimeProof); deferred-rows=$($result.realCallbackTrampolineGate.hasDeferredRowEvidence)")
    $lines.Add("- real callback trampoline diagnostic: $($result.realCallbackTrampolineGate.diagnostic)")
    $lines.Add("- real callback runtime evidence schema: $($result.realCallbackRuntimeEvidenceSchema.status); marker=``$($result.realCallbackRuntimeEvidenceSchema.marker)``; evidence-kind=$($result.realCallbackRuntimeEvidenceSchema.evidenceKind); runtime-evidence=$($result.realCallbackRuntimeEvidenceSchema.runtimeEvidenceKind)")
    $lines.Add("- real callback runtime evidence: $($result.realCallbackRuntimeEvidence.status); marker=``$($result.realCallbackRuntimeEvidence.marker)``; evidence-kind=$($result.realCallbackRuntimeEvidence.evidenceKind); proof=$($result.realCallbackRuntimeEvidence.isRealCallbackRuntimeProof); source=$($result.realCallbackRuntimeEvidence.source)")
    $lines.Add("- split components: $($result.splitPackages.status) $($result.splitPackages.foundCount)/$($result.splitPackages.expectedCount)")
    $lines.Add("- split collection package: $($result.splitCollectionPackage.status) ``$($result.splitCollectionPackage.version)``")
    $lines.Add("- split collection consumer: $($result.splitCollectionConsumer.status) $($result.splitCollectionConsumer.smokeResult); runtime-execution=$($result.splitCollectionConsumer.isRuntimeExecutionEvidence)")
    $lines.Add("- full runtime package: $($result.fullRuntimePackage.status) ``$($result.fullRuntimePackage.version)``")
    $lines.Add("- full package consumer: $($result.fullPackageConsumer.status) $($result.fullPackageConsumer.smokeResult)")
    $lines.Add("- full package consumer evidence scope: $($result.fullPackageConsumer.evidenceKind); classification=$($result.fullPackageConsumer.runtimeSmokeClassification); full-runtime=$($result.fullPackageConsumer.isFullRuntimeEvidence); runtime-execution=$($result.fullPackageConsumer.isRuntimeExecutionEvidence); dependency-probe-only=$($result.fullPackageConsumer.isDependencyProbeOnly); real-callback-proof=$($result.fullPackageConsumer.isRealCallbackRuntimeProof)")
    $lines.Add("- full package consumer callback runtime report: status=$($result.fullPackageConsumer.callbackRuntimeEvidenceStatus); evidence-kind=$($result.fullPackageConsumer.callbackRuntimeEvidenceKind); proof=$($result.fullPackageConsumer.callbackRuntimeIsProof); diagnostic=$($result.fullPackageConsumer.callbackRuntimeDiagnostic)")
    $lines.Add("- runtime execution smoke: $($result.runtimeExecution.status); $($result.runtimeExecution.diagnostic)")
    $lines.Add("- runtime proof status: $($result.runtimeProofStatus); release-required=$($result.runtimeProofRequiredForRelease); $($result.runtimeProofDiagnostic)")
    $lines.Add("- runtime proof blocker owner action: $($result.runtimeProofBlockerOwnerAction.status); category=$($result.runtimeProofBlockerOwnerAction.blockerCategory); external-input=$($result.runtimeProofBlockerOwnerAction.externalInputRequired); why-not-smoke-passed=$($result.runtimeProofBlockerOwnerAction.whyNotSmokePassed)")
    $lines.Add("- vendor roots: TensorRT=``$($result.fullVendorInputs.tensorRtRoot)`` CUDA=``$($result.fullVendorInputs.cudaRoot)`` cuDNN=``$($result.fullVendorInputs.cudnnRoot)``")
    $lines.Add("- readiness blockers: $(@($result.readinessBlockers).Count)")

    if ($result.runtimeProofBlockerOwnerAction) {
      $lines.Add("")
      $lines.Add("| Runtime proof owner action | Value |")
      $lines.Add("| --- | --- |")
      $lines.Add("| status | ``$(ConvertTo-MarkdownCell -Value ([string]$result.runtimeProofBlockerOwnerAction.status))`` |")
      $lines.Add("| blocker category | ``$(ConvertTo-MarkdownCell -Value ([string]$result.runtimeProofBlockerOwnerAction.blockerCategory))`` |")
      $lines.Add("| summary | $(ConvertTo-MarkdownCell -Value ([string]$result.runtimeProofBlockerOwnerAction.summary)) |")
      $lines.Add("| next action | $(ConvertTo-MarkdownCell -Value ([string]$result.runtimeProofBlockerOwnerAction.nextAction)) |")
      $lines.Add("| evidence path | ``$(ConvertTo-MarkdownCell -Value ([string]$result.runtimeProofBlockerOwnerAction.packageConsumerReportPath))`` |")
      foreach ($command in @($result.runtimeProofBlockerOwnerAction.suggestedCommands)) {
        $lines.Add("| suggested command | ``$(ConvertTo-MarkdownCell -Value ([string]$command))`` |")
      }
    }

    if (@($result.runtimeExecution.consumers).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Runtime execution scope | Consumer status | Smoke requested | Smoke result | Classification | Exit code | Evidence kind | Runtime execution | Dependency probe only | Real callback proof | Diagnostic | Report |")
      $lines.Add("| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |")
      foreach ($consumer in @($result.runtimeExecution.consumers)) {
        $exitCode = if ($null -eq $consumer.smokeExitCode) { "" } else { [string]$consumer.smokeExitCode }
        $lines.Add("| $($consumer.scope) | $($consumer.status) | $($consumer.smokeRequested) | $($consumer.smokeResult) | $($consumer.runtimeSmokeClassification) | $exitCode | $($consumer.evidenceKind) | $($consumer.isRuntimeExecutionEvidence) | $($consumer.isDependencyProbeOnly) | $($consumer.isRealCallbackRuntimeProof) | $(ConvertTo-MarkdownCell -Value ([string]$consumer.smokeDiagnostic)) | ``$(ConvertTo-MarkdownCell -Value ([string]$consumer.reportPath))`` |")
      }
    }

    if (@($result.splitPackages.packages).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Split role | Package | Status | Version |")
      $lines.Add("| --- | --- | --- | --- |")
      foreach ($splitPackage in @($result.splitPackages.packages)) {
        $lines.Add("| $($splitPackage.role) | ``$($splitPackage.packageId)`` | $($splitPackage.status) | ``$($splitPackage.version)`` |")
      }
    }

    if (@($result.readinessBlockers).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Blocker category | Status | Detail | Next action | Suggested command | External input | Evidence path |")
      $lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
      foreach ($blocker in @($result.readinessBlockers)) {
        $lines.Add("| $(ConvertTo-MarkdownCell -Value ([string]$blocker.category)) | $(ConvertTo-MarkdownCell -Value ([string]$blocker.status)) | $(ConvertTo-MarkdownCell -Value ([string]$blocker.detail)) | $(ConvertTo-MarkdownCell -Value ([string]$blocker.nextAction)) | ``$(ConvertTo-MarkdownCell -Value ([string]$blocker.suggestedCommand))`` | $($blocker.isExternalInputRequired) | ``$(ConvertTo-MarkdownCell -Value ([string]$blocker.evidencePath))`` |")
      }
    }
    else {
      $lines.Add("- readiness blockers: none")
    }

    Add-MissingEvidenceTable -Lines (,$lines) -Title "Error recorder diagnostics design gate missing evidence" -Evidence $result.errorRecorderDiagnosticsDesignGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Dimension expression snapshot design gate missing evidence" -Evidence $result.dimensionExpressionSnapshotDesignGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Calibrator metadata design gate missing evidence" -Evidence $result.calibratorMetadataDesignGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Runtime deserialization boundary precheck missing evidence" -Evidence $result.runtimeDeserializationBoundaryPrecheck
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Runtime deserialization dependency diagnostics missing evidence" -Evidence $result.runtimeDeserializationDependencyDiagnostics
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Callback owner closure matrix missing evidence" -Evidence $result.callbackOwnerClosureMatrix

    if (@($result.allocatorOwnerLedgerDesignGate.missingMarkers).Count -gt 0 -or @($result.allocatorOwnerLedgerDesignGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Allocator owner ledger missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.allocatorOwnerLedgerDesignGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.allocatorOwnerLedgerDesignGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.allocatorOwnerInternalRuntimePrototype.missingMarkers).Count -gt 0 -or @($result.allocatorOwnerInternalRuntimePrototype.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Allocator owner internal runtime prototype missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.allocatorOwnerInternalRuntimePrototype.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.allocatorOwnerInternalRuntimePrototype.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.allocatorOwnerLedgerSafetyGate.missingMarkers).Count -gt 0 -or @($result.allocatorOwnerLedgerSafetyGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Allocator owner ledger safety gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.allocatorOwnerLedgerSafetyGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.allocatorOwnerLedgerSafetyGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.outputAllocatorInternalRuntimeGate.missingMarkers).Count -gt 0 -or @($result.outputAllocatorInternalRuntimeGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Output allocator internal runtime gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.outputAllocatorInternalRuntimeGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.outputAllocatorInternalRuntimeGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.outputAllocatorCallbackOwnerDesign.missingMarkers).Count -gt 0 -or @($result.outputAllocatorCallbackOwnerDesign.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Output allocator callback owner design missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.outputAllocatorCallbackOwnerDesign.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.outputAllocatorCallbackOwnerDesign.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.outputAllocatorAttachDetachDesignGate.missingMarkers).Count -gt 0 -or @($result.outputAllocatorAttachDetachDesignGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Output allocator attach/detach design gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.outputAllocatorAttachDetachDesignGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.outputAllocatorAttachDetachDesignGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.outputBufferOwnershipSafetyGate.missingMarkers).Count -gt 0 -or @($result.outputBufferOwnershipSafetyGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Output buffer ownership safety gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.outputBufferOwnershipSafetyGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.outputBufferOwnershipSafetyGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.outputAllocatorRuntimeProofPrecheck.missingMarkers).Count -gt 0 -or @($result.outputAllocatorRuntimeProofPrecheck.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Output allocator runtime proof precheck missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.outputAllocatorRuntimeProofPrecheck.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.outputAllocatorRuntimeProofPrecheck.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerCallbackOwnerDesign.missingMarkers).Count -gt 0 -or @($result.debugListenerCallbackOwnerDesign.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener callback owner design missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerCallbackOwnerDesign.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerCallbackOwnerDesign.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerAttachDetachDesignGate.missingMarkers).Count -gt 0 -or @($result.debugListenerAttachDetachDesignGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener attach/detach design gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerAttachDetachDesignGate.missingMarkers)) {
        $lines.Add("| missing-marker | ``$marker`` |")
      }

      foreach ($row in @($result.debugListenerAttachDetachDesignGate.missingDeferredRows)) {
        $lines.Add("| missing-deferred-row | ``$row`` |")
      }
    }

    if (@($result.debugListenerBorrowedTensorSafetyGate.missingMarkers).Count -gt 0 -or @($result.debugListenerBorrowedTensorSafetyGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener borrowed tensor safety gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerBorrowedTensorSafetyGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerBorrowedTensorSafetyGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerAttachVTableSafetyGate.missingMarkers).Count -gt 0 -or @($result.debugListenerAttachVTableSafetyGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener attach/vtable safety gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerAttachVTableSafetyGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerAttachVTableSafetyGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeAttachNoThrowPreflight.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeAttachNoThrowPreflight.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native attach/no-throw preflight missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeAttachNoThrowPreflight.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeAttachNoThrowPreflight.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeOwnerAddressDesignGate.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeOwnerAddressDesignGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native owner address design gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeOwnerAddressDesignGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeOwnerAddressDesignGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeNoThrowVTableDesignGate.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeNoThrowVTableDesignGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native no-throw vtable design gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeNoThrowVTableDesignGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeNoThrowVTableDesignGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeAttachEntryDesignGate.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeAttachEntryDesignGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native attach entry design gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeAttachEntryDesignGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeAttachEntryDesignGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeDetachBeforeReleaseDesignGate.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeDetachBeforeReleaseDesignGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native detach-before-release design gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeDetachBeforeReleaseDesignGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeDetachBeforeReleaseDesignGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeOwnerLifecycleDryRun.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeOwnerLifecycleDryRun.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native owner lifecycle dry-run missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeOwnerLifecycleDryRun.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeOwnerLifecycleDryRun.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeAttachEntryRuntimeScaffold.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeAttachEntryRuntimeScaffold.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native attach entry runtime scaffold missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeAttachEntryRuntimeScaffold.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeAttachEntryRuntimeScaffold.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeAttachEntryMinimalSafety.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeAttachEntryMinimalSafety.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native attach entry minimal safety missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeAttachEntryMinimalSafety.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeAttachEntryMinimalSafety.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeOwnerStableIdentity.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeOwnerStableIdentity.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native owner stable identity missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeOwnerStableIdentity.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeOwnerStableIdentity.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeOwnerNonCopyableStorage.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeOwnerNonCopyableStorage.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native owner non-copyable storage missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeOwnerNonCopyableStorage.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeOwnerNonCopyableStorage.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeNoThrowDestructor.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeNoThrowDestructor.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native no-throw destructor missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeNoThrowDestructor.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeNoThrowDestructor.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.debugListenerNativeOwnerLifecycleGate.missingMarkers).Count -gt 0 -or @($result.debugListenerNativeOwnerLifecycleGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener native owner lifecycle gate missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerNativeOwnerLifecycleGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerNativeOwnerLifecycleGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener native attach bridge shape gate missing evidence" -Evidence $result.debugListenerNativeAttachBridgeShapeGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener exception/status mapping gate missing evidence" -Evidence $result.debugListenerExceptionStatusMappingGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener in-flight accounting gate missing evidence" -Evidence $result.debugListenerInFlightAccountingGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener native no-throw vtable scaffold gate missing evidence" -Evidence $result.debugListenerNativeNoThrowVTableScaffoldGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener no-throw vtable callback stub missing evidence" -Evidence $result.debugListenerNoThrowVTableCallbackStub
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener borrowed debug tensor metadata runtime gate missing evidence" -Evidence $result.debugListenerBorrowedDebugTensorMetadataRuntimeGate
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener native vtable install preflight missing evidence" -Evidence $result.debugListenerNativeVTableInstallPreflight
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener native owner/vtable install experiment missing evidence" -Evidence $result.debugListenerNativeOwnerVTableInstallExperiment
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener runtime proof attempt preflight missing evidence" -Evidence $result.debugListenerRuntimeProofAttemptPreflight
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener real non-null attach runtime smoke missing evidence" -Evidence $result.debugListenerRealNonNullAttachRuntimeSmoke
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener processDebugTensor callback trampoline missing evidence" -Evidence $result.debugListenerProcessDebugTensorCallbackTrampoline
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener real callback runtime proof gate missing evidence" -Evidence $result.debugListenerRealCallbackRuntimeProof
    Add-MissingEvidenceTable -Lines (,$lines) -Title "Debug listener callback proof gap report missing evidence" -Evidence $result.debugListenerCallbackProofGapReport

    if (@($result.debugListenerRuntimeProofPrecheck.missingMarkers).Count -gt 0 -or @($result.debugListenerRuntimeProofPrecheck.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Debug listener runtime proof precheck missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.debugListenerRuntimeProofPrecheck.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.debugListenerRuntimeProofPrecheck.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.realCallbackTrampolineGate.missingMarkers).Count -gt 0 -or @($result.realCallbackTrampolineGate.missingDeferredRows).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Real callback trampoline missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.realCallbackTrampolineGate.missingMarkers)) {
        $lines.Add("| marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($row in @($result.realCallbackTrampolineGate.missingDeferredRows)) {
        $lines.Add("| deferred-row | ``$(ConvertTo-MarkdownCell -Value ([string]$row))`` |")
      }
    }

    if (@($result.realCallbackRuntimeEvidenceSchema.missingMarkers).Count -gt 0 -or @($result.realCallbackRuntimeEvidence.missingSmokeMarkers).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Real callback runtime missing evidence | Value |")
      $lines.Add("| --- | --- |")
      foreach ($marker in @($result.realCallbackRuntimeEvidenceSchema.missingMarkers)) {
        $lines.Add("| schema-marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }

      foreach ($marker in @($result.realCallbackRuntimeEvidence.missingSmokeMarkers)) {
        $lines.Add("| smoke-marker | ``$(ConvertTo-MarkdownCell -Value ([string]$marker))`` |")
      }
    }

    if (@($result.fullVendorInputs.blockers).Count -eq 0) {
      $lines.Add("- vendor blockers: none")
    }
    else {
      foreach ($blocker in @($result.fullVendorInputs.blockers)) {
        $lines.Add("- vendor blocker: $blocker")
      }
    }

    if (@($result.fullVendorInputs.missingAssetSummaryByKind).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Missing vendor kind | Count | Missing roots | Missing files | First missing relative path | Root |")
      $lines.Add("| --- | ---: | ---: | ---: | --- | --- |")
      foreach ($summary in @($result.fullVendorInputs.missingAssetSummaryByKind)) {
        $lines.Add("| $($summary.kind) | $($summary.missingCount) | $($summary.missingRootCount) | $($summary.missingAssetCount) | ``$($summary.firstMissingRelativePath)`` | ``$($summary.root)`` |")
      }
    }

    if (@($result.fullVendorInputs.vendorRootDiagnostics).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Vendor root | Exists | DLLs | LIBs | Missing expected assets | Import-library-only | Diagnostic |")
      $lines.Add("| --- | --- | ---: | ---: | ---: | --- | --- |")
      foreach ($diagnostic in @($result.fullVendorInputs.vendorRootDiagnostics)) {
        $lines.Add("| $($diagnostic.kind) | $($diagnostic.rootExists) | $($diagnostic.dllCount) | $($diagnostic.libCount) | $($diagnostic.expectedMissingCount) | $($diagnostic.hasImportLibrariesOnly) | $(ConvertTo-MarkdownCell -Value ([string]$diagnostic.diagnostic)) |")
      }
    }

    if (@($result.fullVendorInputs.missingAssets).Count -gt 0) {
      $lines.Add("")
      $lines.Add("| Missing kind | Relative path | Expected path |")
      $lines.Add("| --- | --- | --- |")
      foreach ($asset in @($result.fullVendorInputs.missingAssets)) {
        $lines.Add("| $($asset.kind) | ``$($asset.relativePath)`` | ``$($asset.path)`` |")
      }
    }

    $lines.Add("")
  }

  $lines.Add('Generated by `eng/Test-RuntimePackageReadiness.ps1`.')
  $lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

  Write-Host "Runtime package readiness summary written to $jsonPath"
  Write-Host "Runtime package readiness summary written to $markdownPath"
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

if ([string]::IsNullOrWhiteSpace($SplitPackageRoot)) {
  $SplitPackageRoot = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg"
}
elseif (-not [System.IO.Path]::IsPathRooted($SplitPackageRoot)) {
  $SplitPackageRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $SplitPackageRoot))
}

if ([string]::IsNullOrWhiteSpace($PackageConsumerReportDirectory)) {
  $PackageConsumerReportDirectory = Join-Path $RepositoryRoot "artifacts\package-consumer"
}
elseif (-not [System.IO.Path]::IsPathRooted($PackageConsumerReportDirectory)) {
  $PackageConsumerReportDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $PackageConsumerReportDirectory))
}

if ([string]::IsNullOrWhiteSpace($SplitCollectionConsumerReportDirectory)) {
  $SplitCollectionConsumerReportDirectory = Join-Path $PackageConsumerReportDirectory "split-collection"
}
elseif (-not [System.IO.Path]::IsPathRooted($SplitCollectionConsumerReportDirectory)) {
  $SplitCollectionConsumerReportDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $SplitCollectionConsumerReportDirectory))
}

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\package-readiness"
}
elseif (-not [System.IO.Path]::IsPathRooted($ReportDirectory)) {
  $ReportDirectory = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ReportDirectory))
}

$keys = @(Expand-KeyList -Values $RuntimePackageKey)
if ($keys.Count -eq 0) {
  throw "At least one runtime package key is required."
}

$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifest = if (Test-Path -LiteralPath $splitManifestPath -PathType Leaf) {
  Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
}
else {
  [pscustomobject]@{ packages = @() }
}

$results = New-Object System.Collections.Generic.List[object]
foreach ($key in $keys) {
  $package = $runtimeManifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $package) {
    throw "Runtime package key '$key' was not found."
  }

  $managedEvidence = New-PackageEvidence -Directory $ManagedPackageDirectory -PackageId "JYPPX.TensorRT.CSharp.API"
  $fullRuntimeEvidence = New-PackageEvidence -Directory $RuntimePackageDirectory -PackageId ([string]$package.packageId)

  $splitPackages = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $key })
  $splitPackageDirectory = Join-Path $SplitPackageRoot $key
  $splitCollectionPackageEvidence = New-PackageEvidence -Directory $splitPackageDirectory -PackageId ([string]$package.packageId)
  $splitPackageEvidence = @(
    foreach ($splitPackage in $splitPackages) {
      $evidence = New-PackageEvidence -Directory $splitPackageDirectory -PackageId ([string]$splitPackage.packageId)
      [pscustomobject]@{
        key = [string]$splitPackage.key
        role = [string]$splitPackage.role
        packageId = [string]$splitPackage.packageId
        status = [string]$evidence.status
        version = [string]$evidence.version
        path = [string]$evidence.path
      }
    }
  )

  $bridgeSplitPackage = $splitPackages | Where-Object { $_.role -eq "bridge" } | Select-Object -First 1
  $bridgePackageEvidence = if ($bridgeSplitPackage) {
    New-PackageEvidence -Directory $splitPackageDirectory -PackageId ([string]$bridgeSplitPackage.packageId)
  }
  else {
    [pscustomobject]@{
      status = "missing"
      packageId = ""
      version = ""
      path = ""
      directory = $splitPackageDirectory
    }
  }

  $roots = Resolve-RuntimeRootsForKey -Key $key
  $assetChecks = New-Object System.Collections.Generic.List[object]
  if ([string]$package.platform -eq "windows") {
    foreach ($relativePath in @($package.tensorRtFiles)) {
      $assetChecks.Add((Test-RelativeAsset -BaseRoot ([string]$roots.tensorRtRoot) -RelativePath ([string]$relativePath) -Kind "TensorRT"))
    }
    foreach ($relativePath in @($package.cudaFiles)) {
      $assetChecks.Add((Test-RelativeAsset -BaseRoot ([string]$roots.cudaRoot) -RelativePath ([string]$relativePath) -Kind "CUDA"))
    }
    foreach ($relativePath in @($package.cudnnFiles)) {
      $assetChecks.Add((Test-RelativeAsset -BaseRoot ([string]$roots.cudnnRoot) -RelativePath ([string]$relativePath) -Kind "cuDNN"))
    }
  }

  $missingAssets = @($assetChecks | Where-Object { [string]$_.status -ne "present" })
  $vendorBlockers = New-Object System.Collections.Generic.List[string]
  if ([string]$package.platform -ne "windows") {
    $vendorStatus = "skipped"
    $vendorBlockers.Add("vendor input readiness is currently implemented for Windows runtime packages only.")
  }
  elseif ($roots.PSObject.Properties.Name.Contains("error") -and -not [string]::IsNullOrWhiteSpace([string]$roots.error)) {
    $vendorStatus = "blocked"
    $vendorBlockers.Add([string]$roots.error)
  }
  elseif ($missingAssets.Count -gt 0) {
    $vendorStatus = "blocked"
    $vendorBlockers.Add("$($missingAssets.Count) expected vendor runtime asset(s) were not found.")
  }
  else {
    $vendorStatus = "ready"
  }

  $bridgeConsumerEvidence = Get-BridgeConsumerEvidence -Key $key
  $fullPackageConsumerEvidence = Get-FullPackageConsumerEvidence -Key $key
  $splitCollectionConsumerEvidence = Get-SplitCollectionConsumerEvidence -Key $key -PackageId ([string]$package.packageId)
  $runtimeExecutionEvidence = New-RuntimeExecutionEvidence -SplitCollectionConsumer $splitCollectionConsumerEvidence -FullPackageConsumer $fullPackageConsumerEvidence
  $errorRecorderDiagnosticsDesignGateEvidence = New-ErrorRecorderDiagnosticsDesignGateEvidence
  $dimensionExpressionSnapshotDesignGateEvidence = New-DimensionExpressionSnapshotDesignGateEvidence
  $calibratorMetadataDesignGateEvidence = New-CalibratorMetadataDesignGateEvidence
  $runtimeDeserializationBoundaryPrecheckEvidence = New-RuntimeDeserializationBoundaryPrecheckEvidence
  $runtimeDeserializationDependencyDiagnosticsEvidence = New-RuntimeDeserializationDependencyDiagnosticsEvidence -BoundaryPrecheck $runtimeDeserializationBoundaryPrecheckEvidence -FullPackageConsumer $fullPackageConsumerEvidence
  $allocatorOwnerLedgerDesignGateEvidence = New-AllocatorOwnerLedgerDesignGateEvidence
  $allocatorOwnerInternalRuntimePrototypeEvidence = New-AllocatorOwnerInternalRuntimePrototypeEvidence
  $allocatorOwnerLedgerSafetyGateEvidence = New-AllocatorOwnerLedgerSafetyGateEvidence
  $outputAllocatorInternalRuntimeGateEvidence = New-OutputAllocatorInternalRuntimeGateEvidence
  $outputAllocatorCallbackOwnerDesignEvidence = New-OutputAllocatorCallbackOwnerDesignEvidence
  $outputAllocatorAttachDetachDesignGateEvidence = New-OutputAllocatorAttachDetachDesignGateEvidence
  $outputBufferOwnershipSafetyGateEvidence = New-OutputBufferOwnershipSafetyGateEvidence
  $outputAllocatorRuntimeProofPrecheckEvidence = New-OutputAllocatorRuntimeProofPrecheckEvidence
  $callbackOwnerClosureMatrixEvidence = New-CallbackOwnerClosureMatrixEvidence
  $debugListenerCallbackOwnerDesignEvidence = New-DebugListenerCallbackOwnerDesignEvidence
  $debugListenerAttachDetachDesignGateEvidence = New-DebugListenerAttachDetachDesignGateEvidence
  $debugListenerBorrowedTensorSafetyGateEvidence = New-DebugListenerBorrowedTensorSafetyGateEvidence
  $debugListenerAttachVTableSafetyGateEvidence = New-DebugListenerAttachVTableSafetyGateEvidence
  $debugListenerNativeAttachNoThrowPreflightEvidence = New-DebugListenerNativeAttachNoThrowPreflightEvidence
  $debugListenerNativeOwnerAddressDesignGateEvidence = New-DebugListenerNativeOwnerAddressDesignGateEvidence
  $debugListenerNativeNoThrowVTableDesignGateEvidence = New-DebugListenerNativeNoThrowVTableDesignGateEvidence
  $debugListenerNativeAttachEntryDesignGateEvidence = New-DebugListenerNativeAttachEntryDesignGateEvidence
  $debugListenerNativeDetachBeforeReleaseDesignGateEvidence = New-DebugListenerNativeDetachBeforeReleaseDesignGateEvidence
  $debugListenerNativeOwnerLifecycleDryRunEvidence = New-DebugListenerNativeOwnerLifecycleDryRunEvidence
  $debugListenerNativeAttachEntryRuntimeScaffoldEvidence = New-DebugListenerNativeAttachEntryRuntimeScaffoldEvidence
  $debugListenerNativeAttachEntryMinimalSafetyEvidence = New-DebugListenerNativeAttachEntryMinimalSafetyEvidence
  $debugListenerNativeOwnerStableIdentityEvidence = New-DebugListenerNativeOwnerStableIdentityEvidence
  $debugListenerNativeOwnerNonCopyableStorageEvidence = New-DebugListenerNativeOwnerNonCopyableStorageEvidence
  $debugListenerNativeNoThrowDestructorEvidence = New-DebugListenerNativeNoThrowDestructorEvidence
  $debugListenerNativeOwnerLifecycleGateEvidence = New-DebugListenerNativeOwnerLifecycleGateEvidence
  $debugListenerNativeAttachBridgeShapeGateEvidence = New-DebugListenerNativeAttachBridgeShapeGateEvidence
  $debugListenerExceptionStatusMappingGateEvidence = New-DebugListenerExceptionStatusMappingGateEvidence
  $debugListenerInFlightAccountingGateEvidence = New-DebugListenerInFlightAccountingGateEvidence
  $debugListenerNativeNoThrowVTableScaffoldGateEvidence = New-DebugListenerNativeNoThrowVTableScaffoldGateEvidence
  $debugListenerNoThrowVTableCallbackStubEvidence = New-DebugListenerNoThrowVTableCallbackStubEvidence
  $debugListenerBorrowedDebugTensorMetadataRuntimeGateEvidence = New-DebugListenerBorrowedDebugTensorMetadataRuntimeGateEvidence
  $debugListenerNativeVTableInstallPreflightEvidence = New-DebugListenerNativeVTableInstallPreflightEvidence
  $debugListenerNativeOwnerVTableInstallExperimentEvidence = New-DebugListenerNativeOwnerVTableInstallExperimentEvidence
  $debugListenerRuntimeProofPrecheckEvidence = New-DebugListenerRuntimeProofPrecheckEvidence
  $debugListenerRuntimeProofAttemptPreflightEvidence = New-DebugListenerRuntimeProofAttemptPreflightEvidence
  $debugListenerRealNonNullAttachRuntimeSmokeEvidence = New-DebugListenerRealNonNullAttachRuntimeSmokeEvidence
  $debugListenerProcessDebugTensorCallbackTrampolineEvidence = New-DebugListenerProcessDebugTensorCallbackTrampolineEvidence
  $debugListenerRealCallbackRuntimeProofEvidence = New-DebugListenerRealCallbackRuntimeProofEvidence
  $debugListenerCallbackProofGapReportEvidence = New-DebugListenerCallbackProofGapReportEvidence
  $realCallbackTrampolineGateEvidence = New-RealCallbackTrampolineGateEvidence
  $realCallbackRuntimeEvidenceSchema = New-RealCallbackRuntimeEvidenceSchema
  $realCallbackRuntimeEvidence = New-RealCallbackRuntimeEvidence -FullPackageConsumer $fullPackageConsumerEvidence -Schema $realCallbackRuntimeEvidenceSchema

  $fullRuntimeStatus = if ($fullRuntimeEvidence.status -eq "ready" -and $vendorStatus -eq "ready" -and $fullPackageConsumerEvidence.status -eq "ready") {
    "ready"
  }
  elseif ($vendorStatus -eq "skipped") {
    "skipped"
  }
  else {
    "blocked"
  }

  $splitPackageEvidenceRows = @($splitPackageEvidence)
  $assetCheckRows = @($assetChecks.ToArray())
  $missingAssetRows = @($missingAssets)
  $missingAssetSummaryByKindRows = @(New-MissingAssetSummaryByKind -MissingAssets $missingAssetRows -TensorRtRoot ([string]$roots.tensorRtRoot) -CudaRoot ([string]$roots.cudaRoot) -CudnnRoot ([string]$roots.cudnnRoot))
  $vendorRootDiagnosticRows = @(New-VendorRootDiagnostics -MissingAssets $missingAssetRows -TensorRtRoot ([string]$roots.tensorRtRoot) -CudaRoot ([string]$roots.cudaRoot) -CudnnRoot ([string]$roots.cudnnRoot))
  $vendorBlockerRows = @($vendorBlockers.ToArray())
  $readySplitPackageEvidenceRows = @($splitPackageEvidenceRows | Where-Object { $_.status -eq "ready" })
  $nonReadySplitPackageEvidenceRows = @($splitPackageEvidenceRows | Where-Object { $_.status -ne "ready" })
  $splitReadinessStatus = if ($splitPackageEvidenceRows.Count -gt 0 -and $nonReadySplitPackageEvidenceRows.Count -eq 0) { "ready" } else { "incomplete" }

  $overallStatus = if ($managedEvidence.status -eq "ready" -and $bridgePackageEvidence.status -eq "ready" -and $bridgeConsumerEvidence.status -eq "ready" -and $splitReadinessStatus -eq "ready" -and $splitCollectionPackageEvidence.status -eq "ready" -and $splitCollectionConsumerEvidence.status -eq "ready" -and $fullRuntimeStatus -eq "ready") {
    "ready"
  }
  else {
    "blocked"
  }

  $managedPackageCommand = "dotnet pack .\pack\JYPPX.TensorRT.CSharp.API\JYPPX.TensorRT.CSharp.API.csproj -c Debug -o .\artifacts\managed -p:JYPPXPackageVersion=4.0.0 /p:UseSharedCompilation=false"
  $bridgePackageCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 -SourceRuntimeKey $key -SplitPackageRole bridge"
  $bridgeConsumerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageConsumer.ps1 -SourceRuntimeKey $key -BridgePackageDirectory .\artifacts\runtime-split-nupkg\$key -SkipProbe"
  $splitAllCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 -SourceRuntimeKey $key -SplitPackageRole all"
  $splitCollectionCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 -SourceRuntimeKey $key -SplitPackageRole collection"
  $splitCollectionConsumerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $key -RuntimePackageDirectory .\artifacts\runtime-split-nupkg\$key -ReportDirectory .\artifacts\package-consumer\split-collection"
  $vendorRootsCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 -RuntimePackageKey $key"
  $vendorMaterializeCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Materialize-WindowsVendorRuntimeAssets.ps1 -RuntimePackageKey $key -DryRun"
  $fullRuntimeCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalRuntimePackage.ps1 -RuntimePackageKey $key"
  $fullPackageConsumerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $key"

  $readinessBlockers = New-Object System.Collections.Generic.List[object]
  if ([string]$managedEvidence.status -ne "ready") {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "managed-package" -Status ([string]$managedEvidence.status) -Detail "managed package '$($managedEvidence.packageId)' was not found." -NextAction "Pack the managed package into artifacts\managed before runtime consumer validation." -SuggestedCommand $managedPackageCommand -EvidencePath ([string]$managedEvidence.directory)))
  }

  if ([string]$bridgePackageEvidence.status -ne "ready") {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "bridge-package" -Status ([string]$bridgePackageEvidence.status) -Detail "bridge package '$($bridgePackageEvidence.packageId)' was not found." -NextAction "Build the bridge split package for this runtime key." -SuggestedCommand $bridgePackageCommand -EvidencePath ([string]$bridgePackageEvidence.directory)))
  }

  if ([string]$bridgeConsumerEvidence.status -ne "ready") {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "bridge-consumer" -Status ([string]$bridgeConsumerEvidence.status) -Detail ([string]$bridgeConsumerEvidence.diagnostic) -NextAction "Run bridge-only package consumer validation and refresh the bridge report." -SuggestedCommand $bridgeConsumerCommand -EvidencePath ([string]$bridgeConsumerEvidence.reportPath)))
  }

  if ($splitPackageEvidenceRows.Count -eq 0) {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "split-package" -Status "missing" -Detail "no split packages were defined for runtime key '$key'." -NextAction "Add split package manifest rows for this runtime key before packaging." -EvidencePath $splitManifestPath))
  }
  else {
    foreach ($splitPackage in @($nonReadySplitPackageEvidenceRows)) {
      $splitRoleCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 -SourceRuntimeKey $key -SplitPackageRole $($splitPackage.role)"
      $splitPackageNextAction = "Build or download the missing split component package, then rerun readiness."
      $splitPackageSuggestedCommand = $splitRoleCommand
      $splitPackageExternalInputRequired = [string]$splitPackage.role -ne "bridge"
      $splitPackageDetail = "split role '$($splitPackage.role)' package '$($splitPackage.packageId)' was not found."
      if ([string]$vendorStatus -eq "blocked" -and [string]$splitPackage.role -ne "bridge") {
        $splitPackageNextAction = "Resolve vendor input blockers first; this split component is collected from the full runtime staging assets and will fail until TensorRT/CUDA/cuDNN runtime files are present."
        $splitPackageSuggestedCommand = $vendorMaterializeCommand
        $splitPackageExternalInputRequired = $true
        $splitPackageDetail = "$splitPackageDetail Vendor inputs are blocked, so the split package cannot be built from complete runtime staging yet."
      }

      $readinessBlockers.Add((New-ReadinessBlocker -Category "split-package" -Status ([string]$splitPackage.status) -Detail $splitPackageDetail -NextAction $splitPackageNextAction -SuggestedCommand $splitPackageSuggestedCommand -IsExternalInputRequired $splitPackageExternalInputRequired -EvidencePath $splitPackageDirectory))
    }
  }

  if ([string]$splitCollectionPackageEvidence.status -ne "ready") {
    $splitCollectionAction = if ($nonReadySplitPackageEvidenceRows.Count -gt 0) { "Complete split component packages first, then build the lightweight split collection package." } else { "Build the lightweight split collection package that references the ready split components." }
    $splitCollectionSuggestedCommand = if ($nonReadySplitPackageEvidenceRows.Count -gt 0) { $splitAllCommand } else { $splitCollectionCommand }
    $splitCollectionExternalInputRequired = $false
    if ([string]$vendorStatus -eq "blocked" -and $nonReadySplitPackageEvidenceRows.Count -gt 0) {
      $splitCollectionAction = "Resolve vendor input blockers first; the split collection cannot reference a complete component set until non-bridge split packages exist."
      $splitCollectionSuggestedCommand = $vendorMaterializeCommand
      $splitCollectionExternalInputRequired = $true
    }
    $readinessBlockers.Add((New-ReadinessBlocker -Category "split-collection-package" -Status ([string]$splitCollectionPackageEvidence.status) -Detail "split collection package '$($splitCollectionPackageEvidence.packageId)' was not found." -NextAction $splitCollectionAction -SuggestedCommand $splitCollectionSuggestedCommand -IsExternalInputRequired $splitCollectionExternalInputRequired -EvidencePath ([string]$splitCollectionPackageEvidence.directory)))
  }

  if ([string]$splitCollectionConsumerEvidence.status -ne "ready") {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "split-collection-consumer" -Status ([string]$splitCollectionConsumerEvidence.status) -Detail ([string]$splitCollectionConsumerEvidence.diagnostic) -NextAction "Run package consumer validation against the split collection package source." -SuggestedCommand $splitCollectionConsumerCommand -EvidencePath ([string]$splitCollectionConsumerEvidence.reportPath)))
  }

  if ([string]$vendorStatus -ne "ready") {
    foreach ($blocker in @($vendorBlockerRows)) {
      $readinessBlockers.Add((New-ReadinessBlocker -Category "vendor-inputs" -Status ([string]$vendorStatus) -Detail ([string]$blocker) -NextAction "Populate the TensorRT/CUDA/cuDNN vendor roots reported by Resolve-RuntimeRoots, then rerun packaging/readiness. Use Materialize-WindowsVendorRuntimeAssets.ps1 when local NVIDIA archives are available." -SuggestedCommand $vendorMaterializeCommand -IsExternalInputRequired $true -EvidencePath $runtimeManifestPath))
    }
  }

  if ([string]$fullRuntimeEvidence.status -ne "ready") {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "full-runtime-package" -Status ([string]$fullRuntimeEvidence.status) -Detail "full runtime package '$($fullRuntimeEvidence.packageId)' was not found." -NextAction "Build the full runtime package after vendor inputs are complete." -SuggestedCommand $fullRuntimeCommand -IsExternalInputRequired ([string]$vendorStatus -ne "ready") -EvidencePath ([string]$fullRuntimeEvidence.directory)))
  }

  if ([string]$fullPackageConsumerEvidence.status -ne "ready") {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "full-package-consumer" -Status ([string]$fullPackageConsumerEvidence.status) -Detail ([string]$fullPackageConsumerEvidence.diagnostic) -NextAction "Run full runtime package consumer validation and refresh the consumer report." -SuggestedCommand $fullPackageConsumerCommand -IsExternalInputRequired ([string]$fullRuntimeStatus -ne "ready") -EvidencePath ([string]$fullPackageConsumerEvidence.reportPath)))
  }

  if ([string]$callbackOwnerClosureMatrixEvidence.status -ne "closure-matrix-ready" -or [bool]$callbackOwnerClosureMatrixEvidence.isRealCallbackRuntimeProof) {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "callback-owner-closure-matrix" -Status ([string]$callbackOwnerClosureMatrixEvidence.status) -Detail "callback owner closure matrix must be complete, classified as closure-matrix, and remain non-proof before release readiness can be trusted." -NextAction "Refresh callback closure matrix source/smoke/docs/readiness evidence; do not promote closure-matrix evidence to real-callback-runtime." -SuggestedCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1" -EvidencePath "docs\articles\zh-cn\callback-owner-closure-matrix.md"))
  }

  if ([string]$debugListenerCallbackProofGapReportEvidence.status -ne "proof-gap-report-ready" -or [bool]$debugListenerCallbackProofGapReportEvidence.isRealCallbackRuntimeProof -or [string]$debugListenerCallbackProofGapReportEvidence.runtimeEvidenceKind -ne "proof-gap-report") {
    $readinessBlockers.Add((New-ReadinessBlocker -Category "debug-listener-callback-proof-gap-report" -Status ([string]$debugListenerCallbackProofGapReportEvidence.status) -Detail "DebugListener callback proof gap report must be complete, machine-readable, classified as proof-gap-report, and non-proof until full package consumer invocation evidence exists." -NextAction "Refresh DebugListener callback proof gap report source/smoke/docs/package/readiness markers, then rerun package/readiness validation." -SuggestedCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RuntimePackageReadiness.ps1" -EvidencePath "src\JYPPX.TensorRtSharp\TensorRtDebugListenerCallbackProofGapReport.cs"))
  }

  $readinessBlockerRows = @($readinessBlockers.ToArray())
  if ($readinessBlockerRows.Count -gt 0) {
    $overallStatus = "blocked"
  }

  $runtimeProofStatus = [string]$runtimeExecutionEvidence.status
  $runtimeProofDiagnostic = [string]$runtimeExecutionEvidence.diagnostic
  $runtimeProofRequiredForRelease = [string]$runtimeProofStatus -ne "ready"
  $runtimeProofBlockerOwnerAction = New-RuntimeProofBlockerOwnerAction -Key $key -RuntimeExecutionEvidence $runtimeExecutionEvidence -FullPackageConsumer $fullPackageConsumerEvidence -RuntimeProofStatus $runtimeProofStatus -RuntimeProofRequiredForRelease $runtimeProofRequiredForRelease -RuntimeProofDiagnostic $runtimeProofDiagnostic

  $result = [pscustomobject]@{
    key = [string]$package.key
    packageId = [string]$package.packageId
    rid = [string]$package.rid
    platform = [string]$package.platform
    distributionTier = [string]$package.distributionTier
    validationState = [string]$package.validationState
    managedPackage = $managedEvidence
    bridgePackage = $bridgePackageEvidence
    bridgeConsumer = $bridgeConsumerEvidence
    splitPackages = [pscustomobject]@{
      status = $splitReadinessStatus
      expectedCount = $splitPackages.Count
      foundCount = $readySplitPackageEvidenceRows.Count
      packages = @($splitPackageEvidenceRows)
    }
    splitCollectionPackage = $splitCollectionPackageEvidence
    splitCollectionConsumer = $splitCollectionConsumerEvidence
    fullVendorInputs = [pscustomobject]@{
      status = $vendorStatus
      tensorRtRoot = [string]$roots.tensorRtRoot
      cudaRoot = [string]$roots.cudaRoot
      cudnnRoot = [string]$roots.cudnnRoot
      checkedAssets = @($assetCheckRows)
      missingAssets = @($missingAssetRows)
      missingAssetSummaryByKind = @($missingAssetSummaryByKindRows)
      vendorRootDiagnostics = @($vendorRootDiagnosticRows)
      blockers = @($vendorBlockerRows)
    }
    fullRuntimePackage = [pscustomobject]@{
      status = if ($fullRuntimeEvidence.status -eq "ready") { "ready" } else { "missing" }
      packageId = [string]$fullRuntimeEvidence.packageId
      version = [string]$fullRuntimeEvidence.version
      path = [string]$fullRuntimeEvidence.path
      directory = [string]$fullRuntimeEvidence.directory
    }
    fullPackageConsumer = $fullPackageConsumerEvidence
    packageConsumerEvidenceKind = [string]$fullPackageConsumerEvidence.packageConsumerEvidenceKind
    runtimeSmokeClassification = [string]$fullPackageConsumerEvidence.runtimeSmokeClassification
    isRuntimeExecutionEvidence = [bool]$fullPackageConsumerEvidence.isRuntimeExecutionEvidence
    isDependencyProbeOnly = [bool]$fullPackageConsumerEvidence.isDependencyProbeOnly
    isRealCallbackRuntimeProof = [bool]$fullPackageConsumerEvidence.isRealCallbackRuntimeProof
    runtimeProofStatus = $runtimeProofStatus
    runtimeProofDiagnostic = $runtimeProofDiagnostic
    runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
    runtimeExecution = $runtimeExecutionEvidence
    runtimeProofBlockerOwnerAction = $runtimeProofBlockerOwnerAction
    errorRecorderDiagnosticsDesignGate = $errorRecorderDiagnosticsDesignGateEvidence
    dimensionExpressionSnapshotDesignGate = $dimensionExpressionSnapshotDesignGateEvidence
    calibratorMetadataDesignGate = $calibratorMetadataDesignGateEvidence
    runtimeDeserializationBoundaryPrecheck = $runtimeDeserializationBoundaryPrecheckEvidence
    runtimeDeserializationDependencyDiagnostics = $runtimeDeserializationDependencyDiagnosticsEvidence
    allocatorOwnerLedgerDesignGate = $allocatorOwnerLedgerDesignGateEvidence
    allocatorOwnerInternalRuntimePrototype = $allocatorOwnerInternalRuntimePrototypeEvidence
    allocatorOwnerLedgerSafetyGate = $allocatorOwnerLedgerSafetyGateEvidence
    outputAllocatorInternalRuntimeGate = $outputAllocatorInternalRuntimeGateEvidence
    outputAllocatorCallbackOwnerDesign = $outputAllocatorCallbackOwnerDesignEvidence
    outputAllocatorAttachDetachDesignGate = $outputAllocatorAttachDetachDesignGateEvidence
    outputBufferOwnershipSafetyGate = $outputBufferOwnershipSafetyGateEvidence
    outputAllocatorRuntimeProofPrecheck = $outputAllocatorRuntimeProofPrecheckEvidence
    callbackOwnerClosureMatrix = $callbackOwnerClosureMatrixEvidence
    debugListenerCallbackOwnerDesign = $debugListenerCallbackOwnerDesignEvidence
    debugListenerAttachDetachDesignGate = $debugListenerAttachDetachDesignGateEvidence
    debugListenerBorrowedTensorSafetyGate = $debugListenerBorrowedTensorSafetyGateEvidence
    debugListenerAttachVTableSafetyGate = $debugListenerAttachVTableSafetyGateEvidence
    debugListenerNativeAttachNoThrowPreflight = $debugListenerNativeAttachNoThrowPreflightEvidence
    debugListenerNativeOwnerAddressDesignGate = $debugListenerNativeOwnerAddressDesignGateEvidence
    debugListenerNativeNoThrowVTableDesignGate = $debugListenerNativeNoThrowVTableDesignGateEvidence
    debugListenerNativeAttachEntryDesignGate = $debugListenerNativeAttachEntryDesignGateEvidence
    debugListenerNativeDetachBeforeReleaseDesignGate = $debugListenerNativeDetachBeforeReleaseDesignGateEvidence
    debugListenerNativeOwnerLifecycleDryRun = $debugListenerNativeOwnerLifecycleDryRunEvidence
    debugListenerNativeAttachEntryRuntimeScaffold = $debugListenerNativeAttachEntryRuntimeScaffoldEvidence
    debugListenerNativeAttachEntryMinimalSafety = $debugListenerNativeAttachEntryMinimalSafetyEvidence
    debugListenerNativeOwnerStableIdentity = $debugListenerNativeOwnerStableIdentityEvidence
    debugListenerNativeOwnerNonCopyableStorage = $debugListenerNativeOwnerNonCopyableStorageEvidence
    debugListenerNativeNoThrowDestructor = $debugListenerNativeNoThrowDestructorEvidence
    debugListenerNativeOwnerLifecycleGate = $debugListenerNativeOwnerLifecycleGateEvidence
    debugListenerNativeAttachBridgeShapeGate = $debugListenerNativeAttachBridgeShapeGateEvidence
    debugListenerExceptionStatusMappingGate = $debugListenerExceptionStatusMappingGateEvidence
    debugListenerInFlightAccountingGate = $debugListenerInFlightAccountingGateEvidence
    debugListenerNativeNoThrowVTableScaffoldGate = $debugListenerNativeNoThrowVTableScaffoldGateEvidence
    debugListenerNoThrowVTableCallbackStub = $debugListenerNoThrowVTableCallbackStubEvidence
    debugListenerBorrowedDebugTensorMetadataRuntimeGate = $debugListenerBorrowedDebugTensorMetadataRuntimeGateEvidence
    debugListenerNativeVTableInstallPreflight = $debugListenerNativeVTableInstallPreflightEvidence
    debugListenerNativeOwnerVTableInstallExperiment = $debugListenerNativeOwnerVTableInstallExperimentEvidence
    debugListenerRuntimeProofPrecheck = $debugListenerRuntimeProofPrecheckEvidence
    debugListenerRuntimeProofAttemptPreflight = $debugListenerRuntimeProofAttemptPreflightEvidence
    debugListenerRealNonNullAttachRuntimeSmoke = $debugListenerRealNonNullAttachRuntimeSmokeEvidence
    debugListenerProcessDebugTensorCallbackTrampoline = $debugListenerProcessDebugTensorCallbackTrampolineEvidence
    debugListenerRealCallbackRuntimeProof = $debugListenerRealCallbackRuntimeProofEvidence
    debugListenerCallbackProofGapReport = $debugListenerCallbackProofGapReportEvidence
    realCallbackTrampolineGate = $realCallbackTrampolineGateEvidence
    realCallbackRuntimeEvidenceSchema = $realCallbackRuntimeEvidenceSchema
    realCallbackRuntimeEvidence = $realCallbackRuntimeEvidence
    readinessBlockers = @($readinessBlockerRows)
    fullRuntimeStatus = $fullRuntimeStatus
    overallStatus = $overallStatus
  }

  $results.Add($result)
}

Write-ReadinessReports -Results @($results.ToArray())

$blockedResults = @($results | Where-Object { [string]$_.overallStatus -ne "ready" })
if ($FailOnBlocked.IsPresent -and $blockedResults.Count -gt 0) {
  foreach ($result in $blockedResults) {
    Write-Error "$($result.key): overallStatus=$($result.overallStatus) fullRuntimeStatus=$($result.fullRuntimeStatus)"
  }
  exit 1
}
