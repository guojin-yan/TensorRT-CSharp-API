[CmdletBinding()]
param(
  [string]$RepositoryRoot = "",
  [string]$InputPath = "",
  [string]$OutputPath = "",
  [switch]$Strict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
if ([string]::IsNullOrWhiteSpace($InputPath)) {
  $InputPath = Join-Path $RepositoryRoot "artifacts/interface-coverage/trt8-legacy-parser-readonly-runtime-evidence.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "artifacts/interface-coverage/trt8-legacy-parser-readonly-runtime-evidence-validation.json"
}

$evidence = Get-Content -LiteralPath $InputPath -Raw | ConvertFrom-Json
$checks = [System.Collections.Generic.List[object]]::new()
function Add-Check([string]$Name, [bool]$Passed, [string]$Detail) {
  $checks.Add([ordered]@{ name = $Name; passed = $Passed; detail = $Detail }) | Out-Null
}

Add-Check "schema" ($evidence.schemaVersion -eq "trt8-legacy-parser-readonly-runtime-evidence.v1") ([string]$evidence.schemaVersion)
Add-Check "vendor-assets" ($evidence.vendorAbi.nvparsersImportLibraryPresent -and $evidence.vendorAbi.nvparsersRuntimeLibraryPresent -and $evidence.vendorAbi.createUffParserSymbolPresent -and $evidence.vendorAbi.createCaffeParserSymbolPresent) "nvparsers LIB/DLL and both factories are present"
Add-Check "vendor-dependencies" (@($evidence.vendorAbi.runtimeDependencies).Count -eq 2 -and $evidence.vendorAbi.runtimeDependencies[0] -eq "nvinfer.dll" -and $evidence.vendorAbi.runtimeDependencies[1] -eq "KERNEL32.dll") "nvparsers dependency set is bounded"
Add-Check "bindings" ($evidence.bindings.manifestCount -eq 197 -and $evidence.bindings.apiRecordCount -eq 3975 -and $evidence.bindings.generatedTwiceIdempotent -and $evidence.bindings.outputValidationPassed) "197 manifests / 3975 records"
Add-Check "coverage-inventory" ($evidence.coverage.tensorRt8InterfacesScannedPerPackage -eq 880 -and $evidence.coverage.implementedPerPackage -eq 760 -and $evidence.coverage.deferredOnlyPerPackage -eq 120) "TRT8 880 scanned, 760 implemented, 120 deferred-only"
Add-Check "coverage-history" ($evidence.coverage.promotedRows -eq 7 -and $evidence.coverage.promotedState -eq "implemented-with-deferred-history" -and $evidence.coverage.neighboringParseAndDestroyRowsRemainDeferredOnly -and $evidence.deferredHistoryRetained) "Seven safe rows retain deferred history"
Add-Check "abi-parity" ($evidence.nativeAbi.tensorRt8.missing -eq 0 -and $evidence.nativeAbi.tensorRt10.missing -eq 0 -and $evidence.nativeAbi.tensorRt11.missing -eq 0) "TRT8/10/11 ABI/export missing counts are zero"
Add-Check "trt8-runtime" ($evidence.tensorRt8.state -eq "copied-readonly-runtime-passed" -and $evidence.tensorRt8.exitCode -eq 0 -and $evidence.tensorRt8.bridgeInitialized -and $evidence.tensorRt8.uffRequiredVersion -eq "0.6.9") "TRT8 runtime and UFF version passed"
Add-Check "binaryproto-metadata" (@($evidence.tensorRt8.binaryProto.shape).Count -eq 4 -and $evidence.tensorRt8.binaryProto.shape[0] -eq 1 -and $evidence.tensorRt8.binaryProto.shape[1] -eq 1 -and $evidence.tensorRt8.binaryProto.shape[2] -eq 28 -and $evidence.tensorRt8.binaryProto.shape[3] -eq 28 -and $evidence.tensorRt8.binaryProto.dataType -eq "Float" -and $evidence.tensorRt8.binaryProto.dataLength -eq 3136) "Float [1,1,28,28], 3136 copied bytes"
Add-Check "binaryproto-hashes" ($evidence.tensorRt8.binaryProto.sourceSha256 -eq "337CF38DD3A69F25BA7E732D25CA3176576CA8F208B0A916FA3AC2A669A894BE" -and $evidence.tensorRt8.binaryProto.copiedDataSha256 -eq "DF7D560B482098FAC1C6122C22BD0A54499ED9F8EC3AC6BAE8FC917D3A01774A") "Container and copied payload hashes match"
Add-Check "copy-semantics" ($evidence.tensorRt8.binaryProto.independentManagedCopies -and $evidence.tensorRt8.binaryProto.pointerFreeCopiedData -and -not $evidence.tensorRt8.binaryProto.retainsNativeObject -and $evidence.tensorRt8.pointerFreeUffMetadata -and -not $evidence.tensorRt8.retainsUffParser) "Copies are independent and pointer-free"
Add-Check "protobuf-shutdown" (-not $evidence.tensorRt8.callsProcessGlobalProtobufShutdown) "Process-global protobuf shutdown is not called"
Add-Check "version-guard" ($evidence.nonTensorRt8GuardPassed) "TRT10/11 managed guard passed"
Add-Check "package-consumers" (@($evidence.packageConsumer.lines).Count -eq 3 -and @($evidence.packageConsumer.lines | Where-Object { -not $_.restoreSucceeded -or -not $_.buildSucceeded }).Count -eq 0 -and -not $evidence.packageConsumer.usesProjectReference) "Three no-ProjectReference restore/build validations"
Add-Check "package-classification" ($evidence.packageConsumer.wrapperSurfaceEvidenceKind -eq "compile-surface-proof" -and -not $evidence.packageConsumer.isRuntimeExecutionProof -and @($evidence.packageConsumer.compiledSurfaceMarkers) -contains "TensorRtLegacyParserDiagnostics") "Compile-surface classification retained"
Add-Check "non-publish-boundary" (-not $evidence.publicNativePointerAdded -and -not $evidence.isPackageConsumerRuntimeProof -and -not $evidence.isPostPublishProof -and -not $evidence.canPublishPublicly -and -not $evidence.canCloseReleaseIssue -and -not $evidence.performsPublish) "No public pointer, runtime-proof promotion, or publication"

$failures = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "trt8-legacy-parser-readonly-runtime-evidence-validation.v1"
  validationState = if ($failures.Count -eq 0) { "passed" } else { "failed" }
  checkCount = $checks.Count
  failureCount = $failures.Count
  checks = @($checks)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
}

$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output "LegacyParserReadonlyDiagnosticsEvidenceState=$($result.validationState) Checks=$($checks.Count) Failures=$($failures.Count)"
if ($Strict -and $failures.Count -ne 0) {
  throw "Legacy parser readonly diagnostics evidence validation failed: $(@($failures.name) -join ', ')"
}
