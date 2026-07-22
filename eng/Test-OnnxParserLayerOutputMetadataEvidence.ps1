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
  $InputPath = Join-Path $RepositoryRoot "artifacts/interface-coverage/onnx-parser-layer-output-metadata-runtime-evidence.json"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "artifacts/interface-coverage/onnx-parser-layer-output-metadata-runtime-evidence-validation.json"
}

$evidence = Get-Content -LiteralPath $InputPath -Raw | ConvertFrom-Json
$checks = [System.Collections.Generic.List[object]]::new()
function Add-Check([string]$Name, [bool]$Passed, [string]$Detail) {
  $checks.Add([ordered]@{ name = $Name; passed = $Passed; detail = $Detail }) | Out-Null
}

Add-Check "schema" ($evidence.schemaVersion -eq "onnx-parser-layer-output-metadata-runtime-evidence.v1") ([string]$evidence.schemaVersion)
Add-Check "inventory" ($evidence.candidateInventory.tensorRtDeferredRows -eq 598 -and $evidence.candidateInventory.lowRiskRows -eq 0) "598 deferred rows and no low-risk rows"
Add-Check "vendor-version-guards" ($evidence.vendorAbi.tensorRt8HeaderMethod -eq "absent" -and $evidence.vendorAbi.tensorRt10HeaderMethod -eq "present-pure-virtual" -and $evidence.vendorAbi.tensorRt11HeaderMethod -eq "present-pure-virtual") "TRT8 absent; TRT10/11 vtable method present"
Add-Check "bindings" ($evidence.bindings.manifestCount -eq 196 -and $evidence.bindings.apiRecordCount -eq 3973 -and $evidence.bindings.outputValidationPassed) "196 manifests / 3973 records"
Add-Check "abi-parity" ($evidence.nativeAbi.tensorRt8.missing -eq 0 -and $evidence.nativeAbi.tensorRt10.missing -eq 0 -and $evidence.nativeAbi.tensorRt11.missing -eq 0) "TRT8/10/11 missing ABI/export count is zero"
Add-Check "trt8-guard" ($evidence.tensorRt8.state -eq "parser-dependency-controlled-skip" -and $evidence.tensorRt8.exitCode -eq 0 -and $evidence.tensorRt8.controlledSkip -and $evidence.tensorRt8.parserConstructionAttempted -and -not $evidence.tensorRt8.metadataQueryAttempted -and -not $evidence.tensorRt8.nativeEntrypointPresent) "TRT8 parser dependency is a controlled skip and the unavailable API is not dispatched"
Add-Check "trt10-runtime" ($evidence.tensorRt10.exitCode -eq 0 -and $evidence.tensorRt10.state -eq "copied-metadata-runtime-passed" -and $evidence.tensorRt10.parsed) "TRT10 parser metadata smoke passed"
Add-Check "trt10-identity-metadata" ($evidence.tensorRt10.tensorName -eq "output" -and @($evidence.tensorRt10.shape).Count -eq 2 -and $evidence.tensorRt10.shape[0] -eq -1 -and $evidence.tensorRt10.shape[1] -eq 4 -and $evidence.tensorRt10.dataType -eq "Float") "output Float [-1,4]"
Add-Check "trt10-copy-semantics" ($evidence.tensorRt10.tryGetFound -and $evidence.tensorRt10.getMatchesTryGet -and -not $evidence.tensorRt10.missingLayerFound -and $evidence.tensorRt10.pointerFreeCopiedMetadata -and -not $evidence.tensorRt10.retainsNativeTensor) "Try/Get match; missing false; pointer-free"
Add-Check "trt10-enqueue" ($evidence.tensorRt10.enqueueSucceeded -and $evidence.tensorRt10.outputMatch) "Enclosing identity enqueue/output match passed"
Add-Check "trt11-probe-boundary" ($evidence.tensorRt11.state -eq "dependency-runtime-probe-only" -and $evidence.tensorRt11.controlledSkip -and -not $evidence.tensorRt11.metadataQueryAttempted) "TRT11 metadata not claimed after owner creation failed"
Add-Check "package-consumers" (@($evidence.packageConsumer.lines).Count -eq 3 -and @($evidence.packageConsumer.lines | Where-Object { -not $_.restoreSucceeded -or -not $_.buildSucceeded }).Count -eq 0 -and -not $evidence.packageConsumer.usesProjectReference) "Three no-ProjectReference restore/build validations"
Add-Check "package-classification" ($evidence.packageConsumer.wrapperMarker -eq "onnx-parser-layer-output-copied-metadata" -and $evidence.packageConsumer.wrapperSurfaceEvidenceKind -eq "compile-surface-proof" -and -not $evidence.packageConsumer.isRuntimeExecutionProof) "Compile-surface classification retained"
Add-Check "deferred-history" ($evidence.deferredHistoryRetained -and -not $evidence.canDeleteDeferredRecord) "Deferred history retained"
Add-Check "non-publish-boundary" (-not $evidence.isRuntimeExecutionProof -and -not $evidence.isPackageConsumerRuntimeProof -and -not $evidence.isPostPublishProof -and -not $evidence.canPublishPublicly -and -not $evidence.canCloseReleaseIssue -and -not $evidence.performsPublish) "No runtime/release/publication promotion"

$failures = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "onnx-parser-layer-output-metadata-runtime-evidence-validation.v1"
  validationState = if ($failures.Count -eq 0) { "passed" } else { "failed" }
  checkCount = $checks.Count
  failureCount = $failures.Count
  checks = @($checks)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
}

$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output "OnnxParserLayerOutputMetadataEvidenceState=$($result.validationState) Checks=$($checks.Count) Failures=$($failures.Count)"
if ($Strict -and $failures.Count -ne 0) {
  throw "ONNX parser layer output metadata evidence validation failed: $(@($failures.name) -join ', ')"
}
