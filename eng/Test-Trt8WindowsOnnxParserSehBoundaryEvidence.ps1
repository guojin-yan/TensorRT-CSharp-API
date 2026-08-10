[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/interface-coverage/trt8-windows-onnx-parser-seh-boundary-evidence.json",
  [string]$OutputPath = "artifacts/interface-coverage/trt8-windows-onnx-parser-seh-boundary-validation.json",
  [switch]$Strict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
  return [IO.Path]::GetFullPath((Join-Path $repositoryRoot ($Path -replace "/", "\")))
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath (Resolve-RepositoryPath $Path) -Algorithm SHA256).Hash.ToLowerInvariant()
}

$record = Get-Content -LiteralPath (Resolve-RepositoryPath $InputPath) -Raw | ConvertFrom-Json
$checks = [Collections.Generic.List[object]]::new()

function Add-Check {
  param([string]$Name, [bool]$Passed, [string]$Detail)
  $checks.Add([pscustomobject][ordered]@{ name = $Name; passed = $Passed; detail = $Detail }) | Out-Null
}

function Add-ArtifactCheck {
  param([string]$Name, [object]$Artifact)
  $path = Resolve-RepositoryPath $Artifact.path
  $exists = Test-Path -LiteralPath $path -PathType Leaf
  $length = if ($exists) { (Get-Item -LiteralPath $path).Length } else { -1 }
  $sha256 = if ($exists) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
  Add-Check $Name ($exists -and $length -eq [long]$Artifact.lengthBytes -and $sha256 -eq $Artifact.sha256) "$($Artifact.path)/$length/$sha256"
}

Add-Check "schema" ($record.schemaVersion -eq "trt8-windows-onnx-parser-seh-boundary-evidence.v1") ([string]$record.schemaVersion)
Add-ArtifactCheck "source" $record.sourceBoundary
Add-ArtifactCheck "bridge" $record.nativeBuild.bridge
Add-ArtifactCheck "native-build-stdout" $record.nativeBuild.stdout
Add-ArtifactCheck "native-build-stderr" $record.nativeBuild.stderr
Add-ArtifactCheck "focused-test-trx" $record.focusedContractTest.trx
Add-ArtifactCheck "runtime-command" $record.runtimeProbe.command
Add-ArtifactCheck "runtime-stdout" $record.runtimeProbe.stdout
Add-ArtifactCheck "runtime-stderr" $record.runtimeProbe.stderr
Add-ArtifactCheck "runtime-report" $record.runtimeProbe.report
Add-ArtifactCheck "runtime-strict-validation" $record.runtimeProbe.strictValidation
Add-ArtifactCheck "runtime-engine" $record.runtimeProbe.engine
Add-ArtifactCheck "runtime-output" $record.runtimeProbe.output
Add-ArtifactCheck "model" $record.inputs.model
Add-ArtifactCheck "input" $record.inputs.input
Add-ArtifactCheck "reference" $record.inputs.reference

$source = Get-Content -LiteralPath (Resolve-RepositoryPath $record.sourceBoundary.path) -Raw
Add-Check "native-seh-boundary" (
  [bool]$record.sourceBoundary.sehConvertedToStatus -and
  [bool]$record.sourceBoundary.cppExceptionConvertedToStatus -and
  [bool]$record.sourceBoundary.outParserClearedOnFailure -and
  [bool]$record.sourceBoundary.ownerHandleCreatedOnlyAfterSuccessfulVendorReturn -and
  -not [bool]$record.sourceBoundary.borrowedPointerEscaped -and
  $source.Contains("create_onnx_parser_with_seh_guard") -and
  $source.Contains("capture_vendor_seh_exception_code") -and
  $source.Contains("create_onnx_parser_with_guard") -and
  -not $source.Contains("ONNX parser creation is disabled on Windows")) "SEH/C++ exceptions convert to status before owner creation"
Add-Check "native-build" ([int]$record.nativeBuild.exitCode -eq 0) "$($record.nativeBuild.preset)/exit=$($record.nativeBuild.exitCode)"
Add-Check "focused-contract-test" ([bool]$record.focusedContractTest.passed -and [int]$record.focusedContractTest.total -eq 1 -and [int]$record.focusedContractTest.failed -eq 0) "total=$($record.focusedContractTest.total)/failed=$($record.focusedContractTest.failed)"

$report = Get-Content -LiteralPath (Resolve-RepositoryPath $record.runtimeProbe.report.path) -Raw | ConvertFrom-Json
Add-Check "child-process-probe" ($record.runtimeProbe.processIsolation -eq "dedicated-TensorRtExec-child-process" -and [int]$record.runtimeProbe.exitCode -eq 0) "$($record.runtimeProbe.processIsolation)/exit=$($record.runtimeProbe.exitCode)"
Add-Check "parse-build" (
  $report.State -eq "external-onnx-reference-validated-runtime" -and
  [bool]$report.Success -and [bool]$report.Parsed -and [bool]$report.EngineSaved -and [bool]$report.EngineFileRoundTrip -and
  [bool]$report.ParserPreflightSnapshot.ParseSucceeded -and [int]$report.ParserPreflightSnapshot.ErrorCount -eq 0) "$($report.State)/parsed=$($report.Parsed)/parserErrors=$($report.ParserPreflightSnapshot.ErrorCount)"
Add-Check "enqueue-reference" (
  [bool]$report.InferenceRan -and [bool]$report.OutputMatch -and [bool]$report.OutputValidated -and
  $report.ProofClassification -eq "synthetic-input-runtime") "inference=$($report.InferenceRan)/validated=$($report.OutputValidated)/proof=$($report.ProofClassification)"
Add-Check "engine-readonly-roundtrip" ([bool]$report.LoadedEngineDiagnostics.Succeeded -and $report.LoadedEngineDiagnostics.DiagnosticsState -eq "readonly-deserialize-succeeded") "$($report.LoadedEngineDiagnostics.DiagnosticsState)"

$output = Get-Content -LiteralPath (Resolve-RepositoryPath $record.runtimeProbe.output.path) -Raw | ConvertFrom-Json
$comparison = @($output.ReferenceValidation.TensorComparisons)[0]
$capturedOutput = @($output.OutputTensors)[0]
Add-Check "output-hash" ([int]$capturedOutput.ElementCount -eq 10 -and [int]$capturedOutput.ByteLength -eq 40 -and $capturedOutput.Sha256 -eq $record.runtimeProbe.outputSha256) "$($capturedOutput.ElementCount)/$($capturedOutput.ByteLength)/$($capturedOutput.Sha256)"
Add-Check "reference-comparison" ([bool]$output.ReferenceValidation.Completed -and [bool]$output.ReferenceValidation.Passed -and [int]$comparison.MismatchCount -eq 0) "mismatch=$($comparison.MismatchCount)/maxAbs=$($comparison.MaximumAbsoluteError)/maxRel=$($comparison.MaximumRelativeError)"

$strictValidation = Get-Content -LiteralPath (Resolve-RepositoryPath $record.runtimeProbe.strictValidation.path) -Raw | ConvertFrom-Json
Add-Check "strict-report-validation" ($strictValidation.validationState -eq "tensor-rt-exec-report-ready" -and @($strictValidation.validationItems).Count -eq 69 -and @($strictValidation.validationItems | Where-Object { -not $_.passed }).Count -eq 0) "$($strictValidation.validationState)/$(@($strictValidation.validationItems).Count)/failed=$(@($strictValidation.validationItems | Where-Object { -not $_.passed }).Count)"

$boundary = $record.proofBoundary
Add-Check "proof-boundary" (
  [bool]$boundary.isParserCreationSehBoundaryProof -and [bool]$boundary.isOnnxParseBuildEnqueueEvidence -and [bool]$boundary.isReferenceRegressionEvidence -and
  -not [bool]$boundary.isModelAccuracyProof -and -not [bool]$boundary.isPackageConsumerRuntimeProof -and
  -not [bool]$boundary.isPublicFeedProof -and -not [bool]$boundary.isPostPublishProof -and
  -not [bool]$boundary.isReleaseProof -and -not [bool]$boundary.canPublishPublicly -and
  -not [bool]$boundary.publicReleaseSideEffectsExecuted) "synthetic runtime is not model/package/release proof"
Add-Check "historical-boundary-retained" ([bool]$record.historicalBoundary.historicalEvidenceRetained -and $record.historicalBoundary.previousState -eq "dependency-probe-only") "$($record.historicalBoundary.previousState)"

$failures = @($checks | Where-Object { -not $_.passed })
$validation = [pscustomobject][ordered]@{
  schemaVersion = "trt8-windows-onnx-parser-seh-boundary-validation.v1"
  validationState = if ($failures.Count -eq 0) { "passed" } else { "failed" }
  strict = [bool]$Strict
  checkCount = $checks.Count
  passedCount = @($checks | Where-Object passed).Count
  failureCount = $failures.Count
  checks = @($checks)
}

$outputFullPath = Resolve-RepositoryPath $OutputPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
[IO.File]::WriteAllText($outputFullPath, ($validation | ConvertTo-Json -Depth 10) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

$markdownPath = [IO.Path]::ChangeExtension($outputFullPath, ".md")
$rows = @($checks | ForEach-Object { "| ``$($_.name)`` | ``$($_.passed)`` | ``$($_.detail)`` |" })
$markdown = @(
  "# TRT8 Windows ONNX Parser SEH Boundary Validation", "",
  "- State: ``$($validation.validationState)``", "- Checks: ``$($validation.checkCount)``", "- Failures: ``$($validation.failureCount)``", "",
  "| Check | Passed | Detail |", "|---|---:|---|"
) + $rows
[IO.File]::WriteAllText($markdownPath, ($markdown -join [Environment]::NewLine) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

Write-Host "TRT8 Windows ONNX parser SEH boundary validation: $($validation.validationState); checks=$($validation.checkCount); failures=$($validation.failureCount)"
if ($Strict -and $failures.Count -gt 0) { throw "TRT8 Windows ONNX parser SEH boundary validation failed: $($failures.name -join ', ')" }
