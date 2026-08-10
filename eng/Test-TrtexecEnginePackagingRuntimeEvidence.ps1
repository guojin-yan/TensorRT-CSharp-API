[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/interface-coverage/trtexec-engine-packaging-runtime-evidence.json",
  [string]$OutputPath = "artifacts/interface-coverage/trtexec-engine-packaging-runtime-evidence-validation.json",
  [switch]$Strict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }

  return [IO.Path]::GetFullPath((Join-Path $repositoryRoot ($Path -replace "/", "\")))
}

$inputFullPath = Resolve-RepositoryPath $InputPath
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Evidence file was not found: $inputFullPath"
}

$record = Get-Content -LiteralPath $inputFullPath -Raw | ConvertFrom-Json
$checks = [Collections.Generic.List[object]]::new()

function Add-Check {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][bool]$Passed,
    [Parameter(Mandatory = $true)][string]$Detail
  )

  $checks.Add([pscustomobject][ordered]@{ name = $Name; passed = $Passed; detail = $Detail }) | Out-Null
}

Add-Check "schema" ($record.schemaVersion -eq "trtexec-engine-packaging-runtime-evidence.v1") ([string]$record.schemaVersion)
Add-Check "trt8-version-compatible" ([bool]$record.tensorRt8.versionCompatibleApplied) "TRT8 version-compatible flag readback"
Add-Check "trt8-exclude-lean" ([bool]$record.tensorRt8.excludeLeanRuntimeApplied) "TRT8 exclude-lean flag readback"
Add-Check "trt8-refit-conflict-guard" ([bool]$record.tensorRt8.refitVersionCompatibleConflictGuarded) "TRT8 version-compatible+refit remains guarded"
Add-Check "trt8-strip-guard" ([bool]$record.tensorRt8.stripWeightsGuarded) "TRT8 strip remains guarded"
Add-Check "trt8-streaming-guard" ([bool]$record.tensorRt8.weightStreamingGuarded) "TRT8 streaming remains guarded"

$versionRefit = $record.tensorRt10.versionCompatibleRefit
Add-Check "trt10-version-refit-runtime" ([bool]$versionRefit.inferenceRan -and [bool]$versionRefit.outputMatch) "TRT10 version-compatible/refit identity runtime"
Add-Check "trt10-refittable-readback" ([bool]$versionRefit.refitConfigReadbackMatch -and [bool]$versionRefit.refittableEngineReadbackMatch) "Config and engine refit readback"
Add-Check "trt10-runtime-host-code" ([bool]$versionRefit.runtimeHostCodeReadbackMatch) "Runtime host code enabled for version-compatible plan"
Add-Check "trt10-strip-refit-identical" ([bool]$record.tensorRt10.stripWeights.stripPlanReadbackMatch -and $record.tensorRt10.stripWeights.defaultRefitMode -eq "RefitIdentical") "StripPlan plus RefitIdentical"

$streaming = $record.tensorRt10.weightStreaming
$halfBudget = [long]$streaming.streamableWeightsBytes / 2
Add-Check "trt10-weighted-model" ([long]$streaming.streamableWeightsBytes -gt 0 -and [long]$record.model.lengthBytes -gt 0) "Non-zero real model streamable weights"
Add-Check "trt10-percentage-budget" ([long]$streaming.resolvedBudgetBytes -eq $halfBudget -and [long]$streaming.readbackBudgetBytes -eq [long]$streaming.resolvedBudgetBytes) "50 percent budget resolved and read back"
Add-Check "trt10-streaming-scratch" ([long]$streaming.scratchBytes -gt 0) "Non-zero scratch bytes prove active streaming budget"
Add-Check "trt10-context-order" ([bool]$streaming.contextCreatedAfterBudgetReadback) "Budget applied before execution context"
Add-Check "trt10-weighted-enqueue" ([bool]$streaming.inferenceRan -and $streaming.outputValidationState -eq "captured-unverified") "Weighted model enqueue with conservative output classification"

$load = $record.tensorRt10.loadEngineAutomaticBudget
Add-Check "trt10-load-diagnostics" ([bool]$load.diagnosticsSucceeded) "Load-engine readonly diagnostics succeeded"
Add-Check "trt10-load-auto-budget" ([bool]$load.readbackMatch -and [long]$load.resolvedBudgetBytes -eq [long]$load.automaticBudgetBytes) "Automatic budget resolved and read back"
Add-Check "trt10-load-enqueue" ([bool]$load.inferenceRan) "Load-engine bounded runtime executed"

$trt11ReportPath = Resolve-RepositoryPath $record.reports.trt11.path
$trt11HistoricalPath = Resolve-RepositoryPath $record.reports.trt11HistoricalDependencyProbe.path
$trt11Report = Get-Content -LiteralPath $trt11ReportPath -Raw | ConvertFrom-Json
Add-Check "trt11-refit-runtime" (
  (Get-FileHash -LiteralPath $trt11ReportPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $record.reports.trt11.sha256 -and
  $trt11Report.State -eq "external-onnx-refit-reload-reference-validated-runtime" -and
  [bool]$trt11Report.RefitSnapshot.Succeeded -and
  [bool]$trt11Report.RefitPersistenceSnapshot.Succeeded -and
  [bool]$trt11Report.OutputValidated -and
  @($trt11Report.OptionImplementationStatus.AppliedOptions) -contains "--refitFromOnnx" -and
  @($trt11Report.OptionImplementationStatus.AppliedOptions) -contains "--saveRefittedEngine") "TRT11 stripped-plan refit, persistence, reload, and reference validation"
Add-Check "trt11-historical-probe-retained" (
  (Get-FileHash -LiteralPath $trt11HistoricalPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $record.reports.trt11HistoricalDependencyProbe.sha256 -and
  [bool]$record.tensorRt11.historicalDependencyProbeRetained) "Historical TRT11 dependency probe remains immutable"
$boundary = $record.proofBoundary
Add-Check "proof-boundary" (
  [bool]$boundary.isBuilderAndEnginePolicyEvidence -and
  [bool]$boundary.isWeightedModelEnqueueEvidence -and
  -not [bool]$boundary.isModelAccuracyProof -and
  -not [bool]$boundary.isCrossVersionLeanRuntimeProof -and
  [bool]$boundary.isStrippedPlanRefitLifecycleProof -and
  -not [bool]$boundary.isPackageConsumerRuntimeProof -and
  -not [bool]$boundary.isPostPublishProof -and
  -not [bool]$boundary.canPublishPublicly -and
  -not [bool]$boundary.publicReleaseSideEffectsExecuted) "No proof or publish promotion"

$failures = @($checks | Where-Object { -not [bool]$_.passed })
$validation = [pscustomobject][ordered]@{
  schemaVersion = "trtexec-engine-packaging-runtime-evidence-validation.v1"
  validationState = if ($failures.Count -eq 0) { "passed" } else { "failed" }
  checkCount = $checks.Count
  failureCount = $failures.Count
  checks = @($checks)
}

$outputFullPath = Resolve-RepositoryPath $OutputPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
$json = $validation | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText($outputFullPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

$markdownPath = [IO.Path]::ChangeExtension($outputFullPath, ".md")
$rows = @($checks | ForEach-Object { "| $($_.name) | $($_.passed) | $($_.detail) |" })
$markdown = @(
  "# Trtexec Engine Packaging Runtime Evidence Validation",
  "",
  "- State: ``$($validation.validationState)``",
  "- Checks: ``$($validation.checkCount)``",
  "- Failures: ``$($validation.failureCount)``",
  "",
  "| Check | Passed | Detail |",
  "|---|---:|---|"
) + $rows
[IO.File]::WriteAllText($markdownPath, ($markdown -join [Environment]::NewLine) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

Write-Host "Trtexec engine packaging runtime evidence validation: $($validation.validationState); checks=$($validation.checkCount); failures=$($validation.failureCount)"
if ($Strict -and $failures.Count -gt 0) {
  throw "Trtexec engine packaging runtime evidence validation failed: $($failures.name -join ', ')"
}
