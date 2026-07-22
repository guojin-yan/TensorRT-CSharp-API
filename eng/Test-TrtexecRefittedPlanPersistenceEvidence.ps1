param(
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$evidencePath = Join-Path $root "artifacts\interface-coverage\trtexec-refitted-plan-persistence-evidence.json"
$validationPath = Join-Path $root "artifacts\interface-coverage\trtexec-refitted-plan-persistence-validation.json"
$validationMarkdownPath = Join-Path $root "artifacts\interface-coverage\trtexec-refitted-plan-persistence-validation.md"
$evidence = Get-Content -LiteralPath $evidencePath -Raw | ConvertFrom-Json
$checks = [System.Collections.Generic.List[object]]::new()

function Add-Check([string]$Id, [bool]$Passed, [string]$Actual) {
  $checks.Add([pscustomobject]@{ id = $Id; passed = $Passed; actual = $Actual })
}

function Read-Record([string]$RelativePath) {
  return Get-Content -LiteralPath (Join-Path $root $RelativePath) -Raw | ConvertFrom-Json
}

function Get-RecordSha256([string]$RelativePath) {
  return (Get-FileHash -LiteralPath (Join-Path $root $RelativePath) -Algorithm SHA256).Hash.ToLowerInvariant()
}

$trt10 = $evidence.tensorRt10
$sameReport = Read-Record $trt10.sameProcessReportPath
$sameOutput = Read-Record $trt10.sameProcessOutputPath
$secondReport = Read-Record $trt10.secondProcessReportPath
$secondOutput = Read-Record $trt10.secondProcessOutputPath
$baselineReport = Read-Record $trt10.baselineReportPath
$baselineOutput = Read-Record $trt10.baselineOutputPath
$trt8Report = Read-Record $evidence.tensorRt8.dryRunReportPath
$trt11Report = Read-Record $evidence.tensorRt11.reportPath
$sameSnapshot = $sameReport.RefitPersistenceSnapshot

Add-Check "schema" ($evidence.schemaVersion -eq "trtexec-refitted-plan-persistence-evidence.v1") ([string]$evidence.schemaVersion)
Add-Check "contract-requires-refit-source" (@($evidence.contract.requires) -contains "--refitFromOnnx") (@($evidence.contract.requires) -join ",")
Add-Check "trt10-persisted" ([bool]$trt10.persisted -and [bool]$sameSnapshot.Succeeded) "$($trt10.persisted)/$($sameSnapshot.Succeeded)"
Add-Check "trt10-artifact-differs" ([bool]$trt10.artifactDiffersFromStrippedPlan -and $trt10.strippedPlanSha256 -ne $trt10.persistedPlanSha256) ([string]$trt10.persistedPlanSha256)
Add-Check "trt10-exclude-weights-present-before" (([int]$trt10.serializationFlagsBefore -band [int]$trt10.excludeWeightsFlag) -ne 0) ([string]$trt10.serializationFlagsBefore)
Add-Check "trt10-exclude-weights-cleared-after" (([int]$trt10.serializationFlagsAfter -band [int]$trt10.excludeWeightsFlag) -eq 0) ([string]$trt10.serializationFlagsAfter)
Add-Check "trt10-weights-included" ([bool]$trt10.refittableWeightsIncludedInSerialization -and [bool]$sameSnapshot.RefittableWeightsIncludedInSerialization) "$($trt10.refittableWeightsIncludedInSerialization)/$($sameSnapshot.RefittableWeightsIncludedInSerialization)"
Add-Check "trt10-original-disposed" ([bool]$trt10.originalEngineDisposedBeforeReload -and [bool]$sameSnapshot.OriginalRefittedEngineDisposedBeforeReload) "$($trt10.originalEngineDisposedBeforeReload)/$($sameSnapshot.OriginalRefittedEngineDisposedBeforeReload)"
Add-Check "trt10-same-process-reload" ([bool]$trt10.sameProcessReloadSucceeded -and [bool]$sameSnapshot.ReloadSucceeded) "$($trt10.sameProcessReloadSucceeded)/$($sameSnapshot.ReloadSucceeded)"
Add-Check "trt10-full-plan-refittable-fact" (-not [bool]$trt10.reloadEngineRefittable -and -not [bool]$sameSnapshot.ReloadEngineRefittable) "$($trt10.reloadEngineRefittable)/$($sameSnapshot.ReloadEngineRefittable)"
Add-Check "trt10-reload-metadata" ([int]$trt10.reloadIoTensorCount -gt 0 -and [int]$trt10.reloadLayerCount -gt 0 -and [int]$trt10.reloadOptimizationProfileCount -gt 0) "$($trt10.reloadIoTensorCount)/$($trt10.reloadLayerCount)/$($trt10.reloadOptimizationProfileCount)"
Add-Check "trt10-context-gate" ([bool]$trt10.reloadContextCreationAllowed -and [bool]$sameSnapshot.ReloadContextCreationAllowed) "$($trt10.reloadContextCreationAllowed)/$($sameSnapshot.ReloadContextCreationAllowed)"
Add-Check "trt10-runtime-selected" ([bool]$trt10.reloadEngineSelectedForRuntime -and [bool]$sameSnapshot.ReloadEngineSelectedForRuntime) "$($trt10.reloadEngineSelectedForRuntime)/$($sameSnapshot.ReloadEngineSelectedForRuntime)"
Add-Check "trt10-same-process-inference" ([bool]$trt10.sameProcessInferenceRan -and [bool]$sameReport.InferenceRan) "$($trt10.sameProcessInferenceRan)/$($sameReport.InferenceRan)"
Add-Check "trt10-same-report-hash" ((Get-RecordSha256 $trt10.sameProcessReportPath) -eq $trt10.sameProcessReportSha256) (Get-RecordSha256 $trt10.sameProcessReportPath)
Add-Check "trt10-same-output-artifact-hash" ((Get-RecordSha256 $trt10.sameProcessOutputPath) -eq $trt10.sameProcessOutputArtifactSha256) (Get-RecordSha256 $trt10.sameProcessOutputPath)
Add-Check "trt10-second-report-hash" ((Get-RecordSha256 $trt10.secondProcessReportPath) -eq $trt10.secondProcessReportSha256) (Get-RecordSha256 $trt10.secondProcessReportPath)
Add-Check "trt10-second-output-artifact-hash" ((Get-RecordSha256 $trt10.secondProcessOutputPath) -eq $trt10.secondProcessOutputArtifactSha256) (Get-RecordSha256 $trt10.secondProcessOutputPath)
Add-Check "trt10-baseline-report-hash" ((Get-RecordSha256 $trt10.baselineReportPath) -eq $trt10.baselineReportSha256) (Get-RecordSha256 $trt10.baselineReportPath)
Add-Check "trt10-baseline-output-artifact-hash" ((Get-RecordSha256 $trt10.baselineOutputPath) -eq $trt10.baselineOutputArtifactSha256) (Get-RecordSha256 $trt10.baselineOutputPath)
Add-Check "trt10-second-process-independent-command" (-not $secondReport.NormalizedCommandLine.Contains("--onnx", [StringComparison]::Ordinal) -and -not $secondReport.NormalizedCommandLine.Contains("--refitFromOnnx", [StringComparison]::Ordinal) -and $secondReport.NormalizedCommandLine.Contains("--loadEngine", [StringComparison]::Ordinal)) ([string]$secondReport.NormalizedCommandLine)
Add-Check "trt10-second-process-reload" ([bool]$trt10.secondProcessReloadSucceeded -and [bool]$secondReport.LoadedEngineDiagnostics.Succeeded -and [bool]$secondReport.InferenceRan) "$($trt10.secondProcessReloadSucceeded)/$($secondReport.LoadedEngineDiagnostics.Succeeded)/$($secondReport.InferenceRan)"
Add-Check "trt10-three-way-output-match" ([bool]$trt10.sameProcessOutputExactMatch -and [bool]$trt10.secondProcessOutputExactMatch -and $sameOutput.OutputSha256 -eq $secondOutput.OutputSha256 -and $sameOutput.OutputSha256 -eq $baselineOutput.OutputSha256 -and $sameOutput.OutputSha256 -eq $trt10.baselineOutputSha256) ([string]$sameOutput.OutputSha256)
Add-Check "trt10-process-exits" ([int]$trt10.sameProcessExitCode -eq 0 -and [int]$trt10.secondProcessExitCode -eq 0 -and [int]$trt10.baselineProcessExitCode -eq 0) "$($trt10.sameProcessExitCode)/$($trt10.secondProcessExitCode)/$($trt10.baselineProcessExitCode)"
Add-Check "trt8-report-hash" ((Get-RecordSha256 $evidence.tensorRt8.dryRunReportPath) -eq $evidence.tensorRt8.dryRunReportSha256) (Get-RecordSha256 $evidence.tensorRt8.dryRunReportPath)
Add-Check "trt8-dry-parse-only" ($trt8Report.State -eq "dry-run-precheck" -and @($trt8Report.OptionImplementationStatus.ParseOnlyOptions) -contains "--saveRefittedEngine" -and -not [bool]$evidence.tensorRt8.dryRunPersistedFileWritten) "$($trt8Report.State)/$($evidence.tensorRt8.saveRefittedEngineParseOnly)"
Add-Check "trt8-nondry-guard" ([int]$evidence.tensorRt8.nonDryExitCode -eq 2 -and [bool]$evidence.tensorRt8.nonDryGuardedBeforeNativeExecution -and $evidence.tensorRt8.nonDryGuard.Contains("no ONNX parser-refitter API", [StringComparison]::Ordinal)) "$($evidence.tensorRt8.nonDryExitCode)/$($evidence.tensorRt8.nonDryGuard)"
Add-Check "trt11-report-hash" ((Get-RecordSha256 $evidence.tensorRt11.reportPath) -eq $evidence.tensorRt11.reportSha256) (Get-RecordSha256 $evidence.tensorRt11.reportPath)
Add-Check "trt11-dependency-only" ($trt11Report.State -eq "dependency-probe-only" -and $trt11Report.ProofClassification -eq "dependency-probe-only") "$($trt11Report.State)/$($trt11Report.ProofClassification)"
Add-Check "trt11-not-applied" (-not (@($trt11Report.OptionImplementationStatus.AppliedOptions) -contains "--saveRefittedEngine") -and @($trt11Report.OptionImplementationStatus.ParseOnlyOptions) -contains "--saveRefittedEngine") (@($trt11Report.OptionImplementationStatus.ParseOnlyOptions) -join ",")
Add-Check "boundary-local" ([bool]$evidence.proofBoundary.isLocalSourceTreePersistenceEvidence) ([string]$evidence.proofBoundary.isLocalSourceTreePersistenceEvidence)
Add-Check "boundary-no-accuracy" (-not [bool]$evidence.proofBoundary.isModelAccuracyProof) ([string]$evidence.proofBoundary.isModelAccuracyProof)
Add-Check "boundary-no-package" (-not [bool]$evidence.proofBoundary.isPackageConsumerRuntimeProof -and -not [bool]$sameReport.IsPackageConsumerRuntimeProof -and -not [bool]$secondReport.IsPackageConsumerRuntimeProof -and -not [bool]$baselineReport.IsPackageConsumerRuntimeProof) "$($evidence.proofBoundary.isPackageConsumerRuntimeProof)/$($sameReport.IsPackageConsumerRuntimeProof)/$($secondReport.IsPackageConsumerRuntimeProof)/$($baselineReport.IsPackageConsumerRuntimeProof)"
Add-Check "boundary-no-publish" (-not [bool]$evidence.proofBoundary.canPublishPublicly -and -not [bool]$evidence.proofBoundary.publicReleaseSideEffectsExecuted) "$($evidence.proofBoundary.canPublishPublicly)/$($evidence.proofBoundary.publicReleaseSideEffectsExecuted)"
Add-Check "boundary-no-release-close" (-not [bool]$evidence.proofBoundary.canCloseReleaseIssue) ([string]$evidence.proofBoundary.canCloseReleaseIssue)

$failed = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "trtexec-refitted-plan-persistence-validation.v1"
  strict = [bool]$Strict
  checkCount = $checks.Count
  passedCount = $checks.Count - $failed.Count
  failureCount = $failed.Count
  checks = $checks
}
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $validationPath -Encoding utf8

$lines = @(
  "# TensorRtExec Refitted Plan Persistence Validation",
  "",
  "- Strict: ``$([bool]$Strict)``",
  "- Checks: ``$($checks.Count)``",
  "- Passed: ``$($checks.Count - $failed.Count)``",
  "- Failed: ``$($failed.Count)``",
  "",
  "| Check | Passed | Actual |",
  "| --- | --- | --- |"
)
foreach ($check in $checks) {
  $actual = ([string]$check.actual).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
  $lines += "| ``$($check.id)`` | ``$($check.passed)`` | ``$actual`` |"
}
$lines | Set-Content -LiteralPath $validationMarkdownPath -Encoding utf8

Write-Host "TensorRtExec refitted-plan persistence evidence: $($checks.Count - $failed.Count)/$($checks.Count) checks passed."
if ($Strict -and $failed.Count -gt 0) {
  throw "TensorRtExec refitted-plan persistence evidence validation failed: $($failed.id -join ', ')"
}
