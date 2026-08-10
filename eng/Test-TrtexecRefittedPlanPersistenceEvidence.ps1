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
$trt11 = $evidence.tensorRt11
$trt11SameReport = Read-Record $trt11.sameProcessReportPath
$trt11SameOutput = Read-Record $trt11.sameProcessOutputPath
$trt11SameValidation = Read-Record $trt11.sameProcessStrictValidationPath
$trt11SecondReport = Read-Record $trt11.secondProcessReportPath
$trt11SecondOutput = Read-Record $trt11.secondProcessOutputPath
$trt11SecondValidation = Read-Record $trt11.secondProcessStrictValidationPath
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
Add-Check "trt11-same-report-hash" ((Get-RecordSha256 $trt11.sameProcessReportPath) -eq $trt11.sameProcessReportSha256) (Get-RecordSha256 $trt11.sameProcessReportPath)
Add-Check "trt11-same-output-artifact-hash" ((Get-RecordSha256 $trt11.sameProcessOutputPath) -eq $trt11.sameProcessOutputArtifactSha256) (Get-RecordSha256 $trt11.sameProcessOutputPath)
Add-Check "trt11-same-validation-hash" ((Get-RecordSha256 $trt11.sameProcessStrictValidationPath) -eq $trt11.sameProcessStrictValidationSha256) (Get-RecordSha256 $trt11.sameProcessStrictValidationPath)
Add-Check "trt11-second-report-hash" ((Get-RecordSha256 $trt11.secondProcessReportPath) -eq $trt11.secondProcessReportSha256) (Get-RecordSha256 $trt11.secondProcessReportPath)
Add-Check "trt11-second-output-artifact-hash" ((Get-RecordSha256 $trt11.secondProcessOutputPath) -eq $trt11.secondProcessOutputArtifactSha256) (Get-RecordSha256 $trt11.secondProcessOutputPath)
Add-Check "trt11-second-validation-hash" ((Get-RecordSha256 $trt11.secondProcessStrictValidationPath) -eq $trt11.secondProcessStrictValidationSha256) (Get-RecordSha256 $trt11.secondProcessStrictValidationPath)
Add-Check "trt11-state" ($trt11SameReport.State -eq "external-onnx-refit-reload-reference-validated-runtime" -and [bool]$trt11SameReport.Success -and -not [bool]$trt11SameReport.Skipped) "$($trt11SameReport.State)/$($trt11SameReport.Success)/$($trt11SameReport.Skipped)"
Add-Check "trt11-applied" (@($trt11SameReport.OptionImplementationStatus.AppliedOptions) -contains "--refitFromOnnx" -and @($trt11SameReport.OptionImplementationStatus.AppliedOptions) -contains "--saveRefittedEngine") (@($trt11SameReport.OptionImplementationStatus.AppliedOptions) -join ",")
Add-Check "trt11-plan-hashes" ((Get-RecordSha256 $trt11.strippedPlanPath) -eq $trt11.strippedPlanSha256 -and (Get-RecordSha256 $trt11.persistedPlanPath) -eq $trt11.persistedPlanSha256 -and $trt11.strippedPlanSha256 -ne $trt11.persistedPlanSha256) "$((Get-RecordSha256 $trt11.strippedPlanPath))/$((Get-RecordSha256 $trt11.persistedPlanPath))"
Add-Check "trt11-exclude-weights" (([int]$trt11.serializationFlagsBefore -band [int]$trt11.excludeWeightsFlag) -ne 0 -and ([int]$trt11.serializationFlagsAfter -band [int]$trt11.excludeWeightsFlag) -eq 0 -and [bool]$trt11SameReport.RefitPersistenceSnapshot.RefittableWeightsIncludedInSerialization) "$($trt11.serializationFlagsBefore)/$($trt11.serializationFlagsAfter)"
Add-Check "trt11-owner-dispose-reload" ([bool]$trt11SameReport.RefitPersistenceSnapshot.OriginalRefittedEngineDisposedBeforeReload -and [bool]$trt11SameReport.RefitPersistenceSnapshot.ReloadSucceeded -and [bool]$trt11SameReport.RefitPersistenceSnapshot.ReloadContextCreationAllowed) "$($trt11SameReport.RefitPersistenceSnapshot.OriginalRefittedEngineDisposedBeforeReload)/$($trt11SameReport.RefitPersistenceSnapshot.ReloadSucceeded)/$($trt11SameReport.RefitPersistenceSnapshot.ReloadContextCreationAllowed)"
Add-Check "trt11-reload-metadata" ([int]$trt11SameReport.RefitPersistenceSnapshot.ReloadIOTensorCount -eq 2 -and [int]$trt11SameReport.RefitPersistenceSnapshot.ReloadLayerCount -eq 5 -and [int]$trt11SameReport.RefitPersistenceSnapshot.ReloadOptimizationProfileCount -eq 1) "$($trt11SameReport.RefitPersistenceSnapshot.ReloadIOTensorCount)/$($trt11SameReport.RefitPersistenceSnapshot.ReloadLayerCount)/$($trt11SameReport.RefitPersistenceSnapshot.ReloadOptimizationProfileCount)"
Add-Check "trt11-runtime-selected" ([bool]$trt11SameReport.RefitPersistenceSnapshot.ReloadEngineSelectedForRuntime -and [bool]$trt11SameReport.RefitPersistenceSnapshot.InferenceRanFromReloadedEngine -and [bool]$trt11SameReport.InferenceRan) "$($trt11SameReport.RefitPersistenceSnapshot.ReloadEngineSelectedForRuntime)/$($trt11SameReport.RefitPersistenceSnapshot.InferenceRanFromReloadedEngine)"
Add-Check "trt11-second-process-independent-command" (-not $trt11SecondReport.NormalizedCommandLine.Contains("--onnx", [StringComparison]::Ordinal) -and -not $trt11SecondReport.NormalizedCommandLine.Contains("--refitFromOnnx", [StringComparison]::Ordinal) -and $trt11SecondReport.NormalizedCommandLine.Contains("--loadEngine", [StringComparison]::Ordinal)) ([string]$trt11SecondReport.NormalizedCommandLine)
Add-Check "trt11-second-process-reload" ($trt11SecondReport.State -eq "load-engine-reference-validated-runtime" -and [bool]$trt11SecondReport.LoadedEngineDiagnostics.Succeeded -and [bool]$trt11SecondReport.BindingMetadata.ContextReadinessAttached -and [bool]$trt11SecondReport.BindingMetadata.IsReadyForEnqueue -and [bool]$trt11SecondReport.InferenceRan) "$($trt11SecondReport.State)/$($trt11SecondReport.LoadedEngineDiagnostics.Succeeded)/$($trt11SecondReport.BindingMetadata.IsReadyForEnqueue)"
Add-Check "trt11-output-match" ([bool]$trt11SameOutput.OutputValidated -and [bool]$trt11SecondOutput.OutputValidated -and $trt11SameOutput.OutputSha256 -eq $trt11SecondOutput.OutputSha256 -and $trt11SameOutput.OutputSha256 -eq $trt11.sameProcessOutputSha256 -and [int]$trt11SameOutput.ReferenceValidation.TensorComparisons[0].MismatchCount -eq 0 -and [int]$trt11SecondOutput.ReferenceValidation.TensorComparisons[0].MismatchCount -eq 0) "$($trt11SameOutput.OutputSha256)/$($trt11SecondOutput.OutputSha256)"
Add-Check "trt11-strict-validations" ($trt11SameValidation.validationState -eq "tensor-rt-exec-report-ready" -and [int]$trt11SameValidation.failedBlockers -eq 0 -and @($trt11SameValidation.validationItems).Count -eq 69 -and $trt11SecondValidation.validationState -eq "tensor-rt-exec-report-ready" -and [int]$trt11SecondValidation.failedBlockers -eq 0 -and @($trt11SecondValidation.validationItems).Count -eq 69) "$($trt11SameValidation.validationState)/$($trt11SameValidation.failedBlockers)/$($trt11SecondValidation.validationState)/$($trt11SecondValidation.failedBlockers)"
Add-Check "trt11-process-exits" ([int]$trt11.sameProcessExitCode -eq 0 -and [int]$trt11.secondProcessExitCode -eq 0) "$($trt11.sameProcessExitCode)/$($trt11.secondProcessExitCode)"
Add-Check "trt11-historical-probe-retained" ($trt11.historicalDependencyProbe.state -eq "dependency-probe-only" -and (Get-RecordSha256 $trt11.historicalDependencyProbe.reportPath) -eq $trt11.historicalDependencyProbe.reportSha256) "$($trt11.historicalDependencyProbe.state)/$($trt11.historicalDependencyProbe.reportPath)"
Add-Check "boundary-local" ([bool]$evidence.proofBoundary.isLocalSourceTreePersistenceEvidence) ([string]$evidence.proofBoundary.isLocalSourceTreePersistenceEvidence)
Add-Check "boundary-no-accuracy" (-not [bool]$evidence.proofBoundary.isModelAccuracyProof) ([string]$evidence.proofBoundary.isModelAccuracyProof)
Add-Check "boundary-trt11-local" ([bool]$evidence.proofBoundary.isTensorRt11RefitPersistenceEvidence -and [bool]$evidence.proofBoundary.isTensorRt11SecondProcessReloadEvidence) "$($evidence.proofBoundary.isTensorRt11RefitPersistenceEvidence)/$($evidence.proofBoundary.isTensorRt11SecondProcessReloadEvidence)"
Add-Check "boundary-no-package" (-not [bool]$evidence.proofBoundary.isPackageConsumerRuntimeProof -and -not [bool]$sameReport.IsPackageConsumerRuntimeProof -and -not [bool]$secondReport.IsPackageConsumerRuntimeProof -and -not [bool]$baselineReport.IsPackageConsumerRuntimeProof -and -not [bool]$trt11SameReport.IsPackageConsumerRuntimeProof -and -not [bool]$trt11SecondReport.IsPackageConsumerRuntimeProof) "$($evidence.proofBoundary.isPackageConsumerRuntimeProof)/$($sameReport.IsPackageConsumerRuntimeProof)/$($secondReport.IsPackageConsumerRuntimeProof)/$($baselineReport.IsPackageConsumerRuntimeProof)/$($trt11SameReport.IsPackageConsumerRuntimeProof)/$($trt11SecondReport.IsPackageConsumerRuntimeProof)"
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
