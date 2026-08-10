param(
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$evidencePath = Join-Path $root "artifacts\interface-coverage\trtexec-onnx-refit-lifecycle-evidence.json"
$validationPath = Join-Path $root "artifacts\interface-coverage\trtexec-onnx-refit-lifecycle-validation.json"
$validationMarkdownPath = Join-Path $root "artifacts\interface-coverage\trtexec-onnx-refit-lifecycle-validation.md"
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

$trt11 = $evidence.tensorRt11
$trt11Report = Read-Record $trt11.reportPath
$trt11Output = Read-Record $trt11.outputPath
$trt11Validation = Read-Record $trt11.strictValidationPath

Add-Check "schema" ($evidence.schemaVersion -eq "trtexec-onnx-refit-lifecycle-evidence.v1") ([string]$evidence.schemaVersion)
Add-Check "contract-requires-onnx" (@($evidence.contract.requires) -contains "--onnx") (@($evidence.contract.requires) -join ",")
Add-Check "contract-requires-strip" (@($evidence.contract.requires) -contains "--stripWeights") (@($evidence.contract.requires) -join ",")
Add-Check "contract-requires-refit" (@($evidence.contract.requires) -contains "--refit") (@($evidence.contract.requires) -join ",")
Add-Check "trt10-strip-applied" ([bool]$evidence.tensorRt10.stripPlanApplied) ([string]$evidence.tensorRt10.stripPlanApplied)
Add-Check "trt10-refit-applied" ([bool]$evidence.tensorRt10.explicitRefitApplied) ([string]$evidence.tensorRt10.explicitRefitApplied)
Add-Check "trt10-refittable-before" ([bool]$evidence.tensorRt10.engineRefittableBefore) ([string]$evidence.tensorRt10.engineRefittableBefore)
Add-Check "trt10-parser-refit" ([bool]$evidence.tensorRt10.parserRefitReturned) ([string]$evidence.tensorRt10.parserRefitReturned)
Add-Check "trt10-engine-refit-commit" ([bool]$evidence.tensorRt10.engineRefitReturned) ([string]$evidence.tensorRt10.engineRefitReturned)
Add-Check "trt10-refittable-after" ([bool]$evidence.tensorRt10.engineRefittableAfter) ([string]$evidence.tensorRt10.engineRefittableAfter)
Add-Check "trt10-all-inventory" ([int]$evidence.tensorRt10.allWeightsBefore -gt 0 -and [int]$evidence.tensorRt10.allWeightsAfter -gt 0) "$($evidence.tensorRt10.allWeightsBefore)/$($evidence.tensorRt10.allWeightsAfter)"
Add-Check "trt10-missing-zero" ([int]$evidence.tensorRt10.missingWeightsAfter -eq 0) ([string]$evidence.tensorRt10.missingWeightsAfter)
Add-Check "trt10-parser-errors-zero" ([int]$evidence.tensorRt10.parserErrorCount -eq 0) ([string]$evidence.tensorRt10.parserErrorCount)
Add-Check "trt10-context-gate" ([bool]$evidence.tensorRt10.contextCreationAllowed -and [bool]$evidence.tensorRt10.contextCreatedAfterRefitCommit) "$($evidence.tensorRt10.contextCreationAllowed)/$($evidence.tensorRt10.contextCreatedAfterRefitCommit)"
Add-Check "trt10-inference" ([bool]$evidence.tensorRt10.inferenceRan) ([string]$evidence.tensorRt10.inferenceRan)
Add-Check "trt10-output-shape" ([int]$evidence.tensorRt10.outputElementCount -eq 10 -and [int]$evidence.tensorRt10.outputByteLength -eq 40) "$($evidence.tensorRt10.outputElementCount)/$($evidence.tensorRt10.outputByteLength)"
Add-Check "trt10-output-match" ([bool]$evidence.tensorRt10.outputExactMatch -and $evidence.tensorRt10.refitOutputSha256 -eq $evidence.tensorRt10.baselineOutputSha256) ([string]$evidence.tensorRt10.refitOutputSha256)
Add-Check "trt8-precheck" ($evidence.tensorRt8.state -eq "dry-run-precheck" -and [bool]$evidence.tensorRt8.refitFromOnnxParseOnly) "$($evidence.tensorRt8.state)/$($evidence.tensorRt8.refitFromOnnxParseOnly)"
Add-Check "trt11-report-hash" ((Get-RecordSha256 $trt11.reportPath) -eq $trt11.reportSha256) (Get-RecordSha256 $trt11.reportPath)
Add-Check "trt11-output-artifact-hash" ((Get-RecordSha256 $trt11.outputPath) -eq $trt11.outputArtifactSha256) (Get-RecordSha256 $trt11.outputPath)
Add-Check "trt11-strict-validation-hash" ((Get-RecordSha256 $trt11.strictValidationPath) -eq $trt11.strictValidationSha256) (Get-RecordSha256 $trt11.strictValidationPath)
Add-Check "trt11-state" ($trt11Report.State -eq "external-onnx-refit-reload-reference-validated-runtime" -and [bool]$trt11Report.Success -and -not [bool]$trt11Report.Skipped) "$($trt11Report.State)/$($trt11Report.Success)/$($trt11Report.Skipped)"
Add-Check "trt11-options-applied" (@($trt11Report.OptionImplementationStatus.AppliedOptions) -contains "--refitFromOnnx" -and @($trt11Report.OptionImplementationStatus.AppliedOptions) -contains "--saveRefittedEngine") (@($trt11Report.OptionImplementationStatus.AppliedOptions) -join ",")
Add-Check "trt11-refit-gates" ([bool]$trt11Report.RefitSnapshot.Succeeded -and [bool]$trt11Report.RefitSnapshot.ParserRefitReturned -and [bool]$trt11Report.RefitSnapshot.EngineRefitReturned -and [bool]$trt11Report.RefitSnapshot.EngineRefittableBefore -and [bool]$trt11Report.RefitSnapshot.EngineRefittableAfter -and [bool]$trt11Report.RefitSnapshot.ContextCreationAllowed) "$($trt11Report.RefitSnapshot.State)/$($trt11Report.RefitSnapshot.ContextCreationAllowed)"
Add-Check "trt11-refit-inventory" (@($trt11Report.RefitSnapshot.MissingWeightsAfter).Count -eq 0 -and @($trt11Report.RefitSnapshot.AllWeightsAfter).Count -eq 6 -and [int]$trt11Report.RefitSnapshot.ParserErrorCount -eq 0) "$(@($trt11Report.RefitSnapshot.MissingWeightsAfter).Count)/$(@($trt11Report.RefitSnapshot.AllWeightsAfter).Count)/$($trt11Report.RefitSnapshot.ParserErrorCount)"
Add-Check "trt11-model-hash" ([long]$trt11Report.RefitSnapshot.SourceLengthBytes -eq [long]$trt11.modelLengthBytes -and $trt11Report.RefitSnapshot.SourceSha256 -eq $trt11.modelSha256) "$($trt11Report.RefitSnapshot.SourceLengthBytes)/$($trt11Report.RefitSnapshot.SourceSha256)"
Add-Check "trt11-runtime-output" ([bool]$trt11Report.InferenceRan -and [bool]$trt11Report.OutputValidated -and [bool]$trt11Output.OutputValidated -and [int]$trt11Output.OutputElementCount -eq 10 -and [int]$trt11Output.OutputByteLength -eq 40) "$($trt11Report.InferenceRan)/$($trt11Report.OutputValidated)/$($trt11Output.OutputSha256)"
Add-Check "trt11-reference-comparison" ([bool]$trt11Output.ReferenceValidation.Passed -and [int]$trt11Output.ReferenceValidation.TensorComparisons[0].MismatchCount -eq 0 -and $trt11Output.ReferenceValidation.TensorComparisons[0].ReferenceSha256 -eq $trt11.referenceSha256 -and $trt11Output.OutputSha256 -eq $trt11.outputSha256) "$($trt11Output.ReferenceValidation.TensorComparisons[0].MismatchCount)/$($trt11Output.ReferenceValidation.TensorComparisons[0].MaximumAbsoluteError)"
Add-Check "trt11-strict-validation" ($trt11Validation.validationState -eq "tensor-rt-exec-report-ready" -and [int]$trt11Validation.failedBlockers -eq 0 -and @($trt11Validation.validationItems).Count -eq [int]$trt11.strictValidationCheckCount) "$($trt11Validation.validationState)/$($trt11Validation.failedBlockers)/$(@($trt11Validation.validationItems).Count)"
Add-Check "trt11-historical-probe-retained" ($trt11.historicalDependencyProbe.state -eq "dependency-probe-only" -and (Test-Path -LiteralPath (Join-Path $root $trt11.historicalDependencyProbe.reportPath) -PathType Leaf)) "$($trt11.historicalDependencyProbe.state)/$($trt11.historicalDependencyProbe.reportPath)"
Add-Check "boundary-local" ([bool]$evidence.proofBoundary.isLocalSourceTreeRefitLifecycleEvidence) ([string]$evidence.proofBoundary.isLocalSourceTreeRefitLifecycleEvidence)
Add-Check "boundary-no-accuracy" (-not [bool]$evidence.proofBoundary.isModelAccuracyProof) ([string]$evidence.proofBoundary.isModelAccuracyProof)
Add-Check "boundary-local-persistence" ([bool]$evidence.proofBoundary.isRefittedPlanPersistenceProof) ([string]$evidence.proofBoundary.isRefittedPlanPersistenceProof)
Add-Check "boundary-no-package" (-not [bool]$evidence.proofBoundary.isPackageConsumerRuntimeProof) ([string]$evidence.proofBoundary.isPackageConsumerRuntimeProof)
Add-Check "boundary-no-publish" (-not [bool]$evidence.proofBoundary.canPublishPublicly -and -not [bool]$evidence.proofBoundary.publicReleaseSideEffectsExecuted) "$($evidence.proofBoundary.canPublishPublicly)/$($evidence.proofBoundary.publicReleaseSideEffectsExecuted)"

$failed = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "trtexec-onnx-refit-lifecycle-validation.v1"
  strict = [bool]$Strict
  checkCount = $checks.Count
  passedCount = $checks.Count - $failed.Count
  failureCount = $failed.Count
  checks = $checks
}
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $validationPath -Encoding utf8

$lines = @(
  "# TensorRtExec ONNX Refit Lifecycle Validation",
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
  $lines += "| ``$($check.id)`` | ``$($check.passed)`` | ``$($check.actual)`` |"
}
$lines | Set-Content -LiteralPath $validationMarkdownPath -Encoding utf8

Write-Host "TensorRtExec ONNX refit lifecycle evidence: $($checks.Count - $failed.Count)/$($checks.Count) checks passed."
if ($Strict -and $failed.Count -gt 0) {
  throw "TensorRtExec ONNX refit lifecycle evidence validation failed: $($failed.id -join ', ')"
}
