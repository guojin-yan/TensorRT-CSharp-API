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
Add-Check "trt11-dependency" ($evidence.tensorRt11.state -eq "dependency-probe-only" -and [bool]$evidence.tensorRt11.refitFromOnnxParseOnly) "$($evidence.tensorRt11.state)/$($evidence.tensorRt11.refitFromOnnxParseOnly)"
Add-Check "boundary-local" ([bool]$evidence.proofBoundary.isLocalSourceTreeRefitLifecycleEvidence) ([string]$evidence.proofBoundary.isLocalSourceTreeRefitLifecycleEvidence)
Add-Check "boundary-no-accuracy" (-not [bool]$evidence.proofBoundary.isModelAccuracyProof) ([string]$evidence.proofBoundary.isModelAccuracyProof)
Add-Check "boundary-no-persistence" (-not [bool]$evidence.proofBoundary.isRefittedPlanPersistenceProof) ([string]$evidence.proofBoundary.isRefittedPlanPersistenceProof)
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
