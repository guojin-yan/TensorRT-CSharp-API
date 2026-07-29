[CmdletBinding()]
param(
  [switch]$Strict,
  [switch]$RequireRuntimeArtifacts,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else {
  $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
}

$evidencePath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-reference-validation-evidence.json"
$validationPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-reference-validation.json"
$validationMarkdownPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-reference-validation.md"
$evidenceText = Get-Content -LiteralPath $evidencePath -Raw -Encoding utf8
$evidence = $evidenceText | ConvertFrom-Json -Depth 100
$checks = [Collections.Generic.List[object]]::new()

function Add-Check {
  param([string]$Id, [bool]$Passed, [AllowNull()][object]$Actual)
  $checks.Add([pscustomobject][ordered]@{ id = $Id; passed = $Passed; actual = [string]$Actual })
}

function Test-Sha256 {
  param([AllowNull()][string]$Value)
  return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -match '^[0-9a-f]{64}$'
}

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { throw "Evidence paths must be repository-relative: $Path" }
  $fullPath = [IO.Path]::GetFullPath((Join-Path $RepositoryRoot ($Path -replace '/', '\')))
  $rootPrefix = $RepositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
  if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Evidence path escapes the repository: $Path"
  }
  return $fullPath
}

function Add-FileHashCheck {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256
  )
  $fullPath = Resolve-RepositoryPath $Path
  $exists = Test-Path -LiteralPath $fullPath -PathType Leaf
  $actual = if ($exists) { (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { "missing" }
  Add-Check $Id ($exists -and $actual -eq $ExpectedSha256) "$Path/$actual"
}

function Test-ValidatedRunShape {
  param([Parameter(Mandatory = $true)][object]$Run, [Parameter(Mandatory = $true)][string]$ExpectedState)
  return $Run.state -eq $ExpectedState -and
    [bool]($Run.success) -and [bool]($Run.inferenceRan) -and [bool]($Run.outputMatch) -and [bool]($Run.outputValidated) -and
    -not [bool]($Run.identityOutputMatch) -and $Run.genericToolProofClassification -eq "synthetic-input-runtime" -and
    [int]($Run.inputTensorCount) -eq 1 -and [int]($Run.outputTensorCount) -eq 1 -and
    [int]($Run.comparedElementCount) -eq 10 -and [int]($Run.mismatchCount) -eq 0 -and [int]($Run.firstMismatchIndex) -eq -1
}

$model = $evidence.model
$inputArtifact = $evidence.input
$referenceArtifact = $evidence.reference
$buildRun = $evidence.sourceTreeBuild
$loadRun = $evidence.independentLoadEngine
$consumer = $evidence.localPackageConsumer
$owner = $evidence.ownerReview
$boundary = $evidence.proofBoundary

Add-Check "schema" ($evidence.schemaVersion -eq "tensorrtexec-mnist-reference-validation-evidence.v1") $evidence.schemaVersion
Add-Check "state" ($evidence.state -eq "mnist-reference-regression-runtime-passed-owner-review-required") $evidence.state
Add-Check "classification" ($evidence.evidenceClassification -eq "real-model-reference-candidate-runtime") $evidence.evidenceClassification
Add-Check "runtime-key" ($evidence.runtime.runtimeKey -eq "win-x64-trt10.11-cuda12.9-cudnn9.22" -and $evidence.runtime.tensorRtVersion -eq "10.11.0" -and $evidence.runtime.cudaToolkitVersion -eq "12.9") "$($evidence.runtime.runtimeKey)/$($evidence.runtime.tensorRtVersion)/$($evidence.runtime.cudaToolkitVersion)"
Add-Check "model-identity" ((Test-Sha256 $model.sha256) -and $model.statedSource -match "ONNX Model Zoo" -and $model.sourceReadmePath -notmatch '^[A-Za-z]:[\\/]' -and $model.licenseReadmePath -notmatch '^[A-Za-z]:[\\/]') "$($model.sha256)/$($model.statedSource)"
Add-Check "license-review-boundary" ($model.licenseReviewState -eq "owner-review-required" -and -not [bool]($model.redistributionApproved)) "$($model.licenseReviewState)/$($model.redistributionApproved)"
Add-Check "input-contract" ((Test-Sha256 $inputArtifact.sourceAssetSha256) -and (Test-Sha256 $inputArtifact.tensorSha256) -and $inputArtifact.preprocessing -eq "float32 1-pixel/255" -and $inputArtifact.tensorName -eq "Input3" -and (@($inputArtifact.shape) -join ',') -eq "1,1,28,28" -and [int]($inputArtifact.elementCount) -eq 784) "$($inputArtifact.tensorName)/$(@($inputArtifact.shape) -join ',')/$($inputArtifact.elementCount)"
Add-Check "reference-identity" ((Test-Sha256 $referenceArtifact.sha256) -and (Test-Sha256 $referenceArtifact.sidecarSha256) -and $referenceArtifact.tensorName -eq "Plus214_Output_0" -and (@($referenceArtifact.shape) -join ',') -eq "1,10" -and [int]($referenceArtifact.elementCount) -eq 10) "$($referenceArtifact.tensorName)/$(@($referenceArtifact.shape) -join ',')/$($referenceArtifact.elementCount)"
Add-Check "reference-source-boundary" ($referenceArtifact.sourceClassification -eq "repository-mnist-runtime-output-derived-unreviewed" -and -not [bool]($referenceArtifact.independentFrameworkGolden) -and -not [bool]($referenceArtifact.ownerReviewedGolden)) "$($referenceArtifact.sourceClassification)/$($referenceArtifact.independentFrameworkGolden)/$($referenceArtifact.ownerReviewedGolden)"
Add-Check "reference-policy" (([double]($referenceArtifact.absoluteTolerance) -eq 0.0001) -and ([double]($referenceArtifact.relativeTolerance) -eq 0.0001) -and $referenceArtifact.nanPolicy -eq "reject" -and $referenceArtifact.infinityPolicy -eq "exact") "$($referenceArtifact.absoluteTolerance)/$($referenceArtifact.relativeTolerance)/$($referenceArtifact.nanPolicy)/$($referenceArtifact.infinityPolicy)"
Add-Check "source-tree-build" (Test-ValidatedRunShape $buildRun "external-onnx-reference-validated-runtime") "$($buildRun.state)/$($buildRun.outputValidated)/$($buildRun.mismatchCount)"
Add-Check "independent-load-engine" (Test-ValidatedRunShape $loadRun "load-engine-reference-validated-runtime") "$($loadRun.state)/$($loadRun.outputValidated)/$($loadRun.mismatchCount)"
Add-Check "build-load-engine-identity" ($buildRun.engineSha256 -eq $loadRun.engineSha256 -and (Test-Sha256 $buildRun.engineSha256)) "$($buildRun.engineSha256)/$($loadRun.engineSha256)"
Add-Check "build-load-raw-output" ($buildRun.rawOutputSha256 -eq $loadRun.rawOutputSha256 -and (Test-Sha256 $buildRun.rawOutputSha256)) "$($buildRun.rawOutputSha256)/$($loadRun.rawOutputSha256)"
Add-Check "build-error-within-policy" (([double]($buildRun.maximumAbsoluteError) -le [double]($referenceArtifact.absoluteTolerance)) -and ([double]($buildRun.maximumRelativeError) -le [double]($referenceArtifact.relativeTolerance))) "$($buildRun.maximumAbsoluteError)/$($buildRun.maximumRelativeError)"
Add-Check "load-error-within-policy" (([double]($loadRun.maximumAbsoluteError) -le [double]($referenceArtifact.absoluteTolerance)) -and ([double]($loadRun.maximumRelativeError) -le [double]($referenceArtifact.relativeTolerance))) "$($loadRun.maximumAbsoluteError)/$($loadRun.maximumRelativeError)"
Add-Check "runtime-artifact-hashes" ((Test-Sha256 $buildRun.outputSha256) -and (Test-Sha256 $buildRun.reportSha256) -and (Test-Sha256 $loadRun.outputSha256) -and (Test-Sha256 $loadRun.reportSha256)) "$($buildRun.outputSha256)/$($buildRun.reportSha256)/$($loadRun.outputSha256)/$($loadRun.reportSha256)"
Add-Check "consumer-state" ($consumer.state -eq "local-package-consumer-refitted-plan-runtime-passed" -and $consumer.classification -eq "local-package-consumer-refitted-plan-runtime" -and [bool]$consumer.runtimePassed) "$($consumer.state)/$($consumer.classification)/$($consumer.runtimePassed)"
Add-Check "consumer-isolation" ([bool]$consumer.packageReferenceOnly -and -not [bool]$consumer.projectReference -and -not [bool]$consumer.publicFeedEnabled -and [bool]$consumer.workspaceRemovedAfterValidation) "$($consumer.packageReferenceOnly)/$($consumer.projectReference)/$($consumer.publicFeedEnabled)/$($consumer.workspaceRemovedAfterValidation)"
Add-Check "consumer-reference" ([bool]$consumer.outputExactMatch -and [bool]$consumer.referenceValidationCompleted -and [bool]$consumer.referenceValidationPassed -and [int]$consumer.referenceComparedElementCount -eq 10 -and [int]$consumer.referenceMismatchCount -eq 0 -and [int]$consumer.referenceFirstMismatchIndex -eq -1 -and $consumer.referenceSha256 -eq $referenceArtifact.sha256) "$($consumer.outputExactMatch)/$($consumer.referenceValidationPassed)/$($consumer.referenceComparedElementCount)/$($consumer.referenceMismatchCount)/$($consumer.referenceSha256)"
Add-Check "consumer-not-public-proof" (-not [bool]$consumer.isPublicPackageProof) $consumer.isPublicPackageProof
Add-Check "owner-review-open" ($owner.status -eq "not-provided" -and -not [bool]$owner.acceptedAsGoldenReference -and -not [bool]$owner.acceptedForRepositoryRedistribution -and -not [bool]$owner.canPromoteRealModelRuntime) "$($owner.status)/$($owner.acceptedAsGoldenReference)/$($owner.acceptedForRepositoryRedistribution)/$($owner.canPromoteRealModelRuntime)"
Add-Check "proof-boundary" ([bool]$boundary.usesExistingRealOnnxModel -and [bool]$boundary.provesStructuredReferenceRegressionAcrossBuildLoadAndLocalPackageConsumer -and -not [bool]$boundary.isIndependentNumericalGoldenProof -and -not [bool]$boundary.isOwnerAcceptedRealModelProof -and -not [bool]$boundary.isPublicPackageConsumerProof -and -not [bool]$boundary.isPostPublishProof -and -not [bool]$boundary.canPublishPublicly -and -not [bool]$boundary.canCloseReleaseIssue) $boundary.statement
Add-Check "path-free-compact-evidence" ($evidenceText -notmatch '(?i)[A-Z]:\\') "absolute-windows-path-present=$($evidenceText -match '(?i)[A-Z]:\\')"

$referencePath = Resolve-RepositoryPath $referenceArtifact.path
$sidecarPath = Resolve-RepositoryPath $referenceArtifact.sidecarPath
$packageEvidencePath = Resolve-RepositoryPath $consumer.evidencePath
Add-FileHashCheck "reference-file-hash" $referenceArtifact.path $referenceArtifact.sha256
Add-FileHashCheck "reference-sidecar-hash" $referenceArtifact.sidecarPath $referenceArtifact.sidecarSha256

$referenceJson = Get-Content -LiteralPath $referencePath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 20
$sidecarJson = Get-Content -LiteralPath $sidecarPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 30
$packageEvidence = Get-Content -LiteralPath $packageEvidencePath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
Add-Check "reference-file-contract" ([int]$referenceJson.schemaVersion -eq 1 -and $referenceJson.tensorName -eq $referenceArtifact.tensorName -and (@($referenceJson.shape) -join ',') -eq "1,10" -and @($referenceJson.values).Count -eq 10 -and $referenceJson.sourceClassification -eq $referenceArtifact.sourceClassification) "$($referenceJson.schemaVersion)/$($referenceJson.tensorName)/$(@($referenceJson.shape) -join ',')/$(@($referenceJson.values).Count)/$($referenceJson.sourceClassification)"
Add-Check "sidecar-owner-boundary" ($sidecarJson.proofClassification -eq "real-model-reference-candidate-owner-review-required" -and $sidecarJson.ownerReview.status -eq "not-provided" -and -not [bool]$sidecarJson.ownerReview.acceptedAsGoldenReference -and -not [bool]$sidecarJson.canPromoteRealModelRuntime -and -not [bool]$sidecarJson.canPromotePackageConsumerRuntime) "$($sidecarJson.proofClassification)/$($sidecarJson.ownerReview.status)/$($sidecarJson.canPromoteRealModelRuntime)/$($sidecarJson.canPromotePackageConsumerRuntime)"
Add-Check "package-evidence-cross-check" ($packageEvidence.state -eq $consumer.state -and [bool]$packageEvidence.execution.runtimePassed -and [bool]$packageEvidence.runtime.referenceValidationPassed -and $packageEvidence.artifacts.copiedReferenceSha256 -eq $referenceArtifact.sha256 -and $packageEvidence.artifacts.outputSha256 -eq $consumer.outputSha256) "$($packageEvidence.state)/$($packageEvidence.execution.runtimePassed)/$($packageEvidence.runtime.referenceValidationPassed)/$($packageEvidence.artifacts.copiedReferenceSha256)/$($packageEvidence.artifacts.outputSha256)"

if ($RequireRuntimeArtifacts.IsPresent) {
  Add-FileHashCheck "model-file-hash" $model.path $model.sha256
  Add-FileHashCheck "model-source-readme-hash" $model.sourceReadmePath $model.sourceReadmeSha256
  Add-FileHashCheck "model-license-readme-hash" $model.licenseReadmePath $model.licenseReadmeSha256
  Add-FileHashCheck "source-input-file-hash" $inputArtifact.sourceAsset $inputArtifact.sourceAssetSha256
  Add-FileHashCheck "input-tensor-file-hash" $inputArtifact.tensorPath $inputArtifact.tensorSha256
  Add-FileHashCheck "source-output-file-hash" $referenceArtifact.sourceOutputPath $referenceArtifact.sourceOutputSha256
  Add-FileHashCheck "build-engine-file-hash" $buildRun.enginePath $buildRun.engineSha256
  Add-FileHashCheck "build-output-file-hash" $buildRun.outputPath $buildRun.outputSha256
  Add-FileHashCheck "build-report-file-hash" $buildRun.reportPath $buildRun.reportSha256
  Add-FileHashCheck "build-raw-file-hash" $buildRun.rawOutputPath $buildRun.rawOutputSha256
  Add-FileHashCheck "load-output-file-hash" $loadRun.outputPath $loadRun.outputSha256
  Add-FileHashCheck "load-report-file-hash" $loadRun.reportPath $loadRun.reportSha256
  Add-FileHashCheck "load-raw-file-hash" $loadRun.rawOutputPath $loadRun.rawOutputSha256

  $buildOutput = Get-Content -LiteralPath (Resolve-RepositoryPath $buildRun.outputPath) -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
  $buildReport = Get-Content -LiteralPath (Resolve-RepositoryPath $buildRun.reportPath) -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
  $loadOutput = Get-Content -LiteralPath (Resolve-RepositoryPath $loadRun.outputPath) -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
  $loadReport = Get-Content -LiteralPath (Resolve-RepositoryPath $loadRun.reportPath) -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
  Add-Check "build-runtime-report-contract" ([bool]$buildReport.Success -and [bool]$buildReport.InferenceRan -and [bool]$buildReport.OutputValidated -and [bool]$buildOutput.OutputValidated -and [bool]$buildOutput.ReferenceValidation.Passed -and [int]$buildOutput.ReferenceValidation.TensorComparisons[0].MismatchCount -eq 0) "$($buildReport.Success)/$($buildReport.InferenceRan)/$($buildReport.OutputValidated)/$($buildOutput.OutputValidated)/$($buildOutput.ReferenceValidation.Passed)"
  Add-Check "load-runtime-report-contract" ([bool]$loadReport.Success -and [bool]$loadReport.InferenceRan -and [bool]$loadReport.OutputValidated -and [bool]$loadOutput.OutputValidated -and [bool]$loadOutput.ReferenceValidation.Passed -and [int]$loadOutput.ReferenceValidation.TensorComparisons[0].MismatchCount -eq 0) "$($loadReport.Success)/$($loadReport.InferenceRan)/$($loadReport.OutputValidated)/$($loadOutput.OutputValidated)/$($loadOutput.ReferenceValidation.Passed)"
}

$failed = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "tensorrtexec-mnist-reference-validation.v1"
  strict = [bool]$Strict
  runtimeArtifactChecksRequired = [bool]$RequireRuntimeArtifacts
  checkCount = $checks.Count
  passedCount = $checks.Count - $failed.Count
  failureCount = $failed.Count
  checks = @($checks)
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $validationPath -Encoding utf8

$lines = @(
  "# TensorRtExec MNIST Reference Validation",
  "",
  "- strict: ``$([bool]$Strict)``",
  "- runtime artifacts required: ``$([bool]$RequireRuntimeArtifacts)``",
  "- checks: ``$($checks.Count)``",
  "- passed: ``$($checks.Count - $failed.Count)``",
  "- failed: ``$($failed.Count)``",
  "",
  "| Check | Passed | Actual |",
  "| --- | --- | --- |"
)
foreach ($check in $checks) {
  $actual = ([string]$check.actual).Replace('|', '\|').Replace("`r", ' ').Replace("`n", ' ')
  $lines += "| ``$($check.id)`` | ``$($check.passed)`` | ``$actual`` |"
}
$lines | Set-Content -LiteralPath $validationMarkdownPath -Encoding utf8

Write-Host "TensorRtExec MNIST reference evidence: $($checks.Count - $failed.Count)/$($checks.Count) checks passed."
if ($Strict -and $failed.Count -gt 0) {
  throw "TensorRtExec MNIST reference evidence validation failed: $($failed.id -join ', ')"
}
