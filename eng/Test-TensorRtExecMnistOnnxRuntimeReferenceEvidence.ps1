[CmdletBinding()]
param(
  [switch]$Strict,
  [switch]$RequireRuntimeArtifacts,
  [string]$GlobalPackageRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else { $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot) }

if ([string]::IsNullOrWhiteSpace($GlobalPackageRoot)) {
  $GlobalPackageRoot = Join-Path $env:USERPROFILE ".nuget\packages"
}
else { $GlobalPackageRoot = [IO.Path]::GetFullPath($GlobalPackageRoot) }

$evidencePath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-onnxruntime-reference-evidence.json"
$validationPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-onnxruntime-reference-validation.json"
$markdownPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-onnxruntime-reference-validation.md"
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

function Test-FiniteNonNegativeFloat {
  param([AllowNull()][object]$Value)
  $parsed = [single]0
  $ok = [single]::TryParse(
    [string]$Value,
    [Globalization.NumberStyles]::Float,
    [Globalization.CultureInfo]::InvariantCulture,
    [ref]$parsed)
  return $ok -and [single]::IsFinite($parsed) -and $parsed -ge [single]0
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

function Add-RepositoryFileHashCheck {
  param([string]$Id, [string]$Path, [string]$ExpectedSha256)
  $fullPath = Resolve-RepositoryPath -Path $Path
  $exists = Test-Path -LiteralPath $fullPath -PathType Leaf
  $actual = if ($exists) { (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { "missing" }
  Add-Check $Id ($exists -and $actual -eq $ExpectedSha256) "$Path/$actual"
}

$runtime = $evidence.runtime
$model = $evidence.model
$inputArtifact = $evidence.input
$reference = $evidence.reference
$comparison = $evidence.tensorRtComparison
$execution = $evidence.execution
$owner = $evidence.ownerReview
$boundary = $evidence.proofBoundary
$packages = @($runtime.packages)

Add-Check "schema" ($evidence.schemaVersion -eq "tensorrtexec-mnist-onnxruntime-reference-evidence.v1") $evidence.schemaVersion
Add-Check "state" ($evidence.state -eq "independent-onnxruntime-cpu-reference-runtime-passed-owner-review-required") $evidence.state
Add-Check "classification" ($evidence.evidenceClassification -eq "independent-framework-reference-candidate-runtime") $evidence.evidenceClassification
Add-Check "runtime-identity" ($runtime.name -eq "ONNX Runtime" -and $runtime.version -eq "1.23.2" -and $runtime.requestedProvider -eq "CPUExecutionProvider") "$($runtime.name)/$($runtime.version)/$($runtime.requestedProvider)"
Add-Check "provider-available" (@($runtime.availableProviders) -contains "CPUExecutionProvider") (@($runtime.availableProviders) -join ",")
Add-Check "provider-profile-only-cpu" ([bool]$runtime.providerValidated -and (@($runtime.profileProviders) -join ",") -eq "CPUExecutionProvider") "$($runtime.providerValidated)/$(@($runtime.profileProviders) -join ',')"
Add-Check "runtime-repository-commit" ($runtime.repositoryCommit -eq "a83fc4d58cb48eb68890dd689f94f28288cf2278") $runtime.repositoryCommit
Add-Check "package-count" ($packages.Count -eq 4) $packages.Count
Add-Check "package-inventory" ((@($packages.id | Sort-Object) -join ",") -eq "microsoft.ml.onnxruntime,microsoft.ml.onnxruntime.managed,system.memory,system.numerics.tensors") (@($packages.id | Sort-Object) -join ",")
Add-Check "package-versions" ((@($packages | Where-Object { $_.id -like 'microsoft.ml.onnxruntime*' -and $_.version -ne '1.23.2' }).Count -eq 0) -and (@($packages | Where-Object { $_.id -eq 'system.numerics.tensors' -and $_.version -eq '9.0.0' }).Count -eq 1) -and (@($packages | Where-Object { $_.id -eq 'system.memory' -and $_.version -eq '4.5.5' }).Count -eq 1)) (@($packages | ForEach-Object { "$($_.id)/$($_.version)" }) -join ";")
Add-Check "package-hashes" (@($packages | Where-Object { -not (Test-Sha256 $_.sha256) -or [int64]$_.length -le 0 }).Count -eq 0) (@($packages | ForEach-Object sha256) -join ",")
Add-Check "runtime-binary-hashes" ((Test-Sha256 $runtime.managedAssemblySha256) -and (Test-Sha256 $runtime.nativeLibrarySha256)) "$($runtime.managedAssemblySha256)/$($runtime.nativeLibrarySha256)"
Add-Check "model-contract" ($model.sha256 -eq "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf" -and -not [bool]$model.ownerRedistributionApproved) "$($model.sha256)/$($model.ownerRedistributionApproved)"
Add-Check "model-source-boundary" ((Test-Sha256 $model.sourceReadmeSha256) -and (Test-Sha256 $model.licenseReadmeSha256) -and $model.sourceReadmePath -notmatch '^[A-Za-z]:[\/]' -and $model.licenseReadmePath -notmatch '^[A-Za-z]:[\/]') "$($model.sourceReadmeSha256)/$($model.licenseReadmeSha256)"
Add-Check "input-contract" ($inputArtifact.sha256 -eq "81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564" -and $inputArtifact.tensorName -eq "Input3" -and (@($inputArtifact.shape) -join ",") -eq "1,1,28,28" -and [int]$inputArtifact.elementCount -eq 784) "$($inputArtifact.sha256)/$($inputArtifact.tensorName)/$(@($inputArtifact.shape) -join ',')"
Add-Check "reference-identity" ($reference.sha256 -eq "1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571" -and $reference.tensorName -eq "Plus214_Output_0" -and (@($reference.shape) -join ",") -eq "1,10" -and [int]$reference.elementCount -eq 10) "$($reference.sha256)/$($reference.tensorName)/$(@($reference.shape) -join ',')"
Add-Check "reference-source-boundary" ($reference.sourceClassification -eq "onnxruntime-cpu-1.23.2-derived-unreviewed" -and [bool]$reference.deterministicOutput -and [int]$reference.predictedIndex -eq 7) "$($reference.sourceClassification)/$($reference.deterministicOutput)/$($reference.predictedIndex)"
Add-Check "reference-artifact-hashes" ((Test-Sha256 $reference.sidecarSha256) -and (Test-Sha256 $reference.rawOutputSha256)) "$($reference.sidecarSha256)/$($reference.rawOutputSha256)"
Add-Check "comparison-reference" ($comparison.referenceSha256 -eq "07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef") $comparison.referenceSha256
Add-Check "comparison-result" ([bool]$comparison.passed -and [int]$comparison.comparedElementCount -eq 10 -and [int]$comparison.mismatchCount -eq 0 -and [int]$comparison.firstMismatchIndex -eq -1) "$($comparison.passed)/$($comparison.comparedElementCount)/$($comparison.mismatchCount)/$($comparison.firstMismatchIndex)"
Add-Check "comparison-errors" ((Test-FiniteNonNegativeFloat $comparison.maximumAbsoluteError) -and (Test-FiniteNonNegativeFloat $comparison.maximumRelativeError) -and [single]$comparison.maximumAbsoluteError -le [single]$comparison.absoluteTolerance -and [single]$comparison.maximumRelativeError -le [single]$comparison.relativeTolerance) "$($comparison.maximumAbsoluteError)/$($comparison.maximumRelativeError)/$($comparison.absoluteTolerance)/$($comparison.relativeTolerance)"
Add-Check "offline-restore-boundary" ([bool]$execution.remoteSourcesCleared -and [bool]$execution.sourcePackagesCopiedToTemporaryEdriveFeed -and [bool]$execution.sourcePackageCacheReadOnly -and [bool]$execution.isolatedRestoreCache) "$($execution.remoteSourcesCleared)/$($execution.sourcePackagesCopiedToTemporaryEdriveFeed)/$($execution.sourcePackageCacheReadOnly)/$($execution.isolatedRestoreCache)"
Add-Check "workspace-cleanup" (-not [bool]$execution.workspaceOnSystemDrive -and [bool]$execution.workspaceRemovedAfterValidation) "$($execution.workspaceOnSystemDrive)/$($execution.workspaceRemovedAfterValidation)"
Add-Check "process-exits" ([int]$execution.restoreExitCode -eq 0 -and [int]$execution.buildExitCode -eq 0 -and [int]$execution.runtimeExitCode -eq 0) "$($execution.restoreExitCode)/$($execution.buildExitCode)/$($execution.runtimeExitCode)"
Add-Check "source-and-log-hashes" ((Test-Sha256 $execution.projectSha256) -and (Test-Sha256 $execution.programSha256) -and (Test-Sha256 $execution.stdoutSha256) -and (Test-Sha256 $execution.stderrSha256) -and (Test-Sha256 $execution.runReportSha256) -and (Test-Sha256 $execution.profileSha256)) "$($execution.projectSha256)/$($execution.programSha256)/$($execution.runReportSha256)/$($execution.profileSha256)"
Add-Check "relative-runtime-artifact-paths" (@($execution.stdoutPath,$execution.stderrPath,$execution.runReportPath,$execution.profilePath,$reference.path,$reference.sidecarPath,$reference.rawOutputPath,$comparison.referencePath | Where-Object { $_ -match '^[A-Za-z]:[\/]' }).Count -eq 0) "relative-only"
Add-Check "owner-review-open" ($owner.status -eq "not-provided" -and -not [bool]$owner.acceptedAsGoldenReference -and -not [bool]$owner.acceptedForRepositoryRedistribution -and -not [bool]$owner.canPromoteRealModelRuntime) "$($owner.status)/$($owner.acceptedAsGoldenReference)/$($owner.acceptedForRepositoryRedistribution)/$($owner.canPromoteRealModelRuntime)"
Add-Check "independent-execution-boundary" ([bool]$boundary.independentFromTensorRtExecution -and [bool]$boundary.independentFrameworkReferenceCandidate -and -not [bool]$boundary.ownerReviewedGolden) "$($boundary.independentFromTensorRtExecution)/$($boundary.independentFrameworkReferenceCandidate)/$($boundary.ownerReviewedGolden)"
Add-Check "no-release-promotion" (-not [bool]$boundary.repositoryRedistributionApproved -and -not [bool]$boundary.publicPackageProof -and -not [bool]$boundary.postPublishProof -and -not [bool]$boundary.canPublishPublicly -and -not [bool]$boundary.canCloseReleaseIssue) "$($boundary.repositoryRedistributionApproved)/$($boundary.publicPackageProof)/$($boundary.postPublishProof)/$($boundary.canPublishPublicly)/$($boundary.canCloseReleaseIssue)"
Add-Check "proof-statement" ($boundary.statement -match "independent" -and $boundary.statement -match "Owner" -and $boundary.statement -match "not accepted real-model") $boundary.statement
Add-Check "path-free-compact-evidence" ($evidenceText -notmatch '(?i)[A-Z]:[\\/]') "absolute-windows-path-present=$($evidenceText -match '(?i)[A-Z]:[\\/]')"

$referencePath = Resolve-RepositoryPath -Path $reference.path
$sidecarPath = Resolve-RepositoryPath -Path $reference.sidecarPath
$referenceDocument = Get-Content -LiteralPath $referencePath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 20
$sidecarDocument = Get-Content -LiteralPath $sidecarPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
Add-Check "reference-file-contract" ([int]$referenceDocument.schemaVersion -eq 1 -and $referenceDocument.tensorName -eq $reference.tensorName -and (@($referenceDocument.shape) -join ",") -eq "1,10" -and @($referenceDocument.values).Count -eq 10 -and @($referenceDocument | Select-Object -ExpandProperty values | Where-Object { -not [single]::IsFinite([single]$_) }).Count -eq 0 -and $referenceDocument.sourceClassification -eq $reference.sourceClassification) "$($referenceDocument.schemaVersion)/$($referenceDocument.tensorName)/$(@($referenceDocument.shape) -join ',')/$(@($referenceDocument.values).Count)/$($referenceDocument.sourceClassification)"
Add-Check "sidecar-contract" ($sidecarDocument.schemaVersion -eq "tensorrtexec-mnist-onnxruntime-reference-sidecar.v1" -and $sidecarDocument.state -eq "independent-onnxruntime-cpu-reference-candidate-owner-review-required" -and $sidecarDocument.onnxRuntime.version -eq "1.23.2" -and (@($sidecarDocument.onnxRuntime.profileProviders) -join ",") -eq "CPUExecutionProvider") "$($sidecarDocument.schemaVersion)/$($sidecarDocument.state)/$($sidecarDocument.onnxRuntime.version)/$(@($sidecarDocument.onnxRuntime.profileProviders) -join ',')"
Add-Check "sidecar-cross-check" ($sidecarDocument.output.referenceSha256 -eq $reference.sha256 -and $sidecarDocument.output.rawSha256 -eq $reference.rawOutputSha256 -and $sidecarDocument.tensorRtComparison.referenceSha256 -eq $comparison.referenceSha256 -and [bool]$sidecarDocument.tensorRtComparison.passed) "$($sidecarDocument.output.referenceSha256)/$($sidecarDocument.output.rawSha256)/$($sidecarDocument.tensorRtComparison.referenceSha256)/$($sidecarDocument.tensorRtComparison.passed)"
Add-RepositoryFileHashCheck "reference-file-hash" $reference.path $reference.sha256
Add-RepositoryFileHashCheck "sidecar-file-hash" $reference.sidecarPath $reference.sidecarSha256

$priorPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-reference-validation-evidence.json"
$prior = Get-Content -LiteralPath $priorPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
Add-Check "prior-mnist-cross-check" ($prior.model.sha256 -eq $model.sha256 -and $prior.input.tensorSha256 -eq $inputArtifact.sha256 -and $prior.reference.sha256 -eq $comparison.referenceSha256) "$($prior.model.sha256)/$($prior.input.tensorSha256)/$($prior.reference.sha256)"

if ($RequireRuntimeArtifacts) {
  Add-RepositoryFileHashCheck "model-file-hash" $model.path $model.sha256
  Add-RepositoryFileHashCheck "model-source-readme-hash" $model.sourceReadmePath $model.sourceReadmeSha256
  Add-RepositoryFileHashCheck "model-license-readme-hash" $model.licenseReadmePath $model.licenseReadmeSha256
  Add-RepositoryFileHashCheck "input-file-hash" $inputArtifact.path $inputArtifact.sha256
  Add-RepositoryFileHashCheck "tensorrt-reference-file-hash" $comparison.referencePath $comparison.referenceSha256
  Add-RepositoryFileHashCheck "raw-output-file-hash" $reference.rawOutputPath $reference.rawOutputSha256
  Add-RepositoryFileHashCheck "stdout-file-hash" $execution.stdoutPath $execution.stdoutSha256
  Add-RepositoryFileHashCheck "stderr-file-hash" $execution.stderrPath $execution.stderrSha256
  Add-RepositoryFileHashCheck "run-report-file-hash" $execution.runReportPath $execution.runReportSha256
  Add-RepositoryFileHashCheck "profile-file-hash" $execution.profilePath $execution.profileSha256

  foreach ($package in $packages) {
    $nupkgPath = Join-Path $GlobalPackageRoot "$($package.id)\$($package.version)\$($package.id).$($package.version).nupkg"
    $exists = Test-Path -LiteralPath $nupkgPath -PathType Leaf
    $actual = if ($exists) { (Get-FileHash -LiteralPath $nupkgPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { "missing" }
    Add-Check "cached-package-$($package.id)" ($exists -and $actual -eq $package.sha256 -and (Get-Item -LiteralPath $nupkgPath).Length -eq [int64]$package.length) "$($package.version)/$actual"
  }
}

$failures = @($checks | Where-Object { -not $_.passed })
$validation = [ordered]@{
  schemaVersion = "tensorrtexec-mnist-onnxruntime-reference-validation.v1"
  strict = [bool]$Strict
  runtimeArtifactChecksRequired = [bool]$RequireRuntimeArtifacts
  checkCount = $checks.Count
  passedCount = $checks.Count - $failures.Count
  failureCount = $failures.Count
  checks = @($checks)
}
$validation | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $validationPath -Encoding utf8

$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# TensorRtExec MNIST ONNX Runtime Reference Validation")
$lines.Add("")
$lines.Add("- Strict: ``$([bool]$Strict)``")
$lines.Add("- Runtime artifact checks required: ``$([bool]$RequireRuntimeArtifacts)``")
$lines.Add("- Passed: ``$($validation.passedCount)/$($validation.checkCount)``")
$lines.Add("- Failures: ``$($validation.failureCount)``")
$lines.Add("")
$lines.Add("| Check | Passed | Actual |")
$lines.Add("| --- | ---: | --- |")
foreach ($check in $checks) {
  $actual = ([string]$check.actual).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
  $lines.Add("| $($check.id) | $($check.passed) | $actual |")
}
$lines.Add("")
$lines.Add("CPUExecutionProvider profiling proves an execution path independent from TensorRT. It does not supply Owner model/license/redistribution/golden acceptance, public-package proof, post-publish proof, or release authorization.")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "TensorRtExec MNIST ONNX Runtime reference evidence: $($validation.passedCount)/$($validation.checkCount) checks passed."
if ($Strict -and $failures.Count -gt 0) {
  throw "TensorRtExec MNIST ONNX Runtime reference evidence failed: $(@($failures.id) -join ', ')"
}
