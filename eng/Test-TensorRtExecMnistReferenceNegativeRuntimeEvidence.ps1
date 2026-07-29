[CmdletBinding()]
param(
  [string]$EvidencePath,
  [string]$RuntimeArtifactDirectory,
  [string]$OutputPath,
  [string]$MarkdownPath,
  [switch]$RequireRuntimeArtifacts,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else { $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot) }
if ([string]::IsNullOrWhiteSpace($EvidencePath)) {
  $EvidencePath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-reference-negative-runtime-evidence.json"
}
else { $EvidencePath = [IO.Path]::GetFullPath($EvidencePath) }
if ([string]::IsNullOrWhiteSpace($RuntimeArtifactDirectory)) {
  $RuntimeArtifactDirectory = Join-Path $RepositoryRoot "artifacts\real-case\tensorrtexec-mnist-reference-negative-runtime"
}
else { $RuntimeArtifactDirectory = [IO.Path]::GetFullPath($RuntimeArtifactDirectory) }
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-reference-negative-runtime-validation.json"
}
else { $OutputPath = [IO.Path]::GetFullPath($OutputPath) }
if ([string]::IsNullOrWhiteSpace($MarkdownPath)) {
  $MarkdownPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\tensorrtexec-mnist-reference-negative-runtime-validation.md"
}
else { $MarkdownPath = [IO.Path]::GetFullPath($MarkdownPath) }

$utf8 = [Text.UTF8Encoding]::new($false)
$checks = [Collections.Generic.List[object]]::new()

function Add-Check {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][bool]$Passed,
    [AllowEmptyString()][string]$Actual = ""
  )
  $checks.Add([pscustomobject][ordered]@{ id = $Id; passed = $Passed; actual = $Actual }) | Out-Null
}

function Test-Sha256 {
  param([AllowEmptyString()][string]$Value)
  return $Value -match '^[0-9a-f]{64}$'
}

function Get-Sha256 {
  param([Parameter(Mandatory = $true)][string]$Path)
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
}

function Add-RepositoryFileHashCheck {
  param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256
  )
  $fullPath = Resolve-RepositoryPath -Path $Path
  $actual = if (Test-Path -LiteralPath $fullPath -PathType Leaf) { Get-Sha256 -Path $fullPath } else { "missing" }
  Add-Check $Id ($actual -eq $ExpectedSha256) "$Path/$actual"
}

function Get-MarkerValue {
  param(
    [string[]]$Lines,
    [Parameter(Mandatory = $true)][string]$Prefix
  )
  foreach ($line in @($Lines)) {
    if ([string]$line -like "$Prefix*") { return ([string]$line).Substring($Prefix.Length) }
  }
  return ""
}

if (-not (Test-Path -LiteralPath $EvidencePath -PathType Leaf)) {
  throw "Negative runtime evidence was not found: $EvidencePath"
}
$evidenceText = Get-Content -LiteralPath $EvidencePath -Raw -Encoding utf8
$evidence = $evidenceText | ConvertFrom-Json -Depth 100
$runtime = $evidence.runtime
$baseReference = $evidence.baseReference
$engine = $evidence.engine
$inputArtifact = $evidence.input
$consumer = $evidence.localPackageConsumer
$boundary = $evidence.proofBoundary
$cases = @($evidence.cases)
$expectedIds = @("name-mismatch", "shape-mismatch", "value-count-mismatch", "nan-reject", "infinity-reject")

Add-Check "schema" ($evidence.schemaVersion -eq "tensorrtexec-mnist-reference-negative-runtime-evidence.v1") $evidence.schemaVersion
Add-Check "state" ($evidence.state -eq "controlled-reference-negative-runtime-passed") $evidence.state
Add-Check "classification" ($evidence.evidenceClassification -eq "controlled-negative-runtime") $evidence.evidenceClassification
Add-Check "runtime-key" ($runtime.runtimeKey -eq "win-x64-trt10.11-cuda12.9-cudnn9.22" -and [int]$runtime.tensorRtLine -eq 10) "$($runtime.runtimeKey)/$($runtime.tensorRtLine)"
Add-Check "build-processes" ([int]$runtime.applicationBuildExitCode -eq 0 -and [int]$runtime.consumerRestoreExitCode -eq 0 -and [int]$runtime.consumerBuildExitCode -eq 0) "$($runtime.applicationBuildExitCode)/$($runtime.consumerRestoreExitCode)/$($runtime.consumerBuildExitCode)"
Add-Check "base-reference" ($baseReference.sha256 -eq "1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571" -and $baseReference.sourceClassification -eq "onnxruntime-cpu-1.23.2-derived-unreviewed" -and -not [bool]$baseReference.ownerReviewedGolden) "$($baseReference.sha256)/$($baseReference.sourceClassification)/$($baseReference.ownerReviewedGolden)"
Add-Check "engine-input-hashes" ($engine.sha256 -eq "14044b5d345a68bebe7b01c3a48ce10c1665bde40088fbbfd61ae236f1221a89" -and $inputArtifact.sha256 -eq "81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564") "$($engine.sha256)/$($inputArtifact.sha256)"
Add-Check "consumer-isolation" ([bool]$consumer.usesPackageReferenceOnly -and -not [bool]$consumer.usesProjectReference -and [bool]$consumer.remoteSourcesCleared -and [bool]$consumer.isolatedRestoreCache -and -not [bool]$consumer.workspaceOnSystemDrive -and [bool]$consumer.workspaceRemovedAfterValidation) "$($consumer.usesPackageReferenceOnly)/$($consumer.usesProjectReference)/$($consumer.remoteSourcesCleared)/$($consumer.isolatedRestoreCache)/$($consumer.workspaceOnSystemDrive)/$($consumer.workspaceRemovedAfterValidation)"
Add-Check "consumer-package-contract" ((Test-Sha256 $consumer.managedPackage.sha256) -and (Test-Sha256 $consumer.bridgePackage.sha256) -and [int64]$consumer.managedPackage.length -gt 0 -and [int64]$consumer.bridgePackage.length -gt 0 -and (Test-Sha256 $consumer.programSha256)) "$($consumer.managedPackage.id)/$($consumer.managedPackage.version)/$($consumer.bridgePackage.id)/$($consumer.bridgePackage.version)"
Add-Check "case-count" ([int]$evidence.caseCount -eq 5 -and $cases.Count -eq 5) "$($evidence.caseCount)/$($cases.Count)"
Add-Check "case-ids" ((@($cases | ForEach-Object id) -join ",") -eq ($expectedIds -join ",")) (@($cases | ForEach-Object id) -join ",")
Add-Check "fail-closed-counts" ([int]$evidence.sourceTreeFailClosedCount -eq 5 -and [int]$evidence.localPackageConsumerFailClosedCount -eq 5) "$($evidence.sourceTreeFailClosedCount)/$($evidence.localPackageConsumerFailClosedCount)"
Add-Check "path-free-compact-evidence" ($evidenceText -notmatch '(?i)[A-Z]:[\\/]') "absolute-windows-path-present=$($evidenceText -match '(?i)[A-Z]:[\\/]')"
Add-RepositoryFileHashCheck "base-reference-file-hash" $baseReference.path $baseReference.sha256

foreach ($case in $cases) {
  $id = [string]$case.id
  $source = $case.sourceTree
  $localConsumer = $case.localPackageConsumer
  $metadataCase = $id -in @("name-mismatch", "shape-mismatch", "value-count-mismatch")
  $expectedCompleted = -not $metadataCase
  $expectedCompared = if ($metadataCase) { 0 } else { 10 }
  $expectedMismatches = if ($metadataCase) { 0 } else { 1 }
  $expectedFirstMismatch = if ($metadataCase) { -1 } else { 0 }

  Add-Check "$id-source-execution" ([int]$source.exitCode -eq 2 -and [bool]$source.inferenceRan -and [bool]$source.outputCaptureAvailable -and -not [bool]$source.outputValidated) "$($source.exitCode)/$($source.inferenceRan)/$($source.outputCaptureAvailable)/$($source.outputValidated)"
  Add-Check "$id-source-validation" ([bool]$source.validationCompleted -eq $expectedCompleted -and -not [bool]$source.validationPassed -and [int]$source.comparedElementCount -eq $expectedCompared -and [int]$source.mismatchCount -eq $expectedMismatches -and [int]$source.firstMismatchIndex -eq $expectedFirstMismatch) "$($source.validationCompleted)/$($source.validationPassed)/$($source.comparedElementCount)/$($source.mismatchCount)/$($source.firstMismatchIndex)"
  Add-Check "$id-source-hashes" ($source.outputSha256 -eq "0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5" -and (Test-Sha256 $source.referenceSha256) -and (Test-Sha256 $source.reportSha256) -and (Test-Sha256 $source.outputArtifactSha256) -and (Test-Sha256 $source.stdoutSha256) -and (Test-Sha256 $source.stderrSha256)) "$($source.outputSha256)/$($source.referenceSha256)"
  Add-Check "$id-consumer-execution" ([int]$localConsumer.exitCode -eq 1 -and [bool]$localConsumer.enqueueCompleted -and [bool]$localConsumer.ownerScopeExited -and -not [bool]$localConsumer.outputValidated) "$($localConsumer.exitCode)/$($localConsumer.enqueueCompleted)/$($localConsumer.ownerScopeExited)/$($localConsumer.outputValidated)"
  Add-Check "$id-consumer-validation" ([bool]$localConsumer.validationCompleted -eq $expectedCompleted -and -not [bool]$localConsumer.validationPassed -and [int]$localConsumer.comparedElementCount -eq $expectedCompared -and [int]$localConsumer.mismatchCount -eq $expectedMismatches -and [int]$localConsumer.firstMismatchIndex -eq $expectedFirstMismatch) "$($localConsumer.validationCompleted)/$($localConsumer.validationPassed)/$($localConsumer.comparedElementCount)/$($localConsumer.mismatchCount)/$($localConsumer.firstMismatchIndex)"
  Add-Check "$id-consumer-hashes" ($localConsumer.outputSha256 -eq $source.outputSha256 -and $localConsumer.referenceSha256 -eq $source.referenceSha256 -and (Test-Sha256 $localConsumer.stdoutSha256) -and (Test-Sha256 $localConsumer.stderrSha256)) "$($localConsumer.outputSha256)/$($localConsumer.referenceSha256)"
  Add-Check "$id-diagnostic" ([string]$source.diagnostic -like "*$($case.expectedDiagnostic)*" -and [string]$localConsumer.diagnostic -like "*$($case.expectedDiagnostic)*") "$($source.diagnostic)/$($localConsumer.diagnostic)"

  if ($RequireRuntimeArtifacts) {
    $referencePath = Join-Path $RuntimeArtifactDirectory "references\$id.reference.json"
    $sourceRoot = Join-Path $RuntimeArtifactDirectory "source-tree\$id"
    $consumerRoot = Join-Path $RuntimeArtifactDirectory "package-consumer\$id"
    $sourceReportPath = Join-Path $sourceRoot "report.json"
    $sourceOutputPath = Join-Path $sourceRoot "output.json"
    $sourceRawPath = Join-Path $sourceRoot "output.raw"
    $sourceStdoutPath = Join-Path $sourceRoot "stdout.log"
    $sourceStderrPath = Join-Path $sourceRoot "stderr.log"
    $consumerRawPath = Join-Path $consumerRoot "output.raw"
    $consumerStdoutPath = Join-Path $consumerRoot "stdout.log"
    $consumerStderrPath = Join-Path $consumerRoot "stderr.log"

    $required = @($referencePath, $sourceReportPath, $sourceOutputPath, $sourceRawPath, $sourceStdoutPath, $sourceStderrPath, $consumerRawPath, $consumerStdoutPath, $consumerStderrPath)
    Add-Check "$id-runtime-files" (@($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -eq 0) (@($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }) -join ",")
    if (@($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -eq 0) {
      Add-Check "$id-runtime-file-hashes" ((Get-Sha256 $referencePath) -eq $source.referenceSha256 -and (Get-Sha256 $sourceReportPath) -eq $source.reportSha256 -and (Get-Sha256 $sourceOutputPath) -eq $source.outputArtifactSha256 -and (Get-Sha256 $sourceRawPath) -eq $source.outputSha256 -and (Get-Sha256 $sourceStdoutPath) -eq $source.stdoutSha256 -and (Get-Sha256 $sourceStderrPath) -eq $source.stderrSha256 -and (Get-Sha256 $consumerRawPath) -eq $localConsumer.outputSha256 -and (Get-Sha256 $consumerStdoutPath) -eq $localConsumer.stdoutSha256 -and (Get-Sha256 $consumerStderrPath) -eq $localConsumer.stderrSha256) "all-recorded-hashes-match"

      $sourceReport = Get-Content -LiteralPath $sourceReportPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
      $sourceOutput = Get-Content -LiteralPath $sourceOutputPath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
      $sourceLog = Get-Content -LiteralPath $sourceStdoutPath -Raw -Encoding utf8
      $consumerStdout = @(Get-Content -LiteralPath $consumerStdoutPath -Encoding utf8)
      $consumerStderr = Get-Content -LiteralPath $consumerStderrPath -Raw -Encoding utf8
      Add-Check "$id-source-runtime-contract" (-not [bool]$sourceReport.Success -and [bool]$sourceReport.InferenceRan -and -not [bool]$sourceReport.OutputValidated -and [bool]$sourceOutput.OutputCaptureAvailable -and -not [bool]$sourceOutput.OutputValidated -and $sourceLog.Contains("BoundedRuntime Attempted=True Succeeded=True", [StringComparison]::Ordinal)) "$($sourceReport.Success)/$($sourceReport.InferenceRan)/$($sourceReport.OutputValidated)/$($sourceOutput.OutputCaptureAvailable)/$($sourceOutput.OutputValidated)"
      Add-Check "$id-consumer-runtime-contract" ((Get-MarkerValue $consumerStdout "PackageReferenceOnly=") -eq "True" -and (Get-MarkerValue $consumerStdout "ProjectReference=") -eq "False" -and (Get-MarkerValue $consumerStdout "EnqueueCompleted=") -eq "True" -and (Get-MarkerValue $consumerStdout "OutputValidated=") -eq "False" -and (Get-MarkerValue $consumerStdout "OwnerScopeExited=") -eq "True" -and $consumerStderr.Contains("PackageConsumerRuntime=Failed", [StringComparison]::Ordinal)) "package-reference/enqueue/fail-closed/owner-exit"
    }
  }
}

Add-Check "proof-runtime-boundary" ([bool]$boundary.provesRealEnqueueBeforeReferenceRejection -and [bool]$boundary.provesSourceTreeFailClosedValidation -and [bool]$boundary.provesIsolatedLocalPackageConsumerFailClosedValidation) "$($boundary.provesRealEnqueueBeforeReferenceRejection)/$($boundary.provesSourceTreeFailClosedValidation)/$($boundary.provesIsolatedLocalPackageConsumerFailClosedValidation)"
Add-Check "proof-promotion-boundary" (-not [bool]$boundary.ownerReviewedGolden -and -not [bool]$boundary.publicPackageProof -and -not [bool]$boundary.postPublishProof -and -not [bool]$boundary.canPublishPublicly -and -not [bool]$boundary.canCloseReleaseIssue) "$($boundary.ownerReviewedGolden)/$($boundary.publicPackageProof)/$($boundary.postPublishProof)/$($boundary.canPublishPublicly)/$($boundary.canCloseReleaseIssue)"
Add-Check "proof-statement" ($boundary.statement -match "malformed" -and $boundary.statement -match "real TensorRT enqueue" -and $boundary.statement -match "Owner golden" -and $boundary.statement -match "not promote") $boundary.statement

$failed = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "tensorrtexec-mnist-reference-negative-runtime-validation.v1"
  strict = [bool]$Strict
  runtimeArtifactChecksRequired = [bool]$RequireRuntimeArtifacts
  checkCount = $checks.Count
  passedCount = $checks.Count - $failed.Count
  failureCount = $failed.Count
  checks = @($checks)
}
New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null
[IO.File]::WriteAllText($OutputPath, ($result | ConvertTo-Json -Depth 20) + [Environment]::NewLine, $utf8)
$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# TensorRtExec MNIST Reference Negative Runtime Validation")
$lines.Add("")
$lines.Add("- strict: ``$([bool]$Strict)``")
$lines.Add("- runtime artifacts required: ``$([bool]$RequireRuntimeArtifacts)``")
$lines.Add("- checks: ``$($checks.Count)``")
$lines.Add("- passed: ``$($checks.Count - $failed.Count)``")
$lines.Add("- failed: ``$($failed.Count)``")
$lines.Add("")
$lines.Add("| Check | Passed | Actual |")
$lines.Add("| --- | --- | --- |")
foreach ($check in $checks) {
  $actual = ([string]$check.actual).Replace("|", "\\|").Replace("`r", " ").Replace("`n", " ")
  $lines.Add("| ``$($check.id)`` | ``$($check.passed)`` | $actual |")
}
$lines.Add("")
$lines.Add("Controlled negative references prove fail-closed behavior after real enqueue/readback; they do not promote Owner, public-package, post-publish, or release proof.")
[IO.File]::WriteAllLines($MarkdownPath, $lines, $utf8)

Write-Output "TensorRtExecMnistReferenceNegativeRuntimeEvidence=$($checks.Count - $failed.Count)/$($checks.Count)"
Write-Output "RuntimeArtifactsRequired=$([bool]$RequireRuntimeArtifacts)"
Write-Output "Validation=$OutputPath"
if ($Strict -and $failed.Count -gt 0) {
  throw "TensorRtExec MNIST reference negative runtime evidence validation failed: $($failed.Count) check(s)."
}
