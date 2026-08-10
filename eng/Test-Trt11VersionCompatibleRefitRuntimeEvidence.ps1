[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/interface-coverage/trt11-version-compatible-refit-runtime-evidence.json",
  [string]$OutputPath = "artifacts/interface-coverage/trt11-version-compatible-refit-runtime-validation.json",
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

function Contains-Text {
  param([object]$Values, [Parameter(Mandatory = $true)][string]$Expected)
  return @($Values | ForEach-Object { [string]$_ }) -contains $Expected
}

$record = Get-Content -LiteralPath (Resolve-RepositoryPath $InputPath) -Raw | ConvertFrom-Json
$checks = [Collections.Generic.List[object]]::new()

function Add-Check {
  param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][bool]$Passed, [Parameter(Mandatory = $true)][string]$Detail)
  $checks.Add([pscustomobject][ordered]@{ name = $Name; passed = $Passed; detail = $Detail }) | Out-Null
}

function Add-ArtifactCheck {
  param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][object]$Artifact)
  $path = Resolve-RepositoryPath $Artifact.path
  $exists = Test-Path -LiteralPath $path -PathType Leaf
  $length = if ($exists) { (Get-Item -LiteralPath $path).Length } else { -1 }
  $sha256 = if ($exists) { Get-Sha256 $Artifact.path } else { "" }
  Add-Check $Name ($exists -and $length -eq [long]$Artifact.lengthBytes -and $sha256 -eq [string]$Artifact.sha256) "$($Artifact.path)/$length/$sha256"
}

Add-Check "schema" ($record.schemaVersion -eq "trt11-version-compatible-refit-runtime-evidence.v1") ([string]$record.schemaVersion)
Add-Check "source-tree-boundary" (-not [bool]$record.sourceTreeDirtyAtExecution) "sourceTreeDirtyAtExecution=$($record.sourceTreeDirtyAtExecution)"

foreach ($processName in @("sameProcess", "secondProcess")) {
  $process = $record.processes.$processName
  Add-ArtifactCheck "$processName-command" $process.command
  Add-ArtifactCheck "$processName-stdout" $process.stdout
  Add-ArtifactCheck "$processName-stderr" $process.stderr
  Add-ArtifactCheck "$processName-exit-code" $process.exitCode
  Add-ArtifactCheck "$processName-report" $process.report
  Add-ArtifactCheck "$processName-strict-validation" $process.strictValidation
  Add-ArtifactCheck "$processName-output" $process.output
}

foreach ($name in @("model", "input", "reference", "nativeBridge", "vendorRuntime", "cudaRuntime")) {
  Add-ArtifactCheck "asset-$name" $record.assets.$name
}

Add-ArtifactCheck "stripped-plan" $record.buildArtifacts.strippedPlan
Add-ArtifactCheck "refitted-plan" $record.buildArtifacts.refittedPlan
Add-ArtifactCheck "environment" $record.environment

$main = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.sameProcess.report.path) -Raw | ConvertFrom-Json
$second = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.secondProcess.report.path) -Raw | ConvertFrom-Json
$mainOutput = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.sameProcess.output.path) -Raw | ConvertFrom-Json
$secondOutput = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.secondProcess.output.path) -Raw | ConvertFrom-Json
$mainStrict = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.sameProcess.strictValidation.path) -Raw | ConvertFrom-Json
$secondStrict = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.secondProcess.strictValidation.path) -Raw | ConvertFrom-Json
$mainStdout = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.sameProcess.stdout.path) -Raw
$secondStdout = Get-Content -LiteralPath (Resolve-RepositoryPath $record.processes.secondProcess.stdout.path) -Raw

Add-Check "main-exit" ([int]$record.processes.sameProcess.exitCode.value -eq 0) "exit=$($record.processes.sameProcess.exitCode.value)"
Add-Check "second-exit" ([int]$record.processes.secondProcess.exitCode.value -eq 0) "exit=$($record.processes.secondProcess.exitCode.value)"
Add-Check "main-runtime-state" ([bool]$main.Success -and $main.State -eq "external-onnx-refit-reload-reference-validated-runtime" -and $main.ProofClassification -eq "synthetic-input-runtime") "$($main.State)/$($main.ProofClassification)"
Add-Check "second-runtime-state" ([bool]$second.Success -and $second.State -eq "load-engine-reference-validated-runtime" -and $second.ProofClassification -eq "synthetic-input-runtime") "$($second.State)/$($second.ProofClassification)"
Add-Check "main-preflight" ([bool]$main.CapabilityProbe.RuntimeAvailable -and [bool]$main.CapabilityProbe.BuilderAvailable -and $main.CapabilityProbe.TensorRtVersion -eq "11.0.0") "runtime=$($main.CapabilityProbe.RuntimeAvailable)/builder=$($main.CapabilityProbe.BuilderAvailable)/trt=$($main.CapabilityProbe.TensorRtVersion)"
Add-Check "main-version-compatible-applied" ([bool]$main.DeploymentOptions.VersionCompatible -and (Contains-Text $main.OptionImplementationStatus.AppliedOptions "--versionCompatible") -and $mainStdout.Contains("TrtexecDeploymentControl Name=VersionCompatible Applied=True Requested=True Readback=True ReadbackMatch=True")) "VersionCompatible set/readback"
Add-Check "main-refit-applied" ((Contains-Text $main.OptionImplementationStatus.AppliedOptions "--refit") -and (Contains-Text $main.OptionImplementationStatus.AppliedOptions "--refitFromOnnx") -and [bool]$main.RefitSnapshot.Attempted -and [bool]$main.RefitSnapshot.Succeeded -and [bool]$main.RefitSnapshot.ParserRefitReturned -and [bool]$main.RefitSnapshot.EngineRefitReturned -and [int]$main.RefitSnapshot.ParserErrorCount -eq 0) "refit lifecycle completed without parser errors"
Add-Check "main-refit-inventory" ([int]$main.RefitSnapshot.MissingWeightsBefore.Count -eq 0 -and [int]$main.RefitSnapshot.MissingWeightsAfter.Count -eq 0 -and [int]$main.RefitSnapshot.AllWeightsBefore.Count -gt 0 -and [int]$main.RefitSnapshot.AllWeightsAfter.Count -gt 0 -and [bool]$main.RefitSnapshot.ContextCreationAllowed) "missingBefore=$($main.RefitSnapshot.MissingWeightsBefore.Count)/missingAfter=$($main.RefitSnapshot.MissingWeightsAfter.Count)"
Add-Check "main-persistence" ([bool]$main.RefitPersistenceSnapshot.Attempted -and [bool]$main.RefitPersistenceSnapshot.Succeeded -and [bool]$main.RefitPersistenceSnapshot.OriginalRefittedEngineDisposedBeforeReload -and [bool]$main.RefitPersistenceSnapshot.ReloadSucceeded -and [bool]$main.RefitPersistenceSnapshot.RefittableWeightsIncludedInSerialization -and [bool]$main.RefitPersistenceSnapshot.ArtifactDiffersFromStrippedPlan) "persisted reload and distinct full-weight artifact"
Add-Check "main-host-code" ($mainStdout.Contains("TrtexecRuntimePolicy Name=EngineHostCodeAllowed Applied=True Requested=True Readback=True ReadbackMatch=True Source=Build")) "build host-code policy readback"
Add-Check "main-reference" ([bool]$main.InferenceRan -and [bool]$main.OutputMatch -and [bool]$main.OutputValidated -and $mainStdout.Contains("ReferenceOutputValidation Requested=True Completed=True Passed=True") -and $mainStdout.Contains("Mismatches=0")) "enqueue/reference mismatch=0"
Add-Check "main-output-hash" ([string]$mainOutput.OutputSha256 -eq [string]$record.outputSha256 -and [int]$mainOutput.OutputByteLength -eq 40) "$($mainOutput.OutputSha256)/$($mainOutput.OutputByteLength)"
Add-Check "second-load-diagnostics" ([bool]$second.LoadedEngineDiagnostics.Attempted -and [bool]$second.LoadedEngineDiagnostics.Succeeded -and $second.LoadedEngineDiagnostics.DiagnosticsState -eq "readonly-deserialize-succeeded" -and $secondStdout.Contains("LoadEngineReadonlyDiagnostics Attempted=True Succeeded=True")) "independent deserialize diagnostics"
Add-Check "second-host-code" ($secondStdout.Contains("TrtexecRuntimePolicy Name=EngineHostCodeAllowed Applied=True Requested=True Readback=True ReadbackMatch=True Source=LoadEngineDiagnostics") -and $secondStdout.Contains("Source=LoadEngineRuntime")) "load-engine host-code policy readback"
Add-Check "second-reference" ([bool]$second.InferenceRan -and [bool]$second.OutputMatch -and [bool]$second.OutputValidated -and $secondStdout.Contains("ReferenceOutputValidation Requested=True Completed=True Passed=True") -and $secondStdout.Contains("Mismatches=0")) "independent enqueue/reference mismatch=0"
Add-Check "output-hash-stable" ([string]$mainOutput.OutputSha256 -eq [string]$secondOutput.OutputSha256 -and [string]$secondOutput.OutputSha256 -eq [string]$record.outputSha256) "same-process=$($mainOutput.OutputSha256)/second-process=$($secondOutput.OutputSha256)"
Add-Check "strict-main" ($mainStrict.validationState -eq "tensor-rt-exec-report-ready" -and @($mainStrict.validationItems).Count -eq 69 -and @($mainStrict.validationItems | Where-Object { -not $_.passed }).Count -eq 0) "strict report checks=$(@($mainStrict.validationItems).Count) failures=$(@($mainStrict.validationItems | Where-Object { -not $_.passed }).Count)"
Add-Check "strict-second" ($secondStrict.validationState -eq "tensor-rt-exec-report-ready" -and @($secondStrict.validationItems).Count -eq 69 -and @($secondStrict.validationItems | Where-Object { -not $_.passed }).Count -eq 0) "strict report checks=$(@($secondStrict.validationItems).Count) failures=$(@($secondStrict.validationItems | Where-Object { -not $_.passed }).Count)"
Add-Check "plan-hash-record" ((Get-Sha256 $record.buildArtifacts.strippedPlan.path) -eq [string]$record.buildArtifacts.strippedPlan.sha256 -and (Get-Sha256 $record.buildArtifacts.refittedPlan.path) -eq [string]$record.buildArtifacts.refittedPlan.sha256 -and [string]$record.buildArtifacts.strippedPlan.sha256 -ne [string]$record.buildArtifacts.refittedPlan.sha256) "stripped/refitted hashes are distinct and recorded"
Add-Check "proof-boundary" ([bool]$record.proofBoundary.isTrt11VersionCompatibleRefitRuntimeEvidence -and [bool]$record.proofBoundary.isBuilderAndEnginePolicyEvidence -and [bool]$record.proofBoundary.isReferenceValidatedSyntheticRuntime -and -not [bool]$record.proofBoundary.isCrossVersionLeanRuntimeProof -and -not [bool]$record.proofBoundary.isModelAccuracyProof -and -not [bool]$record.proofBoundary.isPackageConsumerRuntimeProof -and -not [bool]$record.proofBoundary.isPostPublishProof -and -not [bool]$record.proofBoundary.canPublishPublicly) "synthetic runtime and release boundaries remain explicit"

$failures = @($checks | Where-Object { -not $_.passed })
$validation = [pscustomobject][ordered]@{
  schemaVersion = "trt11-version-compatible-refit-runtime-validation.v1"
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
  "# TRT11 Version-Compatible Refit Runtime Validation", "",
  "- State: ``$($validation.validationState)``", "- Checks: ``$($validation.checkCount)``", "- Failures: ``$($validation.failureCount)``", "",
  "| Check | Passed | Detail |", "|---|---:|---|"
) + $rows
[IO.File]::WriteAllText($markdownPath, ($markdown -join [Environment]::NewLine) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
Write-Host "TRT11 version-compatible refit runtime validation: $($validation.validationState); checks=$($validation.checkCount); failures=$($validation.failureCount)"
if ($Strict -and $failures.Count -gt 0) { throw "TRT11 version-compatible refit runtime validation failed: $($failures.name -join ', ')" }
