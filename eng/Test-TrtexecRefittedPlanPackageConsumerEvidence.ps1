[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$evidencePath = Join-Path $RepositoryRoot "artifacts\interface-coverage\trtexec-refitted-plan-package-consumer-evidence.json"
$priorEvidencePath = Join-Path $RepositoryRoot "artifacts\interface-coverage\trtexec-refitted-plan-persistence-evidence.json"
$validationPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\trtexec-refitted-plan-package-consumer-validation.json"
$validationMarkdownPath = Join-Path $RepositoryRoot "artifacts\interface-coverage\trtexec-refitted-plan-package-consumer-validation.md"

if (-not (Test-Path -LiteralPath $evidencePath -PathType Leaf)) { throw "Package-consumer evidence is missing: $evidencePath" }
if (-not (Test-Path -LiteralPath $priorEvidencePath -PathType Leaf)) { throw "Prior persistence evidence is missing: $priorEvidencePath" }
$evidenceText = Get-Content -LiteralPath $evidencePath -Raw -Encoding utf8
$evidence = $evidenceText | ConvertFrom-Json -Depth 100
$prior = Get-Content -LiteralPath $priorEvidencePath -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
$checks = [Collections.Generic.List[object]]::new()

function Add-Check {
  param([string]$Id, [bool]$Passed, [AllowNull()][object]$Actual)
  $checks.Add([pscustomobject][ordered]@{ id = $Id; passed = $Passed; actual = [string]$Actual })
}

function Test-Sha256 {
  param([AllowNull()][string]$Value)
  return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -match '^[0-9a-f]{64}$'
}

$contract = $evidence.packageContract
$consumer = $evidence.consumer
$artifacts = $evidence.artifacts
$execution = $evidence.execution
$runtime = $evidence.runtime
$boundary = $evidence.proofBoundary
$nativeAssets = @($evidence.nativeAssets)

Add-Check "schema" ($evidence.schemaVersion -eq "trtexec-refitted-plan-package-consumer-evidence.v1") $evidence.schemaVersion
Add-Check "state" ($evidence.state -eq "local-package-consumer-refitted-plan-runtime-passed") $evidence.state
Add-Check "classification" ($evidence.evidenceClassification -eq "local-package-consumer-refitted-plan-runtime") $evidence.evidenceClassification
Add-Check "trt10-runtime-key" ($evidence.sourceRuntimeKey -eq "win-x64-trt10.11-cuda12.9-cudnn9.22") $evidence.sourceRuntimeKey
Add-Check "two-declared-local-sources" ([int]$contract.declaredLocalSourceCount -eq 2) $contract.declaredLocalSourceCount
Add-Check "nuget-org-disabled" (-not [bool]$contract.nugetOrgEnabled) $contract.nugetOrgEnabled
Add-Check "package-reference-only" ([bool]$contract.usesPackageReferenceOnly) $contract.usesPackageReferenceOnly
Add-Check "no-project-reference" (-not [bool]$contract.usesProjectReference) $contract.usesProjectReference
Add-Check "no-manual-managed-load" (-not [bool]$contract.manualManagedAssemblyLoad) $contract.manualManagedAssemblyLoad
Add-Check "isolated-restore-cache" ([bool]$contract.isolatedRestorePathUsed) $contract.isolatedRestorePathUsed
Add-Check "target-packages-resolved" ([bool]$contract.managedPackageResolved -and [bool]$contract.bridgePackageResolved) "$($contract.managedPackageResolved)/$($contract.bridgePackageResolved)"
Add-Check "managed-package-id" ($evidence.packages.managed.id -eq "JYPPX.TensorRT.CSharp.API") $evidence.packages.managed.id
Add-Check "bridge-package-id" ($evidence.packages.bridge.id -like "JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.*.Bridge") $evidence.packages.bridge.id
Add-Check "package-hashes" ((Test-Sha256 $evidence.packages.managed.sha256) -and (Test-Sha256 $evidence.packages.bridge.sha256)) "$($evidence.packages.managed.sha256)/$($evidence.packages.bridge.sha256)"
Add-Check "consumer-outside-repository" ([bool]$consumer.rootOutsideRepository) $consumer.rootOutsideRepository
Add-Check "consumer-off-system-drive" (-not [bool]$consumer.workspaceOnSystemDrive) $consumer.workspaceOnSystemDrive
Add-Check "consumer-workspace-removed" ([bool]$consumer.workspaceRemovedAfterValidation -and -not [bool]$consumer.outputPreserved) "$($consumer.workspaceRemovedAfterValidation)/$($consumer.outputPreserved)"
Add-Check "managed-assembly-isolated" ([bool]$consumer.managedAssemblyOutsideSourceTree -and [bool]$consumer.managedAssemblyUnderConsumerRoot) "$($consumer.managedAssemblyOutsideSourceTree)/$($consumer.managedAssemblyUnderConsumerRoot)"
Add-Check "bridge-isolated" ([bool]$consumer.bridgeUnderConsumerRoot) $consumer.bridgeUnderConsumerRoot
Add-Check "consumer-source-hashes" ((Test-Sha256 $consumer.projectSha256) -and (Test-Sha256 $consumer.programSha256)) "$($consumer.projectSha256)/$($consumer.programSha256)"
Add-Check "relative-source-artifacts" ($artifacts.sourcePlan -notmatch '^[A-Za-z]:[\\/]' -and $artifacts.sourceInput -notmatch '^[A-Za-z]:[\\/]') "$($artifacts.sourcePlan)/$($artifacts.sourceInput)"
Add-Check "plan-copy-distinct" ([bool]$artifacts.planCopyPathDistinct) $artifacts.planCopyPathDistinct
Add-Check "plan-length-cross-check" ([int64]$artifacts.planLengthBytes -eq [int64]$prior.tensorRt10.persistedPlanLengthBytes) $artifacts.planLengthBytes
Add-Check "plan-hash-cross-check" ($artifacts.sourcePlanSha256 -eq $prior.tensorRt10.persistedPlanSha256 -and $artifacts.copiedPlanSha256 -eq $prior.tensorRt10.persistedPlanSha256) "$($artifacts.sourcePlanSha256)/$($artifacts.copiedPlanSha256)"
Add-Check "input-copy-distinct" ([bool]$artifacts.inputCopyPathDistinct) $artifacts.inputCopyPathDistinct
Add-Check "input-copy-hash" ([int64]$artifacts.inputLengthBytes -eq 3136 -and $artifacts.sourceInputSha256 -eq $artifacts.copiedInputSha256 -and (Test-Sha256 $artifacts.copiedInputSha256)) "$($artifacts.inputLengthBytes)/$($artifacts.copiedInputSha256)"
Add-Check "output-length" ([int64]$artifacts.outputLengthBytes -eq 40) $artifacts.outputLengthBytes
Add-Check "output-hash-cross-check" ($artifacts.outputSha256 -eq $prior.tensorRt10.baselineOutputSha256 -and $artifacts.expectedOutputSha256 -eq $prior.tensorRt10.baselineOutputSha256) "$($artifacts.outputSha256)/$($artifacts.expectedOutputSha256)"
Add-Check "output-exact-match" ([bool]$artifacts.outputExactMatch) $artifacts.outputExactMatch
Add-Check "process-exits" ([int]$execution.restoreExitCode -eq 0 -and [int]$execution.buildExitCode -eq 0 -and [int]$execution.runtimeExitCode -eq 0) "$($execution.restoreExitCode)/$($execution.buildExitCode)/$($execution.runtimeExitCode)"
Add-Check "runtime-passed" ([bool]$execution.runtimePassed) $execution.runtimePassed
Add-Check "execution-log-hashes" ((Test-Sha256 $execution.stdoutSha256) -and (Test-Sha256 $execution.stderrSha256) -and (Test-Sha256 $execution.combinedSha256)) "$($execution.stdoutSha256)/$($execution.stderrSha256)/$($execution.combinedSha256)"
Add-Check "command-shape" ($execution.commandShape -eq "dotnet run --project <consumer-project> -c Release --no-build -- <copied-plan> <copied-input> <raw-output> <expected-output-sha256>") $execution.commandShape
Add-Check "full-weight-refittable-fact" (-not [bool]$runtime.engineRefittable) $runtime.engineRefittable
Add-Check "engine-metadata" ([int]$runtime.engineIOTensorCount -eq 2 -and [int]$runtime.engineLayerCount -eq 5 -and [int]$runtime.engineOptimizationProfileCount -eq 1) "$($runtime.engineIOTensorCount)/$($runtime.engineLayerCount)/$($runtime.engineOptimizationProfileCount)"
Add-Check "input-contract" ($runtime.inputTensor -eq "Input3" -and $runtime.inputShape -eq "[1,1,28,28]" -and [int]$runtime.inputElementCount -eq 784) "$($runtime.inputTensor)/$($runtime.inputShape)/$($runtime.inputElementCount)"
Add-Check "output-contract" ($runtime.outputTensor -eq "Plus214_Output_0" -and $runtime.outputShape -eq "[1,10]" -and [int]$runtime.outputElementCount -eq 10) "$($runtime.outputTensor)/$($runtime.outputShape)/$($runtime.outputElementCount)"
Add-Check "enqueue-gates" ([bool]$runtime.bindingsReadyForEnqueue -and [bool]$runtime.enqueueCompleted -and [bool]$runtime.ownerScopeExited -and [bool]$runtime.outputExactMatch) "$($runtime.bindingsReadyForEnqueue)/$($runtime.enqueueCompleted)/$($runtime.ownerScopeExited)/$($runtime.outputExactMatch)"
Add-Check "mnist-output-index" ([int]$runtime.predictedIndex -eq 7) $runtime.predictedIndex
Add-Check "host-metadata" (-not [string]::IsNullOrWhiteSpace([string]$evidence.host.osDescription) -and $evidence.host.processArchitecture -eq "X64" -and [bool]$evidence.host.nvidiaSmiAvailable -and -not [string]::IsNullOrWhiteSpace([string]$evidence.host.gpuName)) "$($evidence.host.processArchitecture)/$($evidence.host.gpuName)"
Add-Check "native-inventory" (@($nativeAssets | Where-Object role -eq "bridge-package-output").Count -eq 1 -and @($nativeAssets | Where-Object role -eq "tensor-rt-runtime").Count -eq 1 -and @($nativeAssets | Where-Object role -eq "cuda-runtime").Count -eq 1 -and @($nativeAssets | Where-Object { -not (Test-Sha256 $_.sha256) }).Count -eq 0) (@($nativeAssets.role) -join ',')
Add-Check "c-drive-clean" (-not [bool]$evidence.cDriveAudit.consumerWorkspaceUsedSystemDrive -and [int]$evidence.cDriveAudit.namedConsumerArtifactMatchCount -eq 0 -and -not [bool]$evidence.cDriveAudit.unrelatedUserOrSystemCacheTouched) "$($evidence.cDriveAudit.consumerWorkspaceUsedSystemDrive)/$($evidence.cDriveAudit.namedConsumerArtifactMatchCount)"
Add-Check "local-runtime-boundary" ([bool]$boundary.isLocalPackageConsumerRuntimeProof) $boundary.isLocalPackageConsumerRuntimeProof
Add-Check "no-public-package-proof" (-not [bool]$boundary.isPackageConsumerRuntimeProof -and -not [bool]$boundary.packagesDownloadedFromPublicFeed -and -not [bool]$boundary.isPostPublishProof) "$($boundary.isPackageConsumerRuntimeProof)/$($boundary.packagesDownloadedFromPublicFeed)/$($boundary.isPostPublishProof)"
Add-Check "no-release-side-effect" (-not [bool]$boundary.canPublishPublicly -and -not [bool]$boundary.canCloseReleaseIssue -and -not [bool]$boundary.performsPublish) "$($boundary.canPublishPublicly)/$($boundary.canCloseReleaseIssue)/$($boundary.performsPublish)"
Add-Check "compact-evidence-path-free" ($evidenceText -notmatch '(?i)[A-Z]:\\') "absolute-windows-path-present=$($evidenceText -match '(?i)[A-Z]:\\')"

$failed = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
  schemaVersion = "trtexec-refitted-plan-package-consumer-validation.v1"
  strict = [bool]$Strict
  checkCount = $checks.Count
  passedCount = $checks.Count - $failed.Count
  failureCount = $failed.Count
  checks = @($checks)
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $validationPath -Encoding utf8

$lines = @(
  "# TensorRtExec Refitted Plan Package Consumer Validation",
  "",
  "- strict: ``$([bool]$Strict)``",
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

Write-Host "TensorRtExec refitted-plan package-consumer evidence: $($checks.Count - $failed.Count)/$($checks.Count) checks passed."
if ($Strict -and $failed.Count -gt 0) {
  throw "TensorRtExec refitted-plan package-consumer evidence validation failed: $($failed.id -join ', ')"
}
