[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-RunbookStep {
  param(
    [string]$Id,
    [string]$Phase,
    [string]$OwnerAction,
    [string]$TargetArtifact,
    [string[]]$RequiredInputs,
    [string]$Command,
    [string]$ValidatorCommand,
    [string]$FailureRecovery,
    [string[]]$ForbiddenSubstitutes
  )

  [pscustomobject]@{
    id = $Id
    phase = $Phase
    ownerAction = $OwnerAction
    targetArtifact = $TargetArtifact
    requiredInputs = $RequiredInputs
    command = $Command
    validatorCommand = $ValidatorCommand
    failureRecovery = $FailureRecovery
    forbiddenSubstitutes = $ForbiddenSubstitutes
    stepState = "blocked-owner-action-required"
    performsPublish = $false
    canCloseReleaseIssue = $false
  }
}

$ownerProofRealInputConvergence = Read-JsonOrNull "artifacts\final-release\owner-proof-real-input-convergence.json"
$ownerProofRealInputConvergenceValidation = Read-JsonOrNull "artifacts\final-release\owner-proof-real-input-convergence-validation.json"
$releaseIssueCloseRecordRealInputMap = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-real-input-map.json"
$releaseCloseStrictRecordCandidate = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate.json"
$releaseCloseStrictRecordCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate-validation.json"

if ($null -eq $ownerProofRealInputConvergence) {
  throw "Missing owner proof real input convergence. Run Export-OwnerProofRealInputConvergence.ps1 first."
}

$fallbackNonSubstituteKinds = @(
  "template",
  "draft",
  "candidate",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "schema-only",
  "preflight-only",
  "dependency-probe-only",
  "blocked-by-cuda-driver"
)

$nonSubstituteKinds = @(Get-PropertyOrDefault -Object $ownerProofRealInputConvergence -Name "nonSubstituteProofKinds" -DefaultValue (Get-PropertyOrDefault -Object $releaseIssueCloseRecordRealInputMap -Name "nonSubstituteProofKinds" -DefaultValue $fallbackNonSubstituteKinds))
foreach ($kind in $fallbackNonSubstituteKinds) {
  if ($nonSubstituteKinds -notcontains $kind) {
    $nonSubstituteKinds += $kind
  }
}

$strictCloseValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
$strictValidators = @(Get-PropertyOrDefault -Object $ownerProofRealInputConvergence -Name "strictValidators" -DefaultValue @())
foreach ($validator in @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOverlayCandidate.ps1 -Strict",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictRecordCandidate.ps1 -Strict",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerProofRealInputConvergence.ps1 -Strict",
  $strictCloseValidatorCommand
)) {
  if ($strictValidators -notcontains $validator) {
    $strictValidators += $validator
  }
}

$publicPackageSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "candidate", "schema-only")
$runtimeSmokeSubstitutes = @("local feed", "ProjectReference", "direct .nupkg", "dependency-probe-only", "blocked-by-cuda-driver", "preflight-only")
$closeDecisionSubstitutes = @("template", "draft", "candidate", "schema-only", "preflight-only", "hash match only")

$runbookSteps = @(
  New-RunbookStep -Id "confirm-public-package-channel" -Phase "public-package-proof" -OwnerAction "Confirm the managed and runtime package source is the real public channel used by owners, not a local feed or direct package path." -TargetArtifact "artifacts/final-release/post-publish-verification-owner-input.template.json" -RequiredInputs @("publicChannelPackageSource", "managedPackageId", "managedPackageVersion", "runtimePackageKey") -Command "Owner fills artifacts/final-release/post-publish-verification-owner-input.template.json with the public package source." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict" -FailureRecovery "Return to post-publish-verification-owner-input and replace local/template source fields with real public-channel values." -ForbiddenSubstitutes $publicPackageSubstitutes
  New-RunbookStep -Id "collect-public-package-hashes" -Phase "public-package-proof" -OwnerAction "Download packages from the confirmed public source and record package file paths plus SHA256 hashes." -TargetArtifact "artifacts/final-release/post-publish-verification-owner-input.template.json" -RequiredInputs @("managedNupkgSha256", "runtimeNupkgSha256", "downloadedPackagePaths") -Command "Owner downloads packages outside the repository and records SHA256 values in the owner input record." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict" -FailureRecovery "Re-download from the public source and update package hash fields; local build output is not acceptable." -ForbiddenSubstitutes $publicPackageSubstitutes
  New-RunbookStep -Id "prepare-clean-consumer-outside-repo" -Phase "clean-consumer-runtime" -OwnerAction "Create or reuse a clean consumer project outside this repository with no ProjectReference, local feed, or direct .nupkg dependency." -TargetArtifact "artifacts/final-release/package-consumer-runtime-proof-candidate.json" -RequiredInputs @("cleanConsumerProjectPath", "publicPackageSource", "runtimePackageKey") -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerExternalSmokeScaffold.ps1 -OutputRoot <outside-repo-path>" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict" -FailureRecovery "Return to package-consumer-runtime-proof-candidate and replace repo-local consumer evidence with an external clean consumer." -ForbiddenSubstitutes $runtimeSmokeSubstitutes
  New-RunbookStep -Id "run-clean-consumer-runtime-smoke" -Phase "clean-consumer-runtime" -OwnerAction "Run restore/build/runtime smoke in the clean consumer against the public package source and target runtime key." -TargetArtifact "artifacts/final-release/package-consumer-runtime-proof-candidate.json" -RequiredInputs @("restoreCommand", "buildCommand", "smokeCommand", "smokeExitCode", "runtimePackageKey") -Command "Owner runs restore, build, dependency probe, and runtime smoke in the clean consumer directory." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict" -FailureRecovery "Fix the clean consumer or compatible CUDA host and rerun; dependency probe alone cannot become runtime proof." -ForbiddenSubstitutes $runtimeSmokeSubstitutes
  New-RunbookStep -Id "capture-smoke-log-hash-exit-host" -Phase "clean-consumer-runtime" -OwnerAction "Capture smoke log path, smoke log SHA256, exit code, stdout/stderr summary, host metadata, driver/runtime versions, and reviewed command lines." -TargetArtifact "artifacts/final-release/package-consumer-runtime-proof-candidate.json" -RequiredInputs @("smokeLogPath", "smokeLogSha256", "exitCode", "hostMetadata", "driverVersion", "cudaRuntimeVersion") -Command "Owner computes SHA256 for logs and records host metadata from the compatible runtime host." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict" -FailureRecovery "Regenerate log/hash metadata from the same smoke run; mismatched SHA256 or blocked-by-cuda-driver remains blocked." -ForbiddenSubstitutes $runtimeSmokeSubstitutes
  New-RunbookStep -Id "update-post-publish-owner-input" -Phase "owner-input-backfill" -OwnerAction "Backfill the post-publish verification owner input with public source, package identity, hashes, command capture, and clean consumer references." -TargetArtifact "artifacts/final-release/post-publish-verification-owner-input.template.json" -RequiredInputs @("publicChannelPackageSource", "packageIdentity", "nupkgSha256", "cleanConsumerIdentity", "commandCapture") -Command "Owner edits the post-publish owner input template and re-runs its strict validator." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict" -FailureRecovery "Return to post-publish-verification-owner-input; do not promote template, draft, or schema-only records." -ForbiddenSubstitutes $publicPackageSubstitutes
  New-RunbookStep -Id "update-package-consumer-runtime-proof-candidate" -Phase "owner-input-backfill" -OwnerAction "Backfill the package consumer runtime proof candidate with clean external consumer smoke evidence and matching hashes." -TargetArtifact "artifacts/final-release/package-consumer-runtime-proof-candidate.json" -RequiredInputs @("cleanExternalConsumer", "smokeLogSha256", "packageSha256", "runtimeKey", "hostMetadata") -Command "Owner refreshes package-consumer-runtime-proof-candidate from the validated external smoke evidence." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofCandidate.ps1 -Strict" -FailureRecovery "Return to clean consumer smoke execution; local feed and direct nupkg proof are blocked." -ForbiddenSubstitutes $runtimeSmokeSubstitutes
  New-RunbookStep -Id "update-release-issue-id-url" -Phase "release-issue-close-input" -OwnerAction "Record the real release issue id and URL that Owner intends to close after all strict gates pass." -TargetArtifact "artifacts/final-release/release-issue-close-record-owner-input.template.json" -RequiredInputs @("releaseIssueId", "releaseIssueUrl") -Command "Owner backfills release issue id and URL in the close record owner input template." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOwnerInput.ps1 -Strict" -FailureRecovery "Return to release-issue-close-record-real-input-map and fill the missing release issue fields." -ForbiddenSubstitutes $closeDecisionSubstitutes
  New-RunbookStep -Id "update-rollback-owner-trigger-plan" -Phase "release-issue-close-input" -OwnerAction "Record rollback owner, rollback trigger, and rollback plan reviewed against the real public package channel." -TargetArtifact "artifacts/final-release/release-issue-final-close-decision.template.json" -RequiredInputs @("rollbackOwner", "rollbackTrigger", "rollbackPlan") -Command "Owner reviews rollback requirements and fills the final close decision template." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict" -FailureRecovery "Return to release-issue-final-close-decision; placeholder rollback fields keep close blocked." -ForbiddenSubstitutes $closeDecisionSubstitutes
  New-RunbookStep -Id "update-final-close-decision" -Phase "release-issue-close-input" -OwnerAction "Record final Owner close decision only after real post-publish proof, clean consumer runtime smoke, rollback review, and package hash capture pass." -TargetArtifact "artifacts/final-release/release-issue-final-close-decision.template.json" -RequiredInputs @("ownerFinalCloseDecision", "decisionTimestampUtc", "reviewedEvidenceBundleSha256") -Command "Owner fills the final close decision and records reviewed evidence hashes." -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict" -FailureRecovery "Return to release issue final close decision validation; guidance or hash-only matches are not authorization." -ForbiddenSubstitutes $closeDecisionSubstitutes
  New-RunbookStep -Id "refresh-close-record-overlay-candidate" -Phase "strict-close-refresh" -OwnerAction "Refresh the release issue close record overlay candidate from the real owner input and proof artifacts." -TargetArtifact "artifacts/final-release/release-issue-close-record-overlay-candidate.json" -RequiredInputs @("ownerInputPaths", "sourceArtifactHashes", "realProofArtifactPaths") -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordOverlayCandidate.ps1" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOverlayCandidate.ps1 -Strict" -FailureRecovery "Return to release-issue-close-record-candidate or overlay candidate when owner inputs or hashes are missing." -ForbiddenSubstitutes $closeDecisionSubstitutes
  New-RunbookStep -Id "refresh-release-close-strict-candidate" -Phase "strict-close-refresh" -OwnerAction "Refresh the strict release close candidate after all real owner inputs and real proof artifacts are present." -TargetArtifact "artifacts/final-release/release-close-strict-record-candidate.json" -RequiredInputs @("strictCandidateSourceHashes", "realProofValidationStates", "ownerDecisionValidationState") -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseStrictRecordCandidate.ps1" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseStrictRecordCandidate.ps1 -Strict" -FailureRecovery "Return to strict candidate dependencies; failed action-required rows must remain visible." -ForbiddenSubstitutes $closeDecisionSubstitutes
  New-RunbookStep -Id "refresh-owner-proof-convergence" -Phase "strict-close-refresh" -OwnerAction "Refresh the Owner proof convergence matrix to confirm missing real inputs, blocked validators, and blocked proof counters are all resolved before final close." -TargetArtifact "artifacts/final-release/owner-proof-real-input-convergence.json" -RequiredInputs @("realInputMappings", "validatorStates", "proofBlockerCounts") -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerProofRealInputConvergence.ps1" -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerProofRealInputConvergence.ps1 -Strict" -FailureRecovery "Return to owner-proof-real-input-convergence rows and repair every blocked owner action." -ForbiddenSubstitutes $closeDecisionSubstitutes
  New-RunbookStep -Id "run-final-release-issue-close-validator" -Phase "strict-close-gate" -OwnerAction "Run the final release issue close validator with FailOnNotCloseReady only after all real proof and Owner close decision inputs are present." -TargetArtifact "artifacts/final-release/release-issue-close-record.json" -RequiredInputs @("validatorPassingPostPublishProof", "validatorPassingRuntimeProof", "ownerFinalCloseDecision", "rollbackPlan", "releaseIssueId", "releaseIssueUrl") -Command $strictCloseValidatorCommand -ValidatorCommand $strictCloseValidatorCommand -FailureRecovery "Return to release issue close record validation or overlay candidate; do not close the issue until this command passes on real proof." -ForbiddenSubstitutes $closeDecisionSubstitutes
)

$failureRecoveryMap = @(
  [pscustomobject]@{ failure = "public package source validation failed"; recoverAt = "artifacts/final-release/post-publish-verification-owner-input.template.json"; ownerAction = "Replace local/template package source with real public channel identity and downloaded package hashes." }
  [pscustomobject]@{ failure = "clean consumer runtime smoke failed"; recoverAt = "artifacts/final-release/package-consumer-runtime-proof-candidate.json"; ownerAction = "Run or repair clean external consumer runtime smoke on a compatible CUDA host and refresh log/hash/exit-code fields." }
  [pscustomobject]@{ failure = "rollback review failed"; recoverAt = "artifacts/final-release/release-issue-final-close-decision.template.json"; ownerAction = "Fill rollback owner, rollback trigger, rollback plan, and owner final decision only after review." }
  [pscustomobject]@{ failure = "close candidate validation failed"; recoverAt = "artifacts/final-release/release-issue-close-record-candidate.json; artifacts/final-release/release-issue-close-record-overlay-candidate.json"; ownerAction = "Refresh candidate and overlay from real owner input and source artifact hashes." }
  [pscustomobject]@{ failure = "strict validator failed"; recoverAt = "artifacts/final-release/release-issue-close-record-validation.json"; ownerAction = "Keep release issue open and repair failed action-required rows before rerunning the final close validator." }
)

$sourceArtifacts = @(
  "artifacts/final-release/owner-proof-real-input-convergence.json",
  "artifacts/final-release/owner-proof-real-input-convergence-validation.json",
  "artifacts/final-release/release-issue-close-record-real-input-map.json",
  "artifacts/final-release/release-issue-close-record-real-input-map-validation.json",
  "artifacts/final-release/release-close-strict-record-candidate.json",
  "artifacts/final-release/release-close-strict-record-candidate-validation.json"
)

$blockedStepCount = @($runbookSteps | Where-Object { $_.stepState -like "blocked*" }).Count
$ownerActionStepCount = @($runbookSteps | Where-Object { $_.ownerAction -match "Owner|owner" -or $_.stepState -eq "blocked-owner-action-required" }).Count

$runbook = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-close-final-owner-runbook"
  runbookState = "blocked-release-close-final-owner-action-required"
  ownerProofRealInputConvergenceState = [string](Get-PropertyOrDefault -Object $ownerProofRealInputConvergence -Name "convergenceState" -DefaultValue "missing-owner-proof-real-input-convergence")
  ownerProofRealInputConvergenceValidationState = [string](Get-PropertyOrDefault -Object $ownerProofRealInputConvergenceValidation -Name "validationState" -DefaultValue "missing-owner-proof-real-input-convergence-validation")
  releaseIssueCloseRecordRealInputMapState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordRealInputMap -Name "mapState" -DefaultValue "missing-release-issue-close-record-real-input-map")
  releaseCloseStrictRecordCandidateState = [string](Get-PropertyOrDefault -Object $releaseCloseStrictRecordCandidate -Name "candidateState" -DefaultValue "missing-release-close-strict-record-candidate")
  releaseCloseStrictRecordCandidateValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseStrictRecordCandidateValidation -Name "validationState" -DefaultValue "missing-release-close-strict-record-candidate-validation")
  runbookStepCount = $runbookSteps.Count
  blockedStepCount = $blockedStepCount
  ownerActionStepCount = $ownerActionStepCount
  strictValidatorCount = $strictValidators.Count
  runbookSteps = $runbookSteps
  failureRecoveryMap = $failureRecoveryMap
  strictValidators = $strictValidators
  nonSubstituteProofKinds = $nonSubstituteKinds
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Release close final owner runbook is an Owner execution manual only. It does not publish, upload, approve publication, close release issues, collect proof automatically, or promote template/draft/candidate/hash-only/local-feed evidence to proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-close-final-owner-runbook.json"
$markdownPath = Join-Path $artifactRoot "release-close-final-owner-runbook.md"
$runbook | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$stepLines = $runbookSteps | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.phase)`` | $($_.ownerAction.Replace("|", "\|")) | ``$($_.targetArtifact)`` | ``$($_.stepState)`` | $($_.failureRecovery.Replace("|", "\|")) |"
}
$recoveryLines = $failureRecoveryMap | ForEach-Object {
  "| $($_.failure.Replace("|", "\|")) | ``$($_.recoverAt)`` | $($_.ownerAction.Replace("|", "\|")) |"
}
$validatorLines = $strictValidators | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $nonSubstituteKinds | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release Close Final Owner Runbook

生成时间：$($runbook.generatedAtUtc)

该 runbook 是 release close 最后一公里的 Owner 执行手册。它把真实 public package proof、仓库外 clean consumer runtime smoke、Owner 输入、rollback review、final close decision 和 strict close validator 串成一个顺序化执行面。它不会发布包、不会上传资产、不会批准公开发布，也不会关闭 release issue。

| 项目 | 当前值 |
|---|---|
| runbookState | ``$($runbook.runbookState)`` |
| ownerProofRealInputConvergenceState | ``$($runbook.ownerProofRealInputConvergenceState)`` |
| ownerProofRealInputConvergenceValidationState | ``$($runbook.ownerProofRealInputConvergenceValidationState)`` |
| runbookStepCount | ``$($runbook.runbookStepCount)`` |
| blockedStepCount | ``$($runbook.blockedStepCount)`` |
| ownerActionStepCount | ``$($runbook.ownerActionStepCount)`` |
| strictValidatorCount | ``$($runbook.strictValidatorCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Runbook Steps

| ID | Phase | Owner Action | Target Artifact | State | Failure Recovery |
|---|---|---|---|---|---|
$($stepLines -join "`r`n")

## Failure Recovery Map

| Failure | Recover At | Owner Action |
|---|---|---|
$($recoveryLines -join "`r`n")

## Strict Validators

$($validatorLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Boundary

$($runbook.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close final owner runbook written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "RunbookState=$($runbook.runbookState) Steps=$($runbook.runbookStepCount) BlockedSteps=$($runbook.blockedStepCount) OwnerActionSteps=$($runbook.ownerActionStepCount) StrictValidators=$($runbook.strictValidatorCount) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
