[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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

function Convert-ToArray {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Lane {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$SourceArtifact,
    [string]$CurrentState,
    [string[]]$OwnerRequiredFields,
    [string]$FirstOwnerCommand,
    [string[]]$RequiredRealFiles,
    [string[]]$StrictValidators,
    [string]$BlockedReason
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    sourceArtifact = $SourceArtifact
    currentState = $CurrentState
    ownerRequiredFieldCount = @($OwnerRequiredFields).Count
    ownerRequiredFields = @($OwnerRequiredFields)
    firstOwnerCommand = $FirstOwnerCommand
    requiredRealFiles = @($RequiredRealFiles)
    strictValidators = @($StrictValidators)
    blocked = $true
    blockedReason = $BlockedReason
    ownerActionRequired = $true
    performsPublish = $false
    performsRuntimeExecution = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push"
  }
}

function New-GapField {
  param(
    [string]$Id,
    [string]$Group,
    [string]$FieldPath,
    [string]$RequiredEvidence,
    [string]$Status,
    [string[]]$StrictValidators
  )

  [pscustomobject]@{
    id = $Id
    group = $Group
    fieldPath = $FieldPath
    requiredEvidence = $RequiredEvidence
    status = $Status
    strictValidators = @($StrictValidators)
    ownerActionRequired = $true
  }
}

function New-PublicProofStep {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$SourceArtifact,
    [string]$ValidationArtifact,
    [string]$CurrentState,
    [string]$RequiredReadyState,
    [string]$OwnerAction,
    [string]$StrictValidator,
    [string]$BlockedReason
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    sourceArtifact = $SourceArtifact
    validationArtifact = $ValidationArtifact
    currentState = $CurrentState
    requiredReadyState = $RequiredReadyState
    ownerAction = $OwnerAction
    strictValidator = $StrictValidator
    blocked = $true
    blockedReason = $BlockedReason
    forbiddenSubstitutes = @("local feed", "direct .nupkg", "ProjectReference", "dry-run", "package-managed-dry-run", "dashboard-only", "artifact-only", "queued workflow", "missing runner", "sidecar-only", "local test")
    ownerActionRequired = $true
    performsPublish = $false
    performsRuntimeExecution = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "not runtime proof; not post-publish proof; not publish approval; not release close approval; not package push"
  }
}

$cleanRunbook = Read-JsonOrNull "artifacts/final-release/clean-external-package-consumer-owner-runbook.json"
$cleanRunbookValidation = Read-JsonOrNull "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json"
$postPublishRunbook = Read-JsonOrNull "artifacts/final-release/post-publish-owner-verification-runbook.json"
$postPublishRunbookValidation = Read-JsonOrNull "artifacts/final-release/post-publish-owner-verification-runbook-validation.json"
$ownerPublicPublishContract = Read-JsonOrNull "artifacts/final-release/owner-public-publish-execution-result-input-contract.json"
$ownerPublicPublishContractValidation = Read-JsonOrNull "artifacts/final-release/owner-public-publish-execution-result-input-contract-validation.json"
$postPublishContract = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json"
$postPublishContractValidation = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json"
$finalCloseApprovalContract = Read-JsonOrNull "artifacts/final-release/final-release-close-owner-approval-contract.json"
$finalCloseApprovalContractValidation = Read-JsonOrNull "artifacts/final-release/final-release-close-owner-approval-contract-validation.json"
$releaseEvidence = Read-JsonOrNull "artifacts/final-release/release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull "artifacts/final-release/release-evidence-classification-audit.json"
$publicClaimAudit = Read-JsonOrNull "artifacts/final-release/public-proof-claim-boundary-audit.json"
$cleanConsumerClosure = Read-JsonOrNull "artifacts/final-release/clean-consumer-external-proof-closure-pack.json"
$cleanConsumerClosureValidation = Read-JsonOrNull "artifacts/final-release/clean-consumer-external-proof-closure-pack-validation.json"
$githubActionsRunEvidenceValidation = Read-JsonOrNull "artifacts/final-release/github-actions-run-evidence-import-validation.json"
$ownerPublicPublishResultCandidateValidation = Read-JsonOrNull "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json"
$publicPackageDownloadProofCandidateValidation = Read-JsonOrNull "artifacts/final-release/public-package-download-proof-candidate-validation.json"
$postPublishCleanConsumerProofResultValidation = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json"
$finalPublicReleaseClosureBridgeValidation = Read-JsonOrNull "artifacts/final-release/final-public-release-closure-bridge-validation.json"
$releaseIssueCloseOwnerDecisionInputValidation = Read-JsonOrNull "artifacts/final-release/release-issue-close-owner-decision-input-validation.json"

$sourceArtifacts = @(
  "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
  "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
  "artifacts/final-release/post-publish-owner-verification-runbook.json",
  "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
  "artifacts/final-release/owner-public-publish-execution-result-input-contract.json",
  "artifacts/final-release/owner-public-publish-execution-result-input-contract-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
  "artifacts/final-release/final-release-close-owner-approval-contract.json",
  "artifacts/final-release/final-release-close-owner-approval-contract-validation.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-evidence-classification-audit.json",
  "artifacts/final-release/public-proof-claim-boundary-audit.json",
  "artifacts/final-release/clean-consumer-external-proof-closure-pack.json",
  "artifacts/final-release/clean-consumer-external-proof-closure-pack-validation.json",
  "artifacts/final-release/github-actions-run-evidence-import-validation.json",
  "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
  "artifacts/final-release/public-package-download-proof-candidate-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
  "artifacts/final-release/final-public-release-closure-bridge-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json"
)

$sourceStates = [pscustomobject]@{
  cleanExternalPackageConsumerRunbook = [string](Get-PropertyOrDefault -Object $cleanRunbook -Name "runbookState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook")
  cleanExternalPackageConsumerRunbookValidation = [string](Get-PropertyOrDefault -Object $cleanRunbookValidation -Name "validationState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook-validation")
  postPublishOwnerVerificationRunbook = [string](Get-PropertyOrDefault -Object $postPublishRunbook -Name "runbookState" -DefaultValue "missing-post-publish-owner-verification-runbook")
  postPublishOwnerVerificationRunbookValidation = [string](Get-PropertyOrDefault -Object $postPublishRunbookValidation -Name "validationState" -DefaultValue "missing-post-publish-owner-verification-runbook-validation")
  ownerPublicPublishExecutionResultInputContract = [string](Get-PropertyOrDefault -Object $ownerPublicPublishContract -Name "contractState" -DefaultValue "missing-owner-public-publish-execution-result-input-contract")
  ownerPublicPublishExecutionResultInputContractValidation = [string](Get-PropertyOrDefault -Object $ownerPublicPublishContractValidation -Name "validationState" -DefaultValue "missing-owner-public-publish-execution-result-input-contract-validation")
  postPublishCleanConsumerProofRecordContract = [string](Get-PropertyOrDefault -Object $postPublishContract -Name "contractState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-contract")
  postPublishCleanConsumerProofRecordContractValidation = [string](Get-PropertyOrDefault -Object $postPublishContractValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-contract-validation")
  finalReleaseCloseOwnerApprovalContract = [string](Get-PropertyOrDefault -Object $finalCloseApprovalContract -Name "contractState" -DefaultValue "missing-final-release-close-owner-approval-contract")
  finalReleaseCloseOwnerApprovalContractValidation = [string](Get-PropertyOrDefault -Object $finalCloseApprovalContractValidation -Name "validationState" -DefaultValue "missing-final-release-close-owner-approval-contract-validation")
  releaseEvidenceBundle = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  releaseEvidenceClassificationAudit = [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")
  publicProofClaimBoundaryAudit = [string](Get-PropertyOrDefault -Object $publicClaimAudit -Name "auditState" -DefaultValue "missing-public-proof-claim-boundary-audit")
  publicDocsProofBoundaryFreeze = [string](Get-PropertyOrDefault -Object $publicClaimAudit -Name "publicFreezeState" -DefaultValue "missing-public-docs-proof-boundary-freeze")
  cleanConsumerExternalProofClosurePack = [string](Get-PropertyOrDefault -Object $cleanConsumerClosure -Name "closureState" -DefaultValue "missing-clean-consumer-external-proof-closure-pack")
  cleanConsumerExternalProofClosurePackValidation = [string](Get-PropertyOrDefault -Object $cleanConsumerClosureValidation -Name "validationState" -DefaultValue "missing-clean-consumer-external-proof-closure-pack-validation")
  githubActionsRunEvidence = [string](Get-PropertyOrDefault -Object $githubActionsRunEvidenceValidation -Name "validationState" -DefaultValue "missing-github-actions-run-evidence-import-validation")
  ownerPublicPublishResultCandidate = [string](Get-PropertyOrDefault -Object $ownerPublicPublishResultCandidateValidation -Name "validationState" -DefaultValue "missing-owner-public-publish-execution-result-candidate-validation")
  publicPackageDownloadProofCandidate = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProofCandidateValidation -Name "validationState" -DefaultValue "missing-public-package-download-proof-candidate-validation")
  postPublishCleanConsumerProofResult = [string](Get-PropertyOrDefault -Object $postPublishCleanConsumerProofResultValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-validation")
  finalPublicReleaseClosureBridge = [string](Get-PropertyOrDefault -Object $finalPublicReleaseClosureBridgeValidation -Name "validationState" -DefaultValue "missing-final-public-release-closure-bridge-validation")
  releaseIssueCloseOwnerDecisionInput = [string](Get-PropertyOrDefault -Object $releaseIssueCloseOwnerDecisionInputValidation -Name "validationState" -DefaultValue "missing-release-issue-close-owner-decision-input-validation")
}

$lanes = @(
  New-Lane -Order 1 -Id "clean-external-package-consumer" -Title "Clean external package consumer runtime proof collection" -SourceArtifact "clean-external-package-consumer-owner-runbook.json" -CurrentState $sourceStates.cleanExternalPackageConsumerRunbook -OwnerRequiredFields @(
    "cleanConsumer.projectRoot",
    "packageSource.url",
    "managedPackage.id/version/sha256",
    "runtimePackage.id/version/key/sha256",
    "nativeAssetListing.path/sha256",
    "restoreLog.path/sha256",
    "buildLog.path/sha256",
    "runtimeSmoke.stdoutPath/stderrPath/mergedTranscriptPath/sha256",
    "runtimeSmoke.exitCode",
    "runtimeSmoke.smokeStatus",
    "hostMetadata.os/arch/rid/gpu/driver/cuda/tensorrt/cudnn",
    "ownerReview.name/machine/reviewedAtUtc/note"
  ) -FirstOwnerCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CleanExternalPackageConsumerOwnerRunbook.ps1" -RequiredRealFiles @(
    "repository-external clean consumer project",
    "restore/build/runtime smoke logs",
    "managed/runtime nupkg files",
    "native asset listing",
    "owner external proof execution result input"
  ) -StrictValidators @(
    "eng\Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict",
    "eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
    "eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof"
  ) -BlockedReason "Owner must run a repository-external clean consumer and provide real logs, hashes, package metadata, host metadata, and owner review."
  New-Lane -Order 2 -Id "post-publish-owner-verification" -Title "Post-publish clean consumer verification" -SourceArtifact "post-publish-owner-verification-runbook.json" -CurrentState $sourceStates.postPublishOwnerVerificationRunbook -OwnerRequiredFields @(
    "publicPackageSource.url",
    "publishedPackage.url",
    "downloadedManagedPackage.sha256",
    "downloadedRuntimePackage.sha256",
    "postPublishConsumer.projectRoot",
    "postPublishRestoreLog.path/sha256",
    "postPublishBuildLog.path/sha256",
    "postPublishRuntimeSmoke.stdoutPath/stderrPath/mergedTranscriptPath/sha256",
    "postPublishRuntimeSmoke.exitCode",
    "hostMetadata.os/arch/rid/gpu/driver/cuda/tensorrt/cudnn",
    "ownerReview.name/machine/reviewedAtUtc/note"
  ) -FirstOwnerCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishOwnerVerificationRunbook.ps1" -RequiredRealFiles @(
    "public-channel package restore logs",
    "post-publish runtime smoke logs",
    "downloaded nupkg files",
    "native asset listing",
    "post-publish owner result input"
  ) -StrictValidators @(
    "eng\Test-PostPublishOwnerVerificationRunbook.ps1 -Strict",
    "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict",
    "eng\Test-PostPublishCleanConsumerRealProofGate.ps1 -Strict"
  ) -BlockedReason "Owner must publish or point to an approved public channel and run a separate post-publish clean consumer verification."
  New-Lane -Order 3 -Id "owner-public-publish-result-input" -Title "Owner public publish execution result input" -SourceArtifact "owner-public-publish-execution-result-input-contract.json" -CurrentState $sourceStates.ownerPublicPublishExecutionResultInputContract -OwnerRequiredFields @(
    "publicPackageUrl",
    "publicPackageSha256",
    "githubReleaseAssetUrl",
    "githubReleaseAssetSha256",
    "nugetPushTranscriptPath",
    "nugetPushTranscriptSha256",
    "releaseNotesPath",
    "rollbackDecision",
    "ownerReviewer",
    "ownerSignature",
    "nonSubstituteConfirmations"
  ) -FirstOwnerCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerPublicPublishExecutionResultInputContract.ps1" -RequiredRealFiles @(
    "public publish transcript or explicit not-pushed evidence",
    "package URL/hash evidence",
    "release notes",
    "rollback review",
    "owner signature"
  ) -StrictValidators @(
    "eng\Test-OwnerPublicPublishExecutionResultInputContract.ps1 -Strict",
    "eng\Test-OwnerPublicPublishExecutionResultInputTemplate.ps1 -Strict"
  ) -BlockedReason "Owner must fill real public publish result fields; template and contract are not proof."
  New-Lane -Order 4 -Id "post-publish-proof-record-contract" -Title "Post-publish clean consumer proof record contract" -SourceArtifact "post-publish-clean-consumer-proof-record-contract.json" -CurrentState $sourceStates.postPublishCleanConsumerProofRecordContract -OwnerRequiredFields @(
    "postPublish.packageSource",
    "postPublish.downloadedNupkgSha256",
    "postPublish.cleanConsumerLogs",
    "postPublish.hostMetadata",
    "postPublish.validatorOutput",
    "postPublish.ownerReview"
  ) -FirstOwnerCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishCleanConsumerProofRecordContract.ps1" -RequiredRealFiles @(
    "post-publish proof record input",
    "public-channel restore/build/run logs",
    "validator output"
  ) -StrictValidators @(
    "eng\Test-PostPublishCleanConsumerProofRecordContract.ps1 -Strict",
    "eng\Test-FinalPostPublishCleanConsumerProofRecordContract.ps1 -Strict"
  ) -BlockedReason "Post-publish proof contract is blocked until real public-channel evidence exists."
  New-Lane -Order 5 -Id "final-release-close-owner-approval" -Title "Final release close owner approval" -SourceArtifact "final-release-close-owner-approval-contract.json" -CurrentState $sourceStates.finalReleaseCloseOwnerApprovalContract -OwnerRequiredFields @(
    "releaseIssue.id/url",
    "ownerFinalCloseDecision",
    "ownerCloseReviewer",
    "publicPackageSource",
    "publicPackageSha256",
    "publicPublishTranscript",
    "cleanConsumerSmokeLog",
    "postPublishProofReference",
    "rollbackReview",
    "classificationAuditClean"
  ) -FirstOwnerCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleaseCloseOwnerApprovalContract.ps1" -RequiredRealFiles @(
    "final owner close approval input",
    "release evidence bundle",
    "classification audit",
    "post-publish proof",
    "rollback review"
  ) -StrictValidators @(
    "eng\Test-FinalReleaseCloseOwnerApprovalContract.ps1 -Strict",
    "eng\Test-FinalReleaseCloseOwnerApprovalPreflight.ps1 -Strict",
    "eng\Test-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1 -Strict"
  ) -BlockedReason "Final close approval is blocked until real owner approval and accepted proof records exist."
  New-Lane -Order 6 -Id "release-evidence-and-public-docs-freeze" -Title "Release evidence classification and public docs freeze" -SourceArtifact "release-evidence-bundle.json" -CurrentState $sourceStates.releaseEvidenceClassificationAudit -OwnerRequiredFields @(
    "releaseEvidence.bundleState",
    "releaseEvidence.classificationAudit",
    "publicProofClaimBoundaryAudit.auditState",
    "publicDocsProofBoundaryFreeze.publicFreezeState",
    "cleanConsumerExternalClosure.validationState"
  ) -FirstOwnerCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" -RequiredRealFiles @(
    "release-evidence-bundle.json",
    "release-evidence-classification-audit.json",
    "public-proof-claim-boundary-audit.json",
    "clean-consumer-external-proof-closure-pack-validation.json"
  ) -StrictValidators @(
    "eng\Test-PublicProofClaimBoundaryAudit.ps1 -Strict",
    "eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict",
    "eng\Test-CleanConsumerExternalProofClosurePack.ps1 -Strict"
  ) -BlockedReason "Classification and public docs freeze are guardrails only; they do not execute runtime proof or approve publish."
)

$gapFields = @(
  New-GapField -Id "clean-consumer-root" -Group "clean-consumer" -FieldPath "cleanConsumer.projectRoot" -RequiredEvidence "Repository-external clean consumer project root." -Status "missing owner input" -StrictValidators @("Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "package-source-url" -Group "package-source" -FieldPath "packageSource.url" -RequiredEvidence "Public or owner-approved package source URL." -Status "missing owner input" -StrictValidators @("Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "managed-package-identity" -Group "package" -FieldPath "managedPackage.id/version/sha256" -RequiredEvidence "Managed package id, version, nupkg path, and SHA256." -Status "missing owner input" -StrictValidators @("Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "runtime-package-identity" -Group "package" -FieldPath "runtimePackage.id/version/key/sha256" -RequiredEvidence "Runtime package id, version, runtime key, nupkg path, and SHA256." -Status "missing owner input" -StrictValidators @("Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "native-asset-listing" -Group "native-assets" -FieldPath "nativeAssetListing.path/sha256" -RequiredEvidence "Native asset listing file and SHA256." -Status "path missing" -StrictValidators @("Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "restore-build-run-logs" -Group "runtime-logs" -FieldPath "restore/build/run stdout/stderr" -RequiredEvidence "Restore, build, stdout, stderr, and merged transcript logs." -Status "path missing" -StrictValidators @("Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "log-sha256" -Group "runtime-logs" -FieldPath "stdoutSha256/stderrSha256/mergedTranscriptSha256" -RequiredEvidence "64-character SHA256 values matching existing logs." -Status "SHA256 invalid" -StrictValidators @("Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "runtime-exit-and-smoke-status" -Group "runtime-logs" -FieldPath "exitCode/smokeStatus" -RequiredEvidence "exitCode=0 and smokeStatus=passed from real runtime command." -Status "missing owner input" -StrictValidators @("Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof")
  New-GapField -Id "runtime-execution-timestamps" -Group "runtime-logs" -FieldPath "startedAtUtc/finishedAtUtc" -RequiredEvidence "UTC start and finish timestamps for the real clean consumer runtime smoke command." -Status "missing owner input" -StrictValidators @("Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict")
  New-GapField -Id "host-metadata" -Group "host" -FieldPath "host.os/arch/rid/gpu/driver/cuda/tensorrt/cudnn" -RequiredEvidence "OS, architecture, RID, GPU, driver, CUDA, TensorRT, and cuDNN metadata." -Status "missing owner input" -StrictValidators @("Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict")
  New-GapField -Id "owner-review" -Group "owner-review" -FieldPath "owner.name/machine/reviewedAtUtc/note" -RequiredEvidence "Owner reviewer, machine, timestamp, and review note." -Status "missing owner input" -StrictValidators @("Import-OwnerExternalProofExecutionResult.ps1 -Strict")
  New-GapField -Id "post-publish-downloaded-package-hash" -Group "post-publish" -FieldPath "postPublish.downloadedNupkgSha256" -RequiredEvidence "Downloaded public-channel nupkg SHA256." -Status "missing owner input" -StrictValidators @("Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict")
  New-GapField -Id "rollback-review" -Group "release-close" -FieldPath "rollbackReview" -RequiredEvidence "Rollback review and owner decision." -Status "missing owner input" -StrictValidators @("Test-FinalReleaseCloseOwnerApprovalContract.ps1 -Strict")
  New-GapField -Id "final-close-decision" -Group "release-close" -FieldPath "finalCloseDecision" -RequiredEvidence "Final owner close decision and release issue close approval input." -Status "missing owner input" -StrictValidators @("Test-FinalReleaseCloseApprovalRealInputFromOwnerResult.ps1 -Strict")
  New-GapField -Id "strict-validator-chain" -Group "validators" -FieldPath "strictValidators.output" -RequiredEvidence "Accepted strict validator outputs for runtime, post-publish, release evidence, and final close." -Status "strict validator not run" -StrictValidators @("Test-ReleaseEvidenceClassificationAudit.ps1 -Strict", "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady")
)

$finalPublicProofPath = @(
  New-PublicProofStep -Order 1 -Id "github-actions-run-evidence" -Title "GitHub Actions run evidence" -SourceArtifact "artifacts/final-release/github-actions-run-evidence-import-validation.json" -ValidationArtifact "artifacts/final-release/github-actions-run-evidence-import-validation.json" -CurrentState $sourceStates.githubActionsRunEvidence -RequiredReadyState "github-actions-run-evidence-ready" -OwnerAction "Owner imports a real GitHub Actions run URL, run id, head SHA, conclusion, workflow log hash, and artifact manifest hash." -StrictValidator "eng\Test-GitHubActionsRunEvidenceImport.ps1 -Strict" -BlockedReason "No real successful GitHub Actions run evidence has been imported."
  New-PublicProofStep -Order 2 -Id "owner-public-publish-result" -Title "Owner public publish result" -SourceArtifact "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" -ValidationArtifact "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" -CurrentState $sourceStates.ownerPublicPublishResultCandidate -RequiredReadyState "owner-public-publish-execution-result-candidate-ready" -OwnerAction "Owner backfills public package URL/version/SHA, publish transcript hash, GitHub release asset URL/hash, rollback review, and source GitHub Actions linkage." -StrictValidator "eng\Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict" -BlockedReason "Owner public publish result is still missing real public package and publish-result evidence."
  New-PublicProofStep -Order 3 -Id "public-package-download-proof" -Title "Public package download proof" -SourceArtifact "artifacts/final-release/public-package-download-proof-candidate-validation.json" -ValidationArtifact "artifacts/final-release/public-package-download-proof-candidate-validation.json" -CurrentState $sourceStates.publicPackageDownloadProofCandidate -RequiredReadyState "public-package-download-proof-candidate-ready" -OwnerAction "Owner downloads the managed/runtime packages from the public channel and records public URLs, package identities, SHA256 hashes, and GitHub release asset linkage." -StrictValidator "eng\Test-PublicPackageDownloadProofCandidate.ps1 -Strict" -BlockedReason "Public package download proof is missing or still candidate-only."
  New-PublicProofStep -Order 4 -Id "post-publish-clean-consumer-proof-result" -Title "Post-publish clean consumer proof result" -SourceArtifact "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" -ValidationArtifact "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" -CurrentState $sourceStates.postPublishCleanConsumerProofResult -RequiredReadyState "post-publish-clean-consumer-proof-result-validation-ready plus proofCandidateReady=true and sourceProofLinkageReady=true" -OwnerAction "Owner runs repository-external clean consumer restore/build/runtime smoke after public publication and links ready GitHub Actions, Owner publish result, and public download proof." -StrictValidator "eng\Test-PostPublishCleanConsumerProofResult.ps1 -Strict" -BlockedReason "Post-publish clean consumer proof lacks real logs, hashes, host metadata, or upstream source proof linkage."
  New-PublicProofStep -Order 5 -Id "final-public-release-closure-bridge" -Title "Final public release closure bridge" -SourceArtifact "artifacts/final-release/final-public-release-closure-bridge-validation.json" -ValidationArtifact "artifacts/final-release/final-public-release-closure-bridge-validation.json" -CurrentState $sourceStates.finalPublicReleaseClosureBridge -RequiredReadyState "final-public-release-closure-bridge-ready-for-owner-close-review" -OwnerAction "Owner refreshes the read-only final bridge after all upstream proof lanes are ready and verifies URL/version/SHA/source linkage consistency." -StrictValidator "eng\Test-FinalPublicReleaseClosureBridge.ps1 -Strict" -BlockedReason "Final bridge remains blocked until every upstream public proof lane is ready and cross-lane consistency passes."
  New-PublicProofStep -Order 6 -Id "release-issue-close-owner-decision-input" -Title "Release issue close owner decision input" -SourceArtifact "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" -ValidationArtifact "artifacts/final-release/release-issue-close-owner-decision-input-validation.json" -CurrentState $sourceStates.releaseIssueCloseOwnerDecisionInput -RequiredReadyState "release-issue-close-owner-decision-input-ready" -OwnerAction "Owner records final close decision only after final bridge ready, post-publish source linkage ready, evidence bundle hash matches, and rollback review is complete." -StrictValidator "eng\Test-ReleaseIssueCloseOwnerDecisionInput.ps1 -Strict" -BlockedReason "Release issue close decision must remain blocked until real final bridge and post-publish proof linkage are ready."
)

$forbiddenSubstitutes = @(
  "Skipped=True",
  "local smoke",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "build-only",
  "dependency-probe",
  "blocked-by-cuda-driver",
  "dashboard",
  "runbook",
  "candidate",
  "draft",
  "template",
  "preflight-only",
  "dry-run-only",
  "schema-only",
  "owner input without strict validator pass",
  "host metadata without runtime smoke",
  "package hash without existing log validation",
  "native asset listing without runtime smoke",
  "pre-publish smoke reused as post-publish proof",
  "package-managed-dry-run",
  "dashboard-only",
  "artifact-only",
  "queued workflow",
  "missing runner",
  "sidecar-only",
  "local test"
)

$record = [pscustomobject]@{
  recordKind = "final-owner-execution-one-screen-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packState = "blocked-final-owner-execution-one-screen-real-owner-input-required"
  sourceArtifacts = @($sourceArtifacts)
  sourceStates = $sourceStates
  laneCount = $lanes.Count
  blockedLaneCount = $lanes.Count
  lanes = @($lanes)
  ownerInputGapCount = $gapFields.Count
  blockedOwnerInputGapCount = $gapFields.Count
  ownerInputGapTable = @($gapFields)
  finalPublicProofPathCount = $finalPublicProofPath.Count
  blockedFinalPublicProofPathCount = $finalPublicProofPath.Count
  finalPublicProofPath = @($finalPublicProofPath)
  finalPublicProofSourceArtifacts = @($finalPublicProofPath | ForEach-Object { $_.validationArtifact })
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  nonSubstituteProofKinds = @($forbiddenSubstitutes + @("final owner execution one-screen pack", "owner one-screen execution guidance", "owner input gap table", "final public proof path", "release candidate public proof final audit"))
  boundary = "This final owner execution one-screen pack is blocked owner guidance and an owner input gap table only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. It cannot promote proof or close the release until real owner logs, package hashes, host metadata, post-publish evidence, rollback review, final close decision, and strict validators are supplied and accepted."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-one-screen-pack.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-one-screen-pack.md"
$record | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = foreach ($lane in $lanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | ``$($lane.order)`` | $(ConvertTo-MarkdownCell $lane.currentState) | ``$($lane.blocked)`` | ``$($lane.ownerRequiredFieldCount)`` | $(ConvertTo-MarkdownCell $lane.firstOwnerCommand) | $(ConvertTo-MarkdownCell $lane.boundary) |"
}
$gapRows = foreach ($field in $gapFields) {
  "| ``$(ConvertTo-MarkdownCell $field.id)`` | $(ConvertTo-MarkdownCell $field.group) | ``$(ConvertTo-MarkdownCell $field.fieldPath)`` | $(ConvertTo-MarkdownCell $field.status) | $(ConvertTo-MarkdownCell $field.requiredEvidence) |"
}
$finalPublicProofRows = foreach ($step in $finalPublicProofPath) {
  "| ``$(ConvertTo-MarkdownCell $step.id)`` | ``$($step.order)`` | $(ConvertTo-MarkdownCell $step.currentState) | $(ConvertTo-MarkdownCell $step.requiredReadyState) | $(ConvertTo-MarkdownCell $step.ownerAction) | $(ConvertTo-MarkdownCell $step.boundary) |"
}
$sourceRows = foreach ($artifact in $sourceArtifacts) {
  "- ``$artifact``"
}

$markdown = @"
# Final Owner Execution One-Screen Pack

该包把最终 Owner 执行、真实输入缺口、public docs freeze 和 release evidence 分类审计收敛到一页式 blocked handoff。它不执行发布、不运行 runtime smoke、不晋级 proof、不关闭 release issue。

| Field | Value |
|---|---|
| packState | ``$($record.packState)`` |
| laneCount | ``$($record.laneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| ownerInputGapCount | ``$($record.ownerInputGapCount)`` |
| blockedOwnerInputGapCount | ``$($record.blockedOwnerInputGapCount)`` |
| finalPublicProofPathCount | ``$($record.finalPublicProofPathCount)`` |
| blockedFinalPublicProofPathCount | ``$($record.blockedFinalPublicProofPathCount)`` |
| ownerActionRequired | ``$($record.ownerActionRequired)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Owner Execution Lanes

| Lane | Order | Current State | Blocked | Owner Fields | First Owner Command | Boundary |
|---|---:|---|---:|---:|---|---|
$($laneRows -join "`r`n")

## Owner Input Gap Table

| Gap | Group | Field Path | Status | Required Evidence |
|---|---|---|---|---|
$($gapRows -join "`r`n")

## Final Public Proof Path

| Step | Order | Current State | Required Ready State | Owner Action | Boundary |
|---|---:|---|---|---|---|
$($finalPublicProofRows -join "`r`n")

## Source Artifacts

$($sourceRows -join "`r`n")

## Boundary

$($record.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final owner execution one-screen pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "PackState=$($record.packState) Lanes=$($record.laneCount) Gaps=$($record.ownerInputGapCount) CanPublish=False CanClose=False"
