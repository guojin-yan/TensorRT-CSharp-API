[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-BlockerItem {
  param(
    [string]$Id,
    [string]$State,
    [string]$RequiredState,
    [string]$RequiredProof,
    [string]$OwnerNextAction,
    [string]$Validator,
    [string]$WhyNonSubstitute,
    [string[]]$SourceArtifacts
  )

  $ready = [string]::Equals($State, $RequiredState, [StringComparison]::OrdinalIgnoreCase)
  [pscustomobject]@{
    blockerId = $Id
    state = $State
    requiredState = $RequiredState
    ready = $ready
    blockerState = if ($ready) { "ready" } else { "blocked-owner-action-required" }
    requiredProof = $RequiredProof
    ownerNextAction = $OwnerNextAction
    validator = $Validator
    whyNonSubstitute = $WhyNonSubstitute
    sourceArtifacts = $SourceArtifacts
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
  }
}

$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$freezeManifestValidation = Read-JsonOrNull "artifacts\final-release\release-candidate-final-freeze-manifest-validation.json"
$manualHandoffValidation = Read-JsonOrNull "artifacts\final-release\public-publish-owner-manual-command-handoff-validation.json"
$finalOwnerDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-final-owner-decision-audit-validation.json"
$finalPostPublishValidation = Read-JsonOrNull "artifacts\final-release\final-post-publish-audit-pack-validation.json"
$publicPackageValidation = Read-JsonOrNull "artifacts\final-release\public-package-proof-owner-input-validation.json"
$postPublishConfirmationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-proof-owner-confirmation-validation.json"
$releaseClosePublicProofBridgeValidation = Read-JsonOrNull "artifacts\final-release\release-close-public-proof-bridge-validation.json"
$strictCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$packageConsumerRuntimeProofOwnerInputSchema = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input.schema.json"
$packageConsumerRuntimeProofForbiddenSubstituteScan = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-forbidden-substitute-scan.json"
$cleanExternalPackageConsumerOwnerRunbook = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook.json"
$cleanExternalPackageConsumerOwnerRunbookValidation = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook-validation.json"
$postPublishOwnerVerificationRunbook = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook.json"
$postPublishOwnerVerificationRunbookValidation = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook-validation.json"
$ownerExternalProofExecutionBundleValidation = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-bundle-validation.json"
$ownerExternalProofExecutionResultImportValidation = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$realProofRecordCandidateFromOwnerResultImport = Read-JsonOrNull "artifacts\final-release\real-proof-record-candidate-from-owner-result-import.json"
$realProofRecordCandidateFromOwnerResultImportValidation = Read-JsonOrNull "artifacts\final-release\real-proof-record-candidate-from-owner-result-import-validation.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$releasePackageProofBundle = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$postPublishVerificationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$runtimeProofPreflightMatrix = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-preflight-matrix.json"
$yoloVisionRealAssetOwnerProofInputValidation = Read-JsonOrNull "artifacts\user-acceptance\yolovision-real-asset-owner-proof-input-validation.json"
$sampleRunEvidenceRecordValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$ownerRuntimeSmokeFieldAlignment = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment.json"
$ownerRuntimeSmokeFieldAlignmentValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-owner-runtime-smoke-field-alignment-validation.json"

$ownerInputSchemaReady = [string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofOwnerInputSchema -Name "recordKind" -DefaultValue "") -eq "package-consumer-runtime-proof-owner-input-schema"
$forbiddenSubstituteScanState = [string](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofForbiddenSubstituteScan -Name "scanState" -DefaultValue "missing-package-consumer-runtime-proof-forbidden-substitute-scan")
$detectedForbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $packageConsumerRuntimeProofForbiddenSubstituteScan -Name "detectedForbiddenSubstituteCount" -DefaultValue -1)
$cleanOwnerInputReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "cleanOwnerInputReady" -DefaultValue $false)
$ownerInputForbiddenSubstituteFree = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputForbiddenSubstituteFree" -DefaultValue $false)
$ownerInputHashFieldsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputHashFieldsReady" -DefaultValue $false)
$ownerInputSmokeLogReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputSmokeLogReady" -DefaultValue $false)
$ownerInputCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $false)
$ownerInputBlockedReason = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputBlockedReason" -DefaultValue "missing-owner-input-readiness")
$cleanExternalRunbookState = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "runbookState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook")
$cleanExternalRunbookValidationState = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbookValidation -Name "validationState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook-validation")
$cleanExternalRunbookStepCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "stepCount" -DefaultValue 0)
$cleanExternalRunbookFailedBlockerCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbookValidation -Name "failedBlockerCount" -DefaultValue -1)
$postPublishRunbookState = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "runbookState" -DefaultValue "missing-post-publish-owner-verification-runbook")
$postPublishRunbookValidationState = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbookValidation -Name "validationState" -DefaultValue "missing-post-publish-owner-verification-runbook-validation")
$postPublishRunbookStepCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "stepCount" -DefaultValue 0)
$postPublishRunbookFailedBlockerCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbookValidation -Name "failedBlockerCount" -DefaultValue -1)
$ownerExternalProofExecutionBundleState = [string](Get-PropertyOrDefault -Object $ownerExternalProofExecutionBundleValidation -Name "validationState" -DefaultValue "missing-owner-external-proof-execution-bundle-validation")
$ownerExternalProofExecutionBundleFailedBlockers = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionBundleValidation -Name "failedBlockerCount" -DefaultValue -1)
$ownerExternalProofExecutionBundleActionRequired = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionBundleValidation -Name "failedActionRequiredCount" -DefaultValue -1)
$ownerExternalProofResultImportState = [string](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "validationState" -DefaultValue "missing-owner-external-proof-execution-result-import-validation")
$ownerExternalProofResultImportFailedBlockers = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "failedBlockerCount" -DefaultValue -1)
$ownerExternalProofResultImportFailedActionRequired = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "failedActionRequiredCount" -DefaultValue -1)
$ownerExternalProofResultImportCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$ownerExternalProofResultImportIsPostPublishProof = [bool](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "isPostPublishProof" -DefaultValue $false)
$ownerExternalProofResultImportCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$ownerExternalProofResultLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "ownerExternalProofResultLaneCount" -DefaultValue 0)
$ownerExternalProofResultBlockedLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "ownerExternalProofResultBlockedLaneCount" -DefaultValue 0)
$ownerExternalProofResultReadyLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "ownerExternalProofResultReadyLaneCount" -DefaultValue 0)
$ownerExternalProofResultPromotableLaneCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "ownerExternalProofResultPromotableLaneCount" -DefaultValue 0)
$ownerExternalProofResultFileMissingCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "fileMissingCount" -DefaultValue 0)
$ownerExternalProofResultInvalidSha256Count = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "invalidSha256Count" -DefaultValue 0)
$ownerExternalProofResultHashMismatchCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "hashMismatchCount" -DefaultValue 0)
$ownerExternalProofResultOutsideAllowedEvidenceRootCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "outsideAllowedEvidenceRootCount" -DefaultValue 0)
$ownerExternalProofResultForbiddenSubstituteFindingCount = [int](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "forbiddenSubstituteFindingCount" -DefaultValue 0)
$realProofRecordCandidateFromOwnerResultImportState = [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "candidateState" -DefaultValue "missing-real-proof-record-candidate-from-owner-result-import")
$realProofRecordCandidateFromOwnerResultImportValidationState = [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "validationState" -DefaultValue "missing-real-proof-record-candidate-from-owner-result-import-validation")
$realProofRecordCandidateFromOwnerResultImportCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "candidateCount" -DefaultValue 0)
$realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "strictValidatorReadyCandidateCount" -DefaultValue 0)
$realProofRecordCandidateFromOwnerResultImportPackageConsumerCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "packageConsumerRuntimeCandidateCount" -DefaultValue 0)
$realProofRecordCandidateFromOwnerResultImportPostPublishCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "postPublishVerificationCandidateCount" -DefaultValue 0)
$realProofRecordCandidateFromOwnerResultImportFailedBlockers = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "failedBlockerCount" -DefaultValue -1)
$realProofRecordCandidateFromOwnerResultImportActionRequired = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "failedActionRequiredCount" -DefaultValue -1)
$realProofRecordCandidateFromOwnerResultImportCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "canPromoteRuntimeProof" -DefaultValue $false)
$realProofRecordCandidateFromOwnerResultImportIsRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "isRuntimeExecutionProof" -DefaultValue $false)
$realProofRecordCandidateFromOwnerResultImportIsPostPublishProof = [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "isPostPublishProof" -DefaultValue $false)
$realProofRecordCandidateFromOwnerResultImportCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "canCloseReleaseIssue" -DefaultValue $false)
$runtimeProofPreflightEntryCount = @((Get-PropertyOrDefault -Object $runtimeProofPreflightMatrix -Name "entries" -DefaultValue @())).Count
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalRuntimeProofCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalRuntimeProofPreflight = Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeProofPreflight" -DefaultValue $null
$externalRuntimeProofPreflightAligned = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "aligned" -DefaultValue $false)
$releasePackageProofState = [string](Get-PropertyOrDefault -Object $releasePackageProofBundle -Name "proofState" -DefaultValue "missing-release-package-proof-bundle")
$releasePackageCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releasePackageProofBundle -Name "canPromoteRuntimeProof" -DefaultValue $false)
$releasePackageCanCloseIssue = [bool](Get-PropertyOrDefault -Object $releasePackageProofBundle -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishVerificationState = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishCanCloseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishIsProof = [bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$yoloVisionOwnerProofState = [string](Get-PropertyOrDefault -Object $yoloVisionRealAssetOwnerProofInputValidation -Name "validationState" -DefaultValue "missing-yolovision-owner-proof-input-validation")
$sampleRunEvidenceState = [string](Get-PropertyOrDefault -Object $sampleRunEvidenceRecordValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-record-validation")
$ownerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "alignmentState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment")
$ownerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "validationState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment-validation")
$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "runtimeSmokeStatus" -DefaultValue "Smoke=missing")
$ownerRuntimeSmokeFieldAlignmentFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "fieldCount" -DefaultValue 0)
$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "missingRequiredFieldCount" -DefaultValue -1)
$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "failedBlockerCount" -DefaultValue -1)

$blockers = @(
  New-BlockerItem `
    -Id "clean-external-package-consumer-owner-runbook" `
    -State "runbookState=$cleanExternalRunbookState; validationState=$cleanExternalRunbookValidationState; stepCount=$cleanExternalRunbookStepCount; failedBlockerCount=$cleanExternalRunbookFailedBlockerCount" `
    -RequiredState "real-clean-external-package-consumer-proof-imported" `
    -RequiredProof "Owner must execute the clean external package consumer runbook outside this repository and import real logs, SHA256 values, host metadata, and non-substitute confirmations." `
    -OwnerNextAction "Owner must follow clean-external-package-consumer-owner-runbook.md in a repository-external project, then backfill owner-external-proof-execution-result.input.json with real clean consumer evidence." `
    -Validator "Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict; Test-OwnerExternalProofExecutionResultImport.ps1 -Strict" `
    -WhyNonSubstitute "The clean external package consumer runbook is executable guidance only; it cannot replace real external runtime proof, stdout/stderr paths, merged transcript, hashes, or owner review." `
    -SourceArtifacts @("artifacts/final-release/clean-external-package-consumer-owner-runbook.json", "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json", "artifacts/final-release/owner-external-proof-execution-result.input.json")

  New-BlockerItem `
    -Id "post-publish-owner-verification-runbook" `
    -State "runbookState=$postPublishRunbookState; validationState=$postPublishRunbookValidationState; stepCount=$postPublishRunbookStepCount; failedBlockerCount=$postPublishRunbookFailedBlockerCount" `
    -RequiredState "real-post-publish-verification-proof-imported" `
    -RequiredProof "Owner must execute post-publish clean consumer verification only after real selected-channel publication and import public package URLs, package source URL, downloaded nupkg SHA256, logs, hashes, and host metadata." `
    -OwnerNextAction "Owner must follow post-publish-owner-verification-runbook.md after manual publish, then import real post-publish verification results." `
    -Validator "Test-PostPublishOwnerVerificationRunbook.ps1 -Strict; Test-PostPublishVerificationOwnerInput.ps1 -Strict; Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -WhyNonSubstitute "The post-publish runbook does not run dotnet nuget push and cannot replace real public package availability, public package source URL, downloaded nupkg SHA256, or post-publish smoke logs." `
    -SourceArtifacts @("artifacts/final-release/post-publish-owner-verification-runbook.json", "artifacts/final-release/post-publish-owner-verification-runbook-validation.json", "artifacts/final-release/owner-external-proof-execution-result.input.json")

  New-BlockerItem `
    -Id "package-consumer-owner-runtime-smoke-field-alignment" `
    -State "alignmentState=$ownerRuntimeSmokeFieldAlignmentState; validationState=$ownerRuntimeSmokeFieldAlignmentValidationState; runtimeSmokeStatus=$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus; fieldCount=$ownerRuntimeSmokeFieldAlignmentFieldCount; missingRequiredFields=$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount; failedBlockers=$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount" `
    -RequiredState "real-compatible-host-runtime-smoke-imported" `
    -RequiredProof "Owner-facing runtime smoke fields are aligned, but real compatible-host runtime smoke logs, hashes, host metadata, and strict validators are still required before proof promotion." `
    -OwnerNextAction "Owner must execute the clean external package consumer runtime smoke on a compatible CUDA/TensorRT host and import real evidence; field alignment alone is non-proof." `
    -Validator "Test-PackageConsumerOwnerRuntimeSmokeFieldAlignment.ps1 -Strict; Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" `
    -WhyNonSubstitute "Field alignment cannot prove runtime execution and cannot replace real smoke logs, package hashes, host metadata, or owner review." `
    -SourceArtifacts @("artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json", "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json")

  New-BlockerItem `
    -Id "package-consumer-runtime-ownerproof-schema-scan" `
    -State "ownerInputSchemaReady=$ownerInputSchemaReady; forbiddenSubstituteScanState=$forbiddenSubstituteScanState; detectedForbiddenSubstituteCount=$detectedForbiddenSubstituteCount; cleanOwnerInputReady=$cleanOwnerInputReady; ownerInputForbiddenSubstituteFree=$ownerInputForbiddenSubstituteFree; ownerInputHashFieldsReady=$ownerInputHashFieldsReady; ownerInputSmokeLogReady=$ownerInputSmokeLogReady; ownerInputCanPromoteRuntimeProof=$ownerInputCanPromoteRuntimeProof" `
    -RequiredState "real-clean-consumer-owner-input-ready-with-no-forbidden-substitutes" `
    -RequiredProof "Owner-filled package-consumer runtime input with public source, no ProjectReference, no local feed, no direct .nupkg, matching hashes, and real smoke log." `
    -OwnerNextAction "Owner must fill the schema-backed input and rerun forbidden substitute scan after clean external consumer runtime smoke." `
    -Validator "Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict; Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1" `
    -WhyNonSubstitute "Schema-ready and forbidden-substitute scan output cannot prove runtime execution or close release issue; they only block invalid substitutes." `
    -SourceArtifacts @("artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json", "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json")

  New-BlockerItem `
    -Id "owner-external-proof-execution-bundle" `
    -State "validationState=$ownerExternalProofExecutionBundleState; failedBlockerCount=$ownerExternalProofExecutionBundleFailedBlockers; failedActionRequiredCount=$ownerExternalProofExecutionBundleActionRequired; runtimeProofPreflightEntryCount=$runtimeProofPreflightEntryCount" `
    -RequiredState "owner-external-proof-execution-bundle-ready-with-real-results" `
    -RequiredProof "Owner execution bundle with RuntimeProofPreflight options, zero structural blockers, and real owner results imported for package-consumer-runtime, post-publish, release-close, and sample proof lanes." `
    -OwnerNextAction "Owner must execute each bundle lane, choose a concrete RuntimeProofPreflight runtime for package-consumer proof, capture logs/hashes/host metadata, then import real results." `
    -Validator "Export-OwnerExternalProofExecutionBundle.ps1; Test-OwnerExternalProofExecutionBundle.ps1 -Strict; Import-OwnerExternalProofExecutionResult.ps1" `
    -WhyNonSubstitute "An execution bundle cannot replace owner execution logs, SHA256 files, host metadata, external runtime proof, post-publish proof, or release close approval." `
    -SourceArtifacts @("artifacts/final-release/owner-external-proof-execution-bundle.json", "artifacts/final-release/owner-external-proof-execution-bundle-validation.json", "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json")

  New-BlockerItem `
    -Id "owner-external-proof-result-import" `
    -State "validationState=$ownerExternalProofResultImportState; failedBlockerCount=$ownerExternalProofResultImportFailedBlockers; failedActionRequiredCount=$ownerExternalProofResultImportFailedActionRequired; laneCount=$ownerExternalProofResultLaneCount; readyLaneCount=$ownerExternalProofResultReadyLaneCount; blockedLaneCount=$ownerExternalProofResultBlockedLaneCount; promotableLaneCount=$ownerExternalProofResultPromotableLaneCount; fileMissingCount=$ownerExternalProofResultFileMissingCount; invalidSha256Count=$ownerExternalProofResultInvalidSha256Count; hashMismatchCount=$ownerExternalProofResultHashMismatchCount; outsideAllowedEvidenceRootCount=$ownerExternalProofResultOutsideAllowedEvidenceRootCount; forbiddenSubstituteFindingCount=$ownerExternalProofResultForbiddenSubstituteFindingCount; canPromoteRuntimeProof=$ownerExternalProofResultImportCanPromoteRuntimeProof; isPostPublishProof=$ownerExternalProofResultImportIsPostPublishProof; canCloseReleaseIssue=$ownerExternalProofResultImportCanCloseReleaseIssue" `
    -RequiredState "owner-external-proof-result-import-ready-with-strict-real-proof-records" `
    -RequiredProof "Owner-provided execution results for all lanes with existing logs, matching SHA256, non-substitute confirmations, exitCode=0, owner review, and downstream strict real proof validators." `
    -OwnerNextAction "Owner must fill owner-external-proof-execution-result.input.json with real external execution results and rerun Import/Test-OwnerExternalProofExecutionResultImport plus real proof validators." `
    -Validator "Import-OwnerExternalProofExecutionResult.ps1; Test-OwnerExternalProofExecutionResultImport.ps1 -Strict; Test-RealExternalProofRecordImportValidator.ps1 -Strict" `
    -WhyNonSubstitute "Imported result metadata cannot become runtime proof, post-publish proof, or release close approval until strict validators consume real logs/hashes and promote concrete proof records." `
    -SourceArtifacts @("artifacts/final-release/owner-external-proof-execution-result-import.json", "artifacts/final-release/owner-external-proof-execution-result-import-validation.json")

  New-BlockerItem `
    -Id "real-proof-record-candidate-from-owner-result-import" `
    -State "candidateState=$realProofRecordCandidateFromOwnerResultImportState; validationState=$realProofRecordCandidateFromOwnerResultImportValidationState; candidateCount=$realProofRecordCandidateFromOwnerResultImportCandidateCount; strictValidatorReadyCandidateCount=$realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount; packageConsumerRuntimeCandidateCount=$realProofRecordCandidateFromOwnerResultImportPackageConsumerCandidateCount; postPublishVerificationCandidateCount=$realProofRecordCandidateFromOwnerResultImportPostPublishCandidateCount; failedBlockerCount=$realProofRecordCandidateFromOwnerResultImportFailedBlockers; failedActionRequiredCount=$realProofRecordCandidateFromOwnerResultImportActionRequired; canPromoteRuntimeProof=$realProofRecordCandidateFromOwnerResultImportCanPromoteRuntimeProof; isRuntimeExecutionProof=$realProofRecordCandidateFromOwnerResultImportIsRuntimeExecutionProof; isPostPublishProof=$realProofRecordCandidateFromOwnerResultImportIsPostPublishProof; canCloseReleaseIssue=$realProofRecordCandidateFromOwnerResultImportCanCloseReleaseIssue" `
    -RequiredState "strict-real-proof-record-validator-ready-after-owner-candidate-review" `
    -RequiredProof "Ready owner result import candidates consumed by strict real proof validators and promotion guard, without treating candidates as runtime, post-publish, or close proof." `
    -OwnerNextAction "After owner result import has ready contracts, run Export/Test-RealProofRecordCandidateFromOwnerResultImport, then strict real proof validator and promotion guard before any proof promotion." `
    -Validator "Export-RealProofRecordCandidateFromOwnerResultImport.ps1; Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict; Test-RealProofRecordValidator.ps1 -Strict" `
    -WhyNonSubstitute "A ready owner-result candidate cannot substitute real runtime proof, post-publish verification, package publish, or release close approval; it is only strict-validator input." `
    -SourceArtifacts @("artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json", "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json")

  New-BlockerItem `
    -Id "external-runtime-proof-validation" `
    -State "validationState=$externalRuntimeProofState; canPromoteRuntimeProof=$externalRuntimeProofCanPromote; runtimeProofPreflightAligned=$externalRuntimeProofPreflightAligned" `
    -RequiredState "real-runtime-proof" `
    -RequiredProof "Validator-passing external-runtime-proof-record.json from clean package consumer, with real log/hash/host metadata and RuntimeProofPreflight alignment." `
    -OwnerNextAction "Owner must fill external-runtime-proof-record.json from a compatible CUDA/TensorRT host and run Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof." `
    -Validator "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -WhyNonSubstitute "RuntimeProofPreflight, draft/template/example records, local feed, dependency probe, and blocked-by-cuda-driver cannot prove package-consumer runtime execution." `
    -SourceArtifacts @("artifacts/final-release/external-runtime-proof-validation.json", "artifacts/final-release/external-runtime-proof-record.input-template.json", "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json")

  New-BlockerItem `
    -Id "release-package-proof-bundle" `
    -State "proofState=$releasePackageProofState; canPromoteRuntimeProof=$releasePackageCanPromoteRuntimeProof; canCloseReleaseIssue=$releasePackageCanCloseIssue" `
    -RequiredState "release-package-proof-bundle-ready-with-real-owner-proof" `
    -RequiredProof "Release package proof bundle refreshed after real external runtime proof, public/package proof, and post-publish proof all pass strict validators." `
    -OwnerNextAction "After real owner proof import, rerun Export-ReleasePackageProofBundle.ps1 and review proof fields before any close decision." `
    -Validator "Export-ReleasePackageProofBundle.ps1; Test-FinalReleaseCloseBlockerDashboard.ps1 -Strict" `
    -WhyNonSubstitute "The bundle aggregates proof state and cannot replace real package-consumer runtime proof, public package proof, or post-publish proof." `
    -SourceArtifacts @("artifacts/final-release/release-package-proof-bundle.json")

  New-BlockerItem `
    -Id "release-candidate-final-freeze-manifest" `
    -State ([string](Get-PropertyOrDefault -Object $freezeManifestValidation -Name "validationState" -DefaultValue "missing-release-candidate-final-freeze-manifest-validation")) `
    -RequiredState "release-candidate-final-freeze-manifest-ready-for-owner-handoff" `
    -RequiredProof "Local release-facing artifact hash manifest and classification-clean proof boundary review." `
    -OwnerNextAction "Generate and review freeze manifest before copying any publish placeholder." `
    -Validator "Test-ReleaseCandidateFinalFreezeManifest.ps1 -Strict" `
    -WhyNonSubstitute "A freeze manifest is a local hash inventory and cannot prove public package availability or runtime execution." `
    -SourceArtifacts @("artifacts/final-release/release-candidate-final-freeze-manifest-validation.json")

  New-BlockerItem `
    -Id "public-publish-owner-manual-command-handoff" `
    -State ([string](Get-PropertyOrDefault -Object $manualHandoffValidation -Name "validationState" -DefaultValue "missing-public-publish-owner-manual-command-handoff-validation")) `
    -RequiredState "public-publish-owner-manual-command-handoff-ready" `
    -RequiredProof "Owner-reviewed manual command handoff with notExecutedByAutomation=true placeholders." `
    -OwnerNextAction "Owner must manually authorize, materialize, and execute public publish outside automation." `
    -Validator "Test-PublicPublishOwnerManualCommandHandoff.ps1 -Strict" `
    -WhyNonSubstitute "A command handoff can list dotnet nuget push placeholders but cannot execute publish or prove package availability." `
    -SourceArtifacts @("artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json")

  New-BlockerItem `
    -Id "public-package-proof-owner-input" `
    -State ([string](Get-PropertyOrDefault -Object $publicPackageValidation -Name "validationState" -DefaultValue "missing-public-package-proof-owner-input-validation")) `
    -RequiredState "public-package-proof-owner-input-ready" `
    -RequiredProof "Real public package URL, registry source, nupkg SHA256, published timestamp, and owner review fields." `
    -OwnerNextAction "Owner must backfill real public package proof after manual publish." `
    -Validator "Test-PublicPackageProofOwnerInput.ps1 -Strict" `
    -WhyNonSubstitute "Local package files, dry runs, templates, and command placeholders cannot prove public package availability." `
    -SourceArtifacts @("artifacts/final-release/public-package-proof-owner-input-validation.json")

  New-BlockerItem `
    -Id "post-publish-proof-owner-confirmation" `
    -State ([string](Get-PropertyOrDefault -Object $postPublishConfirmationValidation -Name "validationState" -DefaultValue "missing-post-publish-proof-owner-confirmation-validation")) `
    -RequiredState "post-publish-proof-owner-confirmation-ready" `
    -RequiredProof "Post-publish clean consumer proof and owner confirmation gates all ready." `
    -OwnerNextAction "Owner must run clean consumer proof from public source and confirm all post-publish gates." `
    -Validator "Test-PostPublishProofOwnerConfirmation.ps1 -Strict" `
    -WhyNonSubstitute "Owner confirmation cannot become proof until real public-channel logs, hashes, and host metadata exist." `
    -SourceArtifacts @("artifacts/final-release/post-publish-proof-owner-confirmation-validation.json")

  New-BlockerItem `
    -Id "post-publish-verification-validation" `
    -State "validationState=$postPublishVerificationState; isPostPublishVerificationProof=$postPublishIsProof; canCloseReleaseIssue=$postPublishCanCloseIssue" `
    -RequiredState "post-publish-verification-proof" `
    -RequiredProof "Real post-publish clean consumer verification from selected public/private package source with matching logs, hashes, stdout/stderr summaries, and host metadata." `
    -OwnerNextAction "Owner must execute post-publish clean consumer verification only after selected-channel package publication and import the real result." `
    -Validator "Test-PostPublishVerificationOwnerInput.ps1 -Strict; Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -WhyNonSubstitute "Pre-publish local packages, direct nupkg references, ProjectReference, local feed, and template-only records cannot prove post-publish availability." `
    -SourceArtifacts @("artifacts/final-release/post-publish-verification-validation.json")

  New-BlockerItem `
    -Id "yolovision-real-model-proof-boundary" `
    -State "yoloVisionOwnerProofState=$yoloVisionOwnerProofState; sampleRunEvidenceState=$sampleRunEvidenceState" `
    -RequiredState "real-model-runtime-proof-ready-but-not-release-proof" `
    -RequiredProof "YoloVision real model evidence may prove sample real-model-runtime only after owner assets/logs/hashes validate; it still cannot replace package-consumer-runtime or post-publish proof." `
    -OwnerNextAction "Owner must backfill YOLO-family model, labels, input/output, license, logs, hashes, and sample-run evidence separately from release proof." `
    -Validator "Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict; Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -WhyNonSubstitute "YoloVision matrix, asset templates, sidecars, sample logs, TensorRtExec reports, and real-model-runtime evidence cannot replace release package runtime proof or post-publish proof." `
    -SourceArtifacts @("samples/YoloVision", "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json", "artifacts/user-acceptance/sample-run-evidence-record-validation.json")

  New-BlockerItem `
    -Id "release-close-public-proof-bridge" `
    -State ([string](Get-PropertyOrDefault -Object $releaseClosePublicProofBridgeValidation -Name "validationState" -DefaultValue "missing-release-close-public-proof-bridge-validation")) `
    -RequiredState "release-close-public-proof-ready" `
    -RequiredProof "All public package and post-publish proof bridge gates ready." `
    -OwnerNextAction "Owner must complete public package proof, post-publish proof, and close owner bridge records." `
    -Validator "Test-ReleaseClosePublicProofBridge.ps1 -Strict" `
    -WhyNonSubstitute "A bridge aggregates gate status and cannot replace real proof artifacts." `
    -SourceArtifacts @("artifacts/final-release/release-close-public-proof-bridge-validation.json")

  New-BlockerItem `
    -Id "final-post-publish-audit-pack" `
    -State ([string](Get-PropertyOrDefault -Object $finalPostPublishValidation -Name "validationState" -DefaultValue "missing-final-post-publish-audit-pack-validation")) `
    -RequiredState "final-post-publish-audit-ready" `
    -RequiredProof "All final post-publish audit lanes ready." `
    -OwnerNextAction "Owner must complete public package, post-publish, package consumer runtime, public bridge, and strict close lanes." `
    -Validator "Test-FinalPostPublishAuditPack.ps1 -Strict" `
    -WhyNonSubstitute "An audit pack is lane aggregation only and cannot prove post-publish behavior." `
    -SourceArtifacts @("artifacts/final-release/final-post-publish-audit-pack-validation.json")

  New-BlockerItem `
    -Id "release-issue-close-final-owner-decision-audit" `
    -State ([string](Get-PropertyOrDefault -Object $finalOwnerDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-close-final-owner-decision-audit-validation")) `
    -RequiredState "release-issue-close-final-owner-decision-ready" `
    -RequiredProof "All final owner decision gates ready." `
    -OwnerNextAction "Owner must complete public proof, close candidate, final close decision, strict close, and classification gates." `
    -Validator "Test-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1 -Strict" `
    -WhyNonSubstitute "Final owner decision audit is a gate summary and cannot approve close by itself." `
    -SourceArtifacts @("artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json")

  New-BlockerItem `
    -Id "strict-release-close-validator" `
    -State ([string](Get-PropertyOrDefault -Object $strictCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")) `
    -RequiredState "ready-for-owner-release-issue-close" `
    -RequiredProof "Strict close record ready with real public package and post-publish proof references." `
    -OwnerNextAction "Owner must run Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady only after all real proof records validate." `
    -Validator "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" `
    -WhyNonSubstitute "Strict close validator remains the final gate; dashboards, candidates, and audits cannot close the issue." `
    -SourceArtifacts @("artifacts/final-release/release-issue-close-record-validation.json")

  New-BlockerItem `
    -Id "release-evidence-classification-audit" `
    -State ([string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")) `
    -RequiredState "classification-audit-passed-non-proof-boundaries-intact" `
    -RequiredProof "Classification audit findingCount=0 with non-proof boundaries intact." `
    -OwnerNextAction "Keep every freeze/handoff/dashboard item failed/non-proof until real Owner proof exists." `
    -Validator "Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" `
    -WhyNonSubstitute "Classification audit proves boundary discipline only and cannot prove runtime, post-publish, or close approval." `
    -SourceArtifacts @("artifacts/final-release/release-evidence-classification-audit.json")
)

$blocked = @($blockers | Where-Object { -not [bool]$_.ready })
$ready = @($blockers | Where-Object { [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "final-release-close-blocker-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  dashboardState = if ($blocked.Count -eq 0) { "final-release-close-blocker-dashboard-ready" } else { "blocked-final-release-close-owner-action-required" }
  blockerCount = $blockers.Count
  blockedBlockerCount = $blocked.Count
  readyBlockerCount = $ready.Count
  ownerInputSchemaReady = $ownerInputSchemaReady
  forbiddenSubstituteScanState = $forbiddenSubstituteScanState
  detectedForbiddenSubstituteCount = $detectedForbiddenSubstituteCount
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputForbiddenSubstituteFree = $ownerInputForbiddenSubstituteFree
  ownerInputHashFieldsReady = $ownerInputHashFieldsReady
  ownerInputSmokeLogReady = $ownerInputSmokeLogReady
  ownerInputCanPromoteRuntimeProof = $ownerInputCanPromoteRuntimeProof
  ownerInputBlockedReason = $ownerInputBlockedReason
  cleanExternalRunbookState = $cleanExternalRunbookState
  cleanExternalRunbookValidationState = $cleanExternalRunbookValidationState
  cleanExternalRunbookStepCount = $cleanExternalRunbookStepCount
  cleanExternalRunbookFailedBlockerCount = $cleanExternalRunbookFailedBlockerCount
  postPublishRunbookState = $postPublishRunbookState
  postPublishRunbookValidationState = $postPublishRunbookValidationState
  postPublishRunbookStepCount = $postPublishRunbookStepCount
  postPublishRunbookFailedBlockerCount = $postPublishRunbookFailedBlockerCount
  ownerExternalProofExecutionBundleState = $ownerExternalProofExecutionBundleState
  ownerExternalProofExecutionBundleFailedBlockers = $ownerExternalProofExecutionBundleFailedBlockers
  ownerExternalProofExecutionBundleActionRequired = $ownerExternalProofExecutionBundleActionRequired
  ownerExternalProofResultImportState = $ownerExternalProofResultImportState
  ownerExternalProofResultImportFailedBlockers = $ownerExternalProofResultImportFailedBlockers
  ownerExternalProofResultImportFailedActionRequired = $ownerExternalProofResultImportFailedActionRequired
  ownerExternalProofResultImportCanPromoteRuntimeProof = $ownerExternalProofResultImportCanPromoteRuntimeProof
  ownerExternalProofResultImportIsPostPublishProof = $ownerExternalProofResultImportIsPostPublishProof
  ownerExternalProofResultImportCanCloseReleaseIssue = $ownerExternalProofResultImportCanCloseReleaseIssue
  ownerExternalProofResultLaneCount = $ownerExternalProofResultLaneCount
  ownerExternalProofResultBlockedLaneCount = $ownerExternalProofResultBlockedLaneCount
  ownerExternalProofResultReadyLaneCount = $ownerExternalProofResultReadyLaneCount
  ownerExternalProofResultPromotableLaneCount = $ownerExternalProofResultPromotableLaneCount
  ownerExternalProofResultFileMissingCount = $ownerExternalProofResultFileMissingCount
  ownerExternalProofResultInvalidSha256Count = $ownerExternalProofResultInvalidSha256Count
  ownerExternalProofResultHashMismatchCount = $ownerExternalProofResultHashMismatchCount
  ownerExternalProofResultOutsideAllowedEvidenceRootCount = $ownerExternalProofResultOutsideAllowedEvidenceRootCount
  ownerExternalProofResultForbiddenSubstituteFindingCount = $ownerExternalProofResultForbiddenSubstituteFindingCount
  realProofRecordCandidateFromOwnerResultImportState = $realProofRecordCandidateFromOwnerResultImportState
  realProofRecordCandidateFromOwnerResultImportValidationState = $realProofRecordCandidateFromOwnerResultImportValidationState
  realProofRecordCandidateFromOwnerResultImportCandidateCount = $realProofRecordCandidateFromOwnerResultImportCandidateCount
  realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount = $realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount
  realProofRecordCandidateFromOwnerResultImportPackageConsumerCandidateCount = $realProofRecordCandidateFromOwnerResultImportPackageConsumerCandidateCount
  realProofRecordCandidateFromOwnerResultImportPostPublishCandidateCount = $realProofRecordCandidateFromOwnerResultImportPostPublishCandidateCount
  realProofRecordCandidateFromOwnerResultImportFailedBlockers = $realProofRecordCandidateFromOwnerResultImportFailedBlockers
  realProofRecordCandidateFromOwnerResultImportActionRequired = $realProofRecordCandidateFromOwnerResultImportActionRequired
  realProofRecordCandidateFromOwnerResultImportCanPromoteRuntimeProof = $realProofRecordCandidateFromOwnerResultImportCanPromoteRuntimeProof
  realProofRecordCandidateFromOwnerResultImportIsRuntimeExecutionProof = $realProofRecordCandidateFromOwnerResultImportIsRuntimeExecutionProof
  realProofRecordCandidateFromOwnerResultImportIsPostPublishProof = $realProofRecordCandidateFromOwnerResultImportIsPostPublishProof
  realProofRecordCandidateFromOwnerResultImportCanCloseReleaseIssue = $realProofRecordCandidateFromOwnerResultImportCanCloseReleaseIssue
  runtimeProofPreflightEntryCount = $runtimeProofPreflightEntryCount
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofCanPromote = $externalRuntimeProofCanPromote
  externalRuntimeProofPreflightAligned = $externalRuntimeProofPreflightAligned
  releasePackageProofState = $releasePackageProofState
  releasePackageCanPromoteRuntimeProof = $releasePackageCanPromoteRuntimeProof
  postPublishVerificationState = $postPublishVerificationState
  postPublishIsProof = $postPublishIsProof
  yoloVisionOwnerProofState = $yoloVisionOwnerProofState
  sampleRunEvidenceState = $sampleRunEvidenceState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = $ownerRuntimeSmokeFieldAlignmentState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = $ownerRuntimeSmokeFieldAlignmentValidationState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = $ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount = $ownerRuntimeSmokeFieldAlignmentFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = $ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount = $ownerRuntimeSmokeFieldAlignmentFailedBlockerCount
  blockers = $blockers
  sourceArtifacts = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-candidate-final-freeze-manifest-validation.json",
    "artifacts/final-release/public-publish-owner-manual-command-handoff-validation.json",
    "artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json",
    "artifacts/final-release/final-post-publish-audit-pack-validation.json",
    "artifacts/final-release/public-package-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json",
    "artifacts/final-release/release-close-public-proof-bridge-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/release-evidence-classification-audit.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
    "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
    "artifacts/final-release/post-publish-owner-verification-runbook.json",
    "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
    "artifacts/final-release/owner-external-proof-execution-result.input.json",
    "artifacts/final-release/owner-external-proof-execution-bundle-validation.json",
    "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
    "artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json",
    "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/release-package-proof-bundle.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json",
    "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
    "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json",
    "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
    "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Final release close blocker dashboard is owner-action status aggregation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "final-release-close-blocker-dashboard.json"
$markdownPath = Join-Path $artifactRoot "final-release-close-blocker-dashboard.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.blockers | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.blockerId) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` | $(ConvertTo-MarkdownCell $_.ownerNextAction) | $(ConvertTo-MarkdownCell $_.validator) |"
}

$markdown = @"
# Final Release Close Blocker Dashboard

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| dashboardState | ``$($record.dashboardState)`` |
| blockerCount | ``$($record.blockerCount)`` |
| blockedBlockerCount | ``$($record.blockedBlockerCount)`` |
| readyBlockerCount | ``$($record.readyBlockerCount)`` |
| ownerInputSchemaReady | ``$($record.ownerInputSchemaReady)`` |
| forbiddenSubstituteScanState | ``$($record.forbiddenSubstituteScanState)`` |
| detectedForbiddenSubstituteCount | ``$($record.detectedForbiddenSubstituteCount)`` |
| cleanOwnerInputReady | ``$($record.cleanOwnerInputReady)`` |
| ownerInputForbiddenSubstituteFree | ``$($record.ownerInputForbiddenSubstituteFree)`` |
| ownerInputHashFieldsReady | ``$($record.ownerInputHashFieldsReady)`` |
| ownerInputSmokeLogReady | ``$($record.ownerInputSmokeLogReady)`` |
| ownerExternalProofExecutionBundleState | ``$($record.ownerExternalProofExecutionBundleState)`` |
| ownerExternalProofExecutionBundleFailedBlockers | ``$($record.ownerExternalProofExecutionBundleFailedBlockers)`` |
| ownerExternalProofResultImportState | ``$($record.ownerExternalProofResultImportState)`` |
| ownerExternalProofResultLaneCount | ``$($record.ownerExternalProofResultLaneCount)`` |
| ownerExternalProofResultBlockedLaneCount | ``$($record.ownerExternalProofResultBlockedLaneCount)`` |
| ownerExternalProofResultReadyLaneCount | ``$($record.ownerExternalProofResultReadyLaneCount)`` |
| ownerExternalProofResultPromotableLaneCount | ``$($record.ownerExternalProofResultPromotableLaneCount)`` |
| realProofRecordCandidateFromOwnerResultImportState | ``$($record.realProofRecordCandidateFromOwnerResultImportState)`` |
| realProofRecordCandidateFromOwnerResultImportValidationState | ``$($record.realProofRecordCandidateFromOwnerResultImportValidationState)`` |
| realProofRecordCandidateFromOwnerResultImportCandidateCount | ``$($record.realProofRecordCandidateFromOwnerResultImportCandidateCount)`` |
| realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount | ``$($record.realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount)`` |
| runtimeProofPreflightEntryCount | ``$($record.runtimeProofPreflightEntryCount)`` |
| externalRuntimeProofState | ``$($record.externalRuntimeProofState)`` |
| externalRuntimeProofPreflightAligned | ``$($record.externalRuntimeProofPreflightAligned)`` |
| releasePackageProofState | ``$($record.releasePackageProofState)`` |
| postPublishVerificationState | ``$($record.postPublishVerificationState)`` |
| yoloVisionOwnerProofState | ``$($record.yoloVisionOwnerProofState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Blockers

| Blocker | Current State | Required State | Ready | Owner Next Action | Validator |
|---|---|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release close blocker dashboard written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "DashboardState=$($record.dashboardState) Blockers=$($record.blockerCount) Blocked=$($record.blockedBlockerCount) Ready=$($record.readyBlockerCount)"
