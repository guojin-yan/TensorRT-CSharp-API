[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

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

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-StringArray {
  param([AllowNull()][object]$Value)
  return @(ConvertTo-Array $Value | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-RecordState {
  param([AllowNull()][object]$Record)
  foreach ($name in @("worklistState", "packageState", "runbookState", "executionPackState", "crossCheckState", "checkpointState", "dashboardState", "convergenceState", "packState", "validationState")) {
    $value = [string](Get-PropertyOrDefault -Object $Record -Name $name -DefaultValue "")
    if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
  }
  return "missing-state"
}

function New-SourceRecordSummary {
  param(
    [string]$Id,
    [string]$ArtifactPath,
    [AllowNull()][object]$Record,
    [string]$CountField,
    [string]$BlockedCountField
  )

  [pscustomobject]@{
    id = $Id
    artifactPath = $ArtifactPath
    recordKind = [string](Get-PropertyOrDefault -Object $Record -Name "recordKind" -DefaultValue "missing")
    state = Get-RecordState -Record $Record
    count = [int](Get-PropertyOrDefault -Object $Record -Name $CountField -DefaultValue 0)
    blockedCount = [int](Get-PropertyOrDefault -Object $Record -Name $BlockedCountField -DefaultValue 0)
    performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
    canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "canPromoteRuntimeProof" -DefaultValue $false)
    canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
    canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
    isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
    isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
    isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
  }
}

function New-OrderStep {
  param(
    [int]$Sequence,
    [string]$Id,
    [string]$Title,
    [string]$SourceArtifact,
    [string]$SourceState,
    [string[]]$InputArtifacts,
    [string[]]$OutputArtifacts,
    [string[]]$ValidatorScripts,
    [string]$OwnerAction,
    [string]$BlockedUntil,
    [string]$FailureStopRule
  )

  [pscustomobject]@{
    sequence = $Sequence
    id = $Id
    title = $Title
    sourceArtifact = $SourceArtifact
    sourceState = $SourceState
    ownerAction = $OwnerAction
    inputArtifacts = @($InputArtifacts)
    inputArtifactCount = @($InputArtifacts).Count
    outputArtifacts = @($OutputArtifacts)
    outputArtifactCount = @($OutputArtifacts).Count
    validatorScripts = @($ValidatorScripts)
    validatorScriptCount = @($ValidatorScripts).Count
    blockedUntil = $BlockedUntil
    failureStopRule = $FailureStopRule
    ownerExecutionOnly = $true
    notExecutedByAutomation = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner execution order guidance only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$finalOwnerProofActionWorklist = Read-JsonOrNull "artifacts\final-release\final-owner-proof-action-worklist.json"
$finalOwnerExecutionPackage = Read-JsonOrNull "artifacts\final-release\final-owner-execution-package.json"
$cleanExternalPackageConsumerOwnerRunbook = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook.json"
$postPublishOwnerVerificationRunbook = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook.json"
$publicPublishFinalOwnerExecutionPack = Read-JsonOrNull "artifacts\final-release\public-publish-final-owner-execution-pack.json"
$publicPublishCommandCrossCheck = Read-JsonOrNull "artifacts\final-release\public-publish-command-cross-check.json"
$finalOwnerCloseReadinessCheckpoint = Read-JsonOrNull "artifacts\final-release\final-owner-close-readiness-checkpoint.json"
$finalReleaseCloseBlockerDashboard = Read-JsonOrNull "artifacts\final-release\final-release-close-blocker-dashboard.json"
$ownerInputContractConvergence = Read-JsonOrNull "artifacts\final-release\owner-input-contract-convergence.json"
$ownerInputContractConvergenceValidation = Read-JsonOrNull "artifacts\final-release\owner-input-contract-convergence-validation.json"
$finalOwnerExecutionOneScreenPackPath = Join-Path $RepositoryRoot "artifacts\final-release\final-owner-execution-one-screen-pack.json"
$finalOwnerExecutionOneScreenPackValidationPath = Join-Path $RepositoryRoot "artifacts\final-release\final-owner-execution-one-screen-pack-validation.json"
if (-not (Test-Path -LiteralPath $finalOwnerExecutionOneScreenPackPath -PathType Leaf) -or -not (Test-Path -LiteralPath $finalOwnerExecutionOneScreenPackValidationPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerExecutionOneScreenPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  & (Join-Path $RepositoryRoot "eng\Test-FinalOwnerExecutionOneScreenPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict
}
$finalOwnerExecutionOneScreenPack = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack.json"
$finalOwnerExecutionOneScreenPackValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-one-screen-pack-validation.json"

function Get-OneScreenMetric {
  param([string]$Name, [AllowNull()][object]$DefaultValue)
  $value = Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPackValidation -Name $Name -DefaultValue $null
  if ($null -ne $value) { return $value }
  return Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name $Name -DefaultValue $DefaultValue
}

$releaseCloseRealInputChain = @(ConvertTo-Array (Get-PropertyOrDefault -Object $finalOwnerExecutionOneScreenPack -Name "releaseCloseRealInputChain" -DefaultValue @()))
$releaseCloseRealInputChainProjection = foreach ($step in $releaseCloseRealInputChain) {
  [pscustomobject]@{
    order = [int](Get-PropertyOrDefault -Object $step -Name "order" -DefaultValue 0)
    id = [string](Get-PropertyOrDefault -Object $step -Name "id" -DefaultValue "")
    title = [string](Get-PropertyOrDefault -Object $step -Name "title" -DefaultValue "")
    currentState = [string](Get-PropertyOrDefault -Object $step -Name "currentState" -DefaultValue "")
    requiredReadyState = [string](Get-PropertyOrDefault -Object $step -Name "requiredReadyState" -DefaultValue "")
    requiredFieldCount = [int](Get-PropertyOrDefault -Object $step -Name "requiredFieldCount" -DefaultValue 0)
    rejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $step -Name "rejectedSubstituteCount" -DefaultValue 0)
    sourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $step -Name "sourceReadinessSignalCount" -DefaultValue 0)
    blockedRealInputCount = [int](Get-PropertyOrDefault -Object $step -Name "blockedRealInputCount" -DefaultValue 0)
    proofCandidateReady = [bool](Get-PropertyOrDefault -Object $step -Name "proofCandidateReady" -DefaultValue $false)
    sourceLinkageReady = [bool](Get-PropertyOrDefault -Object $step -Name "sourceLinkageReady" -DefaultValue $false)
    strictValidator = [string](Get-PropertyOrDefault -Object $step -Name "strictValidator" -DefaultValue "")
    blockedReason = [string](Get-PropertyOrDefault -Object $step -Name "blockedReason" -DefaultValue "")
    boundary = [string](Get-PropertyOrDefault -Object $step -Name "boundary" -DefaultValue "")
  }
}

$releaseCloseRealInputChainCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainCount" -DefaultValue 0)
$releaseCloseRealInputChainRequiredFieldCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainRequiredFieldCount" -DefaultValue 0)
$releaseCloseRealInputChainRejectedSubstituteCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainRejectedSubstituteCount" -DefaultValue 0)
$releaseCloseRealInputChainSourceReadinessSignalCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainSourceReadinessSignalCount" -DefaultValue 0)
$releaseCloseRealInputChainBlockedRealInputCount = [int](Get-OneScreenMetric -Name "releaseCloseRealInputChainBlockedRealInputCount" -DefaultValue 0)
$publicPackageDownloadProofRequiredFieldCount = [int](Get-OneScreenMetric -Name "publicPackageDownloadProofRequiredFieldCount" -DefaultValue 0)
$publicPackageDownloadProofRejectedSubstituteCount = [int](Get-OneScreenMetric -Name "publicPackageDownloadProofRejectedSubstituteCount" -DefaultValue 0)
$publicPackageDownloadProofSourceReadinessSignalCount = [int](Get-OneScreenMetric -Name "publicPackageDownloadProofSourceReadinessSignalCount" -DefaultValue 0)
$publicPackageDownloadProofCandidateReady = [bool](Get-OneScreenMetric -Name "publicPackageDownloadProofCandidateReady" -DefaultValue $false)
$postPublishCleanConsumerProofRequiredFieldCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofRequiredFieldCount" -DefaultValue 0)
$postPublishCleanConsumerProofRejectedSubstituteCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofRejectedSubstituteCount" -DefaultValue 0)
$postPublishCleanConsumerProofBlockedRealInputCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofBlockedRealInputCount" -DefaultValue 0)
$postPublishCleanConsumerProofSourceReadinessSignalCount = [int](Get-OneScreenMetric -Name "postPublishCleanConsumerProofSourceReadinessSignalCount" -DefaultValue 0)
$postPublishCleanConsumerProofCandidateReady = [bool](Get-OneScreenMetric -Name "postPublishCleanConsumerProofCandidateReady" -DefaultValue $false)
$postPublishCleanConsumerProofSourceProofLinkageReady = [bool](Get-OneScreenMetric -Name "postPublishCleanConsumerProofSourceProofLinkageReady" -DefaultValue $false)
$releaseEvidenceBundleSha256 = [string](Get-OneScreenMetric -Name "releaseEvidenceBundleSha256" -DefaultValue "")
$finalCloseStrictValidatorOutputState = [string](Get-OneScreenMetric -Name "finalCloseStrictValidatorOutputState" -DefaultValue "missing-final-close-strict-validator-output-state")

$sourceRecords = @(
  New-SourceRecordSummary -Id "final-owner-proof-action-worklist" -ArtifactPath "artifacts/final-release/final-owner-proof-action-worklist.json" -Record $finalOwnerProofActionWorklist -CountField "actionCount" -BlockedCountField "blockedActionCount"
  New-SourceRecordSummary -Id "final-owner-execution-package" -ArtifactPath "artifacts/final-release/final-owner-execution-package.json" -Record $finalOwnerExecutionPackage -CountField "executionStepCount" -BlockedCountField "blockedExecutionStepCount"
  New-SourceRecordSummary -Id "clean-external-package-consumer-owner-runbook" -ArtifactPath "artifacts/final-release/clean-external-package-consumer-owner-runbook.json" -Record $cleanExternalPackageConsumerOwnerRunbook -CountField "stepCount" -BlockedCountField "blockedStepCount"
  New-SourceRecordSummary -Id "post-publish-owner-verification-runbook" -ArtifactPath "artifacts/final-release/post-publish-owner-verification-runbook.json" -Record $postPublishOwnerVerificationRunbook -CountField "stepCount" -BlockedCountField "blockedStepCount"
  New-SourceRecordSummary -Id "public-publish-final-owner-execution-pack" -ArtifactPath "artifacts/final-release/public-publish-final-owner-execution-pack.json" -Record $publicPublishFinalOwnerExecutionPack -CountField "executionLaneCount" -BlockedCountField "blockedExecutionLaneCount"
  New-SourceRecordSummary -Id "public-publish-command-cross-check" -ArtifactPath "artifacts/final-release/public-publish-command-cross-check.json" -Record $publicPublishCommandCrossCheck -CountField "crossCheckCount" -BlockedCountField "blockedCrossCheckCount"
  New-SourceRecordSummary -Id "final-owner-close-readiness-checkpoint" -ArtifactPath "artifacts/final-release/final-owner-close-readiness-checkpoint.json" -Record $finalOwnerCloseReadinessCheckpoint -CountField "readinessCheckCount" -BlockedCountField "blockedReadinessCheckCount"
  New-SourceRecordSummary -Id "final-release-close-blocker-dashboard" -ArtifactPath "artifacts/final-release/final-release-close-blocker-dashboard.json" -Record $finalReleaseCloseBlockerDashboard -CountField "blockerCount" -BlockedCountField "blockedBlockerCount"
  New-SourceRecordSummary -Id "owner-input-contract-convergence" -ArtifactPath "artifacts/final-release/owner-input-contract-convergence.json" -Record $ownerInputContractConvergence -CountField "contractSurfaceCount" -BlockedCountField "blockedContractSurfaceCount"
  New-SourceRecordSummary -Id "final-owner-execution-one-screen-pack" -ArtifactPath "artifacts/final-release/final-owner-execution-one-screen-pack.json" -Record $finalOwnerExecutionOneScreenPack -CountField "releaseCloseRealInputChainCount" -BlockedCountField "blockedReleaseCloseRealInputChainCount"
  New-SourceRecordSummary -Id "final-owner-execution-one-screen-pack-validation" -ArtifactPath "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json" -Record $finalOwnerExecutionOneScreenPackValidation -CountField "releaseCloseRealInputChainCount" -BlockedCountField "releaseCloseRealInputChainBlockedRealInputCount"
)

$sourceArtifacts = @(
  "artifacts/final-release/final-owner-proof-action-worklist.json",
  "artifacts/final-release/final-owner-proof-action-worklist-validation.json",
  "artifacts/final-release/final-owner-execution-package.json",
  "artifacts/final-release/final-owner-execution-package-validation.json",
  "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
  "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
  "artifacts/final-release/post-publish-owner-verification-runbook.json",
  "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
  "artifacts/final-release/public-publish-final-owner-execution-pack.json",
  "artifacts/final-release/public-publish-final-owner-execution-pack-validation.json",
  "artifacts/final-release/public-publish-command-cross-check.json",
  "artifacts/final-release/public-publish-command-cross-check-validation.json",
  "artifacts/final-release/final-owner-close-readiness-checkpoint.json",
  "artifacts/final-release/final-owner-close-readiness-checkpoint-validation.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json",
  "artifacts/final-release/final-release-close-blocker-dashboard-validation.json",
  "artifacts/final-release/owner-input-contract-convergence.json",
  "artifacts/final-release/owner-input-contract-convergence-validation.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack.md",
  "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json",
  "artifacts/final-release/final-owner-execution-one-screen-pack-validation.md",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-evidence-classification-audit.json"
)

$steps = @(
  New-OrderStep -Sequence 1 -Id "01-contract-convergence-preflight" -Title "Owner input contract convergence preflight" -SourceArtifact "artifacts/final-release/owner-input-contract-convergence-validation.json" -SourceState ([string](Get-PropertyOrDefault -Object $ownerInputContractConvergenceValidation -Name "validationState" -DefaultValue "missing-owner-input-contract-convergence-validation")) -InputArtifacts @("owner-input-contract-convergence.json", "owner-input-contract-convergence-validation.json") -OutputArtifacts @("owner-external-proof-execution-result.input.json", "public-publish-result-owner-input.json", "release-issue-close-owner-decision-input.json") -ValidatorScripts @("eng/Test-OwnerInputContractConvergence.ps1 -Strict") -OwnerAction "Review the unified owner-fill contract fields before collecting real stdout/stderr/transcript/hash/public package evidence." -BlockedUntil "Real Owner input files are filled from external execution and checked against canonical fields." -FailureStopRule "Stop before collecting or importing proof if canonical public package, hash, transcript, host metadata, reviewer, or non-substitute fields are missing."
  New-OrderStep -Sequence 2 -Id "02-clean-external-consumer-prepublish-proof" -Title "Clean external package consumer pre-publish proof" -SourceArtifact "artifacts/final-release/clean-external-package-consumer-owner-runbook.json" -SourceState ([string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "runbookState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook")) -InputArtifacts @("clean-external-package-consumer-owner-runbook.json", "package-consumer-runtime-proof-owner-input.template.json") -OutputArtifacts @("package-consumer-runtime-proof-record.json", "package-consumer-runtime-proof-record-validation.json", "owner-external-proof-execution-result.input.json") -ValidatorScripts @("eng/Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict", "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof") -OwnerAction "Run a repository-external clean package consumer restore/build/smoke on a compatible host and capture real logs and SHA256 values." -BlockedUntil "Runtime proof record contains real external stdout/stderr/merged transcript, hashes, host metadata, owner review, and non-substitute confirmations." -FailureStopRule "Stop if the run uses local feed, ProjectReference, direct nupkg, repository-internal build output, template data, or missing hashes."
  New-OrderStep -Sequence 3 -Id "03-public-publish-manual-owner-command" -Title "Manual public publish command and cross-check" -SourceArtifact "artifacts/final-release/public-publish-final-owner-execution-pack.json" -SourceState ([string](Get-PropertyOrDefault -Object $publicPublishFinalOwnerExecutionPack -Name "executionPackState" -DefaultValue "missing-public-publish-final-owner-execution-pack")) -InputArtifacts @("public-publish-final-owner-execution-pack.json", "public-publish-command-cross-check.json", "public-publish-result-owner-input.template.json") -OutputArtifacts @("public-publish-result-import.json", "public-package-proof-owner-input.json", "public-package-proof-owner-input-validation.json") -ValidatorScripts @("eng/Test-PublicPublishFinalOwnerExecutionPack.ps1 -Strict", "eng/Test-PublicPublishCommandCrossCheck.ps1 -Strict") -OwnerAction "Owner manually executes the selected public publish command outside automation and records public package URL, source URL, downloaded nupkg SHA256, timestamp, stdout/stderr, and transcript hashes." -BlockedUntil "Owner supplies real public publish result and public package proof from the selected public source." -FailureStopRule "Stop if automation attempts package push or if public package URL/source URL/downloaded nupkg SHA256/timestamp is missing."
  New-OrderStep -Sequence 4 -Id "04-post-publish-clean-consumer-proof" -Title "Post-publish clean consumer verification" -SourceArtifact "artifacts/final-release/post-publish-owner-verification-runbook.json" -SourceState ([string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "runbookState" -DefaultValue "missing-post-publish-owner-verification-runbook")) -InputArtifacts @("post-publish-owner-verification-runbook.json", "post-publish-verification-owner-input.template.json", "public-package-proof-owner-input-validation.json") -OutputArtifacts @("post-publish-verification-record.json", "post-publish-verification-validation.json", "post-publish-proof-owner-confirmation-validation.json") -ValidatorScripts @("eng/Test-PostPublishOwnerVerificationRunbook.ps1 -Strict", "eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof") -OwnerAction "Restore, build, and run a clean consumer from the real public package source after publish, then capture real logs, hashes, host metadata, and owner review." -BlockedUntil "Post-publish clean consumer proof validates against the real public package source without local substitutes." -FailureStopRule "Stop if restore uses local feed, ProjectReference, direct nupkg, unpublished package artifacts, dashboard output, or any missing log/hash field."
  New-OrderStep -Sequence 5 -Id "05-owner-result-import-and-strict-validator" -Title "Owner result import and strict validator bridge" -SourceArtifact "artifacts/final-release/final-owner-execution-package.json" -SourceState ([string](Get-PropertyOrDefault -Object $finalOwnerExecutionPackage -Name "packageState" -DefaultValue "missing-final-owner-execution-package")) -InputArtifacts @("owner-external-proof-execution-result.input.json", "package-consumer-runtime-proof-record.json", "post-publish-verification-record.json", "public-package-proof-owner-input.json") -OutputArtifacts @("owner-external-proof-execution-result-import-validation.json", "real-external-proof-record-import-validator-validation.json", "real-proof-record-candidate-from-owner-result-import-validation.json") -ValidatorScripts @("eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict", "eng/Test-RealExternalProofRecordImportValidator.ps1 -Strict", "eng/Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict") -OwnerAction "Import owner-filled real proof input and run strict validators that reject templates, drafts, candidates, dashboards, dry-runs, local feed, ProjectReference, direct nupkg, and build-only substitutes." -BlockedUntil "Strict validators accept real runtime, public package, and post-publish proof records with matching paths and SHA256 values." -FailureStopRule "Stop if any path is outside the allowed evidence root, any SHA256 mismatches, any required file is missing, or any forbidden substitute is detected."
  New-OrderStep -Sequence 6 -Id "06-final-readiness-and-blocker-dashboard" -Title "Final readiness checkpoint and blocker dashboard review" -SourceArtifact "artifacts/final-release/final-owner-close-readiness-checkpoint.json" -SourceState ([string](Get-PropertyOrDefault -Object $finalOwnerCloseReadinessCheckpoint -Name "checkpointState" -DefaultValue "missing-final-owner-close-readiness-checkpoint")) -InputArtifacts @("final-owner-close-readiness-checkpoint.json", "final-release-close-blocker-dashboard.json", "release-evidence-bundle.json", "release-evidence-classification-audit.json") -OutputArtifacts @("final-owner-close-readiness-checkpoint-validation.json", "final-release-close-blocker-dashboard-validation.json", "release-evidence-classification-audit.json") -ValidatorScripts @("eng/Test-FinalOwnerCloseReadinessCheckpoint.ps1 -Strict", "eng/Test-FinalReleaseCloseBlockerDashboard.ps1 -Strict", "eng/Test-ReleaseEvidenceClassificationAudit.ps1 -Strict") -OwnerAction "Review final readiness and blocker dashboards only after real proof validators pass; treat them as audit views, not proof." -BlockedUntil "All blocker/check lanes point to accepted strict proof instead of handoff, dashboard, runbook, candidate, or dry-run artifacts." -FailureStopRule "Stop if any readiness check or blocker still references non-proof materials as proof or if canCloseReleaseIssue remains false."
  New-OrderStep -Sequence 7 -Id "07-release-issue-close-owner-decision" -Title "Release issue close owner decision" -SourceArtifact "artifacts/final-release/release-issue-close-owner-decision-input.template.json" -SourceState "blocked-release-close-owner-decision-real-proof-required" -InputArtifacts @("release-issue-close-owner-decision-input.json", "release-close-strict-record-candidate-validation.json", "release-issue-close-record-validation.json") -OutputArtifacts @("release-issue-close-record.json", "release-issue-close-record-validation.json", "release-close-final-record.json") -ValidatorScripts @("eng/Test-ReleaseIssueCloseRecord.ps1 -Strict", "eng/Test-ReleaseEvidenceClassificationAudit.ps1 -Strict") -OwnerAction "Owner signs the release close decision only after runtime proof, public package proof, post-publish proof, and strict close validation all pass." -BlockedUntil "Strict close record can be proven by accepted real proof records and final owner decision input." -FailureStopRule "Stop if any required proof lane is missing, blocked, template-only, candidate-only, or dashboard-only."
)

$blockedStepCount = @($steps | Where-Object { -not $_.canCloseReleaseIssue }).Count

$record = [pscustomobject]@{
  recordKind = "final-owner-strict-close-execution-order"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  orderState = "blocked-final-owner-strict-close-owner-execution-required"
  stepCount = $steps.Count
  blockedStepCount = $blockedStepCount
  sourceRecordCount = $sourceRecords.Count
  sourceRecords = @($sourceRecords)
  sourceActionCount = [int](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklist -Name "actionCount" -DefaultValue 0)
  sourceExecutionStepCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionPackage -Name "executionStepCount" -DefaultValue 0)
  cleanExternalRunbookStepCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "stepCount" -DefaultValue 0)
  postPublishRunbookStepCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "stepCount" -DefaultValue 0)
  publicPublishExecutionLaneCount = [int](Get-PropertyOrDefault -Object $publicPublishFinalOwnerExecutionPack -Name "executionLaneCount" -DefaultValue 0)
  publicPublishCrossCheckCount = [int](Get-PropertyOrDefault -Object $publicPublishCommandCrossCheck -Name "crossCheckCount" -DefaultValue 0)
  finalOwnerCloseReadinessCheckCount = [int](Get-PropertyOrDefault -Object $finalOwnerCloseReadinessCheckpoint -Name "readinessCheckCount" -DefaultValue 0)
  finalReleaseCloseBlockerCount = [int](Get-PropertyOrDefault -Object $finalReleaseCloseBlockerDashboard -Name "blockerCount" -DefaultValue 0)
  ownerInputContractSurfaceCount = [int](Get-PropertyOrDefault -Object $ownerInputContractConvergence -Name "contractSurfaceCount" -DefaultValue 0)
  ownerInputContractCanonicalFieldCount = [int](Get-PropertyOrDefault -Object $ownerInputContractConvergence -Name "canonicalFieldCount" -DefaultValue 0)
  ownerInputContractRunbookInputCount = [int](Get-PropertyOrDefault -Object $ownerInputContractConvergence -Name "runbookInputCount" -DefaultValue 0)
  releaseCloseRealInputChainCount = $releaseCloseRealInputChainCount
  releaseCloseRealInputChainRequiredFieldCount = $releaseCloseRealInputChainRequiredFieldCount
  releaseCloseRealInputChainRejectedSubstituteCount = $releaseCloseRealInputChainRejectedSubstituteCount
  releaseCloseRealInputChainSourceReadinessSignalCount = $releaseCloseRealInputChainSourceReadinessSignalCount
  releaseCloseRealInputChainBlockedRealInputCount = $releaseCloseRealInputChainBlockedRealInputCount
  releaseCloseRealInputChain = @($releaseCloseRealInputChainProjection)
  publicPackageDownloadProofRequiredFieldCount = $publicPackageDownloadProofRequiredFieldCount
  publicPackageDownloadProofRejectedSubstituteCount = $publicPackageDownloadProofRejectedSubstituteCount
  publicPackageDownloadProofSourceReadinessSignalCount = $publicPackageDownloadProofSourceReadinessSignalCount
  publicPackageDownloadProofCandidateReady = $publicPackageDownloadProofCandidateReady
  postPublishCleanConsumerProofRequiredFieldCount = $postPublishCleanConsumerProofRequiredFieldCount
  postPublishCleanConsumerProofRejectedSubstituteCount = $postPublishCleanConsumerProofRejectedSubstituteCount
  postPublishCleanConsumerProofBlockedRealInputCount = $postPublishCleanConsumerProofBlockedRealInputCount
  postPublishCleanConsumerProofSourceReadinessSignalCount = $postPublishCleanConsumerProofSourceReadinessSignalCount
  postPublishCleanConsumerProofCandidateReady = $postPublishCleanConsumerProofCandidateReady
  postPublishCleanConsumerProofSourceProofLinkageReady = $postPublishCleanConsumerProofSourceProofLinkageReady
  releaseEvidenceBundleSha256 = $releaseEvidenceBundleSha256
  finalCloseStrictValidatorOutputState = $finalCloseStrictValidatorOutputState
  sourceStates = [pscustomobject]@{
    finalOwnerProofActionWorklist = Get-RecordState -Record $finalOwnerProofActionWorklist
    finalOwnerExecutionPackage = Get-RecordState -Record $finalOwnerExecutionPackage
    cleanExternalPackageConsumerOwnerRunbook = Get-RecordState -Record $cleanExternalPackageConsumerOwnerRunbook
    postPublishOwnerVerificationRunbook = Get-RecordState -Record $postPublishOwnerVerificationRunbook
    publicPublishFinalOwnerExecutionPack = Get-RecordState -Record $publicPublishFinalOwnerExecutionPack
    publicPublishCommandCrossCheck = Get-RecordState -Record $publicPublishCommandCrossCheck
    finalOwnerCloseReadinessCheckpoint = Get-RecordState -Record $finalOwnerCloseReadinessCheckpoint
    finalReleaseCloseBlockerDashboard = Get-RecordState -Record $finalReleaseCloseBlockerDashboard
    ownerInputContractConvergence = Get-RecordState -Record $ownerInputContractConvergence
    ownerInputContractConvergenceValidation = Get-RecordState -Record $ownerInputContractConvergenceValidation
    finalOwnerExecutionOneScreenPack = Get-RecordState -Record $finalOwnerExecutionOneScreenPack
    finalOwnerExecutionOneScreenPackValidation = Get-RecordState -Record $finalOwnerExecutionOneScreenPackValidation
  }
  executionSteps = @($steps)
  forbiddenSubstitutes = @(
    "template",
    "draft",
    "candidate",
    "dashboard",
    "dry-run",
    "build-only",
    "local feed",
    "ProjectReference",
    "direct nupkg",
    "direct .nupkg",
    "runbook as proof",
    "manual handoff as proof",
    "bundle-ready as proof",
    "blocked-by-cuda-driver",
    "public package download proof alone",
    "post-publish validation-ready without proofCandidateReady",
    "release evidence bundle hash only",
    "strict close validator output without real proof"
  )
  ownerManualOnlyCommands = @(
    "dotnet nuget push must be executed by Owner outside automation only after strict proof gates pass",
    "public package source verification must use the selected public package channel",
    "release issue close decision must be filled by Owner only after strict validators pass"
  )
  sourceArtifacts = @($sourceArtifacts)
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner StrictClose execution order is owner guidance and blocked handoff only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-strict-close-execution-order.json"
$markdownPath = Join-Path $OutputRoot "final-owner-strict-close-execution-order.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Final Owner StrictClose Execution Order")
$lines.Add("")
$lines.Add("`final-owner-strict-close-execution-order` 将 Owner 最终执行顺序、输入文件、输出 proof、validator 和失败停点聚合到一个 blocked/non-proof handoff。它不执行发布、不推送包、不证明 runtime/post-publish、不批准公开发布，也不关闭 release issue。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| orderState | ``$($record.orderState)`` |")
$lines.Add("| stepCount | ``$($record.stepCount)`` |")
$lines.Add("| blockedStepCount | ``$($record.blockedStepCount)`` |")
$lines.Add("| sourceActionCount | ``$($record.sourceActionCount)`` |")
$lines.Add("| sourceExecutionStepCount | ``$($record.sourceExecutionStepCount)`` |")
$lines.Add("| cleanExternalRunbookStepCount | ``$($record.cleanExternalRunbookStepCount)`` |")
$lines.Add("| postPublishRunbookStepCount | ``$($record.postPublishRunbookStepCount)`` |")
$lines.Add("| publicPublishExecutionLaneCount | ``$($record.publicPublishExecutionLaneCount)`` |")
$lines.Add("| publicPublishCrossCheckCount | ``$($record.publicPublishCrossCheckCount)`` |")
$lines.Add("| finalOwnerCloseReadinessCheckCount | ``$($record.finalOwnerCloseReadinessCheckCount)`` |")
$lines.Add("| finalReleaseCloseBlockerCount | ``$($record.finalReleaseCloseBlockerCount)`` |")
$lines.Add("| ownerInputContractSurfaceCount | ``$($record.ownerInputContractSurfaceCount)`` |")
$lines.Add("| ownerInputContractCanonicalFieldCount | ``$($record.ownerInputContractCanonicalFieldCount)`` |")
$lines.Add("| ownerInputContractRunbookInputCount | ``$($record.ownerInputContractRunbookInputCount)`` |")
$lines.Add("| releaseCloseRealInputChainCount | ``$($record.releaseCloseRealInputChainCount)`` |")
$lines.Add("| releaseCloseRealInputChainRequiredFieldCount | ``$($record.releaseCloseRealInputChainRequiredFieldCount)`` |")
$lines.Add("| releaseCloseRealInputChainRejectedSubstituteCount | ``$($record.releaseCloseRealInputChainRejectedSubstituteCount)`` |")
$lines.Add("| releaseCloseRealInputChainSourceReadinessSignalCount | ``$($record.releaseCloseRealInputChainSourceReadinessSignalCount)`` |")
$lines.Add("| releaseCloseRealInputChainBlockedRealInputCount | ``$($record.releaseCloseRealInputChainBlockedRealInputCount)`` |")
$lines.Add("| publicPackageDownloadProofRequiredFieldCount | ``$($record.publicPackageDownloadProofRequiredFieldCount)`` |")
$lines.Add("| publicPackageDownloadProofCandidateReady | ``$($record.publicPackageDownloadProofCandidateReady)`` |")
$lines.Add("| postPublishCleanConsumerProofRequiredFieldCount | ``$($record.postPublishCleanConsumerProofRequiredFieldCount)`` |")
$lines.Add("| postPublishCleanConsumerProofCandidateReady | ``$($record.postPublishCleanConsumerProofCandidateReady)`` |")
$lines.Add("| postPublishCleanConsumerProofSourceProofLinkageReady | ``$($record.postPublishCleanConsumerProofSourceProofLinkageReady)`` |")
$lines.Add("| releaseEvidenceBundleSha256 | ``$($record.releaseEvidenceBundleSha256)`` |")
$lines.Add("| finalCloseStrictValidatorOutputState | ``$($record.finalCloseStrictValidatorOutputState)`` |")
$lines.Add("| performsPublish | ``$($record.performsPublish)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Execution Steps")
$lines.Add("")
$lines.Add("| # | ID | Source State | Validators | Failure Stop Rule |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($step in $steps) {
  $lines.Add("| $($step.sequence) | ``$(ConvertTo-MarkdownCell $step.id)`` | ``$(ConvertTo-MarkdownCell $step.sourceState)`` | ``$(ConvertTo-MarkdownCell (($step.validatorScripts -join '; ')))`` | $(ConvertTo-MarkdownCell $step.failureStopRule) |")
}
$lines.Add("")
$lines.Add("## Release Close Real Input Chain")
$lines.Add("")
$lines.Add("| # | ID | State | Required Fields | Rejected Substitutes | Source Signals | Blocked Inputs | Proof Ready | Linkage Ready |")
$lines.Add("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
foreach ($step in $releaseCloseRealInputChainProjection) {
  $lines.Add("| $($step.order) | ``$(ConvertTo-MarkdownCell $step.id)`` | ``$(ConvertTo-MarkdownCell $step.currentState)`` | $($step.requiredFieldCount) | $($step.rejectedSubstituteCount) | $($step.sourceReadinessSignalCount) | $($step.blockedRealInputCount) | ``$($step.proofCandidateReady)`` | ``$($step.sourceLinkageReady)`` |")
}
$lines.Add("")
$lines.Add("## Source Records")
$lines.Add("")
$lines.Add("| ID | State | Count | Blocked |")
$lines.Add("| --- | --- | ---: | ---: |")
foreach ($source in $sourceRecords) {
  $lines.Add("| ``$(ConvertTo-MarkdownCell $source.id)`` | ``$(ConvertTo-MarkdownCell $source.state)`` | $($source.count) | $($source.blockedCount) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)
$lines.Add("")
$lines.Add("Owner 手动执行 public publish 后，仍必须导入真实 public package proof 和 post-publish clean consumer proof，并由 strict validators 接受后，才能进入 release issue close decision。")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final Owner StrictClose execution order written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "OrderState=$($record.orderState) Steps=$($record.stepCount) Sources=$($record.sourceRecordCount)"
