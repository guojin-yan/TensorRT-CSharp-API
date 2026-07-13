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

function Get-RecordState {
  param([AllowNull()][object]$Record)
  foreach ($name in @("readinessState", "validationState", "orchestrationState", "contractState", "importState", "validatorState", "dryRunState", "reportState", "actionPackState", "candidateState", "gateState")) {
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
    [string]$StateField,
    [string]$CountField,
    [string]$BlockedCountField,
    [string]$ActionRequiredField = "failedActionRequiredCount",
    [string]$BlockerField = "failedBlockerCount"
  )

  [pscustomobject]@{
    id = $Id
    artifactPath = $ArtifactPath
    exists = $null -ne $Record
    recordKind = [string](Get-PropertyOrDefault -Object $Record -Name "recordKind" -DefaultValue "missing")
    state = [string](Get-PropertyOrDefault -Object $Record -Name $StateField -DefaultValue (Get-RecordState -Record $Record))
    count = [int](Get-PropertyOrDefault -Object $Record -Name $CountField -DefaultValue 0)
    blockedCount = [int](Get-PropertyOrDefault -Object $Record -Name $BlockedCountField -DefaultValue 0)
    failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Record -Name $ActionRequiredField -DefaultValue 0)
    failedBlockerCount = [int](Get-PropertyOrDefault -Object $Record -Name $BlockerField -DefaultValue 0)
    performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
    canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "canPromoteRuntimeProof" -DefaultValue $false)
    canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
    canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
    isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
    isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
    isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
  }
}

function New-PromotionLane {
  param(
    [int]$Sequence,
    [string]$LaneId,
    [string]$Title,
    [string]$SourceArtifact,
    [string]$CurrentState,
    [string[]]$RequiredRealInputs,
    [string]$StrictValidator,
    [string]$PromotionBlockedUntil
  )

  [pscustomobject]@{
    sequence = $Sequence
    laneId = $LaneId
    title = $Title
    sourceArtifact = $SourceArtifact
    currentState = $CurrentState
    requiredRealInputs = @($RequiredRealInputs)
    requiredRealInputCount = @($RequiredRealInputs).Count
    strictValidator = $StrictValidator
    promotionBlockedUntil = $PromotionBlockedUntil
    canPromote = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    performsPublish = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    nonSubstituteBoundary = "ReleaseClose candidate promotion requires real Owner evidence accepted by strict validators. Dashboard, runbook, candidate, draft, local feed, ProjectReference, direct nupkg, dry-run, build-only, and bundle-ready substitutes cannot promote this lane."
    boundary = "Candidate promotion readiness lane only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$realOwnerEvidenceStrictValidatorOrchestration = Read-JsonOrNull "artifacts\final-release\real-owner-evidence-strict-validator-orchestration.json"
$realOwnerEvidenceStrictValidatorOrchestrationValidation = Read-JsonOrNull "artifacts\final-release\real-owner-evidence-strict-validator-orchestration-validation.json"
$ownerRealInputJsonContract = Read-JsonOrNull "artifacts\final-release\owner-real-input-json-contract.json"
$ownerRealInputJsonContractValidation = Read-JsonOrNull "artifacts\final-release\owner-real-input-json-contract-validation.json"
$ownerRealInputJsonImport = Read-JsonOrNull "artifacts\final-release\owner-real-input-json-import.json"
$ownerRealInputJsonImportValidation = Read-JsonOrNull "artifacts\final-release\owner-real-input-json-import-validation.json"
$ownerRealInputHashAndPathValidator = Read-JsonOrNull "artifacts\final-release\owner-real-input-hash-and-path-validator.json"
$ownerRealInputHashAndPathValidatorValidation = Read-JsonOrNull "artifacts\final-release\owner-real-input-hash-and-path-validator-validation.json"
$ownerRealInputForbiddenSubstituteValidator = Read-JsonOrNull "artifacts\final-release\owner-real-input-forbidden-substitute-validator.json"
$ownerRealInputForbiddenSubstituteValidatorValidation = Read-JsonOrNull "artifacts\final-release\owner-real-input-forbidden-substitute-validator-validation.json"
$strictCloseRealInputDryRun = Read-JsonOrNull "artifacts\final-release\strict-close-real-input-dry-run.json"
$strictCloseRealInputDryRunValidation = Read-JsonOrNull "artifacts\final-release\strict-close-real-input-dry-run-validation.json"
$strictCloseRealInputFindingReport = Read-JsonOrNull "artifacts\final-release\strict-close-real-input-finding-report.json"
$strictCloseRealInputFindingReportValidation = Read-JsonOrNull "artifacts\final-release\strict-close-real-input-finding-report-validation.json"
$strictCloseOwnerActionPack = Read-JsonOrNull "artifacts\final-release\strict-close-owner-action-pack.json"
$strictCloseOwnerActionPackValidation = Read-JsonOrNull "artifacts\final-release\strict-close-owner-action-pack-validation.json"
$releaseCloseStrictRecordCandidate = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate.json"
$releaseCloseStrictRecordCandidateValidation = Read-JsonOrNull "artifacts\final-release\release-close-strict-record-candidate-validation.json"
$finalReleaseCloseRecordRealValidatorValidation = Read-JsonOrNull "artifacts\final-release\final-release-close-record-real-validator-validation.json"
$finalPublishProofGateReport = Read-JsonOrNull "artifacts\final-release\final-publish-proof-gate-report.json"

$sourceRecords = @(
  New-SourceRecordSummary -Id "real-owner-evidence-strict-validator-orchestration" -ArtifactPath "artifacts/final-release/real-owner-evidence-strict-validator-orchestration-validation.json" -Record $realOwnerEvidenceStrictValidatorOrchestrationValidation -StateField "validationState" -CountField "fieldReadinessCount" -BlockedCountField "blockedFieldReadinessCount"
  New-SourceRecordSummary -Id "owner-real-input-json-contract" -ArtifactPath "artifacts/final-release/owner-real-input-json-contract-validation.json" -Record $ownerRealInputJsonContractValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "owner-real-input-json-import" -ArtifactPath "artifacts/final-release/owner-real-input-json-import-validation.json" -Record $ownerRealInputJsonImportValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "owner-real-input-hash-and-path-validator" -ArtifactPath "artifacts/final-release/owner-real-input-hash-and-path-validator-validation.json" -Record $ownerRealInputHashAndPathValidatorValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "owner-real-input-forbidden-substitute-validator" -ArtifactPath "artifacts/final-release/owner-real-input-forbidden-substitute-validator-validation.json" -Record $ownerRealInputForbiddenSubstituteValidatorValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "strict-close-real-input-dry-run" -ArtifactPath "artifacts/final-release/strict-close-real-input-dry-run-validation.json" -Record $strictCloseRealInputDryRunValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "strict-close-real-input-finding-report" -ArtifactPath "artifacts/final-release/strict-close-real-input-finding-report-validation.json" -Record $strictCloseRealInputFindingReportValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "strict-close-owner-action-pack" -ArtifactPath "artifacts/final-release/strict-close-owner-action-pack-validation.json" -Record $strictCloseOwnerActionPackValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "release-close-strict-record-candidate" -ArtifactPath "artifacts/final-release/release-close-strict-record-candidate-validation.json" -Record $releaseCloseStrictRecordCandidateValidation -StateField "validationState" -CountField "missingOwnerInputCount" -BlockedCountField "missingOwnerInputCount"
  New-SourceRecordSummary -Id "final-release-close-record-real-validator" -ArtifactPath "artifacts/final-release/final-release-close-record-real-validator-validation.json" -Record $finalReleaseCloseRecordRealValidatorValidation -StateField "validationState" -CountField "requiredFieldCount" -BlockedCountField "blockedRequiredFieldCount"
  New-SourceRecordSummary -Id "final-publish-proof-gate" -ArtifactPath "artifacts/final-release/final-publish-proof-gate-report.json" -Record $finalPublishProofGateReport -StateField "validationState" -CountField "failedActionRequiredCount" -BlockedCountField "failedActionRequiredCount"
)

$lanes = @(
  New-PromotionLane -Sequence 1 -LaneId "public-package-proof" -Title "Public package proof lane" -SourceArtifact "artifacts/final-release/owner-real-input-json-contract.json" -CurrentState (Get-RecordState -Record $ownerRealInputJsonContract) -RequiredRealInputs @("publicPackageSourceUrl", "downloadedNupkgSha256", "packageId", "packageVersion") -StrictValidator "owner-real-input-json-contract" -PromotionBlockedUntil "Owner supplies public package URL, source kind, id, version, downloaded package SHA256, and non-substitute confirmation."
  New-PromotionLane -Sequence 2 -LaneId "clean-external-consumer-runtime-proof" -Title "Clean external consumer runtime proof lane" -SourceArtifact "artifacts/final-release/owner-real-input-json-import.json" -CurrentState (Get-RecordState -Record $ownerRealInputJsonImport) -RequiredRealInputs @("cleanConsumerProjectPath", "cleanConsumerLogPath", "cleanConsumerLogSha256", "hostMetadata") -StrictValidator "owner-real-input-json-import" -PromotionBlockedUntil "Owner imports real clean external consumer inputs with existing logs and hashes."
  New-PromotionLane -Sequence 3 -LaneId "post-publish-clean-consumer-proof" -Title "Post-publish clean consumer proof lane" -SourceArtifact "artifacts/final-release/owner-real-input-hash-and-path-validator.json" -CurrentState (Get-RecordState -Record $ownerRealInputHashAndPathValidator) -RequiredRealInputs @("postPublishInstallLogPath", "postPublishRunLogPath", "downloadedNupkgSha256", "hostMetadata") -StrictValidator "owner-real-input-hash-and-path-validator" -PromotionBlockedUntil "Owner supplies real post-publish install/run logs and matching hashes from the public package source."
  New-PromotionLane -Sequence 4 -LaneId "hash-path-validation" -Title "Hash and path validation lane" -SourceArtifact "artifacts/final-release/owner-real-input-hash-and-path-validator.json" -CurrentState (Get-RecordState -Record $ownerRealInputHashAndPathValidator) -RequiredRealInputs @("stdoutPath", "stderrPath", "cleanConsumerLogSha256", "downloadedNupkgSha256") -StrictValidator "owner-real-input-hash-and-path-validator" -PromotionBlockedUntil "All owner-supplied file paths exist, remain inside allowed evidence roots, and match SHA256 values."
  New-PromotionLane -Sequence 5 -LaneId "forbidden-substitute-validation" -Title "Forbidden substitute validation lane" -SourceArtifact "artifacts/final-release/owner-real-input-forbidden-substitute-validator.json" -CurrentState (Get-RecordState -Record $ownerRealInputForbiddenSubstituteValidator) -RequiredRealInputs @("nonSubstituteConfirmations", "packageSourceKind", "cleanConsumerProjectPath") -StrictValidator "owner-real-input-forbidden-substitute-validator" -PromotionBlockedUntil "Owner confirms no local feed, ProjectReference, direct nupkg, dry-run, dashboard, runbook, candidate, draft, or build-only substitute was used."
  New-PromotionLane -Sequence 6 -LaneId "strict-close-dry-run" -Title "Strict close real input dry-run lane" -SourceArtifact "artifacts/final-release/strict-close-real-input-dry-run.json" -CurrentState (Get-RecordState -Record $strictCloseRealInputDryRun) -RequiredRealInputs @("stdoutPath", "stderrPath", "hostMetadata", "nonSubstituteConfirmations") -StrictValidator "strict-close-real-input-dry-run" -PromotionBlockedUntil "Strict close dry-run sees real Owner evidence for every required lane without unresolved findings."
  New-PromotionLane -Sequence 7 -LaneId "rollback-review" -Title "Rollback review lane" -SourceArtifact "artifacts/final-release/strict-close-real-input-finding-report.json" -CurrentState (Get-RecordState -Record $strictCloseRealInputFindingReport) -RequiredRealInputs @("rollbackReview", "postPublishRunLogPath", "publicPackageSourceUrl") -StrictValidator "strict-close-real-input-finding-report" -PromotionBlockedUntil "Owner records rollback review after public package and post-publish clean consumer proof are available."
  New-PromotionLane -Sequence 8 -LaneId "final-close-decision" -Title "Final close decision lane" -SourceArtifact "artifacts/final-release/strict-close-owner-action-pack.json" -CurrentState (Get-RecordState -Record $strictCloseOwnerActionPack) -RequiredRealInputs @("finalCloseDecision", "rollbackReview", "nonSubstituteConfirmations") -StrictValidator "strict-close-owner-action-pack" -PromotionBlockedUntil "Owner signs final close decision after every real proof and strict validator lane is accepted."
  New-PromotionLane -Sequence 9 -LaneId "release-close-strict-record-candidate" -Title "ReleaseClose strict record candidate lane" -SourceArtifact "artifacts/final-release/release-close-strict-record-candidate-validation.json" -CurrentState (Get-RecordState -Record $releaseCloseStrictRecordCandidateValidation) -RequiredRealInputs @("finalCloseDecision", "downloadedNupkgSha256", "cleanConsumerLogSha256", "hostMetadata") -StrictValidator "release-close-strict-record-candidate" -PromotionBlockedUntil "ReleaseClose strict record candidate contains no placeholder owner inputs, missing real proof, or mismatched hashes."
  New-PromotionLane -Sequence 10 -LaneId "final-release-close-record-real-validator" -Title "Final release close record real validator lane" -SourceArtifact "artifacts/final-release/final-release-close-record-real-validator-validation.json" -CurrentState (Get-RecordState -Record $finalReleaseCloseRecordRealValidatorValidation) -RequiredRealInputs @("finalCloseDecision", "rollbackReview", "publicPackageSourceUrl", "postPublishRunLogPath") -StrictValidator "final-release-close-record-real-validator" -PromotionBlockedUntil "Final close record real validator accepts all required fields from real Owner proof records."
  New-PromotionLane -Sequence 11 -LaneId "final-publish-proof-gate" -Title "Final publish proof gate lane" -SourceArtifact "artifacts/final-release/final-publish-proof-gate-report.json" -CurrentState (Get-RecordState -Record $finalPublishProofGateReport) -RequiredRealInputs @("publicPackageSourceUrl", "downloadedNupkgSha256", "cleanConsumerLogPath", "postPublishRunLogPath", "finalCloseDecision") -StrictValidator "final-publish-proof-gate" -PromotionBlockedUntil "Final publish proof gate has zero action-required owner proof gaps and all real proof lanes are accepted."
)

$sourceArtifacts = @(
  "artifacts/final-release/real-owner-evidence-strict-validator-orchestration.json",
  "artifacts/final-release/real-owner-evidence-strict-validator-orchestration-validation.json",
  "artifacts/final-release/owner-real-input-json-contract.json",
  "artifacts/final-release/owner-real-input-json-contract-validation.json",
  "artifacts/final-release/owner-real-input-json-import.json",
  "artifacts/final-release/owner-real-input-json-import-validation.json",
  "artifacts/final-release/owner-real-input-hash-and-path-validator.json",
  "artifacts/final-release/owner-real-input-hash-and-path-validator-validation.json",
  "artifacts/final-release/owner-real-input-forbidden-substitute-validator.json",
  "artifacts/final-release/owner-real-input-forbidden-substitute-validator-validation.json",
  "artifacts/final-release/strict-close-real-input-dry-run.json",
  "artifacts/final-release/strict-close-real-input-dry-run-validation.json",
  "artifacts/final-release/strict-close-real-input-finding-report.json",
  "artifacts/final-release/strict-close-real-input-finding-report-validation.json",
  "artifacts/final-release/strict-close-owner-action-pack.json",
  "artifacts/final-release/strict-close-owner-action-pack-validation.json",
  "artifacts/final-release/release-close-strict-record-candidate.json",
  "artifacts/final-release/release-close-strict-record-candidate-validation.json",
  "artifacts/final-release/final-release-close-record-real-validator-validation.json",
  "artifacts/final-release/final-publish-proof-gate-report.json"
)

$failedBlockerCount = @($sourceRecords | Where-Object { $_.failedBlockerCount -gt 0 }).Count
$failedActionRequiredCount = [int](@($sourceRecords | Measure-Object -Property failedActionRequiredCount -Sum).Sum)
$blockedLaneCount = @($lanes | Where-Object { -not $_.canPromote }).Count

$record = [pscustomobject]@{
  recordKind = "release-close-real-input-candidate-promotion-readiness"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  readinessState = "blocked-release-close-real-input-candidate-promotion-real-owner-input-required"
  sourceRecordCount = $sourceRecords.Count
  candidatePromotionLaneCount = $lanes.Count
  blockedCandidatePromotionLaneCount = $blockedLaneCount
  readyCandidatePromotionLaneCount = 0
  failedBlockerCount = $failedBlockerCount
  failedActionRequiredCount = $failedActionRequiredCount
  sourceRecords = @($sourceRecords)
  candidatePromotionMatrix = @($lanes)
  sourceStates = [pscustomobject]@{
    realOwnerEvidenceStrictValidatorOrchestration = Get-RecordState -Record $realOwnerEvidenceStrictValidatorOrchestration
    realOwnerEvidenceStrictValidatorOrchestrationValidation = Get-RecordState -Record $realOwnerEvidenceStrictValidatorOrchestrationValidation
    ownerRealInputJsonContract = Get-RecordState -Record $ownerRealInputJsonContract
    ownerRealInputJsonImport = Get-RecordState -Record $ownerRealInputJsonImport
    ownerRealInputHashAndPathValidator = Get-RecordState -Record $ownerRealInputHashAndPathValidator
    ownerRealInputForbiddenSubstituteValidator = Get-RecordState -Record $ownerRealInputForbiddenSubstituteValidator
    strictCloseRealInputDryRun = Get-RecordState -Record $strictCloseRealInputDryRun
    strictCloseRealInputFindingReport = Get-RecordState -Record $strictCloseRealInputFindingReport
    strictCloseOwnerActionPack = Get-RecordState -Record $strictCloseOwnerActionPack
    releaseCloseStrictRecordCandidate = Get-RecordState -Record $releaseCloseStrictRecordCandidate
    releaseCloseStrictRecordCandidateValidation = Get-RecordState -Record $releaseCloseStrictRecordCandidateValidation
    finalReleaseCloseRecordRealValidatorValidation = Get-RecordState -Record $finalReleaseCloseRecordRealValidatorValidation
    finalPublishProofGateReport = Get-RecordState -Record $finalPublishProofGateReport
  }
  forbiddenSubstitutes = @(
    "template",
    "draft",
    "candidate",
    "dashboard",
    "runbook",
    "dry-run",
    "build-only",
    "local feed",
    "ProjectReference",
    "direct nupkg",
    "direct .nupkg",
    "bundle-ready"
  )
  sourceArtifacts = @($sourceArtifacts)
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "ReleaseClose real input candidate promotion readiness is a blocked promotion map only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 is not ready; real Owner inputs must pass strict validators before promotion."
}

$jsonPath = Join-Path $OutputRoot "release-close-real-input-candidate-promotion-readiness.json"
$markdownPath = Join-Path $OutputRoot "release-close-real-input-candidate-promotion-readiness.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# ReleaseClose Real Input Candidate Promotion Readiness")
$lines.Add("")
$lines.Add("- readiness state: ``$($record.readinessState)``")
$lines.Add("- source records: ``$($record.sourceRecordCount)``")
$lines.Add("- candidate promotion lanes: ``$($record.candidatePromotionLaneCount)``")
$lines.Add("- blocked candidate promotion lanes: ``$($record.blockedCandidatePromotionLaneCount)``")
$lines.Add("- failed blockers: ``$($record.failedBlockerCount)``")
$lines.Add("- failed action-required: ``$($record.failedActionRequiredCount)``")
$lines.Add("- boundary: $($record.boundary)")
$lines.Add("")
$lines.Add("## Candidate Promotion Matrix")
$lines.Add("")
$lines.Add("| lane | state | strict validator | required inputs | blocked until |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($lane in $lanes) {
  $requiredInputs = ($lane.requiredRealInputs -join ", ").Replace("|", "\|")
  $blockedUntil = ([string]$lane.promotionBlockedUntil).Replace("|", "\|")
  $lines.Add("| $($lane.laneId) | $($lane.currentState) | $($lane.strictValidator) | $requiredInputs | $blockedUntil |")
}
$lines.Add("")
$lines.Add("## Source Records")
$lines.Add("")
$lines.Add("| id | state | count | blocked | failed blockers | action required |")
$lines.Add("| --- | --- | ---: | ---: | ---: | ---: |")
foreach ($source in $sourceRecords) {
  $lines.Add("| $($source.id) | $($source.state) | $($source.count) | $($source.blockedCount) | $($source.failedBlockerCount) | $($source.failedActionRequiredCount) |")
}
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "ReleaseClose real input candidate promotion readiness written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ReadinessState=$($record.readinessState) Lanes=$($record.candidatePromotionLaneCount) Sources=$($record.sourceRecordCount) FailedBlockers=$($record.failedBlockerCount)"
