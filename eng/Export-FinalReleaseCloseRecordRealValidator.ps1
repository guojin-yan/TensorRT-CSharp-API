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

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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

function New-FieldContract {
  param(
    [string]$Id,
    [string]$TargetArtifact,
    [string]$TargetField,
    [string]$CurrentState,
    [string]$RequiredEvidence,
    [string]$Validator
  )

  [pscustomobject]@{
    id = $Id
    targetArtifact = $TargetArtifact
    targetField = $TargetField
    currentState = $CurrentState
    requiredEvidence = $RequiredEvidence
    validator = $Validator
    ready = $false
    boundary = "Final close record real validator contract only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$releaseIssueCloseRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$publicPublishDraftValidation = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-record-draft-validation.json"
$cleanConsumerDraftValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-draft-validation.json"
$forbiddenSubstituteScanValidation = Read-JsonOrNull "artifacts\final-release\public-publish-forbidden-substitute-scan-validation.json"
$realProofImportBridgeValidation = Read-JsonOrNull "artifacts\final-release\release-close-real-proof-import-bridge-validation.json"
$finalOwnerCheckpointValidation = Read-JsonOrNull "artifacts\final-release\final-owner-close-readiness-checkpoint-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$releaseIssueCloseRecordValidationState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
$publicPublishDraftValidationState = [string](Get-PropertyOrDefault -Object $publicPublishDraftValidation -Name "validationState" -DefaultValue "missing-public-publish-real-result-record-draft-validation")
$cleanConsumerDraftValidationState = [string](Get-PropertyOrDefault -Object $cleanConsumerDraftValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-draft-validation")
$forbiddenSubstituteScanValidationState = [string](Get-PropertyOrDefault -Object $forbiddenSubstituteScanValidation -Name "validationState" -DefaultValue "missing-public-publish-forbidden-substitute-scan-validation")
$realProofImportBridgeValidationState = [string](Get-PropertyOrDefault -Object $realProofImportBridgeValidation -Name "validationState" -DefaultValue "missing-release-close-real-proof-import-bridge-validation")
$finalOwnerCheckpointValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerCheckpointValidation -Name "validationState" -DefaultValue "missing-final-owner-close-readiness-checkpoint-validation")
$classificationAuditState = [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")

$fieldContracts = @(
  New-FieldContract -Id "release-issue-id" -TargetArtifact "artifacts/final-release/release-issue-close-record.json" -TargetField "releaseIssue.id" -CurrentState $releaseIssueCloseRecordValidationState -RequiredEvidence "Real public release issue id." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
  New-FieldContract -Id "release-issue-url" -TargetArtifact "artifacts/final-release/release-issue-close-record.json" -TargetField "releaseIssue.url" -CurrentState $releaseIssueCloseRecordValidationState -RequiredEvidence "Real release issue URL matching the release issue id." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
  New-FieldContract -Id "owner-final-close-decision" -TargetArtifact "artifacts/final-release/release-issue-close-record.json" -TargetField "ownerDecision.finalCloseDecision" -CurrentState $releaseIssueCloseRecordValidationState -RequiredEvidence "Explicit Owner decision to close after all real proof validators pass." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
  New-FieldContract -Id "owner-close-reviewer" -TargetArtifact "artifacts/final-release/release-issue-close-record.json" -TargetField "ownerDecision.ownerName" -CurrentState $releaseIssueCloseRecordValidationState -RequiredEvidence "Named Owner reviewer accountable for the final close decision." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
  New-FieldContract -Id "public-package-source" -TargetArtifact "artifacts/final-release/public-publish-real-result-record-draft.json" -TargetField "publicPackageSource" -CurrentState $publicPublishDraftValidationState -RequiredEvidence "Real public package source/channel URL, not local feed/direct nupkg/ProjectReference." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict"
  New-FieldContract -Id "public-package-sha256" -TargetArtifact "artifacts/final-release/public-publish-real-result-record-draft.json" -TargetField "downloadedPackageSha256" -CurrentState $publicPublishDraftValidationState -RequiredEvidence "SHA256 for downloaded public package artifact." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict"
  New-FieldContract -Id "public-publish-transcript" -TargetArtifact "artifacts/final-release/public-publish-real-result-record-draft.json" -TargetField "publishTranscript" -CurrentState $publicPublishDraftValidationState -RequiredEvidence "Owner-captured publish command transcript from the real public channel." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict"
  New-FieldContract -Id "clean-consumer-project" -TargetArtifact "artifacts/final-release/post-publish-clean-consumer-proof-record-draft.json" -TargetField "consumerProjectIdentity" -CurrentState $cleanConsumerDraftValidationState -RequiredEvidence "Repository-external clean consumer project identity." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict"
  New-FieldContract -Id "clean-consumer-smoke-log" -TargetArtifact "artifacts/final-release/post-publish-clean-consumer-proof-record-draft.json" -TargetField "smokeLogSha256" -CurrentState $cleanConsumerDraftValidationState -RequiredEvidence "Real restore/build/smoke log hash from a clean external consumer." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict"
  New-FieldContract -Id "forbidden-substitutes-cleared" -TargetArtifact "artifacts/final-release/public-publish-forbidden-substitute-scan.json" -TargetField "substituteChecks" -CurrentState $forbiddenSubstituteScanValidationState -RequiredEvidence "Proof that local feed, ProjectReference, direct nupkg, dry-run, template and dashboard were not substituted." -Validator "eng\Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict"
  New-FieldContract -Id "real-proof-import-bridge-ready" -TargetArtifact "artifacts/final-release/release-close-real-proof-import-bridge.json" -TargetField "proofImportLanes" -CurrentState $realProofImportBridgeValidationState -RequiredEvidence "All real proof import lanes ready after public package and clean consumer records pass." -Validator "eng\Test-ReleaseCloseRealProofImportBridge.ps1 -Strict"
  New-FieldContract -Id "final-owner-close-readiness-ready" -TargetArtifact "artifacts/final-release/final-owner-close-readiness-checkpoint.json" -TargetField "readinessChecks" -CurrentState $finalOwnerCheckpointValidationState -RequiredEvidence "All final owner close readiness checks unblocked." -Validator "eng\Test-FinalOwnerCloseReadinessCheckpoint.ps1 -Strict"
  New-FieldContract -Id "classification-audit-clean" -TargetArtifact "artifacts/final-release/release-evidence-classification-audit.json" -TargetField "findingCount" -CurrentState $classificationAuditState -RequiredEvidence "Release evidence classification audit must remain clean while preserving non-proof boundaries." -Validator "eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
)

$blockedContracts = @($fieldContracts | Where-Object { -not [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "final-release-close-record-real-validator"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validatorState = "blocked-final-release-close-record-real-proof-required"
  requiredFieldCount = $fieldContracts.Count
  blockedRequiredFieldCount = $blockedContracts.Count
  readyRequiredFieldCount = 0
  fieldContracts = @($fieldContracts)
  sourceArtifacts = @(
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/public-publish-real-result-record-draft-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json",
    "artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json",
    "artifacts/final-release/release-close-real-proof-import-bridge-validation.json",
    "artifacts/final-release/final-owner-close-readiness-checkpoint-validation.json",
    "artifacts/final-release/release-evidence-classification-audit.json"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Final release close record real validator is a blocked field contract only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-release-close-record-real-validator.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-record-real-validator.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $fieldContracts | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.currentState)`` | ``$($_.ready)`` | ``$($_.targetField)`` | $($_.requiredEvidence.Replace("|", "\|")) | ``$($_.validator)`` |"
}

$markdown = @"
# Final Release Close Record Real Validator

| Field | Value |
| --- | --- |
| validatorState | ``$($record.validatorState)`` |
| requiredFieldCount | ``$($record.requiredFieldCount)`` |
| blockedRequiredFieldCount | ``$($record.blockedRequiredFieldCount)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Field Contracts

| ID | Current State | Ready | Target Field | Required Evidence | Validator |
| --- | --- | ---: | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release close record real validator written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidatorState=$($record.validatorState) RequiredFields=$($record.requiredFieldCount) Blocked=$($record.blockedRequiredFieldCount)"
