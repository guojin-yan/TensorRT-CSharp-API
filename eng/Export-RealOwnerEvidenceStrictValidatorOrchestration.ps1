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

function Get-RecordState {
  param([AllowNull()][object]$Record)
  foreach ($name in @("orchestrationState", "validationState", "convergenceState", "orderState", "gateState", "validatorState", "importState", "candidateState", "recordState")) {
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
    failedBlockerCount = if ($null -eq $Record) { 1 } else { [int](Get-PropertyOrDefault -Object $Record -Name $BlockerField -DefaultValue 0) }
    performsPublish = [bool](Get-PropertyOrDefault -Object $Record -Name "performsPublish" -DefaultValue $false)
    canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $Record -Name "canPromoteRuntimeProof" -DefaultValue $false)
    canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
    canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
    isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isRuntimeExecutionProof" -DefaultValue $false)
    isPostPublishProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isPostPublishProof" -DefaultValue $false)
    isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseCloseProof" -DefaultValue $false)
  }
}

function New-FieldReadiness {
  param(
    [string]$FieldName,
    [string[]]$RequiredBy,
    [string]$OwnerInputSurface,
    [string]$StrictValidatorConsumer,
    [string]$BlockedUntil
  )

  [pscustomobject]@{
    fieldName = $FieldName
    requiredBy = @($RequiredBy)
    requiredByCount = @($RequiredBy).Count
    ownerInputSurface = $OwnerInputSurface
    strictValidatorConsumer = $StrictValidatorConsumer
    currentState = "blocked-real-owner-input-required"
    blockedUntil = $BlockedUntil
    nonSubstituteBoundary = "Must be supplied as real Owner evidence and checked by strict validators; local dry-run, ProjectReference, direct nupkg, local feed, candidate, dashboard, runbook, and build-only substitutes cannot satisfy this field."
    canBeSatisfiedByLocalDryRun = $false
    canBeSatisfiedByProjectReference = $false
    canBeSatisfiedByDirectNupkg = $false
    canBeSatisfiedByLocalFeed = $false
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$ownerInputContractConvergence = Read-JsonOrNull "artifacts\final-release\owner-input-contract-convergence.json"
$ownerInputContractConvergenceValidation = Read-JsonOrNull "artifacts\final-release\owner-input-contract-convergence-validation.json"
$finalOwnerStrictCloseExecutionOrder = Read-JsonOrNull "artifacts\final-release\final-owner-strict-close-execution-order.json"
$finalOwnerStrictCloseExecutionOrderValidation = Read-JsonOrNull "artifacts\final-release\final-owner-strict-close-execution-order-validation.json"
$publicPublishResultOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-owner-input-validation.json"
$packageConsumerRuntimeProofOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$postPublishCleanConsumerProofRecordContractValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-contract-validation.json"
$finalOwnerRealInputTemplatePackValidation = Read-JsonOrNull "artifacts\final-release\final-owner-real-input-template-pack-validation.json"
$ownerExternalProofExecutionResultImportValidation = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result-import-validation.json"
$realProofRecordCandidateFromOwnerResultImportValidation = Read-JsonOrNull "artifacts\final-release\real-proof-record-candidate-from-owner-result-import-validation.json"
$realProofInputCandidateStrictRecordValidation = Read-JsonOrNull "artifacts\final-release\real-proof-input-candidate-strict-record-validation.json"
$publicPackageHashCrossCheckGate = Read-JsonOrNull "artifacts\final-release\public-package-hash-cross-check-gate.json"
$publicPackageHashCrossCheckGateValidation = Read-JsonOrNull "artifacts\final-release\public-package-hash-cross-check-gate-validation.json"
$finalReleaseCloseRecordRealValidatorValidation = Read-JsonOrNull "artifacts\final-release\final-release-close-record-real-validator-validation.json"
$finalPublishProofGateReport = Read-JsonOrNull "artifacts\final-release\final-publish-proof-gate-report.json"

$sourceRecords = @(
  New-SourceRecordSummary -Id "owner-input-contract-convergence" -ArtifactPath "artifacts/final-release/owner-input-contract-convergence-validation.json" -Record $ownerInputContractConvergenceValidation -StateField "validationState" -CountField "contractSurfaceCount" -BlockedCountField "blockedContractSurfaceCount"
  New-SourceRecordSummary -Id "final-owner-strict-close-execution-order" -ArtifactPath "artifacts/final-release/final-owner-strict-close-execution-order-validation.json" -Record $finalOwnerStrictCloseExecutionOrderValidation -StateField "validationState" -CountField "stepCount" -BlockedCountField "blockedStepCount"
  New-SourceRecordSummary -Id "public-publish-result-owner-input" -ArtifactPath "artifacts/final-release/public-publish-result-owner-input-validation.json" -Record $publicPublishResultOwnerInputValidation -StateField "validationState" -CountField "failedActionRequiredCount" -BlockedCountField "failedActionRequiredCount"
  New-SourceRecordSummary -Id "package-consumer-runtime-proof-owner-input" -ArtifactPath "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json" -Record $packageConsumerRuntimeProofOwnerInputValidation -StateField "validationState" -CountField "failedActionRequiredCount" -BlockedCountField "failedActionRequiredCount"
  New-SourceRecordSummary -Id "post-publish-clean-consumer-proof-record-contract" -ArtifactPath "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json" -Record $postPublishCleanConsumerProofRecordContractValidation -StateField "validationState" -CountField "requiredFieldCount" -BlockedCountField "blockedRequiredFieldCount"
  New-SourceRecordSummary -Id "final-owner-real-input-template-pack-validation" -ArtifactPath "artifacts/final-release/final-owner-real-input-template-pack-validation.json" -Record $finalOwnerRealInputTemplatePackValidation -StateField "validationState" -CountField "laneCount" -BlockedCountField "failedActionRequiredCount"
  New-SourceRecordSummary -Id "owner-external-proof-execution-result-import" -ArtifactPath "artifacts/final-release/owner-external-proof-execution-result-import-validation.json" -Record $ownerExternalProofExecutionResultImportValidation -StateField "validationState" -CountField "resultImportItemCount" -BlockedCountField "blockedResultImportItemCount"
  New-SourceRecordSummary -Id "real-proof-record-candidate-from-owner-result-import" -ArtifactPath "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json" -Record $realProofRecordCandidateFromOwnerResultImportValidation -StateField "validationState" -CountField "candidateCount" -BlockedCountField "strictValidatorReadyCandidateCount"
  New-SourceRecordSummary -Id "real-proof-input-candidate-strict-record" -ArtifactPath "artifacts/final-release/real-proof-input-candidate-strict-record-validation.json" -Record $realProofInputCandidateStrictRecordValidation -StateField "validationState" -CountField "candidateCount" -BlockedCountField "blockedCandidateCount"
  New-SourceRecordSummary -Id "public-package-hash-cross-check-gate" -ArtifactPath "artifacts/final-release/public-package-hash-cross-check-gate-validation.json" -Record $publicPackageHashCrossCheckGateValidation -StateField "validationState" -CountField "findingCount" -BlockedCountField "findingCount"
  New-SourceRecordSummary -Id "final-release-close-record-real-validator" -ArtifactPath "artifacts/final-release/final-release-close-record-real-validator-validation.json" -Record $finalReleaseCloseRecordRealValidatorValidation -StateField "validationState" -CountField "requiredFieldCount" -BlockedCountField "blockedRequiredFieldCount"
  New-SourceRecordSummary -Id "final-publish-proof-gate" -ArtifactPath "artifacts/final-release/final-publish-proof-gate-report.json" -Record $finalPublishProofGateReport -StateField "validationState" -CountField "failedActionRequiredCount" -BlockedCountField "failedActionRequiredCount"
)

$fieldMatrix = @(
  New-FieldReadiness -FieldName "publicPackageSourceUrl" -RequiredBy @("public-publish-result-owner-input", "public-package-hash-cross-check-gate", "post-publish-clean-consumer-proof-record") -OwnerInputSurface "public-publish-result-owner-input" -StrictValidatorConsumer "public-package-hash-cross-check-gate" -BlockedUntil "Owner supplies real public package source URL from the selected public package channel."
  New-FieldReadiness -FieldName "downloadedNupkgSha256" -RequiredBy @("public-publish-result-owner-input", "public-package-hash-cross-check-gate", "final-release-close-record-real-validator") -OwnerInputSurface "public-publish-result-owner-input" -StrictValidatorConsumer "public-package-hash-cross-check-gate" -BlockedUntil "Owner downloads the public package and records the actual SHA256."
  New-FieldReadiness -FieldName "packageId" -RequiredBy @("public-publish-result-owner-input", "package-consumer-runtime-proof-owner-input") -OwnerInputSurface "public-publish-result-owner-input" -StrictValidatorConsumer "final-publish-proof-gate" -BlockedUntil "Owner records the exact public package id used by clean consumer proof."
  New-FieldReadiness -FieldName "packageVersion" -RequiredBy @("public-publish-result-owner-input", "package-consumer-runtime-proof-owner-input") -OwnerInputSurface "public-publish-result-owner-input" -StrictValidatorConsumer "final-publish-proof-gate" -BlockedUntil "Owner records the exact package version installed by clean consumers."
  New-FieldReadiness -FieldName "packageSourceKind" -RequiredBy @("public-publish-result-owner-input", "post-publish-clean-consumer-proof-record") -OwnerInputSurface "public-publish-result-owner-input" -StrictValidatorConsumer "post-publish-clean-consumer-proof-record-contract" -BlockedUntil "Owner classifies the real package source, such as public NuGet or GitHub package channel."
  New-FieldReadiness -FieldName "cleanConsumerProjectPath" -RequiredBy @("package-consumer-runtime-proof-owner-input", "post-publish-clean-consumer-proof-record") -OwnerInputSurface "package-consumer-runtime-proof-owner-input" -StrictValidatorConsumer "owner-external-proof-execution-result-import" -BlockedUntil "Owner uses a repository-external clean consumer project path."
  New-FieldReadiness -FieldName "cleanConsumerLogPath" -RequiredBy @("package-consumer-runtime-proof-owner-input", "post-publish-clean-consumer-proof-record") -OwnerInputSurface "package-consumer-runtime-proof-owner-input" -StrictValidatorConsumer "owner-external-proof-execution-result-import" -BlockedUntil "Owner supplies existing clean consumer restore/build/run log paths."
  New-FieldReadiness -FieldName "cleanConsumerLogSha256" -RequiredBy @("package-consumer-runtime-proof-owner-input", "owner-external-proof-execution-result-import") -OwnerInputSurface "package-consumer-runtime-proof-owner-input" -StrictValidatorConsumer "owner-external-proof-execution-result-import" -BlockedUntil "Owner records SHA256 for clean consumer evidence logs."
  New-FieldReadiness -FieldName "postPublishInstallLogPath" -RequiredBy @("post-publish-clean-consumer-proof-record", "final-publish-proof-gate") -OwnerInputSurface "post-publish-clean-consumer-proof-record" -StrictValidatorConsumer "final-publish-proof-gate" -BlockedUntil "Owner installs from public package source after publish and records install log path."
  New-FieldReadiness -FieldName "postPublishRunLogPath" -RequiredBy @("post-publish-clean-consumer-proof-record", "final-publish-proof-gate") -OwnerInputSurface "post-publish-clean-consumer-proof-record" -StrictValidatorConsumer "final-publish-proof-gate" -BlockedUntil "Owner runs post-publish clean consumer and records run log path."
  New-FieldReadiness -FieldName "stdoutPath" -RequiredBy @("owner-external-proof-execution-result-import", "real-proof-input-candidate-strict-record") -OwnerInputSurface "owner-external-proof-execution-result-import" -StrictValidatorConsumer "real-proof-input-candidate-strict-record" -BlockedUntil "Owner supplies real stdout file paths for every required proof lane."
  New-FieldReadiness -FieldName "stderrPath" -RequiredBy @("owner-external-proof-execution-result-import", "real-proof-input-candidate-strict-record") -OwnerInputSurface "owner-external-proof-execution-result-import" -StrictValidatorConsumer "real-proof-input-candidate-strict-record" -BlockedUntil "Owner supplies real stderr file paths for every required proof lane."
  New-FieldReadiness -FieldName "hostMetadata" -RequiredBy @("owner-external-proof-execution-result-import", "final-release-close-record-real-validator") -OwnerInputSurface "owner-external-proof-execution-result-import" -StrictValidatorConsumer "final-release-close-record-real-validator" -BlockedUntil "Owner supplies compatible host metadata for runtime, package consumer, and post-publish lanes."
  New-FieldReadiness -FieldName "nonSubstituteConfirmations" -RequiredBy @("owner-input-contract-convergence", "owner-external-proof-execution-result-import", "release-evidence-classification-audit") -OwnerInputSurface "owner-input-contract-convergence" -StrictValidatorConsumer "release-evidence-classification-audit" -BlockedUntil "Owner confirms no local feed, ProjectReference, direct nupkg, dry-run, dashboard, candidate, template, or build-only substitute was used."
  New-FieldReadiness -FieldName "rollbackReview" -RequiredBy @("final-release-close-record-real-validator", "final-publish-proof-gate") -OwnerInputSurface "release-issue-close-owner-decision-input" -StrictValidatorConsumer "final-release-close-record-real-validator" -BlockedUntil "Owner records rollback review after public publish and post-publish proof."
  New-FieldReadiness -FieldName "finalCloseDecision" -RequiredBy @("final-release-close-record-real-validator", "release-issue-close-record") -OwnerInputSurface "release-issue-close-owner-decision-input" -StrictValidatorConsumer "final-release-close-record-real-validator" -BlockedUntil "Owner supplies final close decision only after all strict proof validators pass."
)

$sourceArtifacts = @(
  "artifacts/final-release/owner-input-contract-convergence.json",
  "artifacts/final-release/owner-input-contract-convergence-validation.json",
  "artifacts/final-release/final-owner-strict-close-execution-order.json",
  "artifacts/final-release/final-owner-strict-close-execution-order-validation.json",
  "artifacts/final-release/public-publish-result-owner-input-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
  "artifacts/final-release/final-owner-real-input-template-pack-validation.json",
  "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
  "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
  "artifacts/final-release/real-proof-input-candidate-strict-record-validation.json",
  "artifacts/final-release/public-package-hash-cross-check-gate.json",
  "artifacts/final-release/public-package-hash-cross-check-gate-validation.json",
  "artifacts/final-release/final-release-close-record-real-validator-validation.json",
  "artifacts/final-release/final-publish-proof-gate-report.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-evidence-classification-audit.json"
)

$blockedFieldCount = @($fieldMatrix | Where-Object { $_.currentState -like "blocked-*" }).Count
$structuralSourceRecords = @($sourceRecords | Where-Object {
  $_.exists -and
  $_.id -notin @(
    "final-publish-proof-gate-report",
    "final-owner-real-input-template-pack-validation"
  )
})
$failedBlockerCount = @($structuralSourceRecords | Where-Object { $_.failedBlockerCount -gt 0 }).Count
$failedActionRequiredCount = (@($sourceRecords | Measure-Object -Property failedActionRequiredCount -Sum).Sum)
$validatorConsumerCount = @($fieldMatrix | ForEach-Object { $_.strictValidatorConsumer } | Select-Object -Unique).Count
$ownerInputSurfaceCount = @($fieldMatrix | ForEach-Object { $_.ownerInputSurface } | Select-Object -Unique).Count

$record = [pscustomobject]@{
  recordKind = "real-owner-evidence-strict-validator-orchestration"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  orchestrationState = "blocked-real-owner-evidence-strict-validator-real-owner-input-required"
  sourceRecordCount = $sourceRecords.Count
  fieldReadinessCount = $fieldMatrix.Count
  blockedFieldReadinessCount = $blockedFieldCount
  readyFieldReadinessCount = 0
  validatorConsumerCount = $validatorConsumerCount
  ownerInputSurfaceCount = $ownerInputSurfaceCount
  failedBlockerCount = $failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  sourceRecords = @($sourceRecords)
  fieldReadinessMatrix = @($fieldMatrix)
  sourceStates = [pscustomobject]@{
    ownerInputContractConvergence = Get-RecordState -Record $ownerInputContractConvergence
    ownerInputContractConvergenceValidation = Get-RecordState -Record $ownerInputContractConvergenceValidation
    finalOwnerStrictCloseExecutionOrder = Get-RecordState -Record $finalOwnerStrictCloseExecutionOrder
    finalOwnerStrictCloseExecutionOrderValidation = Get-RecordState -Record $finalOwnerStrictCloseExecutionOrderValidation
    publicPublishResultOwnerInputValidation = Get-RecordState -Record $publicPublishResultOwnerInputValidation
    packageConsumerRuntimeProofOwnerInputValidation = Get-RecordState -Record $packageConsumerRuntimeProofOwnerInputValidation
    postPublishCleanConsumerProofRecordContractValidation = Get-RecordState -Record $postPublishCleanConsumerProofRecordContractValidation
    finalOwnerRealInputTemplatePackValidation = Get-RecordState -Record $finalOwnerRealInputTemplatePackValidation
    ownerExternalProofExecutionResultImportValidation = Get-RecordState -Record $ownerExternalProofExecutionResultImportValidation
    realProofRecordCandidateFromOwnerResultImportValidation = Get-RecordState -Record $realProofRecordCandidateFromOwnerResultImportValidation
    realProofInputCandidateStrictRecordValidation = Get-RecordState -Record $realProofInputCandidateStrictRecordValidation
    publicPackageHashCrossCheckGate = Get-RecordState -Record $publicPackageHashCrossCheckGate
    publicPackageHashCrossCheckGateValidation = Get-RecordState -Record $publicPackageHashCrossCheckGateValidation
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
    "bundle-ready",
    "blocked-by-cuda-driver"
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
  boundary = "Real Owner evidence StrictValidator orchestration is a blocked coordination map only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 only means structural blockers are absent; it is not proof ready."
}

$jsonPath = Join-Path $OutputRoot "real-owner-evidence-strict-validator-orchestration.json"
$markdownPath = Join-Path $OutputRoot "real-owner-evidence-strict-validator-orchestration.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Real Owner Evidence StrictValidator Orchestration")
$lines.Add("")
$lines.Add("- orchestration state: ``$($record.orchestrationState)``")
$lines.Add("- source records: ``$($record.sourceRecordCount)``")
$lines.Add("- field readiness rows: ``$($record.fieldReadinessCount)``")
$lines.Add("- blocked field readiness rows: ``$($record.blockedFieldReadinessCount)``")
$lines.Add("- validator consumers: ``$($record.validatorConsumerCount)``")
$lines.Add("- owner input surfaces: ``$($record.ownerInputSurfaceCount)``")
$lines.Add("- failed blockers: ``$($record.failedBlockerCount)``")
$lines.Add("- failed action-required: ``$($record.failedActionRequiredCount)``")
$lines.Add("- boundary: $($record.boundary)")
$lines.Add("")
$lines.Add("## Source Records")
$lines.Add("")
$lines.Add("| id | state | count | blocked | failed blockers | action required |")
$lines.Add("| --- | --- | ---: | ---: | ---: | ---: |")
foreach ($source in $sourceRecords) {
  $lines.Add("| $($source.id) | $($source.state) | $($source.count) | $($source.blockedCount) | $($source.failedBlockerCount) | $($source.failedActionRequiredCount) |")
}
$lines.Add("")
$lines.Add("## Field Readiness Matrix")
$lines.Add("")
$lines.Add("| field | owner input surface | strict validator consumer | current state | blocked until |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($field in $fieldMatrix) {
  $blockedUntil = ([string]$field.blockedUntil).Replace("|", "\|")
  $lines.Add("| $($field.fieldName) | $($field.ownerInputSurface) | $($field.strictValidatorConsumer) | $($field.currentState) | $blockedUntil |")
}
$lines.Add("")
$lines.Add("## Forbidden Substitutes")
$lines.Add("")
foreach ($item in $record.forbiddenSubstitutes) {
  $lines.Add("- ``$item``")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "Real Owner evidence StrictValidator orchestration written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "OrchestrationState=$($record.orchestrationState) Fields=$($record.fieldReadinessCount) Sources=$($record.sourceRecordCount) FailedBlockers=$($record.failedBlockerCount)"
