[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-release-close-blocker-dashboard.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final release close blocker dashboard not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$blockers = @((Get-PropertyOrDefault -Object $record -Name "blockers" -DefaultValue @()))

$requiredIds = @(
  "clean-external-package-consumer-owner-runbook",
  "post-publish-owner-verification-runbook",
  "package-consumer-owner-runtime-smoke-field-alignment",
  "package-consumer-runtime-ownerproof-schema-scan",
  "owner-external-proof-execution-bundle",
  "owner-external-proof-result-import",
  "real-proof-record-candidate-from-owner-result-import",
  "external-runtime-proof-validation",
  "release-package-proof-bundle",
  "release-candidate-final-freeze-manifest",
  "public-publish-owner-manual-command-handoff",
  "public-package-proof-owner-input",
  "post-publish-proof-owner-confirmation",
  "post-publish-verification-validation",
  "yolovision-real-model-proof-boundary",
  "release-close-public-proof-bridge",
  "final-post-publish-audit-pack",
  "release-issue-close-final-owner-decision-audit",
  "strict-release-close-validator",
  "release-evidence-classification-audit"
)

$ids = @($blockers | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "blockerId" -DefaultValue "") })
$missingRequired = @($requiredIds | Where-Object { $ids -notcontains $_ })
$allBlockersSafe = $blockers.Count -ge $requiredIds.Count
foreach ($blocker in $blockers) {
  $allBlockersSafe = $allBlockersSafe -and
    -not [bool](Get-PropertyOrDefault -Object $blocker -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $blocker -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $blocker -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $blocker -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $blocker -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $blocker -Name "isReleaseCloseProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $blocker -Name "isPostPublishProof" -DefaultValue $true) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $blocker -Name "requiredProof" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $blocker -Name "ownerNextAction" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $blocker -Name "validator" -DefaultValue "")) -and
    ([string](Get-PropertyOrDefault -Object $blocker -Name "whyNonSubstitute" -DefaultValue "")).Contains("cannot", [StringComparison]::OrdinalIgnoreCase)
}

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-release-close-blocker-dashboard") -Severity "blocker" -Detail "recordKind must be final-release-close-blocker-dashboard.")) | Out-Null
$items.Add((New-ValidationItem -Id "dashboard-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "dashboardState" -DefaultValue "") -eq "blocked-final-release-close-owner-action-required") -Severity "action-required" -Detail "Dashboard must stay blocked until all real owner proof gates are ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-blockers-present" -Passed ($missingRequired.Count -eq 0) -Severity "blocker" -Detail "Dashboard must include every final release close blocker. Missing: $($missingRequired -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedBlockerCount" -DefaultValue 0) -gt 0) -Severity "action-required" -Detail "Dashboard must expose remaining owner-action blockers until real proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "blockers-safe" -Passed $allBlockersSafe -Severity "blocker" -Detail "Every blocker must carry required proof, owner action, validator, non-substitute rationale, and false proof/publish/close flags.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runbooks-projected" -Passed ($record.PSObject.Properties.Name -contains "cleanExternalRunbookValidationState" -and $record.PSObject.Properties.Name -contains "cleanExternalRunbookStepCount" -and $record.PSObject.Properties.Name -contains "cleanExternalRunbookFailedBlockerCount" -and $record.PSObject.Properties.Name -contains "postPublishRunbookValidationState" -and $record.PSObject.Properties.Name -contains "postPublishRunbookStepCount" -and $record.PSObject.Properties.Name -contains "postPublishRunbookFailedBlockerCount" -and [int](Get-PropertyOrDefault -Object $record -Name "cleanExternalRunbookStepCount" -DefaultValue 0) -gt 0 -and [int](Get-PropertyOrDefault -Object $record -Name "postPublishRunbookStepCount" -DefaultValue 0) -gt 0 -and [int](Get-PropertyOrDefault -Object $record -Name "cleanExternalRunbookFailedBlockerCount" -DefaultValue -1) -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "postPublishRunbookFailedBlockerCount" -DefaultValue -1) -eq 0 -and ($ids -contains "clean-external-package-consumer-owner-runbook") -and ($ids -contains "post-publish-owner-verification-runbook")) -Severity "blocker" -Detail "Dashboard must project both owner runbook lanes while keeping them blocked/non-proof until real owner execution results are imported.")) | Out-Null
$items.Add((New-ValidationItem -Id "ownerproof-schema-scan-projected" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "ownerInputSchemaReady" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteScanState" -DefaultValue "") -ne "" -and [int](Get-PropertyOrDefault -Object $record -Name "detectedForbiddenSubstituteCount" -DefaultValue -1) -ge 0 -and ($ids -contains "package-consumer-runtime-ownerproof-schema-scan")) -Severity "blocker" -Detail "Dashboard must project owner input schema readiness and forbidden substitute scan state as an explicit release-close blocker.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runtime-smoke-field-alignment-projected" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState" -DefaultValue "") -eq "blocked-owner-compatible-host-runtime-smoke-field-alignment-valid" -and [string](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus" -DefaultValue "") -eq "Smoke=not-requested" -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount" -DefaultValue 0) -ge 30 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount" -DefaultValue -1) -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount" -DefaultValue -1) -eq 0 -and ($ids -contains "package-consumer-owner-runtime-smoke-field-alignment")) -Severity "blocker" -Detail "Dashboard must project owner runtime smoke field alignment as zero-missing non-proof coverage and keep runtime smoke blocked until real evidence exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-owner-input-readiness-projected" -Passed ($record.PSObject.Properties.Name -contains "cleanOwnerInputReady" -and $record.PSObject.Properties.Name -contains "ownerInputForbiddenSubstituteFree" -and $record.PSObject.Properties.Name -contains "ownerInputHashFieldsReady" -and $record.PSObject.Properties.Name -contains "ownerInputSmokeLogReady" -and -not [bool](Get-PropertyOrDefault -Object $record -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $true)) -Severity "blocker" -Detail "Dashboard must project clean/hash/smoke owner input readiness while keeping proof promotion false.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-external-proof-execution-bundle-projected" -Passed ($record.PSObject.Properties.Name -contains "ownerExternalProofExecutionBundleState" -and $record.PSObject.Properties.Name -contains "ownerExternalProofExecutionBundleFailedBlockers" -and [int](Get-PropertyOrDefault -Object $record -Name "runtimeProofPreflightEntryCount" -DefaultValue 0) -ge 6 -and ($ids -contains "owner-external-proof-execution-bundle")) -Severity "blocker" -Detail "Dashboard must project owner execution bundle validation and RuntimeProofPreflight option count.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-external-proof-result-import-projected" -Passed ($record.PSObject.Properties.Name -contains "ownerExternalProofResultImportState" -and $record.PSObject.Properties.Name -contains "ownerExternalProofResultImportFailedBlockers" -and $record.PSObject.Properties.Name -contains "ownerExternalProofResultLaneCount" -and $record.PSObject.Properties.Name -contains "ownerExternalProofResultFileMissingCount" -and $record.PSObject.Properties.Name -contains "ownerExternalProofResultInvalidSha256Count" -and $record.PSObject.Properties.Name -contains "ownerExternalProofResultHashMismatchCount" -and $record.PSObject.Properties.Name -contains "ownerExternalProofResultOutsideAllowedEvidenceRootCount" -and $record.PSObject.Properties.Name -contains "ownerExternalProofResultForbiddenSubstituteFindingCount" -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultLaneCount" -DefaultValue 0) -eq 6 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultPromotableLaneCount" -DefaultValue -1) -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultFileMissingCount" -DefaultValue -1) -ge 0 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultInvalidSha256Count" -DefaultValue -1) -ge 0 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultHashMismatchCount" -DefaultValue -1) -ge 0 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultOutsideAllowedEvidenceRootCount" -DefaultValue -1) -ge 0 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultForbiddenSubstituteFindingCount" -DefaultValue -1) -ge 0 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultImportCanPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultImportIsPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "ownerExternalProofResultImportCanCloseReleaseIssue" -DefaultValue $true) -and ($ids -contains "owner-external-proof-result-import")) -Severity "blocker" -Detail "Dashboard must project imported owner result status and classified evidence failures while keeping proof, post-publish, and close flags false.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-result-candidate-bridge-projected" -Passed ($record.PSObject.Properties.Name -contains "realProofRecordCandidateFromOwnerResultImportState" -and $record.PSObject.Properties.Name -contains "realProofRecordCandidateFromOwnerResultImportValidationState" -and $record.PSObject.Properties.Name -contains "realProofRecordCandidateFromOwnerResultImportCandidateCount" -and $record.PSObject.Properties.Name -contains "realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount" -and $record.PSObject.Properties.Name -contains "realProofRecordCandidateFromOwnerResultImportPackageConsumerCandidateCount" -and $record.PSObject.Properties.Name -contains "realProofRecordCandidateFromOwnerResultImportPostPublishCandidateCount" -and [int](Get-PropertyOrDefault -Object $record -Name "realProofRecordCandidateFromOwnerResultImportCandidateCount" -DefaultValue -1) -ge 0 -and [int](Get-PropertyOrDefault -Object $record -Name "realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount" -DefaultValue -1) -ge 0 -and -not [bool](Get-PropertyOrDefault -Object $record -Name "realProofRecordCandidateFromOwnerResultImportCanPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "realProofRecordCandidateFromOwnerResultImportIsRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "realProofRecordCandidateFromOwnerResultImportIsPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "realProofRecordCandidateFromOwnerResultImportCanCloseReleaseIssue" -DefaultValue $true) -and ($ids -contains "real-proof-record-candidate-from-owner-result-import")) -Severity "blocker" -Detail "Dashboard must project owner-result candidate bridge while keeping candidates non-proof and post-publish separated.")) | Out-Null
$items.Add((New-ValidationItem -Id "external-runtime-proof-projected" -Passed ($record.PSObject.Properties.Name -contains "externalRuntimeProofState" -and $record.PSObject.Properties.Name -contains "externalRuntimeProofPreflightAligned" -and -not [bool](Get-PropertyOrDefault -Object $record -Name "externalRuntimeProofCanPromote" -DefaultValue $true) -and ($ids -contains "external-runtime-proof-validation")) -Severity "blocker" -Detail "Dashboard must project external runtime proof validation and keep it non-promoted until real proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "release-package-and-post-publish-projected" -Passed ($record.PSObject.Properties.Name -contains "releasePackageProofState" -and $record.PSObject.Properties.Name -contains "postPublishVerificationState" -and -not [bool](Get-PropertyOrDefault -Object $record -Name "releasePackageCanPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "postPublishIsProof" -DefaultValue $true) -and ($ids -contains "release-package-proof-bundle") -and ($ids -contains "post-publish-verification-validation")) -Severity "blocker" -Detail "Dashboard must project release package and post-publish proof states without promoting local/preflight artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "yolovision-boundary-projected" -Passed ($record.PSObject.Properties.Name -contains "yoloVisionOwnerProofState" -and $record.PSObject.Properties.Name -contains "sampleRunEvidenceState" -and ($ids -contains "yolovision-real-model-proof-boundary")) -Severity "blocker" -Detail "Dashboard must project YoloVision real-model-runtime boundary separately from package-consumer-runtime and post-publish proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Blocker dashboard must not publish, prove runtime, prove post-publish, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-final-release-close-blocker-dashboard"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-final-release-close-owner-action-required"
}
else {
  "final-release-close-blocker-dashboard-ready"
}

$validation = [pscustomobject]@{
  recordKind = "final-release-close-blocker-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  blockerCount = $blockers.Count
  blockedBlockerCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedBlockerCount" -DefaultValue 0)
  readyBlockerCount = [int](Get-PropertyOrDefault -Object $record -Name "readyBlockerCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Final release close blocker dashboard validation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "final-release-close-blocker-dashboard-validation.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-blocker-dashboard-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Final Release Close Blocker Dashboard Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| blockerCount | ``$($validation.blockerCount)`` |
| blockedBlockerCount | ``$($validation.blockedBlockerCount)`` |
| readyBlockerCount | ``$($validation.readyBlockerCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final release close blocker dashboard validation written to $jsonPath"
Write-Host "ValidationState=$validationState Blockers=$($validation.blockerCount) Blocked=$($validation.blockedBlockerCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final release close blocker dashboard has blocker validation failures."
}
