[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-owner-evidence-strict-validator-orchestration.json",
  [string]$OutputRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

$scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
$RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$inputFullPath = if ([System.IO.Path]::IsPathRooted($InputPath)) { $InputPath } else { Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Input file not found: $inputFullPath"
}

$record = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = [System.Collections.Generic.List[object]]::new()

$sourceRecords = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "sourceRecords" -DefaultValue @()))
$fieldRows = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "fieldReadinessMatrix" -DefaultValue @()))
$sourceArtifacts = ConvertTo-StringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "real-owner-evidence-strict-validator-orchestration") -Severity "blocker" -Detail "recordKind must be real-owner-evidence-strict-validator-orchestration.")) | Out-Null
$items.Add((New-ValidationItem -Id "orchestration-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "orchestrationState" -DefaultValue "") -eq "blocked-real-owner-evidence-strict-validator-real-owner-input-required") -Severity "blocker" -Detail "Orchestration must remain blocked until real Owner evidence exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-record-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "sourceRecordCount" -DefaultValue 0) -ge 11 -and $sourceRecords.Count -ge 11) -Severity "blocker" -Detail "Orchestration must aggregate Owner input, strict validator, hash gate, close validator, and final publish gate records.")) | Out-Null
$items.Add((New-ValidationItem -Id "field-readiness-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "fieldReadinessCount" -DefaultValue 0) -ge 16 -and $fieldRows.Count -ge 16) -Severity "blocker" -Detail "Field readiness matrix must cover the canonical real Owner input fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-fields-blocked" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedFieldReadinessCount" -DefaultValue 0) -eq $fieldRows.Count -and [int](Get-PropertyOrDefault -Object $record -Name "readyFieldReadinessCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Every field must remain blocked until real Owner input is provided.")) | Out-Null
$items.Add((New-ValidationItem -Id "validator-consumer-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "validatorConsumerCount" -DefaultValue 0) -ge 6) -Severity "blocker" -Detail "Matrix must map fields to multiple strict validator consumers.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-surface-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "ownerInputSurfaceCount" -DefaultValue 0) -ge 5) -Severity "blocker" -Detail "Matrix must map fields to multiple Owner input surfaces.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-structural-blockers" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "failedBlockerCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "failedBlockerCount must remain zero while still not implying proof readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "action-required-present" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "Action-required count must show real Owner evidence is still missing.")) | Out-Null
$items.Add((New-ValidationItem -Id "not-executed-by-automation" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false)) -Severity "blocker" -Detail "Orchestration must state it is not executed by automation.")) | Out-Null
$items.Add((New-ValidationItem -Id "does-not-publish" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -Severity "blocker" -Detail "Orchestration must not perform publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "does-not-promote-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -Severity "blocker" -Detail "Orchestration must not promote runtime proof, public publish, or release close.")) | Out-Null
$items.Add((New-ValidationItem -Id "not-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Orchestration must not classify itself as runtime, post-publish, or release-close proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-language" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not proof ready", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must explicitly reject proof, approval, package push, and failedBlockerCount=0 promotion.")) | Out-Null

$requiredFields = @(
  "publicPackageSourceUrl",
  "downloadedNupkgSha256",
  "packageId",
  "packageVersion",
  "packageSourceKind",
  "cleanConsumerProjectPath",
  "cleanConsumerLogPath",
  "cleanConsumerLogSha256",
  "postPublishInstallLogPath",
  "postPublishRunLogPath",
  "stdoutPath",
  "stderrPath",
  "hostMetadata",
  "nonSubstituteConfirmations",
  "rollbackReview",
  "finalCloseDecision"
)
$fieldNames = @($fieldRows | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "fieldName" -DefaultValue "") })
foreach ($field in $requiredFields) {
  $items.Add((New-ValidationItem -Id "field-$field-present" -Passed ($fieldNames -contains $field) -Severity "blocker" -Detail "Field readiness matrix must include $field.")) | Out-Null
}

foreach ($field in $fieldRows) {
  $fieldName = [string](Get-PropertyOrDefault -Object $field -Name "fieldName" -DefaultValue "unknown")
  $substituteFlagsClear =
    (-not [bool](Get-PropertyOrDefault -Object $field -Name "canBeSatisfiedByLocalDryRun" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $field -Name "canBeSatisfiedByProjectReference" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $field -Name "canBeSatisfiedByDirectNupkg" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $field -Name "canBeSatisfiedByLocalFeed" -DefaultValue $true))
  $items.Add((New-ValidationItem -Id "field-$fieldName-non-substitute" -Passed $substituteFlagsClear -Severity "blocker" -Detail "Field $fieldName must reject local dry-run, ProjectReference, direct nupkg, and local feed substitutes.")) | Out-Null
}

$requiredArtifacts = @(
  "artifacts/final-release/owner-input-contract-convergence-validation.json",
  "artifacts/final-release/final-owner-strict-close-execution-order-validation.json",
  "artifacts/final-release/public-publish-result-owner-input-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
  "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
  "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
  "artifacts/final-release/real-proof-input-candidate-strict-record-validation.json",
  "artifacts/final-release/public-package-hash-cross-check-gate-validation.json",
  "artifacts/final-release/final-release-close-record-real-validator-validation.json",
  "artifacts/final-release/final-publish-proof-gate-report.json"
)
foreach ($artifact in $requiredArtifacts) {
  $items.Add((New-ValidationItem -Id "source-$($artifact.Replace('.', '-').Replace('/', '-'))" -Passed ($sourceArtifacts -contains $artifact) -Severity "blocker" -Detail "Source artifact $artifact must be listed.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { $_.severity -eq "blocker" -and -not $_.passed })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-real-owner-evidence-strict-validator-orchestration" } else { "blocked-real-owner-evidence-strict-validator-real-owner-input-required" }

$validation = [pscustomobject]@{
  recordKind = "real-owner-evidence-strict-validator-orchestration-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  sourceRecordCount = $sourceRecords.Count
  fieldReadinessCount = $fieldRows.Count
  blockedFieldReadinessCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedFieldReadinessCount" -DefaultValue 0)
  validatorConsumerCount = [int](Get-PropertyOrDefault -Object $record -Name "validatorConsumerCount" -DefaultValue 0)
  ownerInputSurfaceCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerInputSurfaceCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0)
  validationItems = @($items)
  notExecutedByAutomation = [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $true)
  performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
  canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
  canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)
  isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $false)
  isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false)
  boundary = $boundary
}

$jsonPath = Join-Path $OutputRoot "real-owner-evidence-strict-validator-orchestration-validation.json"
$markdownPath = Join-Path $OutputRoot "real-owner-evidence-strict-validator-orchestration-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Real Owner Evidence StrictValidator Orchestration Validation")
$lines.Add("")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- source records: ``$($validation.sourceRecordCount)``")
$lines.Add("- field readiness rows: ``$($validation.fieldReadinessCount)``")
$lines.Add("- failed blockers: ``$($validation.failedBlockerCount)``")
$lines.Add("- failed action-required: ``$($validation.failedActionRequiredCount)``")
$lines.Add("")
$lines.Add("| id | passed | severity | detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $items) {
  $detail = ([string]$item.detail).Replace("|", "\|")
  $lines.Add("| $($item.id) | $($item.passed) | $($item.severity) | $detail |")
}
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "Real Owner evidence StrictValidator orchestration validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState Fields=$($validation.fieldReadinessCount) Sources=$($validation.sourceRecordCount) FailedBlockers=$($validation.failedBlockerCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Real Owner evidence StrictValidator orchestration validation failed with $($failedBlockers.Count) blocker(s)."
}
