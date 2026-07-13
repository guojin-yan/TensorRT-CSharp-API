[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-issue-close-owner-decision-input.template.json",
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

function Read-JsonOrNull {
  param([AllowNull()][object]$Path)
  $pathText = [string]$Path
  if (Test-IsPlaceholder -Value $pathText) { return $null }
  $resolved = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-FileHashMatches {
  param([AllowNull()][object]$Path, [AllowNull()][object]$Sha256)
  $pathText = [string]$Path
  $shaText = [string]$Sha256
  if ((Test-IsPlaceholder -Value $pathText) -or -not (Test-Sha256Format -Value $shaText)) { return $false }
  $resolved = Resolve-RepositoryPath -Path $pathText
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $false }
  $actual = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash
  return $actual.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
}

function Test-PublicHttps {
  param([AllowNull()][object]$Value)
  $text = ([string]$Value).Trim()
  return -not (Test-IsPlaceholder -Value $text) -and $text.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)
}

function Test-ForbiddenSubstituteText {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  foreach ($forbidden in @("file://", "local feed", "local-feed", "direct .nupkg", "ProjectReference", "package-managed-dry-run", "github-actions-runs", "dashboard-only", "artifact-only", "queued workflow", "missing runner", "sidecar-only", "local test")) {
    if ($text.IndexOf($forbidden, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
  }

  return $false
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Add-Item {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  $script:items.Add((New-ValidationItem -Id $Id -Passed $Passed -Severity $Severity -Detail $Detail)) | Out-Null
}

function Compare-StringSnapshot {
  param([string]$Id, [AllowNull()][object]$Expected, [AllowNull()][object]$Actual, [string]$Detail)
  $expectedText = [string]$Expected
  $actualText = [string]$Actual
  Add-Item -Id $Id -Passed ($expectedText.Equals($actualText, [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail $Detail
}

function Compare-IntSnapshot {
  param([string]$Id, [int]$Expected, [AllowNull()][object]$Actual, [string]$Detail)
  $actualInt = [int]$Actual
  Add-Item -Id $Id -Passed ($actualInt -eq $Expected) -Severity "blocker" -Detail $Detail
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release issue close owner decision input not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$bridgePath = [string](Get-PropertyOrDefault -Object $record -Name "finalPublicReleaseClosureBridgePath" -DefaultValue "artifacts/final-release/final-public-release-closure-bridge.json")
$bridgeValidationPath = [string](Get-PropertyOrDefault -Object $record -Name "finalPublicReleaseClosureBridgeValidationPath" -DefaultValue "artifacts/final-release/final-public-release-closure-bridge-validation.json")
$postPublishProofResultValidationPath = [string](Get-PropertyOrDefault -Object $record -Name "postPublishProofResultValidationPath" -DefaultValue "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json")
$publicPackageDownloadProofOwnerExecutionPackValidationPath = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofOwnerExecutionPackValidationPath" -DefaultValue "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json")
$postPublishUserVerificationPackValidationPath = [string](Get-PropertyOrDefault -Object $record -Name "postPublishUserVerificationPackValidationPath" -DefaultValue "artifacts/final-release/post-publish-user-verification-pack-validation.json")
$bridge = Read-JsonOrNull -Path $bridgePath
$bridgeValidation = Read-JsonOrNull -Path $bridgeValidationPath
$postPublishProofResultValidation = Read-JsonOrNull -Path $postPublishProofResultValidationPath
$publicPackageDownloadProofOwnerExecutionPackValidation = Read-JsonOrNull -Path $publicPackageDownloadProofOwnerExecutionPackValidationPath
$postPublishUserVerificationPackValidation = Read-JsonOrNull -Path $postPublishUserVerificationPackValidationPath
$bridgeSummary = Get-PropertyOrDefault -Object $bridge -Name "closureProofSourceSummary" -DefaultValue ([pscustomobject]@{})

$state = [string](Get-PropertyOrDefault -Object $record -Name "ownerDecisionInputState" -DefaultValue "")
$finalCloseDecision = [string](Get-PropertyOrDefault -Object $record -Name "finalCloseDecision" -DefaultValue "")
$approvedClose = $finalCloseDecision.Equals("approved-close-release-issue", [StringComparison]::OrdinalIgnoreCase)
$allowedDecision = (Test-IsPlaceholder -Value $finalCloseDecision) -or @("approved-close-release-issue", "blocked-keep-release-issue-open") -contains $finalCloseDecision

Add-Item -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-issue-close-owner-decision-input") -Severity "blocker" -Detail "recordKind must be release-issue-close-owner-decision-input."
Add-Item -Id "state-valid" -Passed (@("blocked-release-issue-close-owner-decision-input-required", "owner-filled-release-issue-close-owner-decision-input") -contains $state) -Severity "blocker" -Detail "ownerDecisionInputState must be blocked template state or owner-filled input state."
Add-Item -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Owner decision input must not publish, approve, promote proof, or close release issue."
Add-Item -Id "forbidden-substitutes-absent" -Passed (-not (Test-ForbiddenSubstituteText -Value ($record | ConvertTo-Json -Depth 32))) -Severity "blocker" -Detail "Owner close decision input must not contain local feed, direct .nupkg, ProjectReference, dry-run, dashboard-only, artifact-only, queued workflow, missing runner, sidecar-only, or local test substitutes."

foreach ($field in @("ownerName", "ownerEmail", "ownerDecisionTimestampUtc", "releaseIssueUrl", "releaseIssueNumber", "selectedChannel", "approvedPublicPackageProofHash", "approvedPostPublishProofHash", "rollbackPlan", "rollbackOwner", "rollbackTrigger", "knownLimitationsAcknowledgement", "finalCloseDecision")) {
  Add-Item -Id "field-$field" -Passed (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$field must be filled by release owner from real close-ready evidence."
}

Add-Item -Id "release-issue-url-public" -Passed (Test-PublicHttps -Value (Get-PropertyOrDefault -Object $record -Name "releaseIssueUrl" -DefaultValue "")) -Severity "action-required" -Detail "releaseIssueUrl must be a public HTTPS URL."
Add-Item -Id "final-close-decision-allowed" -Passed $allowedDecision -Severity "blocker" -Detail "finalCloseDecision must be approved-close-release-issue or blocked-keep-release-issue-open."
Add-Item -Id "final-close-decision-approved-for-close" -Passed $approvedClose -Severity "action-required" -Detail "Owner must explicitly set finalCloseDecision=approved-close-release-issue before close readiness can become ready."

foreach ($pair in @(
  @("release-evidence-bundle-hash", "releaseEvidenceBundlePath", "releaseEvidenceBundleSha256", "releaseEvidenceBundleSha256 must match releaseEvidenceBundlePath.", "action-required"),
  @("post-publish-validation-hash", "postPublishValidationPath", "postPublishValidationSha256", "postPublishValidationSha256 must match postPublishValidationPath.", "action-required"),
  @("post-publish-proof-result-validation-hash", "postPublishProofResultValidationPath", "postPublishProofResultValidationSha256", "postPublishProofResultValidationSha256 must match postPublishProofResultValidationPath.", "blocker"),
  @("public-package-download-proof-owner-execution-pack-validation-hash", "publicPackageDownloadProofOwnerExecutionPackValidationPath", "publicPackageDownloadProofOwnerExecutionPackValidationSha256", "publicPackageDownloadProofOwnerExecutionPackValidationSha256 must match publicPackageDownloadProofOwnerExecutionPackValidationPath.", "blocker"),
  @("post-publish-user-verification-pack-validation-hash", "postPublishUserVerificationPackValidationPath", "postPublishUserVerificationPackValidationSha256", "postPublishUserVerificationPackValidationSha256 must match postPublishUserVerificationPackValidationPath.", "blocker"),
  @("final-public-release-closure-bridge-hash", "finalPublicReleaseClosureBridgePath", "finalPublicReleaseClosureBridgeSha256", "finalPublicReleaseClosureBridgeSha256 must match finalPublicReleaseClosureBridgePath.", "blocker"),
  @("final-public-release-closure-bridge-validation-hash", "finalPublicReleaseClosureBridgeValidationPath", "finalPublicReleaseClosureBridgeValidationSha256", "finalPublicReleaseClosureBridgeValidationSha256 must match finalPublicReleaseClosureBridgeValidationPath.", "blocker")
)) {
  Add-Item -Id $pair[0] -Passed (Test-FileHashMatches -Path (Get-PropertyOrDefault -Object $record -Name $pair[1] -DefaultValue "") -Sha256 (Get-PropertyOrDefault -Object $record -Name $pair[2] -DefaultValue "")) -Severity $pair[4] -Detail $pair[3]
}

$bridgeValidationState = [string](Get-PropertyOrDefault -Object $bridgeValidation -Name "validationState" -DefaultValue "missing-final-public-release-closure-bridge-validation")
$bridgeReady = $bridgeValidationState -eq "final-public-release-closure-bridge-ready-for-owner-close-review"
$bridgeLaneCount = [int](Get-PropertyOrDefault -Object $bridge -Name "laneCount" -DefaultValue 0)
$bridgeBlockedLaneCount = [int](Get-PropertyOrDefault -Object $bridge -Name "blockedLaneCount" -DefaultValue 0)
$bridgeFailedConsistencyBlockerCount = [int](Get-PropertyOrDefault -Object $bridge -Name "failedConsistencyBlockerCount" -DefaultValue 0)
$bridgeFailedConsistencyActionRequiredCount = [int](Get-PropertyOrDefault -Object $bridge -Name "failedConsistencyActionRequiredCount" -DefaultValue 0)
$publicPackageDownloadProofOwnerExecutionPackValidationState = [string](Get-PropertyOrDefault -Object $publicPackageDownloadProofOwnerExecutionPackValidation -Name "validationState" -DefaultValue "missing-public-package-download-proof-owner-execution-pack-validation")
$postPublishUserVerificationPackValidationState = [string](Get-PropertyOrDefault -Object $postPublishUserVerificationPackValidation -Name "validationState" -DefaultValue "missing-post-publish-user-verification-pack-validation")

Compare-StringSnapshot -Id "final-bridge-validation-state-match" -Expected $bridgeValidationState -Actual (Get-PropertyOrDefault -Object $record -Name "finalPublicReleaseClosureBridgeValidationState" -DefaultValue "") -Detail "Owner decision bridge validation state snapshot must match final-public-release-closure-bridge-validation.json."
Compare-StringSnapshot -Id "public-package-download-owner-execution-validation-state-match" -Expected $publicPackageDownloadProofOwnerExecutionPackValidationState -Actual (Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadProofOwnerExecutionPackValidationState" -DefaultValue "") -Detail "Owner decision public package download owner execution pack validation state snapshot must match its validation artifact."
Compare-StringSnapshot -Id "post-publish-user-verification-validation-state-match" -Expected $postPublishUserVerificationPackValidationState -Actual (Get-PropertyOrDefault -Object $record -Name "postPublishUserVerificationPackValidationState" -DefaultValue "") -Detail "Owner decision post-publish user verification pack validation state snapshot must match its validation artifact."
Compare-StringSnapshot -Id "public-package-download-owner-execution-bridge-state-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "publicPackageDownloadOwnerExecutionPackState" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadOwnerExecutionPackState" -DefaultValue "") -Detail "Owner decision public package download owner execution pack state snapshot must match final bridge source summary."
Compare-StringSnapshot -Id "post-publish-user-verification-bridge-state-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "postPublishUserVerificationPackState" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "postPublishUserVerificationPackState" -DefaultValue "") -Detail "Owner decision post-publish user verification pack state snapshot must match final bridge source summary."
Compare-IntSnapshot -Id "final-bridge-lane-count-match" -Expected $bridgeLaneCount -Actual (Get-PropertyOrDefault -Object $record -Name "closureLaneCount" -DefaultValue -1) -Detail "Owner decision closureLaneCount must match final bridge laneCount."
Compare-IntSnapshot -Id "final-bridge-blocked-lane-count-match" -Expected $bridgeBlockedLaneCount -Actual (Get-PropertyOrDefault -Object $record -Name "closureBlockedLaneCount" -DefaultValue -1) -Detail "Owner decision closureBlockedLaneCount must match final bridge blockedLaneCount."
Compare-IntSnapshot -Id "final-bridge-consistency-blocker-count-match" -Expected $bridgeFailedConsistencyBlockerCount -Actual (Get-PropertyOrDefault -Object $record -Name "closureFailedConsistencyBlockerCount" -DefaultValue -1) -Detail "Owner decision closureFailedConsistencyBlockerCount must match final bridge."
Compare-IntSnapshot -Id "final-bridge-consistency-action-required-count-match" -Expected $bridgeFailedConsistencyActionRequiredCount -Actual (Get-PropertyOrDefault -Object $record -Name "closureFailedConsistencyActionRequiredCount" -DefaultValue -1) -Detail "Owner decision closureFailedConsistencyActionRequiredCount must match final bridge."

$postPublishProofCandidateReady = [bool](Get-PropertyOrDefault -Object $postPublishProofResultValidation -Name "proofCandidateReady" -DefaultValue $false)
$postPublishSourceLinkageReady = [bool](Get-PropertyOrDefault -Object $postPublishProofResultValidation -Name "sourceProofLinkageReady" -DefaultValue $false)
Add-Item -Id "final-bridge-ready-before-close" -Passed $bridgeReady -Severity "action-required" -Detail "Final public release closure bridge must be ready before owner close decision can become ready."
Add-Item -Id "approved-close-requires-ready-final-bridge" -Passed (-not $approvedClose -or $bridgeReady) -Severity "blocker" -Detail "Owner cannot approve release issue close while final public release closure bridge remains blocked."
Add-Item -Id "post-publish-proof-candidate-ready" -Passed $postPublishProofCandidateReady -Severity "action-required" -Detail "Post-publish clean consumer proof result must have proofCandidateReady=true."
Add-Item -Id "post-publish-source-proof-linkage-ready" -Passed $postPublishSourceLinkageReady -Severity "action-required" -Detail "Post-publish clean consumer proof result must link ready GitHub Actions, Owner public publish, and public download proofs."
Add-Item -Id "approved-close-requires-post-publish-source-linkage" -Passed (-not $approvedClose -or $postPublishSourceLinkageReady) -Severity "blocker" -Detail "Owner cannot approve release issue close without post-publish source proof linkage."

Compare-StringSnapshot -Id "github-actions-run-id-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "githubActionsRunId" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "githubActionsRunId" -DefaultValue "") -Detail "Owner decision GitHub Actions run id snapshot must match final bridge source summary."
Compare-StringSnapshot -Id "github-actions-run-url-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "githubActionsRunUrl" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "githubActionsRunUrl" -DefaultValue "") -Detail "Owner decision GitHub Actions run URL snapshot must match final bridge source summary."
Compare-StringSnapshot -Id "github-actions-head-sha-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "githubActionsHeadSha" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "githubActionsHeadSha" -DefaultValue "") -Detail "Owner decision GitHub Actions head SHA snapshot must match final bridge source summary."
Compare-StringSnapshot -Id "public-package-url-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "ownerPublicPackageUrl" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "publicPackageUrl" -DefaultValue "") -Detail "Owner decision public package URL snapshot must match final bridge source summary."
Compare-StringSnapshot -Id "public-package-version-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "ownerPublicPackageVersion" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "publicPackageVersion" -DefaultValue "") -Detail "Owner decision public package version snapshot must match final bridge source summary."
Compare-StringSnapshot -Id "public-package-sha-match" -Expected (Get-PropertyOrDefault -Object $bridgeSummary -Name "ownerPublicPackageSha256" -DefaultValue "") -Actual (Get-PropertyOrDefault -Object $record -Name "publicPackageSha256" -DefaultValue "") -Detail "Owner decision public package SHA256 snapshot must match final bridge source summary."

Add-Item -Id "github-actions-run-url-present" -Passed (Test-PublicHttps -Value (Get-PropertyOrDefault -Object $record -Name "githubActionsRunUrl" -DefaultValue "")) -Severity "action-required" -Detail "githubActionsRunUrl must be public HTTPS evidence."
Add-Item -Id "github-actions-head-sha-format" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "githubActionsHeadSha" -DefaultValue "") -match "^[0-9a-fA-F]{40}$") -Severity "action-required" -Detail "githubActionsHeadSha must be a 40-character commit SHA."
Add-Item -Id "public-package-url-public" -Passed (Test-PublicHttps -Value (Get-PropertyOrDefault -Object $record -Name "publicPackageUrl" -DefaultValue "")) -Severity "action-required" -Detail "publicPackageUrl must be public HTTPS package evidence."
Add-Item -Id "public-package-sha-format" -Passed (Test-Sha256Format -Value (Get-PropertyOrDefault -Object $record -Name "publicPackageSha256" -DefaultValue "")) -Severity "action-required" -Detail "publicPackageSha256 must be a 64-character SHA256."

$strictCommand = [string](Get-PropertyOrDefault -Object $record -Name "strictCloseValidatorCommand" -DefaultValue "")
Add-Item -Id "strict-close-validator-command" -Passed ($strictCommand.Contains("Test-ReleaseIssueCloseRecord.ps1", [StringComparison]::OrdinalIgnoreCase) -and $strictCommand.Contains("-FailOnNotCloseReady", [StringComparison]::Ordinal)) -Severity "blocker" -Detail "strictCloseValidatorCommand must use Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady."

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$ready = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-release-issue-close-owner-decision-input"
}
elseif ($ready) {
  "release-issue-close-owner-decision-input-ready"
}
else {
  "blocked-release-issue-close-owner-decision-input-required"
}

$validation = [pscustomobject]@{
  recordKind = "release-issue-close-owner-decision-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  finalPublicReleaseClosureBridgeValidationState = $bridgeValidationState
  closureLaneCount = $bridgeLaneCount
  closureBlockedLaneCount = $bridgeBlockedLaneCount
  closureFailedConsistencyBlockerCount = $bridgeFailedConsistencyBlockerCount
  closureFailedConsistencyActionRequiredCount = $bridgeFailedConsistencyActionRequiredCount
  publicPackageDownloadProofOwnerExecutionPackValidationState = $publicPackageDownloadProofOwnerExecutionPackValidationState
  postPublishUserVerificationPackValidationState = $postPublishUserVerificationPackValidationState
  publicPackageDownloadOwnerExecutionPackState = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageDownloadOwnerExecutionPackState" -DefaultValue "")
  postPublishUserVerificationPackState = [string](Get-PropertyOrDefault -Object $record -Name "postPublishUserVerificationPackState" -DefaultValue "")
  postPublishProofCandidateReady = $postPublishProofCandidateReady
  postPublishProofSourceLinkageReady = $postPublishSourceLinkageReady
  publicPackageUrl = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageUrl" -DefaultValue "")
  publicPackageVersion = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageVersion" -DefaultValue "")
  publicPackageSha256 = [string](Get-PropertyOrDefault -Object $record -Name "publicPackageSha256" -DefaultValue "")
  githubActionsRunId = [string](Get-PropertyOrDefault -Object $record -Name "githubActionsRunId" -DefaultValue "")
  githubActionsRunUrl = [string](Get-PropertyOrDefault -Object $record -Name "githubActionsRunUrl" -DefaultValue "")
  githubActionsHeadSha = [string](Get-PropertyOrDefault -Object $record -Name "githubActionsHeadSha" -DefaultValue "")
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Validation checks owner close decision input shape and final bridge consistency only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-issue-close-owner-decision-input-validation.json"
$markdownPath = Join-Path $OutputRoot "release-issue-close-owner-decision-input-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Release Issue Close Owner Decision Input Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| finalPublicReleaseClosureBridgeValidationState | ``$($validation.finalPublicReleaseClosureBridgeValidationState)`` |
| closureLaneCount | ``$($validation.closureLaneCount)`` |
| closureBlockedLaneCount | ``$($validation.closureBlockedLaneCount)`` |
| publicPackageDownloadProofOwnerExecutionPackValidationState | ``$($validation.publicPackageDownloadProofOwnerExecutionPackValidationState)`` |
| postPublishUserVerificationPackValidationState | ``$($validation.postPublishUserVerificationPackValidationState)`` |
| publicPackageDownloadOwnerExecutionPackState | ``$($validation.publicPackageDownloadOwnerExecutionPackState)`` |
| postPublishUserVerificationPackState | ``$($validation.postPublishUserVerificationPackState)`` |
| postPublishProofSourceLinkageReady | ``$($validation.postPublishProofSourceLinkageReady)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Release issue close owner decision input validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Release issue close owner decision input validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
