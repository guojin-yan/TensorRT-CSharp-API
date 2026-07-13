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

function New-ChecklistItem {
  param([int]$Order, [string]$Id, [string]$CurrentState, [string]$RequiredRealEvidence, [string]$FirstSafeCommand, [string]$Validator, [string]$CannotSubstitute)
  [pscustomobject]@{
    order = $Order
    id = $Id
    currentState = $CurrentState
    requiredRealEvidence = $RequiredRealEvidence
    firstSafeCommand = $FirstSafeCommand
    validator = $Validator
    cannotSubstitute = $CannotSubstitute
    checked = $false
    ready = $false
    boundary = "Release candidate final owner checklist item only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$items = @(
  New-ChecklistItem -Order 1 -Id "confirm-version-and-public-channel" -CurrentState "blocked-owner-input-required" -RequiredRealEvidence "Final package id, version, public channel, and release issue id." -FirstSafeCommand "Review release-candidate-final-publishability-audit.md" -Validator "eng\Test-ReleaseCandidateFinalOwnerChecklist.ps1 -Strict" -CannotSubstitute "draft release notes"
  New-ChecklistItem -Order 2 -Id "execute-public-publish-manually" -CurrentState "blocked-public-publish-owner-action-required" -RequiredRealEvidence "Manual Owner public publish transcript and package URL." -FirstSafeCommand "Owner manually executes approved public publish command." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict" -CannotSubstitute "dry-run or local feed"
  New-ChecklistItem -Order 3 -Id "download-and-hash-public-package" -CurrentState "blocked-public-package-hash-required" -RequiredRealEvidence "SHA256 from downloaded public package artifact." -FirstSafeCommand "Download from public channel and compute SHA256." -Validator "eng\Test-FinalReleaseCloseRecordRealValidator.ps1 -Strict" -CannotSubstitute "locally packed nupkg hash"
  New-ChecklistItem -Order 4 -Id "run-clean-consumer-smoke" -CurrentState "blocked-clean-consumer-proof-required" -RequiredRealEvidence "External consumer restore/build/smoke log and hashes." -FirstSafeCommand "Create external clean consumer project and install public package." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict" -CannotSubstitute "repository sample or ProjectReference"
  New-ChecklistItem -Order 5 -Id "run-compatible-runtime-proof" -CurrentState "blocked-runtime-proof-required" -RequiredRealEvidence "Compatible CUDA/TensorRT host runtime proof." -FirstSafeCommand "Run runtime proof commands on compatible host." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1" -CannotSubstitute "blocked-by-driver precheck"
  New-ChecklistItem -Order 6 -Id "run-linux-proof" -CurrentState "blocked-linux-proof-required" -RequiredRealEvidence "Linux runner logs, hashes, package source, and environment metadata." -FirstSafeCommand "Run Linux proof package from owner execution pack." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1" -CannotSubstitute "Windows handoff"
  New-ChecklistItem -Order 7 -Id "run-real-model-proof" -CurrentState "blocked-real-model-proof-required" -RequiredRealEvidence "Real model assets, input/output hashes, runtime logs, and validation metadata." -FirstSafeCommand "Run representative real model scenario." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1" -CannotSubstitute "toy model or parser-only output"
  New-ChecklistItem -Order 8 -Id "run-post-publish-verification" -CurrentState "blocked-post-publish-proof-required" -RequiredRealEvidence "Post-publish verification from public channel." -FirstSafeCommand "Run post-publish verification after package is public." -Validator "eng\Test-PostPublishVerification.ps1" -CannotSubstitute "pre-publish package review"
  New-ChecklistItem -Order 9 -Id "clear-non-substitutes" -CurrentState "blocked-non-substitute-scan-required" -RequiredRealEvidence "Final non-substitute scan remains clean and blocked substitutes are not proof." -FirstSafeCommand "Run release candidate non-substitute final scan." -Validator "eng\Test-ReleaseCandidateNonSubstituteFinalScan.ps1 -Strict" -CannotSubstitute "manual statement only"
  New-ChecklistItem -Order 10 -Id "final-owner-approval-and-close" -CurrentState "blocked-release-close-owner-approval-required" -RequiredRealEvidence "Rollback review, final owner decision, strict close record validation." -FirstSafeCommand "Run final close owner approval boundary audit and release issue close record validator." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" -CannotSubstitute "checklist, roadmap, dashboard"
)

$record = [pscustomobject]@{
  recordKind = "release-candidate-final-owner-checklist"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  checklistState = "blocked-release-candidate-final-owner-checklist-owner-proof-required"
  checklistItemCount = $items.Count
  blockedChecklistItemCount = @($items | Where-Object { -not [bool]$_.ready }).Count
  checklistItems = @($items)
  sourceArtifacts = @(
    "artifacts/final-release/release-candidate-final-publishability-audit.json",
    "artifacts/final-release/release-candidate-owner-action-roadmap.json",
    "artifacts/final-release/release-candidate-non-substitute-final-scan.json",
    "artifacts/final-release/final-close-owner-approval-boundary-audit-validation.json"
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
  boundary = "Release candidate final owner checklist is blocked owner guidance only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-final-owner-checklist.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-final-owner-checklist.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $items | ForEach-Object { "| $($_.order) | ``$($_.id)`` | ``$($_.currentState)`` | $($_.requiredRealEvidence.Replace("|", "\|")) | ``$($_.validator)`` | $($_.cannotSubstitute.Replace("|", "\|")) |" }
$markdown = @"
# Release Candidate Final Owner Checklist

| Field | Value |
| --- | --- |
| checklistState | ``$($record.checklistState)`` |
| checklistItemCount | ``$($record.checklistItemCount)`` |
| blockedChecklistItemCount | ``$($record.blockedChecklistItemCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Checklist

| # | ID | Current State | Required Real Evidence | Validator | Cannot Substitute |
| ---: | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate final owner checklist written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ChecklistState=$($record.checklistState) Items=$($record.checklistItemCount) Blocked=$($record.blockedChecklistItemCount)"
