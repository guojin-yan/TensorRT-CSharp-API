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

function New-OwnerAction {
  param(
    [int]$Sequence,
    [string]$Id,
    [string]$Category,
    [string]$CurrentState,
    [string]$RequiredRealEvidence,
    [string]$FirstSafeCommand,
    [string]$Validator,
    [string]$CannotSubstitute
  )

  [pscustomobject]@{
    sequence = $Sequence
    id = $Id
    category = $Category
    currentState = $CurrentState
    requiredRealEvidence = $RequiredRealEvidence
    firstSafeCommand = $FirstSafeCommand
    validator = $Validator
    cannotSubstitute = $CannotSubstitute
    ready = $false
    ownerActionRequired = $true
    boundary = "Release candidate owner action roadmap item only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$actions = @(
  New-OwnerAction -Sequence 1 -Id "owner-confirms-public-channel" -Category "publish" -CurrentState "blocked-owner-public-channel-required" -RequiredRealEvidence "Owner selects real public package channel, package id, version, and release issue." -FirstSafeCommand "Review release evidence bundle and owner action roadmap." -Validator "eng\Test-ReleaseCandidateOwnerActionRoadmap.ps1 -Strict" -CannotSubstitute "local feed, ProjectReference, direct nupkg, template, draft"
  New-OwnerAction -Sequence 2 -Id "public-publish-real-result" -Category "publish" -CurrentState "blocked-public-publish-real-result-record-required" -RequiredRealEvidence "Real public publish result, package URL, package version, timestamp, transcript, and reviewer." -FirstSafeCommand "Owner executes public publish manually outside automation after approvals." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict" -CannotSubstitute "dry-run, runbook, local nupkg, local feed"
  New-OwnerAction -Sequence 3 -Id "public-package-download-hash" -Category "publish" -CurrentState "blocked-public-package-sha256-required" -RequiredRealEvidence "Downloaded public package SHA256 from the public channel." -FirstSafeCommand "Download package from public source and hash the downloaded artifact." -Validator "eng\Test-FinalReleaseCloseRecordRealValidator.ps1 -Strict" -CannotSubstitute "pre-publish package, locally packed artifact, hash-only candidate"
  New-OwnerAction -Sequence 4 -Id "clean-consumer-smoke" -Category "consumer" -CurrentState "blocked-post-publish-clean-consumer-proof-record-required" -RequiredRealEvidence "Repository-external clean consumer restore/build/smoke logs, hashes, and environment metadata." -FirstSafeCommand "Create a clean external consumer and install from public package source." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict" -CannotSubstitute "repository sample, ProjectReference, local feed, build-only log"
  New-OwnerAction -Sequence 5 -Id "compatible-host-runtime-proof" -Category "runtime" -CurrentState "blocked-compatible-host-runtime-proof-required" -RequiredRealEvidence "Compatible CUDA/TensorRT host runtime proof with logs and hashes." -FirstSafeCommand "Run package-consumer runtime proof on compatible host." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1" -CannotSubstitute "blocked-by-driver precheck, local-only scan, sidecar-only report"
  New-OwnerAction -Sequence 6 -Id "linux-runner-proof" -Category "runtime" -CurrentState "blocked-linux-runner-proof-required" -RequiredRealEvidence "Linux runner proof for package consumer and native assets." -FirstSafeCommand "Run Linux runner proof commands from owner execution package." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1" -CannotSubstitute "Windows handoff, dry-run, template"
  New-OwnerAction -Sequence 7 -Id "real-model-runtime-proof" -Category "runtime" -CurrentState "blocked-real-model-runtime-proof-required" -RequiredRealEvidence "Real model assets, inference logs, output hashes, and validation metadata." -FirstSafeCommand "Run representative real model runtime smoke with recorded assets." -Validator "eng\Test-ReleaseRuntimeProofExecutionMatrix.ps1" -CannotSubstitute "toy model, parser-only output, build-only engine"
  New-OwnerAction -Sequence 8 -Id "post-publish-verification" -Category "post-publish" -CurrentState "blocked-post-publish-verification-required" -RequiredRealEvidence "Post-publish verification record from public package channel." -FirstSafeCommand "Run post-publish verification after public package is available." -Validator "eng\Test-PostPublishVerification.ps1" -CannotSubstitute "pre-publish validation, local package review, dashboard"
  New-OwnerAction -Sequence 9 -Id "forbidden-substitute-final-check" -Category "non-substitute" -CurrentState "blocked-forbidden-substitute-final-scan-required" -RequiredRealEvidence "Scan confirms no substitute evidence is promoted as proof." -FirstSafeCommand "Run release candidate non-substitute final scan." -Validator "eng\Test-ReleaseCandidateNonSubstituteFinalScan.ps1 -Strict" -CannotSubstitute "manual assertion without scan evidence"
  New-OwnerAction -Sequence 10 -Id "rollback-review" -Category "release-close" -CurrentState "blocked-rollback-review-required" -RequiredRealEvidence "Owner-reviewed rollback plan tied to public package version." -FirstSafeCommand "Review rollback section of final close decision inputs." -Validator "eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict" -CannotSubstitute "generic rollback note, unchecked checklist"
  New-OwnerAction -Sequence 11 -Id "final-close-record" -Category "release-close" -CurrentState "blocked-final-release-close-record-real-proof-required" -RequiredRealEvidence "All final close record fields, hashes, owner reviewer, and validator outputs." -FirstSafeCommand "Regenerate final close record real validator and projection after owner proof is filled." -Validator "eng\Test-FinalReleaseCloseRecordRealValidator.ps1 -Strict" -CannotSubstitute "template close record, candidate, local-only hash lane"
  New-OwnerAction -Sequence 12 -Id "release-issue-close-decision" -Category "release-close" -CurrentState "blocked-release-issue-close-record-required" -RequiredRealEvidence "Strict release issue close validation passes after all real proof and owner approval." -FirstSafeCommand "Run strict release issue close record validation." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" -CannotSubstitute "owner roadmap, checklist, approval boundary audit"
)

$record = [pscustomobject]@{
  recordKind = "release-candidate-owner-action-roadmap"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  roadmapState = "blocked-release-candidate-owner-action-roadmap-owner-proof-required"
  ownerActionCount = $actions.Count
  blockedOwnerActionCount = @($actions | Where-Object { -not [bool]$_.ready }).Count
  ownerActions = @($actions)
  sourceArtifacts = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-candidate-final-publishability-audit.json",
    "artifacts/final-release/final-release-close-record-real-validator-validation.json",
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
  boundary = "Release candidate owner action roadmap is blocked owner guidance only; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-candidate-owner-action-roadmap.json"
$markdownPath = Join-Path $OutputRoot "release-candidate-owner-action-roadmap.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $actions | ForEach-Object {
  "| $($_.sequence) | ``$($_.id)`` | ``$($_.currentState)`` | $($_.requiredRealEvidence.Replace("|", "\|")) | ``$($_.validator)`` | $($_.cannotSubstitute.Replace("|", "\|")) |"
}

$markdown = @"
# Release Candidate Owner Action Roadmap

| Field | Value |
| --- | --- |
| roadmapState | ``$($record.roadmapState)`` |
| ownerActionCount | ``$($record.ownerActionCount)`` |
| blockedOwnerActionCount | ``$($record.blockedOwnerActionCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Owner Actions

| # | ID | Current State | Required Real Evidence | Validator | Cannot Substitute |
| ---: | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate owner action roadmap written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "RoadmapState=$($record.roadmapState) Actions=$($record.ownerActionCount) Blocked=$($record.blockedOwnerActionCount)"
