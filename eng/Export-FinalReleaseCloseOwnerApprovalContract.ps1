[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-FlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $lines = @(ConvertTo-FlatStringLines -Value $InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

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
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ApprovalLane {
  param([string]$Id, [string]$Description)

  [pscustomobject]@{
    id = $Id
    description = $Description
    approvalState = "blocked-final-release-close-owner-approval-required"
    ownerReviewer = "<owner-fill-final-release-close-owner-reviewer>"
    ownerReviewTimestampUtc = "<owner-fill-final-release-close-owner-review-timestamp-utc>"
    ownerApprovalDecision = "<owner-fill-approved-or-rejected>"
    ownerApprovalRationale = "<owner-fill-final-release-close-rationale>"
    releaseIssueUrl = "<owner-fill-release-issue-url>"
    releaseIssueCloseDecision = "<owner-fill-close-or-keep-open>"
    rollbackDecision = "<owner-fill-rollback-or-no-rollback>"
    rollbackRationale = "<owner-fill-rollback-rationale>"
    releaseNotesPath = "<owner-fill-release-notes-path>"
    releaseNotesSha256 = "<owner-fill-release-notes-sha256>"
    finalPublicPackageUrl = "<owner-fill-final-public-package-url>"
    finalPublicPackageUrlReviewDecision = "<owner-fill-final-public-package-url-approved-or-rejected>"
    finalPublicPackageSha256 = "<owner-fill-final-public-package-sha256>"
    finalPackageIdentity = [pscustomobject]@{
      packageId = "<owner-fill-final-package-id>"
      packageVersion = "<owner-fill-final-package-version>"
      packageSource = "<owner-fill-final-package-source>"
      publicPackageUrl = "<owner-fill-final-public-package-url>"
      nupkgSha256 = "<owner-fill-final-public-package-sha256>"
    }
    postPublishCleanConsumerProofCandidateId = "<owner-fill-post-publish-clean-consumer-proof-candidate-id>"
    postPublishCleanConsumerProofCandidateValidationPath = "<owner-fill-post-publish-clean-consumer-proof-candidate-validation-path>"
    postPublishCleanConsumerProofCandidateValidationSha256 = "<owner-fill-post-publish-clean-consumer-proof-candidate-validation-sha256>"
    classificationAuditPath = "<owner-fill-release-evidence-classification-audit-path>"
    classificationAuditSha256 = "<owner-fill-release-evidence-classification-audit-sha256>"
    releaseEvidenceBundlePath = "<owner-fill-release-evidence-bundle-path>"
    releaseEvidenceBundleSha256 = "<owner-fill-release-evidence-bundle-sha256>"
    nonSubstituteConfirmations = [pscustomobject]@{
      confirmsNoDraftAsProof = $false
      confirmsNoCandidateAsReleaseCloseProof = $false
      confirmsNoDashboardAsProof = $false
      confirmsNoRunbookAsProof = $false
      confirmsNoDryRunAsProof = $false
      confirmsNoLocalFeedAsPublicPackageProof = $false
      confirmsNoDirectNupkgAsPublicPackageProof = $false
      confirmsNoProjectReferenceAsCleanConsumerProof = $false
      confirmsNoSourceCheckoutReferenceAsCleanConsumerProof = $false
      confirmsNoBlockedByDriverAsRuntimeProof = $false
    }
    forbiddenSubstituteMarkers = @(
      "local feed",
      "ProjectReference",
      "direct .nupkg",
      "source checkout reference",
      "template",
      "draft",
      "dry-run",
      "dashboard",
      "candidate",
      "build-only",
      "dependency-probe-only",
      "blocked-by-driver"
    )
    readyForPreflight = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Final release close Owner approval contract lane only. It is not runtime proof, not post-publish proof, not publish approval, not package push, and cannot close the release."
  }
}

$postPublishCandidate = Read-JsonOrNull "artifacts\final-release\final-post-publish-clean-consumer-proof-candidate.json"
$postPublishCandidateValidation = Read-JsonOrNull "artifacts\final-release\final-post-publish-clean-consumer-proof-candidate-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$remoteProofBackfillGate = Read-JsonOrNull "artifacts\final-release\remote-ci-and-public-publish-proof-backfill-gate.json"
$remoteProofBackfillGateValidation = Read-JsonOrNull "artifacts\final-release\remote-ci-and-public-publish-proof-backfill-gate-validation.json"

$approvalLanes = @(
  New-ApprovalLane -Id "owner-final-release-close-decision" -Description "Owner supplies the final decision to close or keep open the release issue."
  New-ApprovalLane -Id "owner-rollback-decision" -Description "Owner supplies rollback or no-rollback decision and rationale."
  New-ApprovalLane -Id "owner-release-notes-approval" -Description "Owner approves release notes content and hash."
  New-ApprovalLane -Id "owner-final-public-package-url-and-hash-approval" -Description "Owner approves final public package URL, identity, and hash."
)

$record = [ordered]@{
  recordKind = "final-release-close-owner-approval-contract"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  contractState = "blocked-final-release-close-owner-approval-required"
  sourcePostPublishCleanConsumerProofCandidateState = [string](Get-PropertyOrDefault -Object $postPublishCandidate -Name "candidateState" -DefaultValue "missing-final-post-publish-clean-consumer-proof-candidate")
  sourcePostPublishCleanConsumerProofCandidateValidationState = [string](Get-PropertyOrDefault -Object $postPublishCandidateValidation -Name "validationState" -DefaultValue "missing-final-post-publish-clean-consumer-proof-candidate-validation")
  sourceReleaseEvidenceBundleRecordKind = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "recordKind" -DefaultValue "missing-release-evidence-bundle")
  sourceClassificationAuditState = [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")
  sourceRemoteCiAndPublicPublishProofBackfillGateState = [string](Get-PropertyOrDefault -Object $remoteProofBackfillGate -Name "gateState" -DefaultValue "missing-remote-ci-and-public-publish-proof-backfill-gate")
  sourceRemoteCiAndPublicPublishProofBackfillGateValidationState = [string](Get-PropertyOrDefault -Object $remoteProofBackfillGateValidation -Name "validationState" -DefaultValue "missing-remote-ci-and-public-publish-proof-backfill-gate-validation")
  requiredRemoteProofLaneIds = @("github-actions-run-proof", "owner-public-publish-result", "public-package-download-proof", "post-publish-clean-consumer-proof")
  approvalLaneCount = $approvalLanes.Count
  blockedApprovalLaneCount = $approvalLanes.Count
  readyForPreflightCount = 0
  requiredOwnerInputFieldCount = 30 * $approvalLanes.Count
  approvalLanes = @($approvalLanes)
  requiredOwnerInputFields = @(
    "ownerReviewer",
    "ownerReviewTimestampUtc",
    "ownerApprovalDecision",
    "ownerApprovalRationale",
    "releaseIssueUrl",
    "releaseIssueCloseDecision",
    "rollbackDecision",
    "rollbackRationale",
    "releaseNotesPath",
    "releaseNotesSha256",
    "finalPublicPackageUrl",
    "finalPublicPackageUrlReviewDecision",
    "finalPublicPackageSha256",
    "finalPackageIdentity",
    "postPublishCleanConsumerProofCandidateId",
    "postPublishCleanConsumerProofCandidateValidationPath",
    "postPublishCleanConsumerProofCandidateValidationSha256",
    "classificationAuditPath",
    "classificationAuditSha256",
    "releaseEvidenceBundlePath",
    "releaseEvidenceBundleSha256",
    "githubActionsRunProofPath",
    "githubActionsRunProofSha256",
    "ownerPublicPublishResultPath",
    "ownerPublicPublishResultSha256",
    "publicPackageDownloadProofPath",
    "publicPackageDownloadProofSha256",
    "postPublishCleanConsumerProofResultPath",
    "postPublishCleanConsumerProofResultSha256",
    "nonSubstituteConfirmations"
  )
  forbiddenSubstituteMarkers = @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "source checkout reference",
    "template",
    "draft",
    "dry-run",
    "dashboard",
    "candidate",
    "build-only",
    "dependency-probe-only",
    "blocked-by-driver"
  )
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.md",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.md",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate.json",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-evidence-classification-audit.json"
  )
  boundary = "Final release close Owner approval contract only. It waits for real Owner decisions, release notes hash, final public package URL/hash review, and rollback decision; it is not runtime proof, not post-publish proof, not publish approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-release-close-owner-approval-contract.json"
$markdownPath = Join-Path $OutputRoot "final-release-close-owner-approval-contract.md"
$record | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($lane in $approvalLanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | ``$(ConvertTo-MarkdownCell $lane.approvalState)`` | ``$($lane.readyForPreflight)`` |"
}

$markdown = @(
  "# Final Release Close Owner Approval Contract",
  "",
  "- contractState: ``$($record.contractState)``",
  "- approvalLaneCount: ``$($record.approvalLaneCount)``",
  "- blockedApprovalLaneCount: ``$($record.blockedApprovalLaneCount)``",
  "- readyForPreflightCount: ``0``",
  "- requiredOwnerInputFieldCount: ``$($record.requiredOwnerInputFieldCount)``",
  "- boundary: $($record.boundary)",
  "",
  "| Approval Lane | State | Ready For Preflight |",
  "|---|---|---|",
  @($rows)
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
