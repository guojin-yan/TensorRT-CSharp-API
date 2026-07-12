[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$requiredOwnerInputFields = @(
  "releaseIssue.id",
  "releaseIssue.url",
  "ownerDecision.ownerName",
  "ownerDecision.ownerIdentityProvider",
  "ownerDecision.approvalTimestampUtc",
  "ownerDecision.finalCloseDecision",
  "ownerDecision.closeDecisionReason",
  "ownerDecision.noRealProofNoCloseAcknowledged",
  "selectedChannel.name",
  "selectedChannel.sourceUri",
  "selectedChannel.packageVersion",
  "packages.managed.packageId",
  "packages.managed.packageVersion",
  "packages.managed.packageUrl",
  "packages.managed.nupkgSha256",
  "packages.runtime.packageId",
  "packages.runtime.packageVersion",
  "packages.runtime.packageUrl",
  "packages.runtime.nupkgSha256",
  "postPublishVerification.validationPath",
  "postPublishVerification.validationState",
  "postPublishVerification.isPostPublishVerificationProof",
  "postPublishVerification.canCloseReleaseIssue",
  "releaseClosePreflight.path",
  "releaseClosePreflight.preflightState",
  "releaseClosePreflight.canCloseReleaseIssue",
  "staleReleaseClaimsAudit.path",
  "staleReleaseClaimsAudit.findingCount",
  "releaseOwnerProofInput.validationPath",
  "releaseOwnerProofInput.validationState",
  "releaseOwnerProofInput.canPromoteOwnerProofInput",
  "releaseEvidenceBundle.path",
  "releaseEvidenceBundle.sha256",
  "releaseEvidenceBundle.bundleState",
  "rollbackPlan.summary",
  "rollbackPlan.packageYankOrDeprecatePlan",
  "rollbackPlan.ownerContact",
  "rollbackPlan.customerNotificationPlan",
  "closureChecklist.ownerProofInputPromoted",
  "closureChecklist.postPublishVerificationPassed",
  "closureChecklist.releaseClosePreflightPassed",
  "closureChecklist.staleClaimsAuditClean",
  "closureChecklist.evidenceBundleHashVerified",
  "closureChecklist.rollbackPlanReady"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-issue-close-record-template"
  recordState = "template-only"
  proofClassification = "template-only"
  closeReadinessState = "blocked-real-proof-required"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteReleaseIssueCloseRecord = $false
  requiredOwnerInputFieldCount = $requiredOwnerInputFields.Count
  requiredOwnerInputFields = $requiredOwnerInputFields
  releaseIssue = [ordered]@{
    id = ""
    url = ""
  }
  ownerDecision = [ordered]@{
    ownerName = ""
    ownerIdentityProvider = ""
    approvalTimestampUtc = ""
    finalCloseDecision = "do-not-close-template"
    closeDecisionReason = ""
    noRealProofNoCloseAcknowledged = $false
  }
  selectedChannel = [ordered]@{
    name = ""
    sourceUri = ""
    packageVersion = ""
  }
  packages = [ordered]@{
    managed = [ordered]@{
      packageId = "JYPPX.TensorRtSharp"
      packageVersion = ""
      packageUrl = ""
      nupkgSha256 = ""
    }
    runtime = [ordered]@{
      packageId = ""
      packageVersion = ""
      packageUrl = ""
      nupkgSha256 = ""
    }
  }
  postPublishVerification = [ordered]@{
    validationPath = "artifacts/final-release/post-publish-verification-validation.json"
    validationState = "template-only"
    isPostPublishVerificationProof = $false
    canCloseReleaseIssue = $false
  }
  releaseClosePreflight = [ordered]@{
    path = "artifacts/final-release/release-close-preflight.json"
    preflightState = "blocked-real-proof-required"
    canCloseReleaseIssue = $false
  }
  staleReleaseClaimsAudit = [ordered]@{
    path = "artifacts/final-release/stale-release-claims-audit.json"
    findingCount = -1
  }
  releaseOwnerProofInput = [ordered]@{
    validationPath = "artifacts/final-release/release-owner-proof-input-record-validation.json"
    validationState = "blocked-template-only"
    canPromoteOwnerProofInput = $false
  }
  releaseEvidenceBundle = [ordered]@{
    path = "artifacts/final-release/release-evidence-bundle.json"
    sha256 = ""
    bundleState = "blocked-evidence-incomplete"
  }
  rollbackPlan = [ordered]@{
    summary = ""
    packageYankOrDeprecatePlan = ""
    ownerContact = ""
    customerNotificationPlan = ""
  }
  closureChecklist = [ordered]@{
    ownerProofInputPromoted = $false
    postPublishVerificationPassed = $false
    releaseClosePreflightPassed = $false
    staleClaimsAuditClean = $false
    evidenceBundleHashVerified = $false
    rollbackPlanReady = $false
  }
  ownerAcknowledgements = @(
    "This template is not release issue close proof.",
    "No release issue can close without real post-publish verification proof.",
    "No release issue can close while release-close-preflight.json is blocked.",
    "No release issue can close while stale release claims remain.",
    "No release issue can close from readiness snapshots, schema-only records, dry-runs, prechecks, or managed-readiness."
  )
  nonSubstituteProofKinds = @(
    "template",
    "draft",
    "schema-only",
    "dry-run-only",
    "precheck-only",
    "managed-readiness",
    "readiness snapshot",
    "collection package",
    "owner handoff",
    "local feed",
    "ProjectReference",
    "direct .nupkg reference",
    "dependency-probe-only",
    "bridge-only package consumer log",
    "CallbackAllocatorReadinessSnapshot",
    "blocked-by-cuda-driver",
    "missing package URL",
    "missing package SHA256",
    "missing post-publish proof",
    "missing owner final close decision",
    "mismatched evidence bundle SHA256"
  )
  boundary = "This template collects final owner close inputs only. It does not publish packages, upload artifacts, authorize publication, or close a release issue."
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$jsonPath = Join-Path $OutputRoot "release-issue-close-record-template.json"
$markdownPath = Join-Path $OutputRoot "release-issue-close-record-template.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$fieldLines = $requiredOwnerInputFields | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $record.nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$ackLines = $record.ownerAcknowledgements | ForEach-Object { "- $_" }

$markdown = @"
# Release Issue Close Record Template

- record state: ``template-only``
- proof classification: ``template-only``
- close readiness state: ``blocked-real-proof-required``
- performs publish: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- required owner input field count: ``$($requiredOwnerInputFields.Count)``

## Required Owner Input Fields

$($fieldLines -join "`r`n")

## Owner Acknowledgements

$($ackLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release issue close record template written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "RecordState=template-only"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
