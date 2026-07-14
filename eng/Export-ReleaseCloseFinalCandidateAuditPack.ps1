[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

function Get-ArtifactHashOrEmpty {
  param([string]$RelativePath)
  $path = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return "" }
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function New-CandidateCheck {
  param(
    [string]$Id,
    [string]$Title,
    [string[]]$SourceArtifacts,
    [string[]]$RequiredRealInputs,
    [string[]]$BlockedReasons
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    checkState = "blocked-real-owner-evidence-required"
    sourceArtifacts = @($SourceArtifacts)
    requiredRealInputs = @($RequiredRealInputs)
    requiredRealInputCount = @($RequiredRealInputs).Count
    blockedReasons = @($BlockedReasons)
    blockedReasonCount = @($BlockedReasons).Count
    passed = $false
    proofReady = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Final close candidate audit check only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

if (-not (Test-Path -LiteralPath (Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path "artifacts/final-release/owner-real-publish-evidence-import-readiness-dashboard-validation.json") -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerRealPublishEvidenceImportReadinessDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  & (Join-Path $RepositoryRoot "eng\Test-OwnerRealPublishEvidenceImportReadinessDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict
}

$bundle = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-evidence-bundle.json"
$classificationAudit = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-evidence-classification-audit.json"
$importDashboardValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/owner-real-publish-evidence-import-readiness-dashboard-validation.json"
$landingPackValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/post-publish-docs-and-samples-final-landing-pack-validation.json"
$ledgerValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/owner-real-publish-evidence-availability-ledger-validation.json"
$publicDownloadValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/public-package-download-proof-input-validation.json"
$postPublishCleanValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json"
$strictCloseValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json"
$closureBridgeValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/final-public-release-closure-bridge-validation.json"
$closeDecisionValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-issue-close-owner-decision-input-validation.json"
$closeRecordValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/release-issue-close-record-validation.json"

$bundleSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/release-evidence-bundle.json"
$classificationAuditSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/release-evidence-classification-audit.json"
$finalLandingPackSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/post-publish-docs-and-samples-final-landing-pack-validation.json"
$importDashboardSha256 = Get-ArtifactHashOrEmpty -RelativePath "artifacts/final-release/owner-real-publish-evidence-import-readiness-dashboard-validation.json"

$checks = @(
  New-CandidateCheck -Id "release-evidence-bundle-hash-lock" -Title "Release evidence bundle hash lock" -SourceArtifacts @("artifacts/final-release/release-evidence-bundle.json", "artifacts/final-release/release-evidence-classification-audit.json") -RequiredRealInputs @("releaseEvidenceBundleSha256", "classificationAuditSha256", "ownerHashReviewDecision") -BlockedReasons @("Evidence bundle hash must be reviewed by Owner.", "Classification audit must remain passed with non-proof boundaries.")
  New-CandidateCheck -Id "owner-import-readiness-no-proof-ready" -Title "Owner import readiness no proof-ready slots" -SourceArtifacts @("artifacts/final-release/owner-real-publish-evidence-import-readiness-dashboard-validation.json", "artifacts/final-release/owner-real-publish-evidence-availability-ledger-validation.json") -RequiredRealInputs @("allRequiredOwnerProofAvailable", "ownerImportDecision", "strictValidatorOutputs") -BlockedReasons @("Owner proof slots are still blocked.", "No real Owner evidence files have been accepted.")
  New-CandidateCheck -Id "final-landing-pack-non-proof" -Title "Final landing pack remains non-proof" -SourceArtifacts @("artifacts/final-release/post-publish-docs-and-samples-final-landing-pack-validation.json") -RequiredRealInputs @("docsPublicUrl", "articlePublishDecision", "publicPackageUrls") -BlockedReasons @("Docs/sample landing material is publication readiness only.", "YoloVision and article assets cannot substitute real model/runtime proof.")
  New-CandidateCheck -Id "public-package-url-hash-proof" -Title "Public package URL and hash proof" -SourceArtifacts @("artifacts/final-release/public-package-download-proof-input-validation.json", "artifacts/final-release/public-package-download-proof-candidate-validation.json") -RequiredRealInputs @("managedPublicDownloadUrl", "managedNupkgSha256", "runtimePublicDownloadUrl", "runtimeNupkgSha256", "downloadLogSha256") -BlockedReasons @("Public package download proof is not proof-ready.", "Package URLs and SHA256 values must come from real public source.")
  New-CandidateCheck -Id "external-clean-consumer-post-publish-proof" -Title "External clean consumer and post-publish proof" -SourceArtifacts @("artifacts/final-release/external-clean-consumer-execution-result-validation.json", "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json") -RequiredRealInputs @("externalWorkspaceRoot", "restoreLogSha256", "buildLogSha256", "smokeLogSha256", "hostMetadata", "noLocalSubstitutes") -BlockedReasons @("Repository-external clean consumer proof is not accepted.", "Post-publish clean consumer proof is not accepted.")
  New-CandidateCheck -Id "strict-close-dashboard-and-bridge" -Title "Strict close dashboard and final bridge" -SourceArtifacts @("artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json", "artifacts/final-release/final-public-release-closure-bridge-validation.json") -RequiredRealInputs @("strictCloseReady", "finalClosureBridgeReady", "allProofLanesAccepted") -BlockedReasons @("Strict close dashboard is still blocked.", "Final public release closure bridge is still blocked.")
  New-CandidateCheck -Id "release-issue-close-owner-decision" -Title "Release Issue close Owner decision" -SourceArtifacts @("artifacts/final-release/release-issue-close-owner-decision-input-validation.json", "artifacts/final-release/release-issue-close-record-validation.json") -RequiredRealInputs @("releaseIssueUrl", "ownerCloseDecision", "postPublishProofSha256", "releaseEvidenceBundleSha256", "rollbackPlanSha256") -BlockedReasons @("Owner close decision is not accepted.", "Release issue close record is not accepted.")
  New-CandidateCheck -Id "rollback-and-known-limitations" -Title "Rollback and known limitations review" -SourceArtifacts @("artifacts/final-release/post-publish-rollback-owner-decision-gate-validation.json", "docs/articles/zh-cn/release-evidence-non-substitute-guide.md") -RequiredRealInputs @("rollbackPlanReviewed", "rollbackPlanSha256", "knownLimitationsUrl", "ownerRollbackDecision") -BlockedReasons @("Rollback owner decision gate is not proof-ready.", "Known limitations need Owner review after publish.")
  New-CandidateCheck -Id "forbidden-substitute-final-scan" -Title "Forbidden substitute final scan" -SourceArtifacts @("artifacts/final-release/release-evidence-classification-audit.json", "artifacts/final-release/public-proof-claim-boundary-audit.json") -RequiredRealInputs @("noLocalFeed", "noProjectReference", "noDirectNupkg", "noDashboardSubstitute", "noDryRunSubstitute") -BlockedReasons @("Classification audit must remain passed.", "Forbidden substitutes remain non-proof even when present as guidance.")
)

$record = [pscustomobject]@{
  recordKind = "release-close-final-candidate-audit-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditState = "blocked-release-close-final-candidate-real-owner-evidence-required"
  sourceStates = [pscustomobject]@{
    releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $bundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
    classificationAuditState = [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")
    ownerImportReadinessValidationState = [string](Get-PropertyOrDefault -Object $importDashboardValidation -Name "validationState" -DefaultValue "missing-owner-real-publish-evidence-import-readiness-dashboard-validation")
    finalLandingPackValidationState = [string](Get-PropertyOrDefault -Object $landingPackValidation -Name "validationState" -DefaultValue "missing-post-publish-docs-and-samples-final-landing-pack-validation")
    ownerLedgerValidationState = [string](Get-PropertyOrDefault -Object $ledgerValidation -Name "validationState" -DefaultValue "missing-owner-real-publish-evidence-availability-ledger-validation")
    publicDownloadValidationState = [string](Get-PropertyOrDefault -Object $publicDownloadValidation -Name "validationState" -DefaultValue "missing-public-package-download-proof-input-validation")
    postPublishCleanConsumerValidationState = [string](Get-PropertyOrDefault -Object $postPublishCleanValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-validation")
    strictCloseValidationState = [string](Get-PropertyOrDefault -Object $strictCloseValidation -Name "validationState" -DefaultValue "missing-strict-close-ready-convergence-dashboard-validation")
    finalClosureBridgeValidationState = [string](Get-PropertyOrDefault -Object $closureBridgeValidation -Name "validationState" -DefaultValue "missing-final-public-release-closure-bridge-validation")
    closeDecisionValidationState = [string](Get-PropertyOrDefault -Object $closeDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-close-owner-decision-input-validation")
    closeRecordValidationState = [string](Get-PropertyOrDefault -Object $closeRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
  }
  releaseEvidenceBundleSha256 = $bundleSha256
  classificationAuditSha256 = $classificationAuditSha256
  finalLandingPackValidationSha256 = $finalLandingPackSha256
  ownerImportReadinessValidationSha256 = $importDashboardSha256
  checkCount = @($checks).Count
  blockedCheckCount = @($checks).Count
  passedCheckCount = 0
  checks = @($checks)
  ownerActionRequired = $true
  passed = $false
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Release close final candidate audit pack is blocked candidate auditing only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-close-final-candidate-audit-pack.json"
$mdPath = Join-Path $OutputRoot "release-close-final-candidate-audit-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)
$rows = foreach ($check in $checks) {
  "| ``$($check.id)`` | ``$($check.checkState)`` | ``$($check.requiredRealInputCount)`` | ``$($check.blockedReasonCount)`` | ``$($check.passed)`` |"
}
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Release Close Final Candidate Audit Pack",
  "",
  "- auditState: ``$($record.auditState)``",
  "- checkCount: ``$($record.checkCount)``",
  "- blockedCheckCount: ``$($record.blockedCheckCount)``",
  "- releaseEvidenceBundleSha256: ``$($record.releaseEvidenceBundleSha256)``",
  "- performsPublish: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Check | State | Required Inputs | Blocked Reasons | Passed |",
  "| --- | --- | ---: | ---: | ---: |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)
Write-Host "ReleaseCloseFinalCandidateAuditPackState=$($record.auditState) Checks=$($record.checkCount) Blocked=$($record.blockedCheckCount)"
