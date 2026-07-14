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

function New-InputLane {
  param(
    [string]$Id,
    [string]$Title,
    [string[]]$RequiredFields,
    [string[]]$RejectedSubstitutes,
    [string[]]$RelatedArtifacts
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    requiredFieldCount = $RequiredFields.Count
    requiredFields = @($RequiredFields)
    rejectedSubstitutes = @($RejectedSubstitutes)
    relatedArtifacts = @($RelatedArtifacts)
    ownerActionRequired = $true
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner input skeleton lane is fillable guidance only; placeholders, local builds, local feeds, ProjectReference, direct nupkg, dashboards, queued workflows, hash-only reviews, and validation-ready records are not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$lanes = @(
  New-InputLane -Id "public-package-url-hash" -Title "Public package URL and hash evidence" -RequiredFields @("channel", "managedPackageId", "managedPackageVersion", "managedPackageUrl", "managedPackageSha256", "runtimePackageId", "runtimePackageVersion", "runtimePackageUrl", "runtimePackageSha256", "ownerReviewed") -RejectedSubstitutes @("local-feed", "direct-nupkg", "ProjectReference", "github-package-listing-only") -RelatedArtifacts @("artifacts/final-release/public-package-url-hash-proof-validator-validation.json", "artifacts/final-release/public-package-url-hash-download-verification-validation.json")
  New-InputLane -Id "external-clean-consumer-post-publish" -Title "Repository-external CleanConsumer post-publish proof" -RequiredFields @("externalWorkspacePath", "restoreCommand", "restoreLogPath", "nativeAssetListingPath", "dependencyProbeLogPath", "smokeCommand", "smokeLogPath", "stdoutSummary", "stderrSummary", "hostMetadataPath", "ownerReviewed") -RejectedSubstitutes @("local-test-only", "project-reference", "direct-nupkg", "pre-publish-smoke", "dependency-probe-only") -RelatedArtifacts @("artifacts/final-release/external-clean-consumer-post-publish-proof-validator-validation.json", "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json")
  New-InputLane -Id "article-publication" -Title "Public article publication proof" -RequiredFields @("articleProofRecordsPath", "articleProofManifestPath", "screenshotsArchivePath", "publicUrls", "publishedAtUtc", "ownerReviewed") -RejectedSubstitutes @("draft", "private-preview", "doc-only", "screenshot-without-public-url") -RelatedArtifacts @("artifacts/final-release/article-publication-proof-validator-validation.json", "artifacts/final-release/article-publication-proof-from-staging-workspace-validation.json")
  New-InputLane -Id "yolovision-real-model" -Title "YoloVision real model asset and runtime proof" -RequiredFields @("assetManifestPath", "modelLicensePath", "runtimeTranscriptPath", "hostMetadataPath", "realExecutionConfirmationPath", "assetSha256Manifest", "ownerReviewed") -RejectedSubstitutes @("readiness-matrix-only", "tutorial-output-only", "synthetic-model-only", "manifest-without-runtime-transcript") -RelatedArtifacts @("artifacts/final-release/yolovision-real-model-post-publish-proof-validator-validation.json", "artifacts/final-release/yolovision-real-model-proof-from-staging-workspace-validation.json")
  New-InputLane -Id "final-owner-rollback-review" -Title "Final Owner rollback review" -RequiredFields @("rollbackPlanReviewed", "rollbackOwner", "rollbackReviewedAtUtc", "rollbackReviewNotes", "ownerReviewed") -RejectedSubstitutes @("template", "dashboard", "unreviewed-plan") -RelatedArtifacts @("artifacts/final-release/final-owner-rollback-review-validation.json")
  New-InputLane -Id "final-owner-close-decision" -Title "Final Owner close decision" -RequiredFields @("finalCloseApproved", "releaseIssueUrl", "releaseEvidenceBundleSha256", "classificationAuditSha256", "ownerApprover", "approvedAtUtc", "ownerReviewed") -RejectedSubstitutes @("candidate", "manual-close-preview", "missing-owner-approval", "bundle-hash-only") -RelatedArtifacts @("artifacts/final-release/final-owner-close-decision-validation.json")
  New-InputLane -Id "github-ci-evidence" -Title "GitHub Actions CI evidence" -RequiredFields @("repository", "branch", "commitSha", "workflowName", "runId", "runUrl", "conclusion", "createdAtUtc", "completedAtUtc", "artifactManifestSha256", "ownerReviewed") -RejectedSubstitutes @("queued-workflow", "in-progress-workflow", "local-build", "local-test", "dry-run") -RelatedArtifacts @("artifacts/final-release/github-ci-evidence-from-owner-input-validation.json")
  New-InputLane -Id "release-evidence-bundle-hash-review" -Title "Release evidence bundle hash review" -RequiredFields @("releaseEvidenceBundleSha256", "reviewedAtUtc", "ownerReviewer", "ownerReviewed") -RejectedSubstitutes @("hash-without-owner-review", "old-bundle-hash", "local-summary-only") -RelatedArtifacts @("artifacts/final-release/release-evidence-bundle-hash-review-validation.json")
  New-InputLane -Id "classification-audit-hash-review" -Title "Classification audit hash review" -RequiredFields @("classificationAuditSha256", "classificationAuditState", "reviewedAtUtc", "ownerReviewer", "ownerReviewed") -RejectedSubstitutes @("hash-without-owner-review", "failed-audit", "old-audit-hash") -RelatedArtifacts @("artifacts/final-release/classification-audit-hash-review-validation.json")
)

$template = [pscustomobject]@{
  recordKind = "final-owner-execution-input-skeleton-template"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  instructions = "Owner fills concrete evidence paths, URLs, hashes, reviewers, and timestamps here. Placeholder/template/hash-only values are never proof."
  publicPackageUrlHash = [pscustomobject]@{
    channel = "<owner-public-channel>"
    managedPackageId = "<owner-managed-package-id>"
    managedPackageVersion = "<owner-managed-package-version>"
    managedPackageUrl = "<owner-public-managed-package-url>"
    managedPackageSha256 = "<owner-managed-package-sha256>"
    runtimePackageId = "<owner-runtime-package-id>"
    runtimePackageVersion = "<owner-runtime-package-version>"
    runtimePackageUrl = "<owner-public-runtime-package-url>"
    runtimePackageSha256 = "<owner-runtime-package-sha256>"
    ownerReviewed = $false
  }
  githubCiEvidence = [pscustomobject]@{
    repository = "<owner-github-repository>"
    branch = "TensorRtSharp4.0"
    commitSha = "<owner-commit-sha>"
    workflowName = "release-quality-gate"
    runId = "<owner-run-id>"
    runUrl = "<owner-run-url>"
    conclusion = "<owner-conclusion>"
    createdAtUtc = "<owner-created-at-utc>"
    completedAtUtc = "<owner-completed-at-utc>"
    artifactManifestSha256 = "<owner-artifact-manifest-sha256>"
    ownerReviewed = $false
  }
  releaseEvidenceBundleHashReview = [pscustomobject]@{
    releaseEvidenceBundleSha256 = "<owner-current-release-evidence-bundle-sha256>"
    reviewedAtUtc = "<owner-reviewed-at-utc>"
    ownerReviewer = "<owner-reviewer>"
    ownerReviewed = $false
  }
  classificationAuditHashReview = [pscustomobject]@{
    classificationAuditSha256 = "<owner-current-classification-audit-sha256>"
    classificationAuditState = "classification-audit-passed-non-proof-boundaries-intact"
    reviewedAtUtc = "<owner-reviewed-at-utc>"
    ownerReviewer = "<owner-reviewer>"
    ownerReviewed = $false
  }
}

$record = [pscustomobject]@{
  recordKind = "final-owner-execution-input-skeleton"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  skeletonState = "final-owner-execution-input-skeleton-ready-non-proof"
  laneCount = $lanes.Count
  requiredFieldCount = @($lanes | ForEach-Object { [int]$_.requiredFieldCount } | Measure-Object -Sum).Sum
  rejectedSubstituteCount = @($lanes | ForEach-Object { $_.rejectedSubstitutes } | Measure-Object).Count
  lanes = @($lanes)
  templateArtifact = "artifacts/final-release/final-owner-execution-input-skeleton.template.json"
  ownerActionRequired = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner execution input skeleton is fillable guidance and schema shape only; it does not publish, does not use tokens, does not close the release issue, and is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-input-skeleton.template.json") -InputObject ($template | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-input-skeleton.json") -InputObject ($record | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath (Join-Path $OutputRoot "final-owner-execution-input-skeleton.md") -InputObject @(
  "# Final Owner Execution Input Skeleton",
  "",
  "- skeletonState: ``$($record.skeletonState)``",
  "- laneCount: ``$($record.laneCount)``",
  "- requiredFieldCount: ``$($record.requiredFieldCount)``",
  "- canPublishPublicly: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $record.boundary
)
Write-Host "FinalOwnerExecutionInputSkeletonState=$($record.skeletonState) Lanes=$($record.laneCount) RequiredFields=$($record.requiredFieldCount)"
