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

# Keep the original field contract as an additive compatibility view for the
# real-input template/import pipeline. The lane view remains the release-facing index.
$compatibilityFieldSpecs = @(
  @("clean-consumer-project-root", "clean-consumer", "cleanConsumer.projectRoot", "path"),
  @("clean-consumer-restore-log-path", "clean-consumer", "cleanConsumer.restoreLog.path", "path"),
  @("clean-consumer-build-log-path", "clean-consumer", "cleanConsumer.buildLog.path", "path"),
  @("package-source-url", "package-source", "packageSource.url", "url"),
  @("managed-package-id", "package-source", "managedPackage.id", "identity"),
  @("managed-package-version", "package-source", "managedPackage.version", "identity"),
  @("managed-package-sha256", "package-source", "managedPackage.sha256", "sha256"),
  @("runtime-package-id", "package-source", "runtimePackage.id", "identity"),
  @("runtime-package-version", "package-source", "runtimePackage.version", "identity"),
  @("runtime-package-key", "package-source", "runtimePackage.key", "identity"),
  @("runtime-package-sha256", "package-source", "runtimePackage.sha256", "sha256"),
  @("native-asset-listing-path", "native-assets", "nativeAssetListing.path", "path"),
  @("native-asset-listing-sha256", "native-assets", "nativeAssetListing.sha256", "sha256"),
  @("stdout-path", "runtime-logs", "runtimeSmoke.stdoutPath", "path"),
  @("stderr-path", "runtime-logs", "runtimeSmoke.stderrPath", "path"),
  @("merged-transcript-path", "runtime-logs", "runtimeSmoke.mergedTranscriptPath", "path"),
  @("stdout-sha256", "runtime-logs", "runtimeSmoke.stdoutSha256", "sha256"),
  @("stderr-sha256", "runtime-logs", "runtimeSmoke.stderrSha256", "sha256"),
  @("merged-transcript-sha256", "runtime-logs", "runtimeSmoke.mergedTranscriptSha256", "sha256"),
  @("exit-code", "runtime-logs", "runtimeSmoke.exitCode", "integer"),
  @("smoke-status", "runtime-logs", "runtimeSmoke.smokeStatus", "status"),
  @("host-os", "host-metadata", "host.os", "identity"),
  @("host-arch", "host-metadata", "host.arch", "identity"),
  @("host-rid", "host-metadata", "host.rid", "identity"),
  @("gpu-name", "host-metadata", "host.gpuName", "identity"),
  @("nvidia-driver", "host-metadata", "host.nvidiaDriver", "identity"),
  @("cuda-runtime-toolkit", "host-metadata", "host.cudaRuntimeToolkit", "identity"),
  @("tensorrt-version", "host-metadata", "host.tensorrt", "identity"),
  @("cudnn-version", "host-metadata", "host.cudnn", "identity"),
  @("owner-name", "owner-review", "owner.name", "identity"),
  @("owner-machine", "owner-review", "owner.machine", "identity"),
  @("reviewed-at-utc", "owner-review", "owner.reviewedAtUtc", "datetime"),
  @("owner-note", "owner-review", "owner.note", "text"),
  @("post-publish-downloaded-package-hash", "post-publish", "postPublish.downloadedPackageSha256", "sha256"),
  @("post-publish-proof-log-path", "post-publish", "postPublish.proofLogPath", "path"),
  @("post-publish-proof-log-sha256", "post-publish", "postPublish.proofLogSha256", "sha256"),
  @("dual-package-nuget-owner-authorization-url", "dual-package-route-proof", "dualPackageRoutes.nugetSmallBridgeCore.ownerAuthorizationUrl", "url"),
  @("dual-package-nuget-public-download-url", "dual-package-route-proof", "dualPackageRoutes.nugetSmallBridgeCore.publicPackageDownloadUrl", "url"),
  @("dual-package-nuget-clean-consumer-log-path", "dual-package-route-proof", "dualPackageRoutes.nugetSmallBridgeCore.cleanConsumerProofLogPath", "path"),
  @("dual-package-nuget-post-publish-proof-log-sha256", "dual-package-route-proof", "dualPackageRoutes.nugetSmallBridgeCore.postPublishProofLogSha256", "sha256"),
  @("dual-package-github-owner-authorization-url", "dual-package-route-proof", "dualPackageRoutes.githubPackagesFullRuntime.ownerAuthorizationUrl", "url"),
  @("dual-package-github-restore-source-url", "dual-package-route-proof", "dualPackageRoutes.githubPackagesFullRuntime.restoreSourceUrl", "url"),
  @("dual-package-github-runtime-dll-resolution-report-path", "dual-package-route-proof", "dualPackageRoutes.githubPackagesFullRuntime.runtimeDllResolutionReportPath", "path"),
  @("dual-package-github-clean-runtime-smoke-log-sha256", "dual-package-route-proof", "dualPackageRoutes.githubPackagesFullRuntime.cleanRuntimeSmokeLogSha256", "sha256"),
  @("rollback-review", "release-close", "rollback.review", "text"),
  @("final-close-decision", "release-close", "finalClose.decision", "decision"),
  @("strict-validator-output-path", "strict-validators", "strictValidators.outputPath", "path"),
  @("strict-validator-output-sha256", "strict-validators", "strictValidators.outputSha256", "sha256"),
  @("strict-validator-chain-state", "strict-validators", "strictValidators.chainState", "status")
)
$compatibilityFields = @(
  foreach ($spec in $compatibilityFieldSpecs) {
    [pscustomobject]@{
      id = [string]$spec[0]
      group = [string]$spec[1]
      fieldPath = [string]$spec[2]
      kind = [string]$spec[3]
      required = $true
      value = "<owner-real-input-required>"
      placeholder = $true
      status = "missing owner input"
      readyForImport = $false
      ownerActionRequired = $true
    }
  }
)
$compatibilityGroupTitles = [ordered]@{
  "clean-consumer" = "Clean external package consumer"
  "package-source" = "Package source and package identity"
  "native-assets" = "Native asset listing"
  "runtime-logs" = "Restore, build, run logs and runtime status"
  "host-metadata" = "Host metadata"
  "owner-review" = "Owner review"
  "post-publish" = "Post-publish proof"
  "dual-package-route-proof" = "Dual-package route proof"
  "release-close" = "Rollback review and final close"
  "strict-validators" = "Strict validator chain"
}
$compatibilityGroups = @(
  foreach ($groupId in $compatibilityGroupTitles.Keys) {
    $groupFields = @($compatibilityFields | Where-Object group -eq $groupId)
    [pscustomobject]@{
      id = $groupId
      title = [string]$compatibilityGroupTitles[$groupId]
      fieldCount = $groupFields.Count
      missingFieldCount = $groupFields.Count
      placeholderFieldCount = $groupFields.Count
      readyFieldCount = 0
      fields = @($groupFields)
      ownerActionRequired = $true
      readyForImport = $false
    }
  }
)

$oneScreen = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/final-owner-execution-one-screen-pack.json"
$oneScreenValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/final-owner-execution-one-screen-pack-validation.json"
$runtimeAlignment = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json"
$runtimeAlignmentValidation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"

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
  skeletonState = "blocked-final-owner-real-input-required"
  sourceOneScreenPackState = [string](Get-PropertyOrDefault -Object $oneScreen -Name "packState" -DefaultValue "missing-final-owner-execution-one-screen-pack")
  sourceOneScreenPackValidationState = [string](Get-PropertyOrDefault -Object $oneScreenValidation -Name "validationState" -DefaultValue "missing-final-owner-execution-one-screen-pack-validation")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $runtimeAlignment -Name "alignmentState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $runtimeAlignmentValidation -Name "validationState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment-validation")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = [string](Get-PropertyOrDefault -Object $runtimeAlignment -Name "runtimeSmokeStatus" -DefaultValue "Smoke=missing")
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount = [int](Get-PropertyOrDefault -Object $runtimeAlignment -Name "fieldCount" -DefaultValue 0)
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $runtimeAlignment -Name "missingRequiredFieldCount" -DefaultValue -1)
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount = [int](Get-PropertyOrDefault -Object $runtimeAlignmentValidation -Name "failedBlockerCount" -DefaultValue -1)
  laneCount = $lanes.Count
  laneRequiredFieldCount = [int](@($lanes | ForEach-Object { [int]$_.requiredFieldCount } | Measure-Object -Sum).Sum)
  fieldGroupCount = $compatibilityGroups.Count
  requiredFieldCount = $compatibilityFields.Count
  missingFieldCount = $compatibilityFields.Count
  placeholderFieldCount = $compatibilityFields.Count
  readyForImportFieldCount = 0
  rejectedSubstituteCount = @($lanes | ForEach-Object { $_.rejectedSubstitutes } | Measure-Object).Count
  lanes = @($lanes)
  fieldGroups = @($compatibilityGroups)
  templateArtifact = "artifacts/final-release/final-owner-execution-input-skeleton.template.json"
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  nonSubstituteProofKinds = @("final owner execution input skeleton", "owner fillable input skeleton", "placeholder owner input", "dual package route proof", "dual package final close lanes", "local feed", "ProjectReference", "direct .nupkg", "pre-publish smoke reused as post-publish proof")
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
