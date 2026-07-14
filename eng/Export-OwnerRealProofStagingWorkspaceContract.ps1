[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$OutputRoot = $ctx.OutputRoot

function New-StagingFile {
  param(
    [int]$Order,
    [string]$Lane,
    [string]$RelativePath,
    [string]$Purpose,
    [bool]$RequiresSha256,
    [string]$EvidenceKind,
    [string]$TargetJson,
    [string]$TargetField,
    [string]$TargetHashField,
    [string[]]$ForbiddenSubstitutes
  )

  [pscustomobject]@{
    order = $Order
    lane = $Lane
    relativePath = $RelativePath
    purpose = $Purpose
    evidenceKind = $EvidenceKind
    targetJson = $TargetJson
    targetField = $TargetField
    targetHashField = $TargetHashField
    requiresSha256 = $RequiresSha256
    existencePolicy = "must-exist-for-strict-import"
    hashMatchPolicy = if ($RequiresSha256) { "must-compute-sha256-for-strict-import" } else { "metadata-or-decision-record-no-download" }
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    forbiddenSubstituteCount = @($ForbiddenSubstitutes).Count
    ownerActionRequired = $true
    passed = $false
  }
}

$commonRejects = @("template", "example", "draft", "dashboard", "audit", "bundle", "local feed", "ProjectReference", "direct .nupkg", "queued workflow", "missing runner", "dry-run", "ready fixture")
$externalRejects = @($commonRejects + @("repository root", "bin/obj", "local package cache", "NuGet.Config local source"))
$yoloRejects = @($commonRejects + @("readiness", "tutorial", "matrix-only", "sample placeholder"))
$articleRejects = @($commonRejects + @("roadmap", "unpublished draft", "private URL"))
$closeRejects = @($commonRejects + @("auto close", "gh issue close", "unreviewed close decision"))

$files = @(
  New-StagingFile 1 "public-package" "public-package/nuget-managed-package.nupkg" "Downloaded public NuGet managed bridge/core package." $true "public-package-file" "owner-post-publish-docs-article-sample-real-input.owner.json" "nugetManagedPackagePath" "nugetManagedPackageSha256" $commonRejects
  New-StagingFile 2 "public-package" "public-package/github-runtime-package.nupkg" "Downloaded public GitHub Packages runtime package." $true "public-package-file" "owner-post-publish-docs-article-sample-real-input.owner.json" "githubRuntimePackagePath" "githubRuntimePackageSha256" $commonRejects
  New-StagingFile 3 "public-package" "public-package/downloaded-packages.json" "Public package URL/version/hash manifest." $true "public-package-manifest" "owner-post-publish-docs-article-sample-real-input.owner.json" "downloadedPackagesManifestPath" "downloadedPackagesManifestSha256" $commonRejects
  New-StagingFile 4 "public-package" "public-package/download-transcript.log" "Transcript for public package downloads and SHA256 capture." $true "transcript" "owner-post-publish-docs-article-sample-real-input.owner.json" "downloadTranscriptPath" "downloadTranscriptSha256" $commonRejects

  New-StagingFile 5 "external-clean-consumer" "external-clean-consumer/consumer.csproj" "Repository-external consumer project file." $true "external-consumer-project" "external-clean-consumer-execution-result.owner.json" "consumerProjectPath" "consumerProjectFileSha256" $externalRejects
  New-StagingFile 6 "external-clean-consumer" "external-clean-consumer/package-source.json" "Public package source URL and restore source metadata." $true "package-source" "external-clean-consumer-execution-result.owner.json" "packageRestoreSourcePath" "packageRestoreSourceSha256" $externalRejects
  New-StagingFile 7 "external-clean-consumer" "external-clean-consumer/resolved-packages.json" "Resolved package list proving public package consumption." $true "resolved-package-list" "external-clean-consumer-execution-result.owner.json" "restoreResolvedPackageListPath" "restoreResolvedPackageListSha256" $externalRejects
  New-StagingFile 8 "external-clean-consumer" "external-clean-consumer/restore.log" "Real package source restore log from external workspace." $true "transcript" "external-clean-consumer-execution-result.owner.json" "restoreLogPath" "restoreLogSha256" $externalRejects
  New-StagingFile 9 "external-clean-consumer" "external-clean-consumer/build.log" "External CleanConsumer build log." $true "transcript" "external-clean-consumer-execution-result.owner.json" "buildLogPath" "buildLogSha256" $externalRejects
  New-StagingFile 10 "external-clean-consumer" "external-clean-consumer/smoke.stdout.log" "External CleanConsumer smoke stdout." $true "transcript" "external-clean-consumer-execution-result.owner.json" "smokeStdoutPath" "smokeStdoutSha256" $externalRejects
  New-StagingFile 11 "external-clean-consumer" "external-clean-consumer/smoke.stderr.log" "External CleanConsumer smoke stderr." $true "transcript" "external-clean-consumer-execution-result.owner.json" "smokeStderrPath" "smokeStderrSha256" $externalRejects
  New-StagingFile 12 "external-clean-consumer" "external-clean-consumer/merged-transcript.log" "Merged restore/build/smoke transcript." $true "transcript" "external-clean-consumer-execution-result.owner.json" "mergedTranscriptPath" "mergedTranscriptSha256" $externalRejects
  New-StagingFile 13 "external-clean-consumer" "external-clean-consumer/native-assets.json" "Restored native asset listing." $true "runtime-asset-manifest" "external-clean-consumer-execution-result.owner.json" "nativeAssetListingPath" "nativeAssetListingSha256" $externalRejects
  New-StagingFile 14 "external-clean-consumer" "external-clean-consumer/host-metadata.json" "Host OS/GPU/CUDA/TensorRT/cuDNN metadata." $true "host-metadata" "external-clean-consumer-execution-result.owner.json" "hostMetadataPath" "hostMetadataSha256" $externalRejects
  New-StagingFile 15 "external-clean-consumer" "external-clean-consumer/package-metadata.json" "Managed/runtime package IDs, versions, public source URL, and package SHA256 values." $true "package-metadata" "external-clean-consumer-execution-result.owner.json" "packageMetadataPath" "packageMetadataSha256" $externalRejects
  New-StagingFile 16 "external-clean-consumer" "external-clean-consumer/no-local-substitute-confirmation.json" "Owner confirmation that no local feed, ProjectReference, or direct nupkg was used." $true "owner-confirmation" "external-clean-consumer-execution-result.owner.json" "noLocalSubstituteConfirmationPath" "noLocalSubstituteConfirmationSha256" $externalRejects

  New-StagingFile 17 "yolovision-real-model" "yolovision/task-metadata.json" "YoloVision task name, model family, and task coverage metadata." $true "yolovision-task-metadata" "yolovision-real-model-proof.owner.json" "taskMetadataPath" "taskMetadataSha256" $yoloRejects
  New-StagingFile 18 "yolovision-real-model" "yolovision/model.onnx" "Real YOLO model file used for execution." $true "real-model" "yolovision-real-model-proof.owner.json" "modelPath" "modelSha256" $yoloRejects
  New-StagingFile 19 "yolovision-real-model" "yolovision/model-license.json" "Model source URL, license, and redistribution review." $true "license" "yolovision-real-model-proof.owner.json" "modelLicensePath" "modelLicenseSha256" $yoloRejects
  New-StagingFile 20 "yolovision-real-model" "yolovision/labels.txt" "Labels file used by the real YoloVision run." $true "labels" "yolovision-real-model-proof.owner.json" "labelsPath" "labelsSha256" $yoloRejects
  New-StagingFile 21 "yolovision-real-model" "yolovision/input.bin" "Input image/tensor used by the real YoloVision run." $true "input" "yolovision-real-model-proof.owner.json" "inputPath" "inputImageSha256" $yoloRejects
  New-StagingFile 22 "yolovision-real-model" "yolovision/asset-manifest.json" "Asset manifest linking model, labels, input, output, logs, and host metadata." $true "asset-manifest" "yolovision-real-model-proof.owner.json" "assetManifestPath" "assetManifestSha256" $yoloRejects
  New-StagingFile 23 "yolovision-real-model" "yolovision/output.json" "YoloVision output JSON generated by the real run." $true "output" "yolovision-real-model-proof.owner.json" "outputJsonPath" "outputJsonSha256" $yoloRejects
  New-StagingFile 24 "yolovision-real-model" "yolovision/stdout.log" "YoloVision stdout log from real execution." $true "transcript" "yolovision-real-model-proof.owner.json" "stdoutLogPath" "stdoutLogSha256" $yoloRejects
  New-StagingFile 25 "yolovision-real-model" "yolovision/stderr.log" "YoloVision stderr log from real execution." $true "transcript" "yolovision-real-model-proof.owner.json" "stderrLogPath" "stderrLogSha256" $yoloRejects
  New-StagingFile 26 "yolovision-real-model" "yolovision/runtime-transcript.log" "Full YoloVision command transcript." $true "transcript" "yolovision-real-model-proof.owner.json" "runtimeTranscriptPath" "runtimeTranscriptSha256" $yoloRejects
  New-StagingFile 27 "yolovision-real-model" "yolovision/host-metadata.json" "Host metadata for the real YoloVision execution." $true "host-metadata" "yolovision-real-model-proof.owner.json" "hostMetadataPath" "hostMetadataSha256" $yoloRejects
  New-StagingFile 28 "yolovision-real-model" "yolovision/real-model-execution-confirmation.json" "Owner confirmation that this is real model execution, not readiness/tutorial/matrix evidence." $true "owner-confirmation" "yolovision-real-model-proof.owner.json" "realModelExecutionConfirmationPath" "realModelExecutionConfirmationSha256" $yoloRejects

  New-StagingFile 29 "article-publication" "article-publication/article-proof-records.json" "Array of public article proof records." $true "article-proof-records" "article-publication-proof.owner.json" "articleProofRecordsPath" "articleProofRecordsSha256" $articleRejects
  New-StagingFile 30 "article-publication" "article-publication/article-proof-manifest.json" "Manifest URL/hash and article proof count." $true "article-proof-manifest" "article-publication-proof.owner.json" "articleProofManifestPath" "articleProofManifestSha256" $articleRejects
  New-StagingFile 31 "article-publication" "article-publication/screenshots.zip" "Screenshots of published public article pages." $true "article-screenshots" "article-publication-proof.owner.json" "articleScreenshotsArchivePath" "articleScreenshotsArchiveSha256" $articleRejects

  New-StagingFile 32 "release-close" "release-close/release-evidence-bundle.sha256" "Owner-reviewed release evidence bundle SHA256." $true "release-evidence-hash" "release-issue-close-material.owner.json" "releaseEvidenceBundleSha256Path" "releaseEvidenceBundleSha256" $closeRejects
  New-StagingFile 33 "release-close" "release-close/classification-audit.sha256" "Owner-reviewed classification audit SHA256." $true "classification-audit-hash" "release-issue-close-material.owner.json" "classificationAuditSha256Path" "classificationAuditSha256" $closeRejects
  New-StagingFile 34 "release-close" "release-close/post-publish-proof.sha256" "Owner-reviewed post-publish proof SHA256." $true "post-publish-proof-hash" "release-issue-close-material.owner.json" "postPublishProofSha256Path" "postPublishProofSha256" $closeRejects
  New-StagingFile 35 "release-close" "release-close/rollback-review.json" "Owner rollback/no-rollback decision." $true "rollback-review" "release-issue-close-material.owner.json" "rollbackReviewPath" "rollbackReviewSha256" $closeRejects
  New-StagingFile 36 "release-close" "release-close/final-close-decision.json" "Owner final manual close decision." $true "final-close-decision" "release-issue-close-material.owner.json" "finalCloseDecisionPath" "finalCloseDecisionSha256" $closeRejects
  New-StagingFile 37 "release-close" "release-close/known-limitations.json" "Known limitations acknowledgement for manual close review." $true "known-limitations" "release-issue-close-material.owner.json" "knownLimitationsPath" "knownLimitationsSha256" $closeRejects
)

$lanes = @($files | Select-Object -ExpandProperty lane -Unique)
$sha256RequiredCount = @($files | Where-Object { [bool]$_.requiresSha256 }).Count
$forbiddenSubstituteCount = [int](($files | Measure-Object -Property forbiddenSubstituteCount -Sum).Sum)

$record = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-contract"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  contractState = "blocked-owner-real-proof-staging-workspace-contract-required"
  laneCount = $lanes.Count
  lanes = @($lanes)
  requiredFileCount = $files.Count
  sha256RequiredFileCount = $sha256RequiredCount
  forbiddenSubstituteCount = $forbiddenSubstituteCount
  externalWorkspaceRequired = $true
  requiredFiles = @($files)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner real proof staging workspace contract defines expected file layout only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-contract.json"
$mdPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-contract.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$rows = foreach ($file in $files) { "| $($file.order) | ``$($file.lane)`` | ``$($file.relativePath)`` | ``$($file.evidenceKind)`` | ``$($file.requiresSha256)`` | $(ConvertTo-MarkdownCell $file.purpose) |" }
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Owner Real Proof Staging Workspace Contract", "", "- laneCount: ``$($record.laneCount)``", "- requiredFileCount: ``$($record.requiredFileCount)``", "- sha256RequiredFileCount: ``$($record.sha256RequiredFileCount)``", "- forbiddenSubstituteCount: ``$($record.forbiddenSubstituteCount)``", "- externalWorkspaceRequired: ``True``", "", "| Order | Lane | Relative Path | Evidence Kind | SHA256 | Purpose |", "|---:|---|---|---|---:|---|", @($rows), "", "## Boundary", "", $record.boundary)
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $mdPath"
