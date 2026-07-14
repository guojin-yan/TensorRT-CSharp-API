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

function New-OwnerActionLane {
  param([string]$Id, [string]$Title, [string[]]$RequiredFields, [string[]]$Commands, [string[]]$ForbiddenSubstitutes)
  [pscustomobject]@{
    id = $Id
    title = $Title
    laneState = "blocked-owner-real-input-required"
    requiredFields = @($RequiredFields)
    requiredFieldCount = @($RequiredFields).Count
    copyableCommands = @($Commands)
    commandCount = @($Commands).Count
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    forbiddenSubstituteCount = @($ForbiddenSubstitutes).Count
    ownerActionRequired = $true
    proofReady = $false
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$forbidden = @("template", "dashboard", "audit", "bundle", "local feed", "ProjectReference", "direct .nupkg", "queued workflow", "missing runner", "dry-run", "ready fixture", "readiness", "tutorial", "matrix", "local package cache", "NuGet.Config local source", ".nupkg")
$lanes = @(
  New-OwnerActionLane -Id "public-package-urls-and-hashes" -Title "Public package URLs and hashes" -RequiredFields @("nugetManagedPackageUrl", "nugetManagedPackageVersion", "nugetManagedPackageSha256", "githubRuntimePackageUrl", "githubRuntimePackageVersion", "githubRuntimePackageSha256", "downloadedAtUtc", "downloadTranscriptSha256") -Commands @("dotnet nuget list source", "pwsh -NoProfile -File .\\eng\\Test-PublicPackageDownloadProofInput.ps1 -Strict") -ForbiddenSubstitutes $forbidden
  New-OwnerActionLane -Id "external-clean-consumer-logs" -Title "External clean consumer logs" -RequiredFields @("externalWorkspaceRoot", "consumerProjectFileSha256", "packageRestoreSourceUrl", "restoreResolvedPackageListSha256", "restoreLogSha256", "buildLogSha256", "smokeLogSha256", "hostMetadataSha256", "noLocalSubstituteConfirmation") -Commands @("dotnet restore <external-consumer.csproj> --source <public-package-source>", "dotnet build <external-consumer.csproj> -c Release --no-restore", "dotnet run --project <external-consumer.csproj> -- --runtime-package-key <runtime-key>") -ForbiddenSubstitutes $forbidden
  New-OwnerActionLane -Id "yolovision-real-model-assets" -Title "YoloVision real model assets" -RequiredFields @("taskName", "modelSourceUrl", "modelLicense", "modelSha256", "labelsSha256", "inputImageSha256", "assetManifestSha256", "outputJsonSha256", "stdoutLogSha256", "stderrLogSha256", "hostMetadataSha256", "runtimeTranscriptSha256", "realModelExecutionConfirmation", "ownerReviewedAtUtc") -Commands @("dotnet run --project .\\samples\\YoloVision -- --model <model.onnx> --labels <labels.txt> --task <det|cls|seg|obb|pose|sem> --input-shape <shape>", "pwsh -NoProfile -File .\\eng\\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict") -ForbiddenSubstitutes $forbidden
  New-OwnerActionLane -Id "article-publication-urls" -Title "Article publication URLs" -RequiredFields @("articleProofCount", "articleProofManifestUrl", "articleProofManifestSha256", "articleProofRecords[]", "articleId", "articleTitle", "articlePublishUrl", "articlePublishedAtUtc", "articleScreenshotSha256", "linkedPublicPackageUrl", "linkedProofHash") -Commands @("Review docs/articles/zh-cn/publishing/article-roadmap-30plus.json", "Record public article URLs only after Owner publish evidence exists", "Attach articleProofRecords[] for every published article") -ForbiddenSubstitutes $forbidden
  New-OwnerActionLane -Id "release-issue-close-material" -Title "Release Issue close material" -RequiredFields @("releaseIssueUrl", "ownerCloseDecision", "releaseEvidenceBundleSha256", "classificationAuditSha256", "postPublishProofSha256", "rollbackDecision", "ownerFinalCloseDecisionImportedAtUtc", "manualCloseReviewConfirmation", "knownLimitationsAcknowledgement") -Commands @("pwsh -NoProfile -File .\\eng\\Test-FinalRealProofImportAndCloseCandidatePack.ps1 -Strict", "pwsh -NoProfile -File .\\eng\\Test-ReleaseIssueCloseRecord.ps1 -Strict") -ForbiddenSubstitutes $forbidden
)

$requiredFieldCount = 0
$commandCount = 0
$forbiddenCount = 0
foreach ($lane in $lanes) {
  $requiredFieldCount += [int]$lane.requiredFieldCount
  $commandCount += [int]$lane.commandCount
  $forbiddenCount += [int]$lane.forbiddenSubstituteCount
}

$record = [pscustomobject]@{
  recordKind = "owner-post-publish-docs-article-sample-execution-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  executionPackState = "blocked-owner-post-publish-docs-article-sample-real-input-required"
  laneCount = @($lanes).Count
  blockedLaneCount = @($lanes).Count
  proofReadyLaneCount = 0
  requiredFieldCount = $requiredFieldCount
  copyableCommandCount = $commandCount
  forbiddenSubstituteCount = $forbiddenCount
  lanes = @($lanes)
  sourceArtifacts = @(
    "artifacts/final-release/post-publish-public-docs-article-sample-readiness-pack.json",
    "artifacts/final-release/public-claim-post-publish-proof-boundary-audit.json",
    "artifacts/final-release/final-real-proof-import-and-close-candidate-pack.json",
    "docs/articles/zh-cn/publishing/article-roadmap-30plus.json",
    "samples/YoloVision/yolo-model-matrix.json",
    "applications/TensorRtExec/tensor-rt-exec-feature-matrix.json"
  )
  ownerActionRequired = $true
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner post-publish docs/article/sample execution pack is copyable Owner guidance only; it does not publish packages or articles, does not run clean consumer, does not close release issue, and is not runtime proof or post-publish proof."
}

$jsonPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-execution-pack.json"
$mdPath = Join-Path $OutputRoot "owner-post-publish-docs-article-sample-execution-pack.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 14)

$rows = foreach ($lane in $lanes) {
  "| ``$($lane.id)`` | ``$($lane.laneState)`` | ``$($lane.requiredFieldCount)`` | ``$($lane.commandCount)`` |"
}

Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Owner Post-Publish Docs Article Sample Execution Pack",
  "",
  "- executionPackState: ``$($record.executionPackState)``",
  "- laneCount: ``$($record.laneCount)``",
  "- blockedLaneCount: ``$($record.blockedLaneCount)``",
  "- proofReadyLaneCount: ``0``",
  "- requiredFieldCount: ``$($record.requiredFieldCount)``",
  "- copyableCommandCount: ``$($record.copyableCommandCount)``",
  "- forbiddenSubstituteCount: ``$($record.forbiddenSubstituteCount)``",
  "- performsPublish: ``False``",
  "- canCloseReleaseIssue: ``False``",
  "",
  "| Lane | State | Fields | Commands |",
  "| --- | --- | ---: | ---: |",
  @($rows),
  "",
  "## Boundary",
  "",
  $record.boundary
)

Write-Host "OwnerPostPublishDocsArticleSampleExecutionPackState=$($record.executionPackState) Lanes=$($record.laneCount) Fields=$($record.requiredFieldCount)"
