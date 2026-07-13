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
  param([int]$Order, [string]$Lane, [string]$RelativePath, [string]$Purpose, [bool]$RequiresSha256)
  [pscustomobject]@{
    order = $Order
    lane = $Lane
    relativePath = $RelativePath
    purpose = $Purpose
    requiresSha256 = $RequiresSha256
    ownerActionRequired = $true
    passed = $false
  }
}

$files = @(
  New-StagingFile 1 "external-clean-consumer" "external-clean-consumer/restore.log" "Real package source restore log." $true
  New-StagingFile 2 "external-clean-consumer" "external-clean-consumer/build.log" "CleanConsumer build log." $true
  New-StagingFile 3 "external-clean-consumer" "external-clean-consumer/smoke.stdout.log" "CleanConsumer smoke stdout." $true
  New-StagingFile 4 "external-clean-consumer" "external-clean-consumer/smoke.stderr.log" "CleanConsumer smoke stderr." $true
  New-StagingFile 5 "external-clean-consumer" "external-clean-consumer/native-assets.json" "Restored native asset listing." $true
  New-StagingFile 6 "external-clean-consumer" "external-clean-consumer/host-metadata.json" "Host OS/GPU/CUDA/TensorRT/cuDNN metadata." $true
  New-StagingFile 7 "external-clean-consumer" "external-clean-consumer/package-metadata.json" "Managed/runtime package IDs, versions, source URL, and package SHA256 values." $false
  New-StagingFile 8 "post-publish" "post-publish/downloaded-packages.json" "Public package URLs and downloaded package SHA256 values." $true
  New-StagingFile 9 "post-publish" "post-publish/install.log" "Public package install/restore log." $true
  New-StagingFile 10 "post-publish" "post-publish/smoke.stdout.log" "Post-publish CleanConsumer smoke stdout." $true
  New-StagingFile 11 "post-publish" "post-publish/smoke.stderr.log" "Post-publish CleanConsumer smoke stderr." $true
  New-StagingFile 12 "post-publish" "post-publish/host-metadata.json" "Post-publish host metadata." $true
  New-StagingFile 13 "owner" "owner/rollback-review.json" "Owner rollback review." $false
  New-StagingFile 14 "owner" "owner/final-close-decision.json" "Owner final close decision." $false
  New-StagingFile 15 "owner" "owner/owner-confirmations.json" "Owner confirmations that no forbidden substitute was used." $false
)

$record = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-contract"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  contractState = "blocked-owner-real-proof-staging-workspace-contract-required"
  requiredFileCount = $files.Count
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
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 10)
$rows = foreach ($file in $files) { "| $($file.order) | ``$($file.lane)`` | ``$($file.relativePath)`` | ``$($file.requiresSha256)`` | $(ConvertTo-MarkdownCell $file.purpose) |" }
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Owner Real Proof Staging Workspace Contract", "", "| Order | Lane | Relative Path | SHA256 | Purpose |", "|---:|---|---|---:|---|", @($rows), "", "## Boundary", "", $record.boundary)
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $mdPath"
