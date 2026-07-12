[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$PostPublishProofPath
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
if ([string]::IsNullOrWhiteSpace($PostPublishProofPath)) { $PostPublishProofPath = Join-Path $ctx.OutputDirectory "post-publish-clean-consumer-real-proof-from-owner-result.json" }
if (-not (Test-Path -LiteralPath $PostPublishProofPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Import-PostPublishCleanConsumerRealProofFromOwnerResult.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$postPublish = Get-Content -LiteralPath $PostPublishProofPath -Raw -Encoding utf8 | ConvertFrom-Json
$readyProofCount = [int](Get-OwnerPropertyOrDefault -Object $postPublish -Name "readyProofCount" -DefaultValue 0)
$approvalFields = @(
  "ownerReviewer", "ownerReviewTimestampUtc", "ownerSignature", "ownerApprovalId", "finalPublicPackageUrlApproval",
  "finalPublicPackageHashApproval", "releaseIssueCloseDecision", "releaseIssueCloseDecisionReason", "rollbackDecision",
  "releaseNotesApprovalId", "postPublishProofValidationHash", "strictValidatorOutputSha256"
)

$record = [ordered]@{
  recordKind = "final-release-close-approval-real-input-from-owner-result"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = "blocked-final-release-close-owner-approval-real-input-required"
  sourcePostPublishProofPath = $PostPublishProofPath
  sourceReadyProofCount = $readyProofCount
  requiredApprovalFieldCount = $approvalFields.Count
  readyCloseApprovalCount = 0
  blockedApprovalFieldCount = $approvalFields.Count
  failedBlockerCount = 0
  failedActionRequiredCount = $approvalFields.Count
  requiredApprovalFields = @($approvalFields | ForEach-Object { [pscustomobject]@{ name = $_; valueState = "owner-input-required"; ready = $false } })
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  closesReleaseIssue = $false
  boundary = "Final release close approval real input from Owner result is blocked until real Owner approval and accepted post-publish proof exist. It is not proof by default, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "final-release-close-approval-real-input-from-owner-result.json"
$markdownPath = Join-Path $ctx.OutputDirectory "final-release-close-approval-real-input-from-owner-result.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Final Release Close Approval Real Input From Owner Result",
  "",
  "- importState: ``$($record.importState)``",
  "- readyCloseApprovalCount: ``$($record.readyCloseApprovalCount)``",
  "- blockedApprovalFieldCount: ``$($record.blockedApprovalFieldCount)``",
  "- canCloseReleaseIssue: ``$($record.canCloseReleaseIssue)``",
  "- isReleaseReady: ``$($record.isReleaseReady)``",
  "",
  "> $($record.boundary)"
)

Write-Host "ImportState=$($record.importState)"
Write-Host "ReadyCloseApprovalCount=$($record.readyCloseApprovalCount)"
