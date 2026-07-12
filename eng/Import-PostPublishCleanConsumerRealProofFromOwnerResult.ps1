[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory,
  [string]$CandidatePath
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPublicPublishExecutionResultCommon.ps1")

$ctx = Initialize-OwnerPublicPublishContext -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
if ([string]::IsNullOrWhiteSpace($CandidatePath)) { $CandidatePath = Join-Path $ctx.OutputDirectory "owner-public-publish-execution-result-candidate.json" }
if (-not (Test-Path -LiteralPath $CandidatePath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Import-OwnerPublicPublishExecutionResultCandidate.ps1") -RepositoryRoot $ctx.RepositoryRoot -OutputDirectory $ctx.OutputDirectory
}

$candidate = Get-Content -LiteralPath $CandidatePath -Raw -Encoding utf8 | ConvertFrom-Json
$candidateItemCount = [int](Get-OwnerPropertyOrDefault -Object $candidate -Name "candidateItemCount" -DefaultValue 0)
$proofLanes = @(
  [pscustomobject]@{ id = "clean-consumer-restore"; requiredFields = @("cleanConsumerRestoreStdoutPath", "cleanConsumerRestoreStderrPath", "cleanConsumerRestoreMergedTranscriptPath", "cleanConsumerRestoreExitCode"); ready = $false },
  [pscustomobject]@{ id = "clean-consumer-build"; requiredFields = @("cleanConsumerBuildStdoutPath", "cleanConsumerBuildStderrPath", "cleanConsumerBuildMergedTranscriptPath", "cleanConsumerBuildExitCode"); ready = $false },
  [pscustomobject]@{ id = "clean-consumer-runtime-smoke"; requiredFields = @("cleanConsumerRuntimeSmokeStdoutPath", "cleanConsumerRuntimeSmokeStderrPath", "cleanConsumerRuntimeSmokeMergedTranscriptPath", "cleanConsumerRuntimeSmokeExitCode"); ready = $false }
)

$record = [ordered]@{
  recordKind = "post-publish-clean-consumer-real-proof-from-owner-result"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = "blocked-post-publish-clean-consumer-real-owner-proof-required"
  sourceCandidatePath = $CandidatePath
  sourceCandidateItemCount = $candidateItemCount
  proofLaneCount = $proofLanes.Count
  readyProofCount = 0
  blockedProofLaneCount = $proofLanes.Count
  failedBlockerCount = 0
  failedActionRequiredCount = $proofLanes.Count
  proofLanes = @($proofLanes)
  missingOwnerEvidence = @("public package URL/hash", "clean external consumer restore log/hash", "clean external consumer build log/hash", "clean external consumer runtime smoke log/hash", "strict validator output/hash", "host identity", "package identity")
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isReleaseReady = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Post-publish clean consumer real proof import from Owner result is blocked until real public-channel clean consumer evidence exists. It is not proof by default, not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $ctx.OutputDirectory "post-publish-clean-consumer-real-proof-from-owner-result.json"
$markdownPath = Join-Path $ctx.OutputDirectory "post-publish-clean-consumer-real-proof-from-owner-result.md"
Write-OwnerUtf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 32)
Write-OwnerUtf8File -LiteralPath $markdownPath -InputObject @(
  "# Post-Publish Clean Consumer Real Proof From Owner Result",
  "",
  "- importState: ``$($record.importState)``",
  "- proofLaneCount: ``$($record.proofLaneCount)``",
  "- readyProofCount: ``$($record.readyProofCount)``",
  "- blockedProofLaneCount: ``$($record.blockedProofLaneCount)``",
  "- isPostPublishProof: ``$($record.isPostPublishProof)``",
  "",
  "> $($record.boundary)"
)

Write-Host "ImportState=$($record.importState)"
Write-Host "ReadyProofCount=$($record.readyProofCount)"
