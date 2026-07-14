[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\external-clean-consumer-post-publish-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
$resolvedInputPath = Resolve-OwnerPath -BaseRoot $RepositoryRoot -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-ExternalCleanConsumerPostPublishCandidate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishLaneCandidateArtifact -Record $record -RecordKind "external-clean-consumer-post-publish-candidate" -LaneId "external-clean-consumer-logs" -ExpectedBlockedState "blocked-external-clean-consumer-post-publish-owner-proof-required" -RequiredBoundaryText "not clean consumer proof")
$items += (New-OwnerValidationItem "external-workspace-required" ([bool](Get-PropertyOrDefault -Object $record -Name "requiresRepositoryExternalWorkspace" -DefaultValue $false)) "blocker" "Candidate must require a repository-external workspace.")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "external-clean-consumer-post-publish-candidate-ready-non-proof" } else { "invalid-external-clean-consumer-post-publish-candidate" }
$validation = [pscustomobject]@{
  recordKind = "external-clean-consumer-post-publish-candidate-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; candidateState = [string]$record.candidateState; requiredFieldCount = [int]$record.requiredFieldCount; blockedFieldCount = [int]$record.blockedFieldCount; failedBlockerCount = $failedBlockers.Count; validationItems = @($items); ownerActionRequired = $true; performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "External clean consumer post-publish candidate validation is non-proof; not clean consumer proof, not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "external-clean-consumer-post-publish-candidate-validation.json"; $mdPath = Join-Path $OutputRoot "external-clean-consumer-post-publish-candidate-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# External Clean Consumer Post-Publish Candidate Validation", "", "- validationState: ``$state``", "- requiredFieldCount: ``$($validation.requiredFieldCount)``", "- blockedFieldCount: ``$($validation.blockedFieldCount)``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "ExternalCleanConsumerPostPublishCandidateValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "External clean consumer post-publish candidate validation failed." }
