[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-issue-close-material-candidate.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseIssueCloseMaterialCandidate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = @(Test-OwnerPostPublishLaneCandidateArtifact -Record $record -RecordKind "release-issue-close-material-candidate" -LaneId "release-issue-close-material" -ExpectedBlockedState "blocked-release-issue-close-material-owner-proof-required" -RequiredBoundaryText "not release close approval")
$items += (New-OwnerValidationItem "dependency-candidates" ([int](Get-PropertyOrDefault -Object $record -Name "dependencyCandidateCount" -DefaultValue 0) -ge 4) "blocker" "Close material candidate must reference package, clean consumer, YoloVision, and article candidates.")
$items += (New-OwnerValidationItem "final-bridge-blocked" ([bool](Get-PropertyOrDefault -Object $record -Name "finalBridgeRequired" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "finalBridgePassed" -DefaultValue $true) -and -not [bool]$record.canCloseReleaseIssue) "blocker" "Final bridge must remain blocked and canCloseReleaseIssue=false.")
$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "release-issue-close-material-candidate-ready-non-proof" } else { "invalid-release-issue-close-material-candidate" }
$validation = [pscustomobject]@{
  recordKind = "release-issue-close-material-candidate-validation"; generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O"); validationState = $state; candidateState = [string]$record.candidateState; requiredFieldCount = [int]$record.requiredFieldCount; blockedFieldCount = [int]$record.blockedFieldCount; dependencyCandidateCount = [int](Get-PropertyOrDefault -Object $record -Name "dependencyCandidateCount" -DefaultValue 0); finalBridgePassed = [bool](Get-PropertyOrDefault -Object $record -Name "finalBridgePassed" -DefaultValue $false); failedBlockerCount = $failedBlockers.Count; validationItems = @($items); ownerActionRequired = $true; performsPublish = $false; usesPublishToken = $false; canPublishPublicly = $false; canCloseReleaseIssue = $false; canPromoteRuntimeProof = $false; isRuntimeExecutionProof = $false; isPostPublishProof = $false; isReleaseCloseProof = $false; boundary = "Release Issue close material candidate validation is non-proof; not release close approval, not post-publish proof, not runtime proof, not publish approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "release-issue-close-material-candidate-validation.json"; $mdPath = Join-Path $OutputRoot "release-issue-close-material-candidate-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Release Issue Close Material Candidate Validation", "", "- validationState: ``$state``", "- requiredFieldCount: ``$($validation.requiredFieldCount)``", "- blockedFieldCount: ``$($validation.blockedFieldCount)``", "- dependencyCandidateCount: ``$($validation.dependencyCandidateCount)``", "- finalBridgePassed: ``False``", "- failedBlockerCount: ``$($validation.failedBlockerCount)``", "", $validation.boundary)
Write-Host "ReleaseIssueCloseMaterialCandidateValidationState=$state FailedBlockers=$($failedBlockers.Count)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Release Issue close material candidate validation failed." }
