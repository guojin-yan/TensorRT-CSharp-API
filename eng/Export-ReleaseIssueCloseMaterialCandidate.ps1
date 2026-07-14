[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input-import.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

$dependencies = @(
  "public-package-url-hash-verification-candidate",
  "external-clean-consumer-post-publish-candidate",
  "yolovision-real-model-post-publish-candidate",
  "article-publication-proof-candidate"
)
$extra = [pscustomobject]@{
  dependencyCandidateIds = @($dependencies)
  dependencyCandidateCount = $dependencies.Count
  finalBridgeRequired = $true
  finalBridgePassed = $false
}

$record = Export-OwnerPostPublishLaneCandidateArtifact `
  -ImportPath $ImportPath `
  -OutputRoot $OutputRoot `
  -RepositoryRoot $RepositoryRoot `
  -LaneId "release-issue-close-material" `
  -RecordKind "release-issue-close-material-candidate" `
  -FileStem "release-issue-close-material-candidate" `
  -CandidateState "blocked-release-issue-close-material-owner-proof-required" `
  -Title "Release Issue Close Material Candidate" `
  -Boundary "Release Issue close material candidate summarizes public package, clean consumer, YoloVision, article publication, Owner close decision, rollback decision, and evidence bundle hashes only; final bridge remains required, it cannot close the release issue, is not release close approval, not post-publish proof, not runtime proof, and not package push." `
  -ExtraProperties $extra

Write-Host "ReleaseIssueCloseMaterialCandidateState=$($record.candidateState) FinalBridgePassed=$($record.finalBridgePassed) ReadyFields=$($record.readyFieldCount)/$($record.requiredFieldCount)"
