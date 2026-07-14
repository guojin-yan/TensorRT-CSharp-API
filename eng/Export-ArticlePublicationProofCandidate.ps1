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

$record = Export-OwnerPostPublishLaneCandidateArtifact `
  -ImportPath $ImportPath `
  -OutputRoot $OutputRoot `
  -RepositoryRoot $RepositoryRoot `
  -LaneId "article-publication-urls" `
  -RecordKind "article-publication-proof-candidate" `
  -FileStem "article-publication-proof-candidate" `
  -CandidateState "blocked-article-publication-owner-proof-required" `
  -Title "Article Publication Proof Candidate" `
  -Boundary "Article publication proof candidate checks public article URL, publish timestamp, screenshot hash, linked package URL, and linked proof hash only; it does not publish articles, does not publish packages, is not post-publish proof, not runtime proof, not release close approval, and not package push." `
  -ExtraProperties ([pscustomobject]@{ articleRoadmapIsNotProof = $true })

Write-Host "ArticlePublicationProofCandidateState=$($record.candidateState) ReadyFields=$($record.readyFieldCount)/$($record.requiredFieldCount)"
