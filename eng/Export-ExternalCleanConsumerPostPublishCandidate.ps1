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
  -LaneId "external-clean-consumer-logs" `
  -RecordKind "external-clean-consumer-post-publish-candidate" `
  -FileStem "external-clean-consumer-post-publish-candidate" `
  -CandidateState "blocked-external-clean-consumer-post-publish-owner-proof-required" `
  -Title "External Clean Consumer Post-Publish Candidate" `
  -Boundary "External clean consumer post-publish candidate checks repository-external restore/build/run logs, host metadata, and no-local-substitute confirmation only; it does not run a consumer, does not publish, is not clean consumer proof, not runtime proof, not post-publish proof, not release close approval, and not package push." `
  -ExtraProperties ([pscustomobject]@{ requiresRepositoryExternalWorkspace = $true })

Write-Host "ExternalCleanConsumerPostPublishCandidateState=$($record.candidateState) ReadyFields=$($record.readyFieldCount)/$($record.requiredFieldCount)"
