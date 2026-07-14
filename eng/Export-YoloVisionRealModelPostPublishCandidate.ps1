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
  -LaneId "yolovision-real-model-assets" `
  -RecordKind "yolovision-real-model-post-publish-candidate" `
  -FileStem "yolovision-real-model-post-publish-candidate" `
  -CandidateState "blocked-yolovision-real-model-owner-proof-required" `
  -Title "YoloVision Real Model Post-Publish Candidate" `
  -Boundary "YoloVision real model post-publish candidate checks real model, labels, input, output JSON, stdout/stderr, host metadata, and Owner review hashes only; it does not run YoloVision, does not publish, is not real-model-runtime proof, not runtime proof, not post-publish proof, not release close approval, and not package push." `
  -ExtraProperties ([pscustomobject]@{ requiredTasks = @("det", "cls", "seg", "obb", "pose", "sem"); matrixIsForbiddenSubstitute = $true })

Write-Host "YoloVisionRealModelPostPublishCandidateState=$($record.candidateState) ReadyFields=$($record.readyFieldCount)/$($record.requiredFieldCount)"
