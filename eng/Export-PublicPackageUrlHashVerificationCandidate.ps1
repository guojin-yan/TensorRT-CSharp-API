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
  -LaneId "public-package-urls-and-hashes" `
  -RecordKind "public-package-url-hash-verification-candidate" `
  -FileStem "public-package-url-hash-verification-candidate" `
  -CandidateState "blocked-public-package-url-hash-owner-proof-required" `
  -Title "Public Package URL Hash Verification Candidate" `
  -Boundary "Public package URL/hash candidate checks Owner supplied public package URL, package version, downloaded nupkg SHA256, and transcript hash only; it does not download packages, does not publish, is not public package proof, not runtime proof, not post-publish proof, not release close approval, and not package push." `
  -ExtraProperties ([pscustomobject]@{ expectedChannels = @("nuget-small-bridge-core", "github-packages-full-runtime") })

Write-Host "PublicPackageUrlHashVerificationCandidateState=$($record.candidateState) ReadyFields=$($record.readyFieldCount)/$($record.requiredFieldCount)"
