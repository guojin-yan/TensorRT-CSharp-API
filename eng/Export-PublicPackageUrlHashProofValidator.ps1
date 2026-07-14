[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input-import.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$record = Export-OwnerPostPublishProofValidatorArtifact -ImportPath $ImportPath -OutputRoot $ctx.OutputRoot -RepositoryRoot $ctx.RepositoryRoot -RecordKind "public-package-url-hash-proof-validator"
Write-Host "PublicPackageUrlHashProofValidatorState=$($record.validatorState) OwnerEvidenceAccepted=$($record.ownerEvidenceAccepted) BlockedReasons=$($record.blockedReasonCount)"
