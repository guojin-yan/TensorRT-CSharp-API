[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\owner-post-publish-docs-article-sample-real-input-import.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$record = Export-OwnerPostPublishProofValidatorArtifact -ImportPath $ImportPath -OutputRoot $ctx.OutputRoot -RepositoryRoot $ctx.RepositoryRoot -RecordKind "release-close-final-bridge-proof-validator"
Write-Host "ReleaseCloseFinalBridgeProofValidatorState=$($record.validatorState) OwnerEvidenceAccepted=$($record.ownerEvidenceAccepted) Dependencies=$($record.dependencyAcceptedCount)/$($record.dependencyRequiredCount) BlockedReasons=$($record.blockedReasonCount)"
