[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot

$validatorScripts = @(
  @("Export-PublicPackageUrlHashProofValidator.ps1", "Test-PublicPackageUrlHashProofValidator.ps1"),
  @("Export-ExternalCleanConsumerPostPublishProofValidator.ps1", "Test-ExternalCleanConsumerPostPublishProofValidator.ps1"),
  @("Export-YoloVisionRealModelPostPublishProofValidator.ps1", "Test-YoloVisionRealModelPostPublishProofValidator.ps1"),
  @("Export-ArticlePublicationProofValidator.ps1", "Test-ArticlePublicationProofValidator.ps1"),
  @("Export-ReleaseCloseFinalBridgeProofValidator.ps1", "Test-ReleaseCloseFinalBridgeProofValidator.ps1")
)

foreach ($pair in $validatorScripts) {
  & (Join-Path $RepositoryRoot "eng\$($pair[0])") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
  & (Join-Path $RepositoryRoot "eng\$($pair[1])") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot -Strict
}

$record = Export-OwnerPostPublishProofAcceptanceManifestArtifact -OutputRoot $OutputRoot -RepositoryRoot $RepositoryRoot
Write-Host "OwnerPostPublishProofAcceptanceManifestState=$($record.manifestState) Validators=$($record.acceptedValidatorCount)/$($record.validatorCount) ReleaseCloseReady=$($record.releaseCloseReady)"
