[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerExternalRealProofReadinessGate.Common.ps1')
New-OwnerExternalRealProofGateArtifact -ArtifactId 'post-publish-clean-consumer-real-proof-gate' -OutputDirectory $OutputDirectory
