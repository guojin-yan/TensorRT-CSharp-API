[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerExternalRealProofReadinessGate.Common.ps1')
Test-OwnerExternalRealProofGateArtifact -ArtifactId 'post-publish-clean-consumer-real-proof-gate' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
