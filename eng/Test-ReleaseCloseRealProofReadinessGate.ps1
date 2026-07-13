[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerExternalRealProofReadinessGate.Common.ps1')
Test-OwnerExternalRealProofGateArtifact -ArtifactId 'release-close-real-proof-readiness-gate' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
