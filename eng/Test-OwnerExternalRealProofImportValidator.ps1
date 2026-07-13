[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerExternalRealProofReadinessGate.Common.ps1')
Test-OwnerExternalRealProofGateArtifact -ArtifactId 'owner-external-real-proof-import-validator' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
