[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerExternalRealProofReadinessGate.Common.ps1')
New-OwnerExternalRealProofGateArtifact -ArtifactId 'release-close-real-proof-readiness-gate' -OutputDirectory $OutputDirectory
