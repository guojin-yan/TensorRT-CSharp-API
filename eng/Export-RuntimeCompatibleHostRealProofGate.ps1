[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerExternalRealProofReadinessGate.Common.ps1')
New-OwnerExternalRealProofGateArtifact -ArtifactId 'runtime-compatible-host-real-proof-gate' -OutputDirectory $OutputDirectory
