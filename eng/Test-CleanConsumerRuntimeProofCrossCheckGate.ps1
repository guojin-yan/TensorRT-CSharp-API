[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
Test-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'clean-consumer-runtime-proof-cross-check-gate' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
