[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
Test-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'release-candidate-real-proof-final-freeze' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
