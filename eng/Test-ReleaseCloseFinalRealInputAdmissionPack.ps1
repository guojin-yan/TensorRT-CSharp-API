[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
Test-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'release-close-final-real-input-admission-pack' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
