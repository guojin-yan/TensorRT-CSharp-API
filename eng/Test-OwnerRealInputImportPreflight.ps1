[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
Test-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'owner-real-input-import-preflight' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
