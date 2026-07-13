[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
New-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'release-close-final-real-input-admission-pack' -OutputDirectory $OutputDirectory
