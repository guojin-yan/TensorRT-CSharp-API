[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
New-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'public-package-hash-cross-check-gate' -OutputDirectory $OutputDirectory
