[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
New-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'post-publish-rollback-owner-decision-gate' -OutputDirectory $OutputDirectory
