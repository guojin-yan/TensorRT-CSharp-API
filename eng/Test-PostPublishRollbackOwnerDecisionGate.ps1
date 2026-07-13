[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'ReleaseCandidateFinalRealInputAdmission.Common.ps1')
Test-ReleaseCandidateFinalRealInputAdmissionArtifact -ArtifactId 'post-publish-rollback-owner-decision-gate' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
