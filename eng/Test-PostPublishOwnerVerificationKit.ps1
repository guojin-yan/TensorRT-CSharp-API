[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerPublicReleaseExecutionKit.Common.ps1')
Test-OwnerPublicReleaseExecutionArtifact -ArtifactId 'post-publish-owner-verification-kit' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
