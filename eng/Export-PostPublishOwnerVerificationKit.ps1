[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerPublicReleaseExecutionKit.Common.ps1')
New-OwnerPublicReleaseExecutionArtifact -ArtifactId 'post-publish-owner-verification-kit' -OutputDirectory $OutputDirectory
