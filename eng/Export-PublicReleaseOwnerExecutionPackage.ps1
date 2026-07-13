[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerPublicReleaseExecutionKit.Common.ps1')
New-OwnerPublicReleaseExecutionArtifact -ArtifactId 'public-release-owner-execution-package' -OutputDirectory $OutputDirectory
