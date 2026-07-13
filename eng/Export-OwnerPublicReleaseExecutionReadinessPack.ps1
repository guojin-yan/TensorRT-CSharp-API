[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerPublicReleaseExecutionKit.Common.ps1')
New-OwnerPublicReleaseExecutionArtifact -ArtifactId 'owner-public-release-execution-readiness-pack' -OutputDirectory $OutputDirectory
