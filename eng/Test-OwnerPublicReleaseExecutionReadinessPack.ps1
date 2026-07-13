[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'OwnerPublicReleaseExecutionKit.Common.ps1')
Test-OwnerPublicReleaseExecutionArtifact -ArtifactId 'owner-public-release-execution-readiness-pack' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
