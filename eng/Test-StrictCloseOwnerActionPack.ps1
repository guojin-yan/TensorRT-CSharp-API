[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'StrictCloseRealInputValidation.Common.ps1')
Test-StrictCloseRealInputValidationArtifact -ArtifactId 'strict-close-owner-action-pack' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
