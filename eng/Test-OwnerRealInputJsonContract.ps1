[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'StrictCloseRealInputValidation.Common.ps1')
Test-StrictCloseRealInputValidationArtifact -ArtifactId 'owner-real-input-json-contract' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
