[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$ArtifactDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'StrictCloseRealInputValidation.Common.ps1')
Test-StrictCloseRealInputValidationArtifact -ArtifactId 'release-close-real-input-final-blocker-ledger' -Strict:$Strict -ArtifactDirectory $ArtifactDirectory
