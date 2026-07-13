[CmdletBinding()]
param(
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'StrictCloseRealInputValidation.Common.ps1')
New-StrictCloseRealInputValidationArtifact -ArtifactId 'owner-real-input-json-contract' -OutputDirectory $OutputDirectory
