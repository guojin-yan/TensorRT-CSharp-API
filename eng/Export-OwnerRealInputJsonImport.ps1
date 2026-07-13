[CmdletBinding()]
param(
  [string]$OwnerInputPath,
  [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\artifacts\final-release')
)

. (Join-Path $PSScriptRoot 'StrictCloseRealInputValidation.Common.ps1')
New-StrictCloseRealInputValidationArtifact -ArtifactId 'owner-real-input-json-import' -OwnerInputPath $OwnerInputPath -OutputDirectory $OutputDirectory
