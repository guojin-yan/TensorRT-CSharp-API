[CmdletBinding()]
param(
  [string]$OutputRoot = 'artifacts\final-release\owner-real-inputs',
  [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FinalOwnerRealInput.Common.ps1')
New-FinalOwnerRealInputTemplate -LaneId 'real-model-runtime' -OutputRoot $OutputRoot -RepositoryRoot $RepositoryRoot
