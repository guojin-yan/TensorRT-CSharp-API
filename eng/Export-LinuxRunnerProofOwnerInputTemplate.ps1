[CmdletBinding()]
param(
  [string]$OutputRoot = 'artifacts\final-release\owner-real-inputs',
  [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FinalOwnerRealInput.Common.ps1')
New-FinalOwnerRealInputTemplate -LaneId 'linux-runner-proof' -OutputRoot $OutputRoot -RepositoryRoot $RepositoryRoot
