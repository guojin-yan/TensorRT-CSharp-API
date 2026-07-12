[CmdletBinding()]
param(
  [switch]$Strict,
  [string]$InputRoot = 'artifacts\final-release\owner-real-inputs',
  [string]$OutputRoot = 'artifacts\final-release',
  [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FinalOwnerRealInput.Common.ps1')
Test-FinalOwnerRealInputTemplate -LaneId 'post-publish-verification' -Strict:$Strict -InputRoot $InputRoot -OutputRoot $OutputRoot -RepositoryRoot $RepositoryRoot
