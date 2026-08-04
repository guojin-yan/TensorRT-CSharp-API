[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$ManagedPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$PackageVersion = "4.0.0",
  [ValidateSet("8", "10", "11")][string]$TensorRtLine = "10",
  [string]$TensorRtRoot,
  [string]$CudaRoot,
  [switch]$KeepWorkspace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$entrypoint = Join-Path $PSScriptRoot "Test-CallbackOwnerLocalPackageConsumer.ps1"
& $entrypoint -Scenario DebugListener @PSBoundParameters
