[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$ReportDirectory,
  [string]$ManagedPackageDirectory,
  [string]$YoloVisionPackageDirectory,
  [string]$BridgePackageDirectory,
  [string]$RuntimePackageKey = "win-x64-trt10.11-cuda12.9-cudnn9.22",
  [string]$BridgePackageId,
  [ValidateSet("8", "10", "11")][string]$TensorRtLine,
  [string]$ModelPath,
  [string]$ModelWeightsPath,
  [string]$LabelsPath,
  [string]$ImagePath,
  [string]$ReferenceOutput0Path,
  [string]$ReferenceClassIndexPath,
  [string]$PythonPath,
  [string]$TensorRtRoot,
  [string]$TensorRtRuntimeRoot,
  [string]$CudaRoot,
  [string]$CudnnRoot,
  [string]$PackageVersion = "4.0.0",
  [switch]$KeepWorkspace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$runnerPath = Join-Path $RepositoryRoot "eng\Test-YoloVisionLocalPackageConsumer.ps1"
$runnerParameters = @{}
foreach ($entry in $PSBoundParameters.GetEnumerator()) {
  $runnerParameters[$entry.Key] = $entry.Value
}
$runnerParameters["RepositoryRoot"] = $RepositoryRoot
$runnerParameters["Scenario"] = "torchvision-lraspp-semantic"

& $runnerPath @runnerParameters
