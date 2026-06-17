[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-distribution"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$distributionPath = Join-Path $outputRoot "runtime-distribution-report.md"
$deliveryStrategyPath = Join-Path $outputRoot "runtime-delivery-strategy.md"
$splitRuntimePackagesPath = Join-Path $outputRoot "split-runtime-packages-report.md"
$publicPreviewPath = Join-Path $outputRoot "publish-readiness-public-preview.md"
$splitDeliveryPath = Join-Path $outputRoot "publish-readiness-split-delivery.md"
$workflowContractPath = Join-Path $RepositoryRoot "artifacts\workflow-contracts\workflow-contract-report.md"

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Gate Summary")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("## Available reports")
$lines.Add("")
foreach ($path in @($distributionPath, $deliveryStrategyPath, $splitRuntimePackagesPath, $publicPreviewPath, $splitDeliveryPath, $workflowContractPath)) {
  if (Test-Path -LiteralPath $path) {
    $lines.Add("- $(Split-Path $path -Leaf)")
  }
}
$lines.Add("")
$lines.Add("## Current guidance")
$lines.Add("")
$lines.Add("- `win-x64-trt8.6-cuda11.8-cudnn8.9` remains the primary public-preview sample candidate.")
$lines.Add("- Large CUDA/cuDNN/TensorRT component packages should stay on GitHub Packages or GitHub Releases unless package size and NVIDIA redistribution terms are cleared for nuget.org.")
$lines.Add("- Managed, bridge, CUDA/cuDNN, TensorRT, and collection packages are versioned independently; routine C# releases should not republish stable NVIDIA component packages.")
$lines.Add("- `NUGET_API_KEY` must be an active plain-text nuget.org key with push permission for `JYPPX.TensorRT.CSharp.API` or its owning account/organization; a nuget.org `403` means the secret needs to be replaced before rerunning managed-only publication.")
$lines.Add("- Ubuntu 20.04, Ubuntu 22.04, and Ubuntu 24.04 x64 Linux packages are published through GitHub-hosted runners with distro-matched Ubuntu job containers; ARM/Jetson still need separate package lines.")
$lines.Add("- Linux package consumer validation is part of the runtime pack workflow and must pass for each published Linux runtime key.")
$lines.Add("- Hosted quality gates must not imply runtime package readiness unless runtime package checks are explicitly enabled.")

$summaryPath = Join-Path $outputRoot "release-gate-summary.md"
$lines | Set-Content -LiteralPath $summaryPath -Encoding utf8

Write-Host "Release gate summary written to $summaryPath"
