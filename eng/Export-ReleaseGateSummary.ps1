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
$splitPrototypePath = Join-Path $outputRoot "split-delivery-prototype-report.md"
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
foreach ($path in @($distributionPath, $deliveryStrategyPath, $splitPrototypePath, $publicPreviewPath, $splitDeliveryPath, $workflowContractPath)) {
  if (Test-Path -LiteralPath $path) {
    $lines.Add("- $(Split-Path $path -Leaf)")
  }
}
$lines.Add("")
$lines.Add("## Current guidance")
$lines.Add("")
$lines.Add("- `win-x64-trt8.6-cuda11.8-cudnn8.9` remains the primary public-preview sample candidate.")
$lines.Add("- `win-x64-trt10.11-cuda11.8-cudnn8.9` remains the primary split-delivery candidate.")
$lines.Add("- Linux combinations remain dry-run-only until a real Linux x64 runner validates build and pack.")
$lines.Add("- Linux package consumer validation is now part of the self-hosted Linux runtime pack workflow, but it still requires a real Linux runner.")
$lines.Add("- Hosted quality gates must not imply runtime package readiness unless runtime package checks are explicitly enabled.")
$lines.Add("- TensorRT 10 split-delivery packages are design-only prototypes until split consumer validation and redistribution review are complete.")

$summaryPath = Join-Path $outputRoot "release-gate-summary.md"
$lines | Set-Content -LiteralPath $summaryPath -Encoding utf8

Write-Host "Release gate summary written to $summaryPath"
