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

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-distribution"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$packages = @(
  $manifest.packages |
    Sort-Object platform, tensorRtLine, cudaLine, key
)

$jsonPath = Join-Path $outputRoot "runtime-distribution-report.json"
$packages | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Runtime Distribution Report")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("| Key | Platform | TRT | CUDA | Tier | Validation | Recommendation |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")

foreach ($package in $packages) {
  $recommendation = switch ($package.distributionTier) {
    "public-sample" { "Public validation sample" }
    "private-feed" { "Private feed or controlled distribution" }
    "split-delivery-candidate" { "Split delivery or private feed" }
    default { "Review manually" }
  }

  $lines.Add("| $($package.key) | $($package.platform) | $($package.tensorRtLine) | $($package.cudaLine) | $($package.distributionTier) | $($package.validationState) | $recommendation |")
}

$lines.Add("")
$lines.Add("## Suggested publication order")
$lines.Add("")

$publicationOrder = @(
  $packages |
    Sort-Object `
      @{ Expression = { if ($_.distributionTier -eq "public-sample") { 0 } elseif ($_.distributionTier -eq "private-feed") { 1 } else { 2 } } }, `
      @{ Expression = { if ($_.validationState -eq "local-validated") { 0 } else { 1 } } }, `
      key
)

$index = 1
foreach ($package in $publicationOrder) {
  $lines.Add("$index. $($package.key) - $($package.distributionTier) / $($package.validationState)")
  $lines.Add("   $($package.distributionNotes)")
  $index++
}

$lines.Add("")
$lines.Add("## Release blockers to review")
$lines.Add("")
$lines.Add("- NVIDIA redistribution terms for the selected package line")
$lines.Add("- Whether the current tier still matches real package size and validation state")
$lines.Add("- Whether Linux packages remain dry-run-only or have moved to real runner validation")
$lines.Add("- Whether TRT10 packages should remain split-delivery candidates")

$markdownPath = Join-Path $outputRoot "runtime-distribution-report.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime distribution report written to $outputRoot"
