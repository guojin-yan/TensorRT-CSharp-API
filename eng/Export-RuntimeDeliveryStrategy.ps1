[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$runtimeManifestPath = Join-Path $RepositoryRoot 'pack\runtime\runtime-packages.manifest.json'
$splitManifestPath = Join-Path $RepositoryRoot 'pack\runtime-split\split-runtime-packages.manifest.json'
$policyPath = Join-Path $RepositoryRoot 'pack\external-vendor-runtime-policy.json'
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding utf8 | ConvertFrom-Json

if ([string]$splitManifest.publicationPolicy.state -ne 'bridge-only') {
  throw 'split-runtime-packages.manifest.json must declare publicationPolicy.state=bridge-only.'
}
if (@($policy.allowedPackageKinds) -join ',' -ne 'managed,bridge') {
  throw 'External vendor runtime policy must allow only managed and bridge package kinds.'
}

$historicalSplitPackages = @($splitManifest.packages)
$results = [System.Collections.Generic.List[object]]::new()
foreach ($package in @($runtimeManifest.packages | Sort-Object platform, tensorRtLine, cudaLine, key)) {
  $bridgeEntry = @(
    $historicalSplitPackages |
      Where-Object { [string]$_.sourceRuntimeKey -eq [string]$package.key -and [string]$_.role -eq 'bridge' }
  ) | Select-Object -First 1
  $bridgePackageId = if ($null -ne $bridgeEntry) {
    [string]$bridgeEntry.packageId
  }
  else {
    "$([string]$package.packageId).Bridge"
  }

  $results.Add([ordered]@{
      key = [string]$package.key
      packageId = $bridgePackageId
      packageKind = 'bridge'
      platform = [string]$package.platform
      rid = [string]$package.rid
      tensorRtLine = [string]$package.tensorRtLine
      cudaLine = [string]$package.cudaLine
      validationState = [string]$package.validationState
      deliveryLane = 'managed-plus-bridge-only'
      publishedAssets = @([string]$package.bridgeFile)
      vendorDependenciesSource = 'consumer-host-installation'
      vendorDependencyPatterns = @(
        @($package.tensorRtFiles) +
        @($package.cudaFiles) +
        @($package.cudnnFiles)
      )
      recommendation = 'Publish the project-owned bridge only. Consumers install matching NVIDIA dependencies.'
      vendorPackageAllowed = $false
    })
}

$outputRoot = Join-Path $RepositoryRoot 'artifacts\runtime-distribution'
[void][System.IO.Directory]::CreateDirectory($outputRoot)
$jsonPath = Join-Path $outputRoot 'runtime-delivery-strategy.json'
$json = [ordered]@{
  schemaVersion = 2
  recordKind = 'runtime-delivery-strategy'
  publicationPolicy = 'managed-plus-bridge-only'
  policyPath = 'pack/external-vendor-runtime-policy.json'
  vendorLibrariesAreExternalDependencies = $true
  performsPackaging = $false
  performsPublish = $false
  packages = @($results)
}
[System.IO.File]::WriteAllText($jsonPath, ($json | ConvertTo-Json -Depth 10), $utf8)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# Runtime Delivery Strategy')
$lines.Add('')
$lines.Add('Only managed C# packages and project-owned `.Bridge` packages are publishable. CUDA, cuDNN, TensorRT, and optional NVRTC are consumer-installed host dependencies.')
$lines.Add('')
$lines.Add('- Policy: `pack/external-vendor-runtime-policy.json`')
$lines.Add('- Publication lane: `managed-plus-bridge-only`')
$lines.Add('- Vendor package allowed: `false`')
$lines.Add('- Performs packaging/publish: `false/false`')
$lines.Add('')
$lines.Add('| Runtime key | Bridge package | Platform | Validation | Vendor source |')
$lines.Add('| --- | --- | --- | --- | --- |')
foreach ($result in $results) {
  $lines.Add("| $($result.key) | $($result.packageId) | $($result.platform) | $($result.validationState) | consumer host installation |")
}
$lines.Add('')
$lines.Add('Historical non-bridge manifest entries remain only for remote cleanup and evidence interpretation. They must never be packed, pushed, or uploaded.')

$markdownPath = Join-Path $outputRoot 'runtime-delivery-strategy.md'
[System.IO.File]::WriteAllText($markdownPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $utf8)

Write-Host "Runtime delivery strategy written to $jsonPath"
Write-Host "Runtime delivery strategy written to $markdownPath"
