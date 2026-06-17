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

function Get-DeliveryLane {
  param(
    [object]$Package
  )

  if ($Package.platform -eq "linux") {
    if ($Package.architecture -ne "x64") {
      return "hold-linux-validation"
    }

    if ($Package.runnerMode -eq "self-hosted") {
      return "hold-linux-validation"
    }
  }

  switch ($Package.distributionTier) {
    "public-sample" { return "public-preview" }
    "private-feed" { return "private-feed" }
    "split-delivery-candidate" { return "split-delivery-design" }
    default { return "manual-review" }
  }
}

function Get-DeliveryRecommendation {
  param(
    [object]$Package,
    [string]$Lane
  )

  switch ($Lane) {
    "public-preview" {
      return "Eligible as a first-wave public validation sample after NVIDIA redistribution terms are reviewed."
    }
    "private-feed" {
      return "Prefer controlled internal feed until validation evidence, size policy, and redistribution terms are finalized."
    }
    "split-delivery-design" {
      return "Keep out of broad public NuGet publication until bridge, CudaCudnn, TensorRt, and collection packages are validated."
    }
    "hold-linux-validation" {
      return "Do not publish until a dedicated runner/dependency line validates build, pack, and package consumer restore."
    }
    default {
      return "Review manually before release."
    }
  }
}

function Get-SplitDeliveryCandidates {
  param(
    [object]$Package
  )

  $core = New-Object System.Collections.Generic.List[string]
  $optional = New-Object System.Collections.Generic.List[string]

  $core.Add($Package.bridgeFile)
  foreach ($file in @($Package.cudaFiles)) {
    $core.Add($file)
  }

  foreach ($file in @($Package.tensorRtFiles)) {
    if ($file -match "builder_resource|plugin|parser|parsers") {
      $optional.Add($file)
    }
    else {
      $core.Add($file)
    }
  }

  return [pscustomobject]@{
    coreAssets = @($core)
    optionalOrSplitAssets = @($optional)
  }
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$packages = @($manifest.packages | Sort-Object platform, tensorRtLine, cudaLine, key)
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitPackages = @()
if (Test-Path -LiteralPath $splitManifestPath -PathType Leaf) {
  $splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
  $splitPackages = @($splitManifest.packages | Sort-Object sourceRuntimeKey, role, key)
}

$results = New-Object System.Collections.Generic.List[object]
foreach ($package in $packages) {
  $lane = Get-DeliveryLane -Package $package
  $split = Get-SplitDeliveryCandidates -Package $package
  $results.Add([pscustomobject]@{
    key = $package.key
    packageId = $package.packageId
    platform = $package.platform
    rid = $package.rid
    tensorRtLine = $package.tensorRtLine
    cudaLine = $package.cudaLine
    distributionTier = $package.distributionTier
    validationState = $package.validationState
    deliveryLane = $lane
    recommendation = Get-DeliveryRecommendation -Package $package -Lane $lane
    licenseBlocker = "NVIDIA TensorRT/CUDA redistribution terms must be reviewed before any public package publication."
    coreAssets = @($split.coreAssets)
    optionalOrSplitAssets = @($split.optionalOrSplitAssets)
  })
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-distribution"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "runtime-delivery-strategy.json"
$results | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Runtime Delivery Strategy")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("## Delivery lanes")
$lines.Add("")
$lines.Add("- public-preview: small enough and locally validated enough to use as a public validation sample after license review.")
$lines.Add("- private-feed: suitable for controlled internal feeds while validation, size, or license constraints remain unresolved.")
$lines.Add("- split-delivery-design: too large or broad for a single default public package; split into bridge, CudaCudnn, TensorRt, and collection packages first.")
$lines.Add("- hold-linux-validation: Linux lines that need a dedicated runner/dependency strategy before publication, such as future ARM/Jetson targets.")
$lines.Add("")
$lines.Add("| Key | Tier | Validation | Delivery lane | Recommendation |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($result in $results) {
  $lines.Add("| $($result.key) | $($result.distributionTier) | $($result.validationState) | $($result.deliveryLane) | $($result.recommendation) |")
}

$lines.Add("")
$lines.Add("## Split-delivery candidates")
$lines.Add("")
foreach ($result in @($results | Where-Object { $_.deliveryLane -eq "split-delivery-design" })) {
  $lines.Add("### $($result.key)")
  $lines.Add("")
  $lines.Add("Core/default candidate assets:")
  foreach ($asset in $result.coreAssets) {
    $lines.Add("- $asset")
  }
  $lines.Add("")
  $lines.Add("Optional/private split candidate assets:")
  if ($result.optionalOrSplitAssets.Count -eq 0) {
    $lines.Add("- none")
  }
  else {
    foreach ($asset in $result.optionalOrSplitAssets) {
      $lines.Add("- $asset")
    }
  }
  $lines.Add("")
}

if ($splitPackages.Count -gt 0) {
  $lines.Add("## Split runtime component packages")
  $lines.Add("")
  $lines.Add("| Split key | Source runtime | Role | Package ID | Assets | State |")
  $lines.Add("| --- | --- | --- | --- | ---: | --- |")
  foreach ($splitPackage in $splitPackages) {
    $lines.Add("| $($splitPackage.key) | $($splitPackage.sourceRuntimeKey) | $($splitPackage.role) | $($splitPackage.packageId) | $(@($splitPackage.assets).Count) | $($splitPackage.prototypeState) |")
  }
  $lines.Add("")
  $lines.Add("Component package naming rule:")
  $lines.Add("")
  $lines.Add("- `<full-runtime-package-id>.Bridge` carries only the local C ABI bridge and may be republished when wrapper native code changes.")
  $lines.Add("- `<full-runtime-package-id>.CudaCudnn` carries CUDA runtime, cuDNN, and related shared assets and should be republished only when the CUDA/cuDNN dependency set changes.")
  $lines.Add("- `<full-runtime-package-id>.TensorRt` carries TensorRT runtime, parser, plugin, and builder-resource assets and should be republished only when the TensorRT dependency set changes.")
  $lines.Add("- The original `<full-runtime-package-id>` remains a lightweight collection package that pins a tested component-version combination.")
  $lines.Add("")
}

$lines.Add("## Release blockers")
$lines.Add("")
$lines.Add("- NVIDIA TensorRT/CUDA redistribution terms are still a blocker before public release.")
$lines.Add("- Linux Ubuntu 22.04 x64 is the hosted publication line once build, pack, package consumer validation, and release/package upload pass.")
$lines.Add("- Linux Ubuntu 20.04 x64, ARM/SBSA, Jetson/L4T, and non-Ubuntu targets remain blocked until their dedicated runner and NVIDIA dependency strategy are validated.")
$lines.Add("- Large Windows runtime component packages should stay on GitHub Packages or GitHub Releases unless their package size fits nuget.org and NVIDIA redistribution terms are cleared.")

$markdownPath = Join-Path $outputRoot "runtime-delivery-strategy.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Runtime delivery strategy written to $jsonPath"
Write-Host "Runtime delivery strategy written to $markdownPath"
