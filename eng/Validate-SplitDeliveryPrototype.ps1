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

function Get-ExpectedFullRuntimeAssetNames {
  param(
    [object]$Package
  )

  $files = @($Package.bridgeFile)
  foreach ($relativePath in @($Package.tensorRtFiles + $Package.cudaFiles + $Package.cudnnFiles)) {
    $files += [System.IO.Path]::GetFileName($relativePath)
  }

  return @($files | Sort-Object -Unique)
}

function Test-AssetMatchesPattern {
  param(
    [string]$Asset,
    [string]$Pattern
  )

  if ([string]::IsNullOrWhiteSpace($Asset) -or [string]::IsNullOrWhiteSpace($Pattern)) {
    return $false
  }

  return $Asset -like $Pattern
}

$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"

if (-not (Test-Path -LiteralPath $splitManifestPath -PathType Leaf)) {
  throw "Split delivery manifest was not found: $splitManifestPath"
}

$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$errors = New-Object System.Collections.Generic.List[string]
$rows = New-Object System.Collections.Generic.List[object]

foreach ($splitPackage in @($splitManifest.packages)) {
  foreach ($property in @("key", "sourceRuntimeKey", "packageId", "rid", "platform", "tensorRtLine", "cudaLine", "role", "prototypeState", "assets")) {
    if (-not $splitPackage.PSObject.Properties.Name.Contains($property) -or [string]::IsNullOrWhiteSpace([string]$splitPackage.$property)) {
      $errors.Add("Split package '$($splitPackage.key)' is missing required property '$property'.")
    }
  }

  if ($splitPackage.role -in @($null, "")) {
    $errors.Add("Split package '$($splitPackage.key)' must define a non-empty role.")
  }

  if ($splitPackage.prototypeState -notin @("design-only", "local-validated", "pending-local-validation")) {
    $errors.Add("Split package '$($splitPackage.key)' has unsupported prototypeState '$($splitPackage.prototypeState)'.")
  }

  $sourcePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $splitPackage.sourceRuntimeKey } | Select-Object -First 1
  if (-not $sourcePackage) {
    $errors.Add("Split package '$($splitPackage.key)' references missing source runtime '$($splitPackage.sourceRuntimeKey)'.")
    continue
  }

  if ($sourcePackage.distributionTier -notin @("public-sample", "split-delivery-candidate", "private-feed")) {
    $errors.Add("Split package '$($splitPackage.key)' source runtime '$($sourcePackage.key)' must be a public-sample, split-delivery, or private-feed candidate.")
  }

  if ($sourcePackage.rid -ne $splitPackage.rid) {
    $errors.Add("Split package '$($splitPackage.key)' rid '$($splitPackage.rid)' does not match source rid '$($sourcePackage.rid)'.")
  }

  if (-not $splitPackage.packageId.StartsWith($sourcePackage.packageId + ".", [System.StringComparison]::Ordinal)) {
    $errors.Add("Split package '$($splitPackage.key)' packageId should start with '$($sourcePackage.packageId).'.")
  }

  $projectPath = Join-Path $RepositoryRoot "pack\runtime-split\$($splitPackage.key)\$($splitPackage.packageId).csproj"
  if (-not (Test-Path -LiteralPath $projectPath -PathType Leaf)) {
    $errors.Add("Split package '$($splitPackage.key)' is missing project file: $projectPath")
  }

  $rows.Add([pscustomobject]@{
    key = $splitPackage.key
    sourceRuntimeKey = $splitPackage.sourceRuntimeKey
    role = $splitPackage.role
    packageId = $splitPackage.packageId
    assetCount = @($splitPackage.assets).Count
  })
}

foreach ($group in @($splitManifest.packages | Group-Object sourceRuntimeKey)) {
  $sourcePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $group.Name } | Select-Object -First 1
  if (-not $sourcePackage) {
    continue
  }

  $fullAssets = @(Get-ExpectedFullRuntimeAssetNames -Package $sourcePackage)
  $splitAssets = @($group.Group | ForEach-Object { $_.assets } | Sort-Object -Unique)
  $duplicateAssets = @($group.Group | ForEach-Object { $_.assets } | Group-Object | Where-Object { $_.Count -gt 1 } | Select-Object -ExpandProperty Name)
  foreach ($asset in $duplicateAssets) {
    $errors.Add("Source runtime '$($group.Name)' assigns asset '$asset' to more than one split package.")
  }

  foreach ($asset in $fullAssets) {
    if (-not ($splitAssets | Where-Object { Test-AssetMatchesPattern -Asset $_ -Pattern $asset })) {
      $errors.Add("Source runtime '$($group.Name)' full asset '$asset' is not assigned to any split package.")
    }
  }

  foreach ($asset in $splitAssets) {
    if (-not ($fullAssets | Where-Object { Test-AssetMatchesPattern -Asset $asset -Pattern $_ })) {
      $errors.Add("Source runtime '$($group.Name)' split asset '$asset' is not present in the full runtime asset set.")
    }
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-distribution"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "split-delivery-prototype-report.json"
$rows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Split Delivery Prototype Report")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("| Split key | Source runtime | Role | Package ID | Assets |")
$lines.Add("| --- | --- | --- | --- | ---: |")
foreach ($row in $rows) {
  $lines.Add("| $($row.key) | $($row.sourceRuntimeKey) | $($row.role) | $($row.packageId) | $($row.assetCount) |")
}
$lines.Add("")
$lines.Add("## Validation result")
$lines.Add("")
if ($errors.Count -eq 0) {
  $lines.Add("- status: passed")
}
else {
  $lines.Add("- status: failed")
  foreach ($errorMessage in $errors) {
    $lines.Add("- error: $errorMessage")
  }
}
$lines.Add("")
$lines.Add("## Publication rule")
$lines.Add("")
$lines.Add("- These packages must not be published before license review, package-size review, and split consumer validation.")
$lines.Add("- Packages that remain `design-only` or `pending-local-validation` are not release-ready.")

$markdownPath = Join-Path $outputRoot "split-delivery-prototype-report.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Split delivery prototype report written to $jsonPath"
Write-Host "Split delivery prototype report written to $markdownPath"

if ($errors.Count -gt 0) {
  foreach ($errorMessage in $errors) {
    Write-Error $errorMessage
  }
  exit 1
}
