[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$SplitPackageKey,
  [string]$SourceAssetsRoot,
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json

$splitPackage = $splitManifest.packages | Where-Object { $_.key -eq $SplitPackageKey } | Select-Object -First 1
if (-not $splitPackage) {
  throw "Split package key '$SplitPackageKey' was not found."
}

$sourcePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $splitPackage.sourceRuntimeKey } | Select-Object -First 1
if (-not $sourcePackage) {
  throw "Source runtime key '$($splitPackage.sourceRuntimeKey)' was not found."
}

if ([string]::IsNullOrWhiteSpace($SourceAssetsRoot)) {
  $SourceAssetsRoot = Join-Path $RepositoryRoot "pack\runtime\$($splitPackage.sourceRuntimeKey)\assets\runtimes\$($sourcePackage.rid)\native"
}

if (-not (Test-Path -LiteralPath $SourceAssetsRoot -PathType Container)) {
  throw "Source runtime native asset folder was not found: $SourceAssetsRoot"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\runtime-split\$SplitPackageKey"
}

$artifactNativeOutput = Join-Path $OutputRoot "runtimes\$($splitPackage.rid)\native"
$packageAssetsRoot = Join-Path $RepositoryRoot "pack\runtime-split\$SplitPackageKey\assets"
$packageNativeOutput = Join-Path $packageAssetsRoot "runtimes\$($splitPackage.rid)\native"

if (Test-Path -LiteralPath $packageAssetsRoot) {
  Remove-Item -LiteralPath $packageAssetsRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $artifactNativeOutput -Force | Out-Null
New-Item -ItemType Directory -Path $packageNativeOutput -Force | Out-Null

$copiedFiles = New-Object System.Collections.Generic.List[string]
foreach ($asset in @($splitPackage.assets)) {
  $sourcePath = Join-Path $SourceAssetsRoot $asset
  if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "Expected split runtime asset was not found: $sourcePath"
  }

  $artifactDestination = Join-Path $artifactNativeOutput $asset
  $packageDestination = Join-Path $packageNativeOutput $asset
  Copy-Item -LiteralPath $sourcePath -Destination $artifactDestination -Force -ErrorAction Stop
  Copy-Item -LiteralPath $sourcePath -Destination $packageDestination -Force -ErrorAction Stop
  $copiedFiles.Add($artifactDestination)
}

$artifactManifest = [ordered]@{
  splitPackageKey = $splitPackage.key
  sourceRuntimeKey = $splitPackage.sourceRuntimeKey
  packageId = $splitPackage.packageId
  rid = $splitPackage.rid
  role = $splitPackage.role
  prototypeState = $splitPackage.prototypeState
  sourceAssetsRoot = $SourceAssetsRoot
  packageAssetsRoot = $packageAssetsRoot
  files = @($copiedFiles)
}

$artifactManifestPath = Join-Path $OutputRoot "artifact-manifest.json"
$artifactManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $artifactManifestPath -Encoding utf8

Write-Host "Collected split runtime assets for $($splitPackage.packageId)"
Write-Host "Output: $OutputRoot"
