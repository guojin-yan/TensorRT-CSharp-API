[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [Parameter(Mandatory = $true)]
  [string]$ConfigurePreset,
  [Parameter(Mandatory = $true)]
  [string]$BuildPreset,
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
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

$errors = New-Object System.Collections.Generic.List[string]
if ($package.buildPreset -ne $ConfigurePreset) {
  $errors.Add("configure_preset '$ConfigurePreset' does not match manifest buildPreset '$($package.buildPreset)'.")
}
if ($package.buildPreset -ne $BuildPreset) {
  $errors.Add("build_preset '$BuildPreset' does not match manifest buildPreset '$($package.buildPreset)'.")
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$summary = [ordered]@{
  runtimeKey = $RuntimePackageKey
  packageId = $package.packageId
  rid = $package.rid
  manifestBuildPreset = $package.buildPreset
  configurePreset = $ConfigurePreset
  buildPreset = $BuildPreset
  matchesManifest = ($errors.Count -eq 0)
  errors = @($errors)
}

$jsonPath = Join-Path $outputRoot "linux-workflow-contract.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Workflow Contract")
$lines.Add("")
$lines.Add("Runtime key: $RuntimePackageKey")
$lines.Add("")
$lines.Add("Package ID: $($package.packageId)")
$lines.Add("")
$lines.Add("- manifest build preset: $($package.buildPreset)")
$lines.Add("- configure preset: $ConfigurePreset")
$lines.Add("- build preset: $BuildPreset")
$lines.Add("- matches manifest: $($errors.Count -eq 0)")
$lines.Add("")

if ($errors.Count -eq 0) {
  $lines.Add("No contract mismatches were detected.")
}
else {
  $lines.Add("## Mismatches")
  $lines.Add("")
  foreach ($error in $errors) {
    $lines.Add("- $error")
  }
}

$markdownPath = Join-Path $outputRoot "linux-workflow-contract.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux workflow contract report written to $outputRoot"

if ($errors.Count -gt 0) {
  $errors | ForEach-Object { Write-Error $_ }
  exit 1
}
