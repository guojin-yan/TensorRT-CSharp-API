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
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$errors = New-Object System.Collections.Generic.List[string]

foreach ($package in $manifest.packages | Where-Object { $_.platform -eq "windows" }) {
  $artifactRoot = Join-Path $RepositoryRoot "artifacts\runtime\$($package.key)"
  $artifactManifestPath = Join-Path $artifactRoot "artifact-manifest.json"
  if (-not (Test-Path -LiteralPath $artifactManifestPath)) {
    $errors.Add("Missing artifact manifest: $artifactManifestPath")
  }

  $nupkgPattern = "$($package.packageId).*" + ".nupkg"
  $nupkgs = Get-ChildItem -Path (Join-Path $RepositoryRoot "artifacts\runtime-nupkg") -Filter $nupkgPattern -ErrorAction SilentlyContinue
  if (-not $nupkgs) {
    $errors.Add("Missing runtime package output for '$($package.packageId)'.")
  }
}

if ($errors.Count -gt 0) {
  $errors | ForEach-Object { Write-Error $_ }
  exit 1
}

Write-Host "Local Windows runtime artifacts are present for all configured packages."
