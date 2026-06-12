[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
$requiredFiles = @(
  (Join-Path $root "linux-runtime-dry-run.json"),
  (Join-Path $root "README.md"),
  (Join-Path $root "linux-runner-checklist.md")
)

$errors = New-Object System.Collections.Generic.List[string]
foreach ($file in $requiredFiles) {
  if (-not (Test-Path -LiteralPath $file)) {
    $errors.Add("Missing Linux dry-run artifact: $file")
  }
}

if ($errors.Count -gt 0) {
  $errors | ForEach-Object { Write-Error $_ }
  exit 1
}

Write-Host "Linux dry-run artifacts are present for $RuntimePackageKey"
