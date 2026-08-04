[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath,
  [string]$ReportPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$workspaceRoot = Split-Path -Parent $RepositoryRoot

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $OutputPath = Join-Path $workspaceRoot "downloads\mnist-owner-generated\digit-7.pgm"
}
elseif (-not [IO.Path]::IsPathRooted($OutputPath)) {
  $OutputPath = Join-Path $workspaceRoot $OutputPath
}
$OutputPath = [IO.Path]::GetFullPath($OutputPath)

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
  $ReportPath = Join-Path (Split-Path -Parent $OutputPath) "digit-7.asset.json"
}
elseif (-not [IO.Path]::IsPathRooted($ReportPath)) {
  $ReportPath = Join-Path $workspaceRoot $ReportPath
}
$ReportPath = [IO.Path]::GetFullPath($ReportPath)

$pixels = New-Object byte[] (28 * 28)
for ($index = 0; $index -lt $pixels.Length; $index++) {
  $pixels[$index] = 255
}
for ($y = 5; $y -le 8; $y++) {
  for ($x = 5; $x -le 22; $x++) {
    $pixels[($y * 28) + $x] = 0
  }
}
for ($y = 8; $y -le 23; $y++) {
  $center = 21 - [int][Math]::Floor(($y - 8) * 0.72)
  for ($offset = -2; $offset -le 2; $offset++) {
    $x = $center + $offset
    if ($x -ge 0 -and $x -lt 28) {
      $pixels[($y * 28) + $x] = 0
    }
  }
}

$header = [Text.Encoding]::ASCII.GetBytes("P5`n28 28`n255`n")
$bytes = New-Object byte[] ($header.Length + $pixels.Length)
[Array]::Copy($header, 0, $bytes, 0, $header.Length)
[Array]::Copy($pixels, 0, $bytes, $header.Length, $pixels.Length)

foreach ($directory in @((Split-Path -Parent $OutputPath), (Split-Path -Parent $ReportPath))) {
  New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
[IO.File]::WriteAllBytes($OutputPath, $bytes)
$sha256 = (Get-FileHash -LiteralPath $OutputPath -Algorithm SHA256).Hash.ToLowerInvariant()

$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "mnist-project-generated-input-asset"
  sourceClassification = "project-generated-deterministic-geometry"
  generator = "eng/New-MnistOwnerGeneratedDigit.ps1"
  width = 28
  height = 28
  format = "P5 PGM"
  length = (Get-Item -LiteralPath $OutputPath).Length
  sha256 = $sha256
  expectedDigit = 7
  thirdPartySourceAsset = $false
  license = "CC0-1.0"
  licenseUrl = "https://creativecommons.org/publicdomain/zero/1.0/"
  publicRedistributionPermitted = $true
  trackedByGit = $false
  performsPublish = $false
}
[IO.File]::WriteAllText(
  $ReportPath,
  ($report | ConvertTo-Json -Depth 5) + "`n",
  [Text.UTF8Encoding]::new($false))

Write-Host "MnistGeneratedAsset Path=$OutputPath Length=$($report.length) Sha256=$sha256"
Write-Host "ThirdPartySourceAsset=False PublicRedistributionPermitted=True PerformsPublish=False"
Write-Host "Report=$ReportPath"
