[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ReleaseTag,
  [string]$Version,
  [string[]]$WindowsRuntimeKey = @(),
  [ValidateSet("full", "split")]
  [string]$WindowsRuntimeDeliveryMode = "split",
  [string[]]$LinuxRuntimeKey = @(),
  [switch]$IncludeManagedPackage,
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Expand-KeyList {
  param(
    [string[]]$Values
  )

  $keys = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $keys.Add($trimmed)
      }
    }
  }

  return @($keys | Select-Object -Unique)
}

function Add-ExpectedPackage {
  param(
    [Parameter(Mandatory = $true)]
    $Set,
    [Parameter(Mandatory = $true)]
    [string]$PackageId,
    [Parameter(Mandatory = $true)]
    [string]$ResolvedVersion
  )

  $null = $Set.Add("$PackageId.$ResolvedVersion.nupkg")
}

$resolvedVersion = if ([string]::IsNullOrWhiteSpace($Version)) {
  $ReleaseTag.TrimStart('v')
}
else {
  & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
}

$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json

$expected = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

if ($IncludeManagedPackage.IsPresent) {
  Add-ExpectedPackage -Set $expected -PackageId "JYPPX.TensorRT.CSharp.API" -ResolvedVersion $resolvedVersion
}

$windowsKeys = @(Expand-KeyList -Values $WindowsRuntimeKey)
foreach ($key in $windowsKeys) {
  $runtimePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $runtimePackage) {
    throw "Windows runtime key '$key' was not found."
  }

  Add-ExpectedPackage -Set $expected -PackageId $runtimePackage.packageId -ResolvedVersion $resolvedVersion

  if ($WindowsRuntimeDeliveryMode -eq "split") {
    $splitPackages = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $key })
    if ($splitPackages.Count -eq 0) {
      throw "No split runtime packages were defined for source runtime '$key'."
    }

    foreach ($splitPackage in $splitPackages) {
      Add-ExpectedPackage -Set $expected -PackageId $splitPackage.packageId -ResolvedVersion $resolvedVersion
    }
  }
}

$linuxKeys = @(Expand-KeyList -Values $LinuxRuntimeKey)
foreach ($key in $linuxKeys) {
  $runtimePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $runtimePackage) {
    throw "Linux runtime key '$key' was not found."
  }

  Add-ExpectedPackage -Set $expected -PackageId $runtimePackage.packageId -ResolvedVersion $resolvedVersion
}

$release = gh release view $ReleaseTag --repo $Repository --json assets,name,tagName,url | ConvertFrom-Json
$actual = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($asset in @($release.assets)) {
  $null = $actual.Add($asset.name)
}

$missing = @($expected | Where-Object { -not $actual.Contains($_) } | Sort-Object)
$unexpected = @($actual | Where-Object { -not $expected.Contains($_) } | Sort-Object)

$rows = New-Object System.Collections.Generic.List[object]
foreach ($name in ($expected | Sort-Object)) {
  $rows.Add([pscustomobject]@{
    package = $name
    present = $actual.Contains($name)
  })
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\release-audit"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$safeTag = ($ReleaseTag -replace '[^A-Za-z0-9._-]', '_')
$jsonPath = Join-Path $outputRoot "release-asset-audit-$safeTag.json"
$markdownPath = Join-Path $outputRoot "release-asset-audit-$safeTag.md"

[pscustomobject]@{
  releaseTag = $ReleaseTag
  releaseUrl = $release.url
  version = $resolvedVersion
  includeManagedPackage = $IncludeManagedPackage.IsPresent
  windowsRuntimeDeliveryMode = $WindowsRuntimeDeliveryMode
  windowsRuntimeKeys = $windowsKeys
  linuxRuntimeKeys = $linuxKeys
  expectedCount = $expected.Count
  actualCount = $actual.Count
  missing = $missing
  unexpected = $unexpected
  rows = $rows
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Release Asset Audit")
$lines.Add("")
$lines.Add("Release: [$($release.tagName)]($($release.url))")
$lines.Add("")
$lines.Add("| Package | Present |")
$lines.Add("| --- | --- |")
foreach ($row in $rows) {
  $lines.Add("| " + $codeQuote + $row.package + $codeQuote + " | " + $row.present + " |")
}

$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Expected packages: $($expected.Count)")
$lines.Add("- Release assets: $($actual.Count)")
$lines.Add("- Missing packages: $($missing.Count)")
$lines.Add("- Unexpected assets: $($unexpected.Count)")

if ($missing.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Missing")
  $lines.Add("")
  foreach ($name in $missing) {
    $lines.Add("- " + $codeQuote + $name + $codeQuote)
  }
}

if ($unexpected.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Unexpected")
  $lines.Add("")
  foreach ($name in $unexpected) {
    $lines.Add("- " + $codeQuote + $name + $codeQuote)
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release asset audit written to $jsonPath"
Write-Host "Release asset audit written to $markdownPath"

if ($missing.Count -gt 0) {
  Write-Warning "Release '$ReleaseTag' is still missing $($missing.Count) expected package asset(s)."
  exit 1
}
