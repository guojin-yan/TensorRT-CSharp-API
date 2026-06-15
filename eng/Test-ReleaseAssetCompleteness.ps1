[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ReleaseTag,
  [string]$Version,
  [string[]]$WindowsRuntimeKey = @(),
  [ValidateSet("full", "split")]
  [string]$WindowsRuntimeDeliveryMode = "split",
  [string[]]$WindowsSplitPackageRoles = @("all"),
  [string]$WindowsCudaCudnnPackageVersion,
  [string]$WindowsCudaCudnnPackageReleaseTag,
  [string]$WindowsTensorRtPackageVersion,
  [string]$WindowsTensorRtPackageReleaseTag,
  [string[]]$LinuxRuntimeKey = @(),
  [switch]$IncludeManagedPackage,
  [switch]$ReportUnexpectedAssets,
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

function Resolve-RolePackageVersion {
  param(
    [string]$Value,
    [Parameter(Mandatory = $true)]
    [string]$Fallback
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $Fallback
  }

  return & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Value
}

function Resolve-PackageReleaseTag {
  param(
    [string]$ExplicitTag,
    [Parameter(Mandatory = $true)]
    [string]$PackageVersion,
    [Parameter(Mandatory = $true)]
    [string]$DefaultVersion,
    [Parameter(Mandatory = $true)]
    [string]$DefaultTag
  )

  if (-not [string]::IsNullOrWhiteSpace($ExplicitTag)) {
    return $ExplicitTag
  }

  if ($PackageVersion -eq $DefaultVersion) {
    return $DefaultTag
  }

  return "v$PackageVersion"
}

function Test-SplitPackageRequested {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SplitPackage,
    [Parameter(Mandatory = $true)]
    [string[]]$RequestedRoles
  )

  $role = ([string]$SplitPackage.role).ToLowerInvariant()
  $key = ([string]$SplitPackage.key).ToLowerInvariant()

  foreach ($requestedRole in $RequestedRoles) {
    switch ($requestedRole) {
      "all" { return $true }
      "stable-dependencies" {
        if ($role -eq "cuda-cudnn" -or $role -eq "tensorrt") {
          return $true
        }
      }
      "nvidia-dependencies" {
        if ($role -eq "cuda-cudnn" -or $role -eq "tensorrt") {
          return $true
        }
      }
      default {
        if ($role -eq $requestedRole -or $key -eq $requestedRole) {
          return $true
        }
      }
    }
  }

  return $false
}

function Test-CollectionRequested {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$RequestedRoles
  )

  return (
    $RequestedRoles -contains "all" -or
    $RequestedRoles -contains "collection" -or
    $RequestedRoles -contains "meta"
  )
}

function Get-SplitPackageTarget {
  param(
    [Parameter(Mandatory = $true)]
    [object]$SplitPackage
  )

  switch -Regex ([string]$SplitPackage.role) {
    '^bridge$' {
      return [pscustomobject]@{
        version = $resolvedVersion
        releaseTag = $ReleaseTag
      }
    }
    '^cuda-cudnn$' {
      return [pscustomobject]@{
        version = $resolvedCudaCudnnPackageVersion
        releaseTag = $resolvedCudaCudnnPackageReleaseTag
      }
    }
    '^tensorrt$' {
      return [pscustomobject]@{
        version = $resolvedTensorRtPackageVersion
        releaseTag = $resolvedTensorRtPackageReleaseTag
      }
    }
    default {
      return [pscustomobject]@{
        version = $resolvedVersion
        releaseTag = $ReleaseTag
      }
    }
  }
}

function Add-ExpectedPackage {
  param(
    [Parameter(Mandatory = $true)]
    $Rows,
    [Parameter(Mandatory = $true)]
    $Set,
    [Parameter(Mandatory = $true)]
    [string]$PackageId,
    [Parameter(Mandatory = $true)]
    [string]$ResolvedVersion,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedReleaseTag,
    [string]$SourceRuntimeKey,
    [string]$Role
  )

  $packageName = "$PackageId.$ResolvedVersion.nupkg"
  $identity = "$ExpectedReleaseTag|$packageName"
  if (-not $Set.Add($identity)) {
    return
  }

  $Rows.Add([pscustomobject]@{
    package = $packageName
    releaseTag = $ExpectedReleaseTag
    sourceRuntimeKey = $SourceRuntimeKey
    role = $Role
    present = $false
  })
}

function Get-Release {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Tag
  )

  if ($releaseCache.ContainsKey($Tag)) {
    return $releaseCache[$Tag]
  }

  $release = gh release view $Tag --repo $Repository --json assets,name,tagName,url | ConvertFrom-Json
  $assets = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($asset in @($release.assets)) {
    $null = $assets.Add($asset.name)
  }

  $entry = [pscustomobject]@{
    tagName = $release.tagName
    url = $release.url
    assets = $assets
    assetCount = $assets.Count
  }
  $releaseCache[$Tag] = $entry
  return $entry
}

$resolvedVersion = if ([string]::IsNullOrWhiteSpace($Version)) {
  $ReleaseTag.TrimStart('v')
}
else {
  & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
}

$resolvedCudaCudnnPackageVersion = Resolve-RolePackageVersion -Value $WindowsCudaCudnnPackageVersion -Fallback $resolvedVersion
$resolvedCudaCudnnPackageReleaseTag = Resolve-PackageReleaseTag `
  -ExplicitTag $WindowsCudaCudnnPackageReleaseTag `
  -PackageVersion $resolvedCudaCudnnPackageVersion `
  -DefaultVersion $resolvedVersion `
  -DefaultTag $ReleaseTag
$resolvedTensorRtPackageVersion = Resolve-RolePackageVersion -Value $WindowsTensorRtPackageVersion -Fallback $resolvedVersion
$resolvedTensorRtPackageReleaseTag = Resolve-PackageReleaseTag `
  -ExplicitTag $WindowsTensorRtPackageReleaseTag `
  -PackageVersion $resolvedTensorRtPackageVersion `
  -DefaultVersion $resolvedVersion `
  -DefaultTag $ReleaseTag

$runtimeManifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json

$expectedSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$rows = New-Object System.Collections.Generic.List[object]
$releaseCache = @{}

if ($IncludeManagedPackage.IsPresent) {
  Add-ExpectedPackage `
    -Rows $rows `
    -Set $expectedSet `
    -PackageId "JYPPX.TensorRT.CSharp.API" `
    -ResolvedVersion $resolvedVersion `
    -ExpectedReleaseTag $ReleaseTag `
    -Role "managed"
}

$windowsKeys = @(Expand-KeyList -Values $WindowsRuntimeKey)
$requestedSplitRoles = @(Expand-KeyList -Values $WindowsSplitPackageRoles | ForEach-Object { $_.ToLowerInvariant() })
if ($requestedSplitRoles.Count -eq 0) {
  $requestedSplitRoles = @("all")
}

foreach ($key in $windowsKeys) {
  $runtimePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $runtimePackage) {
    throw "Windows runtime key '$key' was not found."
  }

  if ($WindowsRuntimeDeliveryMode -eq "full") {
    Add-ExpectedPackage `
      -Rows $rows `
      -Set $expectedSet `
      -PackageId $runtimePackage.packageId `
      -ResolvedVersion $resolvedVersion `
      -ExpectedReleaseTag $ReleaseTag `
      -SourceRuntimeKey $key `
      -Role "full"
    continue
  }

  $allSplitPackages = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $key })
  if ($allSplitPackages.Count -eq 0) {
    throw "No split runtime packages were defined for source runtime '$key'."
  }

  if (Test-CollectionRequested -RequestedRoles $requestedSplitRoles) {
    Add-ExpectedPackage `
      -Rows $rows `
      -Set $expectedSet `
      -PackageId $runtimePackage.packageId `
      -ResolvedVersion $resolvedVersion `
      -ExpectedReleaseTag $ReleaseTag `
      -SourceRuntimeKey $key `
      -Role "collection"
  }

  $selectedSplitPackages = @($allSplitPackages | Where-Object { Test-SplitPackageRequested -SplitPackage $_ -RequestedRoles $requestedSplitRoles })
  foreach ($splitPackage in $selectedSplitPackages) {
    $target = Get-SplitPackageTarget -SplitPackage $splitPackage
    Add-ExpectedPackage `
      -Rows $rows `
      -Set $expectedSet `
      -PackageId $splitPackage.packageId `
      -ResolvedVersion $target.version `
      -ExpectedReleaseTag $target.releaseTag `
      -SourceRuntimeKey $key `
      -Role $splitPackage.role
  }

  if (Test-CollectionRequested -RequestedRoles $requestedSplitRoles) {
    $selectedKeys = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($splitPackage in $selectedSplitPackages) {
      $null = $selectedKeys.Add([string]$splitPackage.key)
    }

    $dependencySplitPackages = @($allSplitPackages | Where-Object {
        -not $selectedKeys.Contains([string]$_.key)
      })

    foreach ($splitPackage in $dependencySplitPackages) {
      $target = Get-SplitPackageTarget -SplitPackage $splitPackage
      Add-ExpectedPackage `
        -Rows $rows `
        -Set $expectedSet `
        -PackageId $splitPackage.packageId `
        -ResolvedVersion $target.version `
        -ExpectedReleaseTag $target.releaseTag `
        -SourceRuntimeKey $key `
        -Role "$($splitPackage.role)-dependency"
    }
  }
}

$linuxKeys = @(Expand-KeyList -Values $LinuxRuntimeKey)
foreach ($key in $linuxKeys) {
  $runtimePackage = $runtimeManifest.packages | Where-Object { $_.key -eq $key } | Select-Object -First 1
  if (-not $runtimePackage) {
    throw "Linux runtime key '$key' was not found."
  }

  Add-ExpectedPackage `
    -Rows $rows `
    -Set $expectedSet `
    -PackageId $runtimePackage.packageId `
    -ResolvedVersion $resolvedVersion `
    -ExpectedReleaseTag $ReleaseTag `
    -SourceRuntimeKey $key `
    -Role "linux"
}

foreach ($row in $rows) {
  $release = Get-Release -Tag $row.releaseTag
  $row.present = $release.assets.Contains($row.package)
}

$expectedByRelease = @{}
foreach ($row in $rows) {
  if (-not $expectedByRelease.ContainsKey($row.releaseTag)) {
    $expectedByRelease[$row.releaseTag] = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  }

  $null = $expectedByRelease[$row.releaseTag].Add($row.package)
}

$missing = @($rows | Where-Object { -not $_.present } | Sort-Object releaseTag, package)
$unexpected = New-Object System.Collections.Generic.List[object]
if ($ReportUnexpectedAssets.IsPresent) {
  foreach ($tag in ($expectedByRelease.Keys | Sort-Object)) {
    $release = Get-Release -Tag $tag
    foreach ($assetName in ($release.assets | Sort-Object)) {
      if (-not $expectedByRelease[$tag].Contains($assetName)) {
        $unexpected.Add([pscustomobject]@{
          releaseTag = $tag
          package = $assetName
        })
      }
    }
  }
}

$releaseSummaries = New-Object System.Collections.Generic.List[object]
foreach ($tag in ($expectedByRelease.Keys | Sort-Object)) {
  $release = Get-Release -Tag $tag
  $expectedForRelease = @($rows | Where-Object { $_.releaseTag -eq $tag })
  $missingForRelease = @($missing | Where-Object { $_.releaseTag -eq $tag })
  $releaseSummaries.Add([pscustomobject]@{
    releaseTag = $tag
    releaseUrl = $release.url
    expectedCount = $expectedForRelease.Count
    actualAssetCount = $release.assetCount
    missingCount = $missingForRelease.Count
  })
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\release-audit"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$safeTag = ($ReleaseTag -replace '[^A-Za-z0-9._-]', '_')
$jsonPath = Join-Path $outputRoot "release-asset-audit-$safeTag.json"
$markdownPath = Join-Path $outputRoot "release-asset-audit-$safeTag.md"

[pscustomobject]@{
  releaseTag = $ReleaseTag
  version = $resolvedVersion
  includeManagedPackage = $IncludeManagedPackage.IsPresent
  windowsRuntimeDeliveryMode = $WindowsRuntimeDeliveryMode
  windowsSplitPackageRoles = $requestedSplitRoles
  windowsCudaCudnnPackageVersion = $resolvedCudaCudnnPackageVersion
  windowsCudaCudnnPackageReleaseTag = $resolvedCudaCudnnPackageReleaseTag
  windowsTensorRtPackageVersion = $resolvedTensorRtPackageVersion
  windowsTensorRtPackageReleaseTag = $resolvedTensorRtPackageReleaseTag
  windowsRuntimeKeys = $windowsKeys
  linuxRuntimeKeys = $linuxKeys
  reportUnexpectedAssets = $ReportUnexpectedAssets.IsPresent
  expectedCount = $rows.Count
  missingCount = $missing.Count
  unexpectedCount = $unexpected.Count
  releases = $releaseSummaries
  missing = $missing
  unexpected = $unexpected
  rows = $rows
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Release Asset Audit")
$lines.Add("")
$lines.Add("Primary release: " + $codeQuote + $ReleaseTag + $codeQuote)
$lines.Add("")
$lines.Add("| Release | Expected | Actual assets | Missing |")
$lines.Add("| --- | ---: | ---: | ---: |")
foreach ($summary in $releaseSummaries) {
  $lines.Add("| " + $codeQuote + $summary.releaseTag + $codeQuote + " | $($summary.expectedCount) | $($summary.actualAssetCount) | $($summary.missingCount) |")
}

$lines.Add("")
$lines.Add("| Release | Package | Role | Present |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($row in ($rows | Sort-Object releaseTag, package)) {
  $lines.Add("| " + $codeQuote + $row.releaseTag + $codeQuote + " | " + $codeQuote + $row.package + $codeQuote + " | " + $codeQuote + $row.role + $codeQuote + " | " + $row.present + " |")
}

$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Expected packages: $($rows.Count)")
$lines.Add("- Missing packages: $($missing.Count)")
$lines.Add("- Unexpected assets: $($unexpected.Count)")

if ($missing.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Missing")
  $lines.Add("")
  foreach ($row in $missing) {
    $lines.Add("- " + $codeQuote + $row.releaseTag + $codeQuote + " / " + $codeQuote + $row.package + $codeQuote)
  }
}

if ($unexpected.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Unexpected")
  $lines.Add("")
  foreach ($row in $unexpected) {
    $lines.Add("- " + $codeQuote + $row.releaseTag + $codeQuote + " / " + $codeQuote + $row.package + $codeQuote)
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release asset audit written to $jsonPath"
Write-Host "Release asset audit written to $markdownPath"

if ($missing.Count -gt 0) {
  Write-Warning "Release asset audit is still missing $($missing.Count) expected package asset(s)."
  exit 1
}
