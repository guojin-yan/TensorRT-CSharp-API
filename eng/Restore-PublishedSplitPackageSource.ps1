param(
  [Parameter(Mandatory = $true)]
  [string]$SourceRuntimeKey,
  [string[]]$SplitPackageRole = @("all"),
  [string]$Version,
  [string]$BridgePackageVersion,
  [string]$VendorPackageVersion,
  [string]$CudaCudnnPackageVersion,
  [string]$TensorRtPackageVersion,
  [string]$VendorPackageReleaseTag,
  [string]$Repository,
  [string]$OutputRoot,
  [string]$OutputPathFile,
  [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Expand-KeyList {
  param(
    [AllowNull()]
    [string[]]$Values
  )

  $items = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in $value -split '[,;]') {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $items.Add($trimmed)
      }
    }
  }

  return @($items)
}

function Resolve-RolePackageVersion {
  param(
    [string]$Value,
    [string]$Fallback
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $Fallback
  }

  return $Value
}

function Resolve-ReleaseTag {
  param(
    [string]$ExplicitTag,
    [string]$ExplicitVendorVersion,
    [string]$ExplicitCudaCudnnVersion,
    [string]$ExplicitTensorRtVersion,
    [string]$VendorVersion,
    [string]$CudaCudnnVersion,
    [string]$TensorRtVersion
  )

  if (-not [string]::IsNullOrWhiteSpace($ExplicitTag)) {
    return $ExplicitTag
  }

  if (-not [string]::IsNullOrWhiteSpace($ExplicitVendorVersion)) {
    return "v$VendorVersion"
  }

  if (-not [string]::IsNullOrWhiteSpace($ExplicitCudaCudnnVersion) -and
      ([string]::IsNullOrWhiteSpace($ExplicitTensorRtVersion) -or $CudaCudnnVersion -eq $TensorRtVersion)) {
    return "v$CudaCudnnVersion"
  }

  if (-not [string]::IsNullOrWhiteSpace($ExplicitTensorRtVersion) -and
      [string]::IsNullOrWhiteSpace($ExplicitCudaCudnnVersion)) {
    return "v$TensorRtVersion"
  }

  return ""
}

function Test-SplitPackageRequested {
  param(
    [Parameter(Mandatory = $true)]
    [psobject]$SplitPackage,
    [Parameter(Mandatory = $true)]
    [string[]]$RequestedRoles
  )

  if ($RequestedRoles -contains "all") {
    return $true
  }

  $role = ([string]$SplitPackage.role).ToLowerInvariant()
  $key = ([string]$SplitPackage.key).ToLowerInvariant()
  return ($RequestedRoles -contains $role) -or ($RequestedRoles -contains $key)
}

function Get-SplitPackageVersion {
  param(
    [Parameter(Mandatory = $true)]
    [psobject]$SplitPackage,
    [Parameter(Mandatory = $true)]
    [string]$BridgeVersion,
    [Parameter(Mandatory = $true)]
    [string]$CudaCudnnVersion,
    [Parameter(Mandatory = $true)]
    [string]$TensorRtVersion
  )

  switch ([string]$SplitPackage.role) {
    "bridge" { return $BridgeVersion }
    "cuda-cudnn" { return $CudaCudnnVersion }
    default { return $TensorRtVersion }
  }
}

function Write-ResolvedPackageSource {
  param(
    [string]$SourceDirectory
  )

  if ([string]::IsNullOrWhiteSpace($OutputPathFile)) {
    return
  }

  $outputDirectory = Split-Path -Parent $OutputPathFile
  if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
  }

  Set-Content -LiteralPath $OutputPathFile -Value $SourceDirectory -Encoding utf8
}

if ([string]::IsNullOrWhiteSpace($Version)) {
  $Version = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1")
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  $Repository = $env:GH_REPO
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    $OutputRoot = Join-Path $RepositoryRoot "artifacts\vendor-package-source"
  }
  else {
    $OutputRoot = Join-Path $env:RUNNER_TEMP "jyppx-release-package-source"
  }
}

$requestedRoles = @(Expand-KeyList -Values $SplitPackageRole | ForEach-Object { $_.ToLowerInvariant() })
if ($requestedRoles.Count -eq 0) {
  $requestedRoles = @("all")
}

$resolvedVendorPackageVersion = Resolve-RolePackageVersion -Value $VendorPackageVersion -Fallback $Version
$resolvedBridgePackageVersion = Resolve-RolePackageVersion -Value $BridgePackageVersion -Fallback $Version
$resolvedCudaCudnnPackageVersion = Resolve-RolePackageVersion -Value $CudaCudnnPackageVersion -Fallback $resolvedVendorPackageVersion
$resolvedTensorRtPackageVersion = Resolve-RolePackageVersion -Value $TensorRtPackageVersion -Fallback $resolvedVendorPackageVersion
$resolvedReleaseTag = Resolve-ReleaseTag `
  -ExplicitTag $VendorPackageReleaseTag `
  -ExplicitVendorVersion $VendorPackageVersion `
  -ExplicitCudaCudnnVersion $CudaCudnnPackageVersion `
  -ExplicitTensorRtVersion $TensorRtPackageVersion `
  -VendorVersion $resolvedVendorPackageVersion `
  -CudaCudnnVersion $resolvedCudaCudnnPackageVersion `
  -TensorRtVersion $resolvedTensorRtPackageVersion

$splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
$splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$allSplitPackages = @($splitManifest.packages | Where-Object { [string]$_.sourceRuntimeKey -eq $SourceRuntimeKey })
if ($allSplitPackages.Count -eq 0) {
  throw "No split runtime packages were defined for source runtime '$SourceRuntimeKey'."
}

$selectedKeys = @(
  $allSplitPackages |
    Where-Object { Test-SplitPackageRequested -SplitPackage $_ -RequestedRoles $requestedRoles } |
    ForEach-Object { [string]$_.key }
)

$missingPackages = @(
  $allSplitPackages |
    Where-Object { $selectedKeys -notcontains [string]$_.key }
)

if ($missingPackages.Count -eq 0) {
  Write-Host "All split component packages for '$SourceRuntimeKey' are built in this run; no published vendor package source is needed."
  Write-ResolvedPackageSource -SourceDirectory ""
  return
}

if ([string]::IsNullOrWhiteSpace($resolvedReleaseTag)) {
  Write-Host "No vendor package release tag was resolved; relying on configured NuGet feeds for previously published split packages."
  Write-ResolvedPackageSource -SourceDirectory ""
  return
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  throw "Repository was not provided. Pass -Repository or set GH_REPO before downloading GitHub Release assets."
}

$safeKey = $SourceRuntimeKey -replace '[^A-Za-z0-9._-]', '-'
$sourceDirectory = Join-Path $OutputRoot $safeKey
New-Item -ItemType Directory -Path $sourceDirectory -Force | Out-Null

foreach ($package in $missingPackages) {
  $packageVersion = Get-SplitPackageVersion `
    -SplitPackage $package `
    -BridgeVersion $resolvedBridgePackageVersion `
    -CudaCudnnVersion $resolvedCudaCudnnPackageVersion `
    -TensorRtVersion $resolvedTensorRtPackageVersion
  $packageFileName = "$($package.packageId).$packageVersion.nupkg"
  $targetPath = Join-Path $sourceDirectory $packageFileName
  if (Test-Path -LiteralPath $targetPath -PathType Leaf) {
    Write-Host "Using cached release asset package: $packageFileName"
    continue
  }

  Write-Host "Downloading published split package '$packageFileName' from GitHub Release '$resolvedReleaseTag'."
  gh release download $resolvedReleaseTag --repo $Repository --pattern $packageFileName --dir $sourceDirectory --clobber
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to download published split package '$packageFileName' from release '$resolvedReleaseTag'."
  }

  if (-not (Test-Path -LiteralPath $targetPath -PathType Leaf)) {
    throw "Published split package '$packageFileName' was not found in release '$resolvedReleaseTag'."
  }
}

Write-Host "Published vendor package source: $sourceDirectory"
Write-ResolvedPackageSource -SourceDirectory $sourceDirectory
