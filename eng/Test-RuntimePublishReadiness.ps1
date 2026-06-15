[CmdletBinding()]
param(
  [ValidateSet("public-preview", "private-feed", "split-delivery")]
  [string]$TargetMode = "public-preview",
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

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json

$packages = @($manifest.packages)
if (-not [string]::IsNullOrWhiteSpace($RuntimePackageKey)) {
  $packages = @($packages | Where-Object { $_.key -eq $RuntimePackageKey })
  if ($packages.Count -eq 0) {
    throw "Runtime package key '$RuntimePackageKey' was not found."
  }
}

$results = New-Object System.Collections.Generic.List[object]
$blockingErrors = New-Object System.Collections.Generic.List[string]

foreach ($package in $packages) {
  $blockers = New-Object System.Collections.Generic.List[string]
  $warnings = New-Object System.Collections.Generic.List[string]

  $nupkgPattern = "$($package.packageId).*" + ".nupkg"
  $hasLocalNupkg = @(Get-ChildItem -Path (Join-Path $RepositoryRoot "artifacts\runtime-nupkg") -Filter $nupkgPattern -ErrorAction SilentlyContinue).Count -gt 0
  $splitManifestPath = Join-Path $RepositoryRoot "pack\runtime-split\split-runtime-packages.manifest.json"
  $splitPackages = @()
  $hasLocalSplitNupkgSet = $false
  if (Test-Path -LiteralPath $splitManifestPath -PathType Leaf) {
    $splitManifest = Get-Content -LiteralPath $splitManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
    $splitPackages = @($splitManifest.packages | Where-Object { $_.sourceRuntimeKey -eq $package.key })
    if ($splitPackages.Count -gt 0) {
      $splitNupkgRoot = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$($package.key)"
      $missingSplitNupkgs = @(
        $splitPackages |
          Where-Object {
            $pattern = "$($_.packageId).*" + ".nupkg"
            @(Get-ChildItem -Path $splitNupkgRoot -Filter $pattern -ErrorAction SilentlyContinue).Count -eq 0
          }
      )
      $hasLocalSplitNupkgSet = $missingSplitNupkgs.Count -eq 0
    }
  }

  switch ($TargetMode) {
    "public-preview" {
      if ($package.distributionTier -ne "public-sample") {
        $blockers.Add("distributionTier '$($package.distributionTier)' is not eligible for public preview.")
      }
      if ($package.validationState -ne "local-validated") {
        $blockers.Add("validationState '$($package.validationState)' is not sufficient for public preview.")
      }
      if (-not $hasLocalNupkg) {
        $blockers.Add("local runtime nupkg was not found under artifacts/runtime-nupkg.")
      }
    }
    "private-feed" {
      if ($package.validationState -ne "local-validated") {
        $blockers.Add("validationState '$($package.validationState)' is not sufficient for private-feed publication.")
      }
      if (-not $hasLocalNupkg) {
        $blockers.Add("local runtime nupkg was not found under artifacts/runtime-nupkg.")
      }
    }
    "split-delivery" {
      if ($package.validationState -ne "local-validated") {
        $blockers.Add("validationState '$($package.validationState)' is not sufficient for split-delivery publication.")
      }
      if ($package.distributionTier -notin @("split-delivery-candidate", "private-feed")) {
        $warnings.Add("package is not marked as a split-delivery candidate, review whether split delivery is actually needed.")
      }

      if ($package.distributionTier -eq "split-delivery-candidate") {
        if (-not (Test-Path -LiteralPath $splitManifestPath -PathType Leaf)) {
          $blockers.Add("split runtime package manifest was not found under pack/runtime-split.")
        }
        else {
          if (($splitPackages | Where-Object { $_.role -eq "bridge" }).Count -eq 0) {
            $blockers.Add("split runtime package set for '$($package.key)' is missing a bridge package.")
          }
          if (($splitPackages | Where-Object { $_.role -eq "cuda-cudnn" }).Count -eq 0) {
            $blockers.Add("split runtime package set for '$($package.key)' is missing a CudaCudnn package.")
          }
          if (($splitPackages | Where-Object { $_.role -eq "tensorrt" }).Count -eq 0) {
            $blockers.Add("split runtime package set for '$($package.key)' is missing a TensorRt package.")
          }
          if (-not $hasLocalSplitNupkgSet) {
            $warnings.Add("complete local split runtime nupkg set was not detected under artifacts/runtime-split-nupkg/$($package.key).")
          }
        }
      }
    }
  }

  if ($package.platform -eq "linux" -and $package.validationState -eq "dry-run-only") {
    $warnings.Add("Linux combination is still dry-run-only and has not been validated on a real Linux x64 runner.")
  }

  if ($package.platform -eq "windows" -and -not $hasLocalNupkg) {
    $warnings.Add("No local runtime nupkg was detected for this Windows package.")
  }

  $status = if ($blockers.Count -eq 0) { "ready" } else { "blocked" }

  $results.Add([pscustomobject]@{
    key = $package.key
    packageId = $package.packageId
    targetMode = $TargetMode
    distributionTier = $package.distributionTier
    validationState = $package.validationState
    status = $status
    blockers = @($blockers)
    warnings = @($warnings)
  })

  if ($blockers.Count -gt 0) {
    $blockingErrors.Add("$($package.key): " + ($blockers -join " | "))
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\runtime-distribution"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "publish-readiness-$TargetMode.json"
$results | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Runtime Publish Readiness")
$lines.Add("")
$lines.Add("Target mode: $TargetMode")
$lines.Add("")
$lines.Add("| Key | Tier | Validation | Status |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($result in $results) {
  $lines.Add("| $($result.key) | $($result.distributionTier) | $($result.validationState) | $($result.status) |")
}
$lines.Add("")

foreach ($result in $results) {
  $lines.Add("## $($result.key)")
  $lines.Add("")
  if ($result.blockers.Count -eq 0) {
    $lines.Add("- blockers: none")
  }
  else {
    foreach ($blocker in $result.blockers) {
      $lines.Add("- blocker: $blocker")
    }
  }

  if ($result.warnings.Count -eq 0) {
    $lines.Add("- warnings: none")
  }
  else {
    foreach ($warning in $result.warnings) {
      $lines.Add("- warning: $warning")
    }
  }
  $lines.Add("")
}

$markdownPath = Join-Path $outputRoot "publish-readiness-$TargetMode.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Publish readiness report written to $outputRoot"

if ($blockingErrors.Count -gt 0) {
  $blockingErrors | ForEach-Object { Write-Error $_ }
  exit 1
}
