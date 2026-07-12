[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$TensorRtSourcePath,
  [string]$CudnnSourcePath,
  [string]$TensorRtRoot,
  [string]$CudnnRoot,
  [string]$ReportDirectory,
  [switch]$DryRun,
  [switch]$Force,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\vendor-runtime-assets\$RuntimePackageKey"
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found in $manifestPath."
}

if ([string]$package.platform -ne "windows") {
  throw "Runtime package key '$RuntimePackageKey' is not a Windows package."
}

function ConvertTo-ForwardSlashPath {
  param([string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return ""
  }

  return $Value -replace '\\', '/'
}

function ConvertTo-LocalRelativePath {
  param([string]$Value)

  return $Value -replace '/', [System.IO.Path]::DirectorySeparatorChar
}

function Expand-LocalPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $null
  }

  $expanded = $Path.Replace("<repo-root>", $RepositoryRoot).Replace("<repository-root>", $RepositoryRoot)
  $expanded = [Environment]::ExpandEnvironmentVariables($expanded)

  if ($expanded -eq "~") {
    $home = if ([string]::IsNullOrWhiteSpace($env:HOME)) { $env:USERPROFILE } else { $env:HOME }
    if (-not [string]::IsNullOrWhiteSpace($home)) {
      return $home
    }
  }

  if ($expanded.StartsWith("~/") -or $expanded.StartsWith("~\")) {
    $home = if ([string]::IsNullOrWhiteSpace($env:HOME)) { $env:USERPROFILE } else { $env:HOME }
    if (-not [string]::IsNullOrWhiteSpace($home)) {
      return (Join-Path $home $expanded.Substring(2))
    }
  }

  return $expanded
}

function Resolve-SourcePath {
  param([string]$Path)

  $expanded = Expand-LocalPath -Path $Path
  if ([string]::IsNullOrWhiteSpace($expanded)) {
    return $null
  }

  if (-not (Test-Path -LiteralPath $expanded)) {
    return $null
  }

  return (Resolve-Path -LiteralPath $expanded).Path
}

function Resolve-TargetRoot {
  param(
    [string]$ExplicitRoot,
    [string]$ResolvedRoot,
    [string]$PackageRootName
  )

  foreach ($candidate in @($ExplicitRoot, $ResolvedRoot, (Join-Path (Join-Path $RepositoryRoot "third_party\nvidia") $PackageRootName))) {
    $expanded = Expand-LocalPath -Path $candidate
    if ([string]::IsNullOrWhiteSpace($expanded)) {
      continue
    }

    return $expanded
  }

  return $null
}

function Find-DefaultSourcePath {
  param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("TensorRT", "cuDNN")]
    [string]$Kind
  )

  $profileRoot = if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { $env:HOME } else { $env:USERPROFILE }
  if ([string]::IsNullOrWhiteSpace($profileRoot)) {
    return $null
  }

  $downloads = Join-Path $profileRoot "Downloads"
  if (-not (Test-Path -LiteralPath $downloads -PathType Container)) {
    return $null
  }

  $patterns = if ($Kind -eq "TensorRT") {
    @(
      "TensorRT*$($package.tensorRtVersion)*cuda-$($package.cudaVersion)*Windows*.zip",
      "TensorRT*$($package.tensorRtVersion)*cuda*$($package.cudaVersion)*Windows*.zip",
      "TensorRT*$($package.tensorRtVersion)*Windows*.zip"
    )
  }
  else {
    @(
      "cudnn_$($package.cudnnVersion)_windows_x86_64.exe",
      "cudnn*$($package.cudnnVersion)*windows*x86_64*.exe",
      "cudnn*$($package.cudnnVersion)*windows*x86_64*.zip",
      "cudnn*cuda$($package.cudaLine)*windows*x86_64*.exe",
      "cudnn*cuda$($package.cudaLine)*windows*x86_64*.zip"
    )
  }

  $matches = New-Object System.Collections.Generic.List[System.IO.FileInfo]
  foreach ($pattern in $patterns) {
    foreach ($match in @(Get-ChildItem -LiteralPath $downloads -Filter $pattern -File -ErrorAction SilentlyContinue)) {
      $matches.Add($match)
    }
  }

  if ($matches.Count -eq 0) {
    return $null
  }

  return (@($matches | Sort-Object LastWriteTime, Length -Descending | Select-Object -First 1).FullName)
}

function Get-ArchiveEntries {
  param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath
  )

  $tarCommand = Get-Command tar -ErrorAction SilentlyContinue
  if (-not $tarCommand) {
    throw "tar was not found. Install an archive tool or provide an extracted source directory."
  }

  $output = & tar -tf $ArchivePath 2>&1
  if ($LASTEXITCODE -ne 0) {
    $text = (($output | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
    throw "Unable to list archive '$ArchivePath' with tar -tf. $text"
  }

  return @($output | ForEach-Object { ConvertTo-ForwardSlashPath -Value ([string]$_) } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Get-DirectoryEntries {
  param(
    [Parameter(Mandatory = $true)]
    [string]$DirectoryPath
  )

  $root = (Resolve-Path -LiteralPath $DirectoryPath).Path
  return @(
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
      ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($root, $_.FullName)
        ConvertTo-ForwardSlashPath -Value $relative
      }
  )
}

function New-SourceCatalog {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  $item = Get-Item -LiteralPath $Path -Force
  if ($item.PSIsContainer) {
    return [pscustomobject]@{
      path = $item.FullName
      kind = "directory"
      entries = @(Get-DirectoryEntries -DirectoryPath $item.FullName)
    }
  }

  return [pscustomobject]@{
    path = $item.FullName
    kind = "archive"
    entries = @(Get-ArchiveEntries -ArchivePath $item.FullName)
  }
}

function Test-EntryAllowedForKind {
  param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("TensorRT", "cuDNN")]
    [string]$Kind,
    [Parameter(Mandatory = $true)]
    [string]$Entry
  )

  $entryPath = ConvertTo-ForwardSlashPath -Value $Entry
  if ($entryPath -notmatch '\.dll$') {
    return $false
  }

  if ($Kind -eq "TensorRT") {
    $preferredPrefix = "TensorRT-$($package.tensorRtVersion)/bin/"
    return $entryPath.StartsWith($preferredPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
      $entryPath.StartsWith("bin/", [System.StringComparison]::OrdinalIgnoreCase) -or
      $entryPath.Contains("/bin/", [System.StringComparison]::OrdinalIgnoreCase)
  }

  $preferredPrefix = "cudnn_cuda$($package.cudaVersion)/libcudnn/bin/$($package.cudaVersion)/x64/"
  return $entryPath.StartsWith($preferredPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
    $entryPath.StartsWith("bin/", [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-ArchiveExtractionPrefix {
  param([string]$Entry)

  $entryPath = ConvertTo-ForwardSlashPath -Value $Entry
  $tensorRtPrefix = "TensorRT-$($package.tensorRtVersion)/bin"
  if ($entryPath.StartsWith("$tensorRtPrefix/", [System.StringComparison]::OrdinalIgnoreCase)) {
    return $tensorRtPrefix
  }

  $cudnnPrefix = "cudnn_cuda$($package.cudaVersion)/libcudnn/bin/$($package.cudaVersion)/x64"
  if ($entryPath.StartsWith("$cudnnPrefix/", [System.StringComparison]::OrdinalIgnoreCase)) {
    return $cudnnPrefix
  }

  return $entryPath
}

function Get-SourceMatches {
  param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("TensorRT", "cuDNN")]
    [string]$Kind,
    [Parameter(Mandatory = $true)]
    [object]$Catalog,
    [Parameter(Mandatory = $true)]
    [string]$RelativePath
  )

  $relativeForward = ConvertTo-ForwardSlashPath -Value $RelativePath
  $fileNamePattern = [System.IO.Path]::GetFileName((ConvertTo-LocalRelativePath -Value $relativeForward))
  if ([string]::IsNullOrWhiteSpace($fileNamePattern) -or $fileNamePattern -notlike "*.dll") {
    return @()
  }

  $matches = @(
    @($Catalog.entries) |
      Where-Object {
        $entry = [string]$_
        $fileName = [System.IO.Path]::GetFileName((ConvertTo-LocalRelativePath -Value $entry))
        $fileName -like $fileNamePattern -and (Test-EntryAllowedForKind -Kind $Kind -Entry $entry)
      } |
      Sort-Object
  )

  return @($matches)
}

$archiveExtractionCache = @{}

function Copy-CatalogEntry {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Catalog,
    [Parameter(Mandatory = $true)]
    [string]$Entry,
    [Parameter(Mandatory = $true)]
    [string]$DestinationPath
  )

  New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($DestinationPath)) -Force | Out-Null

  if ([string]$Catalog.kind -eq "directory") {
    $sourcePath = Join-Path ([string]$Catalog.path) (ConvertTo-LocalRelativePath -Value $Entry)
    Copy-Item -LiteralPath $sourcePath -Destination $DestinationPath -Force -ErrorAction Stop
    return
  }

  $extractionPrefix = Get-ArchiveExtractionPrefix -Entry $Entry
  $cacheKey = "$($Catalog.path)|$extractionPrefix"
  $tempRoot = $archiveExtractionCache[$cacheKey]
  if ([string]::IsNullOrWhiteSpace([string]$tempRoot) -or -not (Test-Path -LiteralPath ([string]$tempRoot) -PathType Container)) {
    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("jyppx-vendor-runtime-assets-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

    $extractOutput = & tar -xf ([string]$Catalog.path) -C $tempRoot $extractionPrefix 2>&1
    if ($LASTEXITCODE -ne 0) {
      $text = (($extractOutput | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
      throw "Unable to extract '$extractionPrefix' from '$($Catalog.path)' with tar -xf. $text"
    }

    $archiveExtractionCache[$cacheKey] = $tempRoot
  }

  $extractedPath = Join-Path ([string]$tempRoot) (ConvertTo-LocalRelativePath -Value $Entry)
  if (-not (Test-Path -LiteralPath $extractedPath -PathType Leaf)) {
    throw "Archive entry '$Entry' did not materialize under temporary extraction root '$tempRoot'."
  }

  Copy-Item -LiteralPath $extractedPath -Destination $DestinationPath -Force -ErrorAction Stop
}

function Test-RelativeAssetPresent {
  param(
    [string]$BaseRoot,
    [string]$RelativePath
  )

  if ([string]::IsNullOrWhiteSpace($BaseRoot) -or -not (Test-Path -LiteralPath $BaseRoot -PathType Container)) {
    return $false
  }

  $normalized = ConvertTo-LocalRelativePath -Value $RelativePath
  $path = Join-Path $BaseRoot $normalized
  if ($normalized.IndexOfAny(@('*', '?')) -ge 0) {
    return @(Resolve-Path -Path $path -ErrorAction SilentlyContinue).Count -gt 0
  }

  return Test-Path -LiteralPath $path -PathType Leaf
}

function Add-MaterializationRows {
  param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("TensorRT", "cuDNN")]
    [string]$Kind,
    [Parameter(Mandatory = $true)]
    [object]$Catalog,
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot,
    [Parameter(Mandatory = $true)]
    [object[]]$RelativeFiles,
    [Parameter(Mandatory = $true)]
    [AllowEmptyCollection()]
    [System.Collections.Generic.List[object]]$Rows
  )

  if ([string]::IsNullOrWhiteSpace($TargetRoot)) {
    foreach ($relativePath in @($RelativeFiles)) {
      $Rows.Add([pscustomobject]@{
          kind = $Kind
          relativePath = [string]$relativePath
          source = [string]$Catalog.path
          sourceEntry = ""
          destinationPath = ""
          status = "missing-target-root"
          bytes = 0
          matchedCount = 0
        })
    }

    return
  }

  foreach ($relativePath in @($RelativeFiles)) {
    if ([string]::IsNullOrWhiteSpace([string]$relativePath)) {
      continue
    }

    $matches = @(Get-SourceMatches -Kind $Kind -Catalog $Catalog -RelativePath ([string]$relativePath))
    if ($matches.Count -eq 0) {
      $Rows.Add([pscustomobject]@{
          kind = $Kind
          relativePath = [string]$relativePath
          source = [string]$Catalog.path
          sourceEntry = ""
          destinationPath = Join-Path $TargetRoot (ConvertTo-LocalRelativePath -Value ([string]$relativePath))
          status = "missing-source"
          bytes = 0
          matchedCount = 0
        })
      continue
    }

    foreach ($entry in $matches) {
      $destinationRelativePath = Join-Path ([System.IO.Path]::GetDirectoryName((ConvertTo-LocalRelativePath -Value ([string]$relativePath)))) ([System.IO.Path]::GetFileName((ConvertTo-LocalRelativePath -Value ([string]$entry))))
      $destinationPath = Join-Path $TargetRoot $destinationRelativePath
      $destinationExists = Test-Path -LiteralPath $destinationPath -PathType Leaf
      $status = "copied"

      if ($destinationExists -and -not $Force) {
        $status = "skipped-existing"
      }
      elseif ($DryRun) {
        $status = if ($destinationExists) { "dry-run-overwrite" } else { "dry-run-copy" }
      }
      else {
        Copy-CatalogEntry -Catalog $Catalog -Entry ([string]$entry) -DestinationPath $destinationPath
      }

      $length = 0
      if (Test-Path -LiteralPath $destinationPath -PathType Leaf) {
        $length = (Get-Item -LiteralPath $destinationPath).Length
      }

      $Rows.Add([pscustomobject]@{
          kind = $Kind
          relativePath = [string]$relativePath
          source = [string]$Catalog.path
          sourceEntry = [string]$entry
          destinationPath = $destinationPath
          status = $status
          bytes = $length
          matchedCount = $matches.Count
        })
    }
  }
}

$resolvedRoots = & (Join-Path $RepositoryRoot "eng\Resolve-RuntimeRoots.ps1") -RuntimePackageKey $RuntimePackageKey -RepositoryRoot $RepositoryRoot | ConvertFrom-Json
$TensorRtRoot = Resolve-TargetRoot -ExplicitRoot $TensorRtRoot -ResolvedRoot ([string]$resolvedRoots.tensorRtRoot) -PackageRootName ([string]$package.tensorRtPackageName)
$CudnnRoot = Resolve-TargetRoot -ExplicitRoot $CudnnRoot -ResolvedRoot ([string]$resolvedRoots.cudnnRoot) -PackageRootName ([string]$package.cudnnPackageName)

if ([string]::IsNullOrWhiteSpace($TensorRtSourcePath)) {
  $TensorRtSourcePath = Find-DefaultSourcePath -Kind "TensorRT"
}
if ([string]::IsNullOrWhiteSpace($CudnnSourcePath)) {
  $CudnnSourcePath = Find-DefaultSourcePath -Kind "cuDNN"
}

$TensorRtSourcePath = Resolve-SourcePath -Path $TensorRtSourcePath
$CudnnSourcePath = Resolve-SourcePath -Path $CudnnSourcePath

$rows = New-Object System.Collections.Generic.List[object]
$sourceDiagnostics = New-Object System.Collections.Generic.List[object]

if ([string]::IsNullOrWhiteSpace($TensorRtSourcePath)) {
  $sourceDiagnostics.Add([pscustomobject]@{ kind = "TensorRT"; status = "missing-source"; path = [string]$TensorRtSourcePath })
}
else {
  $tensorRtCatalog = New-SourceCatalog -Path $TensorRtSourcePath
  Add-MaterializationRows -Kind "TensorRT" -Catalog $tensorRtCatalog -TargetRoot $TensorRtRoot -RelativeFiles @($package.tensorRtFiles) -Rows $rows
}

if (@($package.cudnnFiles).Count -gt 0) {
  if ([string]::IsNullOrWhiteSpace($CudnnSourcePath)) {
    $sourceDiagnostics.Add([pscustomobject]@{ kind = "cuDNN"; status = "missing-source"; path = [string]$CudnnSourcePath })
  }
  else {
    $cudnnCatalog = New-SourceCatalog -Path $CudnnSourcePath
    Add-MaterializationRows -Kind "cuDNN" -Catalog $cudnnCatalog -TargetRoot $CudnnRoot -RelativeFiles @($package.cudnnFiles) -Rows $rows
  }
}

$expectedChecks = New-Object System.Collections.Generic.List[object]
foreach ($relativePath in @($package.tensorRtFiles)) {
  $expectedChecks.Add([pscustomobject]@{
      kind = "TensorRT"
      relativePath = [string]$relativePath
      present = Test-RelativeAssetPresent -BaseRoot $TensorRtRoot -RelativePath ([string]$relativePath)
    })
}
foreach ($relativePath in @($package.cudaFiles)) {
  $expectedChecks.Add([pscustomobject]@{
      kind = "CUDA"
      relativePath = [string]$relativePath
      present = Test-RelativeAssetPresent -BaseRoot ([string]$resolvedRoots.cudaRoot) -RelativePath ([string]$relativePath)
    })
}
foreach ($relativePath in @($package.cudnnFiles)) {
  $expectedChecks.Add([pscustomobject]@{
      kind = "cuDNN"
      relativePath = [string]$relativePath
      present = Test-RelativeAssetPresent -BaseRoot $CudnnRoot -RelativePath ([string]$relativePath)
    })
}

$missingExpectedChecks = @($expectedChecks | Where-Object { -not [bool]$_.present })
$materializationRows = @($rows.ToArray())
$copyFailureRows = @($materializationRows | Where-Object { [string]$_.status -eq "missing-source" -or [string]$_.status -eq "missing-target-root" })
$overallStatus = if ($DryRun) {
  if ($copyFailureRows.Count -eq 0 -and $sourceDiagnostics.Count -eq 0) { "dry-run-ready" } else { "blocked" }
}
elseif ($missingExpectedChecks.Count -eq 0 -and $copyFailureRows.Count -eq 0) {
  "ready"
}
else {
  "blocked"
}

$summary = [ordered]@{
  runtimeKey = [string]$package.key
  status = $overallStatus
  dryRun = [bool]$DryRun
  force = [bool]$Force
  tensorRtRoot = $TensorRtRoot
  cudaRoot = [string]$resolvedRoots.cudaRoot
  cudnnRoot = $CudnnRoot
  tensorRtSourcePath = $TensorRtSourcePath
  cudnnSourcePath = $CudnnSourcePath
  sourceDiagnostics = @($sourceDiagnostics.ToArray())
  materializedAssets = @($materializationRows)
  expectedChecks = @($expectedChecks.ToArray())
  missingExpectedCount = $missingExpectedChecks.Count
  copiedCount = @($materializationRows | Where-Object { [string]$_.status -eq "copied" }).Count
  skippedExistingCount = @($materializationRows | Where-Object { [string]$_.status -eq "skipped-existing" }).Count
  dryRunCopyCount = @($materializationRows | Where-Object { [string]$_.status -like "dry-run-*" }).Count
}

New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
$jsonPath = Join-Path $ReportDirectory "vendor-runtime-assets-summary.json"
$markdownPath = Join-Path $ReportDirectory "vendor-runtime-assets-summary.md"
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Windows Vendor Runtime Asset Materialization")
$lines.Add("")
$lines.Add("- Runtime key: ``$($summary.runtimeKey)``")
$lines.Add("- Status: ``$($summary.status)``")
$lines.Add("- Dry run: ``$($summary.dryRun)``")
$lines.Add("- TensorRT source: ``$($summary.tensorRtSourcePath)``")
$lines.Add("- cuDNN source: ``$($summary.cudnnSourcePath)``")
$lines.Add("- TensorRT root: ``$($summary.tensorRtRoot)``")
$lines.Add("- CUDA root: ``$($summary.cudaRoot)``")
$lines.Add("- cuDNN root: ``$($summary.cudnnRoot)``")
$lines.Add("- Missing expected assets after materialization: $($summary.missingExpectedCount)")
$lines.Add("")
$lines.Add("## Materialized Assets")
$lines.Add("")
$lines.Add("| Kind | Relative pattern | Status | Source entry | Destination |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($row in @($summary.materializedAssets)) {
  $sourceEntry = ([string]$row.sourceEntry) -replace '\|', '\|'
  $destination = ([string]$row.destinationPath) -replace '\|', '\|'
  $relative = ([string]$row.relativePath) -replace '\|', '\|'
  $lines.Add("| $($row.kind) | ``$relative`` | ``$($row.status)`` | ``$sourceEntry`` | ``$destination`` |")
}
$lines.Add("")
$lines.Add("## Expected Asset Check")
$lines.Add("")
$lines.Add("| Kind | Relative pattern | Present |")
$lines.Add("| --- | --- | --- |")
foreach ($check in @($summary.expectedChecks)) {
  $relative = ([string]$check.relativePath) -replace '\|', '\|'
  $lines.Add("| $($check.kind) | ``$relative`` | $($check.present) |")
}
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

$summary | ConvertTo-Json -Depth 12
foreach ($cachedRoot in @($archiveExtractionCache.Values)) {
  if (-not [string]::IsNullOrWhiteSpace([string]$cachedRoot) -and (Test-Path -LiteralPath ([string]$cachedRoot) -PathType Container)) {
    Remove-Item -LiteralPath ([string]$cachedRoot) -Recurse -Force
  }
}
if ($overallStatus -eq "blocked") {
  exit 1
}
