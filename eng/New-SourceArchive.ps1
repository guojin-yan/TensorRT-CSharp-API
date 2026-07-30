[CmdletBinding()]
param(
  [string]$Version = "4.0.0",
  [string]$OutputDirectory,
  [switch]$AllowDirty,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\source"
}

$resolvedVersion = & (Join-Path $RepositoryRoot "eng\Resolve-PackageVersion.ps1") -RequestedVersion $Version
$dirtyFiles = @(& git -C $RepositoryRoot status --porcelain=v1 --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
  throw "Unable to read Git status for source archive creation."
}
if ($dirtyFiles.Count -gt 0 -and -not $AllowDirty.IsPresent) {
  throw "Source archive creation requires a clean tracked worktree. Commit the intended release state or pass -AllowDirty for a diagnostic archive of HEAD."
}

$commit = (& git -C $RepositoryRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($commit)) {
  throw "Unable to resolve source archive commit."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$archiveName = "TensorRtSharp4.0-source-$resolvedVersion.zip"
$archivePath = Join-Path $OutputDirectory $archiveName
if (Test-Path -LiteralPath $archivePath) {
  Remove-Item -LiteralPath $archivePath -Force
}

$prefix = "TensorRtSharp4.0-$resolvedVersion/"
& git -C $RepositoryRoot archive --format=zip "--prefix=$prefix" --output=$archivePath $commit
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
  throw "git archive failed for commit '$commit'."
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$policy = Get-Content -LiteralPath (Join-Path $RepositoryRoot "pack\external-vendor-runtime-policy.json") -Raw -Encoding utf8 | ConvertFrom-Json
$archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
  $forbiddenEntries = New-Object System.Collections.Generic.List[string]
  foreach ($entry in $archive.Entries) {
    $fileName = [IO.Path]::GetFileName($entry.FullName)
    foreach ($pattern in @($policy.forbiddenNativeFileNamePatterns)) {
      if ($fileName -match [string]$pattern) {
        $forbiddenEntries.Add($entry.FullName)
      }
    }
  }

  if ($forbiddenEntries.Count -gt 0) {
    Remove-Item -LiteralPath $archivePath -Force
    throw "Source archive contains forbidden NVIDIA runtime binaries: $($forbiddenEntries -join ', ')"
  }

  $entryCount = $archive.Entries.Count
}
finally {
  $archive.Dispose()
}

$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$result = [pscustomobject]@{
  recordKind = "source-archive"
  version = $resolvedVersion
  commit = $commit
  archivePath = $archivePath
  archiveName = $archiveName
  sha256 = $hash
  entryCount = $entryCount
  trackedFilesOnly = $true
  vendorRuntimeBinaryCount = 0
  dirtyTrackedWorktree = $dirtyFiles.Count -gt 0
  performsPublish = $false
}

$metadataPath = Join-Path $OutputDirectory "TensorRtSharp4.0-source-$resolvedVersion.json"
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8
$result | ConvertTo-Json -Depth 5
