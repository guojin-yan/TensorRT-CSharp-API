param(
  [string]$ManifestPath = "samples\assets\yolovision-yolox-official-assets.json",
  [string]$OutputRoot = "",
  [string]$ReportPath = "artifacts\yolovision\yolox-official-runtime\acquisition-report.json",
  [switch]$Offline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$outerRoot = Split-Path -Parent $repositoryRoot
$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }

  return [IO.Path]::GetFullPath((Join-Path $repositoryRoot $Path))
}

function Get-FileIdentity {
  param([Parameter(Mandatory = $true)][string]$Path)

  $item = Get-Item -LiteralPath $Path
  return [pscustomobject][ordered]@{
    length = $item.Length
    sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

function Write-P6Ppm {
  param(
    [Parameter(Mandatory = $true)][string]$SourcePath,
    [Parameter(Mandatory = $true)][string]$DestinationPath
  )

  Add-Type -AssemblyName System.Drawing.Common
  $bitmap = [Drawing.Bitmap]::new($SourcePath)
  try {
    $stream = [IO.File]::Create($DestinationPath)
    try {
      $header = [Text.Encoding]::ASCII.GetBytes("P6`n$($bitmap.Width) $($bitmap.Height)`n255`n")
      $stream.Write($header, 0, $header.Length)
      $row = [byte[]]::new($bitmap.Width * 3)
      for ($y = 0; $y -lt $bitmap.Height; $y++) {
        for ($x = 0; $x -lt $bitmap.Width; $x++) {
          $pixel = $bitmap.GetPixel($x, $y)
          $offset = $x * 3
          $row[$offset] = $pixel.R
          $row[$offset + 1] = $pixel.G
          $row[$offset + 2] = $pixel.B
        }

        $stream.Write($row, 0, $row.Length)
      }
    }
    finally {
      $stream.Dispose()
    }
  }
  finally {
    $bitmap.Dispose()
  }
}

function Write-CocoLabels {
  param(
    [Parameter(Mandatory = $true)][string]$SourcePath,
    [Parameter(Mandatory = $true)][string]$DestinationPath,
    [Parameter(Mandatory = $true)][int]$ExpectedCount
  )

  $source = Get-Content -LiteralPath $SourcePath -Raw -Encoding utf8
  $matches = [regex]::Matches($source, '(?m)^\s*"([^"]+)",\s*$')
  $labels = @($matches | ForEach-Object { $_.Groups[1].Value })
  if ($labels.Count -ne $ExpectedCount -or $labels[0] -ne "person" -or $labels[-1] -ne "toothbrush") {
    throw "Official COCO class source did not yield the expected $ExpectedCount ordered labels."
  }

  [IO.File]::WriteAllLines($DestinationPath, $labels, $utf8)
}

$resolvedManifestPath = Resolve-RepositoryPath -Path $ManifestPath
$manifest = Get-Content -LiteralPath $resolvedManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($manifest.recordKind -ne "yolovision-yolox-official-asset-acquisition-manifest") {
  throw "Unexpected YOLOX acquisition manifest recordKind."
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $outerRoot "downloads\yolox-apache"
}

$resolvedOutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if ([IO.Path]::GetPathRoot($resolvedOutputRoot).TrimEnd('\') -ieq "C:") {
  throw "YOLOX assets must not be downloaded to the C drive: $resolvedOutputRoot"
}

$sourceRoot = Join-Path $resolvedOutputRoot "source"
$derivedRoot = Join-Path $resolvedOutputRoot "derived"
New-Item -ItemType Directory -Path $sourceRoot, $derivedRoot -Force | Out-Null

$assetResults = [Collections.Generic.List[object]]::new()
foreach ($asset in @($manifest.assets)) {
  $destination = Join-Path $sourceRoot ([string]$asset.fileName)
  $downloaded = $false
  $ready = $false
  if (Test-Path -LiteralPath $destination -PathType Leaf) {
    $identity = Get-FileIdentity -Path $destination
    $ready = $identity.length -eq [int64]$asset.expectedLength -and $identity.sha256 -eq [string]$asset.expectedSha256
  }

  if (-not $ready) {
    if ($Offline) {
      throw "YOLOX asset '$($asset.id)' is missing or invalid in offline mode: $destination"
    }

    Invoke-WebRequest -Uri ([string]$asset.url) -OutFile $destination
    $downloaded = $true
    $identity = Get-FileIdentity -Path $destination
    $ready = $identity.length -eq [int64]$asset.expectedLength -and $identity.sha256 -eq [string]$asset.expectedSha256
  }

  if (-not $ready) {
    throw "YOLOX asset '$($asset.id)' failed length/SHA256 verification."
  }

  $assetResults.Add([pscustomobject][ordered]@{
    id = [string]$asset.id
    role = [string]$asset.role
    path = [IO.Path]::GetFullPath($destination)
    url = [string]$asset.url
    length = $identity.length
    sha256 = $identity.sha256
    verified = $true
    downloaded = $downloaded
  }) | Out-Null
}

$sourceById = @{}
foreach ($item in $assetResults) {
  $sourceById[$item.id] = $item.path
}

$ppmPath = Join-Path $derivedRoot ([string]$manifest.derivedAssets.image.fileName)
$labelsPath = Join-Path $derivedRoot ([string]$manifest.derivedAssets.labels.fileName)
Write-P6Ppm -SourcePath $sourceById[[string]$manifest.derivedAssets.image.sourceAssetId] -DestinationPath $ppmPath
Write-CocoLabels -SourcePath $sourceById[[string]$manifest.derivedAssets.labels.sourceAssetId] -DestinationPath $labelsPath -ExpectedCount ([int]$manifest.derivedAssets.labels.classCount)

$ppmIdentity = Get-FileIdentity -Path $ppmPath
$labelsIdentity = Get-FileIdentity -Path $labelsPath
$expectedPpmSha256 = [string]$manifest.derivedAssets.image.expectedSha256
$expectedLabelsSha256 = [string]$manifest.derivedAssets.labels.expectedSha256
$ppmMatches = [string]::IsNullOrWhiteSpace($expectedPpmSha256) -or $ppmIdentity.sha256 -eq $expectedPpmSha256
$labelsMatch = [string]::IsNullOrWhiteSpace($expectedLabelsSha256) -or $labelsIdentity.sha256 -eq $expectedLabelsSha256
if (-not $ppmMatches -or -not $labelsMatch) {
  throw "A derived YOLOX asset did not match its pinned SHA256."
}

$resolvedReportPath = Resolve-RepositoryPath -Path $ReportPath
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedReportPath) -Force | Out-Null
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-yolox-official-asset-acquisition-report"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  acquisitionState = "official-upstream-assets-verified"
  assetSetId = [string]$manifest.assetSetId
  upstreamRepository = [string]$manifest.upstreamRepository
  upstreamRevision = [string]$manifest.upstreamRevision
  upstreamTag = [string]$manifest.upstreamTag
  license = $manifest.license
  outputRoot = $resolvedOutputRoot
  cDriveOutputRejected = $true
  sourceAssetCount = $assetResults.Count
  sourceAssets = @($assetResults)
  derivedAssets = @(
    [pscustomobject][ordered]@{
      id = "yolox-dog-ppm"
      sourceAssetId = [string]$manifest.derivedAssets.image.sourceAssetId
      path = [IO.Path]::GetFullPath($ppmPath)
      format = [string]$manifest.derivedAssets.image.format
      length = $ppmIdentity.length
      sha256 = $ppmIdentity.sha256
      expectedSha256Matches = $ppmMatches
    },
    [pscustomobject][ordered]@{
      id = "yolox-coco-labels"
      sourceAssetId = [string]$manifest.derivedAssets.labels.sourceAssetId
      path = [IO.Path]::GetFullPath($labelsPath)
      format = "UTF-8 text, one class per line"
      classCount = [int]$manifest.derivedAssets.labels.classCount
      length = $labelsIdentity.length
      sha256 = $labelsIdentity.sha256
      expectedSha256Matches = $labelsMatch
    }
  )
  canProceedToSourceTreeRuntime = $true
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  publicRedistributionOwnerApproval = $false
  canPublishPublicly = $false
  performsPublish = $false
  boundary = "Acquisition and hash verification enable a source-tree runtime attempt. Only a successful real-model run plus strict evidence validation may promote source-tree real-model-runtime; this report does not approve public redistribution or package-consumer-runtime."
}

$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resolvedReportPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($resolvedReportPath, ".md")
$markdown = @(
  "# YOLOX Official Asset Acquisition",
  "",
  "- state: ``$($report.acquisitionState)``",
  "- upstream tag: ``$($report.upstreamTag)``",
  "- upstream revision: ``$($report.upstreamRevision)``",
  "- source assets: ``$($report.sourceAssetCount)``",
  "- E-drive output: ``$resolvedOutputRoot``",
  "- can proceed to source-tree runtime: ``True``",
  "- can publish publicly: ``False``",
  "",
  "## Derived Assets",
  "",
  "- PPM: ``$ppmPath`` / ``$($ppmIdentity.sha256)``",
  "- labels: ``$labelsPath`` / ``$($labelsIdentity.sha256)``",
  "",
  "## Boundary",
  "",
  $report.boundary
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "YOLOX official assets acquired and verified."
Write-Host "OutputRoot=$resolvedOutputRoot"
Write-Host "PpmSha256=$($ppmIdentity.sha256)"
Write-Host "LabelsSha256=$($labelsIdentity.sha256)"
Write-Host "Report=$resolvedReportPath"
