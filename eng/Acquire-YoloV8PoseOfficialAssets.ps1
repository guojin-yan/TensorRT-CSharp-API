param(
  [string]$ManifestPath = "samples\assets\yolovision-yolov8n-pose-official-assets.json",
  [string]$OutputRoot = "",
  [string]$ReportPath = "artifacts\yolovision\yolov8n-pose-official-runtime\acquisition-report.json",
  [string]$PythonPath = "",
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

function Test-DriveIsNotC {
  param([Parameter(Mandatory = $true)][string]$Path)

  return [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path)).TrimEnd('\') -ine "C:"
}

function Resolve-PythonPath {
  if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
    return $PythonPath
  }

  $configured = [Environment]::GetEnvironmentVariable("JYPPX_YOLO_PYTHON")
  if (-not [string]::IsNullOrWhiteSpace($configured)) {
    return $configured
  }

  $command = Get-Command python -ErrorAction SilentlyContinue
  if ($null -eq $command) {
    throw "Python with Pillow is required to derive the P6 RGB PPM input. Set -PythonPath or JYPPX_YOLO_PYTHON."
  }

  return $command.Source
}

$resolvedManifestPath = Resolve-RepositoryPath -Path $ManifestPath
$manifest = Get-Content -LiteralPath $resolvedManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($manifest.recordKind -ne "yolovision-yolov8n-pose-official-asset-acquisition-manifest") {
  throw "Unexpected YOLOv8 pose acquisition manifest recordKind."
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $outerRoot "downloads\yolov8n-pose-ultralytics-v8.3.0"
}

$resolvedOutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (-not (Test-DriveIsNotC -Path $resolvedOutputRoot)) {
  throw "YOLOv8 pose assets must not be downloaded to the C drive: $resolvedOutputRoot"
}

$sourceRoot = Join-Path $resolvedOutputRoot "source"
$derivedRoot = Join-Path $resolvedOutputRoot "derived"
New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null
New-Item -ItemType Directory -Path $derivedRoot -Force | Out-Null

$assetResults = [Collections.Generic.List[object]]::new()
foreach ($asset in @($manifest.assets)) {
  $destination = Join-Path $sourceRoot ([string]$asset.fileName)
  $downloaded = $false
  $identity = $null
  $ready = $false
  if (Test-Path -LiteralPath $destination -PathType Leaf) {
    $identity = Get-FileIdentity -Path $destination
    $ready = $identity.length -eq [int64]$asset.expectedLength -and $identity.sha256 -eq [string]$asset.expectedSha256
  }

  if (-not $ready) {
    if ($Offline) {
      throw "YOLOv8 pose asset '$($asset.id)' is missing or invalid in offline mode: $destination"
    }

    $downloadUrl = [string]$asset.url
    if ($asset.PSObject.Properties.Name -contains "downloadFallbackUrl" -and -not [string]::IsNullOrWhiteSpace([string]$asset.downloadFallbackUrl)) {
      $downloadUrl = [string]$asset.downloadFallbackUrl
    }

    Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $destination
    $downloaded = $true
    $identity = Get-FileIdentity -Path $destination
    $ready = $identity.length -eq [int64]$asset.expectedLength -and $identity.sha256 -eq [string]$asset.expectedSha256
  }

  if (-not $ready) {
    throw "YOLOv8 pose asset '$($asset.id)' failed length/SHA256 verification."
  }

  $assetResults.Add([pscustomobject][ordered]@{
    id = [string]$asset.id
    role = [string]$asset.role
    path = [IO.Path]::GetFullPath($destination)
    url = [string]$asset.url
    length = $identity.length
    sha256 = $identity.sha256
    hashProvenance = [string]$asset.hashProvenance
    verified = $true
    downloaded = $downloaded
  }) | Out-Null
}

$sourceImage = Join-Path $sourceRoot "bus.jpg"
$derivedImage = Join-Path $derivedRoot ([string]$manifest.derivedInput.fileName)
$resolvedPythonPath = Resolve-PythonPath
$conversionCode = "from PIL import Image; import sys; image=Image.open(sys.argv[1]).convert('RGB'); image.save(sys.argv[2], format='PPM'); print(f'{image.width}x{image.height}')"
$derivedDimensions = (& $resolvedPythonPath -c $conversionCode $sourceImage $derivedImage | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0) {
  throw "Python/Pillow failed to derive the pose PPM input."
}

$derivedIdentity = Get-FileIdentity -Path $derivedImage
$derivedReady = $derivedIdentity.length -eq [int64]$manifest.derivedInput.expectedLength -and
  $derivedIdentity.sha256 -eq [string]$manifest.derivedInput.expectedSha256 -and
  $derivedDimensions -eq "$($manifest.derivedInput.width)x$($manifest.derivedInput.height)"
if (-not $derivedReady) {
  throw "Derived YOLOv8 pose PPM input failed dimensions/length/SHA256 verification."
}

$resolvedReportPath = Resolve-RepositoryPath -Path $ReportPath
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedReportPath) -Force | Out-Null
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-yolov8n-pose-official-asset-acquisition-report"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  acquisitionState = "official-release-and-source-assets-repository-hash-pinned-and-verified"
  assetSetId = [string]$manifest.assetSetId
  upstreamAssetsRepository = [string]$manifest.upstreamAssetsRepository
  upstreamReleaseTag = [string]$manifest.upstreamReleaseTag
  upstreamReleaseId = [int64]$manifest.upstreamReleaseId
  upstreamSourceRepository = [string]$manifest.upstreamSourceRepository
  upstreamSourceCommit = [string]$manifest.upstreamSourceCommit
  license = $manifest.license
  modelContract = $manifest.modelContract
  exportContract = $manifest.exportContract
  outputRoot = $resolvedOutputRoot
  cDriveOutputRejected = $true
  sourceAssetCount = $assetResults.Count
  sourceAssets = @($assetResults)
  derivedInput = [pscustomobject][ordered]@{
    path = [IO.Path]::GetFullPath($derivedImage)
    sourcePath = [IO.Path]::GetFullPath($sourceImage)
    dimensions = $derivedDimensions
    length = $derivedIdentity.length
    sha256 = $derivedIdentity.sha256
    verified = $derivedReady
    derivation = [string]$manifest.derivedInput.derivation
  }
  canProceedToSourceTreeRuntime = $derivedReady
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  publicRedistributionOwnerApproval = $false
  canPublishPublicly = $false
  performsExport = $false
  performsRuntime = $false
  performsPublish = $false
  boundary = "Acquisition verifies the exact official Release asset ID, source-commit image/license, and derived PPM against repository-pinned lengths/SHA256 values. The upstream Release did not publish a digest. This report is not export, real-model-runtime, package-consumer, redistribution, post-publish, or release proof."
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedReportPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($resolvedReportPath, ".md")
$markdown = @(
  "# YOLOv8n Pose Official Asset Acquisition",
  "",
  "- state: ``$($report.acquisitionState)``",
  "- release tag/id: ``$($report.upstreamReleaseTag)`` / ``$($report.upstreamReleaseId)``",
  "- source commit: ``$($report.upstreamSourceCommit)``",
  "- source assets: ``$($report.sourceAssetCount)``",
  "- E-drive output: ``$resolvedOutputRoot``",
  "- derived input: ``$derivedImage`` / ``$($derivedIdentity.sha256)``",
  "- can proceed to source-tree runtime: ``$($report.canProceedToSourceTreeRuntime)``",
  "- can publish publicly: ``False``",
  "",
  "## Source Assets",
  ""
)
foreach ($item in $assetResults) {
  $markdown += "- $($item.id): ``$($item.path)`` / ``$($item.sha256)`` / ``$($item.hashProvenance)``"
}

$markdown += @(
  "",
  "## Boundary",
  "",
  $report.boundary
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "YOLOv8n pose official assets acquired and verified."
Write-Host "OutputRoot=$resolvedOutputRoot"
foreach ($item in $assetResults) {
  Write-Host "$($item.id)Sha256=$($item.sha256)"
}
Write-Host "DerivedInputSha256=$($derivedIdentity.sha256)"
Write-Host "CanProceedToSourceTreeRuntime=$($report.canProceedToSourceTreeRuntime)"
Write-Host "Report=$resolvedReportPath"
