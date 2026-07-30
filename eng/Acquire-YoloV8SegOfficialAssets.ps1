param(
  [string]$ManifestPath = "samples\assets\yolovision-yolov8n-seg-official-assets.json",
  [string]$OutputRoot = "",
  [string]$ReportPath = "artifacts\yolovision\yolov8n-seg-official-runtime\acquisition-report.json",
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

$resolvedManifestPath = Resolve-RepositoryPath -Path $ManifestPath
$manifest = Get-Content -LiteralPath $resolvedManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($manifest.recordKind -ne "yolovision-yolov8n-seg-official-asset-acquisition-manifest") {
  throw "Unexpected YOLOv8 segmentation acquisition manifest recordKind."
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $outerRoot "downloads\yolov8n-seg-ultralytics-v8.3.0"
}

$resolvedOutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (-not (Test-DriveIsNotC -Path $resolvedOutputRoot)) {
  throw "YOLOv8 segmentation assets must not be downloaded to the C drive: $resolvedOutputRoot"
}

$sourceRoot = Join-Path $resolvedOutputRoot "source"
New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null

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
      throw "YOLOv8 segmentation asset '$($asset.id)' is missing or invalid in offline mode: $destination"
    }

    Invoke-WebRequest -UseBasicParsing -Uri ([string]$asset.url) -OutFile $destination
    $downloaded = $true
    $identity = Get-FileIdentity -Path $destination
    $ready = $identity.length -eq [int64]$asset.expectedLength -and $identity.sha256 -eq [string]$asset.expectedSha256
  }

  if (-not $ready) {
    throw "YOLOv8 segmentation asset '$($asset.id)' failed length/SHA256 verification."
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

$compatibleInputs = [Collections.Generic.List[object]]::new()
$compatibleRoot = Join-Path $outerRoot "downloads\yolox-apache\derived"
foreach ($entry in @(
  @{ id = "yolox-coco-labels"; kind = "labels"; fileName = "coco.names"; expectedSha256 = [string]$manifest.compatibleRuntimeInputs.labels.expectedSha256 },
  @{ id = "yolox-dog-ppm"; kind = "image"; fileName = "dog.ppm"; expectedSha256 = [string]$manifest.compatibleRuntimeInputs.image.expectedSha256 }
)) {
  $path = Join-Path $compatibleRoot $entry.fileName
  $exists = Test-Path -LiteralPath $path -PathType Leaf
  $identity = if ($exists) { Get-FileIdentity -Path $path } else { $null }
  $verified = $exists -and $identity.sha256 -eq $entry.expectedSha256
  $compatibleInputs.Add([pscustomobject][ordered]@{
    id = $entry.id
    kind = $entry.kind
    path = [IO.Path]::GetFullPath($path)
    expectedSha256 = $entry.expectedSha256
    length = if ($identity -eq $null) { 0 } else { $identity.length }
    sha256 = if ($identity -eq $null) { "" } else { $identity.sha256 }
    verified = $verified
    sourceManifest = [string]$manifest.compatibleRuntimeInputs.source
  }) | Out-Null
}

$allCompatibleInputsReady = @($compatibleInputs | Where-Object { -not $_.verified }).Count -eq 0
$resolvedReportPath = Resolve-RepositoryPath -Path $ReportPath
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedReportPath) -Force | Out-Null
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-yolov8n-seg-official-asset-acquisition-report"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  acquisitionState = "official-release-assets-repository-hash-pinned-and-verified"
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
  compatibleRuntimeInputs = @($compatibleInputs)
  compatibleRuntimeInputsReady = $allCompatibleInputsReady
  canProceedToSourceTreeRuntime = $allCompatibleInputsReady
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  publicRedistributionOwnerApproval = $false
  canPublishPublicly = $false
  performsExport = $false
  performsRuntime = $false
  performsPublish = $false
  boundary = "Acquisition verifies the exact official Release asset ID against repository-pinned length/SHA256 and a source-commit license. The upstream Release did not publish a digest. This report is not export, real-model-runtime, package-consumer, redistribution, post-publish, or release proof."
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedReportPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($resolvedReportPath, ".md")
$markdown = @(
  "# YOLOv8n Segmentation Official Asset Acquisition",
  "",
  "- state: ``$($report.acquisitionState)``",
  "- release tag/id: ``$($report.upstreamReleaseTag)`` / ``$($report.upstreamReleaseId)``",
  "- source commit: ``$($report.upstreamSourceCommit)``",
  "- source assets: ``$($report.sourceAssetCount)``",
  "- E-drive output: ``$resolvedOutputRoot``",
  "- compatible runtime inputs ready: ``$allCompatibleInputsReady``",
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
  "## Compatible Inputs",
  ""
)
foreach ($item in $compatibleInputs) {
  $markdown += "- $($item.id): ``$($item.path)`` / verified=``$($item.verified)``"
}

$markdown += @(
  "",
  "## Boundary",
  "",
  $report.boundary
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "YOLOv8n segmentation official assets acquired and verified."
Write-Host "OutputRoot=$resolvedOutputRoot"
foreach ($item in $assetResults) {
  Write-Host "$($item.id)Sha256=$($item.sha256)"
}
Write-Host "CompatibleRuntimeInputsReady=$allCompatibleInputsReady"
Write-Host "Report=$resolvedReportPath"
