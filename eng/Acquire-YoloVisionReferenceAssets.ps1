[CmdletBinding()]
param(
  [string]$ManifestPath = "samples\assets\yolovision-reference-assets.json",
  [string]$SourceRoot = "",
  [string]$OutputRoot = "artifacts\yolovision\reference-assets",
  [string]$ReportPath = "",
  [switch]$VerifyOnly,
  [switch]$AllowDownload,
  [switch]$RequireLicenseReady
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
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

function Test-Sha256 {
  param([string]$Value)
  return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -match "^[a-fA-F0-9]{64}$"
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
    return $DefaultValue
  }

  $value = $Object.$Name
  if ($null -eq $value) {
    return $DefaultValue
  }

  return $value
}

function Resolve-SourceRoot {
  param([Parameter(Mandatory = $true)][object]$Manifest)

  if (-not [string]::IsNullOrWhiteSpace($SourceRoot)) {
    return [IO.Path]::GetFullPath($SourceRoot)
  }

  $environmentVariable = [string](Get-PropertyOrDefault -Object $Manifest -Name "sourceRootEnvironmentVariable" -DefaultValue "")
  if (-not [string]::IsNullOrWhiteSpace($environmentVariable)) {
    $environmentValue = [Environment]::GetEnvironmentVariable($environmentVariable)
    if (-not [string]::IsNullOrWhiteSpace($environmentValue) -and (Test-Path -LiteralPath $environmentValue -PathType Container)) {
      return [IO.Path]::GetFullPath($environmentValue)
    }
  }

  foreach ($hint in @(Get-PropertyOrDefault -Object $Manifest -Name "sourceRootHints" -DefaultValue @())) {
    $candidate = [Environment]::ExpandEnvironmentVariables([string]$hint)
    if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate -PathType Container)) {
      return [IO.Path]::GetFullPath($candidate)
    }
  }

  return ""
}

function Get-LicenseReady {
  param([AllowNull()][object]$License)

  $status = [string](Get-PropertyOrDefault -Object $License -Name "status" -DefaultValue "")
  $redistributionApproved = [bool](Get-PropertyOrDefault -Object $License -Name "redistributionApproved" -DefaultValue $false)
  return $redistributionApproved -and $status -in @("approved", "redistribution-approved", "owner-approved")
}

$resolvedManifestPath = Resolve-RepositoryPath -Path $ManifestPath
if (-not (Test-Path -LiteralPath $resolvedManifestPath -PathType Leaf)) {
  throw "YoloVision reference asset manifest not found: $resolvedManifestPath"
}

$manifest = Get-Content -LiteralPath $resolvedManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ([string](Get-PropertyOrDefault -Object $manifest -Name "recordKind" -DefaultValue "") -ne "yolovision-reference-asset-acquisition-manifest") {
  throw "Unexpected manifest recordKind in '$resolvedManifestPath'."
}

$resolvedOutputRoot = Resolve-RepositoryPath -Path $OutputRoot
New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
  $resolvedReportPath = Join-Path $resolvedOutputRoot "acquisition-report.json"
}
else {
  $resolvedReportPath = Resolve-RepositoryPath -Path $ReportPath
  New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedReportPath) -Force | Out-Null
}

$resolvedSourceRoot = Resolve-SourceRoot -Manifest $manifest
$assetResults = [Collections.Generic.List[object]]::new()
$failureReasons = [Collections.Generic.List[string]]::new()

foreach ($asset in @($manifest.assets)) {
  $id = [string](Get-PropertyOrDefault -Object $asset -Name "id" -DefaultValue "")
  $role = [string](Get-PropertyOrDefault -Object $asset -Name "role" -DefaultValue "")
  $relativeSourcePath = [string](Get-PropertyOrDefault -Object $asset -Name "relativeSourcePath" -DefaultValue "")
  $cacheFileName = [string](Get-PropertyOrDefault -Object $asset -Name "cacheFileName" -DefaultValue "")
  $expectedSha256 = ([string](Get-PropertyOrDefault -Object $asset -Name "expectedSha256" -DefaultValue "")).ToLowerInvariant()
  $expectedLength = [int64](Get-PropertyOrDefault -Object $asset -Name "expectedLength" -DefaultValue 0)
  $downloadUrl = [string](Get-PropertyOrDefault -Object $asset -Name "downloadUrl" -DefaultValue "")
  $licenseReady = Get-LicenseReady -License (Get-PropertyOrDefault -Object $asset -Name "license" -DefaultValue $null)

  $sourcePath = ""
  $sourceKind = "missing"
  if (-not [string]::IsNullOrWhiteSpace($resolvedSourceRoot) -and -not [string]::IsNullOrWhiteSpace($relativeSourcePath)) {
    $candidate = Join-Path $resolvedSourceRoot $relativeSourcePath
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
      $sourcePath = [IO.Path]::GetFullPath($candidate)
      $sourceKind = "local-installation"
    }
  }

  if ([string]::IsNullOrWhiteSpace($sourcePath) -and $AllowDownload -and $downloadUrl -match "^https?://") {
    $downloadDirectory = Join-Path $resolvedOutputRoot ".downloads"
    New-Item -ItemType Directory -Path $downloadDirectory -Force | Out-Null
    $downloadPath = Join-Path $downloadDirectory $cacheFileName
    Invoke-WebRequest -Uri $downloadUrl -OutFile $downloadPath
    $sourcePath = $downloadPath
    $sourceKind = "download"
  }

  $exists = -not [string]::IsNullOrWhiteSpace($sourcePath) -and (Test-Path -LiteralPath $sourcePath -PathType Leaf)
  $actualLength = if ($exists) { (Get-Item -LiteralPath $sourcePath).Length } else { 0 }
  $actualSha256 = if ($exists) { (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
  $lengthMatches = $exists -and $expectedLength -gt 0 -and $actualLength -eq $expectedLength
  $sha256Matches = $exists -and (Test-Sha256 -Value $expectedSha256) -and $actualSha256 -eq $expectedSha256
  $fileReady = $exists -and $lengthMatches -and $sha256Matches
  $destinationPath = if ([string]::IsNullOrWhiteSpace($cacheFileName)) { "" } else { Join-Path $resolvedOutputRoot $cacheFileName }
  $copied = $false

  if ($fileReady -and -not $VerifyOnly -and -not [string]::IsNullOrWhiteSpace($destinationPath)) {
    Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    $destinationHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($destinationHash -ne $expectedSha256) {
      throw "Copied asset '$id' failed destination SHA256 verification."
    }
    $copied = $true
  }

  $assetFailureReasons = [Collections.Generic.List[string]]::new()
  if (-not $exists) { $assetFailureReasons.Add("source file was not found") }
  if ($exists -and -not $lengthMatches) { $assetFailureReasons.Add("file length does not match the manifest") }
  if ($exists -and -not $sha256Matches) { $assetFailureReasons.Add("SHA256 does not match the manifest") }
  if (-not $licenseReady) { $assetFailureReasons.Add("license owner review is incomplete") }
  foreach ($reason in $assetFailureReasons) {
    $failureReasons.Add("${id}: $reason")
  }

  $assetResults.Add([pscustomobject][ordered]@{
    id = $id
    role = $role
    sourceKind = $sourceKind
    sourcePath = $sourcePath
    destinationPath = if ($copied) { [IO.Path]::GetFullPath($destinationPath) } else { "" }
    exists = $exists
    expectedLength = $expectedLength
    actualLength = $actualLength
    lengthMatches = $lengthMatches
    expectedSha256 = $expectedSha256
    actualSha256 = $actualSha256
    sha256Matches = $sha256Matches
    fileReady = $fileReady
    copied = $copied
    licenseReady = $licenseReady
    redistributionApproved = $licenseReady
    failureReasons = @($assetFailureReasons)
  }) | Out-Null
}

$fileFailureCount = @($assetResults | Where-Object { -not $_.fileReady }).Count
$licenseFailureCount = @($assetResults | Where-Object { -not $_.licenseReady }).Count
$allFilesReady = $fileFailureCount -eq 0 -and $assetResults.Count -gt 0
$allLicensesReady = $licenseFailureCount -eq 0 -and $assetResults.Count -gt 0
$state = if (-not $allFilesReady) {
  "asset-file-verification-failed"
}
elseif (-not $allLicensesReady) {
  "verified-local-source-owner-review-required"
}
else {
  "reference-assets-ready"
}

$record = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-reference-asset-acquisition-report"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  manifestPath = [IO.Path]::GetRelativePath($repositoryRoot, $resolvedManifestPath).Replace("\", "/")
  assetSetId = [string](Get-PropertyOrDefault -Object $manifest -Name "assetSetId" -DefaultValue "")
  acquisitionState = $state
  sourceRoot = $resolvedSourceRoot
  verifyOnly = $VerifyOnly.IsPresent
  allowDownload = $AllowDownload.IsPresent
  assetCount = $assetResults.Count
  fileFailureCount = $fileFailureCount
  licenseFailureCount = $licenseFailureCount
  allFilesReady = $allFilesReady
  allLicensesReady = $allLicensesReady
  canPromoteRealModelRuntime = $allFilesReady -and $allLicensesReady
  canRedistributeInRepository = $allLicensesReady
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  failureReasons = @($failureReasons)
  assets = @($assetResults)
  boundary = "File existence, length, and SHA256 verification do not approve asset redistribution. real-model-runtime promotion remains blocked until every asset license is owner-approved."
}

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $resolvedReportPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($resolvedReportPath, ".md")
$markdown = [Collections.Generic.List[string]]::new()
$markdown.Add("# YoloVision Reference Asset Acquisition")
$markdown.Add("")
$markdown.Add("- state: ``$state``")
$markdown.Add("- asset set: ``$($record.assetSetId)``")
$markdown.Add("- files ready: ``$allFilesReady``")
$markdown.Add("- licenses ready: ``$allLicensesReady``")
$markdown.Add("- can promote real-model-runtime: ``$($record.canPromoteRealModelRuntime)``")
$markdown.Add("- source root: ``$resolvedSourceRoot``")
$markdown.Add("")
$markdown.Add("| Asset | Role | Exists | Length | SHA256 | License | Copied |")
$markdown.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in $assetResults) {
  $markdown.Add("| ``$($item.id)`` | ``$($item.role)`` | ``$($item.exists)`` | ``$($item.lengthMatches)`` | ``$($item.sha256Matches)`` | ``$($item.licenseReady)`` | ``$($item.copied)`` |")
}
$markdown.Add("")
$markdown.Add("## Boundary")
$markdown.Add("")
$markdown.Add($record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "YoloVision reference asset acquisition report written."
Write-Host "JSON=$resolvedReportPath"
Write-Host "Markdown=$markdownPath"
Write-Host "AcquisitionState=$state FilesReady=$allFilesReady LicensesReady=$allLicensesReady CanPromoteRealModelRuntime=$($record.canPromoteRealModelRuntime)"

if (-not $allFilesReady) {
  throw "YoloVision reference asset file verification failed for $fileFailureCount asset(s)."
}

if ($RequireLicenseReady -and -not $allLicensesReady) {
  throw "YoloVision reference assets are hash-verified but license owner review remains incomplete for $licenseFailureCount asset(s)."
}
