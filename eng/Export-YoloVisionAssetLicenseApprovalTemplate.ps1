[CmdletBinding()]
param(
  [string]$ManifestPath = "samples\assets\yolovision-reference-assets.json",
  [string]$AcquisitionReportPath = "artifacts\yolovision\reference-assets\acquisition-report.json",
  [string]$OutputPath = "artifacts\yolovision\reference-assets\asset-license-approval-template.json",
  [string]$MarkdownPath = "artifacts\yolovision\reference-assets\asset-license-approval-template.md",
  [string]$RepositoryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }

  return [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
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

function Get-AssetReportById {
  param(
    [AllowNull()][object]$Report,
    [Parameter(Mandatory = $true)][string]$Id
  )

  foreach ($asset in @((Get-PropertyOrDefault -Object $Report -Name "assets" -DefaultValue @()))) {
    if ([string](Get-PropertyOrDefault -Object $asset -Name "id" -DefaultValue "") -eq $Id) {
      return $asset
    }
  }

  return $null
}

function New-ApprovalAsset {
  param(
    [Parameter(Mandatory = $true)][object]$ManifestAsset,
    [AllowNull()][object]$ReportAsset
  )

  $license = Get-PropertyOrDefault -Object $ManifestAsset -Name "license" -DefaultValue $null
  $id = [string](Get-PropertyOrDefault -Object $ManifestAsset -Name "id" -DefaultValue "")

  [pscustomobject][ordered]@{
    id = $id
    role = [string](Get-PropertyOrDefault -Object $ManifestAsset -Name "role" -DefaultValue "")
    cacheFileName = [string](Get-PropertyOrDefault -Object $ManifestAsset -Name "cacheFileName" -DefaultValue "")
    sourceUri = [string](Get-PropertyOrDefault -Object $ManifestAsset -Name "sourceUrl" -DefaultValue "")
    sourcePath = [string](Get-PropertyOrDefault -Object $ReportAsset -Name "sourcePath" -DefaultValue "")
    expectedSha256 = [string](Get-PropertyOrDefault -Object $ManifestAsset -Name "expectedSha256" -DefaultValue "")
    actualSha256 = [string](Get-PropertyOrDefault -Object $ReportAsset -Name "actualSha256" -DefaultValue "")
    expectedLength = [int64](Get-PropertyOrDefault -Object $ManifestAsset -Name "expectedLength" -DefaultValue 0)
    actualLength = [int64](Get-PropertyOrDefault -Object $ReportAsset -Name "actualLength" -DefaultValue 0)
    fileReady = [bool](Get-PropertyOrDefault -Object $ReportAsset -Name "fileReady" -DefaultValue $false)
    manifestLicenseName = [string](Get-PropertyOrDefault -Object $license -Name "name" -DefaultValue "")
    manifestLicenseUri = [string](Get-PropertyOrDefault -Object $license -Name "url" -DefaultValue "")
    ownerLicenseName = "owner-required"
    ownerLicenseUri = "owner-required"
    ownerRedistributionApproved = $false
    ownerPublicRepositoryApproved = $false
    ownerCommercialUseApproved = $false
    ownerApprovalEvidenceUri = "owner-required"
    ownerApprovalNotes = "owner-required; TensorRT SLA/local file hash verification is not a substitute for upstream model, labels, or image redistribution approval."
  }
}

$resolvedManifestPath = Resolve-RepositoryPath -Path $ManifestPath
$resolvedAcquisitionReportPath = Resolve-RepositoryPath -Path $AcquisitionReportPath
$resolvedOutputPath = Resolve-RepositoryPath -Path $OutputPath
$resolvedMarkdownPath = Resolve-RepositoryPath -Path $MarkdownPath

if (-not (Test-Path -LiteralPath $resolvedManifestPath -PathType Leaf)) {
  throw "YoloVision reference asset manifest not found: $resolvedManifestPath"
}

if (-not (Test-Path -LiteralPath $resolvedAcquisitionReportPath -PathType Leaf)) {
  throw "YoloVision reference asset acquisition report not found: $resolvedAcquisitionReportPath"
}

$manifest = Get-Content -LiteralPath $resolvedManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$acquisitionReport = Get-Content -LiteralPath $resolvedAcquisitionReportPath -Raw -Encoding utf8 | ConvertFrom-Json

if ([string](Get-PropertyOrDefault -Object $manifest -Name "recordKind" -DefaultValue "") -ne "yolovision-reference-asset-acquisition-manifest") {
  throw "Unexpected YoloVision reference asset manifest recordKind."
}

if ([string](Get-PropertyOrDefault -Object $acquisitionReport -Name "recordKind" -DefaultValue "") -ne "yolovision-reference-asset-acquisition-report") {
  throw "Unexpected YoloVision reference asset acquisition report recordKind."
}

$approvalAssets = @()
foreach ($asset in @($manifest.assets)) {
  $id = [string](Get-PropertyOrDefault -Object $asset -Name "id" -DefaultValue "")
  $approvalAssets += New-ApprovalAsset -ManifestAsset $asset -ReportAsset (Get-AssetReportById -Report $acquisitionReport -Id $id)
}

$allHashesMatchAcquisitionReport = $approvalAssets.Count -gt 0
foreach ($asset in @($approvalAssets)) {
  if ([string]::IsNullOrWhiteSpace([string]$asset.expectedSha256) -or
    [string]::IsNullOrWhiteSpace([string]$asset.actualSha256) -or
    [string]$asset.expectedSha256 -ne [string]$asset.actualSha256) {
    $allHashesMatchAcquisitionReport = $false
  }
}

$template = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-reference-asset-license-approval"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  assetSetId = [string](Get-PropertyOrDefault -Object $manifest -Name "assetSetId" -DefaultValue "")
  sampleName = "YoloVision"
  approvalState = "template-only-owner-action-required"
  proofClassification = "license-approval-template-only"
  sourceManifestPath = $ManifestPath.Replace("\", "/")
  sourceAcquisitionReportPath = $AcquisitionReportPath.Replace("\", "/")
  ownerName = "owner-required"
  ownerSignature = "owner-required"
  approvalDateUtc = "owner-required"
  approvalScope = "owner-required; review model weights, labels, and input image independently"
  redistributionAllowed = $false
  publicRepositoryAllowed = $false
  commercialUseAllowed = $false
  allAssetsApproved = $false
  allHashesMatchAcquisitionReport = $allHashesMatchAcquisitionReport
  canPromoteRealModelRuntime = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  forbiddenSubstitutes = @(
    "TensorRT SLA alone",
    "local installation file presence",
    "SHA256 match alone",
    "acquisition-report.json alone",
    "YoloVision Passed=True without license approval",
    "TensorRtExec report",
    "owner-required placeholders",
    "empty license URI"
  )
  validationRules = @(
    "Every asset must keep expectedSha256 aligned with the acquisition manifest and actualSha256 aligned with the acquisition report.",
    "Every asset must have ownerRedistributionApproved=true and ownerPublicRepositoryApproved=true before public publish can be considered.",
    "Model weights, COCO labels, and input image provenance must be reviewed independently; TensorRT SLA does not automatically approve upstream redistribution.",
    "ownerSignature, ownerName, approvalDateUtc, ownerLicenseName, ownerLicenseUri, and ownerApprovalEvidenceUri must be non-placeholder owner input.",
    "This record never performs publish and cannot close the release issue by itself."
  )
  assets = @($approvalAssets)
}

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedOutputPath) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedMarkdownPath) -Force | Out-Null
$template | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8

$lines = [Collections.Generic.List[string]]::new()
$lines.Add("# YoloVision Reference Asset License Approval Template")
$lines.Add("")
$lines.Add("- record kind: ``$($template.recordKind)``")
$lines.Add("- asset set: ``$($template.assetSetId)``")
$lines.Add("- approval state: ``$($template.approvalState)``")
$lines.Add("- proof classification: ``$($template.proofClassification)``")
$lines.Add("- can promote real-model-runtime: ``$($template.canPromoteRealModelRuntime)``")
$lines.Add("- can publish publicly: ``$($template.canPublishPublicly)``")
$lines.Add("")
$lines.Add("| Asset | Role | File Ready | Hash | Owner License | Redistribute | Public Repo |")
$lines.Add("| --- | --- | ---: | --- | --- | ---: | ---: |")
foreach ($asset in $template.assets) {
  $hash = if ($asset.expectedSha256 -eq $asset.actualSha256 -and -not [string]::IsNullOrWhiteSpace($asset.expectedSha256)) { "matched" } else { "pending" }
  $lines.Add("| ``$($asset.id)`` | ``$($asset.role)`` | ``$($asset.fileReady)`` | ``$hash`` | ``$($asset.ownerLicenseName)`` | ``$($asset.ownerRedistributionApproved)`` | ``$($asset.ownerPublicRepositoryApproved)`` |")
}
$lines.Add("")
$lines.Add("## Required Owner Inputs")
$lines.Add("")
$lines.Add("- ownerName, ownerSignature, approvalDateUtc, approvalScope")
$lines.Add("- ownerLicenseName, ownerLicenseUri, ownerApprovalEvidenceUri, ownerApprovalNotes for each asset")
$lines.Add("- explicit redistributionAllowed, publicRepositoryAllowed, and per-asset approval booleans")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add("Hash-ready local TensorRT sample files remain license-review assets until the owner supplies independent approval for the YOLO model, labels, and image. This template is not runtime proof, package-consumer proof, or public publish authorization.")
$lines | Set-Content -LiteralPath $resolvedMarkdownPath -Encoding utf8

Write-Host "YoloVision asset license approval template written."
Write-Host "JSON=$resolvedOutputPath"
Write-Host "Markdown=$resolvedMarkdownPath"
Write-Host "ApprovalState=$($template.approvalState) AssetCount=$(@($template.assets).Count) CanPromoteRealModelRuntime=$($template.canPromoteRealModelRuntime)"
