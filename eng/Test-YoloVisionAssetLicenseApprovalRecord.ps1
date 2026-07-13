[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\yolovision\reference-assets\asset-license-approval-template.json",
  [string]$ManifestPath = "samples\assets\yolovision-reference-assets.json",
  [string]$AcquisitionReportPath = "artifacts\yolovision\reference-assets\acquisition-report.json",
  [string]$OutputPath = "artifacts\yolovision\reference-assets\asset-license-approval-validation.json",
  [switch]$Strict,
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

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  return @($Value)
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or
    $text.Equals("owner-required", [StringComparison]::OrdinalIgnoreCase) -or
    $text.StartsWith("owner-required;", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Contains("template-only", [StringComparison]::OrdinalIgnoreCase) -or
    $text -like "<*>"
}

function Test-Sha256 {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-DateTimeOffset {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) { return $false }

  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject][ordered]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Add-ValidationItem {
  param(
    [Collections.Generic.List[object]]$Items,
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  $Items.Add((New-ValidationItem -Id $Id -Passed $Passed -Severity $Severity -Detail $Detail)) | Out-Null
}

function Get-AssetById {
  param(
    [AllowNull()][object]$Record,
    [Parameter(Mandatory = $true)][string]$Id
  )

  foreach ($asset in @(ConvertTo-Array (Get-PropertyOrDefault -Object $Record -Name "assets" -DefaultValue @()))) {
    if ([string](Get-PropertyOrDefault -Object $asset -Name "id" -DefaultValue "") -eq $Id) {
      return $asset
    }
  }

  return $null
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
$resolvedManifestPath = Resolve-RepositoryPath -Path $ManifestPath
$resolvedAcquisitionReportPath = Resolve-RepositoryPath -Path $AcquisitionReportPath
$resolvedOutputPath = Resolve-RepositoryPath -Path $OutputPath

foreach ($path in @($resolvedInputPath, $resolvedManifestPath, $resolvedAcquisitionReportPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required input was not found: $path"
  }
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $resolvedManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$acquisitionReport = Get-Content -LiteralPath $resolvedAcquisitionReportPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = [Collections.Generic.List[object]]::new()

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$assetSetId = [string](Get-PropertyOrDefault -Object $record -Name "assetSetId" -DefaultValue "")
$manifestAssetSetId = [string](Get-PropertyOrDefault -Object $manifest -Name "assetSetId" -DefaultValue "")
$reportAssetSetId = [string](Get-PropertyOrDefault -Object $acquisitionReport -Name "assetSetId" -DefaultValue "")
$assets = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "assets" -DefaultValue @()))
$manifestAssets = @(ConvertTo-Array (Get-PropertyOrDefault -Object $manifest -Name "assets" -DefaultValue @()))
$reportAssets = @(ConvertTo-Array (Get-PropertyOrDefault -Object $acquisitionReport -Name "assets" -DefaultValue @()))

Add-ValidationItem $items "record-kind" ($recordKind -eq "yolovision-reference-asset-license-approval") "blocker" "recordKind must be yolovision-reference-asset-license-approval."
Add-ValidationItem $items "sample-name" ([string](Get-PropertyOrDefault -Object $record -Name "sampleName" -DefaultValue "") -eq "YoloVision") "blocker" "sampleName must be YoloVision."
Add-ValidationItem $items "asset-set-id" (-not [string]::IsNullOrWhiteSpace($assetSetId) -and $assetSetId -eq $manifestAssetSetId -and $assetSetId -eq $reportAssetSetId) "blocker" "assetSetId must match the manifest and acquisition report."
Add-ValidationItem $items "asset-count" ($assets.Count -eq $manifestAssets.Count -and $assets.Count -eq $reportAssets.Count -and $assets.Count -gt 0) "blocker" "Approval record must cover every manifest/acquisition asset."
Add-ValidationItem $items "does-not-publish" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) "blocker" "License approval validation must not perform publish or close release."
Add-ValidationItem $items "global-owner-name" (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "ownerName" -DefaultValue ""))) "owner-action-required" "ownerName must be real owner input."
Add-ValidationItem $items "global-owner-signature" (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "ownerSignature" -DefaultValue ""))) "owner-action-required" "ownerSignature must be real owner input."
Add-ValidationItem $items "global-approval-date" (Test-DateTimeOffset -Value (Get-PropertyOrDefault -Object $record -Name "approvalDateUtc" -DefaultValue "")) "owner-action-required" "approvalDateUtc must be a parseable date/time."
Add-ValidationItem $items "global-approval-scope" (-not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $record -Name "approvalScope" -DefaultValue ""))) "owner-action-required" "approvalScope must be real owner input."

$allAssetsApproved = $true
$allHashesMatch = $true
foreach ($manifestAsset in $manifestAssets) {
  $id = [string](Get-PropertyOrDefault -Object $manifestAsset -Name "id" -DefaultValue "")
  $approvalAsset = Get-AssetById -Record $record -Id $id
  $reportAsset = Get-AssetById -Record $acquisitionReport -Id $id
  $expectedSha256 = [string](Get-PropertyOrDefault -Object $manifestAsset -Name "expectedSha256" -DefaultValue "")
  $reportActualSha256 = [string](Get-PropertyOrDefault -Object $reportAsset -Name "actualSha256" -DefaultValue "")
  $approvalExpectedSha256 = [string](Get-PropertyOrDefault -Object $approvalAsset -Name "expectedSha256" -DefaultValue "")
  $approvalActualSha256 = [string](Get-PropertyOrDefault -Object $approvalAsset -Name "actualSha256" -DefaultValue "")
  $hashMatches = (Test-Sha256 -Value $expectedSha256) -and $approvalExpectedSha256 -eq $expectedSha256 -and $approvalActualSha256 -eq $reportActualSha256 -and $approvalActualSha256 -eq $expectedSha256
  $fileReady = [bool](Get-PropertyOrDefault -Object $approvalAsset -Name "fileReady" -DefaultValue $false)
  $redistributionApproved = [bool](Get-PropertyOrDefault -Object $approvalAsset -Name "ownerRedistributionApproved" -DefaultValue $false)
  $publicRepositoryApproved = [bool](Get-PropertyOrDefault -Object $approvalAsset -Name "ownerPublicRepositoryApproved" -DefaultValue $false)
  $licenseNameReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $approvalAsset -Name "ownerLicenseName" -DefaultValue ""))
  $licenseUriReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $approvalAsset -Name "ownerLicenseUri" -DefaultValue ""))
  $evidenceReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $approvalAsset -Name "ownerApprovalEvidenceUri" -DefaultValue ""))
  $notesReady = -not (Test-IsPlaceholder -Value (Get-PropertyOrDefault -Object $approvalAsset -Name "ownerApprovalNotes" -DefaultValue ""))
  $assetApproved = $fileReady -and $hashMatches -and $redistributionApproved -and $publicRepositoryApproved -and $licenseNameReady -and $licenseUriReady -and $evidenceReady -and $notesReady

  if (-not $assetApproved) { $allAssetsApproved = $false }
  if (-not $hashMatches) { $allHashesMatch = $false }

  Add-ValidationItem $items "asset-$id-present" ($null -ne $approvalAsset) "blocker" "$id must be present in the approval record."
  Add-ValidationItem $items "asset-$id-file-ready" $fileReady "blocker" "$id must have passed file verification in acquisition."
  Add-ValidationItem $items "asset-$id-hash-match" $hashMatches "blocker" "$id expected/actual SHA256 must match manifest and acquisition report."
  Add-ValidationItem $items "asset-$id-owner-license-name" $licenseNameReady "owner-action-required" "$id ownerLicenseName must be real owner input."
  Add-ValidationItem $items "asset-$id-owner-license-uri" $licenseUriReady "owner-action-required" "$id ownerLicenseUri must be real owner input."
  Add-ValidationItem $items "asset-$id-redistribution-approved" $redistributionApproved "owner-action-required" "$id ownerRedistributionApproved must be true before promotion."
  Add-ValidationItem $items "asset-$id-public-repository-approved" $publicRepositoryApproved "owner-action-required" "$id ownerPublicRepositoryApproved must be true before public publish."
  Add-ValidationItem $items "asset-$id-approval-evidence" $evidenceReady "owner-action-required" "$id ownerApprovalEvidenceUri must be real owner input."
  Add-ValidationItem $items "asset-$id-approval-notes" $notesReady "owner-action-required" "$id ownerApprovalNotes must be real owner input."
}

$redistributionAllowed = [bool](Get-PropertyOrDefault -Object $record -Name "redistributionAllowed" -DefaultValue $false)
$publicRepositoryAllowed = [bool](Get-PropertyOrDefault -Object $record -Name "publicRepositoryAllowed" -DefaultValue $false)
$declaredCanPromoteRealModelRuntime = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$declaredCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
$forbiddenText = (@(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())) -join "`n")

Add-ValidationItem $items "forbidden-substitute-tensorrt-sla" ($forbiddenText.Contains("TensorRT SLA", [StringComparison]::OrdinalIgnoreCase)) "blocker" "forbiddenSubstitutes must state TensorRT SLA alone is not license approval."
Add-ValidationItem $items "forbidden-substitute-hash-only" ($forbiddenText.Contains("SHA256", [StringComparison]::OrdinalIgnoreCase)) "blocker" "forbiddenSubstitutes must state SHA256 match alone is not license approval."
Add-ValidationItem $items "declared-all-assets-approved-consistent" ([bool](Get-PropertyOrDefault -Object $record -Name "allAssetsApproved" -DefaultValue $false) -eq $allAssetsApproved) "blocker" "allAssetsApproved must match computed per-asset approval state."
Add-ValidationItem $items "declared-all-hashes-match-consistent" ([bool](Get-PropertyOrDefault -Object $record -Name "allHashesMatchAcquisitionReport" -DefaultValue $false) -eq $allHashesMatch) "blocker" "allHashesMatchAcquisitionReport must match computed hash state."

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedOwnerActions = @($items | Where-Object { -not $_.passed -and $_.severity -eq "owner-action-required" })
$canPromoteRealModelRuntime = $failedBlockers.Count -eq 0 -and $failedOwnerActions.Count -eq 0 -and $redistributionAllowed -and $allAssetsApproved -and $allHashesMatch
$canPublishPublicly = $canPromoteRealModelRuntime -and $publicRepositoryAllowed

Add-ValidationItem $items "declared-real-model-promotion-consistent" ($declaredCanPromoteRealModelRuntime -eq $canPromoteRealModelRuntime) "blocker" "canPromoteRealModelRuntime must match computed approval state."
Add-ValidationItem $items "declared-public-publish-consistent" ($declaredCanPublishPublicly -eq $canPublishPublicly) "blocker" "canPublishPublicly must match computed approval state."

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedOwnerActions = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "owner-action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid"
}
elseif ($failedOwnerActions.Count -gt 0) {
  "owner-action-required"
}
elseif ($canPublishPublicly) {
  "approved-for-public-repository"
}
else {
  "approved-for-local-real-model-runtime-only"
}

$summary = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-reference-asset-license-approval-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath.Replace("\", "/")
  manifestPath = $ManifestPath.Replace("\", "/")
  acquisitionReportPath = $AcquisitionReportPath.Replace("\", "/")
  validationState = $validationState
  assetSetId = $assetSetId
  assetCount = $assets.Count
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequiredCount = $failedOwnerActions.Count
  allAssetsApproved = $allAssetsApproved
  allHashesMatchAcquisitionReport = $allHashesMatch
  canPromoteRealModelRuntime = $canPromoteRealModelRuntime
  canPublishPublicly = $canPublishPublicly
  canCloseReleaseIssue = $false
  performsPublish = $false
  proofBoundary = "License approval validation only; hash-ready local assets and TensorRT SLA text cannot substitute for owner-approved model, labels, and image redistribution rights."
  validationItems = $validationItems
}

New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedOutputPath) -Force | Out-Null
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedOutputPath -Encoding utf8

$markdownPath = [IO.Path]::ChangeExtension($resolvedOutputPath, ".md")
$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# YoloVision Reference Asset License Approval Validation

| Field | Value |
| --- | --- |
| validationState | ``$($summary.validationState)`` |
| assetSetId | ``$($summary.assetSetId)`` |
| assetCount | ``$($summary.assetCount)`` |
| failedBlockerCount | ``$($summary.failedBlockerCount)`` |
| ownerActionRequiredCount | ``$($summary.ownerActionRequiredCount)`` |
| allAssetsApproved | ``$($summary.allAssetsApproved)`` |
| allHashesMatchAcquisitionReport | ``$($summary.allHashesMatchAcquisitionReport)`` |
| canPromoteRealModelRuntime | ``$($summary.canPromoteRealModelRuntime)`` |
| canPublishPublicly | ``$($summary.canPublishPublicly)`` |

## Validation Items

| ID | Passed | Severity | Detail |
| --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($summary.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "YoloVision asset license approval validation written."
Write-Host "JSON=$resolvedOutputPath"
Write-Host "Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) OwnerActionRequired=$($failedOwnerActions.Count) CanPromoteRealModelRuntime=$canPromoteRealModelRuntime CanPublishPublicly=$canPublishPublicly"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
