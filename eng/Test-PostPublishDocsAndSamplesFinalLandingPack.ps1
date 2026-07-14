[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/post-publish-docs-and-samples-final-landing-pack.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$inputFullPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Missing post-publish docs and samples final landing pack: $InputPath"
}

$pack = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$packText = $pack | ConvertTo-Json -Depth 24
$lanes = @($pack.lanes)
$laneIds = @($lanes | ForEach-Object { [string]$_.id })
$sampleStatus = $pack.sampleStatus
$packageMetadata = $pack.packageMetadata

$requiredLaneIds = @(
  "docs-site-final-links",
  "readme-frontpage-links",
  "nuget-metadata-owner-review",
  "release-notes-and-known-limitations",
  "yolovision-quickstart",
  "yolovision-real-asset-proof",
  "external-clean-install-guide",
  "article-case-asset-index",
  "owner-proof-dependency"
)

$items = @(
  New-ValidationItem -Id "record-kind" -Passed ([string]$pack.recordKind -eq "post-publish-docs-and-samples-final-landing-pack") -Severity "blocker" -Detail "Pack must use post-publish-docs-and-samples-final-landing-pack recordKind."
  New-ValidationItem -Id "pack-state" -Passed ([string]$pack.landingPackState -eq "blocked-owner-real-publish-evidence-required-final-landing-ready") -Severity "blocker" -Detail "Pack must remain blocked on real Owner publish evidence."
  New-ValidationItem -Id "required-lanes" -Passed ((@($requiredLaneIds | Where-Object { $laneIds -notcontains $_ })).Count -eq 0) -Severity "blocker" -Detail "Pack must include the nine final landing lanes."
  New-ValidationItem -Id "lane-counts" -Passed ([int]$pack.landingLaneCount -ge 9 -and [int]$pack.blockedLandingLaneCount -ge 8 -and [int]$pack.proofReadyLandingLaneCount -eq 0) -Severity "blocker" -Detail "Pack must expose broad blocked final landing lanes without proof-ready promotion."
  New-ValidationItem -Id "yolovision-no-yolodet" -Passed ([string]$pack.allowedSampleName -eq "YoloVision" -and [string]$pack.forbiddenLegacySampleName -eq "YoloDet" -and [bool]$pack.yoloVisionSampleReady -and [bool]$sampleStatus.yoloVisionSampleReady -and -not [bool]$sampleStatus.legacyDirectoryExists -and [int]$pack.legacyYoloDetReferenceCount -eq 0) -Severity "blocker" -Detail "YoloVision must be ready and live public-facing YoloDet references must remain zero."
  New-ValidationItem -Id "nuget-metadata" -Passed ([bool]$packageMetadata.packageMetadataReady -and [string]$packageMetadata.packPackageId -eq "JYPPX.TensorRT.CSharp.API" -and [string]$packageMetadata.sourcePackageId -eq "JYPPX.TensorRT.CSharp.API" -and [string]$packageMetadata.repositoryUrl -eq "https://github.com/guojin-yan/TensorRT-CSharp-API") -Severity "blocker" -Detail "Package metadata lane must keep the expected package id and repository URL."
  New-ValidationItem -Id "owner-proof-dependencies" -Passed ($packText.Contains("owner-real-publish-evidence-availability-ledger") -and $packText.Contains("post-publish-docs-article-and-sample-asset-plan") -and $packText.Contains("release-evidence-bundle.json") -and $packText.Contains("strict-close-ready-convergence-dashboard.json") -and $packText.Contains("final-public-release-closure-bridge.json")) -Severity "blocker" -Detail "Pack must link Owner ledger, docs/sample asset plan, release evidence bundle, strict close dashboard, and final closure bridge."
  New-ValidationItem -Id "publication-assets" -Passed ($packText.Contains("docs/index.md") -and $packText.Contains("README.md") -and $packText.Contains("samples/YoloVision/README.md") -and $packText.Contains("samples/assets/yolovision-article-case-pack.json") -and $packText.Contains("package-consumer-runtime-proof-clean-consumer-guide.md")) -Severity "blocker" -Detail "Pack must link final docs, README, YoloVision, article case, and clean install assets."
  New-ValidationItem -Id "owner-proof-requirements" -Passed ([int]$pack.requiredOwnerProofBeforePublishCount -ge 5 -and [int]$pack.requiredOwnerProofBeforeReleaseIssueCloseCount -ge 5 -and $packText.Contains("Owner public publish result") -and $packText.Contains("Repository-external clean consumer") -and $packText.Contains("Release issue close Owner decision")) -Severity "blocker" -Detail "Pack must enumerate publish and release-close Owner proof requirements."
  New-ValidationItem -Id "non-proof-flags" -Passed (-not [bool]$pack.performsPublish -and -not [bool]$pack.canPublishPublicly -and -not [bool]$pack.canCloseReleaseIssue -and -not [bool]$pack.canPromoteRuntimeProof -and -not [bool]$pack.canPromotePackageConsumerRuntime -and -not [bool]$pack.isRuntimeExecutionProof -and -not [bool]$pack.isPostPublishProof -and -not [bool]$pack.isReleaseCloseProof) -Severity "blocker" -Detail "Pack must not publish, close, or promote proof."
  New-ValidationItem -Id "lane-non-proof-flags" -Passed (@($lanes | Where-Object { [bool]$_.performsPublish -or [bool]$_.canPublishPublicly -or [bool]$_.canCloseReleaseIssue -or [bool]$_.canPromoteRuntimeProof -or [bool]$_.canPromotePackageConsumerRuntime -or [bool]$_.isRuntimeExecutionProof -or [bool]$_.isPostPublishProof -or [bool]$_.isReleaseCloseProof -or [bool]$_.proofReady }).Count -eq 0) -Severity "blocker" -Detail "Every landing lane must remain non-proof and non-publish."
  New-ValidationItem -Id "boundary-language" -Passed ($pack.boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $pack.boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $pack.boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $pack.boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $pack.boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must prevent proof, publish, and close promotion."
)

$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockerCount -eq 0) { "post-publish-docs-and-samples-final-landing-pack-passed-non-proof-boundaries-intact" } else { "failed-post-publish-docs-and-samples-final-landing-pack" }

$report = [pscustomobject]@{
  recordKind = "post-publish-docs-and-samples-final-landing-pack-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  sourcePack = $InputPath
  landingPackState = [string]$pack.landingPackState
  landingLaneCount = [int]$pack.landingLaneCount
  readyLandingLaneCount = [int]$pack.readyLandingLaneCount
  blockedLandingLaneCount = [int]$pack.blockedLandingLaneCount
  proofReadyLandingLaneCount = [int]$pack.proofReadyLandingLaneCount
  ownerProofRequiredLaneCount = [int]$pack.ownerProofRequiredLaneCount
  requiredOwnerProofBeforePublishCount = [int]$pack.requiredOwnerProofBeforePublishCount
  requiredOwnerProofBeforeReleaseIssueCloseCount = [int]$pack.requiredOwnerProofBeforeReleaseIssueCloseCount
  yoloVisionSampleReady = [bool]$pack.yoloVisionSampleReady
  legacyYoloDetReferenceCount = [int]$pack.legacyYoloDetReferenceCount
  packageMetadataReady = [bool]$packageMetadata.packageMetadataReady
  failedBlockerCount = [int]$failedBlockerCount
  validationItems = @($items)
  notExecutedByAutomation = $true
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePackageConsumerRuntime = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This validation checks the post-publish docs and samples final landing pack boundaries only: not runtime proof, not post-publish proof, not public package download proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-docs-and-samples-final-landing-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-docs-and-samples-final-landing-pack-validation.md"
$report | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Post-Publish Docs And Samples Final Landing Pack Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- landingLaneCount: ``$($report.landingLaneCount)``
- blockedLandingLaneCount: ``$($report.blockedLandingLaneCount)``
- proofReadyLandingLaneCount: ``$($report.proofReadyLandingLaneCount)``
- yoloVisionSampleReady: ``$($report.yoloVisionSampleReady)``
- legacyYoloDetReferenceCount: ``$($report.legacyYoloDetReferenceCount)``
- packageMetadataReady: ``$($report.packageMetadataReady)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- isPostPublishProof: ``False``

## Items

| Item | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Post-publish docs and samples final landing pack validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) FailedBlockers=$($report.failedBlockerCount) LegacyYoloDet=$($report.legacyYoloDetReferenceCount)"

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Post-publish docs and samples final landing pack validation failed with $failedBlockerCount blocker(s)."
}
