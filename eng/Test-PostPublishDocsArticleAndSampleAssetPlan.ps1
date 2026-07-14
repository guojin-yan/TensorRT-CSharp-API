[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/post-publish-docs-article-and-sample-asset-plan.json",
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
  throw "Missing post-publish docs/article/sample asset plan: $InputPath"
}

$plan = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$planText = $plan | ConvertTo-Json -Depth 20
$lanes = @($plan.publicFacingAssetLanes)
$laneIds = @($lanes | ForEach-Object { [string]$_.id })
$sampleReadiness = $plan.sampleRenameReadiness

$requiredLaneIds = @(
  "article-roadmap-30plus",
  "article-publishing-readiness-map",
  "public-article-readiness-matrix",
  "release-docs-nuget-metadata-audit",
  "sample-yolovision-rename-readiness",
  "owner-proof-dependent-doc-assets",
  "post-publish-case-asset-pack"
)

$items = @(
  New-ValidationItem -Id "record-kind" -Passed ([string]$plan.recordKind -eq "post-publish-docs-article-and-sample-asset-plan") -Severity "blocker" -Detail "Plan must use post-publish-docs-article-and-sample-asset-plan recordKind."
  New-ValidationItem -Id "plan-state" -Passed ([string]$plan.planState -eq "blocked-owner-proof-required-non-proof-planning-ready") -Severity "blocker" -Detail "Plan must remain blocked on Owner proof while ready as planning."
  New-ValidationItem -Id "required-lanes" -Passed ((@($requiredLaneIds | Where-Object { $laneIds -notcontains $_ })).Count -eq 0) -Severity "blocker" -Detail "Plan must cover article roadmap, article readiness, public article matrix, docs metadata, sample naming, owner proof dependencies, and post-publish case assets."
  New-ValidationItem -Id "lane-counts" -Passed ([int]$plan.assetLaneCount -ge 7 -and [int]$plan.blockedAssetLaneCount -ge 6) -Severity "blocker" -Detail "Plan must expose a broad blocked asset lane surface."
  New-ValidationItem -Id "yolovision-sample-ready" -Passed ([string]$plan.allowedSampleName -eq "YoloVision" -and [string]$plan.forbiddenLegacySampleName -eq "YoloDet" -and [bool]$sampleReadiness.ready -and -not [bool]$sampleReadiness.legacyDirectoryExists -and [int]$plan.legacyYoloDetReferenceCount -eq 0) -Severity "blocker" -Detail "YoloVision must be the live sample identity and YoloDet must not appear in public-facing surfaces."
  New-ValidationItem -Id "owner-proof-requirements" -Passed ([int]$plan.requiredOwnerProofBeforePublishCount -ge 5 -and [int]$plan.requiredOwnerProofBeforeReleaseIssueCloseCount -ge 5 -and $planText.Contains("Owner public publish result") -and $planText.Contains("Repository-external clean consumer")) -Severity "blocker" -Detail "Plan must connect docs/assets to Owner public publish and post-publish clean consumer proof."
  New-ValidationItem -Id "source-artifacts-linked" -Passed ($planText.Contains("final-public-release-closure-bridge.json") -and $planText.Contains("strict-close-ready-convergence-dashboard.json") -and $planText.Contains("release-issue-close-owner-decision-input.json") -and $planText.Contains("samples/YoloVision/YoloVision.csproj")) -Severity "blocker" -Detail "Plan must link final close, strict close, release issue decision, and YoloVision sample sources."
  New-ValidationItem -Id "non-proof-flags" -Passed (-not [bool]$plan.performsPublish -and -not [bool]$plan.canPublishPublicly -and -not [bool]$plan.canCloseReleaseIssue -and -not [bool]$plan.canPromoteRuntimeProof -and -not [bool]$plan.canPromotePackageConsumerRuntime -and -not [bool]$plan.isRuntimeExecutionProof -and -not [bool]$plan.isPostPublishProof -and -not [bool]$plan.isReleaseCloseProof) -Severity "blocker" -Detail "Plan must not publish, close, or promote proof."
  New-ValidationItem -Id "lane-non-proof-flags" -Passed (@($lanes | Where-Object { [bool]$_.performsPublish -or [bool]$_.canPublishPublicly -or [bool]$_.canCloseReleaseIssue -or [bool]$_.isRuntimeExecutionProof -or [bool]$_.isPostPublishProof -or [bool]$_.isReleaseCloseProof }).Count -eq 0) -Severity "blocker" -Detail "Every asset lane must remain non-proof and non-publish."
  New-ValidationItem -Id "boundary-language" -Passed ($plan.boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $plan.boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $plan.boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $plan.boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $plan.boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must prevent proof, publish, and close promotion."
)

$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockerCount -eq 0) { "post-publish-docs-article-and-sample-asset-plan-passed-non-proof-boundaries-intact" } else { "failed-post-publish-docs-article-and-sample-asset-plan" }

$report = [pscustomobject]@{
  recordKind = "post-publish-docs-article-and-sample-asset-plan-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  sourcePlan = $InputPath
  planState = [string]$plan.planState
  assetLaneCount = [int]$plan.assetLaneCount
  blockedAssetLaneCount = [int]$plan.blockedAssetLaneCount
  sourceReadinessSignalCount = [int]$plan.sourceReadinessSignalCount
  requiredOwnerProofBeforePublishCount = [int]$plan.requiredOwnerProofBeforePublishCount
  requiredOwnerProofBeforeReleaseIssueCloseCount = [int]$plan.requiredOwnerProofBeforeReleaseIssueCloseCount
  legacyYoloDetReferenceCount = [int]$plan.legacyYoloDetReferenceCount
  sampleRenameReady = [bool]$sampleReadiness.ready
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
  boundary = "This validation checks post-publish docs/article/sample planning boundaries only: not runtime proof, not post-publish proof, not public package download proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-docs-article-and-sample-asset-plan-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-docs-article-and-sample-asset-plan-validation.md"
$report | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Post-Publish Docs, Article, And Sample Asset Plan Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- assetLaneCount: ``$($report.assetLaneCount)``
- blockedAssetLaneCount: ``$($report.blockedAssetLaneCount)``
- legacyYoloDetReferenceCount: ``$($report.legacyYoloDetReferenceCount)``
- sampleRenameReady: ``$($report.sampleRenameReady)``
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

Write-Output "Post-publish docs/article/sample asset plan validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) FailedBlockers=$($report.failedBlockerCount) LegacyYoloDet=$($report.legacyYoloDetReferenceCount)"

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Post-publish docs/article/sample asset plan validation failed with $failedBlockerCount blocker(s)."
}

