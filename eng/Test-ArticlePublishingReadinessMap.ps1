[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/article-publishing-readiness-map.json",
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
  throw "Missing article publishing readiness map: $InputPath"
}

$map = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$mapText = $map | ConvertTo-Json -Depth 18
$areas = @($map.coveredAreas | ForEach-Object { [string]$_ })
$articles = @($map.articles)

$items = @(
  New-ValidationItem -Id "record-kind" -Passed ([string]$map.recordKind -eq "article-publishing-readiness-map") -Severity "blocker" -Detail "Readiness map must use article-publishing-readiness-map recordKind."
  New-ValidationItem -Id "minimum-article-count" -Passed ([int]$map.roadmapArticleCount -ge 30) -Severity "blocker" -Detail "Roadmap should keep at least 30 planned articles."
  New-ValidationItem -Id "required-areas-present" -Passed ((@("YoloVision", "OnnxToEngine", "TensorRtExec", "RuntimePackages", "CleanConsumer", "PostPublish", "ReleaseClose") | Where-Object { $areas -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Readiness map must cover YoloVision, OnnxToEngine, TensorRtExec, runtime packages, clean consumer, PostPublish, and ReleaseClose."
  New-ValidationItem -Id "strict-order-linked" -Passed ($mapText.Contains("release-close-strict-proof-execution-order.json") -and [int]$map.strictExecutionStepCount -ge 6) -Severity "blocker" -Detail "Readiness map must link strict release close execution order."
  New-ValidationItem -Id "final-action-count-linked" -Passed ([int]$map.actionRequiredCount -eq 6) -Severity "blocker" -Detail "Readiness map must keep the six final action-required lanes visible."
  New-ValidationItem -Id "flags-remain-false" -Passed (-not [bool]$map.performsPublish -and -not [bool]$map.canPublishPublicly -and -not [bool]$map.canCloseReleaseIssue -and -not [bool]$map.canPromotePackageConsumerRuntime -and -not [bool]$map.canPromoteRuntimeProof) -Severity "blocker" -Detail "Readiness map must not claim publication, close, or proof promotion."
  New-ValidationItem -Id "forbidden-claims-listed" -Passed ($mapText.Contains("template is proof") -and $mapText.Contains("dashboard is proof") -and $mapText.Contains("build report is proof")) -Severity "blocker" -Detail "Readiness map must explicitly block template/dashboard/build report proof claims."
  New-ValidationItem -Id "articles-are-non-proof" -Passed (@($articles | Where-Object { [bool]$_.canPublishPublicly -or [bool]$_.isRuntimeExecutionProof -or [bool]$_.isPostPublishProof -or [bool]$_.isReleaseCloseProof }).Count -eq 0) -Severity "blocker" -Detail "Focused articles must remain non-proof until real validators pass."
)

$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockerCount -eq 0) { "article-publishing-readiness-map-passed-non-proof-boundaries-intact" } else { "failed-article-publishing-readiness-map" }

$report = [pscustomobject]@{
  recordKind = "article-publishing-readiness-map-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  sourceReadinessMap = $InputPath
  failedBlockerCount = [int]$failedBlockerCount
  validationItems = @($items)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  boundary = "This validation checks article readiness boundaries only. It does not publish, close, or promote proof."
}

$jsonPath = Join-Path $OutputRoot "article-publishing-readiness-map-validation.json"
$markdownPath = Join-Path $OutputRoot "article-publishing-readiness-map-validation.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Article Publishing Readiness Map Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Items

| Item | Passed | Severity | Detail |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($report.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Article publishing readiness map validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) FailedBlockerCount=$($report.failedBlockerCount)"

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Article publishing readiness map validation failed with $failedBlockerCount blocker(s)."
}
