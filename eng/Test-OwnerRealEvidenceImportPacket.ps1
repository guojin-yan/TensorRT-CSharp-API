[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/final-release/owner-real-evidence-import-packet.json",
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
  throw "Missing owner real evidence import packet: $InputPath"
}

$packet = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$packetText = $packet | ConvertTo-Json -Depth 18
$lanes = @($packet.lanes)
$laneIds = @($lanes | ForEach-Object { [string]$_.laneId })
$requiredIds = @(
  "real-model-runtime-owner-proof-required",
  "package-consumer-runtime-owner-proof-required",
  "post-publish-verification-owner-proof-required",
  "final-owner-real-input-template-pack-owner-input-required",
  "owner-external-proof-result-import-owner-proof-required",
  "owner-result-candidate-bridge-real-proof-required"
)

$items = @(
  New-ValidationItem -Id "record-kind" -Passed ([string]$packet.recordKind -eq "owner-real-evidence-import-packet") -Severity "blocker" -Detail "Packet must use owner-real-evidence-import-packet recordKind."
  New-ValidationItem -Id "all-six-lanes-present" -Passed ((@($requiredIds | Where-Object { $laneIds -notcontains $_ })).Count -eq 0 -and @($lanes).Count -eq 6) -Severity "blocker" -Detail "Packet must contain all six final action-required lanes."
  New-ValidationItem -Id "lane-fields-files-hashes-present" -Passed (@($lanes | Where-Object { @($_.requiredFields).Count -lt 3 -or @($_.requiredFiles).Count -lt 3 -or @($_.requiredHashes).Count -lt 3 }).Count -eq 0) -Severity "blocker" -Detail "Each lane must include required fields, files, and hashes."
  New-ValidationItem -Id "lane-validators-present" -Passed (@($lanes | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.validator) -or @($_.validators).Count -lt 1 }).Count -eq 0) -Severity "blocker" -Detail "Each lane must include validator commands."
  New-ValidationItem -Id "owner-input-and-record-paths-present" -Passed (@($lanes | Where-Object { [string]::IsNullOrWhiteSpace([string]$_.ownerInputArtifact) -or [string]::IsNullOrWhiteSpace([string]$_.expectedRecord) }).Count -eq 0) -Severity "blocker" -Detail "Each lane must include owner input artifact and expected record."
  New-ValidationItem -Id "forbidden-substitutes-listed" -Passed ($packetText.Contains("template-only record") -and $packetText.Contains("dashboard-only record") -and $packetText.Contains("local feed") -and $packetText.Contains("ProjectReference") -and $packetText.Contains("direct nupkg")) -Severity "blocker" -Detail "Packet must explicitly list forbidden proof substitutes."
  New-ValidationItem -Id "flags-remain-false" -Passed (-not [bool]$packet.performsPublish -and -not [bool]$packet.canPublishPublicly -and -not [bool]$packet.canCloseReleaseIssue -and -not [bool]$packet.canPromotePackageConsumerRuntime -and -not [bool]$packet.canPromoteRuntimeProof) -Severity "blocker" -Detail "Packet must not publish, close, or promote proof."
)

$failedBlockerCount = @($items | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" }).Count
$failedActionRequiredCount = if ($failedBlockerCount -eq 0) { [int]@($lanes).Count } else { 0 }
$validationState = if ($failedBlockerCount -eq 0) { "blocked-owner-real-evidence-required-packet-shape-valid" } else { "failed-owner-real-evidence-import-packet" }

$report = [pscustomobject]@{
  recordKind = "owner-real-evidence-import-packet-validation"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  validationState = $validationState
  sourcePacket = $InputPath
  laneCount = @($lanes).Count
  failedBlockerCount = [int]$failedBlockerCount
  failedActionRequiredCount = [int]$failedActionRequiredCount
  validationItems = @($items)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  boundary = "This validation checks packet shape and non-proof boundaries only. It does not accept real proof without Owner files, logs, hashes, and strict validators."
}

$jsonPath = Join-Path $OutputRoot "owner-real-evidence-import-packet-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-real-evidence-import-packet-validation.md"
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $items) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Owner Real Evidence Import Packet Validation

Generated at: ``$($report.generatedAtUtc)``

## Summary

- validationState: ``$($report.validationState)``
- laneCount: ``$($report.laneCount)``
- failedBlockerCount: ``$($report.failedBlockerCount)``
- failedActionRequiredCount: ``$($report.failedActionRequiredCount)``
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

Write-Output "Owner real evidence import packet validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($report.validationState) FailedBlockerCount=$($report.failedBlockerCount) ActionRequired=$($report.failedActionRequiredCount)"

if ($Strict -and $failedBlockerCount -gt 0) {
  throw "Owner real evidence import packet validation failed with $failedBlockerCount blocker(s)."
}
