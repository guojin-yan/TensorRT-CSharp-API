[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-real-input-candidate-promotion-readiness.json",
  [string]$OutputRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

$scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
$RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-StringArray {
  param([AllowNull()][object]$Value)
  return @(ConvertTo-Array $Value | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
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

$inputFullPath = if ([System.IO.Path]::IsPathRooted($InputPath)) { $InputPath } else { Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Input file not found: $inputFullPath"
}

$record = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = [System.Collections.Generic.List[object]]::new()

$sourceRecords = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "sourceRecords" -DefaultValue @()))
$lanes = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "candidatePromotionLanes" -DefaultValue @()))
if ($lanes.Count -eq 0) {
  $lanes = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "candidatePromotionMatrix" -DefaultValue @()))
}
$sourceArtifacts = ConvertTo-StringArray (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-close-real-input-candidate-promotion-readiness") -Severity "blocker" -Detail "recordKind must be release-close-real-input-candidate-promotion-readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "readiness-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "readinessState" -DefaultValue "") -eq "blocked-release-close-real-input-candidate-promotion-real-owner-input-required") -Severity "blocker" -Detail "Readiness must remain blocked until real Owner inputs are imported and strict validators accept them.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-record-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "sourceRecordCount" -DefaultValue 0) -ge 10 -and $sourceRecords.Count -ge 10) -Severity "blocker" -Detail "Readiness must aggregate real Owner input, StrictClose, ReleaseClose, and final publish gate records.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "candidatePromotionLaneCount" -DefaultValue 0) -ge 8 -and $lanes.Count -ge 8) -Severity "blocker" -Detail "Candidate promotion readiness must contain the expected promotion lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-lanes-blocked" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedCandidatePromotionLaneCount" -DefaultValue -1) -eq $lanes.Count -and [int](Get-PropertyOrDefault -Object $record -Name "readyCandidatePromotionLaneCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Every candidate promotion lane must remain blocked until real Owner proof inputs pass strict validation.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-structural-blockers" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "failedBlockerCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "failedBlockerCount must remain zero while still not implying proof readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "action-required-present" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0) -gt 0) -Severity "blocker" -Detail "Action-required count must show real Owner evidence is still missing.")) | Out-Null
$items.Add((New-ValidationItem -Id "not-executed-by-automation" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false)) -Severity "blocker" -Detail "Readiness artifact must state it is not executed by automation.")) | Out-Null
$items.Add((New-ValidationItem -Id "does-not-publish" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -Severity "blocker" -Detail "Readiness artifact must not perform publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "does-not-promote-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -Severity "blocker" -Detail "Readiness artifact must not promote runtime proof, public publish, or release close.")) | Out-Null
$items.Add((New-ValidationItem -Id "not-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Readiness artifact must not classify itself as runtime, post-publish, or release-close proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-language" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not ready", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must explicitly reject proof, approval, package push, and failedBlockerCount=0 readiness.")) | Out-Null

$requiredLaneIds = @(
  "public-package-proof",
  "clean-external-consumer-runtime-proof",
  "post-publish-clean-consumer-proof",
  "hash-path-validation",
  "forbidden-substitute-validation",
  "strict-close-dry-run",
  "rollback-review",
  "final-close-decision",
  "release-close-strict-record-candidate",
  "final-release-close-record-real-validator",
  "final-publish-proof-gate"
)
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") })
foreach ($laneId in $requiredLaneIds) {
  $items.Add((New-ValidationItem -Id "lane-$laneId-present" -Passed ($laneIds -contains $laneId) -Severity "blocker" -Detail "Candidate promotion lanes must include $laneId.")) | Out-Null
}

foreach ($lane in $lanes) {
  $laneId = [string](Get-PropertyOrDefault -Object $lane -Name "laneId" -DefaultValue "unknown")
  $requiredRealInputs = @(ConvertTo-Array (Get-PropertyOrDefault -Object $lane -Name "requiredRealInputs" -DefaultValue @()))
  $laneBoundary = [string](Get-PropertyOrDefault -Object $lane -Name "boundary" -DefaultValue "")
  $nonSubstituteBoundary = [string](Get-PropertyOrDefault -Object $lane -Name "nonSubstituteBoundary" -DefaultValue "")
  $laneBlocksPromotion =
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "canPromote" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "canPublishPublicly" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "canCloseReleaseIssue" -DefaultValue $true))
  $laneProofFlagsClear =
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "performsPublish" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "isPostPublishProof" -DefaultValue $true)) -and
    (-not [bool](Get-PropertyOrDefault -Object $lane -Name "isReleaseCloseProof" -DefaultValue $true))
  $laneHasContracts =
    $requiredRealInputs.Count -gt 0 -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "strictValidator" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "promotionBlockedUntil" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace($nonSubstituteBoundary)
  $laneBoundaryValid =
    $laneBoundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and
    $laneBoundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and
    $laneBoundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and
    $laneBoundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and
    $laneBoundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)

  $items.Add((New-ValidationItem -Id "lane-$laneId-blocks-promotion" -Passed $laneBlocksPromotion -Severity "blocker" -Detail "Lane $laneId must not promote proof, public publish, or release close.")) | Out-Null
  $items.Add((New-ValidationItem -Id "lane-$laneId-proof-flags-clear" -Passed $laneProofFlagsClear -Severity "blocker" -Detail "Lane $laneId must not be classified as proof or package push.")) | Out-Null
  $items.Add((New-ValidationItem -Id "lane-$laneId-contracts-present" -Passed $laneHasContracts -Severity "blocker" -Detail "Lane $laneId must list required real inputs, strict validator, blocked-until rule, and non-substitute boundary.")) | Out-Null
  $items.Add((New-ValidationItem -Id "lane-$laneId-boundary-language" -Passed $laneBoundaryValid -Severity "blocker" -Detail "Lane $laneId boundary must reject proof, approval, close, and package push.")) | Out-Null
}

$requiredArtifacts = @(
  "artifacts/final-release/real-owner-evidence-strict-validator-orchestration.json",
  "artifacts/final-release/real-owner-evidence-strict-validator-orchestration-validation.json",
  "artifacts/final-release/owner-real-input-json-contract.json",
  "artifacts/final-release/owner-real-input-json-contract-validation.json",
  "artifacts/final-release/owner-real-input-json-import.json",
  "artifacts/final-release/owner-real-input-json-import-validation.json",
  "artifacts/final-release/owner-real-input-hash-and-path-validator.json",
  "artifacts/final-release/owner-real-input-hash-and-path-validator-validation.json",
  "artifacts/final-release/owner-real-input-forbidden-substitute-validator.json",
  "artifacts/final-release/owner-real-input-forbidden-substitute-validator-validation.json",
  "artifacts/final-release/strict-close-real-input-dry-run.json",
  "artifacts/final-release/strict-close-real-input-dry-run-validation.json",
  "artifacts/final-release/strict-close-real-input-finding-report.json",
  "artifacts/final-release/strict-close-real-input-finding-report-validation.json",
  "artifacts/final-release/strict-close-owner-action-pack.json",
  "artifacts/final-release/strict-close-owner-action-pack-validation.json",
  "artifacts/final-release/release-close-strict-record-candidate.json",
  "artifacts/final-release/release-close-strict-record-candidate-validation.json",
  "artifacts/final-release/final-release-close-record-real-validator-validation.json",
  "artifacts/final-release/final-publish-proof-gate-report.json"
)
foreach ($artifact in $requiredArtifacts) {
  $items.Add((New-ValidationItem -Id "source-$($artifact.Replace('.', '-').Replace('/', '-'))" -Passed ($sourceArtifacts -contains $artifact) -Severity "blocker" -Detail "Source artifact $artifact must be listed.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { $_.severity -eq "blocker" -and -not $_.passed })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-release-close-real-input-candidate-promotion-readiness" } else { "blocked-release-close-real-input-candidate-promotion-real-owner-input-required" }

$validation = [pscustomobject]@{
  recordKind = "release-close-real-input-candidate-promotion-readiness-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  sourceRecordCount = $sourceRecords.Count
  candidatePromotionLaneCount = $lanes.Count
  blockedCandidatePromotionLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCandidatePromotionLaneCount" -DefaultValue 0)
  readyCandidatePromotionLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "readyCandidatePromotionLaneCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0)
  validationItems = @($items)
  notExecutedByAutomation = [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $true)
  performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
  canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $false)
  canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
  canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)
  isRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $false)
  isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $false)
  isReleaseCloseProof = [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $false)
  boundary = $boundary
}

$jsonPath = Join-Path $OutputRoot "release-close-real-input-candidate-promotion-readiness-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-real-input-candidate-promotion-readiness-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# ReleaseClose Real Input Candidate Promotion Readiness Validation")
$lines.Add("")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- source records: ``$($validation.sourceRecordCount)``")
$lines.Add("- candidate promotion lanes: ``$($validation.candidatePromotionLaneCount)``")
$lines.Add("- blocked lanes: ``$($validation.blockedCandidatePromotionLaneCount)``")
$lines.Add("- ready lanes: ``$($validation.readyCandidatePromotionLaneCount)``")
$lines.Add("- failed blockers: ``$($validation.failedBlockerCount)``")
$lines.Add("- failed action-required: ``$($validation.failedActionRequiredCount)``")
$lines.Add("")
$lines.Add("| id | passed | severity | detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $items) {
  $detail = ([string]$item.detail).Replace("|", "\|")
  $lines.Add("| $($item.id) | $($item.passed) | $($item.severity) | $detail |")
}
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Output "ReleaseClose real input candidate promotion readiness validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState Lanes=$($validation.candidatePromotionLaneCount) Sources=$($validation.sourceRecordCount) FailedBlockers=$($validation.failedBlockerCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "ReleaseClose real input candidate promotion readiness validation failed with $($failedBlockers.Count) blocker(s)."
}
