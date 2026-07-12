[CmdletBinding()]
param(
  [string]$RuntimeProofLaneDryRunSummaryPath = "artifacts\final-release\runtime-proof-lane-dry-run-summary.json",
  [string]$ReleaseCloseStrictBridgePath = "artifacts\final-release\release-close-strict-validation-bridge.json",
  [string]$ReleaseEvidenceBundlePath = "artifacts\final-release\release-evidence-bundle.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolved = Resolve-InputPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$laneDryRun = Read-JsonOrNull $RuntimeProofLaneDryRunSummaryPath
$bridge = Read-JsonOrNull $ReleaseCloseStrictBridgePath
$bundle = Read-JsonOrNull $ReleaseEvidenceBundlePath

$laneItems = @(Get-PropertyOrDefault -Object $laneDryRun -Name "laneItems" -DefaultValue @())
$bridgeItems = @(Get-PropertyOrDefault -Object $bridge -Name "bridgeItems" -DefaultValue @())

$closeDryRunItems = foreach ($laneItem in $laneItems) {
  $proofLane = [string](Get-PropertyOrDefault -Object $laneItem -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $bridgeItem = @($bridgeItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "proofLane" -DefaultValue "") -eq $proofLane } | Select-Object -First 1)[0]
  $remainingGaps = New-Object System.Collections.Generic.List[string]
  foreach ($field in @(Get-PropertyOrDefault -Object $laneItem -Name "realEvidenceFieldsMissing" -DefaultValue @())) { $remainingGaps.Add("missing real evidence field: $field") | Out-Null }
  foreach ($reason in @(Get-PropertyOrDefault -Object $laneItem -Name "bridgeBlockedReasons" -DefaultValue @())) { $remainingGaps.Add("bridge blocker: $reason") | Out-Null }
  foreach ($reason in @(Get-PropertyOrDefault -Object $bridgeItem -Name "blockedReasons" -DefaultValue @())) { $remainingGaps.Add("strict close blocker: $reason") | Out-Null }
  if ([string](Get-PropertyOrDefault -Object $bundle -Name "bundleState" -DefaultValue "") -ne "release-evidence-complete") {
    $remainingGaps.Add("release evidence bundle is not complete") | Out-Null
  }

  [pscustomobject]@{
    closeDryRunItemId = "$proofLane-release-close-strict-dry-run"
    proofLane = $proofLane
    executionInputId = [string](Get-PropertyOrDefault -Object $laneItem -Name "executionInputId" -DefaultValue "")
    laneState = [string](Get-PropertyOrDefault -Object $laneItem -Name "laneState" -DefaultValue "")
    bridgeState = [string](Get-PropertyOrDefault -Object $bridgeItem -Name "bridgeState" -DefaultValue "missing-release-close-strict-validation-bridge-item")
    closeDryRunItemState = "blocked-release-close-real-proof-required"
    remainingGaps = @($remainingGaps.ToArray())
    remainingGapCount = $remainingGaps.Count
    strictCloseCommand = [string](Get-PropertyOrDefault -Object $bridgeItem -Name "strictCloseCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady")
    requiredNextCommand = [string](Get-PropertyOrDefault -Object $laneItem -Name "requiredNextCommand" -DefaultValue "Fill real owner runtime proof result input and rerun strict validators.")
    closeImpact = "Cannot close release issue until this lane's gaps are resolved and strict close command passes."
    readyForReleaseClose = $false
    canCloseReleaseIssue = $false
    isReleaseCloseProof = $false
  }
}

$blockedItemCount = @($closeDryRunItems | Where-Object { [string]$_.closeDryRunItemState -eq "blocked-release-close-real-proof-required" }).Count
$readyItemCount = @($closeDryRunItems | Where-Object { [bool]$_.readyForReleaseClose }).Count
$remainingGapCount = ($closeDryRunItems | ForEach-Object { [int]$_.remainingGapCount } | Measure-Object -Sum).Sum

$summary = [pscustomobject]@{
  recordKind = "release-close-strict-dry-run-summary"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  closeDryRunState = "blocked-release-close-real-proof-required"
  closeDryRunItemCount = @($closeDryRunItems).Count
  blockedCloseDryRunItemCount = $blockedItemCount
  readyCloseDryRunItemCount = $readyItemCount
  remainingGapCount = [int]$remainingGapCount
  laneDryRunState = [string](Get-PropertyOrDefault -Object $laneDryRun -Name "dryRunState" -DefaultValue "missing-runtime-proof-lane-dry-run-summary")
  releaseCloseStrictBridgeState = [string](Get-PropertyOrDefault -Object $bridge -Name "bridgeState" -DefaultValue "missing-release-close-strict-validation-bridge")
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $bundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  closeDryRunItems = @($closeDryRunItems)
  sourceArtifacts = @(
    "artifacts/final-release/runtime-proof-lane-dry-run-summary.json",
    "artifacts/final-release/release-close-strict-validation-bridge.json",
    "artifacts/final-release/release-evidence-bundle.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This strict close dry-run summary reports remaining gaps only. It cannot substitute real proof, public publish approval, post-publish verification, rollback approval, or release issue close approval."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-dry-run-summary.json"
$markdownPath = Join-Path $OutputRoot "release-close-strict-dry-run-summary.md"
$summary | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Close Strict Dry-Run Summary")
$lines.Add("")
$lines.Add("`release-close-strict-dry-run-summary` 汇总每条 proof lane 到 strict release close 的剩余 gap。它不是 close approval。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| closeDryRunState | ``$(ConvertTo-MarkdownCell $summary.closeDryRunState)`` |")
$lines.Add("| closeDryRunItemCount | ``$($summary.closeDryRunItemCount)`` |")
$lines.Add("| blockedCloseDryRunItemCount | ``$($summary.blockedCloseDryRunItemCount)`` |")
$lines.Add("| readyCloseDryRunItemCount | ``$($summary.readyCloseDryRunItemCount)`` |")
$lines.Add("| remainingGapCount | ``$($summary.remainingGapCount)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($summary.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Close Dry-Run Items")
$lines.Add("")
$lines.Add("| Lane | State | Remaining Gaps | Strict Close Command |")
$lines.Add("| --- | --- | ---: | --- |")
foreach ($item in $closeDryRunItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.closeDryRunItemState) | ``$($item.remainingGapCount)`` | $(ConvertTo-MarkdownCell $item.strictCloseCommand) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($summary.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close strict dry-run summary written to $jsonPath"
Write-Host "Release close strict dry-run summary markdown written to $markdownPath"
Write-Host "CloseDryRunState=$($summary.closeDryRunState) Items=$($summary.closeDryRunItemCount) Blocked=$($summary.blockedCloseDryRunItemCount) RemainingGaps=$($summary.remainingGapCount)"
