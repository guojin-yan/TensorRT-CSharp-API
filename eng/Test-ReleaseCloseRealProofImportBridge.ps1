[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-real-proof-import-bridge.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Release close real proof import bridge not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "bridgeLanes" -DefaultValue @()))
$blockedLanes = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$requiredLanes = @("public-publish-real-result-record", "post-publish-clean-consumer-proof-record", "forbidden-substitute-scan", "post-publish-verification", "strict-owner-decision-import", "release-issue-close-record")
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$summary = Get-PropertyOrDefault -Object $record -Name "summary" -DefaultValue $null
$forbiddenSubstituteMarkers = @((Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteMarkers" -DefaultValue @()))

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-close-real-proof-import-bridge") -Severity "blocker" -Detail "recordKind must be release-close-real-proof-import-bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "") -eq "blocked-release-close-real-proof-import-required") -Severity "blocker" -Detail "Bridge must stay blocked until real proof import records pass.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes-present" -Passed (@($requiredLanes | Where-Object { $laneIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Bridge must expose all real proof import lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "summary-projected" -Passed ($null -ne $summary -and [bool](Get-PropertyOrDefault -Object $summary -Name "bridgeInputOnly" -DefaultValue $false) -and [int](Get-PropertyOrDefault -Object $summary -Name "blockedLaneCount" -DefaultValue -1) -eq $blockedLanes.Count -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "proofPromotionAllowed" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "releaseCloseAllowed" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must summarize blocked lanes as owner-input-only and non-promotable.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitute-markers-projected" -Passed ((@("local feed", "ProjectReference", "direct .nupkg", "build-only", "dry-run", "candidate", "dashboard", "blocked-by-driver") | Where-Object { $forbiddenSubstituteMarkers -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Bridge must carry forbidden substitute markers into release-close evidence.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-blocked-reasons-and-validators" -Passed (@($lanes | Where-Object { [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "blockedReason" -DefaultValue "")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "validatorCommand" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Each bridge lane must expose owner-action blocked reasons and validator commands.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-required" -Passed ($blockedLanes.Count -eq 0) -Severity "action-required" -Detail "Owner must fill real proof records and pass strict validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not publish, approve, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-release-close-real-proof-import-bridge" } else { "blocked-release-close-real-proof-import-required" }

$validation = [pscustomobject]@{
  recordKind = "release-close-real-proof-import-bridge-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  laneCount = $lanes.Count
  blockedLaneCount = $blockedLanes.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  validationItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "Validation checks release close real proof import bridge shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-close-real-proof-import-bridge-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-real-proof-import-bridge-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Release Close Real Proof Import Bridge Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| laneCount | ``$($validation.laneCount)`` |",
  "| blockedLaneCount | ``$($validation.blockedLaneCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Boundary",
  "",
  $validation.boundary
)

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close real proof import bridge validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Lanes=$($validation.laneCount) Blocked=$($validation.blockedLaneCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Release close real proof import bridge validation failed with $($failedBlockers.Count) blocker(s)."
}
