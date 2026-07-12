[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-strict-validation-bridge.json",
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Release close strict validation bridge not found: $resolvedInputPath"
}

$bridge = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$bridgeItems = @(Get-PropertyOrDefault -Object $bridge -Name "bridgeItems" -DefaultValue @())
$items = New-Object System.Collections.Generic.List[object]
$blockedBridgeItems = @($bridgeItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "bridgeState" -DefaultValue "") -eq "blocked-release-close-strict-validation-required" })
$readyBridgeItems = @($bridgeItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForStrictClose" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $bridge -Name "recordKind" -DefaultValue "") -eq "release-close-strict-validation-bridge") -Severity "blocker" -Detail "recordKind must be release-close-strict-validation-bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "bridge-state" -Passed ([string](Get-PropertyOrDefault -Object $bridge -Name "bridgeState" -DefaultValue "") -eq "blocked-release-close-strict-validation-required") -Severity "blocker" -Detail "Bridge must remain blocked until all strict-close prerequisites pass.")) | Out-Null
$items.Add((New-ValidationItem -Id "bridge-item-count" -Passed ([int](Get-PropertyOrDefault -Object $bridge -Name "bridgeItemCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Bridge must cover 6 proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-bridge-item-count" -Passed ([int](Get-PropertyOrDefault -Object $bridge -Name "blockedBridgeItemCount" -DefaultValue -1) -eq $blockedBridgeItems.Count -and $blockedBridgeItems.Count -ge 6) -Severity "blocker" -Detail "Default bridge must keep all items blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-bridge-item-count" -Passed ([int](Get-PropertyOrDefault -Object $bridge -Name "readyBridgeItemCount" -DefaultValue -1) -eq $readyBridgeItems.Count -and $readyBridgeItems.Count -eq 0) -Severity "blocker" -Detail "Default bridge must not claim ready strict-close items.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $bridge -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $bridge -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $bridge -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $bridge -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $bridge -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $bridge -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not publish or approve public publication.")) | Out-Null

foreach ($bridgeItem in $bridgeItems) {
  $bridgeItemId = [string](Get-PropertyOrDefault -Object $bridgeItem -Name "bridgeItemId" -DefaultValue "unknown-bridge-item")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $bridgeItem -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $bridgeItem -Name "strictCloseCommand" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $bridgeItem -Name "notProofBoundary" -DefaultValue "")) -and
    @((Get-PropertyOrDefault -Object $bridgeItem -Name "blockedReasons" -DefaultValue @())).Count -ge 4
  $items.Add((New-ValidationItem -Id "$bridgeItemId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each bridge item must include lane, blocked reasons, strict close command, and non-proof boundary.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$bridgeItemId-strict-close-required" -Passed ([bool](Get-PropertyOrDefault -Object $bridgeItem -Name "readyForStrictClose" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must complete runtime proof input validation, runbook execution, post-publish verification, close record validation, final runbook, and evidence bundle before strict close.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-release-close-strict-validation-bridge" } else { "blocked-release-close-strict-validation-required" }

$validation = [pscustomobject]@{
  recordKind = "release-close-strict-validation-bridge-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  bridgeItemCount = [int](Get-PropertyOrDefault -Object $bridge -Name "bridgeItemCount" -DefaultValue 0)
  blockedBridgeItemCount = [int](Get-PropertyOrDefault -Object $bridge -Name "blockedBridgeItemCount" -DefaultValue 0)
  readyBridgeItemCount = [int](Get-PropertyOrDefault -Object $bridge -Name "readyBridgeItemCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates strict-close bridge shape only. It is not runtime proof, package publish, post-publish verification, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-validation-bridge-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-strict-validation-bridge-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Close Strict Validation Bridge Validation")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$(ConvertTo-MarkdownCell $validation.validationState)`` |")
$lines.Add("| bridgeItemCount | ``$($validation.bridgeItemCount)`` |")
$lines.Add("| blockedBridgeItemCount | ``$($validation.blockedBridgeItemCount)`` |")
$lines.Add("| readyBridgeItemCount | ``$($validation.readyBridgeItemCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close strict validation bridge validation written to $jsonPath"
Write-Host "Release close strict validation bridge validation markdown written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Release close strict validation bridge validation has blocker failures: $($failedBlockers.Count)"
}
