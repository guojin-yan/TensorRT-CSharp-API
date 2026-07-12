[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-prepublish-quality-freeze-dashboard.json",
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
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath { param([string]$Path) if ([System.IO.Path]::IsPathRooted($Path)) { return $Path } return Join-Path $RepositoryRoot $Path }
function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}
function New-ValidationItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }
function ConvertTo-MarkdownCell { param([AllowNull()][object]$Value) if ($null -eq $Value) { return "" } return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ") }

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalPrepublishQualityFreezeDashboard.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$lanes = @((Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$requiredLaneIds = @(
  "release-script-reference-index",
  "release-script-reference-index-validation",
  "release-evidence-bundle",
  "release-evidence-classification-audit",
  "owner-public-publish-authorization-input",
  "owner-public-publish-authorization-gate",
  "public-publish-result-authorization-convergence-gate",
  "public-publish-final-owner-execution-pack",
  "public-publish-command-cross-check",
  "final-evidence-freeze-non-proof-audit",
  "release-candidate-package-inventory",
  "final-quality-freeze-dashboard"
)
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$missingLaneIds = @($requiredLaneIds | Where-Object { $laneIds -notcontains $_ })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-prepublish-quality-freeze-dashboard") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "state" -Passed (@("blocked-final-prepublish-quality-freeze-owner-action-required", "final-prepublish-quality-freeze-ready-for-owner-manual-public-publish-review") -contains [string](Get-PropertyOrDefault -Object $record -Name "freezeState" -DefaultValue "")) -Severity "blocker" -Detail "freezeState must be blocked or owner-review ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes" -Passed ($missingLaneIds.Count -eq 0) -Severity "blocker" -Detail ("Missing lanes: " + ($missingLaneIds -join ", ")))) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-without-owner-input" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "readyForOwnerPublicPublishExecution" -DefaultValue $true)) -Severity "action-required" -Detail "Default dashboard must stay blocked until real Owner authorization and publish proof exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isGitHubActionsProof" -DefaultValue $true)) -Severity "blocker" -Detail "Dashboard must not publish, close, promote proof, or claim GitHub Actions proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-failures" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "boundaryFailureCount" -DefaultValue 999) -eq 0) -Severity "blocker" -Detail "All lanes must keep non-proof/no-side-effect flags false.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-prepublish-quality-freeze-dashboard" } elseif ($failedActionRequired.Count -gt 0) { "blocked-final-prepublish-quality-freeze-owner-action-required" } else { [string](Get-PropertyOrDefault -Object $record -Name "freezeState" -DefaultValue "blocked-final-prepublish-quality-freeze-owner-action-required") }

$validation = [pscustomobject]@{
  recordKind = "final-prepublish-quality-freeze-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  laneCount = $lanes.Count
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  boundaryFailureCount = [int](Get-PropertyOrDefault -Object $record -Name "boundaryFailureCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  readyForOwnerPublicPublishExecution = $false
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isGitHubActionsProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Final prepublish quality freeze validation is side-effect free; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and not GitHub Actions proof."
}

$jsonPath = Join-Path $OutputRoot "final-prepublish-quality-freeze-dashboard-validation.json"
$markdownPath = Join-Path $OutputRoot "final-prepublish-quality-freeze-dashboard-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $validation.validationItems | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |" }
$markdown = @"
# Final Prepublish Quality Freeze Dashboard Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| laneCount | ``$($validation.laneCount)`` |
| blockedLaneCount | ``$($validation.blockedLaneCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| readyForOwnerPublicPublishExecution | ``$($validation.readyForOwnerPublicPublishExecution)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) { throw "Final prepublish quality freeze dashboard validation failed with $($failedBlockers.Count) blocker(s)." }

Write-Host "Final prepublish quality freeze dashboard validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
