[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-public-publish-execution-consistency-gate.json",
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
function Get-PropertyOrDefault { param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue) if ($null -eq $Object) { return $DefaultValue }; if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }; return $DefaultValue }
function New-ValidationItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }
function ConvertTo-MarkdownCell { param([AllowNull()][object]$Value) if ($null -eq $Value) { return "" } return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ") }

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPublicPublishExecutionConsistencyGate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$consistencyItems = @((Get-PropertyOrDefault -Object $record -Name "consistencyItems" -DefaultValue @()))
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-execution-consistency-gate") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-items" -Passed ($consistencyItems.Count -ge 20) -Severity "blocker" -Detail "Consistency gate must include package fields, forbidden substitutes, and non-proof checks.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-blocker-failures" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "failedBlockerCount" -DefaultValue 999) -eq 0) -Severity "blocker" -Detail "Forbidden substitute and non-proof boundary checks must pass structurally.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-required" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "failedActionRequiredCount" -DefaultValue 0) -gt 0) -Severity "action-required" -Detail "Default consistency gate must remain blocked until real Owner package identity/hash inputs are imported.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Gate must not publish, close, or promote proof.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-public-publish-execution-consistency-gate" } elseif ($failedActionRequired.Count -gt 0) { "blocked-owner-public-publish-execution-consistency-owner-input-required" } else { [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "blocked-owner-public-publish-execution-consistency-owner-input-required") }
$validation = [pscustomobject]@{
  recordKind = "owner-public-publish-execution-consistency-gate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  consistencyItemCount = $consistencyItems.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Consistency validation is side-effect free; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-execution-consistency-gate-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-public-publish-execution-consistency-gate-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$rows = $validation.validationItems | ForEach-Object { "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |" }
$markdown = @"
# Owner Public Publish Execution Consistency Gate Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| consistencyItemCount | ``$($validation.consistencyItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8
if ($Strict -and $failedBlockers.Count -gt 0) { throw "Owner public publish execution consistency gate validation failed with $($failedBlockers.Count) blocker(s)." }
Write-Host "Owner public publish execution consistency gate validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"
