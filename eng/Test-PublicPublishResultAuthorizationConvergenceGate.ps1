[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-result-authorization-convergence-gate.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-PublicPublishResultAuthorizationConvergenceGate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$gateItems = @((Get-PropertyOrDefault -Object $record -Name "gateItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-publish-result-authorization-convergence-gate") -Severity "blocker" -Detail "recordKind must match convergence gate.")) | Out-Null
$items.Add((New-ValidationItem -Id "state" -Passed (@("blocked-public-publish-result-authorization-convergence-required", "public-publish-result-authorization-convergence-ready") -contains [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "")) -Severity "blocker" -Detail "Gate state must be blocked or ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "has-required-gates" -Passed (($gateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "owner-authorization-input" }).Count -eq 1 -and ($gateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "public-publish-result-import" }).Count -eq 1 -and ($gateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "post-publish-clean-consumer-proof" }).Count -eq 1 -and ($gateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "release-close-strict-bridge" }).Count -eq 1) -Severity "blocker" -Detail "Gate must aggregate authorization, publish result, post-publish proof, and close bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed (
      [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "eligibleForOwnerCloseInstruction" -DefaultValue $true)
    ) -Severity "blocker" -Detail "Default convergence gate must stay non-proof, non-publish, non-close, and not close-eligible.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-by-default" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedGateItemCount" -DefaultValue 0) -gt 0) -Severity "action-required" -Detail "Default gate must remain blocked until real Owner inputs and proofs are present.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-public-publish-result-authorization-convergence-gate" } else { [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "blocked-public-publish-result-authorization-convergence-required") }

$validation = [pscustomobject]@{
  recordKind = "public-publish-result-authorization-convergence-gate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  gateItemCount = $gateItems.Count
  blockedGateItemCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedGateItemCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  eligibleForOwnerCloseInstruction = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Convergence gate validation is side-effect free. It does not publish, dispatch workflows, close release issues, or promote proof."
}

$jsonPath = Join-Path $OutputRoot "public-publish-result-authorization-convergence-gate-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-result-authorization-convergence-gate-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |"
}
$markdown = @"
# Public Publish Result Authorization Convergence Gate Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| gateItemCount | ``$($validation.gateItemCount)`` |
| blockedGateItemCount | ``$($validation.blockedGateItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| eligibleForOwnerCloseInstruction | ``$($validation.eligibleForOwnerCloseInstruction)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Public publish result authorization convergence gate validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Public publish result authorization convergence gate validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

