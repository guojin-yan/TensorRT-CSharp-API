[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-runtime-proof-execution-runbook.json",
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
  throw "Owner runtime proof execution runbook not found: $resolvedInputPath"
}

$runbook = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$runbookItems = @(Get-PropertyOrDefault -Object $runbook -Name "runbookItems" -DefaultValue @())
$items = New-Object System.Collections.Generic.List[object]
$blockedItems = @($runbookItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "runbookState" -DefaultValue "") -eq "blocked-owner-runtime-proof-execution-required" })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $runbook -Name "recordKind" -DefaultValue "") -eq "owner-runtime-proof-execution-runbook") -Severity "blocker" -Detail "recordKind must be owner-runtime-proof-execution-runbook.")) | Out-Null
$items.Add((New-ValidationItem -Id "runbook-state" -Passed ([string](Get-PropertyOrDefault -Object $runbook -Name "runbookState" -DefaultValue "") -eq "blocked-owner-runtime-proof-execution-required") -Severity "blocker" -Detail "Owner runtime proof runbook must remain blocked until Owner executes real proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "runbook-item-count" -Passed ([int](Get-PropertyOrDefault -Object $runbook -Name "runbookItemCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Runbook must cover 6 proof lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-runbook-item-count" -Passed ([int](Get-PropertyOrDefault -Object $runbook -Name "blockedRunbookItemCount" -DefaultValue -1) -eq $blockedItems.Count -and $blockedItems.Count -ge 6) -Severity "blocker" -Detail "Runbook must keep all runbook items blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $runbook -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $runbook -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $runbook -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $runbook -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Runbook must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $runbook -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $runbook -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Runbook must not publish or approve public publication.")) | Out-Null

foreach ($item in $runbookItems) {
  $runbookItemId = [string](Get-PropertyOrDefault -Object $item -Name "runbookItemId" -DefaultValue "unknown-runbook-item")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $item -Name "executionInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $item -Name "proofLane" -DefaultValue "")) -and
    @((Get-PropertyOrDefault -Object $item -Name "commandSequence" -DefaultValue @())).Count -ge 4 -and
    @((Get-PropertyOrDefault -Object $item -Name "requiredHashes" -DefaultValue @())).Count -ge 5 -and
    @((Get-PropertyOrDefault -Object $item -Name "validatorCommands" -DefaultValue @())).Count -gt 0 -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $item -Name "nonSubstituteBoundary" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$runbookItemId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each runbook item must include command sequence, hashes, validators, and non-substitute boundary.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$runbookItemId-owner-execution-required" -Passed (-not [bool](Get-PropertyOrDefault -Object $item -Name "ownerActionRequired" -DefaultValue $true)) -Severity "action-required" -Detail "Owner must execute this runbook lane with real logs and hashes before it can become proof.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-owner-runtime-proof-execution-runbook" } else { "blocked-owner-runtime-proof-execution-required" }

$validation = [pscustomobject]@{
  recordKind = "owner-runtime-proof-execution-runbook-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  runbookItemCount = [int](Get-PropertyOrDefault -Object $runbook -Name "runbookItemCount" -DefaultValue 0)
  blockedRunbookItemCount = [int](Get-PropertyOrDefault -Object $runbook -Name "blockedRunbookItemCount" -DefaultValue 0)
  readyRunbookItemCount = [int](Get-PropertyOrDefault -Object $runbook -Name "readyRunbookItemCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates owner runtime proof runbook shape only. It is not runtime proof, package publish, post-publish verification, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-runtime-proof-execution-runbook-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-runtime-proof-execution-runbook-validation.md"
$validation | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Runtime Proof Execution Runbook Validation")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$(ConvertTo-MarkdownCell $validation.validationState)`` |")
$lines.Add("| runbookItemCount | ``$($validation.runbookItemCount)`` |")
$lines.Add("| blockedRunbookItemCount | ``$($validation.blockedRunbookItemCount)`` |")
$lines.Add("| readyRunbookItemCount | ``$($validation.readyRunbookItemCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner runtime proof execution runbook validation written to $jsonPath"
Write-Host "Owner runtime proof execution runbook validation markdown written to $markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner runtime proof execution runbook validation has blocker failures: $($failedBlockers.Count)"
}
