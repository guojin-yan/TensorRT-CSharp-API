[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-public-publish-authorization-gate.json",
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
  & (Join-Path $RepositoryRoot "eng\Export-OwnerPublicPublishAuthorizationGate.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$gateItems = @((Get-PropertyOrDefault -Object $record -Name "gateItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "owner-public-publish-authorization-gate") -Severity "blocker" -Detail "recordKind must match authorization gate.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-or-authorized-state" -Passed (@("blocked-owner-public-publish-authorization-required", "owner-public-publish-authorized-for-manual-execution-review") -contains [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "")) -Severity "blocker" -Detail "Gate must be blocked by default or explicitly authorized for manual review only.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed (
      [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and
      [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and
      -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)
    ) -Severity "blocker" -Detail "Gate must stay non-proof, non-publish, and non-close.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-materialized-command" -Passed ([string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $record -Name "materializedExecutableCommand" -DefaultValue ""))) -Severity "blocker" -Detail "Gate must not materialize executable publish or close commands.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-gate-items" -Passed (($gateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "owner-authorization-present" }).Count -eq 1 -and ($gateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "no-materialized-executable-command" }).Count -eq 1) -Severity "blocker" -Detail "Gate must check explicit owner authorization and missing materialized command.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary-forbids-side-effects" -Passed ($raw.Contains("never executes dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("never publishes GitHub Packages", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("never dispatches workflows", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("never closes a release", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must explicitly forbid publish, workflow dispatch, and release close side effects.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })

$validation = [pscustomobject]@{
  recordKind = "owner-public-publish-authorization-gate-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = if ($failedBlockers.Count -eq 0) { [string](Get-PropertyOrDefault -Object $record -Name "gateState" -DefaultValue "blocked-owner-public-publish-authorization-required") } else { "invalid-owner-public-publish-authorization-gate" }
  gateItemCount = $gateItems.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  blockedActionRequiredCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedActionRequiredCount" -DefaultValue 0)
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
  safetyBoundary = "Authorization gate validation is side-effect free and does not publish, dispatch workflows, close release issues, or promote proof."
}

$jsonPath = Join-Path $OutputRoot "owner-public-publish-authorization-gate-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-public-publish-authorization-gate-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | ``$($_.passed)`` | $(ConvertTo-MarkdownCell $_.severity) | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Owner Public Publish Authorization Gate Validation

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| gateItemCount | ``$($validation.gateItemCount)`` |
| blockedActionRequiredCount | ``$($validation.blockedActionRequiredCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
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
  throw "Owner public publish authorization gate validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Owner public publish authorization gate validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$($validation.failedBlockerCount) Blocked=$($validation.blockedActionRequiredCount)"
