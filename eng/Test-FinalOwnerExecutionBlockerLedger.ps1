[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-blocker-ledger.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
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
  throw "Final owner execution blocker ledger not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$categories = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "categories" -DefaultValue @()))
$blockers = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "blockers" -DefaultValue @()))
$categoryNames = @($categories | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "category" -DefaultValue "") })
$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$requiredCategories = @(
  "missing owner input",
  "placeholder",
  "path missing",
  "SHA256 invalid",
  "strict validator not run",
  "strict validator failed",
  "ready for import candidate"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-blocker-ledger") -Severity "blocker" -Detail "recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "ledgerState" -DefaultValue "") -eq "blocked-final-owner-real-input-required") -Severity "blocker" -Detail "Ledger must remain blocked until real Owner input exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-categories" -Passed (@($requiredCategories | Where-Object { $categoryNames -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Ledger must include all required blocker categories.")) | Out-Null
$items.Add((New-ValidationItem -Id "remaining-blockers" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "remainingBlockerCount" -DefaultValue 0) -gt 0 -and $blockers.Count -gt 0) -Severity "blocker" -Detail "Ledger must report remaining blockers while Owner input is absent.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-ready-candidates" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyForImportCandidateCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default ledger cannot contain ready import candidates.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsRuntimeExecution" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true))) -Severity "blocker" -Detail "Ledger must remain non-proof and non-publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Boundary must exclude proof, publish, close, and package push.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-final-owner-real-input-required" } else { "invalid-final-owner-execution-blocker-ledger" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-blocker-ledger-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  categoryCount = $categories.Count
  blockerCount = $blockers.Count
  remainingBlockerCount = [int](Get-PropertyOrDefault -Object $record -Name "remainingBlockerCount" -DefaultValue 0)
  readyForImportCandidateCount = [int](Get-PropertyOrDefault -Object $record -Name "readyForImportCandidateCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms the final Owner blocker ledger remains blocked owner-action guidance only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-blocker-ledger-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-blocker-ledger-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Final Owner Execution Blocker Ledger Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| categoryCount | ``$($validation.categoryCount)`` |
| blockerCount | ``$($validation.blockerCount)`` |
| remainingBlockerCount | ``$($validation.remainingBlockerCount)`` |
| readyForImportCandidateCount | ``$($validation.readyForImportCandidateCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution blocker ledger validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution blocker ledger validation failed with $($failedBlockers.Count) blocker(s)."
}
