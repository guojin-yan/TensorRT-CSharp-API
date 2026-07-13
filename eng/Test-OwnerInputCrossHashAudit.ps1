[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-input-cross-hash-audit.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner input cross-hash audit not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$auditState = [string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$auditLines = @((Get-PropertyOrDefault -Object $record -Name "auditLines" -DefaultValue @()))
$mismatchedHashCount = [int](Get-PropertyOrDefault -Object $record -Name "mismatchedHashCount" -DefaultValue -1)
$blockedStateLineCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedStateLineCount" -DefaultValue -1)
$sourceArtifacts = @((Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()) | ForEach-Object { [string]$_ })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-input-cross-hash-audit") -Severity "blocker" -Detail "recordKind must be owner-input-cross-hash-audit.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-audit-state" -Passed ($auditState -eq "blocked-owner-input-cross-hash-audit-owner-proof-required") -Severity "blocker" -Detail "Audit must remain blocked because hash consistency is not proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Audit must not publish, approve publication, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "audit-lines-present" -Passed ($auditLines.Count -ge 8) -Severity "blocker" -Detail "Audit must include all final release proof/overlay artifacts.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts-present" -Passed ($sourceArtifacts.Count -ge 8) -Severity "blocker" -Detail "Audit must preserve source artifact references.")) | Out-Null
$items.Add((New-ValidationItem -Id "hashes-match-current-local-files" -Passed ($mismatchedHashCount -eq 0) -Severity "blocker" -Detail "All audited local hashes must match current files.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-proof-still-required" -Passed $false -Severity "action-required" -Detail "$blockedStateLineCount audited line(s) remain blocked/missing/incomplete; hash consistency cannot promote proof.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-input-cross-hash-audit-ready"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-owner-input-cross-hash-audit-owner-proof-required"
}
else {
  "invalid-owner-input-cross-hash-audit"
}

$validation = [pscustomobject]@{
  recordKind = "owner-input-cross-hash-audit-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  auditState = $auditState
  isValidCrossHashAuditShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  auditLineCount = $auditLines.Count
  mismatchedHashCount = $mismatchedHashCount
  blockedStateLineCount = $blockedStateLineCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates local hash consistency only. It cannot publish packages, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-input-cross-hash-audit-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-input-cross-hash-audit-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Owner Input Cross-Hash Audit Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| auditState | ``$($validation.auditState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| auditLineCount | ``$($validation.auditLineCount)`` |
| mismatchedHashCount | ``$($validation.mismatchedHashCount)`` |
| blockedStateLineCount | ``$($validation.blockedStateLineCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner input cross-hash audit validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner input cross-hash audit validation failed with $($failedBlockers.Count) blocker(s)."
}
