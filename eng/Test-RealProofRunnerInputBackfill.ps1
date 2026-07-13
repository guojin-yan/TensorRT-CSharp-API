[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-runner-input-backfill.template.json",
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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof runner input backfill template not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$trackInputs = @(Get-PropertyOrDefault -Object $record -Name "trackInputs" -DefaultValue @())
$forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "real-proof-runner-input-backfill") -Severity "blocker" -Detail "recordKind must be real-proof-runner-input-backfill.")) | Out-Null
$items.Add((New-ValidationItem -Id "track-count" -Passed ($trackInputs.Count -eq 6) -Severity "blocker" -Detail "Template must include exactly 6 proof runner tracks.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Input backfill must not promote runtime proof, release close proof, or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Input backfill must not publish or approve public publication.")) | Out-Null

foreach ($required in @("local feed", "ProjectReference", "direct .nupkg", "DependencyProbe", "build-only", "template", "Windows handoff for Linux proof", "hash-only audit")) {
  $items.Add((New-ValidationItem -Id "forbidden-$($required.Replace(' ', '-').Replace('.', '').ToLowerInvariant())" -Passed ($forbiddenSubstitutes -contains $required) -Severity "blocker" -Detail "Forbidden substitute must be listed: $required.")) | Out-Null
}

foreach ($track in $trackInputs) {
  $trackId = [string](Get-PropertyOrDefault -Object $track -Name "trackId" -DefaultValue "unknown-track")
  $hostOs = Get-PropertyOrDefault -Object $track -Name "hostOs" -DefaultValue ""
  $runtimePackageKey = Get-PropertyOrDefault -Object $track -Name "runtimePackageKey" -DefaultValue ""
  $commands = @(Get-PropertyOrDefault -Object $track -Name "commands" -DefaultValue @())
  $logs = @(Get-PropertyOrDefault -Object $track -Name "logs" -DefaultValue @())
  $hashes = @(Get-PropertyOrDefault -Object $track -Name "hashes" -DefaultValue @())
  $validatorOutputs = @(Get-PropertyOrDefault -Object $track -Name "validatorOutputs" -DefaultValue @())
  $forbiddenChecks = @(Get-PropertyOrDefault -Object $track -Name "forbiddenSubstituteChecks" -DefaultValue @())

  $items.Add((New-ValidationItem -Id "$trackId-host-input-required" -Passed (-not (Test-IsPlaceholder -Value $hostOs)) -Severity "action-required" -Detail "Owner must replace host OS placeholder with the real proof runner host before this can be promoted.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$trackId-runtime-key-input-required" -Passed (-not (Test-IsPlaceholder -Value $runtimePackageKey)) -Severity "action-required" -Detail "Owner must replace runtimePackageKey placeholder with the real runtime package key before this can be promoted.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$trackId-command-shape" -Passed ($commands.Count -ge 1) -Severity "blocker" -Detail "Each track must include at least one command placeholder.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$trackId-log-shape" -Passed ($logs.Count -ge 1) -Severity "blocker" -Detail "Each track must include at least one log placeholder.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$trackId-hash-shape" -Passed ($hashes.Count -ge 1) -Severity "blocker" -Detail "Each track must include at least one hash placeholder.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$trackId-validator-shape" -Passed ($validatorOutputs.Count -ge 1) -Severity "blocker" -Detail "Each track must include validator output placeholder.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$trackId-forbidden-checks" -Passed ($forbiddenChecks.Count -ge 1) -Severity "blocker" -Detail "Each track must include forbidden substitute checks.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$trackId-no-proof-flags" -Passed (-not [bool](Get-PropertyOrDefault -Object $track -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $track -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $track -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $track -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Track-level proof flags must remain false in template.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-real-proof-runner-input-backfill"
}
else {
  "blocked-owner-runner-input-required"
}

$validation = [pscustomobject]@{
  recordKind = "real-proof-runner-input-backfill-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  trackCount = $trackInputs.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates owner input template shape only. It is not proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-runner-input-backfill-validation.json"
$markdownPath = Join-Path $OutputRoot "real-proof-runner-input-backfill-validation.md"
$validation | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Real Proof Runner Input Backfill Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| trackCount | ``$($validation.trackCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isReleaseCloseProof | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join [Environment]::NewLine)

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof runner input backfill validation written to $jsonPath"
Write-Host "Real proof runner input backfill validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState Tracks=$($validation.trackCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Real proof runner input backfill validation failed with $($failedBlockers.Count) blocker(s)."
}
