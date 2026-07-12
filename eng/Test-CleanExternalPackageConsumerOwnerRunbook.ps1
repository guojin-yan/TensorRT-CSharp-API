[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\clean-external-package-consumer-owner-runbook.json",
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
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Clean external package consumer owner runbook not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$steps = @(Get-PropertyOrDefault -Object $record -Name "steps" -DefaultValue @())
$forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())
$requiredFields = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredResultInputFields" -DefaultValue @())
$allText = $record | ConvertTo-Json -Depth 16
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "clean-external-package-consumer-owner-runbook") -Severity "blocker" -Detail "recordKind must be clean-external-package-consumer-owner-runbook.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "runbookState" -DefaultValue "") -eq "blocked-owner-clean-external-package-consumer-execution-required") -Severity "blocker" -Detail "Runbook must remain blocked until owner executes the clean external consumer.")) | Out-Null
$items.Add((New-ValidationItem -Id "step-coverage" -Passed ($steps.Count -ge 9 -and $allText.Contains("dotnet restore", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("dotnet build", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("dotnet run", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Get-FileHash", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Runbook must cover restore/build/runtime smoke/hash execution steps.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-targets" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "requiredInputTarget" -DefaultValue "") -eq "artifacts/final-release/owner-external-proof-execution-result.input.json" -and [string](Get-PropertyOrDefault -Object $record -Name "fillableTemplate" -DefaultValue "") -eq "artifacts/final-release/owner-external-proof-execution-result.input.template.json") -Severity "blocker" -Detail "Runbook must point to the owner external result input target and template.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (($forbiddenSubstitutes -contains "local feed") -and ($forbiddenSubstitutes -contains "ProjectReference") -and ($forbiddenSubstitutes -contains "direct .nupkg") -and ($forbiddenSubstitutes -contains "dry-run") -and ($forbiddenSubstitutes -contains "blocked-by-cuda-driver") -and ($forbiddenSubstitutes -contains "build-only")) -Severity "blocker" -Detail "Runbook must block local feed, ProjectReference, direct .nupkg, dry-run, blocked-by-driver, and build-only substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-result-fields" -Passed (($requiredFields -contains "stdoutPath") -and ($requiredFields -contains "stderrPath") -and ($requiredFields -contains "mergedTranscriptPath") -and ($requiredFields -contains "validatorOutputSha256") -and ($requiredFields -contains "passed") -and ($requiredFields -contains "nonSubstituteConfirmations")) -Severity "blocker" -Detail "Runbook must require logs, hashes, passed, owner review, and non-substitute confirmations.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-or-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Runbook must not publish, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-owner-clean-external-package-consumer-execution-required" } else { "invalid-clean-external-package-consumer-owner-runbook" }

$validation = [pscustomobject]@{
  recordKind = "clean-external-package-consumer-owner-runbook-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  stepCount = $steps.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  boundary = "This validation checks owner runbook shape only. It is not package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "clean-external-package-consumer-owner-runbook-validation.json"
$markdownPath = Join-Path $OutputRoot "clean-external-package-consumer-owner-runbook-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Clean External Package Consumer Owner Runbook Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| stepCount | ``$($validation.stepCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean external package consumer owner runbook validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) PerformsPublish=False CanPromoteRuntimeProof=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Clean external package consumer owner runbook validation failed with $($failedBlockers.Count) blocker(s)."
}
