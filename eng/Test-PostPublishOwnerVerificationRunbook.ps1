[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-owner-verification-runbook.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release" }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath { param([string]$Path) if ([System.IO.Path]::IsPathRooted($Path)) { return $Path } return Join-Path $RepositoryRoot $Path }
function Get-PropertyOrDefault { param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue) if ($null -eq $Object) { return $DefaultValue } if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value } return $DefaultValue }
function Convert-ToStringArray { param([AllowNull()][object]$Values) return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ }) }
function New-ValidationItem { param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail) [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail } }

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { throw "Post publish owner verification runbook not found: $resolvedInputPath" }

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$steps = @(Get-PropertyOrDefault -Object $record -Name "steps" -DefaultValue @())
$forbiddenSubstitutes = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())
$publicEvidence = Convert-ToStringArray (Get-PropertyOrDefault -Object $record -Name "requiredPublicChannelEvidence" -DefaultValue @())
$allText = $record | ConvertTo-Json -Depth 16
$items = New-Object System.Collections.Generic.List[object]

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-owner-verification-runbook") -Severity "blocker" -Detail "recordKind must be post-publish-owner-verification-runbook.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "runbookState" -DefaultValue "") -eq "blocked-owner-post-publish-verification-required") -Severity "blocker" -Detail "Runbook must remain blocked until real post-publish owner verification is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-channel-coverage" -Passed ($steps.Count -ge 6 -and $allText.Contains("public package source", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("published package", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("post-publish", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("owner-external-proof-execution-result.input.json", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Runbook must cover public channel package identity and owner external result input.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-public-evidence" -Passed (($publicEvidence -join "`n").Contains("public package source URL", [StringComparison]::OrdinalIgnoreCase) -and ($publicEvidence -join "`n").Contains("published package URL", [StringComparison]::OrdinalIgnoreCase) -and ($publicEvidence -join "`n").Contains("validator output", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Runbook must require public source, package URL, logs, hashes, validator output, and owner review.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed (($forbiddenSubstitutes -contains "local feed") -and ($forbiddenSubstitutes -contains "ProjectReference") -and ($forbiddenSubstitutes -contains "direct .nupkg") -and ($forbiddenSubstitutes -contains "pre-publish package-consumer proof") -and ($forbiddenSubstitutes -contains "dry-run") -and ($forbiddenSubstitutes -contains "template-only")) -Severity "blocker" -Detail "Runbook must forbid local feed, ProjectReference, direct .nupkg, pre-publish proof, dry-run, and template-only substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-proof-close-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Runbook must not publish, prove post-publish verification, promote proof, or close release.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-owner-post-publish-verification-required" } else { "invalid-post-publish-owner-verification-runbook" }

$validation = [pscustomobject]@{
  recordKind = "post-publish-owner-verification-runbook-validation"
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
  boundary = "This validation checks post-publish owner runbook shape only. It is not post-publish proof."
}

$jsonPath = Join-Path $OutputRoot "post-publish-owner-verification-runbook-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-owner-verification-runbook-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Post Publish Owner Verification Runbook Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| stepCount | ``$($validation.stepCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post publish owner verification runbook validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Post publish owner verification runbook validation failed with $($failedBlockers.Count) blocker(s)."
}
