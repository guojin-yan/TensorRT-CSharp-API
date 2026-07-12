[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-command-cross-check.json",
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

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) { throw "Public publish command cross-check not found: $resolvedInputPath" }

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$checks = @((Get-PropertyOrDefault -Object $record -Name "crossChecks" -DefaultValue @()))
$blockedChecks = @($checks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) })
$checkIds = @($checks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-publish-command-cross-check") -Severity "blocker" -Detail "recordKind must be public-publish-command-cross-check.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "crossCheckState" -DefaultValue "") -eq "blocked-public-publish-command-cross-check-owner-action-required") -Severity "blocker" -Detail "Cross-check must remain blocked until owner fills real channel/package/hash values.")) | Out-Null
$items.Add((New-ValidationItem -Id "check-count" -Passed ($checks.Count -ge 11 -and $blockedChecks.Count -gt 0) -Severity "action-required" -Detail "At least eleven package/channel/hash/runbook/rollback/credential checks must be exposed and blocked by default.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runbook-checks-present" -Passed (($checkIds -contains "clean-external-package-consumer-runbook") -and ($checkIds -contains "post-publish-owner-verification-runbook") -and $raw.Contains("public package source URL", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("downloaded nupkg SHA256", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("nonSubstituteConfirmations", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("does not run dotnet nuget push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Command cross-check must include both owner runbooks and preserve public source/hash/non-substitute/publish-boundary wording.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Cross-check must not publish, approve, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-public-publish-command-cross-check" } else { "blocked-public-publish-command-cross-check-owner-action-required" }

$validation = [pscustomobject]@{
  recordKind = "public-publish-command-cross-check-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  crossCheckCount = $checks.Count
  blockedCrossCheckCount = $blockedChecks.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Validation confirms blocked owner command cross-check only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-command-cross-check-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-command-cross-check-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8
$validation.validationItems | ForEach-Object { } | Out-Null
$markdown = @"
# Public Publish Command Cross Check Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| crossCheckCount | ``$($validation.crossCheckCount)`` |
| blockedCrossCheckCount | ``$($validation.blockedCrossCheckCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) { throw "Public publish command cross-check validation failed with $($failedBlockers.Count) blocker(s)." }

Write-Host "Public publish command cross-check validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) CrossChecks=$($validation.crossCheckCount) Blocked=$($validation.blockedCrossCheckCount) FailedBlockers=$($validation.failedBlockerCount)"
