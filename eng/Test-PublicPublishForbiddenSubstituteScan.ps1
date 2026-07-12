[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-forbidden-substitute-scan.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Public publish forbidden substitute scan not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$checks = @((Get-PropertyOrDefault -Object $record -Name "substituteChecks" -DefaultValue @()))
$blockedChecks = @($checks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$requiredKinds = @(
  "local .nupkg",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "template",
  "dry-run",
  "dashboard",
  "audit pack",
  "local-only artifact scan",
  "manual approval",
  "queued GitHub Actions run",
  "missing self-hosted runner",
  "sidecar-only",
  "TensorRtExec report"
)
$kinds = @($checks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "forbiddenKind" -DefaultValue "") })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-publish-forbidden-substitute-scan") -Severity "blocker" -Detail "recordKind must be public-publish-forbidden-substitute-scan.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "scanState" -DefaultValue "") -eq "blocked-public-publish-forbidden-substitute-scan-owner-proof-required") -Severity "blocker" -Detail "Scan must stay blocked until owner supplies real proof inputs.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-forbidden-kinds" -Passed (@($requiredKinds | Where-Object { $kinds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Scan must cover all forbidden substitute kinds.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-required" -Passed ($blockedChecks.Count -eq 0) -Severity "action-required" -Detail "Owner must provide real records proving forbidden substitutes were not used.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Scan must not publish, approve, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-public-publish-forbidden-substitute-scan" } else { "blocked-public-publish-forbidden-substitute-scan-owner-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "public-publish-forbidden-substitute-scan-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  substituteCheckCount = $checks.Count
  blockedSubstituteCheckCount = $blockedChecks.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  validationItems = @($items.ToArray())
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "Validation checks forbidden substitute scan shape only and confirms manual approval, queued workflow, missing self-hosted runner, sidecar-only, TensorRtExec report, local feed, ProjectReference, and direct nupkg remain non-proof substitutes. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-forbidden-substitute-scan-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-forbidden-substitute-scan-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Public Publish Forbidden Substitute Scan Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| substituteCheckCount | ``$($validation.substituteCheckCount)`` |",
  "| blockedSubstituteCheckCount | ``$($validation.blockedSubstituteCheckCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Boundary",
  "",
  $validation.boundary
)

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish forbidden substitute scan validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Checks=$($validation.substituteCheckCount) Blocked=$($validation.blockedSubstituteCheckCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Public publish forbidden substitute scan validation failed with $($failedBlockers.Count) blocker(s)."
}
