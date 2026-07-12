[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-final-owner-execution-pack.json",
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
  throw "Public publish final owner execution pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$lanes = @((Get-PropertyOrDefault -Object $record -Name "executionLanes" -DefaultValue @()))
$sourceArtifacts = @((Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()))
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredOwnerInputs = @((Get-PropertyOrDefault -Object $record -Name "requiredOwnerInputs" -DefaultValue @()))

$allLanesBlocked = $lanes.Count -ge 10
foreach ($lane in $lanes) {
  $allLanesBlocked = $allLanesBlocked -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "ready" -DefaultValue $true) -and
    [bool](Get-PropertyOrDefault -Object $lane -Name "notExecutedByAutomation" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $lane -Name "ownerExecutionOnly" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canCloseReleaseIssue" -DefaultValue $true)
}

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-publish-final-owner-execution-pack") -Severity "blocker" -Detail "recordKind must be public-publish-final-owner-execution-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "executionPackState" -DefaultValue "") -eq "blocked-public-publish-final-owner-execution-required") -Severity "blocker" -Detail "Execution pack must remain blocked until real owner execution proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "execution-lanes-blocked" -Passed $allLanesBlocked -Severity "blocker" -Detail "All execution lanes must be owner-only, not executed by automation, and blocked by default.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-artifacts" -Passed ($sourceArtifacts.Count -ge 10 -and (($sourceArtifacts -join "`n").Contains("clean-external-package-consumer-owner-runbook-validation.json", [StringComparison]::OrdinalIgnoreCase)) -and (($sourceArtifacts -join "`n").Contains("post-publish-owner-verification-runbook-validation.json", [StringComparison]::OrdinalIgnoreCase))) -Severity "blocker" -Detail "Execution pack must preserve upstream source artifacts, including both owner runbook validations, for auditability.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runbook-lanes-present" -Passed (($laneIds -contains "clean-external-package-consumer-owner-runbook") -and ($laneIds -contains "post-publish-owner-verification-runbook") -and (($requiredOwnerInputs -join "`n").Contains("publicPackageSourceUrl", [StringComparison]::OrdinalIgnoreCase)) -and (($requiredOwnerInputs -join "`n").Contains("downloadedNupkgSha256", [StringComparison]::OrdinalIgnoreCase)) -and (($requiredOwnerInputs -join "`n").Contains("nonSubstituteConfirmations", [StringComparison]::OrdinalIgnoreCase))) -Severity "blocker" -Detail "Execution pack must include both owner runbook lanes and real public package/source/hash owner input fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "manual-publish-boundary-visible" -Passed ($raw.Contains("does not run dotnet nuget push", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("public package source URL", [StringComparison]::OrdinalIgnoreCase) -and $raw.Contains("downloaded nupkg SHA256", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Post-publish runbook lane must state it does not publish and requires public source plus downloaded package hash proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Execution pack must not publish, approve, promote proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-public-publish-final-owner-execution-pack" } else { "blocked-public-publish-final-owner-execution-required" }

$validation = [pscustomobject]@{
  recordKind = "public-publish-final-owner-execution-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  executionLaneCount = $lanes.Count
  blockedExecutionLaneCount = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
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
  safetyBoundary = "Validation confirms a blocked owner execution handoff only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-final-owner-execution-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-final-owner-execution-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Public Publish Final Owner Execution Pack Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| executionLaneCount | ``$($validation.executionLaneCount)`` |
| blockedExecutionLaneCount | ``$($validation.blockedExecutionLaneCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Public publish final owner execution pack validation failed with $($failedBlockers.Count) blocker(s)."
}

Write-Host "Public publish final owner execution pack validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Lanes=$($validation.executionLaneCount) Blocked=$($validation.blockedExecutionLaneCount) FailedBlockers=$($validation.failedBlockerCount)"
