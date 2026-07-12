[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-publish-owner-manual-command-handoff.json",
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
  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Public publish owner manual command handoff not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$commands = @((Get-PropertyOrDefault -Object $record -Name "manualCommands" -DefaultValue @()))
$prerequisites = @((Get-PropertyOrDefault -Object $record -Name "manualPrerequisites" -DefaultValue @()))

$hasNuGetPush = @($commands | Where-Object { ([string](Get-PropertyOrDefault -Object $_ -Name "command" -DefaultValue "")).Contains("dotnet nuget push", [StringComparison]::Ordinal) }).Count -gt 0
$allCommandsManual = $commands.Count -ge 4
foreach ($command in $commands) {
  $allCommandsManual = $allCommandsManual -and
    [bool](Get-PropertyOrDefault -Object $command -Name "notExecutedByAutomation" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $command -Name "placeholderOnly" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $command -Name "ownerExecutionOnly" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $command -Name "modelExecutionForbidden" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $command -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $command -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $command -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $command -Name "materializedExecutableCommand" -DefaultValue "unexpected"))
}

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-publish-owner-manual-command-handoff") -Severity "blocker" -Detail "recordKind must be public-publish-owner-manual-command-handoff.")) | Out-Null
$items.Add((New-ValidationItem -Id "handoff-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "handoffState" -DefaultValue "") -eq "blocked-owner-public-publish-required") -Severity "blocker" -Detail "Handoff must remain blocked until owner public publish proof exists.")) | Out-Null
$items.Add((New-ValidationItem -Id "manual-prerequisites" -Passed ($prerequisites.Count -ge 5 -and [int](Get-PropertyOrDefault -Object $record -Name "blockedManualPrerequisiteCount" -DefaultValue 0) -gt 0) -Severity "action-required" -Detail "Manual handoff must expose blocked prerequisites before publish/close.")) | Out-Null
$items.Add((New-ValidationItem -Id "nuget-push-placeholder" -Passed ($hasNuGetPush -and [bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false)) -Severity "blocker" -Detail "dotnet nuget push may appear only as notExecutedByAutomation=true placeholder.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-commands-manual" -Passed $allCommandsManual -Severity "blocker" -Detail "All command entries must be placeholder-only, owner-execution-only, and non-publishing.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Manual command handoff must not publish, prove runtime, prove post-publish, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-public-publish-owner-manual-command-handoff"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-owner-public-publish-required"
}
else {
  "public-publish-owner-manual-command-handoff-ready"
}

$validation = [pscustomobject]@{
  recordKind = "public-publish-owner-manual-command-handoff-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  commandCount = $commands.Count
  manualPrerequisiteCount = $prerequisites.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Public publish owner manual command handoff validation only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-owner-manual-command-handoff-validation.json"
$markdownPath = Join-Path $OutputRoot "public-publish-owner-manual-command-handoff-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Public Publish Owner Manual Command Handoff Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| commandCount | ``$($validation.commandCount)`` |
| manualPrerequisiteCount | ``$($validation.manualPrerequisiteCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| notExecutedByAutomation | ``$($validation.notExecutedByAutomation)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish owner manual command handoff validation written to $jsonPath"
Write-Host "ValidationState=$validationState Commands=$($validation.commandCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Public publish owner manual command handoff has blocker validation failures."
}
