[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-proof-real-backfill-execution-pack.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

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
  throw "Owner proof real backfill execution pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$packState = [string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "")
$ownerInputTaskCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerInputTaskCount" -DefaultValue -1)
$realProofTaskCount = [int](Get-PropertyOrDefault -Object $record -Name "realProofTaskCount" -DefaultValue -1)
$hashCheckTaskCount = [int](Get-PropertyOrDefault -Object $record -Name "hashCheckTaskCount" -DefaultValue -1)
$blockedTaskCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedTaskCount" -DefaultValue -1)
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$nonSubstituteKinds = @(Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @())
$ownerInputTasks = @(Get-PropertyOrDefault -Object $record -Name "ownerInputTasks" -DefaultValue @())
$realProofTasks = @(Get-PropertyOrDefault -Object $record -Name "realProofTasks" -DefaultValue @())
$hashCheckTasks = @(Get-PropertyOrDefault -Object $record -Name "hashCheckTasks" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-proof-real-backfill-execution-pack") -Severity "blocker" -Detail "recordKind must be owner-proof-real-backfill-execution-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-pack-state" -Passed ($packState -eq "blocked-owner-real-proof-backfill-required") -Severity "blocker" -Detail "Pack must remain blocked until real owner proof is backfilled.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-task-count" -Passed ($ownerInputTaskCount -ge 8 -and $ownerInputTasks.Count -ge 8) -Severity "blocker" -Detail "Pack must include owner input tasks from strict candidate requiredOwnerFields.")) | Out-Null
$items.Add((New-ValidationItem -Id "real-proof-task-count" -Passed ($realProofTaskCount -ge 5 -and $realProofTasks.Count -ge 5) -Severity "blocker" -Detail "Pack must include real proof tasks from strict candidate blockers.")) | Out-Null
$items.Add((New-ValidationItem -Id "hash-check-task-count" -Passed ($hashCheckTaskCount -ge 9 -and $hashCheckTasks.Count -ge 9) -Severity "blocker" -Detail "Pack must include hash check tasks from strict candidate hash lines.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-task-count" -Passed ($blockedTaskCount -ge 1) -Severity "action-required" -Detail "Real owner proof backfill remains required.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Pack must not publish, approve publication, or close release issue.")) | Out-Null

foreach ($task in @($ownerInputTasks + $realProofTasks + $hashCheckTasks)) {
  $taskId = [string](Get-PropertyOrDefault -Object $task -Name "id" -DefaultValue "unknown")
  $firstCommand = [string](Get-PropertyOrDefault -Object $task -Name "firstCommand" -DefaultValue "")
  $validatorCommand = [string](Get-PropertyOrDefault -Object $task -Name "validatorCommand" -DefaultValue "")
  $items.Add((New-ValidationItem -Id "task-command-$taskId" -Passed (-not [string]::IsNullOrWhiteSpace($firstCommand) -and -not [string]::IsNullOrWhiteSpace($validatorCommand)) -Severity "blocker" -Detail "$taskId must include firstCommand and validatorCommand.")) | Out-Null
}

foreach ($requiredKind in @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "candidate", "schema-only", "preflight-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "non-substitute-$($requiredKind.Replace(' ', '-').Replace('.', ''))" -Passed ($nonSubstituteKinds -contains $requiredKind) -Severity "blocker" -Detail "nonSubstituteProofKinds must include $requiredKind.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "owner-real-proof-backfill-still-required" -Passed ($blockedTaskCount -eq 0) -Severity "action-required" -Detail "blockedTaskCount must reach zero only after real owner inputs and real proof are backfilled.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0) {
  "owner-proof-real-backfill-execution-pack-ready"
}
elseif ($failedBlockers.Count -eq 0) {
  "blocked-owner-real-proof-backfill-required"
}
else {
  "invalid-owner-proof-real-backfill-execution-pack"
}

$validation = [pscustomobject]@{
  recordKind = "owner-proof-real-backfill-execution-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  packState = $packState
  isValidExecutionPackShape = ($failedBlockers.Count -eq 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  ownerInputTaskCount = $ownerInputTaskCount
  realProofTaskCount = $realProofTaskCount
  hashCheckTaskCount = $hashCheckTaskCount
  blockedTaskCount = $blockedTaskCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates owner backfill execution pack shape only. Passing blocker checks does not publish packages, promote proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-proof-real-backfill-execution-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-proof-real-backfill-execution-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Owner Proof Real Backfill Execution Pack Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| packState | ``$($validation.packState)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| ownerInputTaskCount | ``$($validation.ownerInputTaskCount)`` |
| realProofTaskCount | ``$($validation.realProofTaskCount)`` |
| hashCheckTaskCount | ``$($validation.hashCheckTaskCount)`` |
| blockedTaskCount | ``$($validation.blockedTaskCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner proof real backfill execution pack validation written to $jsonPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) BlockedTaskCount=$blockedTaskCount PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Owner proof real backfill execution pack validation failed with $($failedBlockers.Count) blocker(s)."
}
