[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-final-owner-runbook.json",
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
  throw "Release close final owner runbook not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$runbookState = [string](Get-PropertyOrDefault -Object $record -Name "runbookState" -DefaultValue "")
$runbookSteps = @(Get-PropertyOrDefault -Object $record -Name "runbookSteps" -DefaultValue @())
$failureRecoveryMap = @(Get-PropertyOrDefault -Object $record -Name "failureRecoveryMap" -DefaultValue @())
$strictValidators = @(Get-PropertyOrDefault -Object $record -Name "strictValidators" -DefaultValue @())
$nonSubstituteProofKinds = @(Get-PropertyOrDefault -Object $record -Name "nonSubstituteProofKinds" -DefaultValue @())
$runbookStepCount = [int](Get-PropertyOrDefault -Object $record -Name "runbookStepCount" -DefaultValue 0)
$blockedStepCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedStepCount" -DefaultValue 0)
$ownerActionStepCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerActionStepCount" -DefaultValue 0)
$strictValidatorCount = [int](Get-PropertyOrDefault -Object $record -Name "strictValidatorCount" -DefaultValue 0)
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)

$stepText = ($runbookSteps | ConvertTo-Json -Depth 10)
$validatorText = ($strictValidators -join "`n")

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "release-close-final-owner-runbook") -Severity "blocker" -Detail "recordKind must be release-close-final-owner-runbook.")) | Out-Null
$items.Add((New-ValidationItem -Id "runbook-state" -Passed ($runbookState -eq "blocked-release-close-final-owner-action-required") -Severity "blocker" -Detail "runbookState must remain blocked-release-close-final-owner-action-required.")) | Out-Null
$items.Add((New-ValidationItem -Id "step-count" -Passed ($runbookStepCount -ge 12 -and $runbookSteps.Count -eq $runbookStepCount) -Severity "blocker" -Detail "Runbook must contain at least 12 concrete owner steps and count must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-step-count" -Passed ($blockedStepCount -ge 1) -Severity "action-required" -Detail "Runbook must remain blocked until real Owner proof and close inputs are present.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-action-step-count" -Passed ($ownerActionStepCount -ge 1) -Severity "action-required" -Detail "Runbook must expose owner action steps.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-count" -Passed ($strictValidatorCount -ge 1 -and $strictValidators.Count -eq $strictValidatorCount) -Severity "blocker" -Detail "Runbook must include strict validators.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-channel-step" -Passed ($stepText -match "public package channel|public package source|public-channel") -Severity "blocker" -Detail "Runbook must include public package channel/source confirmation.")) | Out-Null
$items.Add((New-ValidationItem -Id "clean-consumer-runtime-smoke-step" -Passed ($stepText -match "clean consumer runtime smoke|clean consumer|runtime smoke") -Severity "blocker" -Detail "Runbook must include clean consumer runtime smoke execution.")) | Out-Null
$items.Add((New-ValidationItem -Id "final-close-decision-step" -Passed ($stepText -match "final close decision") -Severity "blocker" -Detail "Runbook must include final close decision input.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-close-validator-step" -Passed ($stepText -match "FailOnNotCloseReady" -and $validatorText -match "Test-ReleaseIssueCloseRecord.ps1") -Severity "blocker" -Detail "Runbook must include the strict close validator with FailOnNotCloseReady.")) | Out-Null
$items.Add((New-ValidationItem -Id "failure-recovery-map" -Passed ($failureRecoveryMap.Count -ge 5 -and (($failureRecoveryMap | ConvertTo-Json -Depth 8) -match "post-publish-verification-owner-input") -and (($failureRecoveryMap | ConvertTo-Json -Depth 8) -match "package-consumer-runtime-proof-candidate")) -Severity "blocker" -Detail "Runbook must map key failures back to owner input/proof repair artifacts.")) | Out-Null

foreach ($requiredKind in @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "candidate", "schema-only", "preflight-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "non-substitute-$($requiredKind.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($nonSubstituteProofKinds -contains $requiredKind) -Severity "blocker" -Detail "nonSubstituteProofKinds must include $requiredKind.")) | Out-Null
}

$items.Add((New-ValidationItem -Id "no-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Runbook must not publish, approve public release, or close release issues.")) | Out-Null

$failedBlockers = @($items | Where-Object { $_.severity -eq "blocker" -and -not $_.passed })
$failedActionRequired = @($items | Where-Object { $_.severity -eq "action-required" -and $_.passed })
$failedBlockerCount = $failedBlockers.Count
$failedActionRequiredCount = $failedActionRequired.Count
$isValidRunbookShape = $failedBlockerCount -eq 0
$validationState = if ($isValidRunbookShape -and $failedActionRequiredCount -gt 0) { "blocked-release-close-final-owner-action-required" } else { "invalid-release-close-final-owner-runbook" }

$validation = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-close-final-owner-runbook-validation"
  validationState = $validationState
  isValidRunbookShape = $isValidRunbookShape
  failedBlockerCount = $failedBlockerCount
  failedActionRequiredCount = $failedActionRequiredCount
  runbookStepCount = $runbookStepCount
  blockedStepCount = $blockedStepCount
  ownerActionStepCount = $ownerActionStepCount
  strictValidatorCount = $strictValidatorCount
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  validationItems = $items
}

$jsonPath = Join-Path $OutputRoot "release-close-final-owner-runbook-validation.json"
$markdownPath = Join-Path $OutputRoot "release-close-final-owner-runbook-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$itemLines = $items | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.severity)`` | ``$($_.passed)`` | $($_.detail.Replace("|", "\|")) |"
}

$markdown = @"
# Release Close Final Owner Runbook Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| isValidRunbookShape | ``$($validation.isValidRunbookShape)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| runbookStepCount | ``$($validation.runbookStepCount)`` |
| blockedStepCount | ``$($validation.blockedStepCount)`` |
| ownerActionStepCount | ``$($validation.ownerActionStepCount)`` |
| strictValidatorCount | ``$($validation.strictValidatorCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Validation Items

| ID | Severity | Passed | Detail |
|---|---|---|---|
$($itemLines -join "`r`n")
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close final owner runbook validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($validation.validationState) FailedBlockers=$failedBlockerCount FailedActionRequired=$failedActionRequiredCount Steps=$runbookStepCount BlockedSteps=$blockedStepCount OwnerActionSteps=$ownerActionStepCount StrictValidators=$strictValidatorCount PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($Strict -and -not $isValidRunbookShape) {
  throw "Release close final owner runbook validation failed blocker checks. FailedBlockerCount=$failedBlockerCount"
}
