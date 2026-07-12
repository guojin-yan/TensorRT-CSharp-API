[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-close-readiness-checkpoint.json",
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

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  throw "Final owner close readiness checkpoint not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$checks = @((Get-PropertyOrDefault -Object $record -Name "readinessChecks" -DefaultValue @()))
$blockedChecks = @($checks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$requiredChecks = @("real-public-package-result-ready", "clean-consumer-smoke-ready", "clean-external-package-consumer-owner-runbook-ready", "post-publish-owner-verification-runbook-ready", "forbidden-substitute-scan-ready", "owner-external-result-import-ready", "real-external-import-validator-ready", "owner-result-candidate-bridge-ready", "real-proof-import-bridge-ready", "strict-owner-decision-ready", "final-close-gate-convergence-ready", "release-issue-close-record-ready")
$checkIds = @($checks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$sourceArtifacts = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @())) | ForEach-Object { [string]$_ })
$forbiddenSubstituteMarkers = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteMarkers" -DefaultValue @())) | ForEach-Object { [string]$_ })
$acceptedProofSources = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "finalCloseAcceptedProofSources" -DefaultValue @())) | ForEach-Object { [string]$_ })
$summary = Get-PropertyOrDefault -Object $record -Name "summary" -DefaultValue $null

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-close-readiness-checkpoint") -Severity "blocker" -Detail "recordKind must be final-owner-close-readiness-checkpoint.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "checkpointState" -DefaultValue "") -eq "blocked-final-owner-close-readiness-owner-proof-required") -Severity "blocker" -Detail "Checkpoint must stay blocked until owner supplies real proof records and strict validators pass.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-checks-present" -Passed (@($requiredChecks | Where-Object { $checkIds -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checkpoint must expose all final owner close readiness checks.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-runbook-checks-present" -Passed (($checkIds -contains "clean-external-package-consumer-owner-runbook-ready") -and ($checkIds -contains "post-publish-owner-verification-runbook-ready") -and (($sourceArtifacts -join "`n").Contains("clean-external-package-consumer-owner-runbook-validation.json", [StringComparison]::OrdinalIgnoreCase)) -and (($sourceArtifacts -join "`n").Contains("post-publish-owner-verification-runbook-validation.json", [StringComparison]::OrdinalIgnoreCase))) -Severity "blocker" -Detail "Checkpoint must include both clean external consumer and post-publish owner runbook readiness lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-input-required" -Passed ($blockedChecks.Count -eq 0) -Severity "action-required" -Detail "Owner must satisfy all real proof and close readiness checks.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Checkpoint must not publish, approve, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-boundary" -Passed ([bool](Get-PropertyOrDefault -Object $summary -Name "strictValidatorRequired" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $summary -Name "candidateInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $summary -Name "bridgeInputOnly" -DefaultValue $false) -and $acceptedProofSources -contains "strict-validator-accepted-real-external-proof-record") -Severity "blocker" -Detail "Checkpoint must require strict validator accepted real proof and keep candidates/bridges input only.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitutes" -Passed ((@("candidate","draft","dashboard","dry-run","local feed","ProjectReference","direct .nupkg","template","build-only","blocked-by-cuda-driver") | Where-Object { $forbiddenSubstituteMarkers -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Checkpoint must list forbidden substitutes that cannot close release.")) | Out-Null
$items.Add((New-ValidationItem -Id "readiness-check-boundary-fields" -Passed (@($checks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "strictValidatorRequired" -DefaultValue $false) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "strictValidatorInputOnly" -DefaultValue $false) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "blockedReason" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Every readiness check must carry strict validator and blocked reason boundary fields.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-owner-close-readiness-checkpoint" } else { "blocked-final-owner-close-readiness-owner-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-close-readiness-checkpoint-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  readinessCheckCount = $checks.Count
  blockedReadinessCheckCount = $blockedChecks.Count
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
  boundary = "Validation checks final owner close readiness checkpoint shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-close-readiness-checkpoint-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-close-readiness-checkpoint-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Owner Close Readiness Checkpoint Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| readinessCheckCount | ``$($validation.readinessCheckCount)`` |",
  "| blockedReadinessCheckCount | ``$($validation.blockedReadinessCheckCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Boundary",
  "",
  $validation.boundary
)

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final owner close readiness checkpoint validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Checks=$($validation.readinessCheckCount) Blocked=$($validation.blockedReadinessCheckCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Final owner close readiness checkpoint validation failed with $($failedBlockers.Count) blocker(s)."
}
