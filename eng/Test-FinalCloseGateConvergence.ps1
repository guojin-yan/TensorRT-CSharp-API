[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-close-gate-convergence.json",
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
  throw "Final close gate convergence not found: $InputPath"
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "gateLanes" -DefaultValue @()))
$blockedLanes = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$forbiddenSubstituteMarkers = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteMarkers" -DefaultValue @())) | ForEach-Object { [string]$_ })
$rejectedCloseSubstitutes = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "rejectedCloseSubstitutes" -DefaultValue @())) | ForEach-Object { [string]$_ })
$strictValidatorSourceArtifacts = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "strictValidatorSourceArtifacts" -DefaultValue @())) | ForEach-Object { [string]$_ })
$acceptedProofSources = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "finalCloseAcceptedProofSources" -DefaultValue @())) | ForEach-Object { [string]$_ })
$summary = Get-PropertyOrDefault -Object $record -Name "summary" -DefaultValue $null

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-close-gate-convergence") -Severity "blocker" -Detail "recordKind must be final-close-gate-convergence.")) | Out-Null
$items.Add((New-ValidationItem -Id "state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "convergenceState" -DefaultValue "") -eq "blocked-final-close-gate-owner-proof-required") -Severity "blocker" -Detail "Convergence must stay blocked until all real close proof lanes pass.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-shape" -Passed ($lanes.Count -ge 10) -Severity "blocker" -Detail "Convergence must expose all final close gate lanes, including owner import and strict validator bridge lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-lanes-required" -Passed ($blockedLanes.Count -eq 0) -Severity "action-required" -Detail "Owner must complete all final close gate lanes.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $record -Name "ownerExecutionOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Convergence must not publish, approve, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-sources" -Passed ($strictValidatorSourceArtifacts -contains "artifacts/final-release/real-external-proof-record-import-validator-validation.json" -and $strictValidatorSourceArtifacts -contains "artifacts/final-release/release-close-real-proof-import-bridge-validation.json" -and $acceptedProofSources -contains "strict-validator-accepted-real-external-proof-record") -Severity "blocker" -Detail "Final close must name strict validator accepted real proof as the only promotable source.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-bridge-input-only" -Passed ([bool](Get-PropertyOrDefault -Object $summary -Name "candidateInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $summary -Name "bridgeInputOnly" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $summary -Name "strictValidatorRequired" -DefaultValue $false)) -Severity "blocker" -Detail "Candidate and bridge records must remain strict-validator input only.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitute-markers" -Passed ((@("candidate","draft","dashboard","dry-run","local feed","ProjectReference","direct .nupkg","template","build-only","blocked-by-cuda-driver") | Where-Object { $forbiddenSubstituteMarkers -notcontains $_ -or $rejectedCloseSubstitutes -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Final close must reject candidate, draft, dashboard, dry-run, local feed, ProjectReference, direct nupkg, template, build-only, and blocked-by-driver substitutes.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-boundary-fields" -Passed (@($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "strictValidatorRequired" -DefaultValue $false) -or -not [bool](Get-PropertyOrDefault -Object $_ -Name "strictValidatorInputOnly" -DefaultValue $false) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "blockedReason" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Every lane must carry strict-validator and blocked reason boundary fields.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-final-close-gate-convergence" } else { "blocked-final-close-gate-owner-proof-required" }

$validation = [pscustomobject]@{
  recordKind = "final-close-gate-convergence-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $InputPath
  validationState = $validationState
  laneCount = $lanes.Count
  blockedLaneCount = $blockedLanes.Count
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
  boundary = "Validation checks final close convergence shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-close-gate-convergence-validation.json"
$markdownPath = Join-Path $OutputRoot "final-close-gate-convergence-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Final Close Gate Convergence Validation",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| validationState | ``$($validation.validationState)`` |",
  "| laneCount | ``$($validation.laneCount)`` |",
  "| blockedLaneCount | ``$($validation.blockedLaneCount)`` |",
  "| failedBlockerCount | ``$($validation.failedBlockerCount)`` |",
  "| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |",
  "| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |",
  "",
  "## Boundary",
  "",
  $validation.boundary
)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final close gate convergence validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Lanes=$($validation.laneCount) Blocked=$($validation.blockedLaneCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Final close gate convergence validation failed with $($failedBlockers.Count) blocker(s)."
}
