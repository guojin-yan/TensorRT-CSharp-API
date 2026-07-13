[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-public-release-closure-bridge.json",
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
  throw "Final public release closure bridge not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "closureLanes" -DefaultValue @()))
$sourceArtifacts = @((Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()) | ForEach-Object { [string]$_ })
$items = New-Object System.Collections.Generic.List[object]

$requiredLaneIds = @(
  "owner-publish-authorization",
  "owner-publish-execution-result",
  "public-package-download-proof",
  "clean-external-consumer-smoke",
  "post-publish-proof",
  "release-issue-close-owner-decision",
  "strict-close-ready-convergence-dashboard"
)

$requiredArtifacts = @(
  "artifacts/final-release/owner-publish-authorization-input-validation.json",
  "artifacts/final-release/owner-publish-execution-result-input-validation.json",
  "artifacts/final-release/public-package-download-proof-candidate-validation.json",
  "artifacts/final-release/clean-external-consumer-smoke-input-validation.json",
  "artifacts/final-release/post-publish-proof-input-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
  "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json"
)

$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") })
$missingLaneIds = @($requiredLaneIds | Where-Object { $laneIds -notcontains $_ })
$missingArtifacts = @($requiredArtifacts | Where-Object { $sourceArtifacts -notcontains $_ })

$allLanesSafe = $lanes.Count -ge $requiredLaneIds.Count
foreach ($lane in $lanes) {
  $allLanesSafe = $allLanesSafe -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "usesPublishToken" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isReleaseCloseProof" -DefaultValue $true) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "ownerAction" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "boundary" -DefaultValue ""))
}

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-public-release-closure-bridge") -Severity "blocker" -Detail "recordKind must be final-public-release-closure-bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes-present" -Passed ($missingLaneIds.Count -eq 0) -Severity "blocker" -Detail "Missing required lanes: $($missingLaneIds -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "required-source-artifacts-present" -Passed ($missingArtifacts.Count -eq 0) -Severity "blocker" -Detail "Missing source artifacts: $($missingArtifacts -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-count-consistent" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "laneCount" -DefaultValue 0) -eq $lanes.Count) -Severity "blocker" -Detail "laneCount must match closureLanes count.")) | Out-Null
$items.Add((New-ValidationItem -Id "lanes-safe" -Passed $allLanesSafe -Severity "blocker" -Detail "Every lane must keep publish/token/close/proof flags false and include ownerAction plus boundary.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not publish, use tokens, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-close-lanes-ready" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0) -eq 0 -and [string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "") -eq "final-public-release-closure-bridge-ready-for-owner-close-review") -Severity "action-required" -Detail "Bridge remains blocked until all owner authorization, owner publish execution result, public download, external smoke, post-publish proof, close decision, and strict dashboard lanes are ready.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-final-public-release-closure-bridge"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-final-public-release-closure-real-owner-proof-required"
}
else {
  "final-public-release-closure-bridge-ready-for-owner-close-review"
}

$validation = [pscustomobject]@{
  recordKind = "final-public-release-closure-bridge-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  laneCount = $lanes.Count
  readyLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "readyLaneCount" -DefaultValue 0)
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  missingArtifactCount = [int](Get-PropertyOrDefault -Object $record -Name "missingArtifactCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  notExecutedByAutomation = $true
  performsPublish = $false
  usesPublishToken = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $false)
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Final public release closure bridge validation only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and not a substitute for real public package or clean external consumer smoke evidence."
}

$jsonPath = Join-Path $OutputRoot "final-public-release-closure-bridge-validation.json"
$markdownPath = Join-Path $OutputRoot "final-public-release-closure-bridge-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object { "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace("|", "\|")) |" }
$markdown = @"
# Final Public Release Closure Bridge Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| laneCount | ``$($validation.laneCount)`` |
| readyLaneCount | ``$($validation.readyLaneCount)`` |
| blockedLaneCount | ``$($validation.blockedLaneCount)`` |
| missingArtifactCount | ``$($validation.missingArtifactCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| usesPublishToken | ``$($validation.usesPublishToken)`` |
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

Write-Host "Final public release closure bridge validation written to $jsonPath"
Write-Host "ValidationState=$($validation.validationState) Lanes=$($validation.laneCount) Blocked=$($validation.blockedLaneCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Final public release closure bridge validation failed with $($failedBlockers.Count) blocker(s)."
}
