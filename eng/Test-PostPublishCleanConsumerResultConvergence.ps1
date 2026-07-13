[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-clean-consumer-result-convergence.json",
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
  throw "Post-publish clean consumer result convergence not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$items = New-Object System.Collections.Generic.List[object]
$requiredIds = @(
  "public-publish-result-import",
  "post-publish-verification",
  "post-publish-clean-consumer-proof-result",
  "package-consumer-runtime-proof",
  "clean-consumer-source-scan",
  "final-post-publish-audit-pack"
)
$ids = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") })
$missingRequired = @($requiredIds | Where-Object { $ids -notcontains $_ })
$crossLaneConsistencyChecks = @((Get-PropertyOrDefault -Object $record -Name "crossLaneConsistencyChecks" -DefaultValue @()))
$checkIds = @($crossLaneConsistencyChecks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredCheckIds = @(
  "post-publish-proof-result-validation-ready",
  "post-publish-proof-candidate-ready",
  "post-publish-source-proof-linkage-ready",
  "post-publish-source-proof-flags-ready",
  "post-publish-public-package-url-match",
  "post-publish-public-package-version-match",
  "post-publish-public-package-sha-match"
)
$missingCheckIds = @($requiredCheckIds | Where-Object { $checkIds -notcontains $_ })
$failedConsistencyBlockers = @($crossLaneConsistencyChecks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "blocker" })
$failedConsistencyActionRequired = @($crossLaneConsistencyChecks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "action-required" })

$allLanesSafe = $lanes.Count -ge $requiredIds.Count
foreach ($lane in $lanes) {
  $allLanesSafe = $allLanesSafe -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isReleaseCloseProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isPostPublishProof" -DefaultValue $true) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "ownerNextAction" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "validator" -DefaultValue ""))
}

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-clean-consumer-result-convergence") -Severity "blocker" -Detail "recordKind must be post-publish-clean-consumer-result-convergence.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes-present" -Passed ($missingRequired.Count -eq 0) -Severity "blocker" -Detail "Missing required lanes: $($missingRequired -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "required-consistency-checks-present" -Passed ($missingCheckIds.Count -eq 0) -Severity "blocker" -Detail "Missing required consistency checks: $($missingCheckIds -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "cross-lane-consistency-no-blockers" -Passed ($failedConsistencyBlockers.Count -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "failedConsistencyBlockerCount" -DefaultValue 0) -eq 0) -Severity "blocker" -Detail "Post-publish convergence consistency checks must not contain blocker failures.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-lanes-ready" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0) -eq 0 -and [string](Get-PropertyOrDefault -Object $record -Name "convergenceState" -DefaultValue "") -eq "post-publish-clean-consumer-result-convergence-ready") -Severity "action-required" -Detail "Convergence remains blocked until every public publish, clean consumer, package-consumer runtime, scan, and audit lane is ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "lanes-safe" -Passed $allLanesSafe -Severity "blocker" -Detail "Every lane must keep proof/publish/close flags false and include owner action plus validator.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Convergence validation must not publish, prove runtime/post-publish, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-post-publish-clean-consumer-result-convergence"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-post-publish-clean-consumer-result-required"
}
else {
  "post-publish-clean-consumer-result-convergence-ready"
}

$validation = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-result-convergence-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  laneCount = $lanes.Count
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  readyLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "readyLaneCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  crossLaneConsistencyCheckCount = $crossLaneConsistencyChecks.Count
  failedConsistencyBlockerCount = $failedConsistencyBlockers.Count
  failedConsistencyActionRequiredCount = $failedConsistencyActionRequired.Count
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Post-publish clean consumer result convergence validation only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-clean-consumer-result-convergence-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-clean-consumer-result-convergence-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Post-Publish Clean Consumer Result Convergence Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| laneCount | ``$($validation.laneCount)`` |
| blockedLaneCount | ``$($validation.blockedLaneCount)`` |
| readyLaneCount | ``$($validation.readyLaneCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| crossLaneConsistencyCheckCount | ``$($validation.crossLaneConsistencyCheckCount)`` |
| failedConsistencyBlockerCount | ``$($validation.failedConsistencyBlockerCount)`` |
| failedConsistencyActionRequiredCount | ``$($validation.failedConsistencyActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish clean consumer result convergence validation written to $jsonPath"
Write-Host "ValidationState=$validationState Lanes=$($validation.laneCount) Blocked=$($validation.blockedLaneCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count) FailedConsistencyBlockers=$($failedConsistencyBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Post-publish clean consumer result convergence has blocker validation failures."
}
