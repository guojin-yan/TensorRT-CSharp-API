[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\strict-close-ready-convergence-dashboard.json",
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

function Get-LaneById {
  param([AllowNull()][object[]]$Lanes, [string]$Id)
  return @($Lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") -eq $Id } | Select-Object -First 1)[0]
}

function Sum-IntProperty {
  param([AllowNull()][object[]]$Items, [string]$Name)
  $total = 0
  foreach ($item in @($Items)) {
    $total += [int](Get-PropertyOrDefault -Object $item -Name $Name -DefaultValue 0)
  }

  return $total
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Strict close ready convergence dashboard not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "closeReadinessLanes" -DefaultValue @()))
$items = New-Object System.Collections.Generic.List[object]
$requiredIds = @(
  "github-actions-run-proof",
  "owner-public-publish-result",
  "public-package-download-proof",
  "post-publish-clean-consumer-proof",
  "final-release-close-blocker-dashboard",
  "public-publish-result-owner-input",
  "public-publish-result-import",
  "post-publish-clean-consumer-proof-record-contract",
  "post-publish-clean-consumer-result-convergence",
  "release-issue-close-owner-decision-input",
  "release-issue-close-final-owner-decision-audit",
  "release-issue-close-record-validation",
  "release-evidence-classification-audit"
)
$ids = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") })
$missingRequired = @($requiredIds | Where-Object { $ids -notcontains $_ })

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

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "strict-close-ready-convergence-dashboard") -Severity "blocker" -Detail "recordKind must be strict-close-ready-convergence-dashboard.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes-present" -Passed ($missingRequired.Count -eq 0) -Severity "blocker" -Detail "Missing required close lanes: $($missingRequired -join ', ')")) | Out-Null
$sourceArtifacts = @((Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()) | ForEach-Object { [string]$_ })
$requiredSourceArtifacts = @(
  "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json",
  "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.md",
  "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json",
  "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.md",
  "artifacts/final-release/public-publish-result-owner-input-validation.json",
  "artifacts/final-release/public-publish-result-import-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
  "artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json",
  "artifacts/final-release/release-issue-close-record-validation.json",
  "artifacts/final-release/release-evidence-classification-audit.json",
  "artifacts/final-release/public-release-owner-execution-package-validation.json",
  "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json",
  "artifacts/final-release/post-publish-user-verification-pack-validation.json"
)
$missingSourceArtifacts = @($requiredSourceArtifacts | Where-Object { $sourceArtifacts -notcontains $_ })
$items.Add((New-ValidationItem -Id "required-source-artifacts-present" -Passed ($missingSourceArtifacts.Count -eq 0) -Severity "blocker" -Detail "Missing required strict-close source artifacts: $($missingSourceArtifacts -join ', ')")) | Out-Null

$remoteProofIds = @("github-actions-run-proof", "owner-public-publish-result", "public-package-download-proof", "post-publish-clean-consumer-proof")
$remoteProofLanes = @($remoteProofIds | ForEach-Object { Get-LaneById -Lanes $lanes -Id $_ })
$remoteProofLanesPresent = @($remoteProofLanes | Where-Object { $null -ne $_ }).Count -eq $remoteProofIds.Count
$remoteProofLanesBlockClose = $remoteProofLanesPresent -and @($remoteProofLanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $true) }).Count -eq 0 -and @($remoteProofLanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "blocksStrictClose" -DefaultValue $false) }).Count -eq 0
$ownerFieldSurfaceLanes = @($lanes | Where-Object { [int](Get-PropertyOrDefault -Object $_ -Name "requiredOwnerFieldCount" -DefaultValue 0) -gt 0 })
$requiredOwnerFieldCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "requiredOwnerFieldCount"
$blockedRequiredOwnerFieldCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "blockedRequiredOwnerFieldCount"
$readyOwnerFieldCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "readyOwnerFieldCount"
$rejectedSubstituteCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "rejectedSubstituteCount"
$sourceReadinessSignalCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "sourceReadinessSignalCount"
$ownerPublicPublishResultLane = Get-LaneById -Lanes $lanes -Id "owner-public-publish-result"
$publicPackageDownloadProofLane = Get-LaneById -Lanes $lanes -Id "public-package-download-proof"
$postPublishProofLane = Get-LaneById -Lanes $lanes -Id "post-publish-clean-consumer-proof"
$ownerFieldSurfaceReady =
  $null -ne $ownerPublicPublishResultLane -and
  $null -ne $publicPackageDownloadProofLane -and
  $null -ne $postPublishProofLane -and
  [int](Get-PropertyOrDefault -Object $ownerPublicPublishResultLane -Name "requiredOwnerFieldCount" -DefaultValue 0) -ge 30 -and
  [int](Get-PropertyOrDefault -Object $publicPackageDownloadProofLane -Name "requiredOwnerFieldCount" -DefaultValue 0) -ge 20 -and
  [int](Get-PropertyOrDefault -Object $postPublishProofLane -Name "requiredOwnerFieldCount" -DefaultValue 0) -ge 19 -and
  $requiredOwnerFieldCountFromLanes -ge 69 -and
  $blockedRequiredOwnerFieldCountFromLanes -eq $requiredOwnerFieldCountFromLanes -and
  $readyOwnerFieldCountFromLanes -eq 0 -and
  $rejectedSubstituteCountFromLanes -ge 30 -and
  $sourceReadinessSignalCountFromLanes -ge 15
$ownerFieldSurfaceTotalsConsistent =
  [int](Get-PropertyOrDefault -Object $record -Name "requiredOwnerFieldCount" -DefaultValue -1) -eq $requiredOwnerFieldCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "blockedRequiredOwnerFieldCount" -DefaultValue -1) -eq $blockedRequiredOwnerFieldCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "readyOwnerFieldCount" -DefaultValue -1) -eq $readyOwnerFieldCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "rejectedSubstituteCount" -DefaultValue -1) -eq $rejectedSubstituteCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "sourceReadinessSignalCount" -DefaultValue -1) -eq $sourceReadinessSignalCountFromLanes
$postPublishProofRequiresCandidate = $null -ne $postPublishProofLane -and
  [bool](Get-PropertyOrDefault -Object $postPublishProofLane -Name "remoteGateStateReady" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $postPublishProofLane -Name "requireProofReady" -DefaultValue $false) -and
  [string](Get-PropertyOrDefault -Object $postPublishProofLane -Name "proofReadyProperty" -DefaultValue "") -eq "proofCandidateReady" -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishProofLane -Name "proofReady" -DefaultValue $true) -and
  -not [bool](Get-PropertyOrDefault -Object $postPublishProofLane -Name "ready" -DefaultValue $true)
$items.Add((New-ValidationItem -Id "remote-proof-lanes-present" -Passed $remoteProofLanesPresent -Severity "blocker" -Detail "Strict close dashboard must include remote proof dependency lanes from remote-ci-and-public-publish-proof-backfill-gate.")) | Out-Null
$items.Add((New-ValidationItem -Id "remote-proof-lanes-block-close" -Passed $remoteProofLanesBlockClose -Severity "blocker" -Detail "Missing real GitHub Actions, public publish, public download, and post-publish proof lanes must block strict close.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-proof-lane-requires-proof-candidate-ready" -Passed $postPublishProofRequiresCandidate -Severity "blocker" -Detail "Post-publish proof lane must stay blocked when validation is ready but proofCandidateReady is false.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-field-surface-lanes-present" -Passed $ownerFieldSurfaceReady -Severity "blocker" -Detail "Strict close dashboard must expose public release, public download, and post-publish owner field surfaces, all still blocked until real owner evidence is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-owner-field-surface-consistent" -Passed $ownerFieldSurfaceTotalsConsistent -Severity "blocker" -Detail "Strict close dashboard top-level owner field counts must match close lane owner field counts.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-close-lanes-ready" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0) -eq 0 -and [string](Get-PropertyOrDefault -Object $record -Name "dashboardState" -DefaultValue "") -eq "strict-close-ready-convergence-ready") -Severity "action-required" -Detail "Dashboard remains blocked until every strict close lane is backed by real Owner proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "lanes-safe" -Passed $allLanesSafe -Severity "blocker" -Detail "Every close readiness lane must keep proof/publish/close flags false and include owner action plus validator.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -Severity "blocker" -Detail "Strict close dashboard must not publish, prove runtime/post-publish, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-strict-close-ready-convergence-dashboard"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-strict-close-ready-owner-action-required"
}
else {
  "strict-close-ready-convergence-dashboard-ready"
}

$validation = [pscustomobject]@{
  recordKind = "strict-close-ready-convergence-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  laneCount = $lanes.Count
  blockedLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0)
  readyLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "readyLaneCount" -DefaultValue 0)
  requiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredOwnerFieldCount" -DefaultValue 0)
  blockedRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0)
  readyOwnerFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "readyOwnerFieldCount" -DefaultValue 0)
  rejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "rejectedSubstituteCount" -DefaultValue 0)
  sourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "sourceReadinessSignalCount" -DefaultValue 0)
  ownerFieldSurfaceLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "ownerFieldSurfaceLaneCount" -DefaultValue 0)
  blockedOwnerFieldSurfaceLaneCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedOwnerFieldSurfaceLaneCount" -DefaultValue 0)
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
  safetyBoundary = "StrictCloseReady convergence dashboard validation only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "strict-close-ready-convergence-dashboard-validation.json"
$markdownPath = Join-Path $OutputRoot "strict-close-ready-convergence-dashboard-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Strict Close Ready Convergence Dashboard Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| laneCount | ``$($validation.laneCount)`` |
| blockedLaneCount | ``$($validation.blockedLaneCount)`` |
| readyLaneCount | ``$($validation.readyLaneCount)`` |
| requiredOwnerFieldCount | ``$($validation.requiredOwnerFieldCount)`` |
| blockedRequiredOwnerFieldCount | ``$($validation.blockedRequiredOwnerFieldCount)`` |
| readyOwnerFieldCount | ``$($validation.readyOwnerFieldCount)`` |
| rejectedSubstituteCount | ``$($validation.rejectedSubstituteCount)`` |
| sourceReadinessSignalCount | ``$($validation.sourceReadinessSignalCount)`` |
| ownerFieldSurfaceLaneCount | ``$($validation.ownerFieldSurfaceLaneCount)`` |
| blockedOwnerFieldSurfaceLaneCount | ``$($validation.blockedOwnerFieldSurfaceLaneCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Strict close ready convergence dashboard validation written to $jsonPath"
Write-Host "ValidationState=$validationState Lanes=$($validation.laneCount) Blocked=$($validation.blockedLaneCount) FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Strict close ready convergence dashboard has blocker validation failures."
}
