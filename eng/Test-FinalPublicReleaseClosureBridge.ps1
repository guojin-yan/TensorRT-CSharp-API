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

function Get-LaneById {
  param([AllowNull()][object[]]$Lanes, [string]$Id)
  foreach ($lane in @($Lanes)) {
    if ([string](Get-PropertyOrDefault -Object $lane -Name "laneId" -DefaultValue "") -eq $Id) {
      return $lane
    }
  }

  return $null
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
  throw "Final public release closure bridge not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "closureLanes" -DefaultValue @()))
$sourceArtifacts = @((Get-PropertyOrDefault -Object $record -Name "sourceArtifacts" -DefaultValue @()) | ForEach-Object { [string]$_ })
$crossLaneConsistencyChecks = @((Get-PropertyOrDefault -Object $record -Name "crossLaneConsistencyChecks" -DefaultValue @()))
$items = New-Object System.Collections.Generic.List[object]

$requiredLaneIds = @(
  "pre-release-package-proof-readiness",
  "github-actions-run-proof",
  "owner-public-publish-result",
  "owner-publish-authorization",
  "owner-publish-execution-result",
  "public-package-download-proof",
  "public-package-download-owner-execution-pack",
  "clean-external-consumer-smoke",
  "post-publish-proof",
  "post-publish-user-verification-pack",
  "release-issue-close-owner-decision",
  "strict-close-ready-convergence-dashboard"
)

$requiredArtifacts = @(
  "artifacts/final-release/pre-release-package-proof-readiness-matrix.json",
  "artifacts/final-release/github-actions-run-evidence-import-validation.json",
  "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json",
  "artifacts/final-release/owner-publish-authorization-input-validation.json",
  "artifacts/final-release/owner-publish-execution-result-input-validation.json",
  "artifacts/final-release/public-package-download-proof-candidate-validation.json",
  "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json",
  "artifacts/final-release/clean-external-consumer-smoke-input-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json",
  "artifacts/final-release/post-publish-user-verification-pack-validation.json",
  "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
  "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json"
)

$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") })
$missingLaneIds = @($requiredLaneIds | Where-Object { $laneIds -notcontains $_ })
$missingArtifacts = @($requiredArtifacts | Where-Object { $sourceArtifacts -notcontains $_ })
$checkIds = @($crossLaneConsistencyChecks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredCheckIds = @(
  "github-actions-run-evidence-ready",
  "pre-release-readiness-matrix-present",
  "pre-release-readiness-lanes-present",
  "pre-release-readiness-lane-metadata-present",
  "pre-release-readiness-no-premature-promote-flags",
  "pre-release-readiness-ready-for-close",
  "github-actions-run-url-present",
  "github-actions-head-sha-format",
  "github-actions-log-and-artifact-hashes",
  "owner-public-publish-result-ready",
  "owner-public-publish-links-github-actions",
  "public-download-proof-ready",
  "public-download-links-source-proofs",
  "public-package-download-owner-execution-pack-present",
  "public-package-download-owner-execution-pack-blocked",
  "public-package-download-owner-execution-pack-field-surface-present",
  "public-package-download-owner-execution-pack-safe",
  "owner-and-public-download-package-url-match",
  "owner-and-public-download-version-match",
  "owner-and-public-download-sha-match",
  "runtime-package-url-public",
  "github-release-asset-consistent",
  "owner-reviewer-and-timestamp-present",
  "post-publish-proof-candidate-ready",
  "post-publish-links-source-proofs",
  "post-publish-user-verification-pack-present",
  "post-publish-user-verification-pack-blocked",
  "post-publish-user-verification-pack-field-surface-present",
  "post-publish-user-verification-pack-safe",
  "post-publish-owner-package-url-match",
  "post-publish-owner-package-version-match",
  "post-publish-owner-package-sha-match",
  "forbidden-substitutes-absent"
)
$missingCheckIds = @($requiredCheckIds | Where-Object { $checkIds -notcontains $_ })
$failedConsistencyBlockers = @($crossLaneConsistencyChecks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "blocker" })
$failedConsistencyActionRequired = @($crossLaneConsistencyChecks | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -eq "action-required" })

$allLanesSafe = $lanes.Count -ge $requiredLaneIds.Count
foreach ($lane in $lanes) {
  $allLanesSafe = $allLanesSafe -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "performsPublish" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "usesPublishToken" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canPublishPublicly" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "canCloseReleaseIssue" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isPostPublishProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $lane -Name "isReleaseCloseProof" -DefaultValue $true) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "ownerAction" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $lane -Name "boundary" -DefaultValue ""))
}

$preReleaseLane = $null
foreach ($lane in $lanes) {
  if ([string](Get-PropertyOrDefault -Object $lane -Name "laneId" -DefaultValue "") -eq "pre-release-package-proof-readiness") {
    $preReleaseLane = $lane
    break
  }
}
$preReleaseLaneCarriesMatrixMetadata = $null -ne $preReleaseLane -and
  -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $preReleaseLane -Name "requiredEvidence" -DefaultValue "")) -and
  -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $preReleaseLane -Name "validatorPath" -DefaultValue ""))
$closureProofSourceSummary = Get-PropertyOrDefault -Object $record -Name "closureProofSourceSummary" -DefaultValue $null
$preReleaseSummaryPresent = $null -ne $closureProofSourceSummary -and
  $closureProofSourceSummary.PSObject.Properties.Name -contains "preReleaseReadinessMatrixState" -and
  $closureProofSourceSummary.PSObject.Properties.Name -contains "preReleaseLaneMetadataReady" -and
  $closureProofSourceSummary.PSObject.Properties.Name -contains "preReleasePromoteFlagsSafe"
$ownerFieldSurfaceLanes = @($lanes | Where-Object { [int](Get-PropertyOrDefault -Object $_ -Name "requiredOwnerFieldCount" -DefaultValue 0) -gt 0 })
$requiredOwnerFieldCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "requiredOwnerFieldCount"
$blockedRequiredOwnerFieldCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "blockedRequiredOwnerFieldCount"
$readyOwnerFieldCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "readyOwnerFieldCount"
$rejectedSubstituteCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "rejectedSubstituteCount"
$sourceReadinessSignalCountFromLanes = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "sourceReadinessSignalCount"
$publicPackageDownloadOwnerExecutionLane = Get-LaneById -Lanes $lanes -Id "public-package-download-owner-execution-pack"
$postPublishUserVerificationLane = Get-LaneById -Lanes $lanes -Id "post-publish-user-verification-pack"
$publicPackageDownloadOwnerExecutionLaneFieldSurface =
  $null -ne $publicPackageDownloadOwnerExecutionLane -and
  [int](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionLane -Name "requiredOwnerFieldCount" -DefaultValue 0) -ge 20 -and
  [int](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionLane -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0) -eq [int](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionLane -Name "requiredOwnerFieldCount" -DefaultValue 0) -and
  [int](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionLane -Name "rejectedSubstituteCount" -DefaultValue 0) -ge 8 -and
  [int](Get-PropertyOrDefault -Object $publicPackageDownloadOwnerExecutionLane -Name "sourceReadinessSignalCount" -DefaultValue 0) -ge 6
$postPublishUserVerificationLaneFieldSurface =
  $null -ne $postPublishUserVerificationLane -and
  [int](Get-PropertyOrDefault -Object $postPublishUserVerificationLane -Name "requiredOwnerFieldCount" -DefaultValue 0) -ge 19 -and
  [int](Get-PropertyOrDefault -Object $postPublishUserVerificationLane -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0) -eq [int](Get-PropertyOrDefault -Object $postPublishUserVerificationLane -Name "requiredOwnerFieldCount" -DefaultValue 0) -and
  [int](Get-PropertyOrDefault -Object $postPublishUserVerificationLane -Name "rejectedSubstituteCount" -DefaultValue 0) -ge 8 -and
  [int](Get-PropertyOrDefault -Object $postPublishUserVerificationLane -Name "sourceReadinessSignalCount" -DefaultValue 0) -ge 9
$topLevelOwnerFieldSurfaceConsistent =
  [int](Get-PropertyOrDefault -Object $record -Name "requiredOwnerFieldCount" -DefaultValue -1) -eq $requiredOwnerFieldCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "blockedRequiredOwnerFieldCount" -DefaultValue -1) -eq $blockedRequiredOwnerFieldCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "readyOwnerFieldCount" -DefaultValue -1) -eq $readyOwnerFieldCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "rejectedSubstituteCount" -DefaultValue -1) -eq $rejectedSubstituteCountFromLanes -and
  [int](Get-PropertyOrDefault -Object $record -Name "sourceReadinessSignalCount" -DefaultValue -1) -eq $sourceReadinessSignalCountFromLanes

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-public-release-closure-bridge") -Severity "blocker" -Detail "recordKind must be final-public-release-closure-bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-lanes-present" -Passed ($missingLaneIds.Count -eq 0) -Severity "blocker" -Detail "Missing required lanes: $($missingLaneIds -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "required-source-artifacts-present" -Passed ($missingArtifacts.Count -eq 0) -Severity "blocker" -Detail "Missing source artifacts: $($missingArtifacts -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "required-cross-lane-consistency-checks-present" -Passed ($missingCheckIds.Count -eq 0) -Severity "blocker" -Detail "Missing cross-lane consistency checks: $($missingCheckIds -join ', ')")) | Out-Null
$items.Add((New-ValidationItem -Id "cross-lane-consistency-no-blockers" -Passed ($failedConsistencyBlockers.Count -eq 0 -and [int](Get-PropertyOrDefault -Object $record -Name "failedConsistencyBlockerCount" -DefaultValue 0) -eq 0) -Severity "blocker" -Detail "Cross-lane consistency checks must not contain blocker failures.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-count-consistent" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "laneCount" -DefaultValue 0) -eq $lanes.Count) -Severity "blocker" -Detail "laneCount must match closureLanes count.")) | Out-Null
$items.Add((New-ValidationItem -Id "lanes-safe" -Passed $allLanesSafe -Severity "blocker" -Detail "Every lane must keep publish/token/close/proof flags false and include ownerAction plus boundary.")) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-readiness-lane-metadata-bridged" -Passed $preReleaseLaneCarriesMatrixMetadata -Severity "blocker" -Detail "Final bridge must carry the pre-release readiness matrix requiredEvidence and validatorPath into a closure lane.")) | Out-Null
$items.Add((New-ValidationItem -Id "pre-release-readiness-summary-bridged" -Passed $preReleaseSummaryPresent -Severity "blocker" -Detail "Final bridge must expose pre-release readiness matrix state, lane metadata readiness, and promote flag safety in closureProofSourceSummary.")) | Out-Null
$items.Add((New-ValidationItem -Id "public-package-download-owner-field-surface-bridged" -Passed $publicPackageDownloadOwnerExecutionLaneFieldSurface -Severity "blocker" -Detail "Public package download owner execution lane must carry required/blocked owner fields, rejected substitute count, and source readiness signal count.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-user-verification-owner-field-surface-bridged" -Passed $postPublishUserVerificationLaneFieldSurface -Severity "blocker" -Detail "Post-publish user verification lane must carry required/blocked owner fields, rejected substitute count, and source readiness signal count.")) | Out-Null
$items.Add((New-ValidationItem -Id "top-level-owner-field-surface-consistent" -Passed ($topLevelOwnerFieldSurfaceConsistent -and $requiredOwnerFieldCountFromLanes -ge 39 -and $blockedRequiredOwnerFieldCountFromLanes -eq $requiredOwnerFieldCountFromLanes -and $rejectedSubstituteCountFromLanes -ge 16 -and $sourceReadinessSignalCountFromLanes -ge 15) -Severity "blocker" -Detail "Final bridge top-level owner field surface totals must match lane totals and remain fully blocked until real owner proof is supplied.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-side-effects" -Passed ([bool](Get-PropertyOrDefault -Object $record -Name "notExecutedByAutomation" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPackageConsumerRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Bridge must not publish, use tokens, promote proof, or close release issue.")) | Out-Null
$items.Add((New-ValidationItem -Id "all-close-lanes-ready" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedLaneCount" -DefaultValue 0) -eq 0 -and [string](Get-PropertyOrDefault -Object $record -Name "bridgeState" -DefaultValue "") -eq "final-public-release-closure-bridge-ready-for-owner-close-review") -Severity "action-required" -Detail "Bridge remains blocked until all owner authorization, owner publish execution result, owner download execution guidance, public download, external smoke, post-publish proof, post-publish user verification, close decision, and strict dashboard lanes are ready.")) | Out-Null

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
  crossLaneConsistencyCheckCount = $crossLaneConsistencyChecks.Count
  failedConsistencyBlockerCount = $failedConsistencyBlockers.Count
  failedConsistencyActionRequiredCount = $failedConsistencyActionRequired.Count
  forbiddenSubstituteFindingCount = [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteFindingCount" -DefaultValue 0)
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
| requiredOwnerFieldCount | ``$($validation.requiredOwnerFieldCount)`` |
| blockedRequiredOwnerFieldCount | ``$($validation.blockedRequiredOwnerFieldCount)`` |
| readyOwnerFieldCount | ``$($validation.readyOwnerFieldCount)`` |
| rejectedSubstituteCount | ``$($validation.rejectedSubstituteCount)`` |
| sourceReadinessSignalCount | ``$($validation.sourceReadinessSignalCount)`` |
| ownerFieldSurfaceLaneCount | ``$($validation.ownerFieldSurfaceLaneCount)`` |
| blockedOwnerFieldSurfaceLaneCount | ``$($validation.blockedOwnerFieldSurfaceLaneCount)`` |
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
