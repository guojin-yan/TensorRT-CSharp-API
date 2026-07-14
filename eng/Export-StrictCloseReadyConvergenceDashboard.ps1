[CmdletBinding()]
param(
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

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Sum-IntProperty {
  param([AllowNull()][object[]]$Items, [string]$Name)
  $total = 0
  foreach ($item in @($Items)) {
    $total += [int](Get-PropertyOrDefault -Object $item -Name $Name -DefaultValue 0)
  }

  return $total
}

function New-CloseLane {
  param(
    [string]$Id,
    [string]$State,
    [string]$RequiredState,
    [string[]]$MissingOwnerInput,
    [string[]]$MissingRealProof,
    [string]$OwnerNextAction,
    [string]$Validator,
    [string]$SourceArtifact,
    [int]$RequiredOwnerFieldCount = 0,
    [int]$BlockedRequiredOwnerFieldCount = 0,
    [int]$ReadyOwnerFieldCount = 0,
    [int]$RejectedSubstituteCount = 0,
    [int]$SourceReadinessSignalCount = 0
  )

  $ready = [string]::Equals($State, $RequiredState, [StringComparison]::OrdinalIgnoreCase) -and @($MissingOwnerInput).Count -eq 0 -and @($MissingRealProof).Count -eq 0
  [pscustomobject]@{
    laneId = $Id
    state = $State
    requiredState = $RequiredState
    ready = $ready
    closeReadinessState = if ($ready) { "ready" } else { "blocked-strict-close-ready-owner-action-required" }
    missingOwnerInput = @($MissingOwnerInput)
    missingRealProof = @($MissingRealProof)
    ownerNextAction = $OwnerNextAction
    validator = $Validator
    sourceArtifact = $SourceArtifact
    requiredOwnerFieldCount = $RequiredOwnerFieldCount
    blockedRequiredOwnerFieldCount = $BlockedRequiredOwnerFieldCount
    readyOwnerFieldCount = $ReadyOwnerFieldCount
    rejectedSubstituteCount = $RejectedSubstituteCount
    sourceReadinessSignalCount = $SourceReadinessSignalCount
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
  }
}

function Get-LaneById {
  param([AllowNull()][object[]]$Lanes, [string]$Id)
  return @($Lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq $Id } | Select-Object -First 1)[0]
}

function New-RemoteProofCloseLane {
  param(
    [string]$Id,
    [AllowNull()][object]$RemoteLane,
    [string]$OwnerNextAction,
    [string]$Validator,
    [string]$MissingRealProof,
    [int]$RequiredOwnerFieldCount = 0,
    [int]$BlockedRequiredOwnerFieldCount = 0,
    [int]$ReadyOwnerFieldCount = 0,
    [int]$RejectedSubstituteCount = 0,
    [int]$SourceReadinessSignalCount = 0
  )

  $state = [string](Get-PropertyOrDefault -Object $RemoteLane -Name "state" -DefaultValue "missing-remote-proof-lane")
  $ready = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "ready" -DefaultValue $false)
  $stateReady = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "stateReady" -DefaultValue $false)
  $requireProofReady = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "requireProofReady" -DefaultValue $false)
  $proofReadyProperty = [string](Get-PropertyOrDefault -Object $RemoteLane -Name "proofReadyProperty" -DefaultValue "")
  $proofReady = [bool](Get-PropertyOrDefault -Object $RemoteLane -Name "proofReady" -DefaultValue $false)
  $requiredEvidence = [string](Get-PropertyOrDefault -Object $RemoteLane -Name "requiredEvidence" -DefaultValue $MissingRealProof)
  $sourceArtifact = [string](Get-PropertyOrDefault -Object $RemoteLane -Name "artifact" -DefaultValue "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json")
  $missingRealProof = if ($ready) { @() } else { @($MissingRealProof) }

  [pscustomobject]@{
    laneId = $Id
    state = $state
    requiredState = "remote-proof-lane-ready"
    ready = $ready
    closeReadinessState = if ($ready) { "ready" } else { "blocked-strict-close-ready-owner-action-required" }
    missingOwnerInput = if ($ready) { @() } else { @("owner-supplied real remote/public proof input") }
    missingRealProof = $missingRealProof
    ownerNextAction = $OwnerNextAction
    validator = $Validator
    sourceArtifact = $sourceArtifact
    remoteProofGateLaneId = $Id
    remoteGateStateReady = $stateReady
    requireProofReady = $requireProofReady
    proofReadyProperty = $proofReadyProperty
    proofReady = $proofReady
    requiredEvidence = $requiredEvidence
    requiredOwnerFieldCount = $RequiredOwnerFieldCount
    blockedRequiredOwnerFieldCount = $BlockedRequiredOwnerFieldCount
    readyOwnerFieldCount = $ReadyOwnerFieldCount
    rejectedSubstituteCount = $RejectedSubstituteCount
    sourceReadinessSignalCount = $SourceReadinessSignalCount
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    isGitHubActionsProof = $false
    blocksStrictClose = -not $ready
  }
}

$finalBlockerDashboardValidation = Read-JsonOrNull "artifacts\final-release\final-release-close-blocker-dashboard-validation.json"
$publicPublishOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-owner-input-validation.json"
$publicPublishImportValidation = Read-JsonOrNull "artifacts\final-release\public-publish-result-import-validation.json"
$postPublishCleanConsumerProofContractValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-contract-validation.json"
$cleanConsumerConvergenceValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-result-convergence-validation.json"
$releaseIssueCloseOwnerDecisionInputValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input-validation.json"
$finalOwnerDecisionValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-final-owner-decision-audit-validation.json"
$closeRecordValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$classificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$remoteProofBackfillGate = Read-JsonOrNull "artifacts\final-release\remote-ci-and-public-publish-proof-backfill-gate.json"
$remoteProofBackfillGateValidation = Read-JsonOrNull "artifacts\final-release\remote-ci-and-public-publish-proof-backfill-gate-validation.json"
$publicReleaseOwnerExecutionPackageValidation = Read-JsonOrNull "artifacts\final-release\public-release-owner-execution-package-validation.json"
$publicPackageDownloadProofOwnerExecutionPackValidation = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-owner-execution-pack-validation.json"
$postPublishUserVerificationPackValidation = Read-JsonOrNull "artifacts\final-release\post-publish-user-verification-pack-validation.json"

$finalBlockerState = [string](Get-PropertyOrDefault -Object $finalBlockerDashboardValidation -Name "validationState" -DefaultValue "missing-final-release-close-blocker-dashboard-validation")
$publicPublishOwnerInputState = [string](Get-PropertyOrDefault -Object $publicPublishOwnerInputValidation -Name "validationState" -DefaultValue "missing-public-publish-result-owner-input-validation")
$publicPublishState = [string](Get-PropertyOrDefault -Object $publicPublishImportValidation -Name "validationState" -DefaultValue "missing-public-publish-result-import-validation")
$postPublishCleanConsumerProofContractState = [string](Get-PropertyOrDefault -Object $postPublishCleanConsumerProofContractValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-contract-validation")
$cleanConsumerState = [string](Get-PropertyOrDefault -Object $cleanConsumerConvergenceValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-result-convergence-validation")
$releaseIssueCloseOwnerDecisionInputState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseOwnerDecisionInputValidation -Name "validationState" -DefaultValue "missing-release-issue-close-owner-decision-input-validation")
$finalOwnerState = [string](Get-PropertyOrDefault -Object $finalOwnerDecisionValidation -Name "validationState" -DefaultValue "missing-release-issue-close-final-owner-decision-audit-validation")
$closeRecordState = [string](Get-PropertyOrDefault -Object $closeRecordValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
$classificationState = [string](Get-PropertyOrDefault -Object $classificationAudit -Name "auditState" -DefaultValue "missing-release-evidence-classification-audit")
$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$remoteProofBackfillGateState = [string](Get-PropertyOrDefault -Object $remoteProofBackfillGateValidation -Name "validationState" -DefaultValue "missing-remote-ci-and-public-publish-proof-backfill-gate-validation")
$remoteProofLanes = @((Get-PropertyOrDefault -Object $remoteProofBackfillGate -Name "lanes" -DefaultValue @()))
$publicReleaseOwnerExecutionRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $publicReleaseOwnerExecutionPackageValidation -Name "requiredOwnerFieldCount" -DefaultValue 0)
$publicReleaseOwnerExecutionBlockedRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $publicReleaseOwnerExecutionPackageValidation -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0)
$publicReleaseOwnerExecutionReadyOwnerFieldCount = [Math]::Max(0, $publicReleaseOwnerExecutionRequiredOwnerFieldCount - $publicReleaseOwnerExecutionBlockedRequiredOwnerFieldCount)
$publicReleaseOwnerExecutionRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $publicReleaseOwnerExecutionPackageValidation -Name "rejectedSubstituteCount" -DefaultValue 0)
$publicReleaseOwnerExecutionSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $publicReleaseOwnerExecutionPackageValidation -Name "sourceReadinessSignalCount" -DefaultValue 0)
$publicPackageDownloadOwnerExecutionRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $publicPackageDownloadProofOwnerExecutionPackValidation -Name "requiredOwnerFieldCount" -DefaultValue 0)
$publicPackageDownloadOwnerExecutionBlockedRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $publicPackageDownloadProofOwnerExecutionPackValidation -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0)
$publicPackageDownloadOwnerExecutionReadyOwnerFieldCount = [Math]::Max(0, $publicPackageDownloadOwnerExecutionRequiredOwnerFieldCount - $publicPackageDownloadOwnerExecutionBlockedRequiredOwnerFieldCount)
$publicPackageDownloadOwnerExecutionRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $publicPackageDownloadProofOwnerExecutionPackValidation -Name "rejectedSubstituteCount" -DefaultValue 0)
$publicPackageDownloadOwnerExecutionSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $publicPackageDownloadProofOwnerExecutionPackValidation -Name "sourceReadinessSignalCount" -DefaultValue 0)
$postPublishUserVerificationRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $postPublishUserVerificationPackValidation -Name "requiredOwnerFieldCount" -DefaultValue 0)
$postPublishUserVerificationBlockedRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $postPublishUserVerificationPackValidation -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0)
$postPublishUserVerificationReadyOwnerFieldCount = [Math]::Max(0, $postPublishUserVerificationRequiredOwnerFieldCount - $postPublishUserVerificationBlockedRequiredOwnerFieldCount)
$postPublishUserVerificationRejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $postPublishUserVerificationPackValidation -Name "rejectedSubstituteCount" -DefaultValue 0)
$postPublishUserVerificationSourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $postPublishUserVerificationPackValidation -Name "sourceReadinessSignalCount" -DefaultValue 0)

$lanes = @(
  New-RemoteProofCloseLane -Id "github-actions-run-proof" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "github-actions-run-proof") -MissingRealProof "real GitHub Actions run URL, run id, head SHA, conclusion, log hash, and artifact hash" -OwnerNextAction "Owner imports real GitHub Actions run evidence; queued workflow, missing runner, local test, or dashboard output remain blocked." -Validator "Test-RemoteCiAndPublicPublishProofBackfillGate.ps1 -Strict"
  New-RemoteProofCloseLane -Id "owner-public-publish-result" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "owner-public-publish-result") -MissingRealProof "owner public publish result with public NuGet/GitHub package URLs, hashes, transcript hashes, reviewer, and authorization linkage" -OwnerNextAction "Owner supplies real public publish result after actual package publication; dry run and command plan are not proof." -Validator "Test-RemoteCiAndPublicPublishProofBackfillGate.ps1 -Strict" -RequiredOwnerFieldCount $publicReleaseOwnerExecutionRequiredOwnerFieldCount -BlockedRequiredOwnerFieldCount $publicReleaseOwnerExecutionBlockedRequiredOwnerFieldCount -ReadyOwnerFieldCount $publicReleaseOwnerExecutionReadyOwnerFieldCount -RejectedSubstituteCount $publicReleaseOwnerExecutionRejectedSubstituteCount -SourceReadinessSignalCount $publicReleaseOwnerExecutionSourceReadinessSignalCount
  New-RemoteProofCloseLane -Id "public-package-download-proof" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "public-package-download-proof") -MissingRealProof "public package download URL/source, package identity, SHA256, timestamp, and non-local source proof" -OwnerNextAction "Owner downloads the public package from the public source and imports SHA256 evidence." -Validator "Test-RemoteCiAndPublicPublishProofBackfillGate.ps1 -Strict" -RequiredOwnerFieldCount $publicPackageDownloadOwnerExecutionRequiredOwnerFieldCount -BlockedRequiredOwnerFieldCount $publicPackageDownloadOwnerExecutionBlockedRequiredOwnerFieldCount -ReadyOwnerFieldCount $publicPackageDownloadOwnerExecutionReadyOwnerFieldCount -RejectedSubstituteCount $publicPackageDownloadOwnerExecutionRejectedSubstituteCount -SourceReadinessSignalCount $publicPackageDownloadOwnerExecutionSourceReadinessSignalCount
  New-RemoteProofCloseLane -Id "post-publish-clean-consumer-proof" -RemoteLane (Get-LaneById -Lanes $remoteProofLanes -Id "post-publish-clean-consumer-proof") -MissingRealProof "post-publication repository-external clean consumer restore/build/smoke proof with proofCandidateReady=true" -OwnerNextAction "Owner provides repository-external clean consumer proof from public packages; validation-ready alone must remain blocked until proofCandidateReady is true." -Validator "Test-RemoteCiAndPublicPublishProofBackfillGate.ps1 -Strict" -RequiredOwnerFieldCount $postPublishUserVerificationRequiredOwnerFieldCount -BlockedRequiredOwnerFieldCount $postPublishUserVerificationBlockedRequiredOwnerFieldCount -ReadyOwnerFieldCount $postPublishUserVerificationReadyOwnerFieldCount -RejectedSubstituteCount $postPublishUserVerificationRejectedSubstituteCount -SourceReadinessSignalCount $postPublishUserVerificationSourceReadinessSignalCount
  New-CloseLane -Id "final-release-close-blocker-dashboard" -State $finalBlockerState -RequiredState "final-release-close-blocker-dashboard-ready" -MissingOwnerInput @("remaining close blockers resolved by Owner") -MissingRealProof @("public package proof", "post-publish proof") -OwnerNextAction "Owner resolves every final release close blocker after real public publish and clean consumer proof." -Validator "Test-FinalReleaseCloseBlockerDashboard.ps1 -Strict" -SourceArtifact "artifacts/final-release/final-release-close-blocker-dashboard-validation.json"
  New-CloseLane -Id "public-publish-result-owner-input" -State $publicPublishOwnerInputState -RequiredState "public-publish-result-owner-input-ready" -MissingOwnerInput @("owner-filled public publish result input") -MissingRealProof @("NuGet/GitHub public URLs, download hashes, release transcript, rollback review, final close decision fields") -OwnerNextAction "Owner fills real public publish result metadata; templates, dry runs, local feeds, and command plans remain non-proof." -Validator "Test-PublicPublishResultOwnerInput.ps1 -Strict" -SourceArtifact "artifacts/final-release/public-publish-result-owner-input-validation.json"
  New-CloseLane -Id "public-publish-result-import" -State $publicPublishState -RequiredState "public-publish-result-import-ready" -MissingOwnerInput @("public publish result owner input") -MissingRealProof @("public package URL/hash/timestamp/transcript") -OwnerNextAction "Owner imports real public publish metadata and runs the strict import validator." -Validator "Test-PublicPublishResultImport.ps1 -Strict" -SourceArtifact "artifacts/final-release/public-publish-result-import-validation.json"
  New-CloseLane -Id "post-publish-clean-consumer-proof-record-contract" -State $postPublishCleanConsumerProofContractState -RequiredState "post-publish-clean-consumer-proof-record-contract-ready" -MissingOwnerInput @("repository-external clean consumer proof record") -MissingRealProof @("restore/build/smoke/stdout/stderr logs, SHA256 values, host metadata, forbidden substitute counts") -OwnerNextAction "Owner supplies post-publish clean consumer proof from a repository-external consumer using only public package sources." -Validator "Test-PostPublishCleanConsumerProofRecordContract.ps1 -Strict" -SourceArtifact "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json"
  New-CloseLane -Id "post-publish-clean-consumer-result-convergence" -State $cleanConsumerState -RequiredState "post-publish-clean-consumer-result-convergence-ready" -MissingOwnerInput @("clean consumer logs and package source metadata") -MissingRealProof @("public-source clean consumer proof") -OwnerNextAction "Owner completes post-publish clean consumer proof and convergence validation." -Validator "Test-PostPublishCleanConsumerResultConvergence.ps1 -Strict" -SourceArtifact "artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json"
  New-CloseLane -Id "release-issue-close-owner-decision-input" -State $releaseIssueCloseOwnerDecisionInputState -RequiredState "release-issue-close-owner-decision-input-ready" -MissingOwnerInput @("release issue close owner decision input") -MissingRealProof @("approved public package proof hash, approved post-publish proof hash, evidence bundle hash, rollback plan, close issue URL") -OwnerNextAction "Owner fills release issue close decision only after public publish proof and post-publish clean consumer proof pass strict gates." -Validator "Test-ReleaseIssueCloseOwnerDecisionInput.ps1 -Strict" -SourceArtifact "artifacts/final-release/release-issue-close-owner-decision-input-validation.json"
  New-CloseLane -Id "release-issue-close-final-owner-decision-audit" -State $finalOwnerState -RequiredState "release-issue-close-final-owner-decision-ready" -MissingOwnerInput @("final close owner decision") -MissingRealProof @("all public proof and close gates ready") -OwnerNextAction "Owner records final close decision only after all strict gates are ready." -Validator "Test-ReleaseIssueCloseFinalOwnerDecisionAudit.ps1 -Strict" -SourceArtifact "artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json"
  New-CloseLane -Id "release-issue-close-record-validation" -State $closeRecordState -RequiredState "ready-for-owner-release-issue-close" -MissingOwnerInput @("owner-filled release issue close record") -MissingRealProof @("real post-publish proof and evidence bundle hash") -OwnerNextAction "Owner runs Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady only when all real gates are ready." -Validator "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady" -SourceArtifact "artifacts/final-release/release-issue-close-record-validation.json"
  New-CloseLane -Id "release-evidence-classification-audit" -State $classificationState -RequiredState "classification-audit-passed-non-proof-boundaries-intact" -MissingOwnerInput @() -MissingRealProof @() -OwnerNextAction "Keep non-proof boundaries intact while real proof remains missing." -Validator "Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" -SourceArtifact "artifacts/final-release/release-evidence-classification-audit.json"
)

$blocked = @($lanes | Where-Object { -not [bool]$_.ready })
$ready = @($lanes | Where-Object { [bool]$_.ready })
$ownerFieldSurfaceLanes = @($lanes | Where-Object { [int](Get-PropertyOrDefault -Object $_ -Name "requiredOwnerFieldCount" -DefaultValue 0) -gt 0 })
$blockedOwnerFieldSurfaceLanes = @($ownerFieldSurfaceLanes | Where-Object { [int](Get-PropertyOrDefault -Object $_ -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0) -gt 0 })
$requiredOwnerFieldCount = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "requiredOwnerFieldCount"
$blockedRequiredOwnerFieldCount = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "blockedRequiredOwnerFieldCount"
$readyOwnerFieldCount = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "readyOwnerFieldCount"
$rejectedSubstituteCount = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "rejectedSubstituteCount"
$sourceReadinessSignalCount = Sum-IntProperty -Items $ownerFieldSurfaceLanes -Name "sourceReadinessSignalCount"

$record = [pscustomobject]@{
  recordKind = "strict-close-ready-convergence-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  dashboardState = if ($blocked.Count -eq 0) { "strict-close-ready-convergence-ready" } else { "blocked-strict-close-ready-owner-action-required" }
  laneCount = $lanes.Count
  blockedLaneCount = $blocked.Count
  readyLaneCount = $ready.Count
  releaseEvidenceBundleState = $releaseEvidenceState
  remoteCiAndPublicPublishProofBackfillGateState = $remoteProofBackfillGateState
  requiredOwnerFieldCount = $requiredOwnerFieldCount
  blockedRequiredOwnerFieldCount = $blockedRequiredOwnerFieldCount
  readyOwnerFieldCount = $readyOwnerFieldCount
  rejectedSubstituteCount = $rejectedSubstituteCount
  sourceReadinessSignalCount = $sourceReadinessSignalCount
  ownerFieldSurfaceLaneCount = $ownerFieldSurfaceLanes.Count
  blockedOwnerFieldSurfaceLaneCount = $blockedOwnerFieldSurfaceLanes.Count
  remoteProofRequiredLaneIds = @("github-actions-run-proof", "owner-public-publish-result", "public-package-download-proof", "post-publish-clean-consumer-proof")
  closeReadinessLanes = $lanes
  sourceArtifacts = @(
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.json",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate.md",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.json",
    "artifacts/final-release/remote-ci-and-public-publish-proof-backfill-gate-validation.md",
    "artifacts/final-release/final-release-close-blocker-dashboard-validation.json",
    "artifacts/final-release/public-publish-result-owner-input-validation.json",
    "artifacts/final-release/public-publish-result-import-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json",
    "artifacts/final-release/release-issue-close-owner-decision-input-validation.json",
    "artifacts/final-release/release-issue-close-final-owner-decision-audit-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json",
    "artifacts/final-release/release-evidence-classification-audit.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/public-release-owner-execution-package-validation.json",
    "artifacts/final-release/public-package-download-proof-owner-execution-pack-validation.json",
    "artifacts/final-release/post-publish-user-verification-pack-validation.json"
  )
  notExecutedByAutomation = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "StrictCloseReady convergence dashboard is owner-action status aggregation only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "strict-close-ready-convergence-dashboard.json"
$markdownPath = Join-Path $artifactRoot "strict-close-ready-convergence-dashboard.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $record.closeReadinessLanes | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.laneId) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` | ``$($_.requiredOwnerFieldCount)`` | ``$($_.blockedRequiredOwnerFieldCount)`` | ``$($_.rejectedSubstituteCount)`` | ``$($_.sourceReadinessSignalCount)`` | $(ConvertTo-MarkdownCell ($_.missingOwnerInput -join ', ')) | $(ConvertTo-MarkdownCell ($_.missingRealProof -join ', ')) | $(ConvertTo-MarkdownCell $_.validator) |"
}

$markdown = @"
# Strict Close Ready Convergence Dashboard

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| dashboardState | ``$($record.dashboardState)`` |
| laneCount | ``$($record.laneCount)`` |
| blockedLaneCount | ``$($record.blockedLaneCount)`` |
| readyLaneCount | ``$($record.readyLaneCount)`` |
| releaseEvidenceBundleState | ``$($record.releaseEvidenceBundleState)`` |
| requiredOwnerFieldCount | ``$($record.requiredOwnerFieldCount)`` |
| blockedRequiredOwnerFieldCount | ``$($record.blockedRequiredOwnerFieldCount)`` |
| readyOwnerFieldCount | ``$($record.readyOwnerFieldCount)`` |
| rejectedSubstituteCount | ``$($record.rejectedSubstituteCount)`` |
| sourceReadinessSignalCount | ``$($record.sourceReadinessSignalCount)`` |
| ownerFieldSurfaceLaneCount | ``$($record.ownerFieldSurfaceLaneCount)`` |
| blockedOwnerFieldSurfaceLaneCount | ``$($record.blockedOwnerFieldSurfaceLaneCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Close Readiness Lanes

| Lane | Current State | Required State | Ready | Owner Fields | Blocked Owner Fields | Rejected Substitutes | Source Signals | Missing Owner Input | Missing Real Proof | Validator |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
$($rows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Strict close ready convergence dashboard written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "DashboardState=$($record.dashboardState) Lanes=$($record.laneCount) Blocked=$($record.blockedLaneCount) Ready=$($record.readyLaneCount)"
