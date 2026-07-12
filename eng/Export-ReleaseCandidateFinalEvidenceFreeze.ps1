[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
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

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-RemainingBlocker {
  param(
    [string]$GapId,
    [string]$ProofClass,
    [string]$CurrentState,
    [string]$Validator,
    [string]$RequiredRealEvidence,
    [string]$OwnerAction,
    [string]$FreezeBoundary
  )

  [pscustomobject]@{
    gapId = $GapId
    proofClass = $ProofClass
    passed = $false
    currentState = $CurrentState
    validator = $Validator
    requiredRealEvidence = $RequiredRealEvidence
    ownerAction = $OwnerAction
    freezeBoundary = $FreezeBoundary
    nonSubstitutes = $script:NonSubstituteProofKinds
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$releasePromotionIssue = Read-JsonOrNull "artifacts\final-release\release-promotion-issue-record.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseCloseGapDashboard = Read-JsonOrNull "artifacts\final-release\release-close-gap-dashboard.json"
$compatibleHostProofExecutionPack = Read-JsonOrNull "artifacts\final-release\compatible-host-proof-execution-pack.json"
$realModelAndPackageProofInputPackage = Read-JsonOrNull "artifacts\final-release\real-model-and-package-proof-input-package.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$releaseProofReadinessSnapshot = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$ownerProofInputReadiness = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness.json"
$ownerProofInputReadinessValidation = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness-validation.json"
$staleReleaseClaimsAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$ownerRealEvidenceInputImport = Read-JsonOrNull "artifacts\final-release\owner-real-evidence-input-import.json"
$ownerRealEvidenceInputValidation = Read-JsonOrNull "artifacts\final-release\owner-real-evidence-input-validation.json"
$ownerRealEvidenceAcceptanceDashboard = Read-JsonOrNull "artifacts\final-release\owner-real-evidence-acceptance-dashboard.json"
$ownerOnlyPublishExecutionCandidate = Read-JsonOrNull "artifacts\final-release\owner-only-publish-execution-candidate.json"
$ownerOnlyPublishExecutionCandidateValidation = Read-JsonOrNull "artifacts\final-release\owner-only-publish-execution-candidate-validation.json"
$finalPublishActionRequiredEvidenceMap = Read-JsonOrNull "artifacts\final-release\final-publish-action-required-evidence-map.json"

$evidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$promotionState = [string](Get-PropertyOrDefault -Object $releasePromotionIssue -Name "promotionState" -DefaultValue "missing-release-promotion-issue-record")
$preflightState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$preflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)
$preflightCanClose = [bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)
$dashboardState = [string](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "dashboardState" -DefaultValue "missing-release-close-gap-dashboard")
$dashboardGapCount = [int](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "gapCount" -DefaultValue -1)
$executionPackState = [string](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "packageState" -DefaultValue "missing-compatible-host-proof-execution-pack")
$executionPackBlockerCount = [int](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "blockerCount" -DefaultValue -1)
$inputPackageState = [string](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "packageState" -DefaultValue "missing-real-model-and-package-proof-input-package")
$ownerPackageState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$oneScreenReleaseHoldChecklist = @(Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "oneScreenReleaseHoldChecklist" -DefaultValue @(Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "oneScreenReleaseHoldChecklist" -DefaultValue @()))
$oneScreenReleaseHoldChecklistCount = if ($oneScreenReleaseHoldChecklist.Count -gt 0) { $oneScreenReleaseHoldChecklist.Count } else { [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "oneScreenReleaseHoldChecklistCount" -DefaultValue 0) }
$releaseProofReadinessSnapshotState = [string](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseProofReadinessSnapshotState" -DefaultValue "missing-release-proof-readiness-snapshot")))
$releaseProofReadinessSnapshotItemCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseProofReadinessSnapshotItemCount" -DefaultValue 0)))
$releaseProofReadinessSnapshotReadyProofItemCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readyProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseProofReadinessSnapshotReadyProofItemCount" -DefaultValue 0)))
$releaseProofReadinessSnapshotBlockedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "blockedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseProofReadinessSnapshotBlockedProofItemCount" -DefaultValue 0)))
$releaseProofReadinessSnapshotPerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseProofReadinessSnapshotPerformsPublish" -DefaultValue $false)))
$releaseProofReadinessSnapshotCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "canPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseProofReadinessSnapshotCanPublishPublicly" -DefaultValue $false)))
$releaseProofReadinessSnapshotCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseProofReadinessSnapshotCanCloseReleaseIssue" -DefaultValue $false)))
$ownerProofInputReadinessState = [string](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "readinessState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessState" -DefaultValue "missing-owner-proof-input-readiness")))
$ownerProofInputReadinessContractCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "contractCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessContractCount" -DefaultValue 0)))
$ownerProofInputReadinessReadyContractCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "readyContractCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessReadyContractCount" -DefaultValue 0)))
$ownerProofInputReadinessBlockedContractCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "blockedContractCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessBlockedContractCount" -DefaultValue 0)))
$ownerProofInputReadinessPerformsPublish = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessPerformsPublish" -DefaultValue $false)))
$ownerProofInputReadinessCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "canPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessCanPublishPublicly" -DefaultValue $false)))
$ownerProofInputReadinessCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadiness -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessCanCloseReleaseIssue" -DefaultValue $false)))
$ownerProofInputReadinessValidationState = [string](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "validationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessValidationState" -DefaultValue "missing-owner-proof-input-readiness-validation")))
$ownerProofInputReadinessIsValid = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "isValidOwnerProofInputReadiness" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessIsValid" -DefaultValue $false)))
$ownerProofInputReadinessValidationFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "failedBlockerCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessValidationFailedBlockerCount" -DefaultValue -1)))
$ownerProofInputReadinessValidationPerformsPublish = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessValidationPerformsPublish" -DefaultValue $false)))
$ownerProofInputReadinessValidationCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "canPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessValidationCanPublishPublicly" -DefaultValue $false)))
$ownerProofInputReadinessValidationCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $ownerProofInputReadinessValidation -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerProofInputReadinessValidationCanCloseReleaseIssue" -DefaultValue $false)))
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleReleaseClaimsAudit -Name "findingCount" -DefaultValue -1)
$ownerRealEvidenceInputImportState = [string](Get-PropertyOrDefault -Object $ownerRealEvidenceInputImport -Name "importState" -DefaultValue "missing-owner-real-evidence-input-import")
$ownerRealEvidenceInputValidationState = [string](Get-PropertyOrDefault -Object $ownerRealEvidenceInputValidation -Name "validationState" -DefaultValue "missing-owner-real-evidence-input-validation")
$ownerRealEvidenceInputAcceptedLaneCount = [int](Get-PropertyOrDefault -Object $ownerRealEvidenceInputImport -Name "acceptedLaneCount" -DefaultValue 0)
$ownerRealEvidenceInputBlockedLaneCount = [int](Get-PropertyOrDefault -Object $ownerRealEvidenceInputImport -Name "blockedLaneCount" -DefaultValue 6)
$ownerRealEvidenceInputValidationFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerRealEvidenceInputValidation -Name "failedBlockerCount" -DefaultValue -1)
$ownerRealEvidenceInputValidationFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $ownerRealEvidenceInputValidation -Name "failedActionRequiredCount" -DefaultValue 6)
$ownerRealEvidenceAcceptanceDashboardState = [string](Get-PropertyOrDefault -Object $ownerRealEvidenceAcceptanceDashboard -Name "dashboardState" -DefaultValue "missing-owner-real-evidence-acceptance-dashboard")
$ownerRealEvidenceAcceptanceCanPrepareOwnerOnlyPublishCandidate = [bool](Get-PropertyOrDefault -Object $ownerRealEvidenceAcceptanceDashboard -Name "canPrepareOwnerOnlyPublishCandidate" -DefaultValue $false)
$ownerOnlyPublishExecutionCandidateState = [string](Get-PropertyOrDefault -Object $ownerOnlyPublishExecutionCandidate -Name "candidateState" -DefaultValue "missing-owner-only-publish-execution-candidate")
$ownerOnlyPublishExecutionCandidateValidationState = [string](Get-PropertyOrDefault -Object $ownerOnlyPublishExecutionCandidateValidation -Name "validationState" -DefaultValue "missing-owner-only-publish-execution-candidate-validation")
$ownerOnlyPublishExecutionCandidateValidationFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerOnlyPublishExecutionCandidateValidation -Name "failedBlockerCount" -DefaultValue -1)
$finalPublishActionRequiredEvidenceMapState = [string](Get-PropertyOrDefault -Object $finalPublishActionRequiredEvidenceMap -Name "mapState" -DefaultValue "missing-final-publish-action-required-evidence-map")
$finalPublishActionRequiredEvidenceMapActionRequiredCount = [int](Get-PropertyOrDefault -Object $finalPublishActionRequiredEvidenceMap -Name "actionRequiredCount" -DefaultValue 6)

$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofState" -DefaultValue "missing-external-runtime-proof-validation")
$externalCanPromote = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPromoteRuntimeProof" -DefaultValue $false)
$linuxProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "sampleRunEvidenceValidationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleCanPromote = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "sampleRunEvidenceCanPromoteRealModelRuntime" -DefaultValue $false)
$postPublishState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "isPostPublishVerificationProof" -DefaultValue $false)
$ownerApprovalState = [string](Get-PropertyOrDefault -Object $releasePromotionIssue -Name "ownerApprovalInputValidationStatus" -DefaultValue "missing-owner-authorization")

$script:NonSubstituteProofKinds = @(
  "helper",
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "DependencyProbe",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "build-only",
  "parse-only",
  "sidecar-only",
  "Windows handoff for Linux proof",
  "owner-action-required without validator pass"
)

$validatorCommands = @(
  "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
  "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
  "Test-LinuxRunnerEvidenceRecord.ps1",
  "Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog",
  "Test-ReleaseOwnerApprovalInput.ps1",
  "Test-OwnerAuthorizedPublishCommandPlan.ps1",
  "Import-OwnerRealEvidenceInput.ps1",
  "Test-OwnerRealEvidenceInput.ps1 -Strict",
  "Export-OwnerRealEvidenceAcceptanceDashboard.ps1",
  "Test-OwnerOnlyPublishExecutionCandidate.ps1 -Strict",
  "Test-StaleReleaseClaims.ps1"
)

$remainingBlockers = @(
  New-RemainingBlocker `
    -GapId "owner-authorization" `
    -ProofClass "owner-authorization" `
    -CurrentState "ownerApprovalInputValidationStatus=$ownerApprovalState; ownerReleaseExecutionPackageState=$ownerPackageState; promotionState=$promotionState" `
    -Validator "Test-ReleaseOwnerApprovalInput.ps1 + Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -RequiredRealEvidence "Explicit owner approval, channel choice, NVIDIA redistribution disposition, and manual publish command materialization." `
    -OwnerAction "Owner must provide a non-template authorization record and keep publication commands manual." `
    -FreezeBoundary "Final evidence freeze does not authorize or execute publication."

  New-RemainingBlocker `
    -GapId "package-consumer-runtime" `
    -ProofClass "package-consumer-runtime" `
    -CurrentState "externalRuntimeProofState=$externalRuntimeProofState; canPromoteRuntimeProof=$externalCanPromote; runtimePackageKey=$RuntimePackageKey" `
    -Validator "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredRealEvidence "Clean external package consumer restore/build/smoke on compatible CUDA/TensorRT host with matching nupkg hashes, host metadata, stdout/stderr summaries, and log SHA256." `
    -OwnerAction "Run compatible-host package consumer smoke and validate the filled external-runtime-proof-record.json." `
    -FreezeBoundary "Template, draft, local feed, ProjectReference, DependencyProbe, build-only, and blocked-by-cuda-driver are not package-consumer-runtime proof."

  New-RemainingBlocker `
    -GapId "linux-runner-proof" `
    -ProofClass "linux-runner-proof" `
    -CurrentState "linuxRuntimePackageKey=$LinuxRuntimePackageKey; isRealLinuxRunnerProof=$linuxProof" `
    -Validator "Test-LinuxRunnerEvidenceRecord.ps1" `
    -RequiredRealEvidence "Real Linux x64 runner record with CUDA/TensorRT/cuDNN metadata, command log, runtime package key, and validator output." `
    -OwnerAction "Run the Linux runner on a real Linux x64 CUDA/TensorRT host and copy back the validated record." `
    -FreezeBoundary "Windows handoff, template-only, and dry-run-only materials are not Linux runner proof."

  New-RemainingBlocker `
    -GapId "real-model-runtime" `
    -ProofClass "real-model-runtime" `
    -CurrentState "sampleRunEvidenceValidationState=$sampleState; canPromoteRealModelRuntime=$sampleCanPromote; inputPackageState=$inputPackageState" `
    -Validator "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -RequiredRealEvidence "Classification/YoloVision models, labels, inputs, licenses, SHA256 values, TensorRtExec sidecar, sample runner log, and sample-run-evidence record. YoloVision scope: YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom and det/cls/seg/obb/pose/sem (det、cls、seg、obb、pose、sem)." `
    -OwnerAction "Provide real Classification and YoloVision assets, run samples, and validate sample-run-evidence." `
    -FreezeBoundary "Sample proof can promote only real-model-runtime; it cannot replace package-consumer-runtime."

  New-RemainingBlocker `
    -GapId "post-publish-verification" `
    -ProofClass "post-publish verification" `
    -CurrentState "postPublishVerificationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof" `
    -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredRealEvidence "Real channel package URL, downloaded nupkg SHA256, clean consumer root, no ProjectReference, native asset listing, dependency probe log, runtime smoke log, stdout/stderr summaries, and matching SHA256 values." `
    -OwnerAction "After owner-approved publication, validate a clean external consumer against the real channel package." `
    -FreezeBoundary "Post-publish proof cannot be collected before real publication; local feed, draft, helper scan, and collection package are not proof."
)

$releaseProofBlockerCount = @($remainingBlockers).Count

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-candidate-final-evidence-freeze"
  freezeState = "blocked-real-proof-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  requiresHumanOwner = $true
  requiresCompatibleHost = $true
  releaseEvidenceBundleState = $evidenceState
  releasePromotionIssueState = $promotionState
  releaseClosePreflightState = $preflightState
  releaseClosePreflightFailedItemCount = $releaseProofBlockerCount
  releaseClosePreflightCanCloseReleaseIssue = $preflightCanClose
  releaseCloseGapDashboardState = $dashboardState
  releaseCloseGapDashboardGapCount = $dashboardGapCount
  compatibleHostProofExecutionPackState = $executionPackState
  compatibleHostProofExecutionPackBlockerCount = $executionPackBlockerCount
  realModelAndPackageProofInputPackageState = $inputPackageState
  ownerReleaseExecutionPackageState = $ownerPackageState
  oneScreenReleaseHoldChecklist = @($oneScreenReleaseHoldChecklist)
  oneScreenReleaseHoldChecklistCount = $oneScreenReleaseHoldChecklistCount
  releaseProofReadinessSnapshotState = $releaseProofReadinessSnapshotState
  releaseProofReadinessSnapshotItemCount = $releaseProofReadinessSnapshotItemCount
  releaseProofReadinessSnapshotReadyProofItemCount = $releaseProofReadinessSnapshotReadyProofItemCount
  releaseProofReadinessSnapshotBlockedProofItemCount = $releaseProofReadinessSnapshotBlockedProofItemCount
  releaseProofReadinessSnapshotPerformsPublish = $releaseProofReadinessSnapshotPerformsPublish
  releaseProofReadinessSnapshotCanPublishPublicly = $releaseProofReadinessSnapshotCanPublishPublicly
  releaseProofReadinessSnapshotCanCloseReleaseIssue = $releaseProofReadinessSnapshotCanCloseReleaseIssue
  ownerProofInputReadinessState = $ownerProofInputReadinessState
  ownerProofInputReadinessContractCount = $ownerProofInputReadinessContractCount
  ownerProofInputReadinessReadyContractCount = $ownerProofInputReadinessReadyContractCount
  ownerProofInputReadinessBlockedContractCount = $ownerProofInputReadinessBlockedContractCount
  ownerProofInputReadinessPerformsPublish = $ownerProofInputReadinessPerformsPublish
  ownerProofInputReadinessCanPublishPublicly = $ownerProofInputReadinessCanPublishPublicly
  ownerProofInputReadinessCanCloseReleaseIssue = $ownerProofInputReadinessCanCloseReleaseIssue
  ownerProofInputReadinessValidationState = $ownerProofInputReadinessValidationState
  ownerProofInputReadinessIsValid = $ownerProofInputReadinessIsValid
  ownerProofInputReadinessValidationFailedBlockerCount = $ownerProofInputReadinessValidationFailedBlockerCount
  ownerProofInputReadinessValidationPerformsPublish = $ownerProofInputReadinessValidationPerformsPublish
  ownerProofInputReadinessValidationCanPublishPublicly = $ownerProofInputReadinessValidationCanPublishPublicly
  ownerProofInputReadinessValidationCanCloseReleaseIssue = $ownerProofInputReadinessValidationCanCloseReleaseIssue
  ownerRealEvidenceInputImportState = $ownerRealEvidenceInputImportState
  ownerRealEvidenceInputValidationState = $ownerRealEvidenceInputValidationState
  ownerRealEvidenceInputAcceptedLaneCount = $ownerRealEvidenceInputAcceptedLaneCount
  ownerRealEvidenceInputBlockedLaneCount = $ownerRealEvidenceInputBlockedLaneCount
  ownerRealEvidenceInputValidationFailedBlockerCount = $ownerRealEvidenceInputValidationFailedBlockerCount
  ownerRealEvidenceInputValidationFailedActionRequiredCount = $ownerRealEvidenceInputValidationFailedActionRequiredCount
  ownerRealEvidenceAcceptanceDashboardState = $ownerRealEvidenceAcceptanceDashboardState
  ownerRealEvidenceAcceptanceCanPrepareOwnerOnlyPublishCandidate = $ownerRealEvidenceAcceptanceCanPrepareOwnerOnlyPublishCandidate
  ownerOnlyPublishExecutionCandidateState = $ownerOnlyPublishExecutionCandidateState
  ownerOnlyPublishExecutionCandidateValidationState = $ownerOnlyPublishExecutionCandidateValidationState
  ownerOnlyPublishExecutionCandidateValidationFailedBlockerCount = $ownerOnlyPublishExecutionCandidateValidationFailedBlockerCount
  finalPublishActionRequiredEvidenceMapState = $finalPublishActionRequiredEvidenceMapState
  finalPublishActionRequiredEvidenceMapActionRequiredCount = $finalPublishActionRequiredEvidenceMapActionRequiredCount
  staleReleaseClaimsFindingCount = $staleFindingCount
  blockerCount = $releaseProofBlockerCount
  remainingBlockers = $remainingBlockers
  validatorCommands = $validatorCommands
  nonSubstituteProofKinds = $script:NonSubstituteProofKinds
  sourceArtifacts = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-promotion-issue-record.json",
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-close-gap-dashboard.json",
    "artifacts/final-release/compatible-host-proof-execution-pack.json",
    "artifacts/final-release/real-model-and-package-proof-input-package.json",
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/owner-release-execution-package.md",
    "artifacts/final-release/release-proof-readiness-snapshot.json",
    "artifacts/final-release/release-proof-readiness-snapshot.md",
    "artifacts/final-release/owner-proof-input-readiness.json",
    "artifacts/final-release/owner-proof-input-readiness.md",
    "artifacts/final-release/owner-proof-input-readiness-validation.json",
    "artifacts/final-release/owner-proof-input-readiness-validation.md",
    "artifacts/final-release/owner-real-evidence-input-import.json",
    "artifacts/final-release/owner-real-evidence-input-import.md",
    "artifacts/final-release/owner-real-evidence-input-validation.json",
    "artifacts/final-release/owner-real-evidence-input-validation.md",
    "artifacts/final-release/owner-real-evidence-acceptance-dashboard.json",
    "artifacts/final-release/owner-real-evidence-acceptance-dashboard.md",
    "artifacts/final-release/owner-only-publish-execution-candidate.json",
    "artifacts/final-release/owner-only-publish-execution-candidate.md",
    "artifacts/final-release/owner-only-publish-execution-candidate-validation.json",
    "artifacts/final-release/owner-only-publish-execution-candidate-validation.md",
    "artifacts/final-release/final-publish-action-required-evidence-map.json",
    "artifacts/final-release/stale-release-claims-audit.json"
  )
  ownerActionRequired = @(
    "Provide owner authorization.",
    "Provide owner-real-evidence-input.json with all six final action-required lanes, matching hashes, logs, host metadata, exit codes, owner review, and public channel metadata.",
    "Run package-consumer-runtime proof on a compatible CUDA/TensorRT host.",
    "Run Linux runner proof on a real Linux x64 runner.",
    "Provide real Classification/YoloVision model assets and sample logs.",
    "Run post-publish verification after real publication."
  )
  safetyNotes = @(
    "Final evidence freeze is a release snapshot and does not publish packages.",
    "performsPublish=false, canPublishPublicly=false, and canCloseReleaseIssue=false are fixed for this freeze.",
    "The freeze keeps blocked-real-proof-required until real proof records pass their validators.",
    "Release proof readiness snapshot is a compact status view only; it cannot publish, close the release issue, or substitute real proof records.",
    "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom and det、cls、seg、obb、pose、sem remain real asset requirements, not built-in proof."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-candidate-final-evidence-freeze.json"
$markdownPath = Join-Path $artifactRoot "release-candidate-final-evidence-freeze.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$blockerRows = $remainingBlockers | ForEach-Object {
  "| ``$($_.gapId)`` | ``$($_.proofClass)`` | ``$($_.passed)`` | $($_.currentState.Replace("|", "\|")) | ``$($_.validator)`` | $($_.freezeBoundary.Replace("|", "\|")) |"
}
$validatorLines = $validatorCommands | ForEach-Object { "- ``$_``" }
$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $script:NonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$ownerActionLines = $record.ownerActionRequired | ForEach-Object { "- $_" }
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }
$oneScreenRows = $oneScreenReleaseHoldChecklist | ForEach-Object {
  $requiredInputs = @($_.requiredRealInputs) -join "<br/>"
  $cannotUse = @($_.cannotUse) -join "<br/>"
  "| ``$($_.id)`` | $($_.ownerVisibleBlocker.Replace("|", "\|")) | $($_.currentState.Replace("|", "\|")) | $($_.ownerNextAction.Replace("|", "\|")) | ``$($_.validatorCommand)`` | $($requiredInputs.Replace("|", "\|")) | $($cannotUse.Replace("|", "\|")) |"
}

$markdown = @"
# Release Candidate Final Evidence Freeze

生成时间：$($record.generatedAtUtc)

## 总结

该 freeze 是发布候选最终证据冻结快照，用于汇总当前 release close 证据链、剩余 blocker、validator 和不可替代 proof 边界。它不执行发布、不上传包、不关闭 release issue。``recordKind=release-candidate-final-evidence-freeze``，``freezeState=blocked-real-proof-required``，``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前状态

| 项目 | 当前值 |
|---|---|
| releaseEvidenceBundleState | ``$evidenceState`` |
| releasePromotionIssueState | ``$promotionState`` |
| releaseClosePreflightState | ``$preflightState`` |
| releaseClosePreflightFailedItemCount | ``$preflightFailedItemCount`` |
| releaseCloseGapDashboardState | ``$dashboardState`` |
| releaseCloseGapDashboardGapCount | ``$dashboardGapCount`` |
| compatibleHostProofExecutionPackState | ``$executionPackState`` |
| compatibleHostProofExecutionPackBlockerCount | ``$executionPackBlockerCount`` |
| ownerReleaseExecutionPackageState | ``$ownerPackageState`` |
| oneScreenReleaseHoldChecklistCount | ``$oneScreenReleaseHoldChecklistCount`` |
| releaseProofReadinessSnapshotState | ``$releaseProofReadinessSnapshotState`` |
| releaseProofReadinessSnapshotItemCount | ``$releaseProofReadinessSnapshotItemCount`` |
| releaseProofReadinessSnapshotReadyProofItemCount | ``$releaseProofReadinessSnapshotReadyProofItemCount`` |
| releaseProofReadinessSnapshotBlockedProofItemCount | ``$releaseProofReadinessSnapshotBlockedProofItemCount`` |
| releaseProofReadinessSnapshotCanPublishPublicly | ``$releaseProofReadinessSnapshotCanPublishPublicly`` |
| releaseProofReadinessSnapshotCanCloseReleaseIssue | ``$releaseProofReadinessSnapshotCanCloseReleaseIssue`` |
| ownerProofInputReadinessState | ``$ownerProofInputReadinessState`` |
| ownerProofInputReadinessContractCount | ``$ownerProofInputReadinessContractCount`` |
| ownerProofInputReadinessReadyContractCount | ``$ownerProofInputReadinessReadyContractCount`` |
| ownerProofInputReadinessBlockedContractCount | ``$ownerProofInputReadinessBlockedContractCount`` |
| ownerProofInputReadinessCanPublishPublicly | ``$ownerProofInputReadinessCanPublishPublicly`` |
| ownerProofInputReadinessCanCloseReleaseIssue | ``$ownerProofInputReadinessCanCloseReleaseIssue`` |
| ownerProofInputReadinessValidationState | ``$ownerProofInputReadinessValidationState`` |
| ownerProofInputReadinessIsValid | ``$ownerProofInputReadinessIsValid`` |
| ownerProofInputReadinessValidationFailedBlockerCount | ``$ownerProofInputReadinessValidationFailedBlockerCount`` |
| ownerProofInputReadinessValidationCanPublishPublicly | ``$ownerProofInputReadinessValidationCanPublishPublicly`` |
| ownerProofInputReadinessValidationCanCloseReleaseIssue | ``$ownerProofInputReadinessValidationCanCloseReleaseIssue`` |
| ownerRealEvidenceInputImportState | ``$ownerRealEvidenceInputImportState`` |
| ownerRealEvidenceInputValidationState | ``$ownerRealEvidenceInputValidationState`` |
| ownerRealEvidenceInputAcceptedLaneCount | ``$ownerRealEvidenceInputAcceptedLaneCount`` |
| ownerRealEvidenceInputBlockedLaneCount | ``$ownerRealEvidenceInputBlockedLaneCount`` |
| ownerRealEvidenceInputValidationFailedBlockerCount | ``$ownerRealEvidenceInputValidationFailedBlockerCount`` |
| ownerRealEvidenceInputValidationFailedActionRequiredCount | ``$ownerRealEvidenceInputValidationFailedActionRequiredCount`` |
| ownerRealEvidenceAcceptanceDashboardState | ``$ownerRealEvidenceAcceptanceDashboardState`` |
| ownerRealEvidenceAcceptanceCanPrepareOwnerOnlyPublishCandidate | ``$ownerRealEvidenceAcceptanceCanPrepareOwnerOnlyPublishCandidate`` |
| ownerOnlyPublishExecutionCandidateState | ``$ownerOnlyPublishExecutionCandidateState`` |
| ownerOnlyPublishExecutionCandidateValidationState | ``$ownerOnlyPublishExecutionCandidateValidationState`` |
| ownerOnlyPublishExecutionCandidateValidationFailedBlockerCount | ``$ownerOnlyPublishExecutionCandidateValidationFailedBlockerCount`` |
| finalPublishActionRequiredEvidenceMapState | ``$finalPublishActionRequiredEvidenceMapState`` |
| finalPublishActionRequiredEvidenceMapActionRequiredCount | ``$finalPublishActionRequiredEvidenceMapActionRequiredCount`` |
| blockerCount | ``$($remainingBlockers.Count)`` |
| staleReleaseClaimsFindingCount | ``$staleFindingCount`` |

## One-Screen Release Hold Checklist

该清单镜像 ``owner-release-execution-package``，只用于 owner 一屏查看最终 release hold 项。它不执行发布、不关闭 release issue，并且所有项都必须由真实 validator proof 回填后才能解除。

| ID | Owner-visible blocker | Current state | Owner next action | Validator command | Required real inputs | Cannot use |
|---|---|---|---|---|---|---|
$($oneScreenRows -join "`r`n")

## Remaining Blockers

| Gap | Proof class | Passed | Current state | Validator | Freeze boundary |
|---|---|---|---|---|---|
$($blockerRows -join "`r`n")

## Validator Commands

$($validatorLines -join "`r`n")

## YoloVision 范围

YoloVision 真实样例 proof 范围固定为 ``YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom``，以及 ``det/cls/seg/obb/pose/sem``（``det、cls、seg、obb、pose、sem``）。这些真实样例 proof 只能晋级 ``real-model-runtime``，不能替代 ``package-consumer-runtime``。

## Owner Action Required

$($ownerActionLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Safety Notes

$($safetyLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release candidate final evidence freeze written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "FreezeState=$($record.freezeState)"
Write-Output "BlockerCount=$($remainingBlockers.Count)"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
