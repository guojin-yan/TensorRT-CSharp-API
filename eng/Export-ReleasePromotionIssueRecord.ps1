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

function New-PromotionItem {
  param(
    [string]$Id,
    [string]$Title,
    [string]$CurrentStatus,
    [string]$RequiredEvidence,
    [string]$OwnerAction,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    currentStatus = $CurrentStatus
    requiredEvidence = $RequiredEvidence
    ownerAction = $OwnerAction
    state = "pending-release-owner-approval"
    boundary = $Boundary
  }
}

function New-ChannelOption {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Preflight,
    [string]$PublishCommandPlaceholder,
    [string]$Rollback,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    preflight = $Preflight
    publishCommandPlaceholder = $PublishCommandPlaceholder
    rollback = $Rollback
    ownerDecision = "pending-release-owner-approval"
    boundary = $Boundary
  }
}

$manifest = Read-JsonOrNull "pack\runtime\runtime-packages.manifest.json"
$package = if ($manifest) { $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1 } else { $null }
$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$decisionRecord = Read-JsonOrNull "artifacts\final-release\release-owner-decision-record.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$releaseProofReadinessSnapshot = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$ownerProofInputReadiness = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness.json"
$ownerProofInputReadinessValidation = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness-validation.json"
$ownerApprovalInputValidation = Read-JsonOrNull "artifacts\final-release\release-owner-approval-input-validation.json"
$publishExecutionChecklist = Read-JsonOrNull "artifacts\final-release\release-publish-execution-checklist.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$releasePackageProof = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$docsPublishReadiness = Read-JsonOrNull "artifacts\final-release\docs-publish-readiness-bundle.json"
$externalRuntimeProof = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$externalRuntimeProofOwnerHandoff = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-owner-handoff.json"
$compatibleHostRunbook = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook.json"
$compatibleHostCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$externalRuntimeProofBackfillPlan = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-backfill-plan.json"
$postPublishVerificationBackfillPlan = Read-JsonOrNull "artifacts\final-release\post-publish-verification-backfill-plan.json"
$externalRuntimeProofCollectionPackage = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-collection-package.json"
$postPublishVerificationCollectionPackage = Read-JsonOrNull "artifacts\final-release\post-publish-verification-collection-package.json"
$postPublishCleanConsumerProjectScan = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-project-scan.json"
$postPublishVerificationInputDraft = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record.input-draft.json"
$realModelAndPackageProofInputPackage = Read-JsonOrNull "artifacts\final-release\real-model-and-package-proof-input-package.json"
$releaseCloseGapDashboard = Read-JsonOrNull "artifacts\final-release\release-close-gap-dashboard.json"
$compatibleHostProofExecutionPack = Read-JsonOrNull "artifacts\final-release\compatible-host-proof-execution-pack.json"
$releaseCandidateFinalEvidenceFreeze = Read-JsonOrNull "artifacts\final-release\release-candidate-final-evidence-freeze.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$postPublishVerificationRecord = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record-template.json"
$postPublishVerificationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$linuxRecordTemplate = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-record-template.json"
$linuxEvidenceTemplate = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-template.json"
$userAcceptance = Read-JsonOrNull "artifacts\user-acceptance\sample-smoke-catalog.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"

$overallStatus = if ($finalRelease) { [string]$finalRelease.overallStatus } else { "missing-final-release-dry-run" }
$blockingIssueCount = if ($finalRelease) { [int]$finalRelease.blockingIssueCount } else { -1 }
$manualApprovalCount = if ($finalRelease) { [int]$finalRelease.manualApprovalCount } else { -1 }
$smokeStatus = if ($finalRelease) { [string]$finalRelease.packageConsumerSmokeStatus } elseif ($packageConsumer) { [string]$packageConsumer.status } else { "missing" }
$runtimeProofStatus = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofStatus" -and -not [string]::IsNullOrWhiteSpace([string]$finalRelease.runtimeProofStatus)) { [string]$finalRelease.runtimeProofStatus } else { $smokeStatus }
$runtimeProofRequiredForRelease = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofRequiredForRelease") { [bool]$finalRelease.runtimeProofRequiredForRelease } else { -not [string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase) }
$allowRuntimeSmokeBlocked = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "allowRuntimeSmokeBlocked") { [bool]$finalRelease.allowRuntimeSmokeBlocked } else { $false }
$runtimeProofBlockerOwnerActionStatus = if ($releaseEvidenceBundle -and $releaseEvidenceBundle.PSObject.Properties.Name -contains "runtimeProofBlockerOwnerActionStatus") { [string]$releaseEvidenceBundle.runtimeProofBlockerOwnerActionStatus } elseif ([string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase)) { "resolved" } else { "owner-action-required" }
$runtimeProofBlockerCategory = if ($releaseEvidenceBundle -and $releaseEvidenceBundle.PSObject.Properties.Name -contains "runtimeProofBlockerOwnerActionCategory") { [string]$releaseEvidenceBundle.runtimeProofBlockerOwnerActionCategory } else {
  switch ($runtimeProofStatus) {
    "ready" { "none"; break }
    "blocked-by-cuda-driver" { "cuda-driver-runtime-compatibility"; break }
    "blocked-by-application-control" { "application-control-policy"; break }
    "not-requested" { "runtime-smoke-not-requested"; break }
    default { "runtime-proof-incomplete"; break }
  }
}
$runtimeProofOwnerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -RunSmoke -AllowSmokeFailure"
$realCallbackProof = if ($finalRelease) { [bool]$finalRelease.realCallbackRuntimeProof } else { $false }
$signingStatus = if ($finalRelease) { [string]$finalRelease.signingStatus } else { "missing" }
$staleFindingCount = if ($staleAudit) { [int]$staleAudit.findingCount } else { -1 }
$linuxProofState = if ($linuxRecordTemplate) { [string]$linuxRecordTemplate.recordState } elseif ($linuxEvidenceTemplate) { [string]$linuxEvidenceTemplate.evidenceState } else { "missing-linux-evidence-template" }
$isRealLinuxRunnerProof = if ($linuxRecordTemplate) { [bool]$linuxRecordTemplate.isRealLinuxRunnerProof } elseif ($linuxEvidenceTemplate) { [bool]$linuxEvidenceTemplate.isRealLinuxRunnerProof } else { $false }
$userAcceptanceStatus = if ($userAcceptance) { "itemCount=$($userAcceptance.itemCount); missingItemCount=$($userAcceptance.missingItemCount)" } else { "missing-user-acceptance-catalog" }
$packageId = if ($package) { [string]$package.packageId } else { "missing-runtime-package" }
$ownerApprovalValidationStatus = if ($ownerApprovalInputValidation) { [string]$ownerApprovalInputValidation.overallStatus } else { "missing-owner-approval-input-validation" }
$releaseEvidenceBundleState = if ($releaseEvidenceBundle) { [string]$releaseEvidenceBundle.bundleState } else { "missing-release-evidence-bundle" }
$releaseEvidenceComplete = if ($releaseEvidenceBundle -and $releaseEvidenceBundle.PSObject.Properties.Name -contains "isReleaseEvidenceComplete") { [bool]$releaseEvidenceBundle.isReleaseEvidenceComplete } else { $false }
$releasePackageProofState = if ($releasePackageProof) { [string]$releasePackageProof.proofState } else { "missing-release-package-proof-bundle" }
$canUseAsPublicPackageProof = if ($releasePackageProof -and $releasePackageProof.PSObject.Properties.Name -contains "canUseAsPublicPackageProof") { [bool]$releasePackageProof.canUseAsPublicPackageProof } else { $false }
$packageProofIsRuntimeExecutionProof = if ($releasePackageProof -and $releasePackageProof.PSObject.Properties.Name -contains "isRuntimeExecutionProof") { [bool]$releasePackageProof.isRuntimeExecutionProof } else { $false }
$docsPublishReadinessState = if ($docsPublishReadiness) { [string]$docsPublishReadiness.readinessState } else { "missing-docs-publish-readiness-bundle" }
$canPublishDocsExternally = if ($docsPublishReadiness -and $docsPublishReadiness.PSObject.Properties.Name -contains "canPublishDocsExternally") { [bool]$docsPublishReadiness.canPublishDocsExternally } else { $false }
$docsArticleCount = if ($docsPublishReadiness -and $docsPublishReadiness.PSObject.Properties.Name -contains "articleCount") { [int]$docsPublishReadiness.articleCount } else { -1 }
$ownerApprovalCanPublishPublicly = if ($ownerApprovalInputValidation -and $ownerApprovalInputValidation.PSObject.Properties.Name -contains "canPublishPublicly") { [bool]$ownerApprovalInputValidation.canPublishPublicly } else { $false }
$publishExecutionState = if ($publishExecutionChecklist) { [string]$publishExecutionChecklist.executionState } else { "missing-release-publish-execution-checklist" }
$canExecutePublicPublish = if ($publishExecutionChecklist -and $publishExecutionChecklist.PSObject.Properties.Name -contains "canExecutePublicPublish") { [bool]$publishExecutionChecklist.canExecutePublicPublish } else { $false }
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewNativeAssetCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "nativeAssetCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$externalRuntimeProofStateFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofState" -DefaultValue "missing-external-runtime-proof-record-template") } else { "missing-external-runtime-proof-record-template" }
$externalRuntimeProofClassificationFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofClassification" -DefaultValue "missing-proof-classification") } else { "missing-proof-classification" }
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofState" -DefaultValue $externalRuntimeProofStateFallback)
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofClassification" -DefaultValue $externalRuntimeProofClassificationFallback)
$externalRuntimeProofRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofPackageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofPackageSourceRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$externalRuntimeProofSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$externalRuntimeProofHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)))
$externalRuntimeProofCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)))
$externalRuntimeProofManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofLogSha256FormatReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256FormatReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256FormatReady" -DefaultValue $false)))
$externalRuntimeProofLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256Matches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256Matches" -DefaultValue $false)))
$externalRuntimeProofFailedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofFailedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)))
$externalRuntimeProofCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeProofOwnerActionStatus = if ($externalRuntimeProofCanPromoteRuntimeProof) { "resolved" } else { "owner-action-required" }
$externalIsRuntimeExecutionEvidence = if ($releaseEvidenceBundle -and $releaseEvidenceBundle.PSObject.Properties.Name -contains "externalRuntimeExecutionEvidence") { [bool]$releaseEvidenceBundle.externalRuntimeExecutionEvidence } elseif ($externalRuntimeProofValidation -and $externalRuntimeProofValidation.PSObject.Properties.Name -contains "isRuntimeExecutionEvidence") { [bool]$externalRuntimeProofValidation.isRuntimeExecutionEvidence } elseif ($externalRuntimeProof -and $externalRuntimeProof.PSObject.Properties.Name -contains "isRuntimeExecutionEvidence") { [bool]$externalRuntimeProof.isRuntimeExecutionEvidence } else { $false }
$externalRuntimeProofDraftState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftState" -DefaultValue "missing-external-runtime-proof-draft")
$externalRuntimeProofDraftCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftCanPromoteRuntimeProof" -DefaultValue $false)
$draftManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftManagedNupkgSha256Ready" -DefaultValue $false)
$draftRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftRuntimeNupkgSha256Ready" -DefaultValue $false)
$draftSmokeLogSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftLogSha256Ready" -DefaultValue $false)
$draftNoProjectReference = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftNoProjectReference" -DefaultValue $false)
$draftSmokeStatus = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofDraftSmokeStatus" -DefaultValue "missing-smoke-status")
$compatibleHostRequired = -not $externalIsRuntimeExecutionEvidence -or [string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)
$requiredHostAction = "Run package consumer smoke on a CUDA-compatible host and replace the draft with a real external-runtime-proof-record."
$promotionBlockedReason = if ($externalRuntimeProofCanPromoteRuntimeProof -and $externalIsRuntimeExecutionEvidence) { "none" } elseif ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -or [string]::Equals($draftSmokeStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)) { "blocked-by-cuda-driver is not smoke passed" } else { "external runtime proof is not promotable package-consumer-runtime proof" }
$externalRuntimeProofOwnerHandoffState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofOwnerHandoff -Name "handoffState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofOwnerHandoffState" -DefaultValue "missing-external-runtime-proof-owner-handoff")))
$externalRuntimeProofOwnerHandoffOwnerActionStatus = [string](Get-PropertyOrDefault -Object $externalRuntimeProofOwnerHandoff -Name "ownerActionStatus" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofOwnerHandoffOwnerActionStatus" -DefaultValue "owner-action-required")))
$compatibleHostRunbookState = [string](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "runbookState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookState" -DefaultValue "missing-compatible-host-runtime-proof-runbook")))
$compatibleHostRunbookCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "compatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookCompatibleHostRequired" -DefaultValue $true)))
$compatibleHostRunbookPerformsPublish = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookPerformsPublish" -DefaultValue $false)))
$compatibleHostRunbookApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "approvesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookApprovesPublicRelease" -DefaultValue $false)))
$compatibleHostRunbookCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostRunbookRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostRunbookPromotionBlockedReason = [string](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "promotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookPromotionBlockedReason" -DefaultValue "missing-compatible-host-runbook-blocked-reason")))
$compatibleHostCollectionBundleCommands = Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "commands" -DefaultValue $null
$compatibleHostCollectionBundleOperatorQuickStart = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "operatorQuickStart" -DefaultValue @())
$compatibleHostCollectionBundlePreflightChecklist = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "preflightChecklist" -DefaultValue @())
$compatibleHostCollectionBundleCopyableExecutionOrder = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "copyableExecutionOrder" -DefaultValue @())
$compatibleHostCollectionBundleOwnerInputArtifacts = @(Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "ownerInputArtifacts" -DefaultValue @())
$compatibleHostCollectionBundleState = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "collectionState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleState" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")))
$compatibleHostCollectionBundleCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "compatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired" -DefaultValue $true)))
$compatibleHostCollectionBundlePerformsPublish = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePerformsPublish" -DefaultValue $false)))
$compatibleHostCollectionBundleApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "approvesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease" -DefaultValue $false)))
$compatibleHostCollectionBundleCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostCollectionBundleRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostCollectionBundlePromotionBlockedReason = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "promotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason" -DefaultValue "missing-compatible-host-collection-bundle-blocked-reason")))
$compatibleHostCollectionBundleRunPackageConsumerSmokeCommand = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundleCommands -Name "runPackageConsumerSmoke" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand" -DefaultValue $runtimeProofOwnerCommand)))
$compatibleHostCollectionBundleValidateFilledRecordCommand = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundleCommands -Name "validateFilledRecord" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof")))
$compatibleHostCollectionBundleQuickStartCount = $compatibleHostCollectionBundleOperatorQuickStart.Count
$compatibleHostCollectionBundlePreflightCount = $compatibleHostCollectionBundlePreflightChecklist.Count
$compatibleHostCollectionBundleExecutionOrderCount = $compatibleHostCollectionBundleCopyableExecutionOrder.Count
$compatibleHostCollectionBundleOwnerInputArtifactCount = $compatibleHostCollectionBundleOwnerInputArtifacts.Count
$externalRuntimeProofBackfillPlanState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "planState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofBackfillPlanState" -DefaultValue "missing-external-runtime-proof-backfill-plan")))
$externalRuntimeProofBackfillStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofBackfillStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count))
$externalRuntimeProofBackfillCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofBackfillPlan -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofBackfillCanPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeProofCollectionPackageState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "packageState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCollectionPackageState" -DefaultValue "missing-external-runtime-proof-collection-package")))
$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCollectionPackageCanPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeProofCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCollectionPackageCanCloseReleaseIssue" -DefaultValue $false)))
$externalRuntimeProofCollectionPackageRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCollectionPackageRuntimeExecutionEvidence" -DefaultValue $false)))
$externalRuntimeProofCollectionPackageStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCollectionPackageStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "collectionSteps" -DefaultValue @())).Count))
$externalRuntimeProofCollectionPackageExecutionOrderCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCollectionPackageExecutionOrderCount" -DefaultValue (@((Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "copyableExecutionOrder" -DefaultValue @())).Count))
$postPublishVerificationBackfillPlanState = [string](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "planState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationBackfillPlanState" -DefaultValue "missing-post-publish-verification-backfill-plan")))
$postPublishVerificationBackfillStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationBackfillStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "backfillSteps" -DefaultValue @())).Count))
$postPublishVerificationBackfillCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationBackfillPlan -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationBackfillCanCloseReleaseIssue" -DefaultValue $false)))
$postPublishVerificationCollectionPackageState = [string](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "packageState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationCollectionPackageState" -DefaultValue "missing-post-publish-verification-collection-package")))
$postPublishVerificationCollectionPackageProof = [bool](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "isPostPublishVerificationProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationCollectionPackageProof" -DefaultValue $false)))
$postPublishVerificationCollectionPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationCollectionPackageCanCloseReleaseIssue" -DefaultValue $false)))
$postPublishVerificationCollectionPackageStepCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationCollectionPackageStepCount" -DefaultValue (@((Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "collectionSteps" -DefaultValue @())).Count))
$postPublishVerificationCollectionPackageExecutionOrderCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationCollectionPackageExecutionOrderCount" -DefaultValue (@((Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "copyableExecutionOrder" -DefaultValue @())).Count))
$postPublishCleanConsumerProjectScanState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCleanConsumerProjectScanState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "scanState" -DefaultValue "missing-post-publish-clean-consumer-project-scan")))
$postPublishCleanConsumerProjectScanPassed = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCleanConsumerProjectScanPassed" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "scanPassed" -DefaultValue $false)))
$postPublishCleanConsumerProjectScanCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCleanConsumerProjectScanCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "canCloseReleaseIssue" -DefaultValue $false)))
$postPublishCleanConsumerProjectScanIsProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCleanConsumerProjectScanIsProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$postPublishVerificationInputDraftKind = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationInputDraftKind" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "recordKind" -DefaultValue "missing-post-publish-verification-record-input-draft")))
$postPublishVerificationInputDraftOnly = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationInputDraftOnly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "inputDraftOnly" -DefaultValue $false)))
$postPublishVerificationInputDraftIsProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationInputDraftIsProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$postPublishVerificationInputDraftCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationInputDraftCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "canCloseReleaseIssue" -DefaultValue $false)))
$realModelAndPackageProofInputPackageState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "realModelAndPackageProofInputPackageState" -DefaultValue ([string](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "packageState" -DefaultValue "missing-real-model-and-package-proof-input-package")))
$realModelAndPackageProofInputPackagePerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "realModelAndPackageProofInputPackagePerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "performsPublish" -DefaultValue $false)))
$realModelAndPackageProofInputPackageCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "realModelAndPackageProofInputPackageCanPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "canPublishPublicly" -DefaultValue $false)))
$realModelAndPackageProofInputPackageCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "realModelAndPackageProofInputPackageCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "canCloseReleaseIssue" -DefaultValue $false)))
$realModelAndPackageProofInputChecklistCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "realModelAndPackageProofInputChecklistCount" -DefaultValue (@((Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "inputChecklists" -DefaultValue @())).Count))
$realModelAndPackageProofInputExecutionOrderCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "realModelAndPackageProofInputExecutionOrderCount" -DefaultValue (@((Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "copyableExecutionOrder" -DefaultValue @())).Count))
$releaseCloseGapDashboardState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCloseGapDashboardState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "dashboardState" -DefaultValue "missing-release-close-gap-dashboard")))
$releaseCloseGapDashboardGapCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCloseGapDashboardGapCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "gapCount" -DefaultValue -1)))
$releaseCloseGapDashboardPerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCloseGapDashboardPerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "performsPublish" -DefaultValue $false)))
$releaseCloseGapDashboardCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCloseGapDashboardCanPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "canPublishPublicly" -DefaultValue $false)))
$releaseCloseGapDashboardCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCloseGapDashboardCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "canCloseReleaseIssue" -DefaultValue $false)))
$compatibleHostProofExecutionPackState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostProofExecutionPackState" -DefaultValue ([string](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "packageState" -DefaultValue "missing-compatible-host-proof-execution-pack")))
$compatibleHostProofExecutionPackBlockerCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostProofExecutionPackBlockerCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "blockerCount" -DefaultValue -1)))
$compatibleHostProofExecutionPackPerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostProofExecutionPackPerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "performsPublish" -DefaultValue $false)))
$compatibleHostProofExecutionPackCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostProofExecutionPackCanPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "canPublishPublicly" -DefaultValue $false)))
$compatibleHostProofExecutionPackCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostProofExecutionPackCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "canCloseReleaseIssue" -DefaultValue $false)))
$ownerReleaseExecutionPackageState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "ownerReleaseExecutionPackageState" -DefaultValue "missing-owner-release-execution-package")))
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
$releaseCandidateFinalEvidenceFreezeState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCandidateFinalEvidenceFreezeState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseCandidateFinalEvidenceFreeze -Name "freezeState" -DefaultValue "missing-release-candidate-final-evidence-freeze")))
$releaseCandidateFinalEvidenceFreezeBlockerCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCandidateFinalEvidenceFreezeBlockerCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseCandidateFinalEvidenceFreeze -Name "blockerCount" -DefaultValue -1)))
$releaseCandidateFinalEvidenceFreezePerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCandidateFinalEvidenceFreezePerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseCandidateFinalEvidenceFreeze -Name "performsPublish" -DefaultValue $false)))
$releaseCandidateFinalEvidenceFreezeCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCandidateFinalEvidenceFreezeCanPublishPublicly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseCandidateFinalEvidenceFreeze -Name "canPublishPublicly" -DefaultValue $false)))
$releaseCandidateFinalEvidenceFreezeCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseCandidateFinalEvidenceFreezeCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseCandidateFinalEvidenceFreeze -Name "canCloseReleaseIssue" -DefaultValue $false)))
$releaseClosePreflightState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseClosePreflightState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")))
$releaseClosePreflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseClosePreflightFailedItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)))
$releaseClosePreflightCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseClosePreflightCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)))
$releaseClosePreflightPerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "releaseClosePreflightPerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "performsPublish" -DefaultValue $true)))
$postPublishVerificationState = if ($postPublishVerificationValidation) { [string]$postPublishVerificationValidation.validationState } elseif ($postPublishVerificationRecord) { [string]$postPublishVerificationRecord.verificationState } else { "missing-post-publish-verification-record-template" }
$postPublishProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")))
$postPublishProofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassificationPromotable" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishProofClassificationPromotable" -DefaultValue $false)))
$postPublishManagedPackageId = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishManagedPackageId" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "managedPackageId" -DefaultValue "missing-managed-package-id")))
$postPublishManagedPackageVersion = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishManagedPackageVersion" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "managedPackageVersion" -DefaultValue "missing-managed-package-version")))
$postPublishRuntimePackageId = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRuntimePackageId" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "runtimePackageId" -DefaultValue "missing-runtime-package-id")))
$postPublishRuntimePackageVersion = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRuntimePackageVersion" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "runtimePackageVersion" -DefaultValue "missing-runtime-package-version")))
$postPublishManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$postPublishRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$postPublishConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$postPublishSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$postPublishHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "hostReady" -DefaultValue $false)))
$postPublishCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "commandsReady" -DefaultValue $false)))
$postPublishStdoutSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStdoutSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "stdoutSummaryReady" -DefaultValue $false)))
$postPublishStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "stderrSummaryReady" -DefaultValue $false)))
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStdoutStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "stdoutStderrSummaryReady" -DefaultValue $false)))
$postPublishAllLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishAllLogSha256Matches" -DefaultValue $false)
$isPostPublishVerificationProof = if ($postPublishVerificationValidation -and $postPublishVerificationValidation.PSObject.Properties.Name -contains "isPostPublishVerificationProof") { [bool]$postPublishVerificationValidation.isPostPublishVerificationProof } elseif ($postPublishVerificationRecord -and $postPublishVerificationRecord.PSObject.Properties.Name -contains "isPostPublishVerificationProof") { [bool]$postPublishVerificationRecord.isPostPublishVerificationProof } else { $false }
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "canCloseReleaseIssue" -DefaultValue $false)))

$promotionItems = @(
  New-PromotionItem `
    -Id "release-evidence-bundle" `
    -Title "Release evidence bundle" `
    -CurrentStatus ("bundleState=" + $releaseEvidenceBundleState + "; isReleaseEvidenceComplete=" + $releaseEvidenceComplete) `
    -RequiredEvidence "artifacts/final-release/release-evidence-bundle.json; artifacts/final-release/release-evidence-bundle.md" `
    -OwnerAction "Review this aggregation layer and fix the source evidence items before promotion." `
    -Boundary "The release evidence bundle does not approve publication or execute package push."
  New-PromotionItem `
    -Id "owner-approval-input" `
    -Title "Release owner approval input validation" `
    -CurrentStatus ("validationStatus=" + $ownerApprovalValidationStatus + "; canPublishPublicly=" + $ownerApprovalCanPublishPublicly) `
    -RequiredEvidence "artifacts/final-release/release-owner-approval-input-record.json; artifacts/final-release/release-owner-approval-input-validation.json" `
    -OwnerAction "Fill and validate the owner approval input record before public promotion." `
    -Boundary "A template-only or blocked owner input validation cannot approve publication."
  New-PromotionItem `
    -Id "publish-execution-checklist" `
    -Title "Publish execution checklist" `
    -CurrentStatus ("executionState=" + $publishExecutionState + "; canExecutePublicPublish=" + $canExecutePublicPublish) `
    -RequiredEvidence "artifacts/final-release/release-publish-execution-checklist.json; artifacts/final-release/release-publish-execution-checklist.md" `
    -OwnerAction "Generate and review the owner-gated publish preflight, rollback, and post-publish verification checklist." `
    -Boundary "The publish execution checklist contains placeholders and does not push packages."
  New-PromotionItem `
    -Id "final-package-review-bundle" `
    -Title "Final package review bundle" `
    -CurrentStatus ("bundleState=" + $finalPackageReviewState + "; packageCount=" + $finalPackageReviewPackageCount + "; nativeAssetCount=" + $finalPackageReviewNativeAssetCount + "; canUseAsPublicPackageProof=" + $finalPackageReviewCanUseAsPublicPackageProof) `
    -RequiredEvidence "artifacts/final-release/final-package-review-bundle.json; artifacts/final-release/final-package-review-bundle.md" `
    -OwnerAction "Review local managed/runtime/split package identities, SHA256 hashes, sizes, and native asset inventory before public channel decisions." `
    -Boundary "The final package review bundle is local package inventory and is not public channel proof."
  New-PromotionItem `
    -Id "release-package-proof-bundle" `
    -Title "Release package proof bundle" `
    -CurrentStatus ("proofState=" + $releasePackageProofState + "; canUseAsPublicPackageProof=" + $canUseAsPublicPackageProof + "; isRuntimeExecutionProof=" + $packageProofIsRuntimeExecutionProof) `
    -RequiredEvidence "artifacts/final-release/release-package-proof-bundle.json; artifacts/final-release/release-package-proof-bundle.md" `
    -OwnerAction "Use this bundle to review local package, split package, native-copy, and consumer evidence before channel promotion." `
    -Boundary "The release package proof bundle does not prove public publication or runtime execution."
  New-PromotionItem `
    -Id "docs-publish-readiness-bundle" `
    -Title "Docs publish readiness bundle" `
    -CurrentStatus ("readinessState=" + $docsPublishReadinessState + "; articleCount=" + $docsArticleCount + "; canPublishDocsExternally=" + $canPublishDocsExternally) `
    -RequiredEvidence "artifacts/final-release/docs-publish-readiness-bundle.json; artifacts/final-release/docs-publish-readiness-bundle.md" `
    -OwnerAction "Review the article set, sample-backed docs, DocFX output, external channel plan, and media/cover asset plan." `
    -Boundary "The docs publish readiness bundle does not publish docs externally."
  New-PromotionItem `
    -Id "external-runtime-proof-record" `
    -Title "External runtime proof record" `
    -CurrentStatus ("proofState=" + $externalRuntimeProofState + "; classification=" + $externalRuntimeProofClassification + "; runtimePackageKeyMatches=" + $externalRuntimeProofRuntimePackageKeyMatches + "; packageSourceRuntimePackageKeyMatches=" + $externalRuntimeProofPackageSourceRuntimePackageKeyMatches + "; consumerProjectIdentityReady=" + $externalRuntimeProofConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $externalRuntimeProofSmokeCommandRuntimeKeyReady + "; hostReady=" + $externalRuntimeProofHostReady + "; commandsReady=" + $externalRuntimeProofCommandsReady + "; managedNupkgSha256Ready=" + $externalRuntimeProofManagedNupkgSha256Ready + "; runtimeNupkgSha256Ready=" + $externalRuntimeProofRuntimeNupkgSha256Ready + "; logSha256FormatReady=" + $externalRuntimeProofLogSha256FormatReady + "; logSha256Matches=" + $externalRuntimeProofLogSha256Matches + "; failedProofItemCount=" + $externalRuntimeProofFailedProofItemCount + "; ownerAction=" + $externalRuntimeProofOwnerActionStatus + "; isRuntimeExecutionEvidence=" + $externalIsRuntimeExecutionEvidence) `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-record-template.json; artifacts/final-release/external-runtime-proof-validation.json" `
    -OwnerAction "Attach a compatible CUDA host smoke record before claiming runtime proof ready; the record must match runtimePackageKey, identify the clean consumer project, include complete CUDA/TensorRT/cuDNN host metadata, include --runtime-package-key in smokeCommand, and include the verified smoke logSha256." `
    -Boundary "A template-only, runtime-key-mismatched, missing consumer identity, missing host metadata, missing runtime-key smoke command, or missing-log-hash external runtime proof record is not runtime execution proof."
  New-PromotionItem `
    -Id "external-runtime-proof-owner-handoff" `
    -Title "External runtime proof owner handoff" `
    -CurrentStatus ("handoffState=" + $externalRuntimeProofOwnerHandoffState + "; ownerAction=" + $externalRuntimeProofOwnerHandoffOwnerActionStatus) `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-owner-handoff.json; artifacts/final-release/external-runtime-proof-owner-handoff.md" `
    -OwnerAction "Use the generated handoff commands on a compatible CUDA host and attach the filled external-runtime-proof-record.json with a matching smoke log SHA256." `
    -Boundary "The handoff is owner guidance only and is not runtime execution proof."
  New-PromotionItem `
    -Id "compatible-host-runtime-proof-runbook" `
    -Title "Compatible host runtime proof runbook" `
    -CurrentStatus ("runbookState=" + $compatibleHostRunbookState + "; compatibleHostRequired=" + $compatibleHostRunbookCompatibleHostRequired + "; canPromoteRuntimeProof=" + $compatibleHostRunbookCanPromoteRuntimeProof + "; isRuntimeExecutionEvidence=" + $compatibleHostRunbookRuntimeExecutionEvidence + "; performsPublish=" + $compatibleHostRunbookPerformsPublish + "; approvesPublicRelease=" + $compatibleHostRunbookApprovesPublicRelease + "; promotionBlockedReason=" + $compatibleHostRunbookPromotionBlockedReason) `
    -RequiredEvidence "artifacts/final-release/compatible-host-runtime-proof-runbook.json; artifacts/final-release/compatible-host-runtime-proof-runbook.md" `
    -OwnerAction "Execute the runbook on a compatible CUDA host, fill the real proof record, and attach the -FailOnNotProof validation output before promotion." `
    -Boundary "The runbook is a command guide and field checklist; it is not runtime execution proof, publication approval, or package push."
  New-PromotionItem `
    -Id "compatible-host-runtime-proof-collection-bundle" `
    -Title "Compatible host runtime proof collection bundle" `
    -CurrentStatus ("collectionState=" + $compatibleHostCollectionBundleState + "; compatibleHostRequired=" + $compatibleHostCollectionBundleCompatibleHostRequired + "; canPromoteRuntimeProof=" + $compatibleHostCollectionBundleCanPromoteRuntimeProof + "; isRuntimeExecutionEvidence=" + $compatibleHostCollectionBundleRuntimeExecutionEvidence + "; performsPublish=" + $compatibleHostCollectionBundlePerformsPublish + "; approvesPublicRelease=" + $compatibleHostCollectionBundleApprovesPublicRelease + "; quickStart=" + $compatibleHostCollectionBundleQuickStartCount + "; preflight=" + $compatibleHostCollectionBundlePreflightCount + "; copyableExecutionOrder=" + $compatibleHostCollectionBundleExecutionOrderCount + "; promotionBlockedReason=" + $compatibleHostCollectionBundlePromotionBlockedReason) `
    -RequiredEvidence "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json; artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md" `
    -OwnerAction "Execute the collection bundle on a compatible CUDA host, fill the real proof record, and attach the -FailOnNotProof validation output before promotion." `
    -Boundary "The collection bundle is a command collection and hash checklist; it is not runtime execution proof, publication approval, or package push."
  New-PromotionItem `
    -Id "external-runtime-proof-backfill-plan" `
    -Title "External runtime proof backfill plan" `
    -CurrentStatus ("planState=" + $externalRuntimeProofBackfillPlanState + "; stepCount=" + $externalRuntimeProofBackfillStepCount + "; canPromoteRuntimeProof=" + $externalRuntimeProofBackfillCanPromoteRuntimeProof) `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-backfill-plan.json; artifacts/final-release/external-runtime-proof-backfill-plan.md" `
    -OwnerAction "Use as guidance for collecting a real compatible-host external-runtime-proof-record.json; do not treat this plan as proof." `
    -Boundary "The backfill plan is guidance only and is not runtime proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "external-runtime-proof-collection-package" `
    -Title "External runtime proof collection package" `
    -CurrentStatus ("packageState=" + $externalRuntimeProofCollectionPackageState + "; stepCount=" + $externalRuntimeProofCollectionPackageStepCount + "; copyableExecutionOrder=" + $externalRuntimeProofCollectionPackageExecutionOrderCount + "; canPromoteRuntimeProof=" + $externalRuntimeProofCollectionPackageCanPromoteRuntimeProof + "; canCloseReleaseIssue=" + $externalRuntimeProofCollectionPackageCanCloseReleaseIssue + "; isRuntimeExecutionEvidence=" + $externalRuntimeProofCollectionPackageRuntimeExecutionEvidence) `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-collection-package.json; artifacts/final-release/external-runtime-proof-collection-package.md" `
    -OwnerAction "Use as a copyable owner package for collecting real compatible-host proof; attach only the validated real external-runtime-proof-record.json as proof." `
    -Boundary "The collection package is guidance only and is not runtime proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "post-publish-verification-record" `
    -Title "Post-publish verification record" `
    -CurrentStatus ("verificationState=" + $postPublishVerificationState + "; classification=" + $postPublishProofClassification + "; promotable=" + $postPublishProofClassificationPromotable + "; managedPackage=" + $postPublishManagedPackageId + "/" + $postPublishManagedPackageVersion + "; runtimePackage=" + $postPublishRuntimePackageId + "/" + $postPublishRuntimePackageVersion + "; managedNupkgSha256Ready=" + $postPublishManagedNupkgSha256Ready + "; runtimeNupkgSha256Ready=" + $postPublishRuntimeNupkgSha256Ready + "; consumerProjectIdentityReady=" + $postPublishConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $postPublishSmokeCommandRuntimeKeyReady + "; hostReady=" + $postPublishHostReady + "; commandsReady=" + $postPublishCommandsReady + "; stdoutSummaryReady=" + $postPublishStdoutSummaryReady + "; stderrSummaryReady=" + $postPublishStderrSummaryReady + "; stdoutStderrSummaryReady=" + $postPublishStdoutStderrSummaryReady + "; allLogSha256Matches=" + $postPublishAllLogSha256Matches + "; isPostPublishVerificationProof=" + $isPostPublishVerificationProof + "; canCloseReleaseIssue=" + $canCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/post-publish-verification-record-template.json; artifacts/final-release/post-publish-verification-validation.json" `
    -OwnerAction "Fill after a real channel publish using clean consumer project identity, target channel package URLs/hashes, compatible host metadata, --runtime-package-key smoke command, stdout/stderr summaries, and SHA256-backed logs." `
    -Boundary "The post-publish verification template, missing consumer identity, missing host metadata, missing runtime-key smoke command, or missing stdout/stderr summary does not publish packages and is not proof."
  New-PromotionItem `
    -Id "post-publish-verification-backfill-plan" `
    -Title "Post-publish verification backfill plan" `
    -CurrentStatus ("planState=" + $postPublishVerificationBackfillPlanState + "; stepCount=" + $postPublishVerificationBackfillStepCount + "; canCloseReleaseIssue=" + $postPublishVerificationBackfillCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/post-publish-verification-backfill-plan.json; artifacts/final-release/post-publish-verification-backfill-plan.md" `
    -OwnerAction "Use after authorized publication as guidance for real post-publish verification; do not treat this plan as close proof." `
    -Boundary "The backfill plan is guidance only and is not post-publish proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "post-publish-verification-collection-package" `
    -Title "Post-publish verification collection package" `
    -CurrentStatus ("packageState=" + $postPublishVerificationCollectionPackageState + "; stepCount=" + $postPublishVerificationCollectionPackageStepCount + "; copyableExecutionOrder=" + $postPublishVerificationCollectionPackageExecutionOrderCount + "; isPostPublishVerificationProof=" + $postPublishVerificationCollectionPackageProof + "; canCloseReleaseIssue=" + $postPublishVerificationCollectionPackageCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/post-publish-verification-collection-package.json; artifacts/final-release/post-publish-verification-collection-package.md" `
    -OwnerAction "Use only after authorized publication to collect real post-publish proof from a clean external consumer; do not treat this package as close proof." `
    -Boundary "The collection package is guidance only and is not post-publish proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "post-publish-clean-consumer-project-scan" `
    -Title "Post-publish clean consumer project scan" `
    -CurrentStatus ("scanState=" + $postPublishCleanConsumerProjectScanState + "; scanPassed=" + $postPublishCleanConsumerProjectScanPassed + "; isPostPublishVerificationProof=" + $postPublishCleanConsumerProjectScanIsProof + "; canCloseReleaseIssue=" + $postPublishCleanConsumerProjectScanCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/post-publish-clean-consumer-project-scan.json; artifacts/final-release/post-publish-clean-consumer-project-scan.md" `
    -OwnerAction "Use the scan to verify the real post-publish consumer project boundary before filling the post-publish verification record." `
    -Boundary "Clean consumer scanning is helper evidence only; it is not post-publish proof, release close approval, or package push."
  New-PromotionItem `
    -Id "post-publish-verification-input-draft" `
    -Title "Post-publish verification input draft" `
    -CurrentStatus ("recordKind=" + $postPublishVerificationInputDraftKind + "; inputDraftOnly=" + $postPublishVerificationInputDraftOnly + "; isPostPublishVerificationProof=" + $postPublishVerificationInputDraftIsProof + "; canCloseReleaseIssue=" + $postPublishVerificationInputDraftCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/post-publish-verification-record.input-draft.json; artifacts/final-release/post-publish-verification-record.input-draft.md" `
    -OwnerAction "Use the input draft as a copy source only after real publication; validate the filled real post-publish record separately." `
    -Boundary "Input drafts can collect paths and hashes, but they are not post-publish proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "real-model-and-package-proof-input-package" `
    -Title "Real model and package proof input package" `
    -CurrentStatus ("packageState=" + $realModelAndPackageProofInputPackageState + "; checklistCount=" + $realModelAndPackageProofInputChecklistCount + "; copyableExecutionOrder=" + $realModelAndPackageProofInputExecutionOrderCount + "; performsPublish=" + $realModelAndPackageProofInputPackagePerformsPublish + "; canPublishPublicly=" + $realModelAndPackageProofInputPackageCanPublishPublicly + "; canCloseReleaseIssue=" + $realModelAndPackageProofInputPackageCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/real-model-and-package-proof-input-package.json; artifacts/final-release/real-model-and-package-proof-input-package.md" `
    -OwnerAction "Use as the consolidated owner input checklist for package-consumer-runtime, real-model-runtime, and post-publish verification backfill; validate the filled real proof records separately." `
    -Boundary "The input package is owner guidance only and is not package-consumer-runtime proof, real-model-runtime proof, post-publish proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "release-close-gap-dashboard" `
    -Title "Release close gap dashboard" `
    -CurrentStatus ("dashboardState=" + $releaseCloseGapDashboardState + "; gapCount=" + $releaseCloseGapDashboardGapCount + "; performsPublish=" + $releaseCloseGapDashboardPerformsPublish + "; canPublishPublicly=" + $releaseCloseGapDashboardCanPublishPublicly + "; canCloseReleaseIssue=" + $releaseCloseGapDashboardCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/release-close-gap-dashboard.json; artifacts/final-release/release-close-gap-dashboard.md" `
    -OwnerAction "Use the dashboard to execute remaining real proof backfill in order; validate the filled real proof records separately." `
    -Boundary "The dashboard is owner guidance only and is not package-consumer-runtime proof, real-model-runtime proof, Linux runner proof, post-publish proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "compatible-host-proof-execution-pack" `
    -Title "Compatible host proof execution pack" `
    -CurrentStatus ("packageState=" + $compatibleHostProofExecutionPackState + "; blockerCount=" + $compatibleHostProofExecutionPackBlockerCount + "; performsPublish=" + $compatibleHostProofExecutionPackPerformsPublish + "; canPublishPublicly=" + $compatibleHostProofExecutionPackCanPublishPublicly + "; canCloseReleaseIssue=" + $compatibleHostProofExecutionPackCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/compatible-host-proof-execution-pack.json; artifacts/final-release/compatible-host-proof-execution-pack.md" `
    -OwnerAction "Use the execution pack as a one-stop owner command index for remaining real proof backfill; validate only the filled real proof records as proof." `
    -Boundary "The execution pack is owner guidance only and is not package-consumer-runtime proof, real-model-runtime proof, Linux runner proof, post-publish proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "release-candidate-final-evidence-freeze" `
    -Title "Release candidate final evidence freeze" `
    -CurrentStatus ("freezeState=" + $releaseCandidateFinalEvidenceFreezeState + "; blockerCount=" + $releaseCandidateFinalEvidenceFreezeBlockerCount + "; performsPublish=" + $releaseCandidateFinalEvidenceFreezePerformsPublish + "; canPublishPublicly=" + $releaseCandidateFinalEvidenceFreezeCanPublishPublicly + "; canCloseReleaseIssue=" + $releaseCandidateFinalEvidenceFreezeCanCloseReleaseIssue) `
    -RequiredEvidence "artifacts/final-release/release-candidate-final-evidence-freeze.json; artifacts/final-release/release-candidate-final-evidence-freeze.md" `
    -OwnerAction "Use the final freeze as a release-candidate evidence snapshot before owner real proof execution." `
    -Boundary "The final evidence freeze is a snapshot only and is not package-consumer-runtime proof, real-model-runtime proof, Linux runner proof, post-publish proof, publication approval, release close approval, or package push."
  New-PromotionItem `
    -Id "release-close-preflight" `
    -Title "Release close preflight" `
    -CurrentStatus ("preflightState=" + $releaseClosePreflightState + "; failedItemCount=" + $releaseClosePreflightFailedItemCount + "; canCloseReleaseIssue=" + $releaseClosePreflightCanCloseReleaseIssue + "; performsPublish=" + $releaseClosePreflightPerformsPublish) `
    -RequiredEvidence "artifacts/final-release/release-close-preflight.json; artifacts/final-release/release-close-preflight.md" `
    -OwnerAction "Use this aggregator to see real-proof gaps before attempting release issue closure." `
    -Boundary "Release close preflight aggregates gaps; it is not owner authorization, runtime proof, post-publish proof, or package push."
  New-PromotionItem `
    -Id "final-dry-run" `
    -Title "Final dry run snapshot" `
    -CurrentStatus $overallStatus `
    -RequiredEvidence "artifacts/final-release/final-release-dry-run-summary.json" `
    -OwnerAction "Review all manual approvals before choosing a promotion channel." `
    -Boundary "ready-needs-manual-approval is not public release approval."
  New-PromotionItem `
    -Id "stale-release-claims" `
    -Title "Stale release claims audit" `
    -CurrentStatus ("findingCount=" + $staleFindingCount) `
    -RequiredEvidence "artifacts/final-release/stale-release-claims-audit.json" `
    -OwnerAction "Keep findingCount at zero after editing promotion issue text and release notes." `
    -Boundary "The audit only checks known stale claim patterns; it does not approve publication."
  New-PromotionItem `
    -Id "runtime-package" `
    -Title "Runtime package target" `
    -CurrentStatus $packageId `
    -RequiredEvidence "pack/runtime/runtime-packages.manifest.json; artifacts/release-candidate/runtime-package-matrix.json" `
    -OwnerAction "Confirm package ID, RID, TensorRT/CUDA/cuDNN line, and distribution tier." `
    -Boundary "A manifest package entry is not proof that the package was pushed to a public source."
  New-PromotionItem `
    -Id "signing" `
    -Title "Signing policy" `
    -CurrentStatus $signingStatus `
    -RequiredEvidence "docs/articles/zh-cn/signing-and-trust-policy.md" `
    -OwnerAction "Approve unsigned RC distribution or attach signing evidence." `
    -Boundary "unsigned-or-not-requested is not signed output."
  New-PromotionItem `
    -Id "nvidia-redistribution" `
    -Title "NVIDIA redistribution approval" `
    -CurrentStatus "pending-legal-or-owner-review" `
    -RequiredEvidence "pack/runtime/runtime-packages.manifest.json; release owner approval notes" `
    -OwnerAction "Confirm redistribution rights for CUDA, cuDNN, and TensorRT artifacts." `
    -Boundary "Do not publicly publish NVIDIA runtime components without redistribution approval."
  New-PromotionItem `
    -Id "linux-runner-proof" `
    -Title "Linux runner proof" `
    -CurrentStatus ("recordState=" + $linuxProofState + "; isRealLinuxRunnerProof=" + $isRealLinuxRunnerProof) `
    -RequiredEvidence "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record-template.md" `
    -OwnerAction "Attach real Linux x64 runner output or keep Linux line as handoff only." `
    -Boundary "template-only and dry-run-only are not Linux runner proof."
  New-PromotionItem `
    -Id "runtime-smoke" `
    -Title "Full package runtime smoke" `
    -CurrentStatus ("packageConsumerSmokeStatus=" + $smokeStatus + "; runtimeProofStatus=" + $runtimeProofStatus + "; runtimeProofRequiredForRelease=" + $runtimeProofRequiredForRelease + "; allowRuntimeSmokeBlocked=" + $allowRuntimeSmokeBlocked + "; ownerAction=" + $runtimeProofBlockerOwnerActionStatus + "; blockerCategory=" + $runtimeProofBlockerCategory) `
    -RequiredEvidence "artifacts/package-consumer/package-consumer-validation-summary.json" `
    -OwnerAction "Accept the environment blocker for RC documentation or rerun on a CUDA-compatible GPU host. Suggested command: $runtimeProofOwnerCommand" `
    -Boundary "blocked-by-cuda-driver, dependency-probe-only, runtime-deserialization-dependency-diagnostics, runtimeProofRequiredForRelease=true, and allowRuntimeSmokeBlocked=true are not smoke passed."
  New-PromotionItem `
    -Id "callback-proof" `
    -Title "Real callback runtime proof" `
    -CurrentStatus ("realCallbackRuntimeProof=" + $realCallbackProof) `
    -RequiredEvidence "artifacts/release-candidate/release-candidate-readiness-summary.json; docs/articles/zh-cn/real-callback-runtime-evidence-schema.md" `
    -OwnerAction "Keep proof false or require InvocationCount>0 package-consumer evidence before promotion." `
    -Boundary "schema-ready, precheck, design gate, and InvocationCount=0 are not real callback runtime proof."
  New-PromotionItem `
    -Id "user-acceptance" `
    -Title "User acceptance sample catalog" `
    -CurrentStatus $userAcceptanceStatus `
    -RequiredEvidence "artifacts/user-acceptance/sample-smoke-catalog.json" `
    -OwnerAction "Keep asset-required samples separate from smoke passes and attach model asset evidence only after real runs." `
    -Boundary "Classification/YoloVision asset candidates are not sample smoke passes."
)

$channels = @(
  New-ChannelOption `
    -Id "local-feed" `
    -Title "Local release candidate feed" `
    -Preflight "Run Test-LocalNuGetFeedConsumer.ps1 and verify no ProjectReference is used." `
    -PublishCommandPlaceholder "No public push; local feed is a dry-run consumer path." `
    -Rollback "Delete the local feed folder and rebuild packages." `
    -Boundary "Local feed validation is not nuget.org or GitHub Packages publication."
  New-ChannelOption `
    -Id "nuget-org" `
    -Title "nuget.org" `
    -Preflight "Verify package ownership, NUGET_API_KEY scope, signing policy, package size, and NVIDIA redistribution approval." `
    -PublishCommandPlaceholder "dotnet nuget push <package>.nupkg --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json" `
    -Rollback "nuget.org versions cannot be overwritten; unlist or publish a corrected version according to owner policy." `
    -Boundary "This record does not execute dotnet nuget push."
  New-ChannelOption `
    -Id "github-packages" `
    -Title "GitHub Packages" `
    -Preflight "Verify package source URL, package owner, token package:write permission, retention policy, and restore credentials." `
    -PublishCommandPlaceholder "dotnet nuget push <package>.nupkg --api-key <GITHUB_TOKEN> --source <github-packages-source>" `
    -Rollback "Delete or supersede the package version according to organization permissions and retention policy." `
    -Boundary "This record does not upload to GitHub Packages."
  New-ChannelOption `
    -Id "github-release-assets" `
    -Title "GitHub Release assets" `
    -Preflight "Verify tag, release notes, asset checksums, and consumer instructions for adding a local source." `
    -PublishCommandPlaceholder "gh release upload <tag> <package>.nupkg <package>.sha256" `
    -Rollback "Delete the asset or publish corrected release notes and replacement assets." `
    -Boundary "Release assets are not a NuGet restore source by themselves."
  New-ChannelOption `
    -Id "private-feed" `
    -Title "Private feed" `
    -Preflight "Verify organization feed retention, access control, package size limits, and consumer NuGet.config." `
    -PublishCommandPlaceholder "dotnet nuget push <package>.nupkg --source <private-feed-source>" `
    -Rollback "Follow the organization feed delete, deprecate, or supersede policy." `
    -Boundary "Private feed success does not prove public channel success."
)

$postPublishVerification = @(
  "Restore a clean consumer project from the chosen channel without ProjectReference and record consumerProjectName/consumerProjectPath.",
  "Verify native bridge and runtime assets are copied to the output directory.",
  "Capture OS, GPU, driver, CUDA driver/runtime, TensorRT runtime/line, and cuDNN version from the compatible smoke host.",
  "Record DependencyProbe BridgeInitialized output.",
  "Run smoke only on a compatible CUDA driver and GPU host, with smokeCommand containing --runtime-package-key $RuntimePackageKey.",
  "Record stdoutSummary and stderrSummary together with SHA256-backed restore/build/dependency/smoke logs.",
  "Keep callback proof false unless InvocationCount>0 and IsRealCallbackRuntimeProof=True are present.",
  "Refresh final release dry run and stale release claims audit after publication notes are edited."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-promotion-issue-record"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  packageId = $packageId
  promotionState = "pending-release-owner-approval"
  canPublishPublicly = $false
  performsPublish = $false
  ownerActionStatus = "owner-action-required"
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isRealModelRuntimeProof = $false
  requiresHumanOwner = $true
  overallStatus = $overallStatus
  blockingIssueCount = $blockingIssueCount
  manualApprovalCount = $manualApprovalCount
  packageConsumerSmokeStatus = $smokeStatus
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  allowRuntimeSmokeBlocked = $allowRuntimeSmokeBlocked
  runtimeProofBlockerOwnerActionStatus = $runtimeProofBlockerOwnerActionStatus
  runtimeProofBlockerCategory = $runtimeProofBlockerCategory
  runtimeProofOwnerCommand = $runtimeProofOwnerCommand
  realCallbackRuntimeProof = $realCallbackProof
  signingStatus = $signingStatus
  staleReleaseClaimsFindingCount = $staleFindingCount
  ownerApprovalInputValidationStatus = $ownerApprovalValidationStatus
  releaseEvidenceBundleState = $releaseEvidenceBundleState
  isReleaseEvidenceComplete = $releaseEvidenceComplete
  releasePackageProofState = $releasePackageProofState
  canUseAsPublicPackageProof = $canUseAsPublicPackageProof
  packageProofIsRuntimeExecutionProof = $packageProofIsRuntimeExecutionProof
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewNativeAssetCount = $finalPackageReviewNativeAssetCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  docsPublishReadinessState = $docsPublishReadinessState
  canPublishDocsExternally = $canPublishDocsExternally
  docsArticleCount = $docsArticleCount
  ownerApprovalCanPublishPublicly = $ownerApprovalCanPublishPublicly
  publishExecutionChecklistState = $publishExecutionState
  canExecutePublicPublish = $canExecutePublicPublish
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
  externalRuntimeProofRuntimePackageKeyMatches = $externalRuntimeProofRuntimePackageKeyMatches
  externalRuntimeProofPackageSourceRuntimePackageKeyMatches = $externalRuntimeProofPackageSourceRuntimePackageKeyMatches
  externalRuntimeProofConsumerProjectIdentityReady = $externalRuntimeProofConsumerProjectIdentityReady
  externalRuntimeProofSmokeCommandRuntimeKeyReady = $externalRuntimeProofSmokeCommandRuntimeKeyReady
  externalRuntimeProofHostReady = $externalRuntimeProofHostReady
  externalRuntimeProofCommandsReady = $externalRuntimeProofCommandsReady
  externalRuntimeProofManagedNupkgSha256Ready = $externalRuntimeProofManagedNupkgSha256Ready
  externalRuntimeProofRuntimeNupkgSha256Ready = $externalRuntimeProofRuntimeNupkgSha256Ready
  externalRuntimeProofLogSha256FormatReady = $externalRuntimeProofLogSha256FormatReady
  externalRuntimeProofLogSha256Matches = $externalRuntimeProofLogSha256Matches
  externalRuntimeProofFailedProofItemCount = $externalRuntimeProofFailedProofItemCount
  externalRuntimeProofOwnerActionStatus = $externalRuntimeProofOwnerActionStatus
  externalRuntimeProofCanPromoteRuntimeProof = $externalRuntimeProofCanPromoteRuntimeProof
  externalRuntimeExecutionEvidence = $externalIsRuntimeExecutionEvidence
  externalRuntimeProofDraftState = $externalRuntimeProofDraftState
  externalRuntimeProofDraftCanPromoteRuntimeProof = $externalRuntimeProofDraftCanPromoteRuntimeProof
  draftManagedNupkgSha256Ready = $draftManagedNupkgSha256Ready
  draftRuntimeNupkgSha256Ready = $draftRuntimeNupkgSha256Ready
  draftSmokeLogSha256Ready = $draftSmokeLogSha256Ready
  draftNoProjectReference = $draftNoProjectReference
  draftSmokeStatus = $draftSmokeStatus
  compatibleHostRequired = $compatibleHostRequired
  requiredHostAction = $requiredHostAction
  promotionBlockedReason = $promotionBlockedReason
  externalRuntimeProofOwnerHandoffState = $externalRuntimeProofOwnerHandoffState
  externalRuntimeProofOwnerHandoffOwnerActionStatus = $externalRuntimeProofOwnerHandoffOwnerActionStatus
  compatibleHostRuntimeProofRunbookState = $compatibleHostRunbookState
  compatibleHostRuntimeProofRunbookCompatibleHostRequired = $compatibleHostRunbookCompatibleHostRequired
  compatibleHostRuntimeProofRunbookPerformsPublish = $compatibleHostRunbookPerformsPublish
  compatibleHostRuntimeProofRunbookApprovesPublicRelease = $compatibleHostRunbookApprovesPublicRelease
  compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof = $compatibleHostRunbookCanPromoteRuntimeProof
  compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence = $compatibleHostRunbookRuntimeExecutionEvidence
  compatibleHostRuntimeProofRunbookPromotionBlockedReason = $compatibleHostRunbookPromotionBlockedReason
  compatibleHostRuntimeProofCollectionBundleState = $compatibleHostCollectionBundleState
  compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired = $compatibleHostCollectionBundleCompatibleHostRequired
  compatibleHostRuntimeProofCollectionBundlePerformsPublish = $compatibleHostCollectionBundlePerformsPublish
  compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease = $compatibleHostCollectionBundleApprovesPublicRelease
  compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = $compatibleHostCollectionBundleCanPromoteRuntimeProof
  compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence = $compatibleHostCollectionBundleRuntimeExecutionEvidence
  compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason = $compatibleHostCollectionBundlePromotionBlockedReason
  compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand = $compatibleHostCollectionBundleRunPackageConsumerSmokeCommand
  compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand = $compatibleHostCollectionBundleValidateFilledRecordCommand
  compatibleHostRuntimeProofCollectionBundleQuickStartCount = $compatibleHostCollectionBundleQuickStartCount
  compatibleHostRuntimeProofCollectionBundlePreflightChecklistCount = $compatibleHostCollectionBundlePreflightCount
  compatibleHostRuntimeProofCollectionBundleCopyableExecutionOrderCount = $compatibleHostCollectionBundleExecutionOrderCount
  compatibleHostRuntimeProofCollectionBundleOperatorQuickStart = @($compatibleHostCollectionBundleOperatorQuickStart)
  compatibleHostRuntimeProofCollectionBundlePreflightChecklist = @($compatibleHostCollectionBundlePreflightChecklist)
  compatibleHostRuntimeProofCollectionBundleCopyableExecutionOrder = @($compatibleHostCollectionBundleCopyableExecutionOrder)
  compatibleHostRuntimeProofCollectionBundleOwnerInputArtifactCount = $compatibleHostCollectionBundleOwnerInputArtifactCount
  compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts = @($compatibleHostCollectionBundleOwnerInputArtifacts)
  externalRuntimeProofBackfillPlanState = $externalRuntimeProofBackfillPlanState
  externalRuntimeProofBackfillStepCount = $externalRuntimeProofBackfillStepCount
  externalRuntimeProofBackfillCanPromoteRuntimeProof = $externalRuntimeProofBackfillCanPromoteRuntimeProof
  externalRuntimeProofCollectionPackageState = $externalRuntimeProofCollectionPackageState
  externalRuntimeProofCollectionPackageStepCount = $externalRuntimeProofCollectionPackageStepCount
  externalRuntimeProofCollectionPackageExecutionOrderCount = $externalRuntimeProofCollectionPackageExecutionOrderCount
  externalRuntimeProofCollectionPackageCanPromoteRuntimeProof = $externalRuntimeProofCollectionPackageCanPromoteRuntimeProof
  externalRuntimeProofCollectionPackageCanCloseReleaseIssue = $externalRuntimeProofCollectionPackageCanCloseReleaseIssue
  externalRuntimeProofCollectionPackageRuntimeExecutionEvidence = $externalRuntimeProofCollectionPackageRuntimeExecutionEvidence
  postPublishVerificationState = $postPublishVerificationState
  postPublishProofClassification = $postPublishProofClassification
  postPublishProofClassificationPromotable = $postPublishProofClassificationPromotable
  postPublishManagedPackageId = $postPublishManagedPackageId
  postPublishManagedPackageVersion = $postPublishManagedPackageVersion
  postPublishRuntimePackageId = $postPublishRuntimePackageId
  postPublishRuntimePackageVersion = $postPublishRuntimePackageVersion
  postPublishManagedNupkgSha256Ready = $postPublishManagedNupkgSha256Ready
  postPublishRuntimeNupkgSha256Ready = $postPublishRuntimeNupkgSha256Ready
  postPublishConsumerProjectIdentityReady = $postPublishConsumerProjectIdentityReady
  postPublishSmokeCommandRuntimeKeyReady = $postPublishSmokeCommandRuntimeKeyReady
  postPublishHostReady = $postPublishHostReady
  postPublishCommandsReady = $postPublishCommandsReady
  postPublishStdoutSummaryReady = $postPublishStdoutSummaryReady
  postPublishStderrSummaryReady = $postPublishStderrSummaryReady
  postPublishStdoutStderrSummaryReady = $postPublishStdoutStderrSummaryReady
  postPublishAllLogSha256Matches = $postPublishAllLogSha256Matches
  isPostPublishVerificationProof = $isPostPublishVerificationProof
  postPublishVerificationBackfillPlanState = $postPublishVerificationBackfillPlanState
  postPublishVerificationBackfillStepCount = $postPublishVerificationBackfillStepCount
  postPublishVerificationBackfillCanCloseReleaseIssue = $postPublishVerificationBackfillCanCloseReleaseIssue
  postPublishVerificationCollectionPackageState = $postPublishVerificationCollectionPackageState
  postPublishVerificationCollectionPackageStepCount = $postPublishVerificationCollectionPackageStepCount
  postPublishVerificationCollectionPackageExecutionOrderCount = $postPublishVerificationCollectionPackageExecutionOrderCount
  postPublishVerificationCollectionPackageProof = $postPublishVerificationCollectionPackageProof
  postPublishVerificationCollectionPackageCanCloseReleaseIssue = $postPublishVerificationCollectionPackageCanCloseReleaseIssue
  postPublishCleanConsumerProjectScanState = $postPublishCleanConsumerProjectScanState
  postPublishCleanConsumerProjectScanPassed = $postPublishCleanConsumerProjectScanPassed
  postPublishCleanConsumerProjectScanCanCloseReleaseIssue = $postPublishCleanConsumerProjectScanCanCloseReleaseIssue
  postPublishCleanConsumerProjectScanIsProof = $postPublishCleanConsumerProjectScanIsProof
  postPublishVerificationInputDraftKind = $postPublishVerificationInputDraftKind
  postPublishVerificationInputDraftOnly = $postPublishVerificationInputDraftOnly
  postPublishVerificationInputDraftIsProof = $postPublishVerificationInputDraftIsProof
  postPublishVerificationInputDraftCanCloseReleaseIssue = $postPublishVerificationInputDraftCanCloseReleaseIssue
  realModelAndPackageProofInputPackageState = $realModelAndPackageProofInputPackageState
  realModelAndPackageProofInputPackagePerformsPublish = $realModelAndPackageProofInputPackagePerformsPublish
  realModelAndPackageProofInputPackageCanPublishPublicly = $realModelAndPackageProofInputPackageCanPublishPublicly
  realModelAndPackageProofInputPackageCanCloseReleaseIssue = $realModelAndPackageProofInputPackageCanCloseReleaseIssue
  realModelAndPackageProofInputChecklistCount = $realModelAndPackageProofInputChecklistCount
  realModelAndPackageProofInputExecutionOrderCount = $realModelAndPackageProofInputExecutionOrderCount
  releaseCloseGapDashboardState = $releaseCloseGapDashboardState
  releaseCloseGapDashboardGapCount = $releaseCloseGapDashboardGapCount
  releaseCloseGapDashboardPerformsPublish = $releaseCloseGapDashboardPerformsPublish
  releaseCloseGapDashboardCanPublishPublicly = $releaseCloseGapDashboardCanPublishPublicly
  releaseCloseGapDashboardCanCloseReleaseIssue = $releaseCloseGapDashboardCanCloseReleaseIssue
  compatibleHostProofExecutionPackState = $compatibleHostProofExecutionPackState
  compatibleHostProofExecutionPackBlockerCount = $compatibleHostProofExecutionPackBlockerCount
  compatibleHostProofExecutionPackPerformsPublish = $compatibleHostProofExecutionPackPerformsPublish
  compatibleHostProofExecutionPackCanPublishPublicly = $compatibleHostProofExecutionPackCanPublishPublicly
  compatibleHostProofExecutionPackCanCloseReleaseIssue = $compatibleHostProofExecutionPackCanCloseReleaseIssue
  ownerReleaseExecutionPackageState = $ownerReleaseExecutionPackageState
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
  releaseCandidateFinalEvidenceFreezeState = $releaseCandidateFinalEvidenceFreezeState
  releaseCandidateFinalEvidenceFreezeBlockerCount = $releaseCandidateFinalEvidenceFreezeBlockerCount
  releaseCandidateFinalEvidenceFreezePerformsPublish = $releaseCandidateFinalEvidenceFreezePerformsPublish
  releaseCandidateFinalEvidenceFreezeCanPublishPublicly = $releaseCandidateFinalEvidenceFreezeCanPublishPublicly
  releaseCandidateFinalEvidenceFreezeCanCloseReleaseIssue = $releaseCandidateFinalEvidenceFreezeCanCloseReleaseIssue
  releaseClosePreflightState = $releaseClosePreflightState
  releaseClosePreflightFailedItemCount = $releaseClosePreflightFailedItemCount
  releaseClosePreflightCanCloseReleaseIssue = $releaseClosePreflightCanCloseReleaseIssue
  releaseClosePreflightPerformsPublish = $releaseClosePreflightPerformsPublish
  canCloseReleaseIssue = $canCloseReleaseIssue
  isRealLinuxRunnerProof = $isRealLinuxRunnerProof
  sourceEvidence = @(
    "artifacts/final-release/final-release-dry-run-summary.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/owner-release-execution-package.md",
    "artifacts/final-release/release-proof-readiness-snapshot.json",
    "artifacts/final-release/release-proof-readiness-snapshot.md",
    "artifacts/final-release/owner-proof-input-readiness.json",
    "artifacts/final-release/owner-proof-input-readiness.md",
    "artifacts/final-release/owner-proof-input-readiness-validation.json",
    "artifacts/final-release/owner-proof-input-readiness-validation.md",
    "artifacts/final-release/final-package-review-bundle.json",
    "artifacts/final-release/release-package-proof-bundle.json",
    "artifacts/final-release/docs-publish-readiness-bundle.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-publish-execution-checklist.json",
    "artifacts/final-release/external-runtime-proof-record-template.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-owner-handoff.json",
    "artifacts/final-release/compatible-host-runtime-proof-runbook.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/release-candidate-package-inventory.json",
    "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.md",
    "artifacts/final-release/external-runtime-proof-collection-package.json",
    "artifacts/final-release/external-runtime-proof-collection-package.md",
    "artifacts/final-release/post-publish-verification-record-template.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/post-publish-verification-backfill-plan.json",
    "artifacts/final-release/post-publish-verification-backfill-plan.md",
    "artifacts/final-release/post-publish-verification-collection-package.json",
    "artifacts/final-release/post-publish-verification-collection-package.md",
    "artifacts/final-release/post-publish-clean-consumer-project-scan.json",
    "artifacts/final-release/post-publish-clean-consumer-project-scan.md",
    "artifacts/final-release/post-publish-verification-record.input-draft.json",
    "artifacts/final-release/post-publish-verification-record.input-draft.md",
    "artifacts/final-release/real-model-and-package-proof-input-package.json",
    "artifacts/final-release/real-model-and-package-proof-input-package.md",
    "artifacts/final-release/release-close-gap-dashboard.json",
    "artifacts/final-release/release-close-gap-dashboard.md",
    "artifacts/final-release/compatible-host-proof-execution-pack.json",
    "artifacts/final-release/compatible-host-proof-execution-pack.md",
    "artifacts/final-release/release-candidate-final-evidence-freeze.json",
    "artifacts/final-release/release-candidate-final-evidence-freeze.md",
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-close-preflight.md",
    "artifacts/final-release/stale-release-claims-audit.json",
    "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record-template.json",
    "artifacts/user-acceptance/sample-smoke-catalog.json"
  )
  promotionItems = $promotionItems
  channelOptions = $channels
  postPublishVerification = $postPublishVerification
  safetyNotes = @(
    "This issue record does not push packages.",
    "The release evidence bundle is an aggregation layer, not release owner approval.",
    "The final package review bundle is local package inventory, not public package proof.",
    "The release package proof bundle is not public package proof and is not runtime execution proof.",
    "The docs publish readiness bundle is not external documentation publication proof.",
    "canPublishPublicly=false until release owner records an explicit decision.",
    "ownerApprovalCanPublishPublicly=false means owner approval input validation has not approved public promotion.",
    "canExecutePublicPublish=false means the publish execution checklist has not approved running any public push.",
    "External runtime proof templates, runtime-key-mismatched records, missing consumer identity, missing host metadata, missing --runtime-package-key smoke commands, missing package hashes, missing-log-hash records, and post-publish verification templates are not proof.",
    "External runtime proof owner handoff is a backfill guide, not runtime proof.",
    "Compatible host runtime proof runbook is a command guide, not runtime proof, public approval, or package push.",
    "Compatible host runtime proof collection bundle is an external execution package, not runtime proof, publication approval, or package push.",
    "compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts are owner inputs only; package inventory, final package review, release package proof, and local feed consumer summaries cannot promote runtime proof or post-publish proof.",
    "Backfill plans are guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push.",
    "Collection packages are copyable owner guidance only; they are not runtime proof, post-publish proof, publication approval, release close approval, or package push.",
    "Post-publish clean consumer scan and input draft are helper artifacts only; they cannot close the release issue.",
    "Real model and package proof input package is an owner input checklist only; it cannot substitute package-consumer-runtime, real-model-runtime, or post-publish verification proof.",
    "Release close gap dashboard is an owner guidance dashboard only; it cannot substitute any real proof or owner authorization.",
    "Release proof readiness snapshot is a compact status view only; it cannot publish, close the release issue, or substitute real proof records.",
    "Release close preflight aggregates real-proof gaps but cannot substitute external runtime proof, owner authorization, or post-publish verification proof.",
    "Post-publish verification requires clean consumer project identity, compatible host CUDA/TensorRT/cuDNN metadata, --runtime-package-key smoke command, stdout/stderr summaries, and SHA256-backed logs before it can close a release issue.",
    "Local feed validation is a consumer dry run, not public publication.",
    "Linux runner proof remains false until a Linux x64 runner supplies real evidence.",
    "blocked-by-cuda-driver is not smoke passed.",
    "runtime-deserialization-dependency-diagnostics and runtime proof blocker owner action are not runtime execution proof.",
    "runtimeProofRequiredForRelease=true means public release still needs compatible runtime proof or explicit owner disposition.",
    "allowRuntimeSmokeBlocked=true records dry-run intent only; it is not smoke passed.",
    "realCallbackRuntimeProof=false must remain visible until InvocationCount>0 evidence exists."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-promotion-issue-record.json"
$markdownPath = Join-Path $outputRoot "release-promotion-issue-record.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Promotion Issue Record")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Linux runtime key: ``$LinuxRuntimePackageKey``")
$lines.Add("")
$lines.Add("Promotion state: ``pending-release-owner-approval``")
$lines.Add("")
$lines.Add("Can publish publicly: ``false``")
$lines.Add("")
$lines.Add("This record is an issue body for release-owner review. It does not push packages, approve publication, or turn dry-run evidence into release proof.")
$lines.Add("")
$lines.Add("## Evidence Snapshot")
$lines.Add("")
$lines.Add("- package id: ``$packageId``")
$lines.Add("- overall status: ``$overallStatus``")
$lines.Add("- blocking issues: $blockingIssueCount")
$lines.Add("- manual approvals: $manualApprovalCount")
$lines.Add("- package consumer smoke: ``$smokeStatus``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- runtime proof required for release: ``$runtimeProofRequiredForRelease``")
$lines.Add("- allow runtime smoke blocked: ``$allowRuntimeSmokeBlocked``")
$lines.Add("- runtime proof blocker owner action: ``$runtimeProofBlockerOwnerActionStatus``")
$lines.Add("- runtime proof blocker category: ``$runtimeProofBlockerCategory``")
$lines.Add("- runtime proof suggested command: ``$runtimeProofOwnerCommand``")
$lines.Add("- real callback runtime proof: ``$realCallbackProof``")
$lines.Add("- signing status: ``$signingStatus``")
$lines.Add("- stale release claim findings: $staleFindingCount")
$lines.Add("- owner approval input validation: ``$ownerApprovalValidationStatus``")
$lines.Add("- release evidence bundle: ``$releaseEvidenceBundleState``")
$lines.Add("- release evidence complete: ``$releaseEvidenceComplete``")
$lines.Add("- final package review bundle: ``$finalPackageReviewState``")
$lines.Add("- final package review package count: $finalPackageReviewPackageCount")
$lines.Add("- final package review native asset count: $finalPackageReviewNativeAssetCount")
$lines.Add("- final package review can use as public package proof: ``$finalPackageReviewCanUseAsPublicPackageProof``")
$lines.Add("- release package proof bundle: ``$releasePackageProofState``")
$lines.Add("- can use as public package proof: ``$canUseAsPublicPackageProof``")
$lines.Add("- package proof is runtime execution proof: ``$packageProofIsRuntimeExecutionProof``")
$lines.Add("- docs publish readiness bundle: ``$docsPublishReadinessState``")
$lines.Add("- docs article count: $docsArticleCount")
$lines.Add("- can publish docs externally: ``$canPublishDocsExternally``")
$lines.Add("- owner approval can publish publicly: ``$ownerApprovalCanPublishPublicly``")
$lines.Add("- publish execution checklist: ``$publishExecutionState``")
$lines.Add("- can execute public publish: ``$canExecutePublicPublish``")
$lines.Add("- external runtime proof state: ``$externalRuntimeProofState``")
$lines.Add("- external runtime proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- external runtime proof runtime key matches: ``$externalRuntimeProofRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof package source runtime key matches: ``$externalRuntimeProofPackageSourceRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof consumer project identity ready: ``$externalRuntimeProofConsumerProjectIdentityReady``")
$lines.Add("- external runtime proof smoke command runtime key ready: ``$externalRuntimeProofSmokeCommandRuntimeKeyReady``")
$lines.Add("- external runtime proof host metadata ready: ``$externalRuntimeProofHostReady``")
$lines.Add("- external runtime proof commands ready: ``$externalRuntimeProofCommandsReady``")
$lines.Add("- external runtime proof managed nupkg SHA256 ready: ``$externalRuntimeProofManagedNupkgSha256Ready``")
$lines.Add("- external runtime proof runtime nupkg SHA256 ready: ``$externalRuntimeProofRuntimeNupkgSha256Ready``")
$lines.Add("- external runtime proof log SHA256 format ready: ``$externalRuntimeProofLogSha256FormatReady``")
$lines.Add("- external runtime proof log SHA256 matches: ``$externalRuntimeProofLogSha256Matches``")
$lines.Add("- external runtime proof draft state: ``$externalRuntimeProofDraftState``")
$lines.Add("- draft managed nupkg SHA256 ready: ``$draftManagedNupkgSha256Ready``")
$lines.Add("- draft runtime nupkg SHA256 ready: ``$draftRuntimeNupkgSha256Ready``")
$lines.Add("- draft smoke log SHA256 ready: ``$draftSmokeLogSha256Ready``")
$lines.Add("- draft no ProjectReference: ``$draftNoProjectReference``")
$lines.Add("- draft smoke status: ``$draftSmokeStatus``")
$lines.Add("- compatible host required: ``$compatibleHostRequired``")
$lines.Add("- required host action: $requiredHostAction")
$lines.Add("- promotion blocked reason: $promotionBlockedReason")
$lines.Add("- external runtime proof failed proof item count: ``$externalRuntimeProofFailedProofItemCount``")
$lines.Add("- external runtime proof owner action: ``$externalRuntimeProofOwnerActionStatus``")
$lines.Add("- external runtime proof owner handoff: ``$externalRuntimeProofOwnerHandoffState``")
$lines.Add("- compatible host runtime proof runbook: ``$compatibleHostRunbookState``")
$lines.Add("- compatible host runbook can promote proof: ``$compatibleHostRunbookCanPromoteRuntimeProof``")
$lines.Add("- compatible host runbook performs publish: ``$compatibleHostRunbookPerformsPublish``")
$lines.Add("- compatible host runtime proof collection bundle: ``$compatibleHostCollectionBundleState``")
$lines.Add("- compatible host collection bundle can promote proof: ``$compatibleHostCollectionBundleCanPromoteRuntimeProof``")
$lines.Add("- compatible host collection bundle runtime execution evidence: ``$compatibleHostCollectionBundleRuntimeExecutionEvidence``")
$lines.Add("- compatible host collection bundle quick start items: ``$compatibleHostCollectionBundleQuickStartCount``")
$lines.Add("- compatible host collection bundle preflight items: ``$compatibleHostCollectionBundlePreflightCount``")
$lines.Add("- compatible host collection bundle copyable execution order items: ``$compatibleHostCollectionBundleExecutionOrderCount``")
$lines.Add("- compatible host collection bundle smoke command: ``$compatibleHostCollectionBundleRunPackageConsumerSmokeCommand``")
$lines.Add("- compatible host collection bundle validation command: ``$compatibleHostCollectionBundleValidateFilledRecordCommand``")
$lines.Add("- owner release execution package: ``$ownerReleaseExecutionPackageState``")
$lines.Add("- one-screen release hold checklist count: ``$oneScreenReleaseHoldChecklistCount``")
$lines.Add("- release proof readiness snapshot: ``$releaseProofReadinessSnapshotState``")
$lines.Add("- release proof readiness item count: ``$releaseProofReadinessSnapshotItemCount``")
$lines.Add("- release proof readiness ready proof item count: ``$releaseProofReadinessSnapshotReadyProofItemCount``")
$lines.Add("- release proof readiness blocked proof item count: ``$releaseProofReadinessSnapshotBlockedProofItemCount``")
$lines.Add("- release proof readiness can publish publicly: ``$releaseProofReadinessSnapshotCanPublishPublicly``")
$lines.Add("- release proof readiness can close release issue: ``$releaseProofReadinessSnapshotCanCloseReleaseIssue``")
$lines.Add("- owner proof input readiness: ``$ownerProofInputReadinessState``")
$lines.Add("- owner proof input contract count: ``$ownerProofInputReadinessContractCount``")
$lines.Add("- owner proof input ready contract count: ``$ownerProofInputReadinessReadyContractCount``")
$lines.Add("- owner proof input blocked contract count: ``$ownerProofInputReadinessBlockedContractCount``")
$lines.Add("- owner proof input can publish publicly: ``$ownerProofInputReadinessCanPublishPublicly``")
$lines.Add("- owner proof input can close release issue: ``$ownerProofInputReadinessCanCloseReleaseIssue``")
$lines.Add("- owner proof input validation: ``$ownerProofInputReadinessValidationState``")
$lines.Add("- owner proof input validation is valid: ``$ownerProofInputReadinessIsValid``")
$lines.Add("- owner proof input validation failed blocker count: ``$ownerProofInputReadinessValidationFailedBlockerCount``")
$lines.Add("- owner proof input validation can publish publicly: ``$ownerProofInputReadinessValidationCanPublishPublicly``")
$lines.Add("- owner proof input validation can close release issue: ``$ownerProofInputReadinessValidationCanCloseReleaseIssue``")
$lines.Add("- external runtime proof backfill plan: ``$externalRuntimeProofBackfillPlanState``")
$lines.Add("- external runtime proof backfill step count: ``$externalRuntimeProofBackfillStepCount``")
$lines.Add("- external runtime proof backfill can promote proof: ``$externalRuntimeProofBackfillCanPromoteRuntimeProof``")
$lines.Add("- external runtime proof collection package: ``$externalRuntimeProofCollectionPackageState``")
$lines.Add("- external runtime proof collection package step count: ``$externalRuntimeProofCollectionPackageStepCount``")
$lines.Add("- external runtime proof collection package copyable execution order count: ``$externalRuntimeProofCollectionPackageExecutionOrderCount``")
$lines.Add("- external runtime proof collection package can promote proof: ``$externalRuntimeProofCollectionPackageCanPromoteRuntimeProof``")
$lines.Add("- external runtime proof collection package can close release issue: ``$externalRuntimeProofCollectionPackageCanCloseReleaseIssue``")
$lines.Add("- external runtime proof collection package runtime execution evidence: ``$externalRuntimeProofCollectionPackageRuntimeExecutionEvidence``")
$lines.Add("- external runtime execution evidence: ``$externalIsRuntimeExecutionEvidence``")
$lines.Add("- post-publish verification state: ``$postPublishVerificationState``")
$lines.Add("- post-publish proof classification: ``$postPublishProofClassification``")
$lines.Add("- post-publish proof classification promotable: ``$postPublishProofClassificationPromotable``")
$lines.Add("- post-publish managed package: ``$postPublishManagedPackageId`` / ``$postPublishManagedPackageVersion``")
$lines.Add("- post-publish runtime package: ``$postPublishRuntimePackageId`` / ``$postPublishRuntimePackageVersion``")
$lines.Add("- post-publish managed nupkg SHA256 ready: ``$postPublishManagedNupkgSha256Ready``")
$lines.Add("- post-publish runtime nupkg SHA256 ready: ``$postPublishRuntimeNupkgSha256Ready``")
$lines.Add("- post-publish consumer project identity ready: ``$postPublishConsumerProjectIdentityReady``")
$lines.Add("- post-publish smoke command runtime key ready: ``$postPublishSmokeCommandRuntimeKeyReady``")
$lines.Add("- post-publish host metadata ready: ``$postPublishHostReady``")
$lines.Add("- post-publish commands ready: ``$postPublishCommandsReady``")
$lines.Add("- post-publish stdout summary ready: ``$postPublishStdoutSummaryReady``")
$lines.Add("- post-publish stderr summary ready: ``$postPublishStderrSummaryReady``")
$lines.Add("- post-publish stdout/stderr summary ready: ``$postPublishStdoutStderrSummaryReady``")
$lines.Add("- post-publish all log SHA256 matches: ``$postPublishAllLogSha256Matches``")
$lines.Add("- post-publish verification proof: ``$isPostPublishVerificationProof``")
$lines.Add("- post-publish verification backfill plan: ``$postPublishVerificationBackfillPlanState``")
$lines.Add("- post-publish verification backfill step count: ``$postPublishVerificationBackfillStepCount``")
$lines.Add("- post-publish verification backfill can close release issue: ``$postPublishVerificationBackfillCanCloseReleaseIssue``")
$lines.Add("- post-publish verification collection package: ``$postPublishVerificationCollectionPackageState``")
$lines.Add("- post-publish verification collection package step count: ``$postPublishVerificationCollectionPackageStepCount``")
$lines.Add("- post-publish verification collection package copyable execution order count: ``$postPublishVerificationCollectionPackageExecutionOrderCount``")
$lines.Add("- post-publish verification collection package is proof: ``$postPublishVerificationCollectionPackageProof``")
$lines.Add("- post-publish verification collection package can close release issue: ``$postPublishVerificationCollectionPackageCanCloseReleaseIssue``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- real Linux runner proof: ``$isRealLinuxRunnerProof``")
$lines.Add("")
$lines.Add("## One-Screen Release Hold Checklist")
$lines.Add("")
$lines.Add("This section mirrors ``owner-release-execution-package``. It is owner-facing hold guidance only; every item remains non-publishing and cannot close the release issue without real validator proof.")
$lines.Add("")
$lines.Add("| ID | Owner-visible blocker | Current state | Owner next action | Validator command | Required real inputs | Cannot use |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in $oneScreenReleaseHoldChecklist) {
  $requiredInputs = @($item.requiredRealInputs) -join "<br/>"
  $cannotUse = @($item.cannotUse) -join "<br/>"
  $lines.Add("| ``$($item.id)`` | $($item.ownerVisibleBlocker.Replace("|", "\|")) | $($item.currentState.Replace("|", "\|")) | $($item.ownerNextAction.Replace("|", "\|")) | ``$($item.validatorCommand)`` | $($requiredInputs.Replace("|", "\|")) | $($cannotUse.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Compatible Host Collection Bundle Execution")
$lines.Add("")
$lines.Add("This section mirrors the collection bundle for promotion owners. It is executable guidance only; it is not runtime proof, publication approval, or package push.")
$lines.Add("")
$lines.Add("### Operator Quick Start")
$lines.Add("")
foreach ($item in $compatibleHostCollectionBundleOperatorQuickStart) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("### Preflight Checklist")
$lines.Add("")
$lines.Add("| ID | Required input | Why |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $compatibleHostCollectionBundlePreflightChecklist) {
  $lines.Add("| ``$($item.id)`` | $($item.required.Replace("|", "\|")) | $($item.why.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("### Copyable Execution Order")
$lines.Add("")
$lines.Add('```powershell')
foreach ($command in $compatibleHostCollectionBundleCopyableExecutionOrder) {
  $lines.Add($command)
}
$lines.Add('```')
$lines.Add("")
$lines.Add("### Owner Input Artifacts")
$lines.Add("")
$lines.Add("| ID | Path | State | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($artifact in $compatibleHostCollectionBundleOwnerInputArtifacts) {
  $lines.Add("| ``$($artifact.id)`` | ``$($artifact.path)`` | $($artifact.state.Replace("|", "\|")) | $($artifact.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Promotion Checklist")
$lines.Add("")
$lines.Add("| ID | Current status | Required evidence | Owner action | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($item in $promotionItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.currentStatus)`` | ``$($item.requiredEvidence)`` | $($item.ownerAction.Replace("|", "\|")) | $($item.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Channel Options")
$lines.Add("")
$lines.Add("| Channel | Preflight | Publish placeholder | Rollback | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($channel in $channels) {
  $lines.Add("| ``$($channel.id)`` | $($channel.preflight.Replace("|", "\|")) | ``$($channel.publishCommandPlaceholder)`` | $($channel.rollback.Replace("|", "\|")) | $($channel.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Post-Publish Verification")
$lines.Add("")
foreach ($item in $postPublishVerification) {
  $lines.Add("- [ ] $item")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release promotion issue record written to $jsonPath"
Write-Host "Release promotion issue record written to $markdownPath"
