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

function Normalize-PostPublishRequiredEvidence {
  param([AllowNull()][object]$Evidence)

  $items = @($Evidence | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
  foreach ($requiredField in @("noLocalPackageSource", "noLocalNupkgPackageReference")) {
    if ($items -notcontains $requiredField) {
      $items += $requiredField
    }
  }

  return @($items)
}

function New-ChecklistItem {
  param(
    [string]$Id,
    [string]$Title,
    [string]$RequiredEvidence,
    [string]$CurrentStatus,
    [string]$OwnerAction,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    requiredEvidence = $RequiredEvidence
    currentStatus = $CurrentStatus
    state = "pending-owner-action"
    ownerAction = $OwnerAction
    boundary = $Boundary
  }
}

function New-ChannelPlan {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Preflight,
    [string]$PublishPlaceholder,
    [string]$Rollback,
    [string]$PostPublishVerification,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    preflight = $Preflight
    publishPlaceholder = $PublishPlaceholder
    rollback = $Rollback
    postPublishVerification = $PostPublishVerification
    performsPublish = $false
    ownerDecision = "pending-release-owner-approval"
    boundary = $Boundary
  }
}

$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$ownerApprovalInputValidation = Read-JsonOrNull "artifacts\final-release\release-owner-approval-input-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$ownerDecisionRecord = Read-JsonOrNull "artifacts\final-release\release-owner-decision-record.json"
$promotionIssueRecord = Read-JsonOrNull "artifacts\final-release\release-promotion-issue-record.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$releasePackageProof = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$docsPublishReadiness = Read-JsonOrNull "artifacts\final-release\docs-publish-readiness-bundle.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$releaseProofReadinessSnapshot = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$ownerProofInputReadiness = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness.json"
$ownerProofInputReadinessValidation = Read-JsonOrNull "artifacts\final-release\owner-proof-input-readiness-validation.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$linuxValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$linuxRecordTemplate = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-record-template.json"
$externalRuntimeProof = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$externalRuntimeProofOwnerHandoff = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-owner-handoff.json"
$compatibleHostRunbook = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook.json"
$compatibleHostCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$postPublishVerificationRecord = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record-template.json"
$postPublishVerificationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$ownerApprovalStatus = if ($ownerApprovalInputValidation) { [string]$ownerApprovalInputValidation.overallStatus } else { "missing-owner-approval-input-validation" }
$releaseEvidenceBundleState = if ($releaseEvidenceBundle) { [string]$releaseEvidenceBundle.bundleState } else { "missing-release-evidence-bundle" }
$releaseEvidenceComplete = if ($releaseEvidenceBundle -and $releaseEvidenceBundle.PSObject.Properties.Name -contains "isReleaseEvidenceComplete") { [bool]$releaseEvidenceBundle.isReleaseEvidenceComplete } else { $false }
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewNativeAssetCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "nativeAssetCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$releasePackageProofState = if ($releasePackageProof) { [string]$releasePackageProof.proofState } else { "missing-release-package-proof-bundle" }
$canUseAsPublicPackageProof = if ($releasePackageProof -and $releasePackageProof.PSObject.Properties.Name -contains "canUseAsPublicPackageProof") { [bool]$releasePackageProof.canUseAsPublicPackageProof } else { $false }
$packageProofIsRuntimeExecutionProof = if ($releasePackageProof -and $releasePackageProof.PSObject.Properties.Name -contains "isRuntimeExecutionProof") { [bool]$releasePackageProof.isRuntimeExecutionProof } else { $false }
$docsPublishReadinessState = if ($docsPublishReadiness) { [string]$docsPublishReadiness.readinessState } else { "missing-docs-publish-readiness-bundle" }
$canPublishDocsExternally = if ($docsPublishReadiness -and $docsPublishReadiness.PSObject.Properties.Name -contains "canPublishDocsExternally") { [bool]$docsPublishReadiness.canPublishDocsExternally } else { $false }
$docsArticleCount = if ($docsPublishReadiness -and $docsPublishReadiness.PSObject.Properties.Name -contains "articleCount") { [int]$docsPublishReadiness.articleCount } else { -1 }
$ownerApprovalCanPublishPublicly = if ($ownerApprovalInputValidation -and $ownerApprovalInputValidation.PSObject.Properties.Name -contains "canPublishPublicly") { [bool]$ownerApprovalInputValidation.canPublishPublicly } else { $false }
$finalReleaseStatus = if ($finalRelease) { [string]$finalRelease.overallStatus } else { "missing-final-release-dry-run" }
$blockingIssueCount = if ($finalRelease) { [int]$finalRelease.blockingIssueCount } else { -1 }
$manualApprovalCount = if ($finalRelease) { [int]$finalRelease.manualApprovalCount } else { -1 }
$runtimeProofStatus = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofStatus") { [string]$finalRelease.runtimeProofStatus } elseif ($packageConsumer) { [string]$packageConsumer.status } else { "missing-runtime-proof" }
$runtimeProofRequiredForRelease = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofRequiredForRelease") { [bool]$finalRelease.runtimeProofRequiredForRelease } else { -not [string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase) }
$packageConsumerSmokeStatus = if ($finalRelease) { [string]$finalRelease.packageConsumerSmokeStatus } elseif ($packageConsumer) { [string]$packageConsumer.status } else { "missing-package-consumer" }
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
$staleFindingCount = if ($staleAudit) { [int]$staleAudit.findingCount } else { -1 }
$promotionIssueState = if ($promotionIssueRecord) { [string]$promotionIssueRecord.promotionState } else { "missing-release-promotion-issue-record" }
$ownerDecisionState = if ($ownerDecisionRecord) { [string]$ownerDecisionRecord.recordState } else { "missing-release-owner-decision-record" }
$linuxProofState = if ($linuxValidation) { [string]$linuxValidation.validationState } elseif ($linuxRecordTemplate) { [string]$linuxRecordTemplate.recordState } else { "missing-linux-runner-validation" }
$isRealLinuxRunnerProof = if ($linuxValidation) { [bool]$linuxValidation.isRealLinuxRunnerProof } elseif ($linuxRecordTemplate) { [bool]$linuxRecordTemplate.isRealLinuxRunnerProof } else { $false }
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
$postPublishRequiredEvidence = @(
  Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRequiredEvidence" -DefaultValue @(
    Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishRequiredEvidence" -DefaultValue @()
  )
)
$postPublishRequiredEvidence = Normalize-PostPublishRequiredEvidence -Evidence $postPublishRequiredEvidence
$postPublishRequiredEvidenceCount = if ($postPublishRequiredEvidence.Count -gt 0) {
  $postPublishRequiredEvidence.Count
}
else {
  [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRequiredEvidenceCount" -DefaultValue (
    [int](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishRequiredEvidenceCount" -DefaultValue 0)
  ))
}
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

$canExecutePublicPublish = $false
$executionState = if ($ownerApprovalCanPublishPublicly) { "blocked-publish-checklist-not-executed" } else { "blocked-owner-input-required" }

$preflightItems = @(
  New-ChecklistItem `
    -Id "release-evidence-bundle" `
    -Title "Release evidence bundle" `
    -RequiredEvidence "artifacts/final-release/release-evidence-bundle.json; artifacts/final-release/release-evidence-bundle.md" `
    -CurrentStatus ("bundleState=" + $releaseEvidenceBundleState + "; isReleaseEvidenceComplete=" + $releaseEvidenceComplete) `
    -OwnerAction "Review the evidence bundle as an aggregation layer and fix source evidence before publication." `
    -Boundary "The evidence bundle is not release owner approval and does not push packages."
  New-ChecklistItem `
    -Id "owner-approval-input" `
    -Title "Release owner approval input" `
    -RequiredEvidence "artifacts/final-release/release-owner-approval-input-record.json; artifacts/final-release/release-owner-approval-input-validation.json" `
    -CurrentStatus ("validationStatus=" + $ownerApprovalStatus + "; canPublishPublicly=" + $ownerApprovalCanPublishPublicly) `
    -OwnerAction "Fill and validate a non-template owner approval input record before public publication." `
    -Boundary "Template-only or blocked owner input cannot approve publication."
  New-ChecklistItem `
    -Id "final-package-review-bundle" `
    -Title "Final package review bundle" `
    -RequiredEvidence "artifacts/final-release/final-package-review-bundle.json; artifacts/final-release/final-package-review-bundle.md" `
    -CurrentStatus ("bundleState=" + $finalPackageReviewState + "; packageCount=" + $finalPackageReviewPackageCount + "; nativeAssetCount=" + $finalPackageReviewNativeAssetCount + "; canUseAsPublicPackageProof=" + $finalPackageReviewCanUseAsPublicPackageProof) `
    -OwnerAction "Review final managed/runtime/split-runtime nupkg identity, size, SHA256, runtime key, and native asset count before channel work." `
    -Boundary "Final package review is local package inventory only; it is not public channel proof, runtime execution proof, or publish approval."
  New-ChecklistItem `
    -Id "release-package-proof-bundle" `
    -Title "Release package proof bundle" `
    -RequiredEvidence "artifacts/final-release/release-package-proof-bundle.json; artifacts/final-release/release-package-proof-bundle.md" `
    -CurrentStatus ("proofState=" + $releasePackageProofState + "; canUseAsPublicPackageProof=" + $canUseAsPublicPackageProof + "; isRuntimeExecutionProof=" + $packageProofIsRuntimeExecutionProof) `
    -OwnerAction "Review package layout, local feed, split package, native-copy, and consumer evidence before approving any channel." `
    -Boundary "Local package proof bundle output is not public package proof and is not runtime execution proof."
  New-ChecklistItem `
    -Id "docs-publish-readiness-bundle" `
    -Title "Docs publish readiness bundle" `
    -RequiredEvidence "artifacts/final-release/docs-publish-readiness-bundle.json; artifacts/final-release/docs-publish-readiness-bundle.md" `
    -CurrentStatus ("readinessState=" + $docsPublishReadinessState + "; articleCount=" + $docsArticleCount + "; canPublishDocsExternally=" + $canPublishDocsExternally) `
    -OwnerAction "Review article count, sample-backed docs, DocFX output, external channel plan, and media/cover assets before publishing docs externally." `
    -Boundary "Docs readiness bundle output is not external publication proof."
  New-ChecklistItem `
    -Id "final-dry-run" `
    -Title "Final release dry run" `
    -RequiredEvidence "artifacts/final-release/final-release-dry-run-summary.json" `
    -CurrentStatus ("overallStatus=" + $finalReleaseStatus + "; blockingIssueCount=" + $blockingIssueCount + "; manualApprovalCount=" + $manualApprovalCount) `
    -OwnerAction "Review manual approvals and blockers before selecting a channel." `
    -Boundary "ready-needs-manual-approval is not public release approval."
  New-ChecklistItem `
    -Id "runtime-proof" `
    -Title "Full runtime proof disposition" `
    -RequiredEvidence "artifacts/package-consumer/package-consumer-validation-summary.json; artifacts/package-readiness/runtime-package-readiness-summary.json" `
    -CurrentStatus ("packageConsumerSmokeStatus=" + $packageConsumerSmokeStatus + "; runtimeProofStatus=" + $runtimeProofStatus + "; runtimeProofRequiredForRelease=" + $runtimeProofRequiredForRelease + "; ownerAction=" + $runtimeProofBlockerOwnerActionStatus + "; blockerCategory=" + $runtimeProofBlockerCategory) `
    -OwnerAction "Rerun on a compatible CUDA host or explicitly approve a known limitation for RC only. Suggested command: $runtimeProofOwnerCommand" `
    -Boundary "blocked-by-cuda-driver, dependency-probe-only, runtime-deserialization-dependency-diagnostics, and precheck evidence are not runtime execution proof."
  New-ChecklistItem `
    -Id "external-runtime-proof-record" `
    -Title "External runtime proof record" `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-record-template.json; artifacts/final-release/external-runtime-proof-validation.json" `
    -CurrentStatus ("proofState=" + $externalRuntimeProofState + "; classification=" + $externalRuntimeProofClassification + "; runtimePackageKeyMatches=" + $externalRuntimeProofRuntimePackageKeyMatches + "; packageSourceRuntimePackageKeyMatches=" + $externalRuntimeProofPackageSourceRuntimePackageKeyMatches + "; consumerProjectIdentityReady=" + $externalRuntimeProofConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $externalRuntimeProofSmokeCommandRuntimeKeyReady + "; hostReady=" + $externalRuntimeProofHostReady + "; commandsReady=" + $externalRuntimeProofCommandsReady + "; managedNupkgSha256Ready=" + $externalRuntimeProofManagedNupkgSha256Ready + "; runtimeNupkgSha256Ready=" + $externalRuntimeProofRuntimeNupkgSha256Ready + "; logSha256FormatReady=" + $externalRuntimeProofLogSha256FormatReady + "; logSha256Matches=" + $externalRuntimeProofLogSha256Matches + "; failedProofItemCount=" + $externalRuntimeProofFailedProofItemCount + "; ownerAction=" + $externalRuntimeProofOwnerActionStatus + "; isRuntimeExecutionEvidence=" + $externalIsRuntimeExecutionEvidence + "; draftState=" + $externalRuntimeProofDraftState + "; draftManagedNupkgSha256Ready=" + $draftManagedNupkgSha256Ready + "; draftRuntimeNupkgSha256Ready=" + $draftRuntimeNupkgSha256Ready + "; draftSmokeLogSha256Ready=" + $draftSmokeLogSha256Ready + "; draftNoProjectReference=" + $draftNoProjectReference + "; draftSmokeStatus=" + $draftSmokeStatus + "; compatibleHostRequired=" + $compatibleHostRequired + "; promotionBlockedReason=" + $promotionBlockedReason) `
    -OwnerAction "Fill this record only from a compatible CUDA host package-consumer smoke run, with matching runtimePackageKey, clean consumer project identity, CUDA/TensorRT/cuDNN host metadata, --runtime-package-key smoke command, nupkg SHA256 values, and the real smoke logSha256." `
    -Boundary "The external runtime proof template, runtime-key-mismatched record, missing consumer identity, missing host metadata, missing runtime-key smoke command, missing package hash, or missing-log-hash record is not runtime execution proof."
  New-ChecklistItem `
    -Id "external-runtime-proof-owner-handoff" `
    -Title "External runtime proof owner handoff" `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-owner-handoff.json; artifacts/final-release/external-runtime-proof-owner-handoff.md" `
    -CurrentStatus ("handoffState=" + $externalRuntimeProofOwnerHandoffState + "; ownerAction=" + $externalRuntimeProofOwnerHandoffOwnerActionStatus) `
    -OwnerAction "Use the handoff commands and expected artifact paths on a compatible CUDA host, then validate the filled external runtime proof record with -RequireExistingLog." `
    -Boundary "The handoff is a backfill guide, not runtime execution proof."
  New-ChecklistItem `
    -Id "compatible-host-runtime-proof-runbook" `
    -Title "Compatible host runtime proof runbook" `
    -RequiredEvidence "artifacts/final-release/compatible-host-runtime-proof-runbook.json; artifacts/final-release/compatible-host-runtime-proof-runbook.md" `
    -CurrentStatus ("runbookState=" + $compatibleHostRunbookState + "; compatibleHostRequired=" + $compatibleHostRunbookCompatibleHostRequired + "; canPromoteRuntimeProof=" + $compatibleHostRunbookCanPromoteRuntimeProof + "; isRuntimeExecutionEvidence=" + $compatibleHostRunbookRuntimeExecutionEvidence + "; performsPublish=" + $compatibleHostRunbookPerformsPublish + "; approvesPublicRelease=" + $compatibleHostRunbookApprovesPublicRelease + "; promotionBlockedReason=" + $compatibleHostRunbookPromotionBlockedReason) `
    -OwnerAction "Follow the runbook on a compatible CUDA host to produce a real external-runtime-proof-record.json, then rerun the -FailOnNotProof validation." `
    -Boundary "The runbook is a command guide and field checklist; it is not runtime execution proof, publication approval, or package push."
  New-ChecklistItem `
    -Id "compatible-host-runtime-proof-collection-bundle" `
    -Title "Compatible host runtime proof collection bundle" `
    -RequiredEvidence "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json; artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md" `
    -CurrentStatus ("collectionState=" + $compatibleHostCollectionBundleState + "; compatibleHostRequired=" + $compatibleHostCollectionBundleCompatibleHostRequired + "; canPromoteRuntimeProof=" + $compatibleHostCollectionBundleCanPromoteRuntimeProof + "; isRuntimeExecutionEvidence=" + $compatibleHostCollectionBundleRuntimeExecutionEvidence + "; performsPublish=" + $compatibleHostCollectionBundlePerformsPublish + "; approvesPublicRelease=" + $compatibleHostCollectionBundleApprovesPublicRelease + "; quickStart=" + $compatibleHostCollectionBundleQuickStartCount + "; preflight=" + $compatibleHostCollectionBundlePreflightCount + "; copyableExecutionOrder=" + $compatibleHostCollectionBundleExecutionOrderCount + "; promotionBlockedReason=" + $compatibleHostCollectionBundlePromotionBlockedReason) `
    -OwnerAction "Use the collection bundle on a compatible CUDA host, fill the real proof record, and validate it with: $compatibleHostCollectionBundleValidateFilledRecordCommand" `
    -Boundary "The collection bundle is a command collection and hash checklist; it is not runtime execution proof, publication approval, or package push."
  New-ChecklistItem `
    -Id "linux-runner-proof" `
    -Title "Linux runner proof disposition" `
    -RequiredEvidence "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json" `
    -CurrentStatus ("validationState=" + $linuxProofState + "; isRealLinuxRunnerProof=" + $isRealLinuxRunnerProof) `
    -OwnerAction "Attach real Linux runner evidence or keep Linux package promotion blocked." `
    -Boundary "template-only, handoff-only, and Windows-generated records are not Linux runner proof."
  New-ChecklistItem `
    -Id "callback-proof" `
    -Title "Real callback runtime proof disposition" `
    -RequiredEvidence "artifacts/release-candidate/release-candidate-readiness-summary.json; docs/articles/zh-cn/real-callback-runtime-evidence-schema.md" `
    -CurrentStatus ("realCallbackRuntimeProof=" + $realCallbackProof) `
    -OwnerAction "Require InvocationCount>0 evidence or approve the known callback proof limitation for RC." `
    -Boundary "precheck, schema-ready, and attempted-no-invocation are not callback runtime proof."
  New-ChecklistItem `
    -Id "stale-release-claims" `
    -Title "Stale release claims audit" `
    -RequiredEvidence "artifacts/final-release/stale-release-claims-audit.json" `
    -CurrentStatus ("findingCount=" + $staleFindingCount) `
    -OwnerAction "Keep findingCount at zero after editing release notes, issue body, and channel instructions." `
    -Boundary "The stale-claim audit is a guardrail, not a publication approval."
  New-ChecklistItem `
    -Id "promotion-issue" `
    -Title "Release promotion issue record" `
    -RequiredEvidence "artifacts/final-release/release-promotion-issue-record.json" `
    -CurrentStatus ("promotionState=" + $promotionIssueState + "; ownerDecisionState=" + $ownerDecisionState) `
    -OwnerAction "Use the generated issue body as review material and record the final owner decision separately." `
    -Boundary "The promotion issue record does not publish packages."
  New-ChecklistItem `
    -Id "post-publish-verification-record" `
    -Title "Post-publish verification record" `
    -RequiredEvidence "artifacts/final-release/post-publish-verification-record-template.json; artifacts/final-release/post-publish-verification-validation.json" `
    -CurrentStatus ("verificationState=" + $postPublishVerificationState + "; classification=" + $postPublishProofClassification + "; promotable=" + $postPublishProofClassificationPromotable + "; managedPackage=" + $postPublishManagedPackageId + "/" + $postPublishManagedPackageVersion + "; runtimePackage=" + $postPublishRuntimePackageId + "/" + $postPublishRuntimePackageVersion + "; managedNupkgSha256Ready=" + $postPublishManagedNupkgSha256Ready + "; runtimeNupkgSha256Ready=" + $postPublishRuntimeNupkgSha256Ready + "; consumerProjectIdentityReady=" + $postPublishConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $postPublishSmokeCommandRuntimeKeyReady + "; hostReady=" + $postPublishHostReady + "; commandsReady=" + $postPublishCommandsReady + "; stdoutSummaryReady=" + $postPublishStdoutSummaryReady + "; stderrSummaryReady=" + $postPublishStderrSummaryReady + "; stdoutStderrSummaryReady=" + $postPublishStdoutStderrSummaryReady + "; allLogSha256Matches=" + $postPublishAllLogSha256Matches + "; isPostPublishVerificationProof=" + $isPostPublishVerificationProof + "; canCloseReleaseIssue=" + $canCloseReleaseIssue) `
    -OwnerAction "Fill after a real channel publish using package id/version/URL, downloaded nupkg SHA256, clean consumer project identity, compatible host metadata, --runtime-package-key smoke command, stdout/stderr summaries, and a clean consumer with no ProjectReference." `
    -Boundary "The post-publish verification template or missing package identity/hash/consumer/host/smoke-command/log-summary record does not publish packages and is not proof."
)

$channelPlans = @(
  New-ChannelPlan `
    -Id "local-feed" `
    -Title "Local release candidate feed" `
    -Preflight "Run local feed consumer validation; verify no ProjectReference and native assets copied." `
    -PublishPlaceholder "No public push; local feed is a dry-run consumer path." `
    -Rollback "Delete the local feed folder and rebuild packages." `
    -PostPublishVerification "Not public; repeat clean consumer restore/build/native-copy locally." `
    -Boundary "Local feed validation is not nuget.org or GitHub Packages publication."
  New-ChannelPlan `
    -Id "nuget-org" `
    -Title "nuget.org" `
    -Preflight "Verify owner approval, package ownership, NUGET_API_KEY scope, signing policy, package size, and NVIDIA redistribution approval." `
    -PublishPlaceholder "dotnet nuget push <package>.nupkg --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json" `
    -Rollback "Unlist the version or publish a corrected version according to owner policy; nuget.org versions cannot be overwritten." `
    -PostPublishVerification "Restore a clean consumer from nuget.org without ProjectReference and verify package source plus native copy." `
    -Boundary "This checklist does not execute dotnet nuget push."
  New-ChannelPlan `
    -Id "github-packages" `
    -Title "GitHub Packages" `
    -Preflight "Verify owner approval, source URL, package owner, token package:write permission, retention policy, and restore credentials." `
    -PublishPlaceholder "dotnet nuget push <package>.nupkg --api-key <GITHUB_TOKEN> --source <github-packages-source>" `
    -Rollback "Delete, deprecate, or supersede the package version according to organization permissions and retention policy." `
    -PostPublishVerification "Restore a clean consumer from GitHub Packages with explicit credentials and verify native copy." `
    -Boundary "This checklist does not upload to GitHub Packages."
  New-ChannelPlan `
    -Id "github-release-assets" `
    -Title "GitHub Release assets" `
    -Preflight "Verify owner approval, tag, release notes, asset SHA256, and instructions for adding a local package source." `
    -PublishPlaceholder "gh release upload <tag> <package>.nupkg <package>.sha256" `
    -Rollback "Delete assets or publish corrected release notes and replacement assets." `
    -PostPublishVerification "Download release assets into a local package source before clean consumer restore." `
    -Boundary "Release assets are not a NuGet restore source by themselves."
  New-ChannelPlan `
    -Id "private-feed" `
    -Title "Private feed" `
    -Preflight "Verify owner approval, access control, retention, package size limits, and consumer NuGet.config." `
    -PublishPlaceholder "dotnet nuget push <package>.nupkg --source <private-feed-source>" `
    -Rollback "Follow the organization feed delete, deprecate, or supersede policy." `
    -PostPublishVerification "Restore a clean consumer from the private feed and record package source, native copy, and dependency probe." `
    -Boundary "Private feed success does not prove public channel success."
)

$postPublishVerification = @(
  [pscustomobject]@{ id = "clean-consumer"; requiredEvidence = "fresh directory with no ProjectReference"; status = "pending"; boundary = "Existing repo build is not clean consumer proof." }
  [pscustomobject]@{ id = "clean-consumer-project-identity"; requiredEvidence = "consumerProjectName and consumerProjectPath pointing to the clean consumer .csproj"; status = "pending"; boundary = "A directory name alone is not enough to audit the consumer project." }
  [pscustomobject]@{ id = "managed-package-source"; requiredEvidence = "restore log showing managed package from selected channel"; status = "pending"; boundary = "ProjectReference or local bin output is not channel proof." }
  [pscustomobject]@{ id = "runtime-package-source"; requiredEvidence = "restore log showing runtime package from selected channel"; status = "pending"; boundary = "GitHub Release assets require an explicit local source after download." }
  [pscustomobject]@{ id = "host-runtime-metadata"; requiredEvidence = "OS, GPU, driver, CUDA runtime, TensorRT runtime/line, and cuDNN version from the smoke host"; status = "pending"; boundary = "A passed flag without host metadata is not reproducible proof." }
  [pscustomobject]@{ id = "native-assets-copied"; requiredEvidence = "consumer output directory native bridge and vendor runtime listing"; status = "pending"; boundary = "Package restore alone is not native-copy proof." }
  [pscustomobject]@{ id = "dependency-probe"; requiredEvidence = "DependencyProbe BridgeInitialized output"; status = "pending"; boundary = "Dependency probe is not runtime execution proof." }
  [pscustomobject]@{ id = "runtime-key-smoke-command"; requiredEvidence = "smokeCommand containing --runtime-package-key $RuntimePackageKey"; status = "pending"; boundary = "A smoke command without the release runtime package key cannot prove the selected runtime package." }
  [pscustomobject]@{ id = "stdout-stderr-summary"; requiredEvidence = "stdoutSummary and stderrSummary fields from restore/build/smoke execution"; status = "pending"; boundary = "Raw logs need short summaries for release issue review; summaries do not replace SHA256 logs." }
  [pscustomobject]@{ id = "compatible-host-smoke"; requiredEvidence = "CUDA-compatible host smoke log"; status = "pending"; boundary = "blocked-by-cuda-driver is not smoke passed." }
  [pscustomobject]@{ id = "runtime-proof-owner-action"; requiredEvidence = "runtimeProofBlockerOwnerAction from runtime-package-readiness-summary.json"; status = $runtimeProofBlockerOwnerActionStatus; boundary = "Owner action guidance is not package-consumer-runtime proof." }
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-publish-execution-checklist"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  executionState = $executionState
  canExecutePublicPublish = $canExecutePublicPublish
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $canCloseReleaseIssue
  ownerActionStatus = "owner-action-required"
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isRealModelRuntimeProof = $false
  requiresHumanOwner = $true
  ownerApprovalInputValidationStatus = $ownerApprovalStatus
  releaseEvidenceBundleState = $releaseEvidenceBundleState
  isReleaseEvidenceComplete = $releaseEvidenceComplete
  releasePackageProofState = $releasePackageProofState
  canUseAsPublicPackageProof = $canUseAsPublicPackageProof
  packageProofIsRuntimeExecutionProof = $packageProofIsRuntimeExecutionProof
  docsPublishReadinessState = $docsPublishReadinessState
  canPublishDocsExternally = $canPublishDocsExternally
  docsArticleCount = $docsArticleCount
  ownerApprovalCanPublishPublicly = $ownerApprovalCanPublishPublicly
  finalReleaseDryRunStatus = $finalReleaseStatus
  blockingIssueCount = $blockingIssueCount
  manualApprovalCount = $manualApprovalCount
  packageConsumerSmokeStatus = $packageConsumerSmokeStatus
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  runtimeProofBlockerOwnerActionStatus = $runtimeProofBlockerOwnerActionStatus
  runtimeProofBlockerCategory = $runtimeProofBlockerCategory
  runtimeProofOwnerCommand = $runtimeProofOwnerCommand
  realCallbackRuntimeProof = $realCallbackProof
  linuxRunnerValidationState = $linuxProofState
  isRealLinuxRunnerProof = $isRealLinuxRunnerProof
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
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidenceCount
  isPostPublishVerificationProof = $isPostPublishVerificationProof
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
  staleReleaseClaimsFindingCount = $staleFindingCount
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewNativeAssetCount = $finalPackageReviewNativeAssetCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  preflightItems = $preflightItems
  channelPlans = $channelPlans
  postPublishVerification = $postPublishVerification
  sourceEvidence = @(
    "artifacts/final-release/release-owner-approval-input-validation.json",
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
    "artifacts/final-release/final-release-dry-run-summary.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/release-promotion-issue-record.json",
    "artifacts/final-release/stale-release-claims-audit.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
    "artifacts/final-release/external-runtime-proof-record-template.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-owner-handoff.json",
    "artifacts/final-release/compatible-host-runtime-proof-runbook.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/release-candidate-package-inventory.json",
    "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json",
    "artifacts/final-release/post-publish-verification-record-template.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/real-model-and-package-proof-input-package.json"
  )
  safetyNotes = @(
    "This checklist does not push packages.",
    "The release evidence bundle is an aggregation layer, not publication approval.",
    "The final package review bundle is local package inventory, not public package proof.",
    "The release package proof bundle is local/package evidence, not public package proof.",
    "The docs publish readiness bundle is local readiness, not external publication proof.",
    "canExecutePublicPublish=false until a real owner approval input record validates cleanly.",
    "Publish placeholders are commands for human review, not commands executed by this script.",
    "blocked-by-cuda-driver is not smoke passed.",
    "runtime-deserialization-dependency-diagnostics and runtime proof blocker owner action are not runtime execution proof.",
    "DependencyProbe output is not runtime execution proof.",
    "template-only and handoff-only are not Linux runner proof.",
    "External runtime proof templates, runtime-key-mismatched records, missing consumer identity, missing host metadata, missing --runtime-package-key smoke commands, missing package hashes, missing-log-hash records, and post-publish verification templates are not proof.",
    "External runtime proof owner handoff is a backfill guide, not runtime proof.",
    "Compatible host runtime proof runbook is a command guide, not runtime proof, public approval, or package push.",
    "Compatible host runtime proof collection bundle is an external execution package, not runtime proof, publication approval, or package push.",
    "compatibleHostRuntimeProofCollectionBundleOwnerInputArtifacts are owner inputs only; package inventory, final package review, release package proof, and local feed consumer summaries cannot promote runtime proof or post-publish proof.",
    "Release proof readiness snapshot is a compact status view only; it cannot publish, close the release issue, or substitute real proof records.",
    "Post-publish verification requires clean consumer project identity, compatible host CUDA/TensorRT/cuDNN metadata, --runtime-package-key smoke command, stdout/stderr summaries, and SHA256-backed logs before it can close a release issue.",
    "IsRealCallbackRuntimeProof=false is not callback proof complete."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-publish-execution-checklist.json"
$markdownPath = Join-Path $outputRoot "release-publish-execution-checklist.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Publish Execution Checklist")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Linux runtime key: ``$LinuxRuntimePackageKey``")
$lines.Add("")
$lines.Add("Execution state: ``$executionState``")
$lines.Add("")
$lines.Add("Can execute public publish: ``false``")
$lines.Add("")
$lines.Add("This checklist does not push packages. It only gathers owner-gated preflight, channel placeholder, rollback, and post-publish verification items.")
$lines.Add("")
$lines.Add("## Evidence Snapshot")
$lines.Add("")
$lines.Add("- owner approval input validation: ``$ownerApprovalStatus``")
$lines.Add("- release evidence bundle: ``$releaseEvidenceBundleState``")
$lines.Add("- release evidence complete: ``$releaseEvidenceComplete``")
$lines.Add("- release package proof bundle: ``$releasePackageProofState``")
$lines.Add("- can use as public package proof: ``$canUseAsPublicPackageProof``")
$lines.Add("- package proof is runtime execution proof: ``$packageProofIsRuntimeExecutionProof``")
$lines.Add("- docs publish readiness bundle: ``$docsPublishReadinessState``")
$lines.Add("- docs article count: $docsArticleCount")
$lines.Add("- can publish docs externally: ``$canPublishDocsExternally``")
$lines.Add("- owner approval can publish publicly: ``$ownerApprovalCanPublishPublicly``")
$lines.Add("- final release dry run: ``$finalReleaseStatus``")
$lines.Add("- blocking issues: $blockingIssueCount")
$lines.Add("- manual approvals: $manualApprovalCount")
$lines.Add("- package consumer smoke: ``$packageConsumerSmokeStatus``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- runtime proof required for release: ``$runtimeProofRequiredForRelease``")
$lines.Add("- runtime proof blocker owner action: ``$runtimeProofBlockerOwnerActionStatus``")
$lines.Add("- runtime proof blocker category: ``$runtimeProofBlockerCategory``")
$lines.Add("- runtime proof suggested command: ``$runtimeProofOwnerCommand``")
$lines.Add("- real callback runtime proof: ``$realCallbackProof``")
$lines.Add("- Linux runner validation state: ``$linuxProofState``")
$lines.Add("- real Linux runner proof: ``$isRealLinuxRunnerProof``")
$lines.Add("- external runtime proof state: ``$externalRuntimeProofState``")
$lines.Add("- external runtime proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- external runtime proof runtime key matches: ``$externalRuntimeProofRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof package source runtime key matches: ``$externalRuntimeProofPackageSourceRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof consumer project identity ready: ``$externalRuntimeProofConsumerProjectIdentityReady``")
$lines.Add("- external runtime proof smoke command runtime key ready: ``$externalRuntimeProofSmokeCommandRuntimeKeyReady``")
$lines.Add("- external runtime proof host metadata ready: ``$externalRuntimeProofHostReady``")
$lines.Add("- external runtime proof commands ready: ``$externalRuntimeProofCommandsReady``")
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
$lines.Add("- external runtime execution evidence: ``$externalIsRuntimeExecutionEvidence``")
$lines.Add("- post-publish verification state: ``$postPublishVerificationState``")
$lines.Add("- post-publish consumer project identity ready: ``$postPublishConsumerProjectIdentityReady``")
$lines.Add("- post-publish smoke command runtime key ready: ``$postPublishSmokeCommandRuntimeKeyReady``")
$lines.Add("- post-publish host metadata ready: ``$postPublishHostReady``")
$lines.Add("- post-publish commands ready: ``$postPublishCommandsReady``")
$lines.Add("- post-publish stdout summary ready: ``$postPublishStdoutSummaryReady``")
$lines.Add("- post-publish stderr summary ready: ``$postPublishStderrSummaryReady``")
$lines.Add("- post-publish stdout/stderr summary ready: ``$postPublishStdoutStderrSummaryReady``")
$lines.Add("- post-publish all log SHA256 matches: ``$postPublishAllLogSha256Matches``")
$lines.Add("- post-publish verification proof: ``$isPostPublishVerificationProof``")
$lines.Add("- post-publish required evidence count: ``$postPublishRequiredEvidenceCount``")
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
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- stale release claim findings: $staleFindingCount")
$lines.Add("")
$lines.Add("## One-Screen Release Hold Checklist")
$lines.Add("")
$lines.Add("This section mirrors ``owner-release-execution-package``. It is operator guidance only; it does not run public publish and cannot close the release issue.")
$lines.Add("")
$lines.Add("| ID | Owner-visible blocker | Current state | Owner next action | Validator command | Required real inputs | Cannot use |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in $oneScreenReleaseHoldChecklist) {
  $requiredInputs = @($item.requiredRealInputs) -join "<br/>"
  $cannotUse = @($item.cannotUse) -join "<br/>"
  $lines.Add("| ``$($item.id)`` | $($item.ownerVisibleBlocker.Replace("|", "\|")) | $($item.currentState.Replace("|", "\|")) | $($item.ownerNextAction.Replace("|", "\|")) | ``$($item.validatorCommand)`` | $($requiredInputs.Replace("|", "\|")) | $($cannotUse.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Compatible Host Collection Bundle Execution")
$lines.Add("")
$lines.Add("This section mirrors the collection bundle for a release operator. It is executable guidance only; it is not runtime proof, publication approval, or package push.")
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
$lines.Add("## Preflight Items")
$lines.Add("")
$lines.Add("| ID | Current status | Required evidence | Owner action | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($item in $preflightItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.currentStatus)`` | ``$($item.requiredEvidence)`` | $($item.ownerAction.Replace("|", "\|")) | $($item.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Channel Plans")
$lines.Add("")
$lines.Add("| Channel | Preflight | Publish placeholder | Rollback | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($channel in $channelPlans) {
  $lines.Add("| ``$($channel.id)`` | $($channel.preflight.Replace("|", "\|")) | ``$($channel.publishPlaceholder)`` | $($channel.rollback.Replace("|", "\|")) | $($channel.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Post-Publish Verification")
$lines.Add("")
$lines.Add("| ID | Required evidence | Status | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $postPublishVerification) {
  $lines.Add("| ``$($item.id)`` | $($item.requiredEvidence.Replace("|", "\|")) | ``$($item.status)`` | $($item.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release publish execution checklist written to $jsonPath"
Write-Host "Release publish execution checklist written to $markdownPath"
