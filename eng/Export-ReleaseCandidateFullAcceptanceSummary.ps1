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

function Get-ArrayCount {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  if ($null -eq $Object -or -not ($Object.PSObject.Properties.Name -contains $Name) -or $null -eq $Object.$Name) {
    return 0
  }

  return @($Object.$Name).Count
}

function New-AcceptanceItem {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [string]$State,
    [AllowNull()][object]$Passed,
    [AllowNull()][object]$RequiredForOwnerHandoff,
    [string]$Boundary,
    [string]$NextAction
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    artifact = $Artifact
    state = $State
    passed = [bool]$Passed
    requiredForOwnerHandoff = [bool]$RequiredForOwnerHandoff
    boundary = $Boundary
    nextAction = $NextAction
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$releasePackageProof = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$docsPublishReadiness = Read-JsonOrNull "artifacts\final-release\docs-publish-readiness-bundle.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$promotionIssue = Read-JsonOrNull "artifacts\final-release\release-promotion-issue-record.json"
$freezeSummary = Read-JsonOrNull "artifacts\release\release-candidate-freeze-summary.json"
$freezeChecklist = Read-JsonOrNull "artifacts\release\release-candidate-freeze-checklist.json"
$freezeValidation = Read-JsonOrNull "artifacts\release\release-candidate-freeze-validation.json"
$ownerCommandPlan = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan.json"
$ownerCommandPlanValidation = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan-validation.json"
$externalCollectionPackage = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-collection-package.json"
$postPublishCollectionPackage = Read-JsonOrNull "artifacts\final-release\post-publish-verification-collection-package.json"
$postPublishCleanConsumerProjectScan = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-project-scan.json"
$postPublishVerificationInputDraft = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record.input-draft.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$externalBackfillPlan = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-backfill-plan.json"
$postPublishBackfillPlan = Read-JsonOrNull "artifacts\final-release\post-publish-verification-backfill-plan.json"
$externalRuntimeValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$userAcceptance = Read-JsonOrNull "artifacts\user-acceptance\sample-smoke-catalog.json"

$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewNativeAssetCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "nativeAssetCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$finalPackageReviewReadyForOwner = ([string]::Equals($finalPackageReviewState, "owner-review-required", [System.StringComparison]::Ordinal)) -and ($finalPackageReviewPackageCount -gt 0) -and ($finalPackageReviewNativeAssetCount -gt 0) -and (-not $finalPackageReviewCanUseAsPublicPackageProof)

$releasePackageProofState = [string](Get-PropertyOrDefault -Object $releasePackageProof -Name "proofState" -DefaultValue "missing-release-package-proof-bundle")
$releasePackageCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $releasePackageProof -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$releasePackageIsRuntimeExecutionProof = [bool](Get-PropertyOrDefault -Object $releasePackageProof -Name "isRuntimeExecutionProof" -DefaultValue $false)
$releasePackageReadyForOwner = ([string]::Equals($releasePackageProofState, "package-evidence-owner-review-required", [System.StringComparison]::Ordinal)) -and (-not $releasePackageCanUseAsPublicPackageProof) -and (-not $releasePackageIsRuntimeExecutionProof)

$docsReadinessState = [string](Get-PropertyOrDefault -Object $docsPublishReadiness -Name "readinessState" -DefaultValue "missing-docs-publish-readiness-bundle")
$docsArticleCount = [int](Get-PropertyOrDefault -Object $docsPublishReadiness -Name "articleCount" -DefaultValue 0)
$canPublishDocsExternally = [bool](Get-PropertyOrDefault -Object $docsPublishReadiness -Name "canPublishDocsExternally" -DefaultValue $false)
$docsReadyForOwner = ([string]::Equals($docsReadinessState, "ready-for-owner-review", [System.StringComparison]::Ordinal)) -and ($docsArticleCount -ge 30) -and (-not $canPublishDocsExternally)

$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$releaseEvidenceComplete = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isReleaseEvidenceComplete" -DefaultValue $false)
$releaseEvidenceCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canCloseReleaseIssue" -DefaultValue $false)
$releaseEvidenceCanPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canPublishPublicly" -DefaultValue $false)

$promotionStatus = [string](Get-PropertyOrDefault -Object $promotionIssue -Name "overallStatus" -DefaultValue ([string](Get-PropertyOrDefault -Object $promotionIssue -Name "promotionState" -DefaultValue "missing-release-promotion-issue-record")))
$promotionCanClose = [bool](Get-PropertyOrDefault -Object $promotionIssue -Name "canCloseReleaseIssue" -DefaultValue $false)
$promotionPerformsPublish = [bool](Get-PropertyOrDefault -Object $promotionIssue -Name "performsPublish" -DefaultValue $true)
$promotionBlockingIssueCount = [int](Get-PropertyOrDefault -Object $promotionIssue -Name "blockingIssueCount" -DefaultValue 0)

$freezeState = [string](Get-PropertyOrDefault -Object $freezeSummary -Name "freezeState" -DefaultValue "missing-release-candidate-freeze-summary")
$freezeBlockingItemCount = [int](Get-PropertyOrDefault -Object $freezeSummary -Name "blockingItemCount" -DefaultValue -1)
$freezeCanClose = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "canCloseReleaseIssue" -DefaultValue $false)
$freezePerformsPublish = [bool](Get-PropertyOrDefault -Object $freezeSummary -Name "performsPublish" -DefaultValue $true)
$freezeChecklistState = [string](Get-PropertyOrDefault -Object $freezeChecklist -Name "freezeState" -DefaultValue "missing-release-candidate-freeze-checklist")
$freezeChecklistCanClose = [bool](Get-PropertyOrDefault -Object $freezeChecklist -Name "canCloseReleaseIssue" -DefaultValue $false)
$freezeValidationState = [string](Get-PropertyOrDefault -Object $freezeValidation -Name "validationState" -DefaultValue "missing-release-candidate-freeze-validation")
$freezeValidationFailedCount = [int](Get-PropertyOrDefault -Object $freezeValidation -Name "failedValidationItemCount" -DefaultValue -1)

$ownerPlanState = [string](Get-PropertyOrDefault -Object $ownerCommandPlan -Name "planState" -DefaultValue "missing-owner-authorized-publish-command-plan")
$ownerPlanPerformsPublish = [bool](Get-PropertyOrDefault -Object $ownerCommandPlan -Name "performsPublish" -DefaultValue $true)
$ownerPlanCanMaterialize = [bool](Get-PropertyOrDefault -Object $ownerCommandPlan -Name "canMaterializeExecutableCommands" -DefaultValue $true)
$ownerPlanValidationState = [string](Get-PropertyOrDefault -Object $ownerCommandPlanValidation -Name "validationState" -DefaultValue "missing-owner-command-plan-validation")
$ownerPlanValidationFailedCount = [int](Get-PropertyOrDefault -Object $ownerCommandPlanValidation -Name "failedValidationItemCount" -DefaultValue -1)

$externalCollectionState = [string](Get-PropertyOrDefault -Object $externalCollectionPackage -Name "packageState" -DefaultValue "missing-external-runtime-proof-collection-package")
$externalCollectionStepCount = Get-ArrayCount -Object $externalCollectionPackage -Name "collectionSteps"
$externalCollectionExecutionOrderCount = Get-ArrayCount -Object $externalCollectionPackage -Name "copyableExecutionOrder"
$externalCollectionCanPromote = [bool](Get-PropertyOrDefault -Object $externalCollectionPackage -Name "canPromoteRuntimeProof" -DefaultValue $true)
$externalCollectionCanClose = [bool](Get-PropertyOrDefault -Object $externalCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue $true)
$externalCollectionIsRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalCollectionPackage -Name "isRuntimeExecutionEvidence" -DefaultValue $true)

$postPublishCollectionState = [string](Get-PropertyOrDefault -Object $postPublishCollectionPackage -Name "packageState" -DefaultValue "missing-post-publish-verification-collection-package")
$postPublishCollectionStepCount = Get-ArrayCount -Object $postPublishCollectionPackage -Name "collectionSteps"
$postPublishCollectionExecutionOrderCount = Get-ArrayCount -Object $postPublishCollectionPackage -Name "copyableExecutionOrder"
$postPublishCollectionIsProof = [bool](Get-PropertyOrDefault -Object $postPublishCollectionPackage -Name "isPostPublishVerificationProof" -DefaultValue $true)
$postPublishCollectionCanClose = [bool](Get-PropertyOrDefault -Object $postPublishCollectionPackage -Name "canCloseReleaseIssue" -DefaultValue $true)
$postPublishCleanConsumerProjectScanState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "scanState" -DefaultValue "missing-post-publish-clean-consumer-project-scan")))
$postPublishCleanConsumerProjectScanPassed = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanPassed" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "scanPassed" -DefaultValue $false)))
$postPublishCleanConsumerProjectScanCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "canCloseReleaseIssue" -DefaultValue $false)))
$postPublishCleanConsumerProjectScanIsProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishCleanConsumerProjectScanIsProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishCleanConsumerProjectScan -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$postPublishVerificationInputDraftKind = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftKind" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "recordKind" -DefaultValue "missing-post-publish-verification-record-input-draft")))
$postPublishVerificationInputDraftOnly = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftOnly" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "inputDraftOnly" -DefaultValue $false)))
$postPublishVerificationInputDraftIsProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftIsProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$postPublishVerificationInputDraftCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationInputDraftCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublishVerificationInputDraft -Name "canCloseReleaseIssue" -DefaultValue $false)))
$releaseClosePreflightState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")))
$releaseClosePreflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightFailedItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)))
$releaseClosePreflightCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightCanCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)))
$releaseClosePreflightPerformsPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseClosePreflightPerformsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "performsPublish" -DefaultValue $true)))

$externalBackfillState = [string](Get-PropertyOrDefault -Object $externalBackfillPlan -Name "planState" -DefaultValue "missing-external-runtime-proof-backfill-plan")
$externalBackfillStepCount = Get-ArrayCount -Object $externalBackfillPlan -Name "backfillSteps"
$externalBackfillCanPromote = [bool](Get-PropertyOrDefault -Object $externalBackfillPlan -Name "canPromoteRuntimeProof" -DefaultValue $true)
$postPublishBackfillState = [string](Get-PropertyOrDefault -Object $postPublishBackfillPlan -Name "planState" -DefaultValue "missing-post-publish-verification-backfill-plan")
$postPublishBackfillStepCount = Get-ArrayCount -Object $postPublishBackfillPlan -Name "backfillSteps"
$postPublishBackfillCanClose = [bool](Get-PropertyOrDefault -Object $postPublishBackfillPlan -Name "canCloseReleaseIssue" -DefaultValue $true)

$externalValidationState = [string](Get-PropertyOrDefault -Object $externalRuntimeValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalIsRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeValidation -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$postPublishValidationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishIsProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)

$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)
$userAcceptanceItemCount = [int](Get-PropertyOrDefault -Object $userAcceptance -Name "itemCount" -DefaultValue 0)
$userAcceptanceMissingItemCount = [int](Get-PropertyOrDefault -Object $userAcceptance -Name "missingItemCount" -DefaultValue -1)

$items = @(
  New-AcceptanceItem -Id "final-package-review-bundle" -Title "Final package review bundle" -Artifact "artifacts/final-release/final-package-review-bundle.json" -State "$finalPackageReviewState; packageCount=$finalPackageReviewPackageCount; nativeAssetCount=$finalPackageReviewNativeAssetCount; canUseAsPublicPackageProof=$finalPackageReviewCanUseAsPublicPackageProof" -Passed $finalPackageReviewReadyForOwner -RequiredForOwnerHandoff $true -Boundary "Local package inventory is owner review input only, not public package proof." -NextAction "Review package IDs, versions, SHA256, and native asset counts before owner proof collection."
  New-AcceptanceItem -Id "release-package-proof-bundle" -Title "Release package proof bundle" -Artifact "artifacts/final-release/release-package-proof-bundle.json" -State "$releasePackageProofState; canUseAsPublicPackageProof=$releasePackageCanUseAsPublicPackageProof; isRuntimeExecutionProof=$releasePackageIsRuntimeExecutionProof" -Passed $releasePackageReadyForOwner -RequiredForOwnerHandoff $true -Boundary "Package proof bundle is local/package-layout proof, not runtime execution proof or public channel proof." -NextAction "Keep public package proof blocked until owner-approved channel publication evidence exists."
  New-AcceptanceItem -Id "docs-publish-readiness-bundle" -Title "Docs publish readiness bundle" -Artifact "artifacts/final-release/docs-publish-readiness-bundle.json" -State "$docsReadinessState; articleCount=$docsArticleCount; canPublishDocsExternally=$canPublishDocsExternally" -Passed $docsReadyForOwner -RequiredForOwnerHandoff $true -Boundary "Docs readiness is not external docs publication proof." -NextAction "Owner reviews generated docs and publication channel policy before external docs publish."
  New-AcceptanceItem -Id "release-evidence-bundle" -Title "Release evidence bundle" -Artifact "artifacts/final-release/release-evidence-bundle.json" -State "$releaseEvidenceState; complete=$releaseEvidenceComplete; canPublishPublicly=$releaseEvidenceCanPublish; canCloseReleaseIssue=$releaseEvidenceCanClose" -Passed (-not $releaseEvidenceComplete -and -not $releaseEvidenceCanPublish -and -not $releaseEvidenceCanClose) -RequiredForOwnerHandoff $true -Boundary "Blocked evidence state is expected until real runtime and post-publish proof exist." -NextAction "Use this as the aggregation surface for owner review; do not override blocked fields."
  New-AcceptanceItem -Id "release-promotion-issue-record" -Title "Release promotion issue record" -Artifact "artifacts/final-release/release-promotion-issue-record.json" -State "$promotionStatus; blockingIssueCount=$promotionBlockingIssueCount; performsPublish=$promotionPerformsPublish; canCloseReleaseIssue=$promotionCanClose" -Passed (-not $promotionPerformsPublish -and -not $promotionCanClose) -RequiredForOwnerHandoff $true -Boundary "Issue record is a draft/review artifact and never publishes." -NextAction "Attach only after owner reviews all blocked proof boundaries."
  New-AcceptanceItem -Id "release-candidate-freeze" -Title "Release candidate freeze summary/checklist/validation" -Artifact "artifacts/release/release-candidate-freeze-summary.json" -State "summary=$freezeState; checklist=$freezeChecklistState; validation=$freezeValidationState; blockingItemCount=$freezeBlockingItemCount; failedValidationItemCount=$freezeValidationFailedCount" -Passed ([string]::Equals($freezeState, "blocked-freeze-owner-action-required", [System.StringComparison]::Ordinal) -and [string]::Equals($freezeChecklistState, $freezeState, [System.StringComparison]::Ordinal) -and [string]::Equals($freezeValidationState, $freezeState, [System.StringComparison]::Ordinal) -and $freezeValidationFailedCount -eq 0 -and -not $freezeCanClose -and -not $freezeChecklistCanClose -and -not $freezePerformsPublish) -RequiredForOwnerHandoff $true -Boundary "Freeze validates the blocked release-candidate state; it is not publish authorization." -NextAction "Keep freeze blocked until real proof and owner authorization are attached."
  New-AcceptanceItem -Id "owner-authorized-publish-command-plan" -Title "Owner authorized publish command plan" -Artifact "artifacts/final-release/owner-authorized-publish-command-plan.json" -State "$ownerPlanState; validation=$ownerPlanValidationState; failedValidationItemCount=$ownerPlanValidationFailedCount; performsPublish=$ownerPlanPerformsPublish; canMaterializeExecutableCommands=$ownerPlanCanMaterialize" -Passed ([string]::Equals($ownerPlanState, "blocked-owner-authorization-required", [System.StringComparison]::Ordinal) -and [string]::Equals($ownerPlanValidationState, $ownerPlanState, [System.StringComparison]::Ordinal) -and $ownerPlanValidationFailedCount -eq 0 -and -not $ownerPlanPerformsPublish -and -not $ownerPlanCanMaterialize) -RequiredForOwnerHandoff $true -Boundary "Publish commands remain placeholders until explicit owner authorization; this script never executes publish." -NextAction "Owner fills approval/decision records and reruns validators before any command materialization."
  New-AcceptanceItem -Id "external-runtime-proof-collection-package" -Title "External runtime proof collection package" -Artifact "artifacts/final-release/external-runtime-proof-collection-package.json" -State "$externalCollectionState; stepCount=$externalCollectionStepCount; executionOrderCount=$externalCollectionExecutionOrderCount; canPromoteRuntimeProof=$externalCollectionCanPromote; canCloseReleaseIssue=$externalCollectionCanClose; isRuntimeExecutionEvidence=$externalCollectionIsRuntimeExecutionEvidence" -Passed ([string]::Equals($externalCollectionState, "owner-action-required", [System.StringComparison]::Ordinal) -and $externalCollectionStepCount -ge 7 -and $externalCollectionExecutionOrderCount -ge 5 -and -not $externalCollectionCanPromote -and -not $externalCollectionCanClose -and -not $externalCollectionIsRuntimeExecutionEvidence) -RequiredForOwnerHandoff $true -Boundary "Collection package is copyable owner guidance only; it is not compatible-host runtime proof." -NextAction "Owner runs it on a compatible CUDA/TensorRT host and validates the filled real proof record."
  New-AcceptanceItem -Id "post-publish-verification-collection-package" -Title "Post publish verification collection package" -Artifact "artifacts/final-release/post-publish-verification-collection-package.json" -State "$postPublishCollectionState; stepCount=$postPublishCollectionStepCount; executionOrderCount=$postPublishCollectionExecutionOrderCount; isPostPublishVerificationProof=$postPublishCollectionIsProof; canCloseReleaseIssue=$postPublishCollectionCanClose" -Passed ([string]::Equals($postPublishCollectionState, "blocked-real-publication-required", [System.StringComparison]::Ordinal) -and $postPublishCollectionStepCount -ge 8 -and $postPublishCollectionExecutionOrderCount -ge 5 -and -not $postPublishCollectionIsProof -and -not $postPublishCollectionCanClose) -RequiredForOwnerHandoff $true -Boundary "Post-publish collection package cannot be proof until an authorized real channel publish exists." -NextAction "Use only after owner-authorized publication to collect clean-consumer post-publish proof."
  New-AcceptanceItem -Id "post-publish-clean-consumer-project-scan" -Title "Post publish clean consumer project scan" -Artifact "artifacts/final-release/post-publish-clean-consumer-project-scan.json" -State "$postPublishCleanConsumerProjectScanState; scanPassed=$postPublishCleanConsumerProjectScanPassed; isPostPublishVerificationProof=$postPublishCleanConsumerProjectScanIsProof; canCloseReleaseIssue=$postPublishCleanConsumerProjectScanCanClose" -Passed (-not $postPublishCleanConsumerProjectScanIsProof -and -not $postPublishCleanConsumerProjectScanCanClose) -RequiredForOwnerHandoff $true -Boundary "Clean consumer scan is helper evidence only and cannot close the release issue." -NextAction "Run the scan against the real external clean consumer project after owner-authorized publication."
  New-AcceptanceItem -Id "post-publish-verification-input-draft" -Title "Post publish verification input draft" -Artifact "artifacts/final-release/post-publish-verification-record.input-draft.json" -State "$postPublishVerificationInputDraftKind; inputDraftOnly=$postPublishVerificationInputDraftOnly; isPostPublishVerificationProof=$postPublishVerificationInputDraftIsProof; canCloseReleaseIssue=$postPublishVerificationInputDraftCanClose" -Passed ($postPublishVerificationInputDraftOnly -and -not $postPublishVerificationInputDraftIsProof -and -not $postPublishVerificationInputDraftCanClose) -RequiredForOwnerHandoff $true -Boundary "Input draft is a helper artifact only and cannot close the release issue." -NextAction "Use the draft as fill guidance, then validate a real post-publish verification record."
  New-AcceptanceItem -Id "release-close-preflight" -Title "Release close preflight" -Artifact "artifacts/final-release/release-close-preflight.json" -State "$releaseClosePreflightState; failedItemCount=$releaseClosePreflightFailedItemCount; canCloseReleaseIssue=$releaseClosePreflightCanClose; performsPublish=$releaseClosePreflightPerformsPublish" -Passed ([string]::Equals($releaseClosePreflightState, "blocked-real-proof-required", [System.StringComparison]::Ordinal) -and $releaseClosePreflightFailedItemCount -gt 0 -and -not $releaseClosePreflightCanClose -and -not $releaseClosePreflightPerformsPublish) -RequiredForOwnerHandoff $true -Boundary "Release close preflight aggregates missing real proof and cannot substitute proof." -NextAction "Rerun after owner authorization, external runtime proof, and real post-publish proof are attached."
  New-AcceptanceItem -Id "backfill-plans" -Title "External and post-publish backfill plans" -Artifact "artifacts/final-release/external-runtime-proof-backfill-plan.json" -State "external=$externalBackfillState stepCount=$externalBackfillStepCount canPromote=$externalBackfillCanPromote; postPublish=$postPublishBackfillState stepCount=$postPublishBackfillStepCount canClose=$postPublishBackfillCanClose" -Passed ([string]::Equals($externalBackfillState, "blocked-compatible-host-proof-required", [System.StringComparison]::Ordinal) -and $externalBackfillStepCount -ge 7 -and -not $externalBackfillCanPromote -and [string]::Equals($postPublishBackfillState, "blocked-real-post-publish-proof-required", [System.StringComparison]::Ordinal) -and $postPublishBackfillStepCount -ge 9 -and -not $postPublishBackfillCanClose) -RequiredForOwnerHandoff $true -Boundary "Backfill plans are execution guidance, not proof or release close approval." -NextAction "Use the collection packages first; keep plans as fallback detailed guidance."
  New-AcceptanceItem -Id "real-proof-boundary" -Title "Real external/post-publish proof boundary" -Artifact "artifacts/final-release/external-runtime-proof-validation.json" -State "external=$externalValidationState canPromote=$externalCanPromote isRuntimeProof=$externalIsRuntimeProof; postPublish=$postPublishValidationState canClose=$postPublishCanClose isProof=$postPublishIsProof" -Passed (-not $externalCanPromote -and -not $externalIsRuntimeProof -and -not $postPublishCanClose -and -not $postPublishIsProof) -RequiredForOwnerHandoff $true -Boundary "The current state intentionally does not claim real runtime proof or post-publish proof." -NextAction "Collect real compatible-host runtime proof, then real post-publish clean-consumer proof."
  New-AcceptanceItem -Id "stale-release-claims-audit" -Title "Stale release claims audit" -Artifact "artifacts/final-release/stale-release-claims-audit.json" -State "findingCount=$staleFindingCount" -Passed ($staleFindingCount -eq 0) -RequiredForOwnerHandoff $true -Boundary "Zero findings is a text guardrail, not proof or publication approval." -NextAction "Rerun the audit after editing release-facing artifacts."
  New-AcceptanceItem -Id "sample-surface-yolovision" -Title "Sample and YoloVision surface" -Artifact "artifacts/user-acceptance/sample-smoke-catalog.json" -State "itemCount=$userAcceptanceItemCount; missingItemCount=$userAcceptanceMissingItemCount" -Passed ($userAcceptanceItemCount -gt 0 -and $userAcceptanceMissingItemCount -eq 0) -RequiredForOwnerHandoff $false -Boundary "Asset-required Classification/YoloVision entries are not smoke passes." -NextAction "Keep YoloVision naming consistent and collect real model assets separately when needed."
)

$failedRequiredItems = @($items | Where-Object { $_.requiredForOwnerHandoff -and -not $_.passed })
$failedOptionalItems = @($items | Where-Object { -not $_.requiredForOwnerHandoff -and -not $_.passed })
$canPromoteToOwnerProofCollection = $failedRequiredItems.Count -eq 0
$canCloseReleaseIssue = $false
$canPublishPublicly = $false
$performsPublish = $false
$canUseAsPublicPackageProof = $false
$canPublishDocsExternallyForRelease = $false
$acceptanceState = if ($canPromoteToOwnerProofCollection) { "ready-for-owner-proof-collection" } else { "blocked-release-candidate-acceptance" }

$sourceEvidence = @(
  "artifacts/final-release/final-package-review-bundle.json",
  "artifacts/final-release/release-package-proof-bundle.json",
  "artifacts/final-release/docs-publish-readiness-bundle.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-promotion-issue-record.json",
  "artifacts/release/release-candidate-freeze-summary.json",
  "artifacts/release/release-candidate-freeze-checklist.json",
  "artifacts/release/release-candidate-freeze-validation.json",
  "artifacts/final-release/owner-authorized-publish-command-plan.json",
  "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
  "artifacts/final-release/external-runtime-proof-backfill-plan.json",
  "artifacts/final-release/post-publish-verification-backfill-plan.json",
  "artifacts/final-release/external-runtime-proof-collection-package.json",
  "artifacts/final-release/post-publish-verification-collection-package.json",
  "artifacts/final-release/post-publish-clean-consumer-project-scan.json",
  "artifacts/final-release/post-publish-verification-record.input-draft.json",
  "artifacts/final-release/release-close-preflight.json",
  "artifacts/final-release/external-runtime-proof-validation.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/final-release/stale-release-claims-audit.json",
  "artifacts/user-acceptance/sample-smoke-catalog.json"
)

$safetyNotes = @(
  "This full acceptance summary does not publish packages.",
  "This summary aggregates non-publish release-candidate evidence only.",
  "canCloseReleaseIssue remains false until real external runtime proof, real post-publish proof, and owner authorization are all present.",
  "canPublishPublicly remains false until release owner records an explicit authorization.",
  "Final package review, release package proof, local package inventory, docs readiness, collection package, backfill plan, template, draft, example, runbook, dependency-probe-only, and blocked-by-cuda-driver outputs cannot close the release issue.",
  "External runtime proof collection package is owner guidance only, not runtime proof.",
  "Post-publish verification collection package is owner guidance only, not post-publish proof.",
  "Post-publish clean consumer scan and input draft are helper artifacts only; they cannot close the release issue.",
  "Release close preflight aggregates real-proof gaps but cannot substitute external runtime proof, owner authorization, or post-publish verification proof.",
  "blocked-by-cuda-driver is not smoke passed.",
  "YoloVision sample asset candidates are not sample smoke passes."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-candidate-full-acceptance-summary"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  acceptanceState = $acceptanceState
  performsPublish = $performsPublish
  canPromoteToOwnerProofCollection = $canPromoteToOwnerProofCollection
  canPublishPublicly = $canPublishPublicly
  canUseAsPublicPackageProof = $canUseAsPublicPackageProof
  canPublishDocsExternally = $canPublishDocsExternallyForRelease
  canCloseReleaseIssue = $canCloseReleaseIssue
  requiredItemCount = @($items | Where-Object { $_.requiredForOwnerHandoff }).Count
  failedRequiredItemCount = $failedRequiredItems.Count
  optionalItemCount = @($items | Where-Object { -not $_.requiredForOwnerHandoff }).Count
  failedOptionalItemCount = $failedOptionalItems.Count
  finalPackageReviewState = $finalPackageReviewState
  releasePackageProofState = $releasePackageProofState
  docsPublishReadinessState = $docsReadinessState
  releaseEvidenceBundleState = $releaseEvidenceState
  promotionIssueStatus = $promotionStatus
  freezeState = $freezeState
  freezeValidationState = $freezeValidationState
  ownerCommandPlanState = $ownerPlanState
  ownerCommandPlanValidationState = $ownerPlanValidationState
  externalRuntimeProofCollectionPackageState = $externalCollectionState
  postPublishVerificationCollectionPackageState = $postPublishCollectionState
  postPublishCleanConsumerProjectScanState = $postPublishCleanConsumerProjectScanState
  postPublishVerificationInputDraftKind = $postPublishVerificationInputDraftKind
  releaseClosePreflightState = $releaseClosePreflightState
  releaseClosePreflightFailedItemCount = $releaseClosePreflightFailedItemCount
  externalRuntimeProofValidationState = $externalValidationState
  postPublishVerificationValidationState = $postPublishValidationState
  staleReleaseClaimsFindingCount = $staleFindingCount
  acceptanceItems = $items
  sourceEvidence = $sourceEvidence
  safetyNotes = $safetyNotes
  nextOwnerActions = @(
    "Run the External Runtime Proof Collection Package on a compatible CUDA/TensorRT host.",
    "Validate the filled external runtime proof record with Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof.",
    "After owner authorization and real channel publication, run the Post Publish Verification Collection Package from a clean consumer.",
    "Run the post-publish clean consumer project scan and use the input draft as fill guidance only.",
    "Validate the filled post-publish verification record with Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof.",
    "Rerun Export-ReleaseClosePreflight.ps1 to verify all close gates remain real-proof backed.",
    "Regenerate release evidence, freeze, promotion issue, owner command plan, and this full acceptance summary after real proof changes."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-candidate-full-acceptance-summary.json"
$markdownPath = Join-Path $outputRoot "release-candidate-full-acceptance-summary.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Release Candidate Full Acceptance Summary")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Linux runtime key: ``$LinuxRuntimePackageKey``")
$lines.Add("")
$lines.Add("Acceptance state: ``$acceptanceState``")
$lines.Add("")
$lines.Add("This summary aggregates non-publish release-candidate evidence. It does not publish packages, approve publication, create proof, or close the release issue.")
$lines.Add("")
$lines.Add("## Decision Fields")
$lines.Add("")
$lines.Add("- performs publish: ``$performsPublish``")
$lines.Add("- can promote to owner proof collection: ``$canPromoteToOwnerProofCollection``")
$lines.Add("- can publish publicly: ``$canPublishPublicly``")
$lines.Add("- can use as public package proof: ``$canUseAsPublicPackageProof``")
$lines.Add("- can publish docs externally: ``$canPublishDocsExternallyForRelease``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- failed required items: ``$($failedRequiredItems.Count)``")
$lines.Add("- failed optional items: ``$($failedOptionalItems.Count)``")
$lines.Add("")
$lines.Add("## Acceptance Items")
$lines.Add("")
$lines.Add("| ID | Passed | Required | State | Artifact | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($item in $items) {
  $lines.Add("| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$($item.requiredForOwnerHandoff)`` | $(ConvertTo-MarkdownCell $item.state) | ``$(ConvertTo-MarkdownCell $item.artifact)`` | $(ConvertTo-MarkdownCell $item.boundary) |")
}

$lines.Add("")
$lines.Add("## Source Evidence")
$lines.Add("")
foreach ($source in $sourceEvidence) {
  $lines.Add("- ``$source``")
}

$lines.Add("")
$lines.Add("## Next Owner Actions")
$lines.Add("")
foreach ($action in $record.nextOwnerActions) {
  $lines.Add("- $action")
}

$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release candidate full acceptance summary written to $jsonPath"
Write-Host "Release candidate full acceptance summary written to $markdownPath"
Write-Host "AcceptanceState=$acceptanceState FailedRequiredItemCount=$($failedRequiredItems.Count) PerformsPublish=False CanCloseReleaseIssue=False"
