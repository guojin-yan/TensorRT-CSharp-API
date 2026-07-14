$script:RealOwnerProofContractsCommonLoaded = $true

if (-not $script:OwnerPostPublishRealInputCommonLoaded) {
  . (Join-Path $PSScriptRoot "OwnerPostPublishRealInput.Common.ps1")
}

function New-RealOwnerProofLaneContract {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Category,
    [string[]]$RequiredFields,
    [string]$OwnerInputArtifact,
    [string]$TemplateArtifact,
    [string]$RefreshScript,
    [string]$ValidationScript,
    [string]$RecordArtifact,
    [string]$ValidationArtifact,
    [string]$ExpectedRecordKind,
    [string]$ExpectedValidationKind,
    [string]$AcceptedProperty,
    [ValidateSet("record", "validation")][string]$AcceptedSource,
    [string]$StateProperty,
    [string]$ActionCountProperty,
    [string]$ProofScope
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    category = $Category
    requiredFields = @($RequiredFields)
    requiredFieldCount = @($RequiredFields).Count
    ownerInputArtifact = $OwnerInputArtifact
    templateArtifact = $TemplateArtifact
    refreshScript = $RefreshScript
    validationScript = $ValidationScript
    recordArtifact = $RecordArtifact
    validationArtifact = $ValidationArtifact
    expectedRecordKind = $ExpectedRecordKind
    expectedValidationKind = $ExpectedValidationKind
    acceptedProperty = $AcceptedProperty
    acceptedSource = $AcceptedSource
    stateProperty = $StateProperty
    actionCountProperty = $ActionCountProperty
    proofScope = $ProofScope
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

function Get-PostPublishLaneRequiredFields {
  param([string]$LaneId)

  $lane = @(Get-OwnerPostPublishLaneSpecs | Where-Object { [string]$_.id -eq $LaneId }) | Select-Object -First 1
  if ($null -eq $lane) { return @() }
  return @($lane.fields | ForEach-Object { [string]$_.name })
}

function Get-RealOwnerProofLaneContracts {
  @(
    New-RealOwnerProofLaneContract `
      -Id "public-package-url-hash" `
      -Title "Public package URL and hash Owner proof" `
      -Category "post-publish" `
      -RequiredFields (Get-PostPublishLaneRequiredFields "public-package-urls-and-hashes") `
      -OwnerInputArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.json" `
      -TemplateArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.template.json" `
      -RefreshScript "Export-PublicPackageUrlHashProofValidator.ps1" `
      -ValidationScript "Test-PublicPackageUrlHashProofValidator.ps1" `
      -RecordArtifact "artifacts/final-release/public-package-url-hash-proof-validator.json" `
      -ValidationArtifact "artifacts/final-release/public-package-url-hash-proof-validator-validation.json" `
      -ExpectedRecordKind "public-package-url-hash-proof-validator" `
      -ExpectedValidationKind "public-package-url-hash-proof-validator-validation" `
      -AcceptedProperty "ownerEvidenceAccepted" `
      -AcceptedSource "validation" `
      -StateProperty "validatorState" `
      -ActionCountProperty "blockedReasonCount" `
      -ProofScope "public-package-download-hash-traceability"
    New-RealOwnerProofLaneContract `
      -Id "external-clean-consumer-post-publish" `
      -Title "External CleanConsumer post-publish Owner proof" `
      -Category "post-publish" `
      -RequiredFields (Get-PostPublishLaneRequiredFields "external-clean-consumer-logs") `
      -OwnerInputArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.json" `
      -TemplateArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.template.json" `
      -RefreshScript "Export-ExternalCleanConsumerPostPublishProofValidator.ps1" `
      -ValidationScript "Test-ExternalCleanConsumerPostPublishProofValidator.ps1" `
      -RecordArtifact "artifacts/final-release/external-clean-consumer-post-publish-proof-validator.json" `
      -ValidationArtifact "artifacts/final-release/external-clean-consumer-post-publish-proof-validator-validation.json" `
      -ExpectedRecordKind "external-clean-consumer-post-publish-proof-validator" `
      -ExpectedValidationKind "external-clean-consumer-post-publish-proof-validator-validation" `
      -AcceptedProperty "ownerEvidenceAccepted" `
      -AcceptedSource "validation" `
      -StateProperty "validatorState" `
      -ActionCountProperty "blockedReasonCount" `
      -ProofScope "repository-external-clean-consumer-runtime"
    New-RealOwnerProofLaneContract `
      -Id "article-publication" `
      -Title "Article publication Owner proof" `
      -Category "post-publish" `
      -RequiredFields (Get-PostPublishLaneRequiredFields "article-publication-urls") `
      -OwnerInputArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.json" `
      -TemplateArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.template.json" `
      -RefreshScript "Export-ArticlePublicationProofValidator.ps1" `
      -ValidationScript "Test-ArticlePublicationProofValidator.ps1" `
      -RecordArtifact "artifacts/final-release/article-publication-proof-validator.json" `
      -ValidationArtifact "artifacts/final-release/article-publication-proof-validator-validation.json" `
      -ExpectedRecordKind "article-publication-proof-validator" `
      -ExpectedValidationKind "article-publication-proof-validator-validation" `
      -AcceptedProperty "ownerEvidenceAccepted" `
      -AcceptedSource "validation" `
      -StateProperty "validatorState" `
      -ActionCountProperty "blockedReasonCount" `
      -ProofScope "public-article-publication"
    New-RealOwnerProofLaneContract `
      -Id "yolovision-real-model" `
      -Title "YoloVision real model Owner proof" `
      -Category "post-publish" `
      -RequiredFields (Get-PostPublishLaneRequiredFields "yolovision-real-model-assets") `
      -OwnerInputArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.json" `
      -TemplateArtifact "artifacts/final-release/owner-post-publish-docs-article-sample-real-input.template.json" `
      -RefreshScript "Export-YoloVisionRealModelPostPublishProofValidator.ps1" `
      -ValidationScript "Test-YoloVisionRealModelPostPublishProofValidator.ps1" `
      -RecordArtifact "artifacts/final-release/yolovision-real-model-post-publish-proof-validator.json" `
      -ValidationArtifact "artifacts/final-release/yolovision-real-model-post-publish-proof-validator-validation.json" `
      -ExpectedRecordKind "yolovision-real-model-post-publish-proof-validator" `
      -ExpectedValidationKind "yolovision-real-model-post-publish-proof-validator-validation" `
      -AcceptedProperty "ownerEvidenceAccepted" `
      -AcceptedSource "validation" `
      -StateProperty "validatorState" `
      -ActionCountProperty "blockedReasonCount" `
      -ProofScope "real-model-runtime-execution"
    New-RealOwnerProofLaneContract `
      -Id "final-rollback-review" `
      -Title "Final Owner rollback review" `
      -Category "governance" `
      -RequiredFields @("reviewer", "reviewedAtUtc", "decision", "acceptedRisk", "rollbackPlan", "packageVersion", "rollbackTargetFeed", "rollbackPackageIds", "rollbackPackageVersion", "rollbackExecutionApprovedBy", "rollbackDecisionTimestampUtc", "rollbackScope", "rollbackCommandPlanPath", "rollbackCommandPlanSha256", "withdrawCommandSha256", "delistCommandSha256", "deprecateCommandSha256", "riskAssessment", "userImpactAssessment", "evidenceBundleSha256", "externalCleanConsumerProofReady", "postPublishProofReady", "confirmsNoRollbackExecutionByAutomation", "confirmsNoDeleteDelistWithdrawDeprecateExecution", "confirmsRollbackPlanNotReleaseCloseProof") `
      -OwnerInputArtifact "artifacts/final-release/final-owner-rollback-review.owner.json" `
      -TemplateArtifact "artifacts/final-release/final-owner-rollback-review.template.json" `
      -RefreshScript "Import-FinalOwnerRollbackReview.ps1" `
      -ValidationScript "Test-FinalOwnerRollbackReview.ps1" `
      -RecordArtifact "artifacts/final-release/final-owner-rollback-review-import.json" `
      -ValidationArtifact "artifacts/final-release/final-owner-rollback-review-validation.json" `
      -ExpectedRecordKind "final-owner-rollback-review-import" `
      -ExpectedValidationKind "final-owner-rollback-review-validation" `
      -AcceptedProperty "rollbackReviewReady" `
      -AcceptedSource "record" `
      -StateProperty "importState" `
      -ActionCountProperty "failedActionRequiredCount" `
      -ProofScope "owner-rollback-governance-review"
    New-RealOwnerProofLaneContract `
      -Id "final-close-decision" `
      -Title "Final Owner close decision" `
      -Category "governance" `
      -RequiredFields @("reviewer", "reviewedAtUtc", "ownerReviewed", "decision", "closeReason", "acceptedRisk", "rollbackPlan", "packageId", "packageVersion", "releaseIssueId", "releaseIssueUrl", "evidenceBundleSha256", "classificationAuditSha256", "publicPackageProofSha256", "externalCleanConsumerProofSha256", "postPublishProofSha256", "postPublishProofUrls", "externalCleanConsumerProofReady", "postPublishProofReady", "rollbackReviewReady", "classificationAuditPassed", "manualCloseOnlyConfirmation", "confirmsNoIssueCloseByAutomation") `
      -OwnerInputArtifact "artifacts/final-release/final-owner-close-decision.owner.json" `
      -TemplateArtifact "artifacts/final-release/final-owner-close-decision.template.json" `
      -RefreshScript "Import-FinalOwnerCloseDecision.ps1" `
      -ValidationScript "Test-FinalOwnerCloseDecision.ps1" `
      -RecordArtifact "artifacts/final-release/final-owner-close-decision-import.json" `
      -ValidationArtifact "artifacts/final-release/final-owner-close-decision-validation.json" `
      -ExpectedRecordKind "final-owner-close-decision-import" `
      -ExpectedValidationKind "final-owner-close-decision-validation" `
      -AcceptedProperty "finalCloseDecisionReady" `
      -AcceptedSource "record" `
      -StateProperty "importState" `
      -ActionCountProperty "failedActionRequiredCount" `
      -ProofScope "owner-final-close-governance-review"
    New-RealOwnerProofLaneContract `
      -Id "github-ci-evidence" `
      -Title "GitHub CI evidence Owner review" `
      -Category "verification" `
      -RequiredFields @("repository", "branch", "commitSha", "workflowName", "runId", "runUrl", "conclusion", "createdAtUtc", "completedAtUtc", "artifactManifestSha256", "ownerReviewer", "ownerReviewed") `
      -OwnerInputArtifact "artifacts/final-release/github-ci-evidence.owner-input.json" `
      -TemplateArtifact "artifacts/final-release/github-ci-evidence.owner-input.template.json" `
      -RefreshScript "Import-GitHubCiEvidenceFromOwnerInput.ps1" `
      -ValidationScript "Test-GitHubCiEvidenceFromOwnerInput.ps1" `
      -RecordArtifact "artifacts/final-release/github-ci-evidence-from-owner-input.json" `
      -ValidationArtifact "artifacts/final-release/github-ci-evidence-from-owner-input-validation.json" `
      -ExpectedRecordKind "github-ci-evidence-from-owner-input" `
      -ExpectedValidationKind "github-ci-evidence-from-owner-input-validation" `
      -AcceptedProperty "ciEvidenceAccepted" `
      -AcceptedSource "validation" `
      -StateProperty "validationState" `
      -ActionCountProperty "failedActionRequiredCount" `
      -ProofScope "completed-github-actions-run-traceability"
    New-RealOwnerProofLaneContract `
      -Id "release-evidence-bundle-hash-review" `
      -Title "Release evidence bundle hash Owner review" `
      -Category "verification" `
      -RequiredFields @("releaseEvidenceBundlePath", "releaseEvidenceBundleSha256", "reviewedAtUtc", "ownerReviewer", "ownerReviewed") `
      -OwnerInputArtifact "artifacts/final-release/release-evidence-bundle-hash-review.owner-input.json" `
      -TemplateArtifact "artifacts/final-release/release-evidence-bundle-hash-review.owner-input.template.json" `
      -RefreshScript "Import-ReleaseEvidenceBundleHashReview.ps1" `
      -ValidationScript "Test-ReleaseEvidenceBundleHashReview.ps1" `
      -RecordArtifact "artifacts/final-release/release-evidence-bundle-hash-review.json" `
      -ValidationArtifact "artifacts/final-release/release-evidence-bundle-hash-review-validation.json" `
      -ExpectedRecordKind "release-evidence-bundle-hash-review" `
      -ExpectedValidationKind "release-evidence-bundle-hash-review-validation" `
      -AcceptedProperty "reviewAccepted" `
      -AcceptedSource "validation" `
      -StateProperty "validationState" `
      -ActionCountProperty "failedActionRequiredCount" `
      -ProofScope "release-evidence-bundle-hash-traceability"
    New-RealOwnerProofLaneContract `
      -Id "classification-audit-hash-review" `
      -Title "Classification audit hash Owner review" `
      -Category "verification" `
      -RequiredFields @("classificationAuditPath", "classificationAuditSha256", "classificationAuditState", "reviewedAtUtc", "ownerReviewer", "ownerReviewed") `
      -OwnerInputArtifact "artifacts/final-release/classification-audit-hash-review.owner-input.json" `
      -TemplateArtifact "artifacts/final-release/classification-audit-hash-review.owner-input.template.json" `
      -RefreshScript "Import-ClassificationAuditHashReview.ps1" `
      -ValidationScript "Test-ClassificationAuditHashReview.ps1" `
      -RecordArtifact "artifacts/final-release/classification-audit-hash-review.json" `
      -ValidationArtifact "artifacts/final-release/classification-audit-hash-review-validation.json" `
      -ExpectedRecordKind "classification-audit-hash-review" `
      -ExpectedValidationKind "classification-audit-hash-review-validation" `
      -AcceptedProperty "reviewAccepted" `
      -AcceptedSource "validation" `
      -StateProperty "validationState" `
      -ActionCountProperty "failedActionRequiredCount" `
      -ProofScope "classification-audit-hash-traceability"
  )
}

function Get-RealOwnerProofLaneObservation {
  param(
    [object]$Contract,
    [string]$RepositoryRoot
  )

  $record = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath ([string]$Contract.recordArtifact)
  $validation = Read-JsonOrNull -RepositoryRoot $RepositoryRoot -RelativePath ([string]$Contract.validationArtifact)
  $acceptedSource = if ([string]$Contract.acceptedSource -eq "record") { $record } else { $validation }
  $recordKindMatches = $null -ne $record -and [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq [string]$Contract.expectedRecordKind
  $validationKindMatches = $null -ne $validation -and [string](Get-PropertyOrDefault -Object $validation -Name "recordKind" -DefaultValue "") -eq [string]$Contract.expectedValidationKind
  $failedBlockerCount = [int](Get-PropertyOrDefault -Object $validation -Name "failedBlockerCount" -DefaultValue 999)
  $accepted = $null -ne $acceptedSource -and [bool](Get-PropertyOrDefault -Object $acceptedSource -Name ([string]$Contract.acceptedProperty) -DefaultValue $false)
  $stateSource = if ($null -ne $record -and $record.PSObject.Properties.Name -contains [string]$Contract.stateProperty) { $record } else { $validation }
  $actionSource = if ($null -ne $record -and $record.PSObject.Properties.Name -contains [string]$Contract.actionCountProperty) { $record } else { $validation }
  $actionCount = [int](Get-PropertyOrDefault -Object $actionSource -Name ([string]$Contract.actionCountProperty) -DefaultValue 0)
  $boundary = [string](Get-PropertyOrDefault -Object $validation -Name "boundary" -DefaultValue (Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue ""))

  [pscustomobject]@{
    id = [string]$Contract.id
    title = [string]$Contract.title
    category = [string]$Contract.category
    proofScope = [string]$Contract.proofScope
    recordArtifact = [string]$Contract.recordArtifact
    validationArtifact = [string]$Contract.validationArtifact
    recordPresent = $null -ne $record
    validationPresent = $null -ne $validation
    recordKindMatches = $recordKindMatches
    validationKindMatches = $validationKindMatches
    structuralReady = $recordKindMatches -and $validationKindMatches -and $failedBlockerCount -eq 0
    accepted = $accepted
    state = [string](Get-PropertyOrDefault -Object $stateSource -Name ([string]$Contract.stateProperty) -DefaultValue "missing-owner-proof-state")
    failedBlockerCount = $failedBlockerCount
    failedActionRequiredCount = $actionCount
    ownerActionRequired = -not $accepted
    boundary = $boundary
  }
}
