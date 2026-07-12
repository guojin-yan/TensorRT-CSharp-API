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

function Get-ArrayOrEmpty {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue @()
  if ($null -eq $value) {
    return @()
  }

  return @($value)
}

function Get-TriageTierCount {
  param(
    [object[]]$TierSummaries,
    [string]$SafetyTier
  )

  $tier = $TierSummaries | Where-Object { $_.safetyTier -eq $SafetyTier } | Select-Object -First 1
  return [int](Get-PropertyOrDefault -Object $tier -Name "candidateCount" -DefaultValue 0)
}

$releaseCloseGapDashboard = Read-JsonOrNull "artifacts\final-release\release-close-gap-dashboard.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$releasePackageProofBundle = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseCandidateFinalEvidenceFreeze = Read-JsonOrNull "artifacts\final-release\release-candidate-final-evidence-freeze.json"
$deferredSafetyTriage = Read-JsonOrNull "artifacts\interface-coverage\deferred-candidate-safety-triage.json"
$deferredBTierProofClosureDashboard = Read-JsonOrNull "artifacts\interface-coverage\deferred-btier-proof-closure-dashboard.json"
$deferredBTierAliasProofClosureRecord = Read-JsonOrNull "artifacts\interface-coverage\deferred-btier-alias-proof-closure-record.json"
$releaseRuntimeProofExecutionMatrix = Read-JsonOrNull "artifacts\final-release\release-runtime-proof-execution-matrix.json"
$releaseProofReadinessSnapshot = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$externalRuntimeProofTemplate = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$packageConsumerProofExecutionPack = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-execution-pack.json"
$packageConsumerProofPackValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-pack-validation.json"
$linuxRunnerProofTemplate = Read-JsonOrNull "artifacts\final-release\linux-runner-evidence-record.template.json"
$linuxRunnerProofValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$linuxRunnerProofExecutionPack = Read-JsonOrNull "artifacts\final-release\linux-runner-proof-execution-pack.json"
$linuxRunnerProofPackValidation = Read-JsonOrNull "artifacts\final-release\linux-runner-proof-pack-validation.json"
$sampleRunEvidenceValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$onnxEngineBuildEvidenceSidecarAudit = Read-JsonOrNull "artifacts\user-acceptance\onnx-engine-build-evidence-sidecar-audit.json"
$realModelAndPackageProofInputPackage = Read-JsonOrNull "artifacts\final-release\real-model-and-package-proof-input-package.json"
$realCaseProofExecutionPack = Read-JsonOrNull "artifacts\final-release\real-case-proof-execution-pack.json"
$realCaseEvidenceRecordTemplate = Read-JsonOrNull "artifacts\final-release\real-case-evidence-record-template.json"
$realCaseEvidenceRecordValidation = Read-JsonOrNull "artifacts\final-release\real-case-evidence-record-validation.json"
$postPublishVerificationTemplate = Read-JsonOrNull "artifacts\final-release\post-publish-verification-record-template.json"
$postPublishVerificationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$gapItems = Get-ArrayOrEmpty -Object $releaseCloseGapDashboard -Name "gapItems"
$remainingBlockers = Get-ArrayOrEmpty -Object $releaseCandidateFinalEvidenceFreeze -Name "remainingBlockers"
$preflightItems = Get-ArrayOrEmpty -Object $releaseClosePreflight -Name "preflightItems"
$ownerAuthorizationRequiredFields = Get-ArrayOrEmpty -Object $releaseClosePreflight -Name "ownerAuthorizationRequiredFields"
$manualMaterializationPrerequisites = Get-ArrayOrEmpty -Object $releaseClosePreflight -Name "manualMaterializationPrerequisites"
$postPublishRequiredEvidence = Get-ArrayOrEmpty -Object $releaseClosePreflight -Name "postPublishRequiredEvidence"
$validatorCommands = Get-ArrayOrEmpty -Object $releaseCandidateFinalEvidenceFreeze -Name "validatorCommands"
$tierSummaries = Get-ArrayOrEmpty -Object $deferredSafetyTriage -Name "tierSummaries"

$triageKind = [string](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "triageKind" -DefaultValue "missing-deferred-candidate-safety-triage")
$triageTotalRows = [int](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "totalTriageRowCount" -DefaultValue 0)
$tierA = Get-TriageTierCount -TierSummaries $tierSummaries -SafetyTier "A - immediate-safe"
$tierB = Get-TriageTierCount -TierSummaries $tierSummaries -SafetyTier "B - safe-alternative-or-alias"
$tierC = Get-TriageTierCount -TierSummaries $tierSummaries -SafetyTier "C - design-gate-required"
$tierD = Get-TriageTierCount -TierSummaries $tierSummaries -SafetyTier "D - keep-deferred"
$triageState = if ($triageKind -eq "deferred-candidate-safety-triage") { "triage-ready-planning-input-only" } else { "missing-deferred-safety-triage" }
$triageBoundary = "Deferred safety triage is planning input and boundary disclosure only: safe-alternative rows need alias/proof closure, design-gate rows require explicit design approval, keep-deferred rows remain deferred, and no tier is release proof or permission to delete deferred records."
$bTierClosureDashboardState = [string](Get-PropertyOrDefault -Object $deferredBTierProofClosureDashboard -Name "closureState" -DefaultValue "missing-deferred-btier-proof-closure-dashboard")
$bTierClosureDashboardTotalCount = [int](Get-PropertyOrDefault -Object $deferredBTierProofClosureDashboard -Name "totalBTierCount" -DefaultValue 0)
$bTierClosureDashboardSelectedCount = [int](Get-PropertyOrDefault -Object $deferredBTierProofClosureDashboard -Name "selectedCandidateCount" -DefaultValue 0)
$bTierClosureDashboardBoundary = [string](Get-PropertyOrDefault -Object $deferredBTierProofClosureDashboard -Name "boundary" -DefaultValue "B-tier proof closure dashboard is engineering closure input only and is not release proof.")
$bTierAliasClosureState = [string](Get-PropertyOrDefault -Object $deferredBTierAliasProofClosureRecord -Name "closureState" -DefaultValue "missing-deferred-btier-alias-proof-closure-record")
$bTierAliasClosureCandidateCount = [int](Get-PropertyOrDefault -Object $deferredBTierAliasProofClosureRecord -Name "closureCandidateCount" -DefaultValue 0)
$bTierAliasClosureBoundary = [string](Get-PropertyOrDefault -Object $deferredBTierAliasProofClosureRecord -Name "boundary" -DefaultValue "B-tier alias proof closure record is engineering closure evidence only and is not release proof.")
$runtimeProofExecutionMatrixState = [string](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "matrixState" -DefaultValue "missing-release-runtime-proof-execution-matrix")
$runtimeProofExecutionMatrixProofItemCount = [int](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "proofItemCount" -DefaultValue 0)
$runtimeProofExecutionMatrixBlockedCount = [int](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "blockedProofItemCount" -DefaultValue 0)
$runtimeProofExecutionMatrixBoundary = [string](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "boundary" -DefaultValue "Release runtime proof execution matrix is a blocked execution plan and is not proof.")
$runtimeProofExecutionMatrixTemplateCoverageState = [string](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "ownerProofTemplateCoverageState" -DefaultValue "missing-owner-proof-template-coverage")
$runtimeProofExecutionMatrixValidatorCoverageState = [string](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "validatorCoverageState" -DefaultValue "missing-validator-coverage")
$runtimeProofExecutionMatrixArticleCoverageState = [string](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "ownerProofExecutionArticleCoverageState" -DefaultValue "missing-owner-proof-article-coverage")
$runtimeProofExecutionMatrixExpectedInputTemplateCount = [int](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "expectedInputTemplateCount" -DefaultValue 0)
$runtimeProofExecutionMatrixExpectedValidatedRecordCount = [int](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "expectedValidatedRecordCount" -DefaultValue 0)
$runtimeProofExecutionMatrixValidatorCommandCount = [int](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "validatorCommandCount" -DefaultValue 0)
$runtimeProofExecutionMatrixArticleMappedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "articleMappedProofItemCount" -DefaultValue 0)
$runtimeProofExecutionMatrixPostPublishRequiredEvidence = Get-ArrayOrEmpty -Object $releaseRuntimeProofExecutionMatrix -Name "postPublishRequiredEvidence"
$runtimeProofExecutionMatrixPostPublishRequiredEvidenceCount = [int](Get-PropertyOrDefault -Object $releaseRuntimeProofExecutionMatrix -Name "postPublishRequiredEvidenceCount" -DefaultValue @($runtimeProofExecutionMatrixPostPublishRequiredEvidence).Count)
$releaseProofReadinessSnapshotState = [string](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessState" -DefaultValue "missing-release-proof-readiness-snapshot")
$releaseProofReadinessSnapshotItemCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessItemCount" -DefaultValue 0)
$releaseProofReadinessSnapshotReadyCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readyProofItemCount" -DefaultValue 0)
$releaseProofReadinessSnapshotBlockedCount = [int](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "blockedProofItemCount" -DefaultValue 0)
$releaseProofReadinessSnapshotIsComplete = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "isReleaseProofComplete" -DefaultValue $false)
$releaseProofReadinessSnapshotCanPublish = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "canPublishPublicly" -DefaultValue $false)
$releaseProofReadinessSnapshotCanClose = [bool](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "canCloseReleaseIssue" -DefaultValue $false)
$releaseProofReadinessSnapshotBoundary = [string](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "boundary" -DefaultValue "Release proof readiness snapshot is a status view only and is not proof.")
$packageConsumerRuntimeProofTemplateState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofTemplate -Name "proofState" -DefaultValue "missing-external-runtime-proof-record-template")
$packageConsumerRuntimeProofTemplateKind = [string](Get-PropertyOrDefault -Object $externalRuntimeProofTemplate -Name "recordKind" -DefaultValue "missing-external-runtime-proof-record-template")
$packageConsumerRuntimeValidatorState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$packageConsumerRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$packageConsumerRuntimeProofCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$packageConsumerProofExecutionPackState = [string](Get-PropertyOrDefault -Object $packageConsumerProofExecutionPack -Name "packState" -DefaultValue "missing-package-consumer-runtime-proof-execution-pack")
$packageConsumerProofExecutionPackMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $packageConsumerProofExecutionPack -Name "missingOwnerInputCount" -DefaultValue 0)
$packageConsumerProofExecutionPackCanPromote = [bool](Get-PropertyOrDefault -Object $packageConsumerProofExecutionPack -Name "canPromotePackageConsumerRuntime" -DefaultValue $false)
$packageConsumerProofPackValidationState = [string](Get-PropertyOrDefault -Object $packageConsumerProofPackValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-pack-validation")
$packageConsumerProofPackValidationCanPromote = [bool](Get-PropertyOrDefault -Object $packageConsumerProofPackValidation -Name "canPromotePackageConsumerRuntime" -DefaultValue $false)
$linuxRunnerProofTemplateState = [string](Get-PropertyOrDefault -Object $linuxRunnerProofTemplate -Name "finalReleaseTemplateState" -DefaultValue ([string](Get-PropertyOrDefault -Object $linuxRunnerProofTemplate -Name "recordState" -DefaultValue "missing-linux-runner-evidence-record-template")))
$linuxRunnerProofTemplateKind = [string](Get-PropertyOrDefault -Object $linuxRunnerProofTemplate -Name "recordKind" -DefaultValue "missing-linux-runner-evidence-record-template")
$linuxRunnerValidatorState = [string](Get-PropertyOrDefault -Object $linuxRunnerProofValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxRunnerProofCanPromote = [bool](Get-PropertyOrDefault -Object $linuxRunnerProofValidation -Name "canPromoteLinuxPackage" -DefaultValue $false)
$linuxRunnerProofIsReal = [bool](Get-PropertyOrDefault -Object $linuxRunnerProofValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$linuxRunnerProofExecutionPackState = [string](Get-PropertyOrDefault -Object $linuxRunnerProofExecutionPack -Name "packState" -DefaultValue "missing-linux-runner-proof-execution-pack")
$linuxRunnerProofExecutionPackMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $linuxRunnerProofExecutionPack -Name "missingOwnerInputCount" -DefaultValue 0)
$linuxRunnerProofExecutionPackCanPromote = [bool](Get-PropertyOrDefault -Object $linuxRunnerProofExecutionPack -Name "canPromoteLinuxRunnerProof" -DefaultValue $false)
$linuxRunnerProofPackValidationState = [string](Get-PropertyOrDefault -Object $linuxRunnerProofPackValidation -Name "validationState" -DefaultValue "missing-linux-runner-proof-pack-validation")
$linuxRunnerProofPackValidationCanPromote = [bool](Get-PropertyOrDefault -Object $linuxRunnerProofPackValidation -Name "canPromoteLinuxRunnerProof" -DefaultValue $false)
$realModelRuntimeProofInputPackageState = [string](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "packageState" -DefaultValue "missing-real-model-and-package-proof-input-package")
$realModelRuntimeProofInputPackageKind = [string](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "recordKind" -DefaultValue "missing-real-model-and-package-proof-input-package")
$sampleRunEvidenceValidatorState = [string](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleRunEvidenceProofClassification = [string](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$sampleRunEvidenceTemplateOnly = [bool](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "templateOnly" -DefaultValue $true)
$sampleRunEvidenceCanPromoteRealModelRuntime = [bool](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$sampleRunEvidenceRealModelEvidenceReady = [bool](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "realModelEvidenceReady" -DefaultValue $false)
$sampleRunEvidenceOwnerActionRequiredCount = [int](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "ownerActionRequiredCount" -DefaultValue 0)
$onnxEngineBuildEvidenceSidecarAuditState = [string](Get-PropertyOrDefault -Object $onnxEngineBuildEvidenceSidecarAudit -Name "auditState" -DefaultValue "missing-onnx-engine-build-evidence-sidecar-audit")
$onnxEngineBuildEvidenceSidecarOwnerActionRequiredCount = [int](Get-PropertyOrDefault -Object $onnxEngineBuildEvidenceSidecarAudit -Name "ownerActionRequiredCount" -DefaultValue 0)
$onnxEngineBuildEvidenceSidecarErrorCount = [int](Get-PropertyOrDefault -Object $onnxEngineBuildEvidenceSidecarAudit -Name "errorCount" -DefaultValue 0)
$realCaseProofExecutionPackState = [string](Get-PropertyOrDefault -Object $realCaseProofExecutionPack -Name "packState" -DefaultValue "missing-real-case-proof-execution-pack")
$realCaseProofExecutionPackCaseCount = [int](Get-PropertyOrDefault -Object $realCaseProofExecutionPack -Name "caseCount" -DefaultValue 0)
$realCaseProofExecutionPackBlockedCaseCount = [int](Get-PropertyOrDefault -Object $realCaseProofExecutionPack -Name "blockedCaseCount" -DefaultValue 0)
$realCaseProofExecutionPackMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $realCaseProofExecutionPack -Name "missingOwnerInputCount" -DefaultValue 0)
$realCaseProofExecutionPackCanPromote = [bool](Get-PropertyOrDefault -Object $realCaseProofExecutionPack -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$realCaseEvidenceRecordTemplateState = [string](Get-PropertyOrDefault -Object $realCaseEvidenceRecordTemplate -Name "templateState" -DefaultValue "missing-real-case-evidence-record-template")
$realCaseEvidenceRecordTemplateKind = [string](Get-PropertyOrDefault -Object $realCaseEvidenceRecordTemplate -Name "recordKind" -DefaultValue "missing-real-case-evidence-record-template")
$realCaseEvidenceRecordValidatorState = [string](Get-PropertyOrDefault -Object $realCaseEvidenceRecordValidation -Name "validationState" -DefaultValue "missing-real-case-evidence-validation")
$realCaseEvidenceRecordProofClassification = [string](Get-PropertyOrDefault -Object $realCaseEvidenceRecordValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$realCaseEvidenceRecordCanPromote = [bool](Get-PropertyOrDefault -Object $realCaseEvidenceRecordValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$postPublishVerificationTemplateState = [string](Get-PropertyOrDefault -Object $postPublishVerificationTemplate -Name "verificationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublishVerificationTemplate -Name "proofState" -DefaultValue "missing-post-publish-verification-record-template")))
$postPublishVerificationTemplateKind = [string](Get-PropertyOrDefault -Object $postPublishVerificationTemplate -Name "recordKind" -DefaultValue "missing-post-publish-verification-record-template")
$postPublishVerificationTemplateOnly = [bool](Get-PropertyOrDefault -Object $postPublishVerificationTemplate -Name "templateOnly" -DefaultValue $true)
$postPublishVerificationValidatorState = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishVerificationProofClassification = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishProofClassification" -DefaultValue "missing-proof-classification")
$postPublishVerificationIsProof = [bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishVerificationCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishVerificationFailedBlockerCount = [int](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "failedBlockerCount" -DefaultValue 0)
$postPublishVerificationFailedProofItemCount = [int](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "failedProofItemCount" -DefaultValue 0)
$ownerAuthorizationProofGateState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "ownerAuthorizationProofGateState" -DefaultValue "missing-owner-authorization-proof-gate")
$ownerAuthorizationRequiredFieldCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "ownerAuthorizationRequiredFieldCount" -DefaultValue 0)
$ownerAuthorizationMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "ownerAuthorizationMissingOwnerInputCount" -DefaultValue -1)
$manualMaterializationPrerequisiteCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "manualMaterializationPrerequisiteCount" -DefaultValue 0)
$postPublishRequiredEvidenceCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "postPublishRequiredEvidenceCount" -DefaultValue 0)

$blockerIds = @($remainingBlockers | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "gapId" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($blockerIds.Count -eq 0) {
  $blockerIds = @($gapItems | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "gapId" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

$nonSubstituteProofKinds = @(
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
  "owner-action-required without validator pass",
  "deferred safety triage",
  "safe-alternative-or-alias planning input",
  "design-gate-required planning input",
  "keep-deferred boundary disclosure",
  "A-tier candidate list",
  "B-tier alias/proof planning list",
  "C-tier design gate list",
  "D-tier deferred boundary list",
  "B-tier proof closure dashboard",
  "B-tier closure planning input",
  "B-tier alias proof closure record",
  "release runtime proof execution matrix",
  "release proof readiness snapshot"
)

$sourceArtifacts = @(
  "artifacts/final-release/release-close-gap-dashboard.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-package-proof-bundle.json",
  "artifacts/final-release/release-close-preflight.json",
  "artifacts/final-release/release-owner-approval-input-template.json",
  "artifacts/final-release/release-owner-approval-input-validation.json",
  "artifacts/final-release/release-owner-decision-record.json",
  "artifacts/final-release/owner-authorized-publish-command-plan.json",
  "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
  "artifacts/final-release/release-candidate-final-evidence-freeze.json",
  "artifacts/interface-coverage/deferred-candidate-safety-triage.json",
  "artifacts/interface-coverage/deferred-candidate-safety-triage.md",
  "artifacts/interface-coverage/deferred-btier-proof-closure-dashboard.json",
  "artifacts/interface-coverage/deferred-btier-proof-closure-dashboard.md",
  "artifacts/interface-coverage/deferred-btier-alias-proof-closure-record.json",
  "artifacts/interface-coverage/deferred-btier-alias-proof-closure-record.md",
  "artifacts/final-release/release-runtime-proof-execution-matrix.json",
  "artifacts/final-release/release-runtime-proof-execution-matrix.md",
  "artifacts/final-release/release-proof-readiness-snapshot.json",
  "artifacts/final-release/release-proof-readiness-snapshot.md",
  "artifacts/final-release/external-runtime-proof-record-template.json",
  "artifacts/final-release/external-runtime-proof-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-execution-pack.json",
  "artifacts/final-release/package-consumer-runtime-proof-pack-validation.json",
  "artifacts/final-release/linux-runner-evidence-record.template.json",
  "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
  "artifacts/final-release/linux-runner-proof-execution-pack.json",
  "artifacts/final-release/linux-runner-proof-pack-validation.json",
  "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
  "artifacts/user-acceptance/onnx-engine-build-evidence-sidecar-audit.json",
  "artifacts/final-release/real-model-and-package-proof-input-package.json",
  "artifacts/final-release/real-case-proof-execution-pack.json",
  "artifacts/final-release/real-case-evidence-record-template.json",
  "artifacts/final-release/real-case-evidence-record-validation.json",
  "artifacts/final-release/post-publish-verification-record-template.json",
  "artifacts/final-release/post-publish-verification-validation.json"
)

$ownerActions = @(
  "Provide explicit owner authorization before any manual publish materialization.",
  "Collect package-consumer-runtime proof on a compatible CUDA/TensorRT host.",
  "Collect Linux runner proof on a real Linux x64 CUDA/TensorRT host.",
  "Collect Classification/YoloVision real-model-runtime proof with real assets and logs.",
  "After real publication, collect post-publish verification proof from the real package channel.",
  "Use B-tier deferred safety triage only for alias/proof closure planning; do not touch C/D public API without design gate approval."
)

$releaseProofFinalAuditItems = @(
  [ordered]@{
    proofId = "owner-authorization"
    currentState = "owner-action-required"
    templatePath = "artifacts/final-release/release-owner-approval-input-template.json"
    expectedValidatedRecord = "artifacts/final-release/release-owner-approval-input-record.json"
    validatorCommand = "Test-ReleaseOwnerApprovalInput.ps1 + Test-OwnerAuthorizedPublishCommandPlan.ps1"
    canPromote = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    missingOwnerInput = "Explicit owner approval, target channel, manual command materialization, and publish intent."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
  [ordered]@{
    proofId = "package-consumer-runtime"
    currentState = $packageConsumerRuntimeValidatorState
    templatePath = "artifacts/final-release/external-runtime-proof-record-template.json"
    expectedValidatedRecord = "artifacts/final-release/external-runtime-proof-record.json"
    validatorCommand = "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof"
    canPromote = $packageConsumerRuntimeProofCanPromote
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    missingOwnerInput = "Clean external consumer, real package source, runtime smoke log, nupkg SHA256 values, and compatible host metadata."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
  [ordered]@{
    proofId = "package-consumer-proof-pack"
    currentState = $packageConsumerProofExecutionPackState
    templatePath = "artifacts/final-release/package-consumer-runtime-proof-execution-pack.json"
    expectedValidatedRecord = "artifacts/final-release/external-runtime-proof-record.json"
    validatorCommand = "Test-PackageConsumerRuntimeProofPack.ps1 + Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof"
    canPromote = ($packageConsumerProofExecutionPackCanPromote -and $packageConsumerProofPackValidationCanPromote)
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    missingOwnerInput = "Clean external consumer outside repository, real package source, restore/build/runtime smoke logs, nupkg/log SHA256 values, no ProjectReference confirmation, and compatible host metadata."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
  [ordered]@{
    proofId = "linux-runner-proof"
    currentState = $linuxRunnerValidatorState
    templatePath = "artifacts/final-release/linux-runner-evidence-record.template.json"
    expectedValidatedRecord = "artifacts/final-release/linux-runner-evidence-record.json"
    validatorCommand = "Test-LinuxRunnerEvidenceRecord.ps1"
    canPromote = $linuxRunnerProofCanPromote
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    missingOwnerInput = "Real Linux x64 CUDA/TensorRT runner, command log, runtime package SHA256, and CUDA/TensorRT/cuDNN/GPU/driver metadata."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
  [ordered]@{
    proofId = "linux-runner-proof-pack"
    currentState = $linuxRunnerProofExecutionPackState
    templatePath = "artifacts/final-release/linux-runner-proof-execution-pack.json"
    expectedValidatedRecord = "artifacts/final-release/linux-runner-evidence-record.json"
    validatorCommand = "Test-LinuxRunnerProofPack.ps1 + Test-LinuxRunnerEvidenceRecord.ps1"
    canPromote = ($linuxRunnerProofExecutionPackCanPromote -and $linuxRunnerProofPackValidationCanPromote)
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    missingOwnerInput = "Real Linux x64 host, runtime package SHA256, Linux runner command/log/SHA256, kernel, CUDA/TensorRT/cuDNN/GPU/driver metadata, and owner review."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
  [ordered]@{
    proofId = "real-model-runtime"
    currentState = $sampleRunEvidenceValidatorState
    templatePath = "artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json"
    expectedValidatedRecord = "artifacts/user-acceptance/sample-run-evidence-record.json"
    validatorCommand = "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    canPromote = $sampleRunEvidenceCanPromoteRealModelRuntime
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    missingOwnerInput = "Real model assets, licenses, ONNX/engine/input/output hashes, sample runner logs, and sidecar audit resolution."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
  [ordered]@{
    proofId = "real-case-proof-pack"
    currentState = $realCaseProofExecutionPackState
    templatePath = "artifacts/final-release/real-case-evidence-record-template.json"
    expectedValidatedRecord = "artifacts/final-release/real-case-evidence-record.json"
    validatorCommand = "Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json"
    canPromote = ($realCaseProofExecutionPackCanPromote -and $realCaseEvidenceRecordCanPromote)
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    missingOwnerInput = "YoloVision/OnnxToEngine/TensorRtExec real case assets, logs, screenshots, SHA256 values, host metadata, and owner review."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
  [ordered]@{
    proofId = "post-publish-verification"
    currentState = $postPublishVerificationValidatorState
    templatePath = "artifacts/final-release/post-publish-verification-record-template.json"
    expectedValidatedRecord = "artifacts/final-release/post-publish-verification-record.json"
    validatorCommand = "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
    canPromote = $postPublishVerificationIsProof
    canPublishPublicly = $false
    canCloseReleaseIssue = $postPublishVerificationCanCloseReleaseIssue
    missingOwnerInput = "Published package channel, downloaded nupkg SHA256 values, clean consumer restore/build/run logs, and post-publish runtime smoke proof."
    nonSubstituteProofKinds = $nonSubstituteProofKinds
  }
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-freeze-final-verification"
  verificationState = "blocked-real-proof-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  ownerActionStatus = "owner-action-required"
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isRealModelRuntimeProof = $false
  isPostPublishVerificationProof = $false
  releaseCloseGapDashboardState = [string](Get-PropertyOrDefault -Object $releaseCloseGapDashboard -Name "dashboardState" -DefaultValue "missing-release-close-gap-dashboard")
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  releasePackageProofBundleState = [string](Get-PropertyOrDefault -Object $releasePackageProofBundle -Name "proofState" -DefaultValue "missing-release-package-proof-bundle")
  releaseClosePreflightState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
  releaseCandidateFinalEvidenceFreezeState = [string](Get-PropertyOrDefault -Object $releaseCandidateFinalEvidenceFreeze -Name "freezeState" -DefaultValue "missing-release-candidate-final-evidence-freeze")
  sourceArtifacts = $sourceArtifacts
  releaseBlockerCount = @($blockerIds).Count
  releaseBlockers = $blockerIds
  gapItems = $gapItems
  remainingBlockers = $remainingBlockers
  preflightItems = $preflightItems
  validatorCommands = $validatorCommands
  nonSubstituteProofKinds = $nonSubstituteProofKinds
  deferredSafetyTriageState = $triageState
  deferredSafetyTriageKind = $triageKind
  deferredSafetyTriageTotalRows = $triageTotalRows
  deferredSafetyTierAImmediateSafeCount = $tierA
  deferredSafetyTierBSafeAlternativeCount = $tierB
  deferredSafetyTierCDesignGateCount = $tierC
  deferredSafetyTierDKeepDeferredCount = $tierD
  deferredSafetyTriageProofBoundary = $triageBoundary
  deferredBTierProofClosureDashboardState = $bTierClosureDashboardState
  deferredBTierProofClosureDashboardTotalCount = $bTierClosureDashboardTotalCount
  deferredBTierProofClosureDashboardSelectedCount = $bTierClosureDashboardSelectedCount
  deferredBTierProofClosureDashboardIsReleaseProof = $false
  deferredBTierProofClosureDashboardCanPublishPublicly = $false
  deferredBTierProofClosureDashboardCanCloseReleaseIssue = $false
  deferredBTierProofClosureDashboardCanDeleteDeferredRecords = $false
  deferredBTierProofClosureDashboardBoundary = $bTierClosureDashboardBoundary
  deferredBTierAliasProofClosureRecordState = $bTierAliasClosureState
  deferredBTierAliasProofClosureRecordCandidateCount = $bTierAliasClosureCandidateCount
  deferredBTierAliasProofClosureRecordIsReleaseProof = $false
  deferredBTierAliasProofClosureRecordCanPublishPublicly = $false
  deferredBTierAliasProofClosureRecordCanCloseReleaseIssue = $false
  deferredBTierAliasProofClosureRecordCanDeleteDeferredRecords = $false
  deferredBTierAliasProofClosureRecordBoundary = $bTierAliasClosureBoundary
  releaseRuntimeProofExecutionMatrixState = $runtimeProofExecutionMatrixState
  releaseRuntimeProofExecutionMatrixProofItemCount = $runtimeProofExecutionMatrixProofItemCount
  releaseRuntimeProofExecutionMatrixBlockedProofItemCount = $runtimeProofExecutionMatrixBlockedCount
  releaseRuntimeProofExecutionMatrixIsReleaseProofComplete = $false
  releaseRuntimeProofExecutionMatrixCanPublishPublicly = $false
  releaseRuntimeProofExecutionMatrixCanCloseReleaseIssue = $false
  releaseRuntimeProofExecutionMatrixBoundary = $runtimeProofExecutionMatrixBoundary
  releaseProofReadinessSnapshotState = $releaseProofReadinessSnapshotState
  releaseProofReadinessSnapshotItemCount = $releaseProofReadinessSnapshotItemCount
  releaseProofReadinessSnapshotReadyProofItemCount = $releaseProofReadinessSnapshotReadyCount
  releaseProofReadinessSnapshotBlockedProofItemCount = $releaseProofReadinessSnapshotBlockedCount
  releaseProofReadinessSnapshotIsReleaseProofComplete = $releaseProofReadinessSnapshotIsComplete
  releaseProofReadinessSnapshotCanPublishPublicly = $releaseProofReadinessSnapshotCanPublish
  releaseProofReadinessSnapshotCanCloseReleaseIssue = $releaseProofReadinessSnapshotCanClose
  releaseProofReadinessSnapshotBoundary = $releaseProofReadinessSnapshotBoundary
  ownerProofTemplateCoverageState = $runtimeProofExecutionMatrixTemplateCoverageState
  ownerProofTemplateCoverageCount = $runtimeProofExecutionMatrixExpectedInputTemplateCount
  ownerProofValidatedRecordCoverageCount = $runtimeProofExecutionMatrixExpectedValidatedRecordCount
  validatorCoverageState = $runtimeProofExecutionMatrixValidatorCoverageState
  validatorCoverageCount = $runtimeProofExecutionMatrixValidatorCommandCount
  ownerProofExecutionArticleCoverageState = $runtimeProofExecutionMatrixArticleCoverageState
  ownerProofExecutionArticleMappedCount = $runtimeProofExecutionMatrixArticleMappedProofItemCount
  releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount = $runtimeProofExecutionMatrixPostPublishRequiredEvidenceCount
  releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidence = $runtimeProofExecutionMatrixPostPublishRequiredEvidence
  ownerAuthorizationProofGateState = $ownerAuthorizationProofGateState
  ownerAuthorizationRequiredFieldCount = $ownerAuthorizationRequiredFieldCount
  ownerAuthorizationMissingOwnerInputCount = $ownerAuthorizationMissingOwnerInputCount
  manualMaterializationPrerequisiteCount = $manualMaterializationPrerequisiteCount
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidenceCount
  ownerAuthorizationRequiredFields = $ownerAuthorizationRequiredFields
  manualMaterializationPrerequisites = $manualMaterializationPrerequisites
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  packageConsumerRuntimeProofTemplateState = $packageConsumerRuntimeProofTemplateState
  packageConsumerRuntimeProofTemplateKind = $packageConsumerRuntimeProofTemplateKind
  packageConsumerRuntimeValidatorState = $packageConsumerRuntimeValidatorState
  packageConsumerRuntimeProofClassification = $packageConsumerRuntimeProofClassification
  packageConsumerRuntimeProofCanPromote = $packageConsumerRuntimeProofCanPromote
  packageConsumerProofExecutionPackPath = "artifacts/final-release/package-consumer-runtime-proof-execution-pack.json"
  packageConsumerProofPackState = $packageConsumerProofExecutionPackState
  packageConsumerProofPackValidationState = $packageConsumerProofPackValidationState
  packageConsumerProofMissingOwnerInputCount = $packageConsumerProofExecutionPackMissingOwnerInputCount
  packageConsumerProofPackCanPromote = ($packageConsumerProofExecutionPackCanPromote -and $packageConsumerProofPackValidationCanPromote)
  packageConsumerRuntimeProofCanPublishPublicly = $false
  packageConsumerRuntimeProofCanCloseReleaseIssue = $false
  linuxRunnerProofTemplateState = $linuxRunnerProofTemplateState
  linuxRunnerProofTemplateKind = $linuxRunnerProofTemplateKind
  linuxRunnerValidatorState = $linuxRunnerValidatorState
  linuxRunnerProofCanPromote = $linuxRunnerProofCanPromote
  linuxRunnerProofIsReal = $linuxRunnerProofIsReal
  linuxRunnerProofExecutionPackPath = "artifacts/final-release/linux-runner-proof-execution-pack.json"
  linuxRunnerProofPackState = $linuxRunnerProofExecutionPackState
  linuxRunnerProofPackValidationState = $linuxRunnerProofPackValidationState
  linuxRunnerProofMissingOwnerInputCount = $linuxRunnerProofExecutionPackMissingOwnerInputCount
  linuxRunnerProofPackCanPromote = ($linuxRunnerProofExecutionPackCanPromote -and $linuxRunnerProofPackValidationCanPromote)
  linuxRunnerProofCanPublishPublicly = $false
  linuxRunnerProofCanCloseReleaseIssue = $false
  realModelRuntimeProofInputPackageState = $realModelRuntimeProofInputPackageState
  realModelRuntimeProofInputPackageKind = $realModelRuntimeProofInputPackageKind
  sampleRunEvidenceValidatorState = $sampleRunEvidenceValidatorState
  sampleRunEvidenceProofClassification = $sampleRunEvidenceProofClassification
  sampleRunEvidenceTemplateOnly = $sampleRunEvidenceTemplateOnly
  sampleRunEvidenceCanPromoteRealModelRuntime = $sampleRunEvidenceCanPromoteRealModelRuntime
  sampleRunEvidenceRealModelEvidenceReady = $sampleRunEvidenceRealModelEvidenceReady
  sampleRunEvidenceOwnerActionRequiredCount = $sampleRunEvidenceOwnerActionRequiredCount
  onnxEngineBuildEvidenceSidecarAuditState = $onnxEngineBuildEvidenceSidecarAuditState
  onnxEngineBuildEvidenceSidecarOwnerActionRequiredCount = $onnxEngineBuildEvidenceSidecarOwnerActionRequiredCount
  onnxEngineBuildEvidenceSidecarErrorCount = $onnxEngineBuildEvidenceSidecarErrorCount
  realCaseProofExecutionPackPath = "artifacts/final-release/real-case-proof-execution-pack.json"
  realCaseEvidenceRecordTemplatePath = "artifacts/final-release/real-case-evidence-record-template.json"
  realCaseEvidenceValidatorCommand = "Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json"
  realCaseProofPackState = $realCaseProofExecutionPackState
  realCaseProofCaseCount = $realCaseProofExecutionPackCaseCount
  realCaseProofBlockedCaseCount = $realCaseProofExecutionPackBlockedCaseCount
  realCaseProofMissingOwnerInputCount = $realCaseProofExecutionPackMissingOwnerInputCount
  realCaseProofCanPromote = ($realCaseProofExecutionPackCanPromote -and $realCaseEvidenceRecordCanPromote)
  realCaseEvidenceRecordTemplateState = $realCaseEvidenceRecordTemplateState
  realCaseEvidenceRecordTemplateKind = $realCaseEvidenceRecordTemplateKind
  realCaseEvidenceValidatorState = $realCaseEvidenceRecordValidatorState
  realCaseEvidenceProofClassification = $realCaseEvidenceRecordProofClassification
  realModelRuntimeProofCanPromote = $false
  realModelRuntimeProofCanPublishPublicly = $false
  realModelRuntimeProofCanCloseReleaseIssue = $false
  postPublishVerificationTemplateState = $postPublishVerificationTemplateState
  postPublishVerificationTemplateKind = $postPublishVerificationTemplateKind
  postPublishVerificationTemplateOnly = $postPublishVerificationTemplateOnly
  postPublishVerificationValidatorState = $postPublishVerificationValidatorState
  postPublishVerificationProofClassification = $postPublishVerificationProofClassification
  postPublishVerificationIsProof = $postPublishVerificationIsProof
  postPublishVerificationCanCloseReleaseIssue = $postPublishVerificationCanCloseReleaseIssue
  postPublishVerificationFailedBlockerCount = $postPublishVerificationFailedBlockerCount
  postPublishVerificationFailedProofItemCount = $postPublishVerificationFailedProofItemCount
  postPublishVerificationCanPublishPublicly = $false
  releaseProofFinalAuditItemCount = @($releaseProofFinalAuditItems).Count
  releaseProofFinalAuditBlockedItemCount = @($releaseProofFinalAuditItems | Where-Object { -not [bool]$_.canPromote }).Count
  releaseProofFinalAuditItems = $releaseProofFinalAuditItems
  releaseProofBlockers = $blockerIds
  proofNonSubstituteBoundary = "Templates, drafts, runbooks, local feeds, ProjectReference consumers, dependency probes, build-only reports, sidecars, deferred safety triage, B-tier alias closure records, and release freeze summaries cannot substitute owner authorization, package-consumer-runtime, Linux runner, real-model-runtime, or post-publish verification proof."
  deferredSafetyTriageSummary = [ordered]@{
    state = $triageState
    kind = $triageKind
    totalRows = $triageTotalRows
    tierAImmediateSafeCount = $tierA
    tierBSafeAlternativeOrAliasCount = $tierB
    tierCDesignGateRequiredCount = $tierC
    tierDKeepDeferredCount = $tierD
    isReleaseProof = $false
    isPackageProof = $false
    isRuntimeExecutionProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canDeleteDeferredRecords = $false
    boundary = $triageBoundary
  }
  ownerActionRequired = $ownerActions
  nextOwnerActions = $ownerActions
  yoloVisionScope = "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom"
  yoloVisionTaskScope = "det/cls/seg/obb/pose/sem; det、cls、seg、obb、pose、sem"
  safetyNotes = @(
    "This final verification is a release freeze front door and never performs publish actions.",
    "canPublishPublicly=false and canCloseReleaseIssue=false remain fixed until real owner/runtime/post-publish proof records pass.",
    "Deferred safety triage is not release proof, not runtime proof, and not permission to delete deferred records.",
    "B-tier proof closure dashboard is engineering closure input only and cannot unlock release publication or issue closure.",
    "B-tier alias proof closure record documents engineering alias closure only and keeps deferred history intact.",
    "Release runtime proof execution matrix is a blocked execution plan until real owner/runtime/post-publish proof records pass.",
    "Release proof readiness snapshot is a 5-blocker status view only and cannot publish packages or close the release issue.",
    "Owner proof templates, validators, and article mapping are coverage scaffolding only and do not promote release proof.",
    "Package-consumer and Linux runner templates remain blocked until real logs, hashes, host metadata, and validators pass.",
    "Real-model runtime remains blocked until real model/sample logs, hashes, licenses, sidecar audit, and sample-run validator pass.",
    "Post-publish verification remains blocked until a real published channel package is consumed by a clean external project and validator passes.",
    "B-tier safe-alternative rows can feed proof closure planning; C/D rows remain design-gated or deferred."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-freeze-final-verification.json"
$markdownPath = Join-Path $artifactRoot "release-freeze-final-verification.md"

$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$blockerLines = $blockerIds | ForEach-Object { "- ``$_``" }
$validatorLines = $validatorCommands | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }
$preflightRows = $preflightItems | ForEach-Object {
  $id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
  $passed = [string](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue "")
  $currentStatus = ([string](Get-PropertyOrDefault -Object $_ -Name "currentStatus" -DefaultValue "")).Replace("|", "\|")
  "| ``$id`` | ``$passed`` | $currentStatus |"
}
$ownerAuthorizationFieldLines = $ownerAuthorizationRequiredFields | ForEach-Object { "- ``$_``" }
$manualMaterializationRows = $manualMaterializationPrerequisites | ForEach-Object {
  $id = ([string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")).Replace("|", "\|")
  $blocks = [string](Get-PropertyOrDefault -Object $_ -Name "blocksMaterialization" -DefaultValue "")
  $currentState = ([string](Get-PropertyOrDefault -Object $_ -Name "currentState" -DefaultValue "")).Replace("|", "\|")
  $requiredEvidence = ([string](Get-PropertyOrDefault -Object $_ -Name "requiredEvidence" -DefaultValue "")).Replace("|", "\|")
  "| ``$id`` | ``$blocks`` | $currentState | $requiredEvidence |"
}
$postPublishEvidenceLines = $postPublishRequiredEvidence | ForEach-Object { "- ``$_``" }
$finalAuditRows = $releaseProofFinalAuditItems | ForEach-Object {
  $proofId = ([string]$_.proofId).Replace("|", "\|")
  $currentState = ([string]$_.currentState).Replace("|", "\|")
  $templatePath = ([string]$_.templatePath).Replace("|", "\|")
  $expectedValidatedRecord = ([string]$_.expectedValidatedRecord).Replace("|", "\|")
  $validatorCommand = ([string]$_.validatorCommand).Replace("|", "\|")
  $missingOwnerInput = ([string]$_.missingOwnerInput).Replace("|", "\|")
  "| ``$proofId`` | ``$currentState`` | ``$templatePath`` | ``$expectedValidatedRecord`` | ``$validatorCommand`` | ``$($_.canPromote)`` | ``$($_.canPublishPublicly)`` | ``$($_.canCloseReleaseIssue)`` | $missingOwnerInput |"
}
$nonSubstituteLines = $nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$ownerActionLines = $ownerActions | ForEach-Object { "- $_" }
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }

$markdown = @"
# Release Freeze Final Verification

生成时间：$($record.generatedAtUtc)

## 总结

``recordKind=release-freeze-final-verification``，``verificationState=blocked-real-proof-required``，``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。该产物只做发布冻结解除前的最终核验汇总，不上传包、不执行发布命令、不关闭 release issue。

## Proof 边界

- runtime execution proof: ``False``
- package consumer runtime proof: ``False``
- real model runtime proof: ``False``
- post-publish verification proof: ``False``
- owner action status: ``owner-action-required``

## Release Blockers

$($blockerLines -join "`r`n")

## Validator Commands

$($validatorLines -join "`r`n")

## Deferred Safety Triage

- state: ``$triageState``
- kind: ``$triageKind``
- total rows: ``$triageTotalRows``
- A immediate-safe: ``$tierA``
- B safe-alternative-or-alias: ``$tierB``
- C design-gate-required: ``$tierC``
- D keep-deferred: ``$tierD``
- boundary: $triageBoundary

## B-tier Proof Closure Dashboard

- state: ``$bTierClosureDashboardState``
- total B-tier rows: ``$bTierClosureDashboardTotalCount``
- selected closure candidates: ``$bTierClosureDashboardSelectedCount``
- is release proof: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- canDeleteDeferredRecords=false
- boundary: $bTierClosureDashboardBoundary

## B-tier Alias Proof Closure Record

- state: ``$bTierAliasClosureState``
- closure candidates: ``$bTierAliasClosureCandidateCount``
- is release proof: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- canDeleteDeferredRecords=false
- boundary: $bTierAliasClosureBoundary

## Runtime Proof Execution Matrix

- state: ``$runtimeProofExecutionMatrixState``
- proof item count: ``$runtimeProofExecutionMatrixProofItemCount``
- blocked proof item count: ``$runtimeProofExecutionMatrixBlockedCount``
- is release proof complete: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- owner proof template coverage state: ``$runtimeProofExecutionMatrixTemplateCoverageState``
- owner proof template count: ``$runtimeProofExecutionMatrixExpectedInputTemplateCount``
- owner proof validated record count: ``$runtimeProofExecutionMatrixExpectedValidatedRecordCount``
- validator coverage state: ``$runtimeProofExecutionMatrixValidatorCoverageState``
- validator command count: ``$runtimeProofExecutionMatrixValidatorCommandCount``
- article coverage state: ``$runtimeProofExecutionMatrixArticleCoverageState``
- article mapped proof item count: ``$runtimeProofExecutionMatrixArticleMappedProofItemCount``
- post-publish required evidence count: ``$runtimeProofExecutionMatrixPostPublishRequiredEvidenceCount``
- boundary: $runtimeProofExecutionMatrixBoundary

## Release Proof Readiness Snapshot

- state: ``$releaseProofReadinessSnapshotState``
- readiness item count: ``$releaseProofReadinessSnapshotItemCount``
- ready proof item count: ``$releaseProofReadinessSnapshotReadyCount``
- blocked proof item count: ``$releaseProofReadinessSnapshotBlockedCount``
- is release proof complete: ``$releaseProofReadinessSnapshotIsComplete``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- boundary: $releaseProofReadinessSnapshotBoundary

## Owner Authorization Proof Gate

- gate state: ``$ownerAuthorizationProofGateState``
- required field count: ``$ownerAuthorizationRequiredFieldCount``
- missing owner input count: ``$ownerAuthorizationMissingOwnerInputCount``
- manual materialization prerequisite count: ``$manualMaterializationPrerequisiteCount``
- post-publish required evidence count: ``$postPublishRequiredEvidenceCount``
- canPublishPublicly=false
- canCloseReleaseIssue=false

### Owner Authorization Required Fields

$($ownerAuthorizationFieldLines -join "`r`n")

### Manual Materialization Prerequisites

| ID | Blocks materialization | Current state | Required evidence |
|---|---:|---|---|
$($manualMaterializationRows -join "`r`n")

### Post-Publish Required Evidence

$($postPublishEvidenceLines -join "`r`n")

## Release Proof Final Audit

- audit item count: ``$($record.releaseProofFinalAuditItemCount)``
- blocked audit item count: ``$($record.releaseProofFinalAuditBlockedItemCount)``
- canPublishPublicly=false
- canCloseReleaseIssue=false

| Proof ID | Current state | Template | Expected validated record | Validator command | Can promote | Can publish | Can close issue | Missing owner input |
|---|---|---|---|---|---:|---:|---:|---|
$($finalAuditRows -join "`r`n")

## Package Consumer Runtime Proof

- template kind: ``$packageConsumerRuntimeProofTemplateKind``
- template state: ``$packageConsumerRuntimeProofTemplateState``
- validator state: ``$packageConsumerRuntimeValidatorState``
- proof classification: ``$packageConsumerRuntimeProofClassification``
- can promote runtime proof: ``$packageConsumerRuntimeProofCanPromote``
- proof pack state: ``$packageConsumerProofExecutionPackState``
- proof pack validation state: ``$packageConsumerProofPackValidationState``
- proof pack missing owner input count: ``$packageConsumerProofExecutionPackMissingOwnerInputCount``
- proof pack can promote package-consumer-runtime: ``$($packageConsumerProofExecutionPackCanPromote -and $packageConsumerProofPackValidationCanPromote)``
- canPublishPublicly=false
- canCloseReleaseIssue=false

## Linux Runner Proof

- template kind: ``$linuxRunnerProofTemplateKind``
- template state: ``$linuxRunnerProofTemplateState``
- validator state: ``$linuxRunnerValidatorState``
- can promote Linux package: ``$linuxRunnerProofCanPromote``
- is real Linux runner proof: ``$linuxRunnerProofIsReal``
- proof pack state: ``$linuxRunnerProofExecutionPackState``
- proof pack validation state: ``$linuxRunnerProofPackValidationState``
- proof pack missing owner input count: ``$linuxRunnerProofExecutionPackMissingOwnerInputCount``
- proof pack can promote Linux runner proof: ``$($linuxRunnerProofExecutionPackCanPromote -and $linuxRunnerProofPackValidationCanPromote)``
- canPublishPublicly=false
- canCloseReleaseIssue=false

## Real Model Runtime Proof

- input package kind: ``$realModelRuntimeProofInputPackageKind``
- input package state: ``$realModelRuntimeProofInputPackageState``
- sample-run validator state: ``$sampleRunEvidenceValidatorState``
- sample-run proof classification: ``$sampleRunEvidenceProofClassification``
- sample-run template only: ``$sampleRunEvidenceTemplateOnly``
- sample-run real model evidence ready: ``$sampleRunEvidenceRealModelEvidenceReady``
- sample-run can promote real-model-runtime: ``$sampleRunEvidenceCanPromoteRealModelRuntime``
- sidecar audit state: ``$onnxEngineBuildEvidenceSidecarAuditState``
- sidecar owner action required count: ``$onnxEngineBuildEvidenceSidecarOwnerActionRequiredCount``
- sidecar error count: ``$onnxEngineBuildEvidenceSidecarErrorCount``
- real case proof pack state: ``$realCaseProofExecutionPackState``
- real case proof case count: ``$realCaseProofExecutionPackCaseCount``
- real case blocked case count: ``$realCaseProofExecutionPackBlockedCaseCount``
- real case missing owner input count: ``$realCaseProofExecutionPackMissingOwnerInputCount``
- real case evidence validator state: ``$realCaseEvidenceRecordValidatorState``
- real case can promote real-model-runtime: ``$($realCaseProofExecutionPackCanPromote -and $realCaseEvidenceRecordCanPromote)``
- canPublishPublicly=false
- canCloseReleaseIssue=false

## Post Publish Verification Proof

- template kind: ``$postPublishVerificationTemplateKind``
- template state: ``$postPublishVerificationTemplateState``
- template only: ``$postPublishVerificationTemplateOnly``
- validator state: ``$postPublishVerificationValidatorState``
- proof classification: ``$postPublishVerificationProofClassification``
- is post-publish verification proof: ``$postPublishVerificationIsProof``
- failed blocker count: ``$postPublishVerificationFailedBlockerCount``
- failed proof item count: ``$postPublishVerificationFailedProofItemCount``
- can close release issue from post-publish proof: ``$postPublishVerificationCanCloseReleaseIssue``
- canPublishPublicly=false

## Proof Non-Substitute Boundary

$($record.proofNonSubstituteBoundary)

## Preflight Items

| ID | Passed | Current status |
|---|---|---|
$($preflightRows -join "`r`n")

## YoloVision 范围

YoloVision 真实样例 proof 范围固定为 ``YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom``，任务覆盖 ``det/cls/seg/obb/pose/sem``（``det、cls、seg、obb、pose、sem``）。这些 proof 只能晋级 ``real-model-runtime``，不能替代 ``package-consumer-runtime``。

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

Write-Output "Release freeze final verification written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "VerificationState=$($record.verificationState)"
Write-Output "ReleaseBlockerCount=$($record.releaseBlockerCount)"
Write-Output "DeferredSafetyTriageTotalRows=$triageTotalRows"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
