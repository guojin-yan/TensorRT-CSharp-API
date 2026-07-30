[CmdletBinding()]
param(
  [string]$FinalGatePath = "artifacts/final-release/final-publish-proof-gate-report.json",
  [string]$YoloVisionIntakePath = "artifacts/user-acceptance/yolovision-owner-real-evidence-intake-dashboard.json",
  [string]$YoloVisionBackfillPath = "artifacts/user-acceptance/yolovision-owner-evidence-batch-backfill-pack.json",
  [string]$PackageConsumerDualRouteProofPlanPath = "artifacts/final-release/package-consumer-dual-route-proof-plan.json",
  [string]$CleanExternalConsumerExecutionKitPath = "artifacts/final-release/clean-external-consumer-execution-kit.json",
  [string]$PostPublishVerificationIntakeMapPath = "artifacts/final-release/post-publish-verification-intake-map.json",
  [string]$ReleaseCloseStrictProofExecutionOrderPath = "artifacts/final-release/release-close-strict-proof-execution-order.json",
  [string]$ArticlePublishingReadinessMapPath = "artifacts/final-release/article-publishing-readiness-map.json",
  [string]$OwnerRealEvidenceImportPacketPath = "artifacts/final-release/owner-real-evidence-import-packet.json",
  [string]$OwnerRealEvidenceImportPacketValidationPath = "artifacts/final-release/owner-real-evidence-import-packet-validation.json",
  [string]$PublicPublishAuthorizationPreflightPath = "artifacts/final-release/public-publish-authorization-preflight.json",
  [string]$PublicPublishAuthorizationPreflightValidationPath = "artifacts/final-release/public-publish-authorization-preflight-validation.json",
  [string]$OwnerRealEvidenceInputImportPath = "artifacts/final-release/owner-real-evidence-input-import.json",
  [string]$OwnerRealEvidenceInputValidationPath = "artifacts/final-release/owner-real-evidence-input-validation.json",
  [string]$OwnerRealEvidenceAcceptanceDashboardPath = "artifacts/final-release/owner-real-evidence-acceptance-dashboard.json",
  [string]$OwnerOnlyPublishExecutionCandidatePath = "artifacts/final-release/owner-only-publish-execution-candidate.json",
  [string]$OwnerOnlyPublishExecutionCandidateValidationPath = "artifacts/final-release/owner-only-publish-execution-candidate-validation.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ActionMapItem {
  param(
    [string]$Id,
    [string]$Detail,
    [string]$ProofLane,
    [string]$OwnerInputArtifact,
    [string]$RequiredRecord,
    [string]$Validator,
    [string]$FirstCommand,
    [string[]]$ForbiddenSubstitutes,
    [string[]]$RouteIds = @(),
    [string]$ExecutionKit = "",
    [string]$PlanningArtifact = "",
    [string]$PublicSourceRequirement = "",
    [string]$IntakeMapArtifact = ""
  )

  [pscustomobject]@{
    id = $Id
    proofLane = $ProofLane
    currentState = "action-required"
    ownerInputArtifact = $OwnerInputArtifact
    requiredRecord = $RequiredRecord
    validator = $Validator
    firstCommand = $FirstCommand
    routeIds = @($RouteIds)
    executionKit = $ExecutionKit
    planningArtifact = $PlanningArtifact
    intakeMapArtifact = $IntakeMapArtifact
    publicSourceRequirement = $PublicSourceRequirement
    blockingReason = $Detail
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    whyNotPublishableYet = "Required real owner evidence has not passed strict validation; publish, close, and proof-promotion flags must remain false."
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromotePackageConsumerRuntime = $false
    canPromoteRuntimeProof = $false
  }
}

$finalGate = Read-JsonOrNull $FinalGatePath
if ($null -eq $finalGate) { throw "Final gate report missing: $FinalGatePath" }

$intake = Read-JsonOrNull $YoloVisionIntakePath
$backfill = Read-JsonOrNull $YoloVisionBackfillPath
$packageConsumerDualRoutePlan = Read-JsonOrNull $PackageConsumerDualRouteProofPlanPath
$cleanExternalConsumerKit = Read-JsonOrNull $CleanExternalConsumerExecutionKitPath
$postPublishIntakeMap = Read-JsonOrNull $PostPublishVerificationIntakeMapPath
$releaseCloseStrictOrder = Read-JsonOrNull $ReleaseCloseStrictProofExecutionOrderPath
$articlePublishingReadinessMap = Read-JsonOrNull $ArticlePublishingReadinessMapPath
$ownerRealEvidenceImportPacket = Read-JsonOrNull $OwnerRealEvidenceImportPacketPath
$ownerRealEvidenceImportPacketValidation = Read-JsonOrNull $OwnerRealEvidenceImportPacketValidationPath
$publicPublishAuthorizationPreflight = Read-JsonOrNull $PublicPublishAuthorizationPreflightPath
$publicPublishAuthorizationPreflightValidation = Read-JsonOrNull $PublicPublishAuthorizationPreflightValidationPath
$ownerRealEvidenceInputImport = Read-JsonOrNull $OwnerRealEvidenceInputImportPath
$ownerRealEvidenceInputValidation = Read-JsonOrNull $OwnerRealEvidenceInputValidationPath
$ownerRealEvidenceAcceptanceDashboard = Read-JsonOrNull $OwnerRealEvidenceAcceptanceDashboardPath
$ownerOnlyPublishExecutionCandidate = Read-JsonOrNull $OwnerOnlyPublishExecutionCandidatePath
$ownerOnlyPublishExecutionCandidateValidation = Read-JsonOrNull $OwnerOnlyPublishExecutionCandidateValidationPath
$actionItems = @($finalGate.validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "action-required" })
$commonForbidden = @(
  "local package-feed result",
  "project-reference run",
  "direct nupkg run",
  "template-only record",
  "build-only report",
  "TensorRtExec report",
  "OnnxToEngine report",
  "YoloVision matrix",
  "screenshot-only evidence"
)

$mapped = @()
foreach ($item in $actionItems) {
  $id = [string]$item.id
  $detail = [string]$item.detail
  switch ($id) {
    "real-model-runtime-owner-proof-required" {
      $mapped += New-ActionMapItem -Id $id -Detail $detail -ProofLane "real-model-runtime" -OwnerInputArtifact "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json" -RequiredRecord "artifacts/final-release/real-case-evidence-record.json" -Validator "eng/Test-RealCaseEvidenceRecord.ps1 -FailOnNotProof; eng/Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict" -FirstCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-YoloVisionOwnerRealEvidenceIntakeDashboard.ps1" -ForbiddenSubstitutes $commonForbidden
    }
    "package-consumer-runtime-owner-proof-required" {
      $routeIds = if ($null -ne $packageConsumerDualRoutePlan) {
        @($packageConsumerDualRoutePlan.routes | ForEach-Object { [string]$_.routeId })
      }
      else {
        @("github-release-managed-plus-bridge-assets", "nuget-managed-plus-bridge-packages")
      }
      $mapped += New-ActionMapItem -Id $id -Detail $detail -ProofLane "package-consumer-runtime" -OwnerInputArtifact "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json" -RequiredRecord "artifacts/final-release/package-consumer-runtime-proof-record.json" -Validator "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -FirstCommand "Create a clean consumer outside the repository and restore from public package source." -ForbiddenSubstitutes $commonForbidden -RouteIds $routeIds -ExecutionKit $CleanExternalConsumerExecutionKitPath -PlanningArtifact $PackageConsumerDualRouteProofPlanPath -PublicSourceRequirement "Owner must choose GitHub Release or NuGet-compatible managed plus bridge-only delivery and provide public package source evidence; NVIDIA dependencies remain machine-installed."
    }
    "post-publish-verification-owner-proof-required" {
      $mapped += New-ActionMapItem -Id $id -Detail $detail -ProofLane "post-publish-verification" -OwnerInputArtifact "artifacts/final-release/post-publish-verification-record.template.json" -RequiredRecord "artifacts/final-release/post-publish-verification-record.json" -Validator "eng/Test-PostPublishVerificationRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -FirstCommand "After real public publication, run install and smoke verification from the public channel." -ForbiddenSubstitutes $commonForbidden -IntakeMapArtifact $PostPublishVerificationIntakeMapPath -PublicSourceRequirement "Owner must provide public channel URL, published version, install/run commands, logs, hashes, host metadata, owner decision, and rollback reference."
    }
    "final-owner-real-input-template-pack-owner-input-required" {
      $mapped += New-ActionMapItem -Id $id -Detail $detail -ProofLane "owner-input-template-pack" -OwnerInputArtifact "artifacts/final-release/final-owner-real-input-template-pack.json" -RequiredRecord "artifacts/final-release/final-owner-real-input-template-pack-validation.json" -Validator "eng\Test-FinalPublishProofGate.ps1 -Strict" -FirstCommand "Fill every configured owner input lane with real logs, hashes, exit codes, host metadata, package metadata, and owner review." -ForbiddenSubstitutes $commonForbidden
    }
    "owner-external-proof-result-import-owner-proof-required" {
      $mapped += New-ActionMapItem -Id $id -Detail $detail -ProofLane "owner-external-proof-result-import" -OwnerInputArtifact "artifacts/final-release/owner-external-proof-execution-result-input.json" -RequiredRecord "artifacts/final-release/owner-external-proof-execution-result-import-validation.json" -Validator "eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict" -FirstCommand "Import owner external execution result with existing logs and matching SHA256 values." -ForbiddenSubstitutes $commonForbidden
    }
    "owner-result-candidate-bridge-real-proof-required" {
      $mapped += New-ActionMapItem -Id $id -Detail $detail -ProofLane "owner-result-candidate-bridge" -OwnerInputArtifact "artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json" -RequiredRecord "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json" -Validator "eng/Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict" -FirstCommand "Bridge only strict-validator-ready owner records into real proof candidates." -ForbiddenSubstitutes $commonForbidden
    }
    default {
      $mapped += New-ActionMapItem -Id $id -Detail $detail -ProofLane "unmapped-owner-action" -OwnerInputArtifact "owner-action-required" -RequiredRecord "owner-action-required" -Validator "eng\Test-FinalPublishProofGate.ps1 -Strict" -FirstCommand "Inspect final gate action-required detail and fill real owner evidence." -ForbiddenSubstitutes $commonForbidden
    }
  }
}

$map = [pscustomobject]@{
  recordKind = "final-publish-action-required-evidence-map"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  mapState = "blocked-real-owner-evidence-required"
  sourceFinalGate = $FinalGatePath
  sourceYoloVisionIntake = $YoloVisionIntakePath
  sourceYoloVisionBackfill = $YoloVisionBackfillPath
  sourcePackageConsumerDualRouteProofPlan = $PackageConsumerDualRouteProofPlanPath
  sourceCleanExternalConsumerExecutionKit = $CleanExternalConsumerExecutionKitPath
  sourcePostPublishVerificationIntakeMap = $PostPublishVerificationIntakeMapPath
  sourceReleaseCloseStrictProofExecutionOrder = $ReleaseCloseStrictProofExecutionOrderPath
  sourceArticlePublishingReadinessMap = $ArticlePublishingReadinessMapPath
  sourceOwnerRealEvidenceImportPacket = $OwnerRealEvidenceImportPacketPath
  sourceOwnerRealEvidenceImportPacketValidation = $OwnerRealEvidenceImportPacketValidationPath
  sourcePublicPublishAuthorizationPreflight = $PublicPublishAuthorizationPreflightPath
  sourcePublicPublishAuthorizationPreflightValidation = $PublicPublishAuthorizationPreflightValidationPath
  sourceOwnerRealEvidenceInputImport = $OwnerRealEvidenceInputImportPath
  sourceOwnerRealEvidenceInputValidation = $OwnerRealEvidenceInputValidationPath
  sourceOwnerRealEvidenceAcceptanceDashboard = $OwnerRealEvidenceAcceptanceDashboardPath
  sourceOwnerOnlyPublishExecutionCandidate = $OwnerOnlyPublishExecutionCandidatePath
  sourceOwnerOnlyPublishExecutionCandidateValidation = $OwnerOnlyPublishExecutionCandidateValidationPath
  finalGateState = [string]$finalGate.validationState
  failedBlockerCount = [int]$finalGate.failedBlockerCount
  actionRequiredCount = @($mapped).Count
  yoloVisionIntakeTaskCount = if ($null -ne $intake) { [int]$intake.taskCount } else { 0 }
  yoloVisionBackfillGroupCount = if ($null -ne $backfill) { [int]$backfill.groupCount } else { 0 }
  packageConsumerRouteCount = if ($null -ne $packageConsumerDualRoutePlan) { [int]$packageConsumerDualRoutePlan.routeCount } else { 0 }
  cleanExternalConsumerStepCount = if ($null -ne $cleanExternalConsumerKit) { @($cleanExternalConsumerKit.executionSteps).Count } else { 0 }
  postPublishRequiredFieldCount = if ($null -ne $postPublishIntakeMap) { [int]$postPublishIntakeMap.requiredFieldCount } else { 0 }
  postPublishIntakeState = if ($null -ne $postPublishIntakeMap) { [string]$postPublishIntakeMap.intakeState } else { "missing" }
  releaseCloseStrictExecutionStepCount = if ($null -ne $releaseCloseStrictOrder) { [int]$releaseCloseStrictOrder.executionStepCount } else { 0 }
  articlePublishingFocusedReadinessArticleCount = if ($null -ne $articlePublishingReadinessMap) { [int]$articlePublishingReadinessMap.focusedReadinessArticleCount } else { 0 }
  articlePublishingReadinessState = if ($null -ne $articlePublishingReadinessMap) { [string]$articlePublishingReadinessMap.readinessState } else { "missing" }
  ownerRealEvidenceImportLaneCount = if ($null -ne $ownerRealEvidenceImportPacket) { [int]$ownerRealEvidenceImportPacket.laneCount } else { 0 }
  ownerRealEvidenceImportValidationState = if ($null -ne $ownerRealEvidenceImportPacketValidation) { [string]$ownerRealEvidenceImportPacketValidation.validationState } else { "missing" }
  publicPublishAuthorizationRequirementCount = if ($null -ne $publicPublishAuthorizationPreflight) { [int]$publicPublishAuthorizationPreflight.requirementCount } else { 0 }
  publicPublishAuthorizationValidationState = if ($null -ne $publicPublishAuthorizationPreflightValidation) { [string]$publicPublishAuthorizationPreflightValidation.validationState } else { "missing" }
  ownerRealEvidenceInputImportState = if ($null -ne $ownerRealEvidenceInputImport) { [string]$ownerRealEvidenceInputImport.importState } else { "missing" }
  ownerRealEvidenceInputValidationState = if ($null -ne $ownerRealEvidenceInputValidation) { [string]$ownerRealEvidenceInputValidation.validationState } else { "missing" }
  ownerRealEvidenceInputAcceptedLaneCount = if ($null -ne $ownerRealEvidenceInputImport) { [int]$ownerRealEvidenceInputImport.acceptedLaneCount } else { 0 }
  ownerRealEvidenceInputBlockedLaneCount = if ($null -ne $ownerRealEvidenceInputImport) { [int]$ownerRealEvidenceInputImport.blockedLaneCount } else { 6 }
  ownerRealEvidenceAcceptanceDashboardState = if ($null -ne $ownerRealEvidenceAcceptanceDashboard) { [string]$ownerRealEvidenceAcceptanceDashboard.dashboardState } else { "missing" }
  ownerOnlyPublishExecutionCandidateState = if ($null -ne $ownerOnlyPublishExecutionCandidate) { [string]$ownerOnlyPublishExecutionCandidate.candidateState } else { "missing" }
  ownerOnlyPublishExecutionCandidateValidationState = if ($null -ne $ownerOnlyPublishExecutionCandidateValidation) { [string]$ownerOnlyPublishExecutionCandidateValidation.validationState } else { "missing" }
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  actions = @($mapped)
  boundary = "This map explains final publish action-required evidence only. It does not publish, does not close issues, and does not promote runtime proof."
}

$jsonPath = Join-Path $OutputRoot "final-publish-action-required-evidence-map.json"
$markdownPath = Join-Path $OutputRoot "final-publish-action-required-evidence-map.md"
$map | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($action in $mapped) {
  $planningOrIntakeArtifact = if (-not [string]::IsNullOrWhiteSpace([string]$action.planningArtifact)) {
    [string]$action.planningArtifact
  }
  elseif (-not [string]::IsNullOrWhiteSpace([string]$action.intakeMapArtifact)) {
    [string]$action.intakeMapArtifact
  }
  elseif (-not [string]::IsNullOrWhiteSpace([string]$action.executionKit)) {
    [string]$action.executionKit
  }
  else {
    ""
  }
  "| ``$(ConvertTo-MarkdownCell $action.id)`` | ``$(ConvertTo-MarkdownCell $action.proofLane)`` | ``$(ConvertTo-MarkdownCell $action.ownerInputArtifact)`` | ``$(ConvertTo-MarkdownCell $planningOrIntakeArtifact)`` | ``$(ConvertTo-MarkdownCell $action.validator)`` | ``False`` |"
}

$markdown = @"
# Final Publish Action Required Evidence Map

Generated at: ``$($map.generatedAtUtc)``

## Summary

- mapState: ``$($map.mapState)``
- finalGateState: ``$($map.finalGateState)``
- failedBlockerCount: ``$($map.failedBlockerCount)``
- actionRequiredCount: ``$($map.actionRequiredCount)``
- yoloVisionIntakeTaskCount: ``$($map.yoloVisionIntakeTaskCount)``
- yoloVisionBackfillGroupCount: ``$($map.yoloVisionBackfillGroupCount)``
- packageConsumerRouteCount: ``$($map.packageConsumerRouteCount)``
- cleanExternalConsumerStepCount: ``$($map.cleanExternalConsumerStepCount)``
- postPublishRequiredFieldCount: ``$($map.postPublishRequiredFieldCount)``
- postPublishIntakeState: ``$($map.postPublishIntakeState)``
- releaseCloseStrictExecutionStepCount: ``$($map.releaseCloseStrictExecutionStepCount)``
- articlePublishingFocusedReadinessArticleCount: ``$($map.articlePublishingFocusedReadinessArticleCount)``
- articlePublishingReadinessState: ``$($map.articlePublishingReadinessState)``
- ownerRealEvidenceImportLaneCount: ``$($map.ownerRealEvidenceImportLaneCount)``
- ownerRealEvidenceImportValidationState: ``$($map.ownerRealEvidenceImportValidationState)``
- publicPublishAuthorizationRequirementCount: ``$($map.publicPublishAuthorizationRequirementCount)``
- publicPublishAuthorizationValidationState: ``$($map.publicPublishAuthorizationValidationState)``
- ownerRealEvidenceInputImportState: ``$($map.ownerRealEvidenceInputImportState)``
- ownerRealEvidenceInputValidationState: ``$($map.ownerRealEvidenceInputValidationState)``
- ownerRealEvidenceInputAcceptedLaneCount: ``$($map.ownerRealEvidenceInputAcceptedLaneCount)``
- ownerRealEvidenceInputBlockedLaneCount: ``$($map.ownerRealEvidenceInputBlockedLaneCount)``
- ownerRealEvidenceAcceptanceDashboardState: ``$($map.ownerRealEvidenceAcceptanceDashboardState)``
- ownerOnlyPublishExecutionCandidateState: ``$($map.ownerOnlyPublishExecutionCandidateState)``
- ownerOnlyPublishExecutionCandidateValidationState: ``$($map.ownerOnlyPublishExecutionCandidateValidationState)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromotePackageConsumerRuntime: ``False``
- canPromoteRuntimeProof: ``False``

## Action Map

| Action | Proof Lane | Owner Input Artifact | Planning/Intake Artifact | Validator | Can Publish |
| --- | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($map.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Final publish action-required evidence map written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "MapState=$($map.mapState) ActionRequiredCount=$($map.actionRequiredCount) FailedBlockerCount=$($map.failedBlockerCount)"
