[CmdletBinding()]
param(
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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ReadinessLane {
  param(
    [string]$Id,
    [string]$State,
    [string[]]$SourceArtifacts,
    [string[]]$RequiredBeforePromotion,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    state = $State
    blocked = $true
    sourceArtifacts = @($SourceArtifacts)
    requiredBeforePromotion = @($RequiredBeforePromotion)
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

$preflight = Read-JsonOrNull "artifacts/final-release/owner-input-preflight-bundle.json"
$dashboardValidation = Read-JsonOrNull "artifacts/final-release/release-proof-dashboard-validation.json"
$dashboard = Read-JsonOrNull "artifacts/final-release/release-proof-dashboard.json"
$finalGate = Read-JsonOrNull "artifacts/final-release/final-publish-proof-gate-report.json"
$finalOwnerProofActionWorklist = Read-JsonOrNull "artifacts/final-release/final-owner-proof-action-worklist.json"
$finalOwnerProofActionWorklistValidation = Read-JsonOrNull "artifacts/final-release/final-owner-proof-action-worklist-validation.json"
$finalOwnerExecutionPackage = Read-JsonOrNull "artifacts/final-release/final-owner-execution-package.json"
$finalOwnerExecutionPackageValidation = Read-JsonOrNull "artifacts/final-release/final-owner-execution-package-validation.json"
$ownerExternalProofExecutionResultImport = Read-JsonOrNull "artifacts/final-release/owner-external-proof-execution-result-import.json"
$ownerExternalProofExecutionResultImportValidation = Read-JsonOrNull "artifacts/final-release/owner-external-proof-execution-result-import-validation.json"
$realExternalProofRecordImportValidator = Read-JsonOrNull "artifacts/final-release/real-external-proof-record-import-validator.json"
$realExternalProofRecordImportValidatorValidation = Read-JsonOrNull "artifacts/final-release/real-external-proof-record-import-validator-validation.json"
$realProofRecordCandidateFromOwnerResultImport = Read-JsonOrNull "artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json"
$realProofRecordCandidateFromOwnerResultImportValidation = Read-JsonOrNull "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json"
$releaseCloseRealProofImportBridge = Read-JsonOrNull "artifacts/final-release/release-close-real-proof-import-bridge.json"
$releaseCloseRealProofImportBridgeValidation = Read-JsonOrNull "artifacts/final-release/release-close-real-proof-import-bridge-validation.json"
$packageValidation = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json"
$postPublishValidation = Read-JsonOrNull "artifacts/final-release/post-publish-verification-validation.json"
$cleanExternalPackageConsumerOwnerRunbook = Read-JsonOrNull "artifacts/final-release/clean-external-package-consumer-owner-runbook.json"
$cleanExternalPackageConsumerOwnerRunbookValidation = Read-JsonOrNull "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json"
$postPublishOwnerVerificationRunbook = Read-JsonOrNull "artifacts/final-release/post-publish-owner-verification-runbook.json"
$postPublishOwnerVerificationRunbookValidation = Read-JsonOrNull "artifacts/final-release/post-publish-owner-verification-runbook-validation.json"
$yoloRepairPack = Read-JsonOrNull "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json"
$publicDocsGate = Read-JsonOrNull "artifacts/final-release/public-docs-package-metadata-gate.json"

$forbiddenSubstitutes = @(
  "build-only",
  "dry-run",
  "template",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "TensorRtExec report",
  "YoloVision matrix",
  "OnnxToEngine report",
  "readonly diagnostics",
  "screenshot",
  "sidecar-only report",
  "skipped run",
  "blocked-by-cuda-driver",
  "blocked-by-driver",
  "candidate",
  "dashboard",
  "sample-run-evidence",
  "package-consumer-runtime as post-publish verification"
)

$readinessLanes = @(
  New-ReadinessLane `
    -Id "real-model-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $yoloRepairPack -Name "repairPackState" -DefaultValue "blocked-owner-action-required")) `
    -SourceArtifacts @("artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json", "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json") `
    -RequiredBeforePromotion @("real model assets", "run logs", "output JSON", "SHA256 values", "host metadata", "owner review") `
    -Boundary "real-model-runtime remains blocked until owner-provided real model runtime evidence is complete."
  New-ReadinessLane `
    -Id "package-consumer-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $packageValidation -Name "validationState" -DefaultValue "blocked-owner-input-required")) `
    -SourceArtifacts @("artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json", "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json") `
    -RequiredBeforePromotion @("public package source", "clean external consumer", "no local feed", "no ProjectReference", "runtime smoke exitCode=0", "stdout/stderr review", "log SHA256") `
    -Boundary "package-consumer-runtime requires a clean external consumer and cannot be replaced by sample-run-evidence."
  New-ReadinessLane `
    -Id "post-publish-verification" `
    -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "template-only")) `
    -SourceArtifacts @("artifacts/final-release/post-publish-verification-validation.json", "artifacts/final-release/post-publish-verification-owner-input.template.json") `
    -RequiredBeforePromotion @("real public channel publish", "published package URLs", "downloaded package hashes", "clean post-publish consumer", "runtime-key smoke", "existing log hashes") `
    -Boundary "post-publish verification requires real public channel publish and cannot be replaced by pre-publish package-consumer proof."
  New-ReadinessLane `
    -Id "public-owner-confirmation" `
    -State "blocked-release-close-real-proof-required" `
    -SourceArtifacts @("artifacts/final-release/release-close-proof-lane-worklist.json", "artifacts/final-release/final-publish-proof-gate-report.json") `
    -RequiredBeforePromotion @("real-model-runtime proof", "package-consumer-runtime proof", "post-publish verification proof", "owner release close decision") `
    -Boundary "public owner confirmation is not proof by itself and can only close after all real proof lanes pass."
  New-ReadinessLane `
    -Id "public-docs-and-package-metadata" `
    -State ([string](Get-PropertyOrDefault -Object $publicDocsGate -Name "gateState" -DefaultValue "blocked-owner-public-postpublish-proof-required")) `
    -SourceArtifacts @("artifacts/final-release/public-docs-package-metadata-gate.json") `
    -RequiredBeforePromotion @("no stale live claims", "no forbidden substitute proof claims", "owner/public/post-publish boundary language") `
    -Boundary "public docs/package metadata gate prevents over-claims but does not prove runtime or publish readiness."
)

$ownerCommandSequence = @()
foreach ($group in @(Get-PropertyOrDefault -Object $preflight -Name "ownerHandoffCommands" -DefaultValue @())) {
  foreach ($command in @(Get-PropertyOrDefault -Object $group -Name "commands" -DefaultValue @())) {
    $ownerCommandSequence += [pscustomobject]@{
      lane = [string](Get-PropertyOrDefault -Object $group -Name "lane" -DefaultValue "")
      command = [string]$command
    }
  }
}
foreach ($action in @(Get-PropertyOrDefault -Object $finalOwnerProofActionWorklist -Name "actions" -DefaultValue @())) {
  foreach ($command in @(Get-PropertyOrDefault -Object $action -Name "ownerCommands" -DefaultValue @())) {
    $ownerCommandSequence += [pscustomobject]@{
      lane = [string](Get-PropertyOrDefault -Object $action -Name "laneId" -DefaultValue "")
      command = [string]$command
    }
  }
}
$ownerCommandSequence += [pscustomobject]@{ lane = "public-docs-and-package-metadata"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicDocsAndPackageMetadataGate.ps1 -Strict" }
$ownerCommandSequence += [pscustomobject]@{ lane = "final-owner-proof-action-worklist"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalOwnerProofActionWorklist.ps1" }
$ownerCommandSequence += [pscustomobject]@{ lane = "final-owner-proof-action-worklist"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalOwnerProofActionWorklist.ps1 -Strict" }
$ownerCommandSequence += [pscustomobject]@{ lane = "final-owner-execution-package"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalOwnerExecutionPackage.ps1" }
$ownerCommandSequence += [pscustomobject]@{ lane = "final-owner-execution-package"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalOwnerExecutionPackage.ps1 -Strict" }
$ownerCommandSequence += [pscustomobject]@{ lane = "clean-external-package-consumer-owner-runbook"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CleanExternalPackageConsumerOwnerRunbook.ps1" }
$ownerCommandSequence += [pscustomobject]@{ lane = "clean-external-package-consumer-owner-runbook"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CleanExternalPackageConsumerOwnerRunbook.ps1 -Strict" }
$ownerCommandSequence += [pscustomobject]@{ lane = "post-publish-owner-verification-runbook"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishOwnerVerificationRunbook.ps1" }
$ownerCommandSequence += [pscustomobject]@{ lane = "post-publish-owner-verification-runbook"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishOwnerVerificationRunbook.ps1 -Strict" }
$ownerCommandSequence += [pscustomobject]@{ lane = "release-readiness-evidence-pack"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePublishReadinessEvidencePack.ps1" }
$ownerCommandSequence += [pscustomobject]@{ lane = "final-publish-gate"; command = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalPublishProofGate.ps1 -Strict" }

$nonProofEvidenceCatalog = @(
  [pscustomobject]@{ id = "owner-input-preflight-bundle"; state = [string](Get-PropertyOrDefault -Object $preflight -Name "bundleState" -DefaultValue "missing"); boundary = "handoff only, not proof" }
  [pscustomobject]@{ id = "final-owner-proof-action-worklist"; state = [string](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklist -Name "worklistState" -DefaultValue "missing"); boundary = "one-screen owner action handoff only, not proof" }
  [pscustomobject]@{ id = "final-owner-proof-action-worklist-validation"; state = [string](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklistValidation -Name "validationState" -DefaultValue "missing"); boundary = "handoff structure validator only, not proof" }
  [pscustomobject]@{ id = "final-owner-execution-package"; state = [string](Get-PropertyOrDefault -Object $finalOwnerExecutionPackage -Name "packageState" -DefaultValue "missing"); boundary = "owner execution package only, not proof or publish approval" }
  [pscustomobject]@{ id = "final-owner-execution-package-validation"; state = [string](Get-PropertyOrDefault -Object $finalOwnerExecutionPackageValidation -Name "validationState" -DefaultValue "missing"); boundary = "execution package structure validator only, not proof" }
  [pscustomobject]@{ id = "owner-external-proof-execution-result-import"; state = [string](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImport -Name "importState" -DefaultValue "missing"); boundary = "owner result import strict-validator input only, not proof" }
  [pscustomobject]@{ id = "owner-external-proof-execution-result-import-validation"; state = [string](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "validationState" -DefaultValue "missing"); boundary = "owner result import validator only, not proof" }
  [pscustomobject]@{ id = "real-external-proof-record-import-validator"; state = [string](Get-PropertyOrDefault -Object $realExternalProofRecordImportValidator -Name "validatorState" -DefaultValue "missing"); boundary = "real proof import contract only, not proof" }
  [pscustomobject]@{ id = "real-external-proof-record-import-validator-validation"; state = [string](Get-PropertyOrDefault -Object $realExternalProofRecordImportValidatorValidation -Name "validationState" -DefaultValue "missing"); boundary = "real proof import validator output only, not proof" }
  [pscustomobject]@{ id = "real-proof-record-candidate-from-owner-result-import"; state = [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "candidateState" -DefaultValue "missing"); boundary = "strict-validator input candidate only, not runtime/post-publish/release-close proof" }
  [pscustomobject]@{ id = "real-proof-record-candidate-from-owner-result-import-validation"; state = [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "validationState" -DefaultValue "missing"); boundary = "candidate bridge shape validator only, not proof" }
  [pscustomobject]@{ id = "release-close-real-proof-import-bridge"; state = [string](Get-PropertyOrDefault -Object $releaseCloseRealProofImportBridge -Name "bridgeState" -DefaultValue "missing"); boundary = "release-close owner input bridge only, not close approval" }
  [pscustomobject]@{ id = "release-close-real-proof-import-bridge-validation"; state = [string](Get-PropertyOrDefault -Object $releaseCloseRealProofImportBridgeValidation -Name "validationState" -DefaultValue "missing"); boundary = "release-close bridge validator only, not close approval" }
  [pscustomobject]@{ id = "clean-external-package-consumer-owner-runbook"; state = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "runbookState" -DefaultValue "missing"); boundary = "owner executable clean external consumer guidance only, not runtime proof" }
  [pscustomobject]@{ id = "clean-external-package-consumer-owner-runbook-validation"; state = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbookValidation -Name "validationState" -DefaultValue "missing"); boundary = "runbook structure validator only, not runtime proof" }
  [pscustomobject]@{ id = "post-publish-owner-verification-runbook"; state = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "runbookState" -DefaultValue "missing"); boundary = "owner executable post-publish guidance only, not post-publish proof" }
  [pscustomobject]@{ id = "post-publish-owner-verification-runbook-validation"; state = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbookValidation -Name "validationState" -DefaultValue "missing"); boundary = "post-publish runbook structure validator only, not post-publish proof" }
  [pscustomobject]@{ id = "release-proof-dashboard"; state = [string](Get-PropertyOrDefault -Object $dashboard -Name "dashboardState" -DefaultValue "missing"); boundary = "navigation only, not proof" }
  [pscustomobject]@{ id = "release-proof-dashboard-validation"; state = [string](Get-PropertyOrDefault -Object $dashboardValidation -Name "validationState" -DefaultValue "missing"); boundary = "structure gate only, not proof" }
  [pscustomobject]@{ id = "public-docs-package-metadata-gate"; state = [string](Get-PropertyOrDefault -Object $publicDocsGate -Name "gateState" -DefaultValue "missing"); boundary = "claim-safety gate only, not proof" }
  [pscustomobject]@{ id = "final-publish-proof-gate-report"; state = [string](Get-PropertyOrDefault -Object $finalGate -Name "validationState" -DefaultValue "missing"); boundary = "blocking gate only, not publish action" }
)

$pack = [ordered]@{
  recordKind = "release-publish-readiness-evidence-pack"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  packState = "blocked-owner-public-postpublish-proof-required"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  ownerInputPreflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "bundleState" -DefaultValue "missing")
  dashboardValidationState = [string](Get-PropertyOrDefault -Object $dashboardValidation -Name "validationState" -DefaultValue "missing")
  finalGateState = [string](Get-PropertyOrDefault -Object $finalGate -Name "validationState" -DefaultValue "missing")
  realModelRuntimeState = [string](Get-PropertyOrDefault -Object $yoloRepairPack -Name "repairPackState" -DefaultValue "blocked-owner-action-required")
  packageConsumerRuntimeState = [string](Get-PropertyOrDefault -Object $packageValidation -Name "validationState" -DefaultValue "blocked-owner-input-required")
  postPublishVerificationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "template-only")
  publicDocsPackageMetadataGateState = [string](Get-PropertyOrDefault -Object $publicDocsGate -Name "gateState" -DefaultValue "missing")
  actionRequiredCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedActionRequiredCount" -DefaultValue 3)
  failedBlockerCount = [int](Get-PropertyOrDefault -Object $finalGate -Name "failedBlockerCount" -DefaultValue 0)
  finalOwnerProofActionWorklistState = [string](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklist -Name "worklistState" -DefaultValue "missing")
  finalOwnerProofActionWorklistValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklistValidation -Name "validationState" -DefaultValue "missing")
  finalOwnerProofActionCount = [int](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklist -Name "actionCount" -DefaultValue 0)
  finalOwnerProofBlockedActionCount = [int](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklist -Name "blockedActionCount" -DefaultValue 0)
  finalOwnerProofMissingActionRequiredIdCount = [int](Get-PropertyOrDefault -Object $finalOwnerProofActionWorklist -Name "missingActionRequiredIdCount" -DefaultValue -1)
  finalOwnerExecutionPackageState = [string](Get-PropertyOrDefault -Object $finalOwnerExecutionPackage -Name "packageState" -DefaultValue "missing")
  finalOwnerExecutionPackageValidationState = [string](Get-PropertyOrDefault -Object $finalOwnerExecutionPackageValidation -Name "validationState" -DefaultValue "missing")
  finalOwnerExecutionStepCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionPackage -Name "executionStepCount" -DefaultValue 0)
  finalOwnerBlockedExecutionStepCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionPackage -Name "blockedExecutionStepCount" -DefaultValue 0)
  finalOwnerExecutionPackageFailedBlockerCount = [int](Get-PropertyOrDefault -Object $finalOwnerExecutionPackageValidation -Name "failedBlockerCount" -DefaultValue -1)
  ownerExternalProofExecutionResultImportState = [string](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImport -Name "importState" -DefaultValue "missing")
  ownerExternalProofExecutionResultImportValidationState = [string](Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImportValidation -Name "validationState" -DefaultValue "missing")
  ownerExternalProofExecutionResultImportReadyForStrictValidatorLaneCount = [int](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImport -Name "summary" -DefaultValue $null) -Name "readyForStrictValidatorLaneCount" -DefaultValue 0)
  ownerExternalProofExecutionResultImportBlockedLaneCount = [int](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImport -Name "summary" -DefaultValue $null) -Name "blockedLaneCount" -DefaultValue 0)
  ownerExternalProofExecutionResultImportPromotableLaneCount = [int](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $ownerExternalProofExecutionResultImport -Name "summary" -DefaultValue $null) -Name "promotableLaneCount" -DefaultValue 0)
  realExternalProofRecordImportValidatorState = [string](Get-PropertyOrDefault -Object $realExternalProofRecordImportValidator -Name "validatorState" -DefaultValue "missing")
  realExternalProofRecordImportValidatorValidationState = [string](Get-PropertyOrDefault -Object $realExternalProofRecordImportValidatorValidation -Name "validationState" -DefaultValue "missing")
  realExternalProofRecordImportValidatorReadyForStrictValidatorContractCount = [int](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $realExternalProofRecordImportValidator -Name "summary" -DefaultValue $null) -Name "readyForStrictValidatorContractCount" -DefaultValue 0)
  realExternalProofRecordImportValidatorBlockedContractCount = [int](Get-PropertyOrDefault -Object $realExternalProofRecordImportValidator -Name "blockedCandidateContractCount" -DefaultValue 0)
  realProofRecordCandidateFromOwnerResultImportState = [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "candidateState" -DefaultValue "missing")
  realProofRecordCandidateFromOwnerResultImportValidationState = [string](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImportValidation -Name "validationState" -DefaultValue "missing")
  realProofRecordCandidateFromOwnerResultImportCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "candidateCount" -DefaultValue 0)
  realProofRecordCandidateFromOwnerResultImportStrictValidatorReadyCandidateCount = [int](Get-PropertyOrDefault -Object $realProofRecordCandidateFromOwnerResultImport -Name "strictValidatorReadyCandidateCount" -DefaultValue 0)
  releaseCloseRealProofImportBridgeState = [string](Get-PropertyOrDefault -Object $releaseCloseRealProofImportBridge -Name "bridgeState" -DefaultValue "missing")
  releaseCloseRealProofImportBridgeValidationState = [string](Get-PropertyOrDefault -Object $releaseCloseRealProofImportBridgeValidation -Name "validationState" -DefaultValue "missing")
  releaseCloseRealProofImportBridgeBlockedLaneCount = [int](Get-PropertyOrDefault -Object $releaseCloseRealProofImportBridgeValidation -Name "blockedLaneCount" -DefaultValue 0)
  releaseCloseRealProofImportBridgeFailedBlockerCount = [int](Get-PropertyOrDefault -Object $releaseCloseRealProofImportBridgeValidation -Name "failedBlockerCount" -DefaultValue -1)
  cleanExternalPackageConsumerOwnerRunbookState = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "runbookState" -DefaultValue "missing")
  cleanExternalPackageConsumerOwnerRunbookValidationState = [string](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbookValidation -Name "validationState" -DefaultValue "missing")
  cleanExternalPackageConsumerOwnerRunbookStepCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "stepCount" -DefaultValue 0)
  cleanExternalPackageConsumerOwnerRunbookBlockedStepCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbook -Name "blockedStepCount" -DefaultValue 0)
  cleanExternalPackageConsumerOwnerRunbookFailedBlockerCount = [int](Get-PropertyOrDefault -Object $cleanExternalPackageConsumerOwnerRunbookValidation -Name "failedBlockerCount" -DefaultValue -1)
  postPublishOwnerVerificationRunbookState = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "runbookState" -DefaultValue "missing")
  postPublishOwnerVerificationRunbookValidationState = [string](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbookValidation -Name "validationState" -DefaultValue "missing")
  postPublishOwnerVerificationRunbookStepCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "stepCount" -DefaultValue 0)
  postPublishOwnerVerificationRunbookBlockedStepCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbook -Name "blockedStepCount" -DefaultValue 0)
  postPublishOwnerVerificationRunbookFailedBlockerCount = [int](Get-PropertyOrDefault -Object $postPublishOwnerVerificationRunbookValidation -Name "failedBlockerCount" -DefaultValue -1)
  readinessLaneCount = $readinessLanes.Count
  blockedReadinessLaneCount = @($readinessLanes | Where-Object { $_.blocked }).Count
  readinessLanes = @($readinessLanes)
  ownerCommandSequence = @($ownerCommandSequence)
  nonProofEvidenceCatalog = @($nonProofEvidenceCatalog)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  sourceArtifacts = @(
    "artifacts/final-release/owner-input-preflight-bundle.json",
    "artifacts/final-release/final-owner-proof-action-worklist.json",
    "artifacts/final-release/final-owner-proof-action-worklist-validation.json",
    "artifacts/final-release/final-owner-execution-package.json",
    "artifacts/final-release/final-owner-execution-package-validation.json",
    "artifacts/final-release/owner-external-proof-execution-result-import.json",
    "artifacts/final-release/owner-external-proof-execution-result-import-validation.json",
    "artifacts/final-release/real-external-proof-record-import-validator.json",
    "artifacts/final-release/real-external-proof-record-import-validator-validation.json",
    "artifacts/final-release/real-proof-record-candidate-from-owner-result-import.json",
    "artifacts/final-release/real-proof-record-candidate-from-owner-result-import-validation.json",
    "artifacts/final-release/release-close-real-proof-import-bridge.json",
    "artifacts/final-release/release-close-real-proof-import-bridge-validation.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook-validation.json",
    "artifacts/final-release/post-publish-owner-verification-runbook.json",
    "artifacts/final-release/post-publish-owner-verification-runbook-validation.json",
    "artifacts/final-release/release-proof-dashboard-validation.json",
    "artifacts/final-release/release-proof-dashboard.json",
    "artifacts/final-release/final-publish-proof-gate-report.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/user-acceptance/yolovision-owner-proof-field-delta-repair-pack.json",
    "artifacts/final-release/public-docs-package-metadata-gate.json"
  )
  boundary = "Release publish readiness evidence pack is a non-publishing, non-proof aggregator. It summarizes owner/public/post-publish blockers and cannot publish, close release issues, or promote runtime proof."
}

$jsonPath = Join-Path $OutputRoot "release-publish-readiness-evidence-pack.json"
$markdownPath = Join-Path $OutputRoot "release-publish-readiness-evidence-pack.md"
$pack | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = foreach ($lane in $readinessLanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | ``$(ConvertTo-MarkdownCell $lane.state)`` | ``$($lane.blocked)`` | $(ConvertTo-MarkdownCell $lane.boundary) |"
}

$catalogRows = foreach ($item in $nonProofEvidenceCatalog) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$(ConvertTo-MarkdownCell $item.state)`` | $(ConvertTo-MarkdownCell $item.boundary) |"
}

$commandLines = $ownerCommandSequence | ForEach-Object { "- ``$($_.lane)``: ``$($_.command)``" }

$markdown = @"
# Release Publish Readiness Evidence Pack

Generated at: ``$($pack.generatedAtUtc)``

## Summary

- recordKind: ``$($pack.recordKind)``
- packState: ``$($pack.packState)``
- ownerInputPreflightState: ``$($pack.ownerInputPreflightState)``
- dashboardValidationState: ``$($pack.dashboardValidationState)``
- finalGateState: ``$($pack.finalGateState)``
- finalOwnerProofActionWorklistState: ``$($pack.finalOwnerProofActionWorklistState)``
- finalOwnerProofActionWorklistValidationState: ``$($pack.finalOwnerProofActionWorklistValidationState)``
- finalOwnerProofActionCount: ``$($pack.finalOwnerProofActionCount)``
- finalOwnerProofBlockedActionCount: ``$($pack.finalOwnerProofBlockedActionCount)``
- finalOwnerExecutionPackageState: ``$($pack.finalOwnerExecutionPackageState)``
- finalOwnerExecutionPackageValidationState: ``$($pack.finalOwnerExecutionPackageValidationState)``
- finalOwnerExecutionStepCount: ``$($pack.finalOwnerExecutionStepCount)``
- finalOwnerBlockedExecutionStepCount: ``$($pack.finalOwnerBlockedExecutionStepCount)``
- ownerExternalProofExecutionResultImportState: ``$($pack.ownerExternalProofExecutionResultImportState)``
- ownerExternalProofExecutionResultImportValidationState: ``$($pack.ownerExternalProofExecutionResultImportValidationState)``
- ownerExternalProofExecutionResultImportReadyForStrictValidatorLaneCount: ``$($pack.ownerExternalProofExecutionResultImportReadyForStrictValidatorLaneCount)``
- ownerExternalProofExecutionResultImportBlockedLaneCount: ``$($pack.ownerExternalProofExecutionResultImportBlockedLaneCount)``
- realExternalProofRecordImportValidatorState: ``$($pack.realExternalProofRecordImportValidatorState)``
- realExternalProofRecordImportValidatorValidationState: ``$($pack.realExternalProofRecordImportValidatorValidationState)``
- realProofRecordCandidateFromOwnerResultImportState: ``$($pack.realProofRecordCandidateFromOwnerResultImportState)``
- releaseCloseRealProofImportBridgeValidationState: ``$($pack.releaseCloseRealProofImportBridgeValidationState)``
- cleanExternalPackageConsumerOwnerRunbookState: ``$($pack.cleanExternalPackageConsumerOwnerRunbookState)``
- cleanExternalPackageConsumerOwnerRunbookValidationState: ``$($pack.cleanExternalPackageConsumerOwnerRunbookValidationState)``
- cleanExternalPackageConsumerOwnerRunbookStepCount: ``$($pack.cleanExternalPackageConsumerOwnerRunbookStepCount)``
- cleanExternalPackageConsumerOwnerRunbookFailedBlockerCount: ``$($pack.cleanExternalPackageConsumerOwnerRunbookFailedBlockerCount)``
- postPublishOwnerVerificationRunbookState: ``$($pack.postPublishOwnerVerificationRunbookState)``
- postPublishOwnerVerificationRunbookValidationState: ``$($pack.postPublishOwnerVerificationRunbookValidationState)``
- postPublishOwnerVerificationRunbookStepCount: ``$($pack.postPublishOwnerVerificationRunbookStepCount)``
- postPublishOwnerVerificationRunbookFailedBlockerCount: ``$($pack.postPublishOwnerVerificationRunbookFailedBlockerCount)``
- publicDocsPackageMetadataGateState: ``$($pack.publicDocsPackageMetadataGateState)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``
- actionRequiredCount: ``$($pack.actionRequiredCount)``
- failedBlockerCount: ``$($pack.failedBlockerCount)``

## Readiness Lanes

| Lane | State | Blocked | Boundary |
| --- | --- | --- | --- |
$($laneRows -join "`r`n")

## Non-Proof Evidence Catalog

| Artifact | State | Boundary |
| --- | --- | --- |
$($catalogRows -join "`r`n")

## Owner Command Sequence

$($commandLines -join "`r`n")

## Boundary

$($pack.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release publish readiness evidence pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackState=$($pack.packState) ReadinessLanes=$($pack.readinessLaneCount) FailedBlockers=$($pack.failedBlockerCount) ActionRequired=$($pack.actionRequiredCount)"
