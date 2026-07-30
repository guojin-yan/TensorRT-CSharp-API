[CmdletBinding()]
param(
  [string]$FinalActionMapPath = "artifacts/final-release/final-publish-action-required-evidence-map.json",
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

function Read-JsonOrThrow {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    throw "Missing required JSON artifact: $Path"
  }

  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-ActionById {
  param([object]$Map, [string]$Id)
  $action = @($Map.actions | Where-Object { [string]$_.id -eq $Id } | Select-Object -First 1)
  if ($null -eq $action -or @($action).Count -eq 0) {
    throw "Missing final action map item: $Id"
  }

  return $action
}

function New-ExecutionStep {
  param(
    [int]$Order,
    [string]$ActionId,
    [string]$Phase,
    [string]$OwnerObjective,
    [string]$ExpectedInputArtifact,
    [string]$ExpectedOutputArtifact,
    [string]$ValidatorCommand,
    [string]$FirstOwnerCommand,
    [string[]]$FailureRepairHints,
    [string[]]$LinkedArtifacts,
    [string[]]$ForbiddenSubstitutes
  )

  [pscustomobject]@{
    order = $Order
    actionId = $ActionId
    phase = $Phase
    stepState = "blocked-owner-action-required"
    ownerObjective = $OwnerObjective
    expectedInputArtifact = $ExpectedInputArtifact
    expectedOutputArtifact = $ExpectedOutputArtifact
    validatorCommand = $ValidatorCommand
    firstOwnerCommand = $FirstOwnerCommand
    failureRepairHints = @($FailureRepairHints)
    linkedArtifacts = @($LinkedArtifacts | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromotePackageConsumerRuntime = $false
    canPromoteRuntimeProof = $false
  }
}

$finalActionMap = Read-JsonOrThrow -Path $FinalActionMapPath
$commonForbidden = @(
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "template-only record",
  "dashboard-only record",
  "build-only report",
  "dry-run record",
  "TensorRtExec report",
  "OnnxToEngine report",
  "YoloVision matrix",
  "screenshot-only evidence"
)

$realModelAction = Get-ActionById -Map $finalActionMap -Id "real-model-runtime-owner-proof-required"
$packageAction = Get-ActionById -Map $finalActionMap -Id "package-consumer-runtime-owner-proof-required"
$postPublishAction = Get-ActionById -Map $finalActionMap -Id "post-publish-verification-owner-proof-required"
$templatePackAction = Get-ActionById -Map $finalActionMap -Id "final-owner-real-input-template-pack-owner-input-required"
$externalImportAction = Get-ActionById -Map $finalActionMap -Id "owner-external-proof-result-import-owner-proof-required"
$candidateBridgeAction = Get-ActionById -Map $finalActionMap -Id "owner-result-candidate-bridge-real-proof-required"

$steps = @(
  New-ExecutionStep -Order 1 -ActionId $templatePackAction.id -Phase "owner-real-input-template-pack" -OwnerObjective "Prepare the owner input template pack before any proof promotion attempt." -ExpectedInputArtifact $templatePackAction.ownerInputArtifact -ExpectedOutputArtifact $templatePackAction.requiredRecord -ValidatorCommand $templatePackAction.validator -FirstOwnerCommand $templatePackAction.firstCommand -FailureRepairHints @("Fill every owner input lane with real logs, hashes, exit codes, host metadata, package metadata, and owner review.", "Keep template rows visible until every strict validator accepts real input.", "Do not convert the pack into proof by deleting action-required rows.") -LinkedArtifacts @($templatePackAction.ownerInputArtifact, $templatePackAction.requiredRecord) -ForbiddenSubstitutes $commonForbidden
  New-ExecutionStep -Order 2 -ActionId $realModelAction.id -Phase "real-model-runtime" -OwnerObjective "Import real YoloVision model assets, labels, licenses, golden outputs, logs, and SHA256 values for all six tasks." -ExpectedInputArtifact $realModelAction.ownerInputArtifact -ExpectedOutputArtifact $realModelAction.requiredRecord -ValidatorCommand $realModelAction.validator -FirstOwnerCommand $realModelAction.firstCommand -FailureRepairHints @("Use real model assets and owner-reviewed labels, not sample matrix placeholders.", "Record model source, license, preprocessing input, output JSON/log, and SHA256.", "Keep det/seg/pose/obb/cls/sem incomplete lanes action-required until validated.") -LinkedArtifacts @($realModelAction.ownerInputArtifact, $realModelAction.requiredRecord, "artifacts/user-acceptance/yolovision-owner-real-evidence-intake-dashboard.json") -ForbiddenSubstitutes $commonForbidden
  New-ExecutionStep -Order 3 -ActionId $packageAction.id -Phase "package-consumer-runtime" -OwnerObjective "Choose the public managed plus bridge-only route and run a clean external package consumer from the public source." -ExpectedInputArtifact $packageAction.ownerInputArtifact -ExpectedOutputArtifact $packageAction.requiredRecord -ValidatorCommand $packageAction.validator -FirstOwnerCommand $packageAction.firstCommand -FailureRepairHints @("Choose either github-release-managed-plus-bridge-assets or nuget-managed-plus-bridge-packages.", "Run restore/build/smoke outside the repository and capture log/hash/exit code.", "Require same-commit managed/bridge provenance and reject project-reference, locally built package-feed, direct nupkg, and vendor-bundle substitutes.") -LinkedArtifacts @($packageAction.ownerInputArtifact, $packageAction.requiredRecord, $packageAction.planningArtifact, $packageAction.executionKit) -ForbiddenSubstitutes $commonForbidden
  New-ExecutionStep -Order 4 -ActionId $postPublishAction.id -Phase "post-publish-verification" -OwnerObjective "After real public publication, install and run from the public channel and capture owner-reviewed logs and rollback data." -ExpectedInputArtifact $postPublishAction.ownerInputArtifact -ExpectedOutputArtifact $postPublishAction.requiredRecord -ValidatorCommand $postPublishAction.validator -FirstOwnerCommand $postPublishAction.firstCommand -FailureRepairHints @("Provide public channel URL, published version, install/run commands, stdout/stderr, smoke log SHA256, host metadata, owner decision, and rollback/deprecation reference.", "Do not treat pre-publish package planning as post-publish proof.", "Keep canCloseReleaseIssue=false until strict post-publish validation passes.") -LinkedArtifacts @($postPublishAction.ownerInputArtifact, $postPublishAction.requiredRecord, $postPublishAction.intakeMapArtifact) -ForbiddenSubstitutes $commonForbidden
  New-ExecutionStep -Order 5 -ActionId $externalImportAction.id -Phase "owner-external-proof-result-import" -OwnerObjective "Import owner external execution results only after real logs, matching hashes, and strict owner result fields are available." -ExpectedInputArtifact $externalImportAction.ownerInputArtifact -ExpectedOutputArtifact $externalImportAction.requiredRecord -ValidatorCommand $externalImportAction.validator -FirstOwnerCommand $externalImportAction.firstCommand -FailureRepairHints @("Import only real external owner execution results with existing logs and matching SHA256.", "Repair missing host/package metadata before bridging candidates.", "Do not import dashboard-only or template-only records.") -LinkedArtifacts @($externalImportAction.ownerInputArtifact, $externalImportAction.requiredRecord) -ForbiddenSubstitutes $commonForbidden
  New-ExecutionStep -Order 6 -ActionId $candidateBridgeAction.id -Phase "owner-result-candidate-bridge" -OwnerObjective "Bridge strict-validator-ready owner results into real proof candidates without weakening proof validators." -ExpectedInputArtifact $candidateBridgeAction.ownerInputArtifact -ExpectedOutputArtifact $candidateBridgeAction.requiredRecord -ValidatorCommand $candidateBridgeAction.validator -FirstOwnerCommand $candidateBridgeAction.firstCommand -FailureRepairHints @("Bridge only strict-validator-ready owner records.", "Keep release issue close blocked until real proof candidates pass final gates.", "Do not lower validator strength or delete action-required rows to pass.") -LinkedArtifacts @($candidateBridgeAction.ownerInputArtifact, $candidateBridgeAction.requiredRecord) -ForbiddenSubstitutes $commonForbidden
)

$mapText = ($finalActionMap | ConvertTo-Json -Depth 18)
$plan = [pscustomobject]@{
  recordKind = "release-close-strict-proof-execution-order"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  orderState = "blocked-owner-real-proof-execution-required"
  sourceFinalActionMap = $FinalActionMapPath
  sourcePackageConsumerDualRouteProofPlan = [string]$finalActionMap.sourcePackageConsumerDualRouteProofPlan
  sourceCleanExternalConsumerExecutionKit = [string]$finalActionMap.sourceCleanExternalConsumerExecutionKit
  sourcePostPublishVerificationIntakeMap = [string]$finalActionMap.sourcePostPublishVerificationIntakeMap
  actionRequiredCount = [int]$finalActionMap.actionRequiredCount
  executionStepCount = @($steps).Count
  actionIds = @($steps | ForEach-Object { [string]$_.actionId })
  blockedStepCount = @($steps | Where-Object { [string]$_.stepState -like "blocked*" }).Count
  packageConsumerRouteCount = [int]$finalActionMap.packageConsumerRouteCount
  cleanExternalConsumerStepCount = [int]$finalActionMap.cleanExternalConsumerStepCount
  postPublishRequiredFieldCount = [int]$finalActionMap.postPublishRequiredFieldCount
  hasPackageConsumerPlanningLinks = $mapText.Contains("package-consumer-dual-route-proof-plan.json") -and $mapText.Contains("clean-external-consumer-execution-kit.json")
  hasPostPublishIntakeLinks = $mapText.Contains("post-publish-verification-intake-map.json")
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  steps = @($steps)
  boundary = "This execution order is a release-close owner action plan only. It does not publish, does not close issues, does not run package consumers, and does not promote proof."
}

$jsonPath = Join-Path $OutputRoot "release-close-strict-proof-execution-order.json"
$markdownPath = Join-Path $OutputRoot "release-close-strict-proof-execution-order.md"
$plan | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($step in $steps) {
  "| ``$($step.order)`` | ``$(ConvertTo-MarkdownCell $step.actionId)`` | ``$(ConvertTo-MarkdownCell $step.phase)`` | ``$(ConvertTo-MarkdownCell $step.validatorCommand)`` | ``False`` |"
}

$markdown = @"
# Release Close Strict Proof Execution Order

Generated at: ``$($plan.generatedAtUtc)``

## Summary

- orderState: ``$($plan.orderState)``
- actionRequiredCount: ``$($plan.actionRequiredCount)``
- executionStepCount: ``$($plan.executionStepCount)``
- blockedStepCount: ``$($plan.blockedStepCount)``
- packageConsumerRouteCount: ``$($plan.packageConsumerRouteCount)``
- cleanExternalConsumerStepCount: ``$($plan.cleanExternalConsumerStepCount)``
- postPublishRequiredFieldCount: ``$($plan.postPublishRequiredFieldCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromotePackageConsumerRuntime: ``False``

## Execution Order

| Order | Action | Phase | Validator | Can Close |
| --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($plan.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release close strict proof execution order written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "OrderState=$($plan.orderState) ExecutionStepCount=$($plan.executionStepCount) BlockedStepCount=$($plan.blockedStepCount)"
