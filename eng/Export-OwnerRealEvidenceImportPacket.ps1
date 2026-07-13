[CmdletBinding()]
param(
  [string]$FinalActionMapPath = "artifacts/final-release/final-publish-action-required-evidence-map.json",
  [string]$ReleaseCloseStrictOrderPath = "artifacts/final-release/release-close-strict-proof-execution-order.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

function Get-StepByActionId {
  param([AllowNull()][object]$Order, [string]$Id)
  if ($null -eq $Order) { return $null }
  return @($Order.steps | Where-Object { [string]$_.actionId -eq $Id } | Select-Object -First 1)
}

function New-Lane {
  param(
    [object]$Action,
    [AllowNull()][object]$Step,
    [string[]]$RequiredFields,
    [string[]]$RequiredFiles,
    [string[]]$RequiredHashes,
    [string[]]$RequiredValidators,
    [string[]]$RequiredOwnerReview,
    [string[]]$LinkedArtifacts
  )

  $firstCommand = if ($null -ne $Step -and -not [string]::IsNullOrWhiteSpace([string]$Step.firstOwnerCommand)) {
    [string]$Step.firstOwnerCommand
  }
  else {
    [string]$Action.firstCommand
  }

  [pscustomobject]@{
    laneId = [string]$Action.id
    proofLane = [string]$Action.proofLane
    laneState = "blocked-owner-real-evidence-required"
    ownerInputArtifact = [string]$Action.ownerInputArtifact
    expectedRecord = [string]$Action.requiredRecord
    validator = [string]$Action.validator
    validators = @($RequiredValidators | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)
    firstOwnerCommand = $firstCommand
    requiredFields = @($RequiredFields)
    requiredFiles = @($RequiredFiles)
    requiredHashes = @($RequiredHashes)
    requiredOwnerReview = @($RequiredOwnerReview)
    linkedArtifacts = @($LinkedArtifacts | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique)
    importPacketFieldPath = "lanes.$([string]$Action.id)"
    missingRealEvidenceReason = "Owner has not supplied real files, logs, hashes, exit codes, host metadata, package metadata, and reviewer decision for this lane."
    forbiddenSubstitutes = @(
      "template-only record",
      "dashboard-only record",
      "build-only report",
      "local feed",
      "ProjectReference",
      "direct nupkg",
      "dry-run record",
      "screenshot-only evidence"
    )
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromotePackageConsumerRuntime = $false
    canPromoteRuntimeProof = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

$finalActionMap = Read-JsonOrThrow -Path $FinalActionMapPath
$strictOrder = Read-JsonOrThrow -Path $ReleaseCloseStrictOrderPath

$ids = @(
  "final-owner-real-input-template-pack-owner-input-required",
  "real-model-runtime-owner-proof-required",
  "package-consumer-runtime-owner-proof-required",
  "post-publish-verification-owner-proof-required",
  "owner-external-proof-result-import-owner-proof-required",
  "owner-result-candidate-bridge-real-proof-required"
)

$lanes = @()
foreach ($id in $ids) {
  $action = Get-ActionById -Map $finalActionMap -Id $id
  $step = Get-StepByActionId -Order $strictOrder -Id $id

  switch ($id) {
    "final-owner-real-input-template-pack-owner-input-required" {
      $lanes += New-Lane -Action $action -Step $step `
        -RequiredFields @("ownerName", "reviewedAtUtc", "laneOwner", "evidenceBundleSha256", "allLaneInputsPresent") `
        -RequiredFiles @("final owner real input template pack", "template pack validation report", "owner review notes") `
        -RequiredHashes @("templatePackSha256", "validationReportSha256", "ownerReviewSha256") `
        -RequiredValidators @("eng/Test-FinalPublishProofGate.ps1 -Strict") `
        -RequiredOwnerReview @("owner confirms every lane has real input", "owner confirms no template-only promotion", "owner confirms action-required rows remain visible") `
        -LinkedArtifacts @($action.ownerInputArtifact, $action.requiredRecord, $strictOrder.sourceFinalActionMap)
    }
    "real-model-runtime-owner-proof-required" {
      $lanes += New-Lane -Action $action -Step $step `
        -RequiredFields @("modelFamily", "task", "modelSourceUrl", "license", "labelsPath", "inputShape", "preprocessedInputPath", "goldenOutputPath", "runtimeLogPath", "exitCode", "hostMetadata") `
        -RequiredFiles @("real ONNX model", "labels file", "preprocessed tensor input", "runtime stdout/stderr", "output JSON", "license/source note") `
        -RequiredHashes @("modelSha256", "labelsSha256", "preprocessedInputSha256", "outputJsonSha256", "runtimeLogSha256") `
        -RequiredValidators @("eng/Test-RealCaseEvidenceRecord.ps1 -FailOnNotProof", "eng/Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict") `
        -RequiredOwnerReview @("owner accepts model source/license", "owner verifies det/seg/pose/obb/cls/sem output coverage", "owner signs off runtime logs") `
        -LinkedArtifacts @($action.ownerInputArtifact, $action.requiredRecord, "artifacts/user-acceptance/yolovision-owner-real-evidence-intake-dashboard.json")
    }
    "package-consumer-runtime-owner-proof-required" {
      $lanes += New-Lane -Action $action -Step $step `
        -RequiredFields @("selectedPackageRoute", "publicPackageSource", "publishedVersion", "cleanExternalConsumerRoot", "restoreCommand", "buildCommand", "smokeCommand", "exitCode", "runtimePackageKey", "hostMetadata") `
        -RequiredFiles @("clean external consumer project", "restore log", "build log", "smoke log", "dependency probe log", "package metadata record") `
        -RequiredHashes @("managedPackageSha256", "nativeBridgeSha256", "runtimePackageSha256", "restoreLogSha256", "buildLogSha256", "smokeLogSha256") `
        -RequiredValidators @("eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof") `
        -RequiredOwnerReview @("owner confirms clean root is outside repository", "owner confirms no local feed substitute", "owner confirms no ProjectReference or direct nupkg substitute") `
        -LinkedArtifacts @($action.ownerInputArtifact, $action.requiredRecord, $action.planningArtifact, $action.executionKit)
    }
    "post-publish-verification-owner-proof-required" {
      $lanes += New-Lane -Action $action -Step $step `
        -RequiredFields @("publicChannelUrl", "publishedVersion", "packageSource", "installCommand", "runCommand", "smokeLogPath", "stdoutSummary", "stderrSummary", "ownerReviewer", "ownerDecision", "rollbackOrDeprecationPlanReference") `
        -RequiredFiles @("public install transcript", "public run transcript", "smoke log", "owner review record", "rollback/deprecation plan") `
        -RequiredHashes @("installTranscriptSha256", "runTranscriptSha256", "smokeLogSha256", "ownerReviewSha256", "rollbackPlanSha256") `
        -RequiredValidators @("eng/Test-PostPublishVerificationRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof") `
        -RequiredOwnerReview @("owner confirms public channel URL", "owner confirms published version", "owner confirms rollback/deprecation decision") `
        -LinkedArtifacts @($action.ownerInputArtifact, $action.requiredRecord, $action.intakeMapArtifact)
    }
    "owner-external-proof-result-import-owner-proof-required" {
      $lanes += New-Lane -Action $action -Step $step `
        -RequiredFields @("ownerResultPath", "ownerResultSha256", "sourceLogPaths", "sourceLogSha256Values", "hostMetadata", "packageMetadata", "importedAtUtc") `
        -RequiredFiles @("owner external execution result", "source logs", "hash manifest", "host metadata", "package metadata") `
        -RequiredHashes @("ownerResultSha256", "sourceLogsSha256", "hashManifestSha256", "hostMetadataSha256") `
        -RequiredValidators @("eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict") `
        -RequiredOwnerReview @("owner confirms imported result uses real logs", "owner confirms SHA256 values match existing files", "owner confirms no dashboard-only import") `
        -LinkedArtifacts @($action.ownerInputArtifact, $action.requiredRecord)
    }
    "owner-result-candidate-bridge-real-proof-required" {
      $lanes += New-Lane -Action $action -Step $step `
        -RequiredFields @("strictValidatorReadyOwnerRecord", "candidateRecordPath", "candidateRecordSha256", "sourceOwnerResultSha256", "bridgeDecision", "bridgedAtUtc") `
        -RequiredFiles @("strict-validator-ready owner record", "real proof candidate record", "bridge validation report") `
        -RequiredHashes @("ownerRecordSha256", "candidateRecordSha256", "bridgeValidationSha256") `
        -RequiredValidators @("eng/Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict") `
        -RequiredOwnerReview @("owner confirms candidate bridge only uses strict-validator-ready records", "owner confirms validators were not weakened", "owner confirms release close remains blocked until final gates pass") `
        -LinkedArtifacts @($action.ownerInputArtifact, $action.requiredRecord)
    }
  }
}

$packet = [pscustomobject]@{
  recordKind = "owner-real-evidence-import-packet"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  packetState = "blocked-owner-real-evidence-required"
  sourceFinalActionMap = $FinalActionMapPath
  sourceReleaseCloseStrictOrder = $ReleaseCloseStrictOrderPath
  actionRequiredCount = [int]$finalActionMap.actionRequiredCount
  laneCount = @($lanes).Count
  requiredLaneIds = @($ids)
  lanes = @($lanes)
  performsPublish = $false
  performsRuntimeExecution = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This packet defines how Owner evidence must be imported. It does not execute runs, publish packages, close issues, or promote proof."
}

$jsonPath = Join-Path $OutputRoot "owner-real-evidence-import-packet.json"
$markdownPath = Join-Path $OutputRoot "owner-real-evidence-import-packet.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($packet | ConvertTo-Json -Depth 18)
$rows = foreach ($lane in $lanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.laneId)`` | ``$(ConvertTo-MarkdownCell $lane.proofLane)`` | ``$($lane.requiredFields.Count)`` | ``$($lane.requiredFiles.Count)`` | ``$($lane.requiredHashes.Count)`` | ``False`` |"
}

$markdown = @"
# Owner Real Evidence Import Packet

Generated at: ``$($packet.generatedAtUtc)``

## Summary

- packetState: ``$($packet.packetState)``
- actionRequiredCount: ``$($packet.actionRequiredCount)``
- laneCount: ``$($packet.laneCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromotePackageConsumerRuntime: ``False``

## Lanes

| Lane | Proof Lane | Fields | Files | Hashes | Can Publish |
| --- | --- | --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($packet.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner real evidence import packet written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PacketState=$($packet.packetState) LaneCount=$($packet.laneCount)"
