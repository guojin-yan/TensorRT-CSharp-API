[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ProofLane {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$State,
    [string]$EvidenceKind,
    [bool]$OwnerActionRequired,
    [string[]]$RequiredInputs,
    [string[]]$StrictValidators,
    [string]$PromotionRule,
    [string]$Boundary
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    state = $State
    evidenceKind = $EvidenceKind
    ownerActionRequired = $OwnerActionRequired
    requiredInputs = @($RequiredInputs)
    strictValidators = @($StrictValidators)
    canPromoteRuntimeProof = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    promotionRule = $PromotionRule
    boundary = $Boundary
  }
}

function New-PromotionRequirement {
  param([string]$Id, [string]$Description, [string]$Validator, [string]$Rejects)

  [pscustomobject]@{
    id = $Id
    description = $Description
    validator = $Validator
    rejects = $Rejects
    ownerActionRequired = $true
  }
}

function New-ExecutionCommand {
  param([int]$Order, [string]$Id, [string]$Command, [string]$ExpectedOutput, [string]$Boundary)

  [pscustomobject]@{
    order = $Order
    id = $Id
    command = $Command
    expectedOutput = $ExpectedOutput
    ownerActionRequired = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    boundary = $Boundary
  }
}

$localSmokeClassification = Read-JsonOrNull "artifacts/final-release/cuda-device-initialization-local-smoke-classification.json"
$localSmokeClassificationValidation = Read-JsonOrNull "artifacts/final-release/cuda-device-initialization-local-smoke-classification-validation.json"
$releaseEvidence = Read-JsonOrNull "artifacts/final-release/release-evidence-bundle.json"
$preflightMatrix = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json"
$externalRuntimeProofTemplate = Read-JsonOrNull "artifacts/final-release/external-runtime-proof-record.input-template.json"
$postPublishContract = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json"
$finalPostPublishContract = Read-JsonOrNull "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json"
$ownerPack = Read-JsonOrNull "artifacts/final-release/clean-consumer-proof-owner-execution-pack.json"
$ownerPackValidation = Read-JsonOrNull "artifacts/final-release/clean-consumer-proof-owner-execution-pack-validation.json"
$checklist = Read-JsonOrNull "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json"
$checklistValidation = Read-JsonOrNull "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist-validation.json"

$sourceArtifacts = @(
  "artifacts/final-release/cuda-device-initialization-local-smoke-classification.json",
  "artifacts/final-release/cuda-device-initialization-local-smoke-classification-validation.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json",
  "artifacts/final-release/external-runtime-proof-record.input-template.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json",
  "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist-validation.json",
  "artifacts/final-release/clean-consumer-proof-owner-execution-pack.json",
  "artifacts/final-release/clean-consumer-proof-owner-execution-pack-validation.json"
)

$sourceStates = [pscustomobject]@{
  cudaDeviceInitializationClassificationPresent = $null -ne $localSmokeClassification
  cudaDeviceInitializationClassificationState = [string](Get-PropertyOrDefault -Object $localSmokeClassification -Name "classificationState" -DefaultValue "missing-cuda-device-initialization-local-smoke-classification")
  cudaDeviceInitializationProofKind = [string](Get-PropertyOrDefault -Object $localSmokeClassification -Name "proofKind" -DefaultValue "missing-proof-kind")
  cudaDeviceInitializationValidationState = [string](Get-PropertyOrDefault -Object $localSmokeClassificationValidation -Name "validationState" -DefaultValue "missing-cuda-device-initialization-local-smoke-classification-validation")
  releaseEvidencePresent = $null -ne $releaseEvidence
  releaseEvidenceStatus = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "status" -DefaultValue "missing-release-evidence-bundle")
  releaseOwnerActionStatus = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerActionStatus" -DefaultValue "owner-action-required")
  preflightMatrixPresent = $null -ne $preflightMatrix
  externalRuntimeProofTemplatePresent = $null -ne $externalRuntimeProofTemplate
  postPublishCleanConsumerProofRecordContractPresent = $null -ne $postPublishContract
  finalPostPublishCleanConsumerProofRecordContractPresent = $null -ne $finalPostPublishContract
  cleanConsumerRuntimeProofExecutionChecklistReady = [string](Get-PropertyOrDefault -Object $checklistValidation -Name "validationState" -DefaultValue "") -eq "clean-consumer-runtime-proof-execution-checklist-ready"
  cleanConsumerProofOwnerExecutionPackReady = [string](Get-PropertyOrDefault -Object $ownerPackValidation -Name "validationState" -DefaultValue "") -eq "clean-consumer-proof-owner-execution-pack-ready"
}

$lanes = @(
  New-ProofLane -Order 1 -Id "local-smoke" -Title "Local smoke classification lane" -State "local-smoke-not-external-proof" -EvidenceKind "non-proof" -OwnerActionRequired $false -RequiredInputs @(
    "CudaDeviceInitializationProofRunner output",
    "pre-init call order marker",
    "Skipped=True forbidden substitute marker"
  ) -StrictValidators @(
    "eng/Test-CudaDeviceInitializationLocalSmokeClassification.ps1 -Strict"
  ) -PromotionRule "Never promotes package-consumer-runtime proof." -Boundary "Local smoke is useful engineering signal only; it is not external clean consumer proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  New-ProofLane -Order 2 -Id "local-feed" -Title "Local feed consumer lane" -State "local-feed-non-proof" -EvidenceKind "non-proof" -OwnerActionRequired $false -RequiredInputs @(
    "local package build",
    "local feed restore/build output",
    "native asset copy listing"
  ) -StrictValidators @(
    "eng/Test-PackageConsumer.ps1"
  ) -PromotionRule "Never promotes public package proof or post-publish proof." -Boundary "Local feed output is not clean public package source proof; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  New-ProofLane -Order 3 -Id "clean-external-package-consumer" -Title "Repository-external clean package consumer runtime lane" -State "owner-action-required" -EvidenceKind "candidate-runtime-proof-after-strict-validation" -OwnerActionRequired $true -RequiredInputs @(
    "consumer project outside repository",
    "no ProjectReference",
    "managed/runtime packages restored from package source",
    "native asset listing and SHA256",
    "runtime smoke stdout/stderr logs and SHA256",
    "host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
    "owner review metadata"
  ) -StrictValidators @(
    "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
    "eng/Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1"
  ) -PromotionRule "May promote only after strict validator accepts existing logs and proofClassification=package-consumer-runtime." -Boundary "Until owner supplies real clean external runtime smoke and strict validation passes, this lane remains owner-action-required and non-proof; it is not runtime proof, not publish approval, not release close approval, and not package push."
  New-ProofLane -Order 4 -Id "compatible-cuda-host-runtime" -Title "Compatible CUDA/TensorRT host runtime lane" -State "owner-action-required" -EvidenceKind "owner-action-required" -OwnerActionRequired $true -RequiredInputs @(
    "compatible NVIDIA driver",
    "compatible CUDA runtime/toolkit",
    "TensorRT runtime matching runtime package key",
    "cuDNN version when applicable",
    "runtime smoke command executed on compatible host"
  ) -StrictValidators @(
    "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
    "eng/Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof"
  ) -PromotionRule "Compatible host is required input; host metadata alone cannot promote proof." -Boundary "Host compatibility metadata is prerequisite evidence only; it is not runtime proof without clean consumer smoke logs and strict validation."
  New-ProofLane -Order 5 -Id "post-publish-clean-consumer" -Title "Post-publish clean consumer lane" -State "owner-action-required-after-publication" -EvidenceKind "post-publish-owner-action-required" -OwnerActionRequired $true -RequiredInputs @(
    "published package source",
    "repository-external post-publish clean consumer",
    "post-publish install log",
    "post-publish run log",
    "downloaded nupkg SHA256",
    "host metadata",
    "owner non-substitute confirmation"
  ) -StrictValidators @(
    "eng/Test-PostPublishCleanConsumerProofRecordContract.ps1 -Strict",
    "eng/Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict",
    "eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
  ) -PromotionRule "May close only after real post-publish proof and strict close validator pass." -Boundary "Post-publish proof is separate owner action after public publish; this bundle cannot publish packages, create proof, or close release."
)

$forbiddenSubstitutes = @(
  "Skipped=True",
  "local-smoke",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "build-only",
  "dependency-probe",
  "blocked-by-cuda-driver",
  "article roadmap",
  "dashboard",
  "runbook",
  "candidate",
  "draft",
  "template",
  "preflight-only",
  "dry-run-only",
  "schema-only",
  "owner input without strict validator pass",
  "native asset listing without runtime smoke",
  "host metadata without runtime smoke",
  "package hash without existing log validation"
)

$promotionRequirements = @(
  New-PromotionRequirement -Id "repository-external-clean-consumer" -Description "Clean consumer project must live outside the source repository and owner proof root must be explicit." -Validator "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Rejects "ProjectReference, in-repo sample, local smoke."
  New-PromotionRequirement -Id "no-project-reference" -Description "Consumer must reference packages only, never source projects." -Validator "eng/Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1" -Rejects "ProjectReference."
  New-PromotionRequirement -Id "public-or-approved-package-source" -Description "Managed and runtime packages must be restored from the declared package source." -Validator "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Rejects "local feed and direct nupkg unless explicitly classified as non-proof."
  New-PromotionRequirement -Id "native-assets-and-sha256" -Description "Native asset listing and SHA256 values must be captured for the exact runtime package." -Validator "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Rejects "asset listing without smoke log."
  New-PromotionRequirement -Id "runtime-smoke-log-sha256" -Description "Runtime smoke stdout/stderr logs must exist and hash-match owner input." -Validator "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Rejects "missing log, mismatched SHA256, copied summary."
  New-PromotionRequirement -Id "host-metadata" -Description "OS, GPU, driver, CUDA, TensorRT, and cuDNN metadata must be captured." -Validator "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Rejects "dependency probe only and blocked-by-driver output."
  New-PromotionRequirement -Id "strict-external-runtime-proof-validator" -Description "External runtime proof validator must pass with existing logs and FailOnNotProof." -Validator "eng/Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" -Rejects "template, draft, preflight, runbook."
  New-PromotionRequirement -Id "post-publish-separate-gate" -Description "Post-publish clean consumer proof remains a separate owner gate after package publication." -Validator "eng/Test-PostPublishCleanConsumerProofRecordContract.ps1 -Strict" -Rejects "pre-publish runtime smoke being reused as post-publish proof."
)

$executionCommands = @(
  New-ExecutionCommand -Order 1 -Id "export-local-smoke-classification" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CudaDeviceInitializationLocalSmokeClassification.ps1" -ExpectedOutput "local-smoke-not-external-proof classification artifact" -Boundary "Generates non-proof classification only."
  New-ExecutionCommand -Order 2 -Id "validate-local-smoke-classification" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CudaDeviceInitializationLocalSmokeClassification.ps1 -Strict" -ExpectedOutput "validationState with local smoke boundaries intact" -Boundary "Validation cannot promote runtime proof."
  New-ExecutionCommand -Order 3 -Id "prepare-owner-clean-consumer" -Command "Follow artifacts/final-release/clean-consumer-proof-owner-execution-pack.md on a compatible external host." -ExpectedOutput "repository-external consumer, package restore/build/smoke logs, hashes, host metadata" -Boundary "Owner action required; this command is guidance only."
  New-ExecutionCommand -Order 4 -Id "strict-runtime-proof-validation" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -ExpectedOutput "proofClassification=package-consumer-runtime and smokeStatus=passed" -Boundary "Only this strict validator can promote package-consumer runtime proof."
  New-ExecutionCommand -Order 5 -Id "forbidden-substitute-scan" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1" -ExpectedOutput "no forbidden substitutes detected" -Boundary "Scan is an audit control, not proof."
  New-ExecutionCommand -Order 6 -Id "refresh-release-evidence" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" -ExpectedOutput "release evidence and classification audit keep non-proof boundaries intact" -Boundary "Refresh aggregates state only."
  New-ExecutionCommand -Order 7 -Id "post-publish-owner-gate" -Command "After owner publish, run post-publish clean consumer proof validators with existing logs." -ExpectedOutput "post-publish clean consumer proof record and strict close validator pass" -Boundary "Not executable before public publication."
)

$record = [pscustomobject]@{
  recordKind = "clean-consumer-proof-execution-bundle"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bundleState = "blocked-owner-clean-consumer-runtime-and-post-publish-proof-required"
  sourceArtifacts = @($sourceArtifacts)
  sourceStates = $sourceStates
  laneCount = $lanes.Count
  lanes = @($lanes)
  forbiddenSubstituteCount = $forbiddenSubstitutes.Count
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  promotionRequirementCount = $promotionRequirements.Count
  promotionRequirements = @($promotionRequirements)
  executionCommandCount = $executionCommands.Count
  executionCommands = @($executionCommands)
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  nonSubstituteProofKinds = @($forbiddenSubstitutes + @("clean-consumer-proof-execution-bundle", "local-smoke-not-external-proof", "owner-action-required"))
  boundary = "This bundle is an execution map and release evidence classifier only. It does not run runtime smoke, does not publish packages, does not approve release close, and cannot promote package-consumer-runtime or post-publish proof without real repository-external clean consumer logs, hashes, package metadata, host metadata, owner review, and strict validators with FailOnNotProof."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-proof-execution-bundle.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-proof-execution-bundle.md"
$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = $lanes | ForEach-Object {
  "| $($_.order) | ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.title) | ``$($_.state)`` | ``$($_.evidenceKind)`` | ``$($_.ownerActionRequired)`` | $(ConvertTo-MarkdownCell $_.boundary) |"
}
$requirementRows = $promotionRequirements | ForEach-Object {
  "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.description) | ``$($_.validator)`` | $(ConvertTo-MarkdownCell $_.rejects) |"
}
$commandRows = $executionCommands | ForEach-Object {
  "| $($_.order) | ``$($_.id)`` | ``$($_.command)`` | $(ConvertTo-MarkdownCell $_.expectedOutput) | $(ConvertTo-MarkdownCell $_.boundary) |"
}
$forbiddenRows = $forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# Clean Consumer Proof Execution Bundle

| Field | Value |
| --- | --- |
| recordKind | ``$($record.recordKind)`` |
| bundleState | ``$($record.bundleState)`` |
| laneCount | ``$($record.laneCount)`` |
| forbiddenSubstituteCount | ``$($record.forbiddenSubstituteCount)`` |
| promotionRequirementCount | ``$($record.promotionRequirementCount)`` |
| executionCommandCount | ``$($record.executionCommandCount)`` |
| ownerActionRequired | ``$($record.ownerActionRequired)`` |
| performsPublish | ``$($record.performsPublish)`` |
| performsRuntimeExecution | ``$($record.performsRuntimeExecution)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |
| isPackageConsumerRuntimeProof | ``$($record.isPackageConsumerRuntimeProof)`` |
| isPostPublishProof | ``$($record.isPostPublishProof)`` |

## Lanes

| # | ID | Title | State | Evidence Kind | Owner Action | Boundary |
|---:|---|---|---|---|---:|---|
$($laneRows -join "`r`n")

## Promotion Requirements

| ID | Description | Validator | Rejects |
|---|---|---|---|
$($requirementRows -join "`r`n")

## Execution Commands

| # | ID | Command | Expected Output | Boundary |
|---:|---|---|---|---|
$($commandRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer proof execution bundle written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "BundleState=$($record.bundleState) Lanes=$($record.laneCount) CanPromote=$($record.canPromoteRuntimeProof) CanPublish=$($record.canPublishPublicly) CanClose=$($record.canCloseReleaseIssue)"
