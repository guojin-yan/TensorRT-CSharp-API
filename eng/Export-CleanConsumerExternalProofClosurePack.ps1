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

function New-ClosureLane {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$CurrentState,
    [string[]]$RequiredInputs,
    [string[]]$ValidatorCommands,
    [string[]]$OutputArtifacts,
    [string]$BlockedUntil,
    [string]$Boundary
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    currentState = $CurrentState
    ownerActionRequired = $true
    requiredInputs = @($RequiredInputs)
    validatorCommands = @($ValidatorCommands)
    outputArtifacts = @($OutputArtifacts)
    blockedUntil = $BlockedUntil
    performsPublish = $false
    performsRuntimeExecution = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    boundary = $Boundary
  }
}

function New-ClosureField {
  param([string]$Id, [string]$FieldPath, [string]$RequiredEvidence, [string]$StrictValidator, [string]$Rejects)

  [pscustomobject]@{
    id = $Id
    fieldPath = $FieldPath
    requiredEvidence = $RequiredEvidence
    strictValidator = $StrictValidator
    rejects = $Rejects
    ownerActionRequired = $true
  }
}

function New-CommandStep {
  param([int]$Order, [string]$Id, [string]$Command, [string]$RequiredOutput, [string]$Boundary)

  [pscustomobject]@{
    order = $Order
    id = $Id
    command = $Command
    requiredOutput = $RequiredOutput
    ownerActionRequired = $true
    performsPublish = $false
    performsRuntimeExecution = $false
    canPromoteRuntimeProof = $false
    boundary = $Boundary
  }
}

$cleanConsumerBundle = Read-JsonOrNull "artifacts/final-release/clean-consumer-proof-execution-bundle.json"
$cleanConsumerBundleValidation = Read-JsonOrNull "artifacts/final-release/clean-consumer-proof-execution-bundle-validation.json"
$cleanConsumerOwnerPack = Read-JsonOrNull "artifacts/final-release/clean-consumer-proof-owner-execution-pack.json"
$cleanConsumerOwnerPackValidation = Read-JsonOrNull "artifacts/final-release/clean-consumer-proof-owner-execution-pack-validation.json"
$ownerExternalExecutionBundle = Read-JsonOrNull "artifacts/final-release/owner-external-proof-execution-bundle.json"
$ownerExternalExecutionBundleValidation = Read-JsonOrNull "artifacts/final-release/owner-external-proof-execution-bundle-validation.json"
$externalCleanConsumerProofKit = Read-JsonOrNull "artifacts/final-release/external-clean-consumer-proof-kit.json"
$externalCleanConsumerProofKitValidation = Read-JsonOrNull "artifacts/final-release/external-clean-consumer-proof-kit-validation.json"
$ownerExternalRealProofInputContract = Read-JsonOrNull "artifacts/final-release/owner-external-real-proof-input-contract.json"
$ownerExternalRealProofInputContractValidation = Read-JsonOrNull "artifacts/final-release/owner-external-real-proof-input-contract-validation.json"
$ownerExternalRealProofImportValidator = Read-JsonOrNull "artifacts/final-release/owner-external-real-proof-import-validator.json"
$ownerExternalRealProofImportValidatorValidation = Read-JsonOrNull "artifacts/final-release/owner-external-real-proof-import-validator-validation.json"
$runtimeCompatibleHostRealProofGate = Read-JsonOrNull "artifacts/final-release/runtime-compatible-host-real-proof-gate.json"
$runtimeCompatibleHostRealProofGateValidation = Read-JsonOrNull "artifacts/final-release/runtime-compatible-host-real-proof-gate-validation.json"
$postPublishCleanConsumerRealProofGate = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-real-proof-gate.json"
$postPublishCleanConsumerRealProofGateValidation = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-real-proof-gate-validation.json"
$postPublishCleanConsumerProofRecordContract = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json"
$postPublishCleanConsumerProofRecordContractValidation = Read-JsonOrNull "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json"
$finalPostPublishCleanConsumerProofRecordContract = Read-JsonOrNull "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json"
$finalPostPublishCleanConsumerProofRecordContractValidation = Read-JsonOrNull "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract-validation.json"
$releaseCloseRealProofReadinessGate = Read-JsonOrNull "artifacts/final-release/release-close-real-proof-readiness-gate.json"
$releaseCloseRealProofReadinessGateValidation = Read-JsonOrNull "artifacts/final-release/release-close-real-proof-readiness-gate-validation.json"
$packageConsumerPreflightMatrix = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json"
$externalRuntimeProofInputTemplate = Read-JsonOrNull "artifacts/final-release/external-runtime-proof-record.input-template.json"
$releaseEvidence = Read-JsonOrNull "artifacts/final-release/release-evidence-bundle.json"

$sourceArtifacts = @(
  "artifacts/final-release/clean-consumer-proof-execution-bundle.json",
  "artifacts/final-release/clean-consumer-proof-execution-bundle-validation.json",
  "artifacts/final-release/clean-consumer-proof-owner-execution-pack.json",
  "artifacts/final-release/clean-consumer-proof-owner-execution-pack-validation.json",
  "artifacts/final-release/owner-external-proof-execution-bundle.json",
  "artifacts/final-release/owner-external-proof-execution-bundle-validation.json",
  "artifacts/final-release/external-clean-consumer-proof-kit.json",
  "artifacts/final-release/external-clean-consumer-proof-kit-validation.json",
  "artifacts/final-release/owner-external-real-proof-input-contract.json",
  "artifacts/final-release/owner-external-real-proof-input-contract-validation.json",
  "artifacts/final-release/owner-external-real-proof-import-validator.json",
  "artifacts/final-release/owner-external-real-proof-import-validator-validation.json",
  "artifacts/final-release/runtime-compatible-host-real-proof-gate.json",
  "artifacts/final-release/runtime-compatible-host-real-proof-gate-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-real-proof-gate.json",
  "artifacts/final-release/post-publish-clean-consumer-real-proof-gate-validation.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
  "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract.json",
  "artifacts/final-release/final-post-publish-clean-consumer-proof-record-contract-validation.json",
  "artifacts/final-release/release-close-real-proof-readiness-gate.json",
  "artifacts/final-release/release-close-real-proof-readiness-gate-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json",
  "artifacts/final-release/external-runtime-proof-record.input-template.json",
  "artifacts/final-release/release-evidence-bundle.json"
)

$sourceStates = [pscustomobject]@{
  cleanConsumerProofExecutionBundle = [string](Get-PropertyOrDefault -Object $cleanConsumerBundleValidation -Name "validationState" -DefaultValue "missing-clean-consumer-proof-execution-bundle-validation")
  cleanConsumerProofOwnerExecutionPack = [string](Get-PropertyOrDefault -Object $cleanConsumerOwnerPackValidation -Name "validationState" -DefaultValue "missing-clean-consumer-proof-owner-execution-pack-validation")
  ownerExternalProofExecutionBundle = [string](Get-PropertyOrDefault -Object $ownerExternalExecutionBundleValidation -Name "validationState" -DefaultValue "missing-owner-external-proof-execution-bundle-validation")
  externalCleanConsumerProofKit = [string](Get-PropertyOrDefault -Object $externalCleanConsumerProofKitValidation -Name "validationState" -DefaultValue "missing-external-clean-consumer-proof-kit-validation")
  ownerExternalRealProofInputContract = [string](Get-PropertyOrDefault -Object $ownerExternalRealProofInputContractValidation -Name "validationState" -DefaultValue "missing-owner-external-real-proof-input-contract-validation")
  ownerExternalRealProofImportValidator = [string](Get-PropertyOrDefault -Object $ownerExternalRealProofImportValidatorValidation -Name "validationState" -DefaultValue "missing-owner-external-real-proof-import-validator-validation")
  runtimeCompatibleHostRealProofGate = [string](Get-PropertyOrDefault -Object $runtimeCompatibleHostRealProofGateValidation -Name "validationState" -DefaultValue "missing-runtime-compatible-host-real-proof-gate-validation")
  postPublishCleanConsumerRealProofGate = [string](Get-PropertyOrDefault -Object $postPublishCleanConsumerRealProofGateValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-real-proof-gate-validation")
  postPublishCleanConsumerProofRecordContract = [string](Get-PropertyOrDefault -Object $postPublishCleanConsumerProofRecordContractValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-record-contract-validation")
  finalPostPublishCleanConsumerProofRecordContract = [string](Get-PropertyOrDefault -Object $finalPostPublishCleanConsumerProofRecordContractValidation -Name "validationState" -DefaultValue "missing-final-post-publish-clean-consumer-proof-record-contract-validation")
  releaseCloseRealProofReadinessGate = [string](Get-PropertyOrDefault -Object $releaseCloseRealProofReadinessGateValidation -Name "validationState" -DefaultValue "missing-release-close-real-proof-readiness-gate-validation")
  packageConsumerRuntimeProofPreflightMatrixPresent = $null -ne $packageConsumerPreflightMatrix
  packageConsumerRuntimeProofPreflightEntryCount = @((Get-PropertyOrDefault -Object $packageConsumerPreflightMatrix -Name "entries" -DefaultValue @())).Count
  externalRuntimeProofInputTemplatePresent = $null -ne $externalRuntimeProofInputTemplate
  releaseEvidenceStatus = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "status" -DefaultValue "missing-release-evidence-bundle")
}

$closureLanes = @(
  New-ClosureLane -Order 1 -Id "owner-input-contracts" -Title "Owner input contracts and field map" -CurrentState "owner-action-required" -RequiredInputs @(
    "owner external real proof input contract",
    "external runtime proof input template",
    "post-publish clean consumer proof record contract",
    "package-consumer runtime proof preflight matrix"
  ) -ValidatorCommands @(
    "eng/Test-OwnerExternalRealProofInputContract.ps1 -Strict",
    "eng/Test-PostPublishCleanConsumerProofRecordContract.ps1 -Strict",
    "eng/Test-CleanConsumerProofExecutionBundle.ps1 -Strict"
  ) -OutputArtifacts @(
    "owner-external-real-proof-input-contract-validation.json",
    "post-publish-clean-consumer-proof-record-contract-validation.json"
  ) -BlockedUntil "Owner fills all required real input fields with file paths, SHA256 values, package metadata, host metadata, and review identity." -Boundary "Input contracts are owner-action surfaces only; they are not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  New-ClosureLane -Order 2 -Id "clean-external-package-consumer-runtime" -Title "Repository-external clean package consumer runtime proof" -CurrentState "owner-action-required" -RequiredInputs @(
    "clean consumer path outside repository",
    "no ProjectReference",
    "managed and runtime packages restored from package source",
    "native asset listing and SHA256",
    "restore/build/runtime smoke stdout and stderr logs",
    "runtime smoke log SHA256",
    "smoke exit code 0 and smokeStatus=passed"
  ) -ValidatorCommands @(
    "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
    "eng/Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
    "eng/Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1"
  ) -OutputArtifacts @(
    "package-consumer-runtime-proof-record-validation.json",
    "external-runtime-proof-validation.json",
    "package-consumer-runtime-proof-forbidden-substitute-scan.json"
  ) -BlockedUntil "A real repository-external clean consumer runtime smoke log exists and strict validators accept it." -Boundary "Clean consumer closure guidance is not runtime proof until real logs, hashes, package metadata, host metadata, owner review, and FailOnNotProof validators pass."
  New-ClosureLane -Order 3 -Id "compatible-cuda-host-runtime" -Title "Compatible CUDA/TensorRT host runtime proof" -CurrentState "owner-action-required" -RequiredInputs @(
    "host OS architecture and RID",
    "GPU name",
    "NVIDIA driver version",
    "CUDA runtime/toolkit version",
    "TensorRT runtime version",
    "cuDNN version when applicable",
    "runtime package key matching selected package"
  ) -ValidatorCommands @(
    "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
    "eng/Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
    "eng/Test-OwnerExternalRealProofImportValidator.ps1 -Strict"
  ) -OutputArtifacts @(
    "owner-external-real-proof-import-validator-validation.json",
    "runtime-compatible-host-real-proof-gate-validation.json"
  ) -BlockedUntil "Compatible host metadata and runtime smoke evidence are captured from the same owner execution." -Boundary "Compatible host metadata is required context only; it is not runtime proof without real clean consumer execution logs and strict validation."
  New-ClosureLane -Order 4 -Id "owner-external-proof-result-import" -Title "Owner external proof result import" -CurrentState "owner-action-required" -RequiredInputs @(
    "owner external proof execution result input",
    "stdout/stderr/merged transcript paths",
    "validator output path",
    "all SHA256 values",
    "non-substitute confirmations",
    "owner review name, machine, timestamp, and note"
  ) -ValidatorCommands @(
    "eng/Import-OwnerExternalProofExecutionResult.ps1 -Strict",
    "eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict",
    "eng/Test-RealExternalProofRecordImportValidator.ps1 -Strict"
  ) -OutputArtifacts @(
    "owner-external-proof-execution-result-import-validation.json",
    "real-external-proof-record-import-validator-validation.json"
  ) -BlockedUntil "Owner imports real result records and import validators accept all paths, hashes, and non-substitute confirmations." -Boundary "Import is a validation and projection step only; it is not publish approval, not release close approval, not package push, and not proof without accepted real execution logs."
  New-ClosureLane -Order 5 -Id "post-publish-clean-consumer-proof" -Title "Post-publish clean consumer proof" -CurrentState "owner-action-required-after-publication" -RequiredInputs @(
    "public package source after publish",
    "downloaded managed and runtime nupkg SHA256",
    "repository-external post-publish consumer project",
    "post-publish restore/build/run logs",
    "post-publish runtime smoke log SHA256",
    "host metadata from post-publish execution",
    "owner non-substitute confirmation"
  ) -ValidatorCommands @(
    "eng/Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict",
    "eng/Test-PostPublishCleanConsumerRealProofGate.ps1 -Strict",
    "eng/Test-PostPublishCleanConsumerProofRecordContract.ps1 -Strict"
  ) -OutputArtifacts @(
    "post-publish-clean-consumer-proof-record-draft-validation.json",
    "post-publish-clean-consumer-real-proof-gate-validation.json"
  ) -BlockedUntil "Public package has been published and real post-publish clean consumer evidence is supplied." -Boundary "Post-publish lane is separate owner proof after publication; pre-publish smoke, local feed, dashboards, drafts, and runbooks are not post-publish proof."
  New-ClosureLane -Order 6 -Id "strict-close-admission" -Title "Strict release close admission" -CurrentState "owner-action-required" -RequiredInputs @(
    "package-consumer runtime proof accepted",
    "compatible host proof accepted",
    "post-publish clean consumer proof accepted",
    "public package hash cross-check accepted",
    "rollback review completed",
    "final owner decision accepted"
  ) -ValidatorCommands @(
    "eng/Test-ReleaseCloseRealProofReadinessGate.ps1 -Strict",
    "eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady",
    "eng/Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
  ) -OutputArtifacts @(
    "release-close-real-proof-readiness-gate-validation.json",
    "release-evidence-classification-audit.json"
  ) -BlockedUntil "All real proof lanes and final owner close decisions pass strict validators." -Boundary "Strict close admission is blocked owner decision workflow; it is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push by itself."
)

$requiredOwnerFields = @(
  New-ClosureField -Id "clean-consumer-root" -FieldPath "consumer.projectRoot" -RequiredEvidence "Repository-external clean consumer path and project hash." -StrictValidator "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Rejects "in-repository sample, ProjectReference"
  New-ClosureField -Id "package-source" -FieldPath "packageSource.url" -RequiredEvidence "Public or owner-approved package source used for restore." -StrictValidator "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Rejects "local feed, direct nupkg"
  New-ClosureField -Id "managed-runtime-package-identity" -FieldPath "packageSource.managedPackageId/runtimePackageId/runtimePackageKey/version" -RequiredEvidence "Exact package ids, versions, runtime key, and restore source." -StrictValidator "eng/Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" -Rejects "mismatched runtime key"
  New-ClosureField -Id "native-assets" -FieldPath "results.nativeAssetsFound/nativeAssetSha256" -RequiredEvidence "Native asset listing and SHA256 for runtime package assets." -StrictValidator "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Rejects "asset listing without runtime smoke"
  New-ClosureField -Id "runtime-smoke-logs" -FieldPath "command.stdoutLogPath/stderrLogPath/logSha256" -RequiredEvidence "Existing runtime smoke logs with matching SHA256 values." -StrictValidator "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Rejects "missing log, mismatched SHA256, copied summary"
  New-ClosureField -Id "host-metadata" -FieldPath "host.os/arch/gpu/driver/cuda/tensorrt/cudnn" -RequiredEvidence "Compatible host metadata from the same execution." -StrictValidator "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Rejects "dependency-probe-only, blocked-by-cuda-driver"
  New-ClosureField -Id "owner-review" -FieldPath "owner.name/machineName/reviewedAtUtc/reviewNote" -RequiredEvidence "Owner review identity, machine, timestamp, and non-substitute confirmation." -StrictValidator "eng/Import-OwnerExternalProofExecutionResult.ps1 -Strict" -Rejects "anonymous draft, candidate only"
  New-ClosureField -Id "post-publish-evidence" -FieldPath "postPublish.installLog/runLog/downloadedNupkgSha256" -RequiredEvidence "Post-publish clean consumer logs and downloaded package hashes." -StrictValidator "eng/Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict" -Rejects "pre-publish smoke reused as post-publish proof"
)

$executionSteps = @(
  New-CommandStep -Order 1 -Id "select-runtime-proof-preflight-option" -Command "Select runtimePackageKey from package-consumer-runtime-proof-preflight-matrix.json and copy package metadata into owner result input." -RequiredOutput "Selected runtimePackageKey, runtimePackageId, restoreSourceMode, nativeAssetsExpected." -Boundary "Preflight selection is metadata alignment only and not proof."
  New-CommandStep -Order 2 -Id "create-clean-consumer-outside-repository" -Command "Create a repository-external .NET consumer project and ensure no ProjectReference points back to this repository." -RequiredOutput "External csproj path, csproj SHA256, no ProjectReference confirmation." -Boundary "Project creation is setup only and not runtime proof."
  New-CommandStep -Order 3 -Id "restore-managed-runtime-packages" -Command "Restore managed and runtime packages from the declared package source; capture restore log and package identities." -RequiredOutput "Restore log, package id/version/runtime key/source URL." -Boundary "Restore alone is build/setup output and not runtime proof."
  New-CommandStep -Order 4 -Id "run-clean-consumer-runtime-smoke" -Command "Run the clean consumer runtime smoke on a compatible CUDA/TensorRT host and capture stdout/stderr." -RequiredOutput "Runtime smoke logs, exit code 0, smokeStatus=passed." -Boundary "This is promotable only after strict validators accept existing logs and hashes."
  New-CommandStep -Order 5 -Id "hash-logs-packages-native-assets" -Command "Compute SHA256 for restore/build/run logs, validator output, managed/runtime nupkg, and native asset listing." -RequiredOutput "SHA256 values matching owner input." -Boundary "Hashes authenticate files but are not proof by themselves."
  New-CommandStep -Order 6 -Id "import-owner-external-result" -Command "Import owner external proof execution result and run strict real proof validators." -RequiredOutput "Accepted owner result import and strict proof validator outputs." -Boundary "Import cannot promote proof while required real execution fields are missing."
  New-CommandStep -Order 7 -Id "refresh-release-evidence" -Command "Run Export-ReleaseEvidenceBundle.ps1 and Test-ReleaseEvidenceClassificationAudit.ps1 -Strict." -RequiredOutput "Release evidence and classification audit keep non-proof boundaries intact." -Boundary "Refresh aggregates evidence only and cannot publish or close."
  New-CommandStep -Order 8 -Id "post-publish-clean-consumer-after-publication" -Command "After public publication, run post-publish clean consumer proof validators with public package source logs." -RequiredOutput "Post-publish clean consumer proof record and strict close validator readiness." -Boundary "Cannot execute before publication and cannot be replaced by pre-publish smoke."
)

$forbiddenSubstitutes = @(
  "Skipped=True",
  "local smoke",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "build-only",
  "dependency-probe",
  "blocked-by-cuda-driver",
  "dashboard",
  "runbook",
  "candidate",
  "draft",
  "template",
  "preflight-only",
  "dry-run-only",
  "schema-only",
  "owner input without strict validator pass",
  "host metadata without runtime smoke",
  "package hash without existing log validation",
  "native asset listing without runtime smoke",
  "pre-publish smoke reused as post-publish proof"
)

$strictValidatorCommands = @(
  "eng/Test-CleanConsumerProofExecutionBundle.ps1 -Strict",
  "eng/Test-OwnerExternalProofExecutionBundle.ps1 -Strict",
  "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
  "eng/Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof",
  "eng/Import-OwnerExternalProofExecutionResult.ps1 -Strict",
  "eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict",
  "eng/Test-OwnerExternalRealProofImportValidator.ps1 -Strict",
  "eng/Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict",
  "eng/Test-PostPublishCleanConsumerRealProofGate.ps1 -Strict",
  "eng/Test-ReleaseCloseRealProofReadinessGate.ps1 -Strict",
  "eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady",
  "eng/Export-ReleaseEvidenceBundle.ps1",
  "eng/Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
)

$record = [pscustomobject]@{
  recordKind = "clean-consumer-external-proof-closure-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  closureState = "blocked-owner-external-clean-consumer-proof-closure-required"
  sourceArtifacts = @($sourceArtifacts)
  sourceStates = $sourceStates
  closureLaneCount = $closureLanes.Count
  closureLanes = @($closureLanes)
  requiredOwnerFieldCount = $requiredOwnerFields.Count
  requiredOwnerFields = @($requiredOwnerFields)
  executionStepCount = $executionSteps.Count
  executionSteps = @($executionSteps)
  strictValidatorCommands = @($strictValidatorCommands)
  forbiddenSubstituteCount = $forbiddenSubstitutes.Count
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  nonSubstituteProofKinds = @($forbiddenSubstitutes + @("clean-consumer-external-proof-closure-pack", "external proof closure guidance", "owner-action closure pack"))
  boundary = "This closure pack is a blocked owner-action convergence layer only. It links clean consumer runtime, compatible host runtime, owner external result import, post-publish clean consumer proof, and strict close admission, but it does not run runtime smoke, does not publish packages, does not approve public release, does not close the release issue, and cannot promote proof without real external logs, SHA256 values, package metadata, native asset evidence, host metadata, owner review, and strict validators with FailOnNotProof."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-external-proof-closure-pack.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-external-proof-closure-pack.md"
$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = $closureLanes | ForEach-Object {
  "| $($_.order) | ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.title) | ``$($_.currentState)`` | $(ConvertTo-MarkdownCell $_.blockedUntil) | $(ConvertTo-MarkdownCell $_.boundary) |"
}
$fieldRows = $requiredOwnerFields | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.fieldPath)`` | $(ConvertTo-MarkdownCell $_.requiredEvidence) | ``$($_.strictValidator)`` | $(ConvertTo-MarkdownCell $_.rejects) |"
}
$stepRows = $executionSteps | ForEach-Object {
  "| $($_.order) | ``$($_.id)`` | ``$($_.command)`` | $(ConvertTo-MarkdownCell $_.requiredOutput) | $(ConvertTo-MarkdownCell $_.boundary) |"
}
$validatorRows = $strictValidatorCommands | ForEach-Object { "- ``$_``" }
$forbiddenRows = $forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# Clean Consumer External Proof Closure Pack

| Field | Value |
| --- | --- |
| recordKind | ``$($record.recordKind)`` |
| closureState | ``$($record.closureState)`` |
| closureLaneCount | ``$($record.closureLaneCount)`` |
| requiredOwnerFieldCount | ``$($record.requiredOwnerFieldCount)`` |
| executionStepCount | ``$($record.executionStepCount)`` |
| forbiddenSubstituteCount | ``$($record.forbiddenSubstituteCount)`` |
| ownerActionRequired | ``$($record.ownerActionRequired)`` |
| performsPublish | ``$($record.performsPublish)`` |
| performsRuntimeExecution | ``$($record.performsRuntimeExecution)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |
| isPackageConsumerRuntimeProof | ``$($record.isPackageConsumerRuntimeProof)`` |
| isPostPublishProof | ``$($record.isPostPublishProof)`` |
| isReleaseCloseProof | ``$($record.isReleaseCloseProof)`` |

## Closure Lanes

| # | ID | Title | State | Blocked Until | Boundary |
|---:|---|---|---|---|---|
$($laneRows -join "`r`n")

## Required Owner Fields

| ID | Field Path | Required Evidence | Strict Validator | Rejects |
|---|---|---|---|---|
$($fieldRows -join "`r`n")

## Execution Steps

| # | ID | Command | Required Output | Boundary |
|---:|---|---|---|---|
$($stepRows -join "`r`n")

## Strict Validators

$($validatorRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer external proof closure pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ClosureState=$($record.closureState) Lanes=$($record.closureLaneCount) CanPromote=$($record.canPromoteRuntimeProof) CanPublish=$($record.canPublishPublicly) CanClose=$($record.canCloseReleaseIssue)"
