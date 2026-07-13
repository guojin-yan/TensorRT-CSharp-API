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

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-OwnerInput {
  param([string]$Id, [string]$Field, [string]$RequiredEvidence, [string]$Validator)

  [pscustomobject]@{
    id = $Id
    field = $Field
    requiredEvidence = $RequiredEvidence
    validator = $Validator
    ownerActionRequired = $true
  }
}

function New-OwnerCommand {
  param([string]$Id, [string]$Command, [string]$Purpose, [string]$Boundary)

  [pscustomobject]@{
    id = $Id
    command = $Command
    purpose = $Purpose
    boundary = $Boundary
    performsPublish = $false
    canPromoteRuntimeProof = $false
  }
}

$prePublish = Read-JsonOrNull "artifacts/final-release/final-release-pre-publish-audit-matrix.json"
$prePublishValidation = Read-JsonOrNull "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json"
$releaseEvidence = Read-JsonOrNull "artifacts/final-release/release-evidence-bundle.json"
$dryRun = Read-JsonOrNull "artifacts/final-release/final-release-dry-run-summary.json"
$closeBlockerDashboard = Read-JsonOrNull "artifacts/final-release/final-release-close-blocker-dashboard.json"
$ownerInputSchema = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json"
$ownerInputImport = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json"
$forbiddenScan = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json"
$packageConsumerRecord = Read-JsonOrNull "artifacts/final-release/package-consumer-runtime-proof-record-validation.json"
$publicPublishPack = Read-JsonOrNull "artifacts/final-release/public-publish-final-owner-execution-pack.json"
$postPublishPack = Read-JsonOrNull "artifacts/final-release/final-post-publish-audit-pack.json"
$ownerRuntimeSmokeFieldAlignment = Read-JsonOrNull "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json"
$ownerRuntimeSmokeFieldAlignmentValidation = Read-JsonOrNull "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"

$prePublishAuditMatrixReady = [string](Get-PropertyOrDefault -Object $prePublishValidation -Name "validationState" -DefaultValue "") -eq "final-release-pre-publish-audit-matrix-ready"
$releaseEvidenceBundleReady = $null -ne $releaseEvidence
$finalDryRunReady = $null -ne $dryRun
$closeBlockerDashboardReady = $null -ne $closeBlockerDashboard
$ownerInputSchemaReady = $null -ne $ownerInputSchema -or [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputSchemaReady" -DefaultValue $false)
$cleanOwnerInputReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "cleanOwnerInputReady" -DefaultValue $false)
$ownerInputCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $false)
$forbiddenSubstituteScanState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "forbiddenSubstituteScanState" -DefaultValue ([string](Get-PropertyOrDefault -Object $forbiddenScan -Name "scanState" -DefaultValue "missing-forbidden-substitute-scan")))
$detectedForbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "detectedForbiddenSubstituteCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $forbiddenScan -Name "detectedForbiddenSubstituteCount" -DefaultValue 0)))
$packageConsumerRuntimeProofRecordReady = [string](Get-PropertyOrDefault -Object $packageConsumerRecord -Name "validationState" -DefaultValue "") -eq "package-consumer-runtime-proof-record-ready"
$externalRuntimeProofReady = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofReady" -DefaultValue $false)
$postPublishVerificationReady = [bool](Get-PropertyOrDefault -Object $postPublishPack -Name "postPublishVerificationReady" -DefaultValue $false)
$publicPublishOwnerCommandReady = [string](Get-PropertyOrDefault -Object $publicPublishPack -Name "executionPackState" -DefaultValue "") -eq "ready-owner-public-publish-execution"
$tensorRtExecNonProofBoundaryReady = [int](Get-PropertyOrDefault -Object $prePublish -Name "tensorRtExecRuntimeProofItems" -DefaultValue 1) -eq 0
$yoloVisionRealModelBackfillReady = [string](Get-PropertyOrDefault -Object $prePublish -Name "yoloVisionEvidenceState" -DefaultValue "") -eq "release-readiness-planning"
$onnxToEngineBuildReportBoundaryReady = [string](Get-PropertyOrDefault -Object $prePublish -Name "onnxToEngineParityState" -DefaultValue "") -eq "onnx-to-engine-trtexec-parity-matrix"
$ownerRuntimeSmokeFieldAlignmentState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "alignmentState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment")
$ownerRuntimeSmokeFieldAlignmentValidationState = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "validationState" -DefaultValue "missing-package-consumer-owner-runtime-smoke-field-alignment-validation")
$ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = [string](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "runtimeSmokeStatus" -DefaultValue "Smoke=missing")
$ownerRuntimeSmokeFieldAlignmentFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "fieldCount" -DefaultValue 0)
$ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignment -Name "missingRequiredFieldCount" -DefaultValue -1)
$ownerRuntimeSmokeFieldAlignmentFailedBlockerCount = [int](Get-PropertyOrDefault -Object $ownerRuntimeSmokeFieldAlignmentValidation -Name "failedBlockerCount" -DefaultValue -1)

$requiredOwnerInputs = @(
  New-OwnerInput -Id "public-package-source-url" -Field "packageSource.publicPackageSourceUrl" -RequiredEvidence "Official public package URL or package feed source used by a clean external consumer." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-OwnerInput -Id "package-id-version" -Field "packageSource.packageId/packageSource.packageVersion" -RequiredEvidence "Exact package id and version consumed by the clean external consumer." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-OwnerInput -Id "package-sha256" -Field "packageSource.managedNupkgSha256/packageSource.runtimeNupkgSha256" -RequiredEvidence "SHA256 for exact managed and runtime nupkg files consumed." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-OwnerInput -Id "clean-consumer-repo" -Field "consumer.projectPath/consumer.projectHash" -RequiredEvidence "Repository-external clean consumer path and project identity hash." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-OwnerInput -Id "restore-build-smoke-command" -Field "commands.restoreCommand/commands.buildCommand/commands.smokeCommand" -RequiredEvidence "Commands executed in the clean consumer, including runtime package key." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-OwnerInput -Id "smoke-exit-code" -Field "results.exitCode" -RequiredEvidence "Exit code 0 from the clean consumer smoke command." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-OwnerInput -Id "smoke-log-hash" -Field "results.stdoutLogPath/results.stderrLogPath/results.logSha256" -RequiredEvidence "Persisted stdout/stderr logs and matching SHA256." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-OwnerInput -Id "host-os" -Field "host.os" -RequiredEvidence "Operating system and architecture for the proof host." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-OwnerInput -Id "gpu-driver-cuda-trt" -Field "host.gpu/host.driverVersion/host.cudaVersion/host.tensorRtVersion" -RequiredEvidence "GPU, NVIDIA driver, CUDA, TensorRT, and cuDNN metadata." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-OwnerInput -Id "runtime-package-metadata" -Field "packageSource.runtimePackageKey/packageSource.runtimePackageVersion" -RequiredEvidence "Runtime package key and version match the consumed package and smoke command." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-OwnerInput -Id "owner-review" -Field "owner.name/owner.reviewedAtUtc/owner.reviewNote" -RequiredEvidence "Owner review identity, timestamp, and review note." -Validator "Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict"
  New-OwnerInput -Id "post-publish-verification" -Field "postPublish.packageUrl/postPublish.installCommand/postPublish.verificationLogSha256" -RequiredEvidence "Public channel package URL, clean install command, and verification log." -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
)

$requiredCommands = @(
  New-OwnerCommand -Id "export-owner-input-template" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1" -Purpose "Refresh owner input template." -Boundary "Template only; not runtime proof."
  New-OwnerCommand -Id "import-owner-input" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict" -Purpose "Import owner-filled clean consumer input and project a proof record." -Boundary "Import performs no package publish and cannot promote proof while owner input is incomplete."
  New-OwnerCommand -Id "validate-package-consumer-record" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof" -Purpose "Validate real clean consumer runtime proof with existing log hash." -Boundary "Strict validation with existing logs and FailOnNotProof is required before proof promotion."
  New-OwnerCommand -Id "validate-forbidden-substitutes" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1" -Purpose "Scan for forbidden substitute proof sources." -Boundary "Scan output is an audit, not runtime proof."
  New-OwnerCommand -Id "validate-post-publish" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" -Purpose "Validate post-publish public channel verification." -Boundary "Cannot run before real public package channel evidence exists; existing logs and FailOnNotProof are required."
  New-OwnerCommand -Id "refresh-pre-publish-matrix" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleasePrePublishAuditMatrix.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleasePrePublishAuditMatrix.ps1 -Strict" -Purpose "Refresh pre-publish non-proof boundary matrix." -Boundary "Pre-publish matrix cannot publish or close release."
)

$requiredValidators = @(
  "eng/Test-FinalReleasePrePublishAuditMatrix.ps1 -Strict",
  "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
  "eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
  "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
  "eng/Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1",
  "eng/Test-FinalReleaseDryRun.ps1",
  "eng/Test-FinalReleaseCloseBlockerDashboard.ps1 -Strict",
  "eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
)

$forbiddenSubstitutes = @(
  "template",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "dry-run",
  "build-only",
  "preflight-only",
  "GUI screenshot",
  "TensorRtExec report",
  "YoloVision matrix",
  "OnnxToEngine report",
  "sample manifest",
  "sidecar-only",
  "readonly diagnostics"
)

$sourceArtifacts = @(
  "artifacts/final-release/final-release-pre-publish-audit-matrix.json",
  "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-release-dry-run-summary.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json",
  "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json",
  "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
  "artifacts/final-release/public-publish-final-owner-execution-pack.json",
  "artifacts/final-release/final-post-publish-audit-pack.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment.json",
  "artifacts/final-release/package-consumer-owner-runtime-smoke-field-alignment-validation.json"
)

$readySignals = @(
  $prePublishAuditMatrixReady,
  $releaseEvidenceBundleReady,
  $finalDryRunReady,
  $closeBlockerDashboardReady,
  $ownerInputSchemaReady,
  $cleanOwnerInputReady,
  $ownerInputCanPromoteRuntimeProof,
  $packageConsumerRuntimeProofRecordReady,
  $externalRuntimeProofReady,
  $postPublishVerificationReady,
  $publicPublishOwnerCommandReady,
  $tensorRtExecNonProofBoundaryReady,
  $yoloVisionRealModelBackfillReady,
  $onnxToEngineBuildReportBoundaryReady
)
$remainingOwnerActionCount = @($readySignals | Where-Object { -not [bool]$_ }).Count

$record = [pscustomobject]@{
  recordKind = "final-proof-owner-handoff-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  handoffState = "blocked-owner-real-proof-required"
  prePublishAuditMatrixReady = $prePublishAuditMatrixReady
  releaseEvidenceBundleReady = $releaseEvidenceBundleReady
  finalDryRunReady = $finalDryRunReady
  closeBlockerDashboardReady = $closeBlockerDashboardReady
  ownerInputSchemaReady = $ownerInputSchemaReady
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputCanPromoteRuntimeProof = $ownerInputCanPromoteRuntimeProof
  forbiddenSubstituteScanState = $forbiddenSubstituteScanState
  detectedForbiddenSubstituteCount = $detectedForbiddenSubstituteCount
  packageConsumerRuntimeProofRecordReady = $packageConsumerRuntimeProofRecordReady
  externalRuntimeProofReady = $externalRuntimeProofReady
  postPublishVerificationReady = $postPublishVerificationReady
  publicPublishOwnerCommandReady = $publicPublishOwnerCommandReady
  tensorRtExecNonProofBoundaryReady = $tensorRtExecNonProofBoundaryReady
  yoloVisionRealModelBackfillReady = $yoloVisionRealModelBackfillReady
  onnxToEngineBuildReportBoundaryReady = $onnxToEngineBuildReportBoundaryReady
  packageConsumerOwnerRuntimeSmokeFieldAlignmentState = $ownerRuntimeSmokeFieldAlignmentState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState = $ownerRuntimeSmokeFieldAlignmentValidationState
  packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus = $ownerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount = $ownerRuntimeSmokeFieldAlignmentFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount = $ownerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount
  packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount = $ownerRuntimeSmokeFieldAlignmentFailedBlockerCount
  remainingOwnerActionCount = $remainingOwnerActionCount
  requiredOwnerInputs = @($requiredOwnerInputs)
  requiredCommands = @($requiredCommands)
  requiredValidators = @($requiredValidators)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  sourceArtifacts = @($sourceArtifacts)
  performsPublish = $false
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  boundary = "Final proof owner handoff pack is an owner execution map only; it does not publish packages, approve release, close an issue, promote runtime proof, or replace clean external package-consumer-runtime plus post-publish verification."
}

$jsonPath = Join-Path $OutputRoot "final-proof-owner-handoff-pack.json"
$markdownPath = Join-Path $OutputRoot "final-proof-owner-handoff-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$inputRows = $requiredOwnerInputs | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.field)`` | $(ConvertTo-MarkdownCell $_.requiredEvidence) | ``$($_.validator)`` |"
}
$commandRows = $requiredCommands | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.command)`` | $(ConvertTo-MarkdownCell $_.purpose) | $(ConvertTo-MarkdownCell $_.boundary) |"
}
$forbiddenRows = $forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# Final Proof Owner Handoff Pack

| Field | Value |
| --- | --- |
| handoffState | ``$($record.handoffState)`` |
| prePublishAuditMatrixReady | ``$($record.prePublishAuditMatrixReady)`` |
| ownerInputSchemaReady | ``$($record.ownerInputSchemaReady)`` |
| cleanOwnerInputReady | ``$($record.cleanOwnerInputReady)`` |
| ownerInputCanPromoteRuntimeProof | ``$($record.ownerInputCanPromoteRuntimeProof)`` |
| forbiddenSubstituteScanState | ``$($record.forbiddenSubstituteScanState)`` |
| detectedForbiddenSubstituteCount | ``$($record.detectedForbiddenSubstituteCount)`` |
| packageConsumerRuntimeProofRecordReady | ``$($record.packageConsumerRuntimeProofRecordReady)`` |
| postPublishVerificationReady | ``$($record.postPublishVerificationReady)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentValidationState)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentRuntimeSmokeStatus)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentMissingRequiredFieldCount)`` |
| packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount | ``$($record.packageConsumerOwnerRuntimeSmokeFieldAlignmentFailedBlockerCount)`` |
| remainingOwnerActionCount | ``$($record.remainingOwnerActionCount)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |

## Required Owner Inputs

| ID | Field | Required Evidence | Validator |
| --- | --- | --- | --- |
$($inputRows -join "`r`n")

## Required Commands

| ID | Command | Purpose | Boundary |
| --- | --- | --- | --- |
$($commandRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Final proof owner handoff pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "HandoffState=$($record.handoffState) RemainingOwnerActionCount=$remainingOwnerActionCount CanPublish=$($record.canPublishPublicly) CanClose=$($record.canCloseReleaseIssue) CanPromote=$($record.canPromoteRuntimeProof)"
