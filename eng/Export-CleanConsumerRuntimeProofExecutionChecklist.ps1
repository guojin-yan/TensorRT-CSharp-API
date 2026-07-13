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

$packageConsumerStrongValidator = "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"

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

function New-ExecutionStep {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$CommandTemplate,
    [string]$RequiredOutput,
    [string]$ProofBoundary
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    commandTemplate = $CommandTemplate
    requiredOutput = $RequiredOutput
    ownerActionRequired = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    proofBoundary = $ProofBoundary
  }
}

function New-RequiredItem {
  param([string]$Id, [string]$Description, [string]$Validator)

  [pscustomobject]@{
    id = $Id
    description = $Description
    validator = $Validator
    ownerActionRequired = $true
  }
}

$releaseEvidence = Read-JsonOrNull "artifacts/final-release/release-evidence-bundle.json"
$handoffPack = Read-JsonOrNull "artifacts/final-release/final-proof-owner-handoff-pack.json"
$handoffValidation = Read-JsonOrNull "artifacts/final-release/final-proof-owner-handoff-pack-validation.json"
$prePublish = Read-JsonOrNull "artifacts/final-release/final-release-pre-publish-audit-matrix.json"
$prePublishValidation = Read-JsonOrNull "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json"
$tensorRtExecParity = Read-JsonOrNull "artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist.json"
$tensorRtExecParityValidation = Read-JsonOrNull "artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist-validation.json"

$sourceArtifacts = @(
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-proof-owner-handoff-pack.json",
  "artifacts/final-release/final-proof-owner-handoff-pack-validation.json",
  "artifacts/final-release/final-release-pre-publish-audit-matrix.json",
  "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json",
  "artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist.json",
  "artifacts/final-release/tensor-rt-exec-gui-cli-parity-checklist-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json",
  "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json",
  "artifacts/final-release/final-release-dry-run-summary.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json"
)

$executionSteps = @(
  New-ExecutionStep -Order 1 -Id "create-repository-external-clean-consumer" -Title "Create repository-external clean consumer" -CommandTemplate 'mkdir C:\trtsharp-clean-consumer; cd C:\trtsharp-clean-consumer; dotnet new console --framework net8.0' -RequiredOutput "Clean consumer project path outside the repository plus project file hash." -ProofBoundary "Creating a project is setup only and not runtime proof."
  New-ExecutionStep -Order 2 -Id "configure-public-package-source" -Title "Configure public package source" -CommandTemplate 'dotnet nuget add source <public-package-source-url> --name trtsharp-public-proof-source' -RequiredOutput "Public package source URL and source name used by the external consumer." -ProofBoundary "Package source configuration is owner input; local feed and direct .nupkg are forbidden substitutes."
  New-ExecutionStep -Order 3 -Id "install-managed-and-runtime-packages" -Title "Install managed and runtime packages" -CommandTemplate 'dotnet add package JYPPX.TensorRtSharp --version <version>; dotnet add package JYPPX.TensorRtSharp.Native.<runtime-key> --version <version>' -RequiredOutput "Managed package id/version and runtime package id/version resolved from public source." -ProofBoundary "Install metadata is required, but install alone is not runtime proof."
  New-ExecutionStep -Order 4 -Id "restore-clean-consumer" -Title "Restore clean consumer" -CommandTemplate 'dotnet restore --no-cache --force-evaluate *> package-consumer-restore.log' -RequiredOutput "Restore log path and restore exit code 0." -ProofBoundary "Restore output is not runtime proof and cannot replace smoke execution."
  New-ExecutionStep -Order 5 -Id "build-clean-consumer" -Title "Build clean consumer" -CommandTemplate 'dotnet build -c Release --no-restore *> package-consumer-build.log' -RequiredOutput "Build log path and build exit code 0." -ProofBoundary "Build-only output is explicitly non-proof."
  New-ExecutionStep -Order 6 -Id "run-smoke-runtime-command" -Title "Run smoke runtime command" -CommandTemplate 'dotnet run -c Release --no-build -- --runtime-smoke *> package-consumer-smoke.stdout.log 2> package-consumer-smoke.stderr.log' -RequiredOutput "Runtime smoke stdout/stderr logs and exit code 0 on a compatible CUDA/TensorRT host." -ProofBoundary "Only this runtime command can contribute to package-consumer-runtime proof after strict validation."
  New-ExecutionStep -Order 7 -Id "persist-stdout-stderr-logs" -Title "Persist stdout/stderr logs" -CommandTemplate 'Copy-Item package-consumer-*.log <owner-proof-log-directory>' -RequiredOutput "Stable stdout/stderr/restore/build log paths in owner proof storage." -ProofBoundary "Copied logs require SHA256 and validator linkage before proof promotion."
  New-ExecutionStep -Order 8 -Id "compute-log-sha256" -Title "Compute log SHA256" -CommandTemplate 'Get-FileHash -Algorithm SHA256 package-consumer-smoke.stdout.log,package-consumer-smoke.stderr.log' -RequiredOutput "SHA256 for stdout and stderr logs." -ProofBoundary "Hash values authenticate logs; they are not proof by themselves."
  New-ExecutionStep -Order 9 -Id "compute-nupkg-sha256" -Title "Compute managed/runtime nupkg SHA256" -CommandTemplate 'Get-FileHash -Algorithm SHA256 <managed-nupkg-path>,<runtime-nupkg-path>' -RequiredOutput "SHA256 for exact managed and runtime packages consumed by the clean project." -ProofBoundary "Package hashes must match the public package source record."
  New-ExecutionStep -Order 10 -Id "fill-owner-input" -Title "Fill owner input" -CommandTemplate 'Copy-Item artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json <owner-input.json>; notepad <owner-input.json>' -RequiredOutput "Owner-filled input with package, host, command, log, hash, and review metadata." -ProofBoundary "Owner input is not proof until imported and strictly validated."
  New-ExecutionStep -Order 11 -Id "import-owner-input" -Title "Import owner input" -CommandTemplate 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 -InputPath <owner-input.json> -Strict' -RequiredOutput "Imported package consumer runtime proof candidate and import report." -ProofBoundary "Import cannot promote proof while required runtime fields are incomplete."
  New-ExecutionStep -Order 12 -Id "strict-validate-proof-record" -Title "Strict validate package consumer runtime proof record" -CommandTemplate 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof' -RequiredOutput "Validator passes with proofClassification=package-consumer-runtime and smokeStatus=passed." -ProofBoundary "This is the minimum promotion gate for package-consumer-runtime proof."
  New-ExecutionStep -Order 13 -Id "run-forbidden-substitute-scan" -Title "Run forbidden substitute scan" -CommandTemplate 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1' -RequiredOutput "Forbidden substitute scan with detectedForbiddenSubstituteCount=0." -ProofBoundary "Scan output is an audit control and not runtime proof."
  New-ExecutionStep -Order 14 -Id "refresh-release-evidence-and-dashboard" -Title "Refresh release evidence, dry run, and close blocker dashboard" -CommandTemplate 'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleaseCloseBlockerDashboard.ps1' -RequiredOutput "Refreshed final-release artifacts showing proof status after owner input validation." -ProofBoundary "Refresh commands do not publish, close release, or replace strict runtime proof."
)

$requiredInputs = @(
  New-RequiredItem -Id "public-package-source-url" -Description "Official public package source URL consumed by the clean external project." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "managed-package-id-version" -Description "Managed package id and version resolved from the public source." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "runtime-package-id-version" -Description "Runtime package id, version, and runtime key consumed by the external project." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "repository-external-consumer-path" -Description "Clean consumer project path outside the source repository." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "restore-build-smoke-commands" -Description "Exact restore, build, and runtime smoke commands that were executed." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "smoke-exit-code-zero" -Description "Runtime smoke command exit code 0." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "startedAtUtc" -Description "UTC timestamp captured immediately before the runtime smoke command started." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "finishedAtUtc" -Description "UTC timestamp captured immediately after the runtime smoke command finished." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "owner-review-identity" -Description "Owner name, review timestamp, and review note." -Validator "Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict"
)

$requiredHashes = @(
  New-RequiredItem -Id "clean-consumer-project-hash" -Description "Hash of the clean consumer project file or lock file." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "managed-nupkg-sha256" -Description "SHA256 of the exact managed package consumed." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "runtime-nupkg-sha256" -Description "SHA256 of the exact runtime package consumed." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "restore-log-sha256" -Description "SHA256 for restore log." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "build-log-sha256" -Description "SHA256 for build log." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "stdout-log-sha256" -Description "SHA256 for runtime smoke stdout log." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "stderr-log-sha256" -Description "SHA256 for runtime smoke stderr log." -Validator $packageConsumerStrongValidator
)

$requiredLogs = @(
  New-RequiredItem -Id "restore-log" -Description "dotnet restore stdout/stderr log." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "build-log" -Description "dotnet build stdout/stderr log." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "smoke-stdout-log" -Description "Runtime smoke stdout log." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "smoke-stderr-log" -Description "Runtime smoke stderr log." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "owner-import-log" -Description "Owner input import log." -Validator "Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict"
  New-RequiredItem -Id "strict-validator-log" -Description "Strict proof validator log." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
)

$requiredHostMetadata = @(
  New-RequiredItem -Id "host-os-architecture" -Description "Operating system, architecture, and runtime identifier." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "gpu-name" -Description "NVIDIA GPU name used by the proof host." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "nvidia-driver-version" -Description "NVIDIA driver version." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "cuda-version" -Description "CUDA runtime/toolkit version observed by the clean consumer." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "tensorrt-version" -Description "TensorRT version observed by the clean consumer." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  New-RequiredItem -Id "cudnn-version" -Description "cuDNN version when applicable to the runtime key." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
)

$requiredPackageMetadata = @(
  New-RequiredItem -Id "managed-package-id" -Description "Managed package id." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "managed-package-version" -Description "Managed package version." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "runtime-package-id" -Description "Runtime package id." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "runtime-package-version" -Description "Runtime package version." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "runtime-package-key" -Description "Runtime package key, for example win-x64-trt11.0-cuda13.2-cudnn9.22." -Validator $packageConsumerStrongValidator
  New-RequiredItem -Id "public-package-source" -Description "Public source URL that served both packages." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1"
)

$requiredValidators = @(
  "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
  "eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
  "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
  "eng/Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1",
  "eng/Export-ReleaseEvidenceBundle.ps1",
  "eng/Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked",
  "eng/Export-FinalReleaseCloseBlockerDashboard.ps1",
  "eng/Export-FinalReleasePrePublishAuditMatrix.ps1",
  "eng/Test-FinalReleasePrePublishAuditMatrix.ps1 -Strict"
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
  "readonly diagnostics",
  "dependency probe",
  "owner input without strict validator pass"
)

$record = [pscustomobject]@{
  recordKind = "clean-consumer-runtime-proof-execution-checklist"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  checklistState = "blocked-owner-clean-consumer-runtime-proof-required"
  sourceArtifacts = @($sourceArtifacts)
  sourceStates = [pscustomobject]@{
    releaseEvidenceBundlePresent = $null -ne $releaseEvidence
    finalProofOwnerHandoffPackPresent = $null -ne $handoffPack
    finalProofOwnerHandoffPackReady = [string](Get-PropertyOrDefault -Object $handoffValidation -Name "validationState" -DefaultValue "") -eq "final-proof-owner-handoff-pack-ready"
    finalReleasePrePublishAuditReady = [string](Get-PropertyOrDefault -Object $prePublishValidation -Name "validationState" -DefaultValue "") -eq "final-release-pre-publish-audit-matrix-ready"
    tensorRtExecGuiCliParityReady = [string](Get-PropertyOrDefault -Object $tensorRtExecParityValidation -Name "validationState" -DefaultValue "") -eq "tensor-rt-exec-gui-cli-parity-checklist-ready"
    tensorRtExecRuntimeProofItems = [int](Get-PropertyOrDefault -Object $tensorRtExecParity -Name "runtimeProofItems" -DefaultValue 0)
    prePublishRuntimeProofItems = [int](Get-PropertyOrDefault -Object $prePublish -Name "tensorRtExecRuntimeProofItems" -DefaultValue 0)
  }
  executionStepCount = $executionSteps.Count
  executionSteps = @($executionSteps)
  requiredInputs = @($requiredInputs)
  requiredHashes = @($requiredHashes)
  requiredLogs = @($requiredLogs)
  requiredHostMetadata = @($requiredHostMetadata)
  requiredPackageMetadata = @($requiredPackageMetadata)
  requiredValidators = @($requiredValidators)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  boundary = "This checklist is owner execution guidance only. It does not publish packages, close the release issue, promote runtime proof, or replace a real clean external package-consumer-runtime smoke with logs, hashes, host metadata, package metadata, owner review, and strict validator pass."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-runtime-proof-execution-checklist.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-runtime-proof-execution-checklist.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$stepRows = $executionSteps | ForEach-Object {
  "| $($_.order) | ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.title) | ``$($_.commandTemplate)`` | $(ConvertTo-MarkdownCell $_.requiredOutput) |"
}
$inputRows = $requiredInputs | ForEach-Object { "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.description) | ``$($_.validator)`` |" }
$hashRows = $requiredHashes | ForEach-Object { "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.description) | ``$($_.validator)`` |" }
$logRows = $requiredLogs | ForEach-Object { "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.description) | ``$($_.validator)`` |" }
$hostRows = $requiredHostMetadata | ForEach-Object { "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.description) | ``$($_.validator)`` |" }
$packageRows = $requiredPackageMetadata | ForEach-Object { "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.description) | ``$($_.validator)`` |" }
$validatorRows = $requiredValidators | ForEach-Object { "- ``$_``" }
$forbiddenRows = $forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# Clean Consumer Runtime Proof Execution Checklist

| Field | Value |
| --- | --- |
| recordKind | ``$($record.recordKind)`` |
| checklistState | ``$($record.checklistState)`` |
| executionStepCount | ``$($record.executionStepCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |
| isPackageConsumerRuntimeProof | ``$($record.isPackageConsumerRuntimeProof)`` |

## Execution Steps

| # | ID | Title | Command Template | Required Output |
|---:|---|---|---|---|
$($stepRows -join "`r`n")

## Required Inputs

| ID | Description | Validator |
|---|---|---|
$($inputRows -join "`r`n")

## Required Hashes

| ID | Description | Validator |
|---|---|---|
$($hashRows -join "`r`n")

## Required Logs

| ID | Description | Validator |
|---|---|---|
$($logRows -join "`r`n")

## Required Host Metadata

| ID | Description | Validator |
|---|---|---|
$($hostRows -join "`r`n")

## Required Package Metadata

| ID | Description | Validator |
|---|---|---|
$($packageRows -join "`r`n")

## Required Validators

$($validatorRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer runtime proof execution checklist written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ChecklistState=$($record.checklistState) Steps=$($record.executionStepCount) CanPublish=$($record.canPublishPublicly) CanClose=$($record.canCloseReleaseIssue) CanPromote=$($record.canPromoteRuntimeProof)"
