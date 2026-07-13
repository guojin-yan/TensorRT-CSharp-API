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

function New-CommandGroup {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string[]]$Commands,
    [string[]]$RequiredOutputs,
    [string]$ProofBoundary
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    commands = @($Commands)
    requiredOutputs = @($RequiredOutputs)
    ownerActionRequired = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    proofBoundary = $ProofBoundary
  }
}

function New-RequiredFile {
  param([string]$Id, [string]$PathPattern, [string]$Purpose, [string]$Sha256Required)

  [pscustomobject]@{
    id = $Id
    pathPattern = $PathPattern
    purpose = $Purpose
    sha256Required = $Sha256Required
    ownerActionRequired = $true
  }
}

function New-RequiredInput {
  param([string]$Id, [string]$FieldPath, [string]$RequiredEvidence, [string]$Validator)

  [pscustomobject]@{
    id = $Id
    fieldPath = $FieldPath
    requiredEvidence = $RequiredEvidence
    validator = $Validator
    ownerActionRequired = $true
  }
}

function New-ProjectionItem {
  param([string]$Id, [string]$SourceInput, [string]$ProjectedField, [string]$PromotionRule)

  [pscustomobject]@{
    id = $Id
    sourceInput = $SourceInput
    projectedField = $ProjectedField
    promotionRule = $PromotionRule
    canPromoteRuntimeProof = $false
  }
}

$checklist = Read-JsonOrNull "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json"
$checklistValidation = Read-JsonOrNull "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist-validation.json"
$handoff = Read-JsonOrNull "artifacts/final-release/final-proof-owner-handoff-pack.json"
$handoffValidation = Read-JsonOrNull "artifacts/final-release/final-proof-owner-handoff-pack-validation.json"
$releaseEvidence = Read-JsonOrNull "artifacts/final-release/release-evidence-bundle.json"
$closeDashboard = Read-JsonOrNull "artifacts/final-release/final-release-close-blocker-dashboard.json"
$prePublishValidation = Read-JsonOrNull "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json"
$technicalCampaignValidation = Read-JsonOrNull "artifacts/final-release/technical-article-campaign-matrix-validation.json"

$checklistReady = [string](Get-PropertyOrDefault -Object $checklistValidation -Name "validationState" -DefaultValue "") -eq "clean-consumer-runtime-proof-execution-checklist-ready"
$handoffReady = [string](Get-PropertyOrDefault -Object $handoffValidation -Name "validationState" -DefaultValue "") -eq "final-proof-owner-handoff-pack-ready"
$prePublishReady = [string](Get-PropertyOrDefault -Object $prePublishValidation -Name "validationState" -DefaultValue "") -eq "final-release-pre-publish-audit-matrix-ready"
$technicalCampaignReady = [string](Get-PropertyOrDefault -Object $technicalCampaignValidation -Name "validationState" -DefaultValue "") -eq "technical-article-campaign-matrix-ready"
$closeDashboardState = [string](Get-PropertyOrDefault -Object $closeDashboard -Name "dashboardState" -DefaultValue "missing-final-release-close-blocker-dashboard")
$cleanOwnerInputReady = [bool](Get-PropertyOrDefault -Object $closeDashboard -Name "cleanOwnerInputReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $handoff -Name "cleanOwnerInputReady" -DefaultValue $false)))
$ownerInputCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $closeDashboard -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $handoff -Name "ownerInputCanPromoteRuntimeProof" -DefaultValue $false)))
$forbiddenSubstituteScanState = [string](Get-PropertyOrDefault -Object $closeDashboard -Name "forbiddenSubstituteScanState" -DefaultValue ([string](Get-PropertyOrDefault -Object $handoff -Name "forbiddenSubstituteScanState" -DefaultValue "missing-forbidden-substitute-scan")))
$detectedForbiddenSubstituteCount = [int](Get-PropertyOrDefault -Object $closeDashboard -Name "detectedForbiddenSubstituteCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $handoff -Name "detectedForbiddenSubstituteCount" -DefaultValue 0)))

$sourceArtifacts = @(
  "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json",
  "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist-validation.json",
  "artifacts/final-release/final-proof-owner-handoff-pack.json",
  "artifacts/final-release/final-proof-owner-handoff-pack-validation.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/final-release-close-blocker-dashboard.json",
  "artifacts/final-release/final-release-pre-publish-audit-matrix-validation.json",
  "artifacts/final-release/technical-article-campaign-matrix-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
  "artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json",
  "artifacts/final-release/package-consumer-runtime-proof-record-validation.json",
  "artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json"
)

$ownerCommandGroups = @(
  New-CommandGroup -Order 1 -Id "prepare-clean-consumer-workspace" -Title "Prepare repository-external clean consumer workspace" -Commands @(
    'New-Item -ItemType Directory -Force C:\trtsharp-clean-consumer | Out-Null',
    'Set-Location C:\trtsharp-clean-consumer',
    'dotnet new console --framework net8.0'
  ) -RequiredOutputs @(
    "Repository-external project path",
    "Clean consumer project file hash",
    "Confirmation that no ProjectReference points back to the source repository"
  ) -ProofBoundary "Workspace creation is setup only; it cannot prove runtime execution."
  New-CommandGroup -Order 2 -Id "configure-public-source-and-install-packages" -Title "Configure public package source and install exact packages" -Commands @(
    'dotnet nuget add source <public-package-source-url> --name trtsharp-public-proof-source',
    'dotnet add package JYPPX.TensorRtSharp --version <managed-version> --source <public-package-source-url>',
    'dotnet add package JYPPX.TensorRtSharp.Native.<runtime-key> --version <runtime-version> --source <public-package-source-url>'
  ) -RequiredOutputs @(
    "Public package source URL",
    "Managed package id/version",
    "Runtime package id/version/runtime key",
    "No local feed, no direct .nupkg, no ProjectReference"
  ) -ProofBoundary "Package restore metadata is required owner input but is not proof until smoke and strict validation pass."
  New-CommandGroup -Order 3 -Id "restore-build-smoke-and-capture-logs" -Title "Restore, build, run smoke, and capture logs" -Commands @(
    'dotnet restore --no-cache --force-evaluate *> package-consumer-restore.log',
    'dotnet build -c Release --no-restore *> package-consumer-build.log',
    'dotnet run -c Release --no-build -- --runtime-smoke *> package-consumer-smoke.stdout.log 2> package-consumer-smoke.stderr.log'
  ) -RequiredOutputs @(
    "restore log and exit code",
    "build log and exit code",
    "runtime smoke stdout/stderr logs",
    "runtime smoke exit code 0 on compatible CUDA/TensorRT host"
  ) -ProofBoundary "Restore/build are non-proof. Runtime smoke can promote only after owner input import and strict validator success."
  New-CommandGroup -Order 4 -Id "hash-logs-packages-and-project" -Title "Hash logs, packages, and clean consumer project" -Commands @(
    'Get-FileHash -Algorithm SHA256 package-consumer-restore.log,package-consumer-build.log,package-consumer-smoke.stdout.log,package-consumer-smoke.stderr.log',
    'Get-FileHash -Algorithm SHA256 <managed-nupkg-path>,<runtime-nupkg-path>',
    'Get-FileHash -Algorithm SHA256 .\*.csproj'
  ) -RequiredOutputs @(
    "restore/build/smoke log SHA256",
    "managed/runtime nupkg SHA256",
    "clean consumer project hash"
  ) -ProofBoundary "Hashes authenticate artifacts and must match owner input; hashes alone are not runtime proof."
  New-CommandGroup -Order 5 -Id "capture-host-package-and-owner-metadata" -Title "Capture host, package, and owner metadata" -Commands @(
    'nvidia-smi',
    'dotnet --info',
    'Write-Output "<owner-name> <machine-name> <reviewed-at-utc> <runtime-key>"'
  ) -RequiredOutputs @(
    "OS and architecture",
    "GPU name and NVIDIA driver",
    "CUDA runtime/toolkit, TensorRT, cuDNN",
    "Owner identity, machine name, review timestamp"
  ) -ProofBoundary "Host metadata is required for traceability but cannot substitute runtime smoke proof."
  New-CommandGroup -Order 6 -Id "fill-import-and-validate-owner-input" -Title "Fill owner input, import candidate, and run strict validators" -Commands @(
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 -InputPath <owner-input.json> -Strict',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof'
  ) -RequiredOutputs @(
    "Owner input import report",
    "Package consumer runtime proof record validation",
    "ProofClassification=package-consumer-runtime",
    "SmokeStatus=passed"
  ) -ProofBoundary "Only strict validator success with existing logs and matching hashes can promote package-consumer-runtime proof."
  New-CommandGroup -Order 7 -Id "scan-forbidden-substitutes-and-refresh-release-state" -Title "Scan forbidden substitutes and refresh release state" -Commands @(
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleaseCloseBlockerDashboard.ps1'
  ) -RequiredOutputs @(
    "Forbidden substitute scan with no detected substitutes",
    "Refreshed release evidence bundle",
    "Refreshed dry run summary",
    "Refreshed close blocker dashboard"
  ) -ProofBoundary "Release refresh commands aggregate state only and do not publish, close, or replace runtime proof."
  New-CommandGroup -Order 8 -Id "post-publish-remains-separate-owner-gate" -Title "Keep post-publish verification as separate owner gate" -Commands @(
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady'
  ) -RequiredOutputs @(
    "Post-publish public package install log after real owner publish",
    "Strict release close validator result only after all real proof gates pass"
  ) -ProofBoundary "This pack cannot execute publish or post-publish verification; those remain owner-only gates."
)

$requiredOwnerFiles = @(
  New-RequiredFile -Id "clean-consumer-project" -PathPattern "C:\trtsharp-clean-consumer\*.csproj" -Purpose "Repository-external consumer identity." -Sha256Required "true"
  New-RequiredFile -Id "restore-log" -PathPattern "package-consumer-restore.log" -Purpose "Restore evidence for public package source." -Sha256Required "true"
  New-RequiredFile -Id "build-log" -PathPattern "package-consumer-build.log" -Purpose "Build evidence for external consumer." -Sha256Required "true"
  New-RequiredFile -Id "smoke-stdout-log" -PathPattern "package-consumer-smoke.stdout.log" -Purpose "Runtime smoke stdout evidence." -Sha256Required "true"
  New-RequiredFile -Id "smoke-stderr-log" -PathPattern "package-consumer-smoke.stderr.log" -Purpose "Runtime smoke stderr evidence." -Sha256Required "true"
  New-RequiredFile -Id "managed-nupkg" -PathPattern "<nuget-global-packages>\JYPPX.TensorRtSharp\<version>\*.nupkg" -Purpose "Exact managed package consumed from public source." -Sha256Required "true"
  New-RequiredFile -Id "runtime-nupkg" -PathPattern "<nuget-global-packages>\JYPPX.TensorRtSharp.Native.<runtime-key>\<version>\*.nupkg" -Purpose "Exact runtime package consumed from public source." -Sha256Required "true"
  New-RequiredFile -Id "owner-input" -PathPattern "artifacts\final-release\package-consumer-runtime-proof-owner-input.json" -Purpose "Owner-filled runtime proof input." -Sha256Required "false"
)

$requiredOwnerInputs = @(
  New-RequiredInput -Id "public-package-source-url" -FieldPath "packageSource.publicPackageSourceUrl" -RequiredEvidence "Public source URL used by restore/install." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict"
  New-RequiredInput -Id "package-ids-and-versions" -FieldPath "packageSource.packageId/packageVersion/runtimePackageKey/runtimePackageVersion" -RequiredEvidence "Exact managed and runtime package identities consumed." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-RequiredInput -Id "consumer-project-identity" -FieldPath "consumer.projectPath/consumer.projectSha256" -RequiredEvidence "Repository-external clean consumer path and project hash." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict"
  New-RequiredInput -Id "commands" -FieldPath "commands.restoreCommand/buildCommand/smokeCommand" -RequiredEvidence "Exact restore/build/smoke commands executed." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-RequiredInput -Id "logs-and-hashes" -FieldPath "results.stdoutLogPath/stderrLogPath/logSha256/packageSha256" -RequiredEvidence "Existing logs and matching SHA256 values." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-RequiredInput -Id "host-metadata" -FieldPath "host.os/architecture/gpuName/driverVersion/cudaVersion/tensorRtVersion/cudnnVersion" -RequiredEvidence "Compatible host metadata." -Validator "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict"
  New-RequiredInput -Id "runtime-status" -FieldPath "results.exitCode/results.smokeStatus/results.nativeAssetsCopied" -RequiredEvidence "Exit code 0, smokeStatus=passed, native assets copied." -Validator "Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  New-RequiredInput -Id "owner-review" -FieldPath "owner.name/machineName/reviewedAtUtc/reviewNote" -RequiredEvidence "Owner review identity and timestamp." -Validator "Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict"
)

$proofRecordProjection = @(
  New-ProjectionItem -Id "classification" -SourceInput "strict proof validator" -ProjectedField "proofClassification=package-consumer-runtime" -PromotionRule "Must be exactly package-consumer-runtime."
  New-ProjectionItem -Id "runtime-execution" -SourceInput "smoke command" -ProjectedField "isRuntimeExecutionProof=true only after validator pass" -PromotionRule "Cannot be set by checklist, template, report, or owner input alone."
  New-ProjectionItem -Id "package-consumer" -SourceInput "clean external consumer" -ProjectedField "isPackageConsumerRuntimeProof=true only after validator pass" -PromotionRule "Requires no local feed, no ProjectReference, no direct .nupkg, public source, matching hashes."
  New-ProjectionItem -Id "publishability" -SourceInput "release evidence bundle" -ProjectedField "canPublishPublicly remains false" -PromotionRule "Public publish is separate owner manual gate."
  New-ProjectionItem -Id "release-close" -SourceInput "strict close validator" -ProjectedField "canCloseReleaseIssue remains false" -PromotionRule "Requires post-publish proof and Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady."
)

$strictValidatorCommands = @(
  "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
  "eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
  "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
  "eng/Export-PackageConsumerRuntimeProofForbiddenSubstituteScan.ps1",
  "eng/Export-ReleaseEvidenceBundle.ps1",
  "eng/Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked",
  "eng/Export-FinalReleaseCloseBlockerDashboard.ps1",
  "eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof",
  "eng/Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
)

$forbiddenSubstituteRules = @(
  "template is not proof",
  "local feed is not clean public package source proof",
  "ProjectReference is not package-consumer-runtime proof",
  "direct .nupkg is not public package proof",
  "dry-run is not runtime proof",
  "build-only is not runtime proof",
  "preflight-only is not runtime proof",
  "GUI screenshot is not runtime proof",
  "TensorRtExec report is not package-consumer-runtime proof",
  "YoloVision matrix is not package-consumer-runtime proof",
  "OnnxToEngine report is not package-consumer-runtime proof",
  "sidecar-only report is not runtime proof",
  "readonly diagnostics are not runtime proof",
  "owner input without strict validator pass is not proof"
)

$releaseRefreshCommands = @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleaseCloseBlockerDashboard.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalReleasePrePublishAuditMatrix.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleasePrePublishAuditMatrix.ps1 -Strict"
)

$ownerHandoffSummary = [pscustomobject]@{
  checklistReady = $checklistReady
  finalProofOwnerHandoffPackReady = $handoffReady
  finalReleasePrePublishAuditReady = $prePublishReady
  technicalArticleCampaignReady = $technicalCampaignReady
  closeDashboardState = $closeDashboardState
  cleanOwnerInputReady = $cleanOwnerInputReady
  ownerInputCanPromoteRuntimeProof = $ownerInputCanPromoteRuntimeProof
  forbiddenSubstituteScanState = $forbiddenSubstituteScanState
  detectedForbiddenSubstituteCount = $detectedForbiddenSubstituteCount
  remainingOwnerActionCount = [int](Get-PropertyOrDefault -Object $handoff -Name "remainingOwnerActionCount" -DefaultValue 0)
  nextOwnerAction = "Run real repository-external clean consumer smoke from public package source, backfill owner input with logs/hashes/host/package metadata, import, run strict proof validator, run forbidden substitute scan, then refresh release evidence and close dashboard."
}

$record = [pscustomobject]@{
  recordKind = "clean-consumer-proof-owner-execution-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packState = "blocked-owner-compatible-host-runtime-proof-required"
  sourceChecklist = "artifacts/final-release/clean-consumer-runtime-proof-execution-checklist.json"
  sourceArtifacts = @($sourceArtifacts)
  ownerCommandGroupCount = $ownerCommandGroups.Count
  ownerCommandGroups = @($ownerCommandGroups)
  requiredOwnerFiles = @($requiredOwnerFiles)
  requiredOwnerInputs = @($requiredOwnerInputs)
  proofRecordProjection = @($proofRecordProjection)
  strictValidatorCommands = @($strictValidatorCommands)
  forbiddenSubstituteRules = @($forbiddenSubstituteRules)
  releaseRefreshCommands = @($releaseRefreshCommands)
  ownerHandoffSummary = $ownerHandoffSummary
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  boundary = "This owner execution pack compresses the real clean consumer proof handoff only. It does not execute package publish, does not run post-publish verification, does not close the release issue, and cannot promote package-consumer-runtime proof until a repository-external clean consumer smoke from a public package source passes strict validation with existing logs, hashes, host metadata, package metadata, and no forbidden substitutes."
}

$jsonPath = Join-Path $OutputRoot "clean-consumer-proof-owner-execution-pack.json"
$markdownPath = Join-Path $OutputRoot "clean-consumer-proof-owner-execution-pack.md"
$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$commandRows = $ownerCommandGroups | ForEach-Object {
  "| $($_.order) | ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.title) | $(ConvertTo-MarkdownCell ($_.commands -join "<br>")) | $(ConvertTo-MarkdownCell ($_.requiredOutputs -join "<br>")) | $(ConvertTo-MarkdownCell $_.proofBoundary) |"
}
$fileRows = $requiredOwnerFiles | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.pathPattern)`` | $(ConvertTo-MarkdownCell $_.purpose) | ``$($_.sha256Required)`` |"
}
$inputRows = $requiredOwnerInputs | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.fieldPath)`` | $(ConvertTo-MarkdownCell $_.requiredEvidence) | ``$($_.validator)`` |"
}
$projectionRows = $proofRecordProjection | ForEach-Object {
  "| ``$($_.id)`` | $(ConvertTo-MarkdownCell $_.sourceInput) | ``$($_.projectedField)`` | $(ConvertTo-MarkdownCell $_.promotionRule) |"
}
$validatorRows = $strictValidatorCommands | ForEach-Object { "- ``$_``" }
$forbiddenRows = $forbiddenSubstituteRules | ForEach-Object { "- $_" }

$markdown = @"
# Clean Consumer Proof Owner Execution Pack

| Field | Value |
| --- | --- |
| recordKind | ``$($record.recordKind)`` |
| packState | ``$($record.packState)`` |
| sourceChecklist | ``$($record.sourceChecklist)`` |
| ownerCommandGroupCount | ``$($record.ownerCommandGroupCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| isRuntimeExecutionProof | ``$($record.isRuntimeExecutionProof)`` |
| isPackageConsumerRuntimeProof | ``$($record.isPackageConsumerRuntimeProof)`` |

## Owner Handoff Summary

- checklistReady: ``$($ownerHandoffSummary.checklistReady)``
- finalProofOwnerHandoffPackReady: ``$($ownerHandoffSummary.finalProofOwnerHandoffPackReady)``
- closeDashboardState: ``$($ownerHandoffSummary.closeDashboardState)``
- cleanOwnerInputReady: ``$($ownerHandoffSummary.cleanOwnerInputReady)``
- ownerInputCanPromoteRuntimeProof: ``$($ownerHandoffSummary.ownerInputCanPromoteRuntimeProof)``
- forbiddenSubstituteScanState: ``$($ownerHandoffSummary.forbiddenSubstituteScanState)``
- detectedForbiddenSubstituteCount: ``$($ownerHandoffSummary.detectedForbiddenSubstituteCount)``
- remainingOwnerActionCount: ``$($ownerHandoffSummary.remainingOwnerActionCount)``
- nextOwnerAction: $($ownerHandoffSummary.nextOwnerAction)

## Owner Command Groups

| # | ID | Title | Commands | Required Outputs | Proof Boundary |
|---:|---|---|---|---|---|
$($commandRows -join "`r`n")

## Required Owner Files

| ID | Path Pattern | Purpose | SHA256 Required |
|---|---|---|---|
$($fileRows -join "`r`n")

## Required Owner Inputs

| ID | Field Path | Required Evidence | Validator |
|---|---|---|---|
$($inputRows -join "`r`n")

## Proof Record Projection

| ID | Source Input | Projected Field | Promotion Rule |
|---|---|---|---|
$($projectionRows -join "`r`n")

## Strict Validators

$($validatorRows -join "`r`n")

## Forbidden Substitute Rules

$($forbiddenRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean consumer proof owner execution pack written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "PackState=$($record.packState) CommandGroups=$($record.ownerCommandGroupCount) CanPublish=$($record.canPublishPublicly) CanClose=$($record.canCloseReleaseIssue) CanPromote=$($record.canPromoteRuntimeProof)"
