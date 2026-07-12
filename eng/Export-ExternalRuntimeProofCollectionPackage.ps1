[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
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

function New-CollectionStep {
  param(
    [string]$Id,
    [int]$Order,
    [string]$Command,
    [string]$RequiredEvidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    order = $Order
    command = $Command
    requiredEvidence = $RequiredEvidence
    boundary = $Boundary
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canCloseReleaseIssue = $false
  }
}

$runbook = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook.json"
$backfillPlan = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-backfill-plan.json"
$externalValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"

$runbookCommands = Get-PropertyOrDefault -Object $runbook -Name "commands" -DefaultValue $null
$expectedArtifacts = Get-PropertyOrDefault -Object $runbook -Name "expectedArtifacts" -DefaultValue $null

$inputTemplatePath = [string](Get-PropertyOrDefault -Object $expectedArtifacts -Name "inputTemplatePath" -DefaultValue "artifacts/final-release/external-runtime-proof-record.input-template.json")
$realRecordPath = [string](Get-PropertyOrDefault -Object $expectedArtifacts -Name "realRecordPath" -DefaultValue "artifacts/final-release/external-runtime-proof-record.json")
$smokeLogPath = [string](Get-PropertyOrDefault -Object $expectedArtifacts -Name "smokeLogPath" -DefaultValue "artifacts/final-release/external-runtime-proof/$RuntimePackageKey/package-consumer-smoke.log")
$packageConsumerSummaryPath = [string](Get-PropertyOrDefault -Object $expectedArtifacts -Name "packageConsumerSummaryPath" -DefaultValue "artifacts/package-consumer/package-consumer-validation-summary.json")
$validationPath = [string](Get-PropertyOrDefault -Object $expectedArtifacts -Name "validationPath" -DefaultValue "artifacts/final-release/external-runtime-proof-validation.json")

$generateTemplateCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "generateInputTemplate" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey")
$smokeCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "runPackageConsumerSmoke" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -SmokeRuntimePackageKey $RuntimePackageKey -RunSmoke -KeepConsumerOutput")
$copyRecordCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "copyInputTemplateToRealRecord" -DefaultValue "Copy-Item -LiteralPath $inputTemplatePath -Destination $realRecordPath")
$validateCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "validateFilledRecord" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath $realRecordPath -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof")
$smokeLogHashCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "computeSmokeLogSha256" -DefaultValue "Get-FileHash -LiteralPath `"$smokeLogPath`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash")
$managedHashCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "computeManagedNupkgSha256" -DefaultValue "Get-FileHash -LiteralPath `"<consumed-managed-nupkg>`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash")
$runtimeHashCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "computeRuntimeNupkgSha256" -DefaultValue "Get-FileHash -LiteralPath `"<consumed-runtime-nupkg>`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash")

$copyableExecutionOrder = @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalPackageReviewBundle.ps1",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofRunbook.ps1 -RuntimePackageKey $RuntimePackageKey",
  $generateTemplateCommand,
  $smokeCommand,
  $smokeLogHashCommand,
  $managedHashCommand,
  $runtimeHashCommand,
  $copyRecordCommand,
  "Fill $realRecordPath with real host metadata, package hashes, log hash, stdoutSummary, stderrSummary, and templateOnly=false.",
  $validateCommand,
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFreezeSummary.ps1"
)

$requiredHostMetadata = @(
  "ownerName",
  "machineName",
  "osDescription",
  "gpuName",
  "driverVersion",
  "cudaDriverSupportedRuntime",
  "cudaRuntimeVersion",
  "tensorRtRuntimeVersion",
  "tensorRtLine",
  "cudnnVersion"
)

$requiredLogArtifacts = @(
  [pscustomobject]@{ name = "package consumer smoke log"; path = $smokeLogPath; requiredHash = "command.logSha256"; boundary = "A log path without matching SHA256 is not promotable runtime proof." },
  [pscustomobject]@{ name = "package consumer summary"; path = $packageConsumerSummaryPath; requiredHash = "optional summary SHA256"; boundary = "Summary metadata helps review but cannot replace the real smoke log." }
)

$requiredPackageHashes = @(
  [pscustomobject]@{ field = "packageSource.managedNupkgSha256"; source = "exact consumed managed nupkg"; boundary = "Local package inventory is not enough; hash the package consumed by the clean project." },
  [pscustomobject]@{ field = "packageSource.runtimeNupkgSha256"; source = "exact consumed runtime nupkg"; boundary = "Runtime package hash must match the target runtime package key." }
)

$collectionSteps = @(
  New-CollectionStep -Id "refresh-package-inventory" -Order 1 -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-FinalPackageReviewBundle.ps1" -RequiredEvidence "Owner-visible local package inventory and hashes." -Boundary "Local package inventory is not runtime proof or public package proof."
  New-CollectionStep -Id "refresh-input-template" -Order 2 -Command $generateTemplateCommand -RequiredEvidence $inputTemplatePath -Boundary "Template-only records are not proof."
  New-CollectionStep -Id "run-compatible-host-smoke" -Order 3 -Command $smokeCommand -RequiredEvidence "Clean package consumer smoke log, no ProjectReference, exitCode=0, smokeStatus=passed." -Boundary "blocked-by-cuda-driver and dependency-probe-only are not smoke passed."
  New-CollectionStep -Id "capture-hashes" -Order 4 -Command "$smokeLogHashCommand; $managedHashCommand; $runtimeHashCommand" -RequiredEvidence "Managed/runtime nupkg SHA256 and smoke log SHA256." -Boundary "Missing or mismatched hashes cannot promote runtime proof."
  New-CollectionStep -Id "review-stdout-stderr" -Order 5 -Command "Review $smokeLogPath and fill results.stdoutSummary plus results.stderrSummary; use no-stderr-emitted only when stderr is empty." -RequiredEvidence "Non-empty stdoutSummary and stderrSummary." -Boundary "A log hash without reviewed stdout/stderr summaries is incomplete."
  New-CollectionStep -Id "fill-real-record" -Order 6 -Command $copyRecordCommand -RequiredEvidence "recordKind=external-runtime-proof-record, templateOnly=false, proofClassification=package-consumer-runtime." -Boundary "Draft, example, input-template, and collection package remain non-proof."
  New-CollectionStep -Id "validate-real-record" -Order 7 -Command $validateCommand -RequiredEvidence $validationPath -Boundary "Only validator-promoted external-runtime-proof-record.json can promote runtime proof."
  New-CollectionStep -Id "refresh-release-gates" -Order 8 -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFreezeSummary.ps1" -RequiredEvidence "Release evidence, promotion issue, freeze summary, freeze validation." -Boundary "Refreshing aggregate records cannot fabricate proof."
)

$safetyNotes = @(
  "This collection package is guidance-only and not runtime proof.",
  "This collection package does not publish packages or approve public release.",
  "This collection package keeps canPromoteRuntimeProof=false and canCloseReleaseIssue=false.",
  "Only external-runtime-proof-record.json validated with -RequireExistingLog -FailOnNotProof can promote package-consumer-runtime proof.",
  "blocked-by-cuda-driver is not smoke passed.",
  "dependency-probe-only, build-only, template, draft, example, runbook, collection bundle, and collection package outputs are not release proof.",
  "Do not infer runtime proof from local package inventory without compatible-host runtime execution.",
  "No dotnet nuget push, GitHub Packages upload, GitHub Release upload, delete, delist, or withdraw is performed."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "external-runtime-proof-collection-package"
  packageState = "owner-action-required"
  runtimePackageKey = $RuntimePackageKey
  compatibleHostRequired = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionEvidence = $false
  currentBackfillPlanState = [string](Get-PropertyOrDefault -Object $backfillPlan -Name "planState" -DefaultValue "missing-external-runtime-proof-backfill-plan")
  currentBackfillStepCount = @((Get-PropertyOrDefault -Object $backfillPlan -Name "backfillSteps" -DefaultValue @())).Count
  currentValidationState = [string](Get-PropertyOrDefault -Object $externalValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
  currentProofClassification = [string](Get-PropertyOrDefault -Object $externalValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
  currentValidationCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
  currentFailedProofItemCount = [int](Get-PropertyOrDefault -Object $externalValidation -Name "failedProofItemCount" -DefaultValue -1)
  packageConsumerSmokeStatus = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeResult" -DefaultValue "missing-package-consumer-smoke")
  packageConsumerEvidenceKind = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "EvidenceKind" -DefaultValue "missing-package-consumer-evidence-kind")
  packageConsumerDependencyProbeOnly = [bool](Get-PropertyOrDefault -Object $packageConsumer -Name "IsDependencyProbeOnly" -DefaultValue $true)
  finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
  expectedArtifacts = [pscustomobject]@{
    inputTemplatePath = $inputTemplatePath
    realRecordPath = $realRecordPath
    smokeLogPath = $smokeLogPath
    packageConsumerSummaryPath = $packageConsumerSummaryPath
    validationPath = $validationPath
    collectionPackageJsonPath = "artifacts/final-release/external-runtime-proof-collection-package.json"
    collectionPackageMarkdownPath = "artifacts/final-release/external-runtime-proof-collection-package.md"
  }
  requiredHostMetadata = $requiredHostMetadata
  requiredLogArtifacts = $requiredLogArtifacts
  requiredPackageHashes = $requiredPackageHashes
  copyableExecutionOrder = $copyableExecutionOrder
  collectionSteps = $collectionSteps
  validationCommand = $validateCommand
  sourceEvidence = @(
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-backfill-plan.json",
    "artifacts/final-release/compatible-host-runtime-proof-runbook.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/final-package-review-bundle.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json"
  )
  safetyNotes = $safetyNotes
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "external-runtime-proof-collection-package.json"
$markdownPath = Join-Path $outputRoot "external-runtime-proof-collection-package.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Collection Package")
$lines.Add("")
$lines.Add("- package state: ``$($record.packageState)``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- can promote runtime proof: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("")
$lines.Add("This package is copyable owner guidance for collecting real compatible-host runtime proof. It is not runtime proof, publication approval, release close approval, or package push.")
$lines.Add("")
$lines.Add("## Copyable Execution Order")
$lines.Add("")
foreach ($command in $copyableExecutionOrder) {
  $lines.Add('```powershell')
  $lines.Add($command)
  $lines.Add('```')
}
$lines.Add("")
$lines.Add("## Required Host Metadata")
$lines.Add("")
foreach ($field in $requiredHostMetadata) { $lines.Add("- ``$field``") }
$lines.Add("")
$lines.Add("## Required Package Hashes")
$lines.Add("")
$lines.Add("| Field | Source | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($hash in $requiredPackageHashes) {
  $lines.Add("| ``$($hash.field)`` | $(ConvertTo-MarkdownCell $hash.source) | $(ConvertTo-MarkdownCell $hash.boundary) |")
}
$lines.Add("")
$lines.Add("## Required Log Artifacts")
$lines.Add("")
$lines.Add("| Name | Path | Required hash | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($artifact in $requiredLogArtifacts) {
  $lines.Add("| $(ConvertTo-MarkdownCell $artifact.name) | ``$($artifact.path)`` | ``$($artifact.requiredHash)`` | $(ConvertTo-MarkdownCell $artifact.boundary) |")
}
$lines.Add("")
$lines.Add("## Collection Steps")
$lines.Add("")
$lines.Add("| Order | ID | Command | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($step in $collectionSteps) {
  $lines.Add("| $($step.order) | ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.command) | $(ConvertTo-MarkdownCell $step.requiredEvidence) | $(ConvertTo-MarkdownCell $step.boundary) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $safetyNotes) { $lines.Add("- $note") }
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "External runtime proof collection package written to $jsonPath"
Write-Host "External runtime proof collection package written to $markdownPath"
Write-Host "PackageState=$($record.packageState) PerformsPublish=False CanPromoteRuntimeProof=False CanCloseReleaseIssue=False IsRuntimeExecutionEvidence=False"
