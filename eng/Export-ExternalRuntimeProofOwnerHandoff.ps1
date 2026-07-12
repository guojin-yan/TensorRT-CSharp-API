[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

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
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-HandoffStep {
  param(
    [string]$Id,
    [string]$Action,
    [string]$RequiredEvidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    action = $Action
    requiredEvidence = $RequiredEvidence
    boundary = $Boundary
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$runtimeReadiness = Read-JsonOrNull "artifacts\package-readiness\runtime-package-readiness-summary.json"

$packageConsumerRuntimeProofStatus = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeResult" -DefaultValue "missing-runtime-proof-status")
$runtimeReadinessProofStatus = [string](Get-PropertyOrDefault -Object $runtimeReadiness -Name "runtimeProofStatus" -DefaultValue $packageConsumerRuntimeProofStatus)
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofStatus" -DefaultValue $runtimeReadinessProofStatus)
$runtimeProofRequiredForRelease = [bool](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofRequiredForRelease" -DefaultValue $true)
$runtimeProofBlockerCategory = [string](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofBlockerCategory" -DefaultValue "runtime-proof-incomplete")
$runtimeProofBlockerOwnerActionStatus = [string](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofBlockerOwnerActionStatus" -DefaultValue "owner-action-required")

$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$runtimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)
$packageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)
$managedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)
$runtimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)
$logSha256FormatReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256FormatReady" -DefaultValue $false)
$logSha256Matches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256Matches" -DefaultValue $false)
$stdoutSummaryReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "stdoutSummaryReady" -DefaultValue $false)
$stderrSummaryReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "stderrSummaryReady" -DefaultValue $false)
$stdoutStderrSummariesReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "stdoutStderrSummariesReady" -DefaultValue $false)
$failedProofItemCount = [int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)
$canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$isRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$noProjectReference = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "noProjectReference" -DefaultValue $false)
$packageSourceReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceReady" -DefaultValue $false)
$commandsReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)
$hostReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)
$consumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)
$smokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)
$compatibleHostRequired = -not $hostReady -or [string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)
$requiredHostAction = "Run package consumer smoke on a CUDA-compatible GPU host with the exact runtime package key and a compatible NVIDIA driver/runtime stack."
$promotionBlockedReason = if ($canPromoteRuntimeProof -and $isRuntimeExecutionEvidence) { "none" } elseif ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase) -or $externalRuntimeProofState -like "*blocked-by-cuda-driver*") { "blocked-by-cuda-driver is not smoke passed" } else { "external runtime proof is not package-consumer-runtime proof" }
$ownerActionStatus = if ($canPromoteRuntimeProof -and $isRuntimeExecutionEvidence) { "resolved" } else { "owner-action-required" }
$handoffState = if ($ownerActionStatus -eq "resolved") { "ready" } else { "owner-action-required" }

$expectedRecordPath = "artifacts/final-release/external-runtime-proof-record.json"
$inputTemplatePath = "artifacts/final-release/external-runtime-proof-record.input-template.json"
$expectedSmokeLogPath = "artifacts/final-release/external-runtime-proof/$RuntimePackageKey/package-consumer-smoke.log"
$expectedConsumerSummaryPath = "artifacts/package-consumer/package-consumer-validation-summary.json"
$expectedManagedNupkgPath = "artifacts/final-release/packages/JYPPX.TensorRtSharp.*.nupkg"
$expectedRuntimeNupkgPath = "artifacts/final-release/packages/JYPPX.TensorRtSharp.runtime.$RuntimePackageKey.*.nupkg"

$generateTemplateCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey"
$smokeCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -SmokeRuntimePackageKey $RuntimePackageKey -RunSmoke -KeepConsumerOutput"
$hashCommand = "Get-FileHash -LiteralPath `"$expectedSmokeLogPath`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash"
$packageHashCommand = "Get-FileHash -LiteralPath `"<downloaded-managed-or-runtime-nupkg>`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash"
$validateCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath $expectedRecordPath -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"

$backfillSteps = @(
  New-HandoffStep `
    -Id "generate-input-template" `
    -Action "Generate or refresh the human-fill external runtime proof input template." `
    -RequiredEvidence $inputTemplatePath `
    -Boundary "The input template is not runtime proof."
  New-HandoffStep `
    -Id "run-compatible-host-smoke" `
    -Action "Run full package consumer smoke on a CUDA-compatible GPU host for the exact runtime package key." `
    -RequiredEvidence "$expectedConsumerSummaryPath; $expectedSmokeLogPath" `
    -Boundary "blocked-by-cuda-driver and DependencyProbe output are not smoke passed."
  New-HandoffStep `
    -Id "capture-log-sha256" `
    -Action "Compute SHA256 for the saved smoke log and copy it into command.logSha256." `
    -RequiredEvidence "64-character SHA256 from Get-FileHash -Algorithm SHA256." `
    -Boundary "A missing or mismatched logSha256 cannot promote runtime proof."
  New-HandoffStep `
    -Id "capture-package-sha256" `
    -Action "Compute SHA256 for the exact managed and runtime nupkg files consumed by the smoke record and copy them into packageSource.managedNupkgSha256 and packageSource.runtimeNupkgSha256." `
    -RequiredEvidence "$expectedManagedNupkgPath; $expectedRuntimeNupkgPath; 64-character SHA256 for each downloaded nupkg." `
    -Boundary "Missing managed/runtime nupkg hashes or a mismatched packageSource.runtimePackageKey cannot promote runtime proof."
  New-HandoffStep `
    -Id "review-stdout-stderr" `
    -Action "Review the preserved smoke log and fill results.stdoutSummary plus results.stderrSummary. If the process emitted no stderr, write an explicit no-stderr-emitted note in results.stderrSummary." `
    -RequiredEvidence "Non-empty results.stdoutSummary and non-empty results.stderrSummary in $expectedRecordPath." `
    -Boundary "A log hash without reviewed stdout/stderr summaries cannot promote runtime proof."
  New-HandoffStep `
    -Id "fill-real-record" `
    -Action "Copy the input template to external-runtime-proof-record.json and fill owner, complete host CUDA/TensorRT/cuDNN metadata, package source, packageSource.runtimePackageKey, consumer project identity, managed/runtime nupkg SHA256, command, result, and summary fields." `
    -RequiredEvidence $expectedRecordPath `
    -Boundary "recordKind must be external-runtime-proof-record, templateOnly=false, proofClassification=package-consumer-runtime, no ProjectReference, command.smokeCommand includes --runtime-package-key."
  New-HandoffStep `
    -Id "validate-with-existing-log" `
    -Action "Validate the filled record with -RequireExistingLog so the validator recomputes the log SHA256." `
    -RequiredEvidence "artifacts/final-release/external-runtime-proof-validation.json" `
    -Boundary "runtimePackageKeyMatches=true alone is not proof; packageSourceRuntimePackageKeyMatches, managed/runtime nupkg SHA256, logSha256Matches, and all proof items must pass."
)

$nonProofBoundaries = @(
  "template-only is not runtime proof.",
  "example-not-for-publication is not runtime proof.",
  "build-only and dependency-probe-only are not runtime proof.",
  "synthetic-input-runtime and real-model-runtime do not replace package-consumer-runtime.",
  "blocked-by-cuda-driver is not smoke passed.",
  "runtimePackageKeyMatches=true without packageSourceRuntimePackageKeyMatches=true is still owner-action-required.",
  "consumerProjectIdentityReady=false is still owner-action-required.",
  "smokeCommandRuntimeKeyReady=false is still owner-action-required.",
  "hostReady=false, including missing cuDNN/TensorRT/CUDA/driver/GPU/OS metadata, is still owner-action-required.",
  "managedNupkgSha256Ready=false or runtimeNupkgSha256Ready=false is still owner-action-required.",
  "runtimePackageKeyMatches=true without logSha256Matches=true is still owner-action-required.",
  "stdoutSummaryReady=false or stderrSummaryReady=false is still owner-action-required.",
  "The handoff record does not publish packages and does not approve public release."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "external-runtime-proof-owner-handoff"
  handoffState = $handoffState
  ownerActionStatus = $ownerActionStatus
  runtimePackageKey = $RuntimePackageKey
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  runtimeProofBlockerCategory = $runtimeProofBlockerCategory
  runtimeProofBlockerOwnerActionStatus = $runtimeProofBlockerOwnerActionStatus
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
  runtimePackageKeyMatches = $runtimePackageKeyMatches
  packageSourceRuntimePackageKeyMatches = $packageSourceRuntimePackageKeyMatches
  managedNupkgSha256Ready = $managedNupkgSha256Ready
  runtimeNupkgSha256Ready = $runtimeNupkgSha256Ready
  logSha256FormatReady = $logSha256FormatReady
  logSha256Matches = $logSha256Matches
  stdoutSummaryReady = $stdoutSummaryReady
  stderrSummaryReady = $stderrSummaryReady
  stdoutStderrSummariesReady = $stdoutStderrSummariesReady
  failedProofItemCount = $failedProofItemCount
  canPromoteRuntimeProof = $canPromoteRuntimeProof
  isRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
  noProjectReference = $noProjectReference
  packageSourceReady = $packageSourceReady
  commandsReady = $commandsReady
  hostReady = $hostReady
  consumerProjectIdentityReady = $consumerProjectIdentityReady
  smokeCommandRuntimeKeyReady = $smokeCommandRuntimeKeyReady
  compatibleHostRequired = $compatibleHostRequired
  requiredHostAction = $requiredHostAction
  promotionBlockedReason = $promotionBlockedReason
  expectedArtifacts = [pscustomobject]@{
    inputTemplatePath = $inputTemplatePath
    realRecordPath = $expectedRecordPath
    smokeLogPath = $expectedSmokeLogPath
    managedNupkgPath = $expectedManagedNupkgPath
    runtimeNupkgPath = $expectedRuntimeNupkgPath
    packageConsumerSummaryPath = $expectedConsumerSummaryPath
    validationPath = "artifacts/final-release/external-runtime-proof-validation.json"
  }
  commands = [pscustomobject]@{
    generateInputTemplate = $generateTemplateCommand
    runPackageConsumerSmoke = $smokeCommand
    computeSmokeLogSha256 = $hashCommand
    computePackageNupkgSha256 = $packageHashCommand
    validateFilledRecord = $validateCommand
  }
  backfillSteps = @($backfillSteps)
  nonProofBoundaries = @($nonProofBoundaries)
  sourceEvidence = @(
    "artifacts/final-release/final-release-dry-run-summary.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/package-readiness/runtime-package-readiness-summary.json"
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "external-runtime-proof-owner-handoff.json"
$markdownPath = Join-Path $outputRoot "external-runtime-proof-owner-handoff.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Owner Handoff")
$lines.Add("")
$lines.Add("- handoff state: ``$handoffState``")
$lines.Add("- owner action status: ``$ownerActionStatus``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- runtime proof required for release: ``$runtimeProofRequiredForRelease``")
$lines.Add("- external proof state: ``$externalRuntimeProofState``")
$lines.Add("- external proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- runtime package key matches: ``$runtimePackageKeyMatches``")
$lines.Add("- package source runtime package key matches: ``$packageSourceRuntimePackageKeyMatches``")
$lines.Add("- managed nupkg SHA256 ready: ``$managedNupkgSha256Ready``")
$lines.Add("- runtime nupkg SHA256 ready: ``$runtimeNupkgSha256Ready``")
$lines.Add("- log SHA256 format ready: ``$logSha256FormatReady``")
$lines.Add("- log SHA256 matches: ``$logSha256Matches``")
$lines.Add("- failed proof item count: ``$failedProofItemCount``")
$lines.Add("- can promote runtime proof: ``$canPromoteRuntimeProof``")
$lines.Add("- no ProjectReference: ``$noProjectReference``")
$lines.Add("- package source ready: ``$packageSourceReady``")
$lines.Add("- commands ready: ``$commandsReady``")
$lines.Add("- host ready: ``$hostReady``")
$lines.Add("- consumer project identity ready: ``$consumerProjectIdentityReady``")
$lines.Add("- smoke command runtime package key ready: ``$smokeCommandRuntimeKeyReady``")
$lines.Add("- compatible host required: ``$compatibleHostRequired``")
$lines.Add("- required host action: $requiredHostAction")
$lines.Add("- promotion blocked reason: $promotionBlockedReason")
$lines.Add("")
$lines.Add("## Commands")
$lines.Add("")
$lines.Add('```powershell')
$lines.Add($generateTemplateCommand)
$lines.Add($smokeCommand)
$lines.Add($hashCommand)
$lines.Add($packageHashCommand)
$lines.Add($validateCommand)
$lines.Add('```')
$lines.Add("")
$lines.Add("## Backfill Steps")
$lines.Add("")
$lines.Add("| ID | Action | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($step in $backfillSteps) {
  $lines.Add("| ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.action) | $(ConvertTo-MarkdownCell $step.requiredEvidence) | $(ConvertTo-MarkdownCell $step.boundary) |")
}
$lines.Add("")
$lines.Add("## Non-Proof Boundaries")
$lines.Add("")
foreach ($boundary in $nonProofBoundaries) {
  $lines.Add("- $boundary")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "External runtime proof owner handoff written to $jsonPath"
Write-Host "External runtime proof owner handoff written to $markdownPath"
Write-Host "HandoffState=$handoffState OwnerActionStatus=$ownerActionStatus RuntimePackageKeyMatches=$runtimePackageKeyMatches PackageSourceRuntimePackageKeyMatches=$packageSourceRuntimePackageKeyMatches ConsumerProjectIdentityReady=$consumerProjectIdentityReady SmokeCommandRuntimeKeyReady=$smokeCommandRuntimeKeyReady ManagedNupkgSha256Ready=$managedNupkgSha256Ready RuntimeNupkgSha256Ready=$runtimeNupkgSha256Ready LogSha256Matches=$logSha256Matches FailedProofItemCount=$failedProofItemCount CanPromoteRuntimeProof=$canPromoteRuntimeProof"
