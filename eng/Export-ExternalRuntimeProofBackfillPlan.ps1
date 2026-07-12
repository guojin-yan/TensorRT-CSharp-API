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

function New-BackfillStep {
  param(
    [string]$Id,
    [int]$Order,
    [string]$Title,
    [string]$Command,
    [string]$RequiredEvidence,
    [string]$OutputArtifact,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    order = $Order
    title = $Title
    command = $Command
    requiredEvidence = $RequiredEvidence
    outputArtifact = $OutputArtifact
    boundary = $Boundary
    performsPublish = $false
    canPromoteRuntimeProof = $false
  }
}

$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$externalRuntimeProofDraft = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record.draft.json"
$ownerHandoff = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-owner-handoff.json"
$collectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"

$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "runtimeProofStatus" -DefaultValue "missing-runtime-proof-status")
$validationState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$proofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$isRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$failedProofItemCount = [int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)
$runtimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)
$packageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)
$managedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)
$runtimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)
$logSha256Matches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256Matches" -DefaultValue $false)
$consumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)
$smokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)
$hostReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)
$commandsReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)
$stdoutStderrSummariesReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "stdoutStderrSummariesReady" -DefaultValue $false)
$draftState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "proofState" -DefaultValue "missing-external-runtime-proof-draft")
$ownerHandoffState = [string](Get-PropertyOrDefault -Object $ownerHandoff -Name "handoffState" -DefaultValue "missing-external-runtime-proof-owner-handoff")
$collectionState = [string](Get-PropertyOrDefault -Object $collectionBundle -Name "collectionState" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)

$realRecordPath = "artifacts/final-release/external-runtime-proof-record.json"
$inputTemplatePath = "artifacts/final-release/external-runtime-proof-record.input-template.json"
$smokeLogPath = "artifacts/final-release/external-runtime-proof/$RuntimePackageKey/package-consumer-smoke.log"
$validationPath = "artifacts/final-release/external-runtime-proof-validation.json"

$steps = @(
  New-BackfillStep -Id "refresh-input-template" -Order 1 -Title "Refresh external proof input template" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey" -RequiredEvidence $inputTemplatePath -OutputArtifact $inputTemplatePath -Boundary "Input template is not runtime proof."
  New-BackfillStep -Id "refresh-owner-handoff" -Order 2 -Title "Refresh owner handoff and compatible-host collection bundle" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofOwnerHandoff.ps1 -RuntimePackageKey $RuntimePackageKey; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofRunbook.ps1 -RuntimePackageKey $RuntimePackageKey; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofCollectionBundle.ps1 -RuntimePackageKey $RuntimePackageKey" -RequiredEvidence "external-runtime-proof-owner-handoff.json; compatible-host-runtime-proof-runbook.json; compatible-host-runtime-proof-collection-bundle.json" -OutputArtifact "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json" -Boundary "Handoff, runbook, and collection bundle are executable guidance, not proof."
  New-BackfillStep -Id "run-compatible-host-smoke" -Order 3 -Title "Run package consumer smoke on compatible CUDA host" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -SmokeRuntimePackageKey $RuntimePackageKey -RunSmoke -KeepConsumerOutput" -RequiredEvidence "clean consumer restore/build log, DependencyProbe output, smoke output, exitCode=0, smokeStatus=passed" -OutputArtifact $smokeLogPath -Boundary "blocked-by-cuda-driver, dependency-probe-only, and local build-only output are not smoke passed."
  New-BackfillStep -Id "capture-package-and-log-hashes" -Order 4 -Title "Capture consumed package and smoke log SHA256" -Command "Get-FileHash -LiteralPath `"<downloaded-managed-nupkg>`" -Algorithm SHA256; Get-FileHash -LiteralPath `"<downloaded-runtime-nupkg>`" -Algorithm SHA256; Get-FileHash -LiteralPath `"$smokeLogPath`" -Algorithm SHA256" -RequiredEvidence "64-character managed/runtime nupkg SHA256 and matching smoke log SHA256" -OutputArtifact $realRecordPath -Boundary "Package hashes and log hash must identify the exact consumed files."
  New-BackfillStep -Id "fill-real-record" -Order 5 -Title "Fill real external-runtime-proof-record" -Command "Copy-Item -LiteralPath $inputTemplatePath -Destination $realRecordPath; edit $realRecordPath with owner, host, packageSource, command, results, stdoutSummary, and stderrSummary fields" -RequiredEvidence "recordKind=external-runtime-proof-record; templateOnly=false; proofClassification=package-consumer-runtime; no ProjectReference; host CUDA/TensorRT/cuDNN metadata" -OutputArtifact $realRecordPath -Boundary "A filled JSON is still not proof until the validator passes with existing-log SHA256 checks."
  New-BackfillStep -Id "validate-real-record" -Order 6 -Title "Validate real runtime proof record" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath $realRecordPath -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof" -RequiredEvidence "validationState=real-runtime-proof; canPromoteRuntimeProof=true; failedProofItemCount=0" -OutputArtifact $validationPath -Boundary "Only this validator output can promote external runtime proof."
  New-BackfillStep -Id "refresh-release-gates" -Order 7 -Title "Refresh release evidence after proof passes" -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionRecord.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1" -RequiredEvidence "release evidence and owner-facing records show realExternalRuntimeProofReady=true only after the real validation passes" -OutputArtifact "artifacts/final-release/release-evidence-bundle.json" -Boundary "Refreshing gates is not publication and does not close post-publish proof."
)

$blockingReasons = New-Object System.Collections.Generic.List[string]
if (-not $canPromoteRuntimeProof) { $blockingReasons.Add("external runtime proof is not promotable") }
if (-not $isRuntimeExecutionEvidence) { $blockingReasons.Add("missing real runtime execution evidence") }
if ([string]::Equals($runtimeProofStatus, "blocked-by-cuda-driver", [System.StringComparison]::OrdinalIgnoreCase)) { $blockingReasons.Add("blocked-by-cuda-driver is not smoke passed") }
if ($failedProofItemCount -ne 0) { $blockingReasons.Add("external runtime proof validator has failed proof items") }
if (-not $runtimePackageKeyMatches) { $blockingReasons.Add("runtimePackageKey does not match release target") }
if (-not $packageSourceRuntimePackageKeyMatches) { $blockingReasons.Add("packageSource.runtimePackageKey does not match release target") }
if (-not ($managedNupkgSha256Ready -and $runtimeNupkgSha256Ready)) { $blockingReasons.Add("managed/runtime nupkg SHA256 evidence is incomplete") }
if (-not $logSha256Matches) { $blockingReasons.Add("smoke log SHA256 is missing or mismatched") }
if (-not $consumerProjectIdentityReady) { $blockingReasons.Add("clean consumer project identity is incomplete") }
if (-not $smokeCommandRuntimeKeyReady) { $blockingReasons.Add("smoke command does not prove the target runtime package key") }
if (-not $hostReady) { $blockingReasons.Add("host CUDA/TensorRT/cuDNN metadata is incomplete") }
if (-not $commandsReady) { $blockingReasons.Add("restore/build/smoke command capture is incomplete") }
if (-not $stdoutStderrSummariesReady) { $blockingReasons.Add("stdout/stderr summaries are incomplete") }

$planState = if ($blockingReasons.Count -eq 0) { "ready-after-real-external-runtime-proof" } else { "blocked-compatible-host-proof-required" }

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "external-runtime-proof-backfill-plan"
  runtimePackageKey = $RuntimePackageKey
  planState = $planState
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canCloseReleaseIssue = $false
  runtimeProofStatus = $runtimeProofStatus
  validationState = $validationState
  proofClassification = $proofClassification
  realRuntimeExecutionEvidenceReady = $isRuntimeExecutionEvidence
  externalRuntimeProofCanPromoteRuntimeProof = $canPromoteRuntimeProof
  failedProofItemCount = $failedProofItemCount
  draftState = $draftState
  ownerHandoffState = $ownerHandoffState
  collectionState = $collectionState
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  runtimePackageKeyMatches = $runtimePackageKeyMatches
  packageSourceRuntimePackageKeyMatches = $packageSourceRuntimePackageKeyMatches
  managedNupkgSha256Ready = $managedNupkgSha256Ready
  runtimeNupkgSha256Ready = $runtimeNupkgSha256Ready
  logSha256Matches = $logSha256Matches
  consumerProjectIdentityReady = $consumerProjectIdentityReady
  smokeCommandRuntimeKeyReady = $smokeCommandRuntimeKeyReady
  hostReady = $hostReady
  commandsReady = $commandsReady
  stdoutStderrSummariesReady = $stdoutStderrSummariesReady
  blockingReasons = @($blockingReasons)
  backfillSteps = @($steps)
  expectedArtifacts = [pscustomobject]@{
    inputTemplatePath = $inputTemplatePath
    realRecordPath = $realRecordPath
    smokeLogPath = $smokeLogPath
    validationPath = $validationPath
  }
  sourceEvidence = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-owner-handoff.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/final-package-review-bundle.json"
  )
  safetyNotes = @(
    "This backfill plan does not publish packages.",
    "This backfill plan is not runtime proof.",
    "Templates, drafts, examples, runbooks, and collection bundles are not promotable proof.",
    "blocked-by-cuda-driver is not smoke passed.",
    "Only external-runtime-proof-record.json validated with -RequireExistingLog -FailOnNotProof can promote external runtime proof.",
    "final-package-review-bundle is local package inventory only, not public package proof."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "external-runtime-proof-backfill-plan.json"
$markdownPath = Join-Path $outputRoot "external-runtime-proof-backfill-plan.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# External Runtime Proof Backfill Plan")
$lines.Add("")
$lines.Add("- plan state: ``$planState``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- proof classification: ``$proofClassification``")
$lines.Add("- can promote runtime proof: ``$canPromoteRuntimeProof``")
$lines.Add("- real runtime execution evidence: ``$isRuntimeExecutionEvidence``")
$lines.Add("- failed proof item count: ``$failedProofItemCount``")
$lines.Add("- blocking reasons: $($blockingReasons.Count)")
$lines.Add("")
$lines.Add("## Backfill Steps")
$lines.Add("")
$lines.Add("| Order | ID | Command | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($step in $steps) {
  $lines.Add("| $($step.order) | ``$($step.id)`` | $($step.command.Replace("|", "\|")) | $($step.requiredEvidence.Replace("|", "\|")) | $($step.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Blocking Reasons")
$lines.Add("")
foreach ($reason in $blockingReasons) {
  $lines.Add("- $reason")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "External runtime proof backfill plan written to $jsonPath"
Write-Host "External runtime proof backfill plan written to $markdownPath"
