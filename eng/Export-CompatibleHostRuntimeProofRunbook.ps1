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

function Test-Sha256Ready {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return -not [string]::IsNullOrWhiteSpace($text) -and [System.Text.RegularExpressions.Regex]::IsMatch($text, "^[0-9a-fA-F]{64}$")
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-RunbookStep {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Command,
    [string]$ExpectedEvidence,
    [string]$ProofBoundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    command = $Command
    expectedEvidence = $ExpectedEvidence
    proofBoundary = $ProofBoundary
  }
}

function New-RequiredRecordField {
  param(
    [string]$Path,
    [string]$ExpectedValue,
    [string]$Why
  )

  [pscustomobject]@{
    path = $Path
    expectedValue = $ExpectedValue
    why = $Why
  }
}

$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$externalRuntimeProofDraft = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record.draft.json"
$externalRuntimeOwnerHandoff = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-owner-handoff.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$releaseCandidatePackageInventory = Read-JsonOrNull "artifacts\final-release\release-candidate-package-inventory.json"
$releasePackageProof = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$localFeedConsumer = Read-JsonOrNull "artifacts\local-feed-consumer\local-nuget-feed-consumer-summary.json"

$draftPackageSource = Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "packageSource" -DefaultValue $null
$draftCommand = Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "command" -DefaultValue $null
$draftResults = Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "results" -DefaultValue $null

$validationState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$proofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$isRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$canPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$failedProofItemCount = [int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofStatus" -DefaultValue ([string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeResult" -DefaultValue "missing-runtime-proof-status")))
$runtimeProofBlockerCategory = [string](Get-PropertyOrDefault -Object $finalRelease -Name "runtimeProofBlockerCategory" -DefaultValue "runtime-proof-incomplete")
$bundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$draftProofState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "proofState" -DefaultValue "missing-external-runtime-proof-draft")
$draftProofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "proofClassification" -DefaultValue "missing-draft-proof-classification")
$draftCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "canPromoteRuntimeProof" -DefaultValue $false)
$draftRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofDraft -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$draftManagedNupkgSha256Ready = Test-Sha256Ready (Get-PropertyOrDefault -Object $draftPackageSource -Name "managedNupkgSha256" -DefaultValue "")
$draftRuntimeNupkgSha256Ready = Test-Sha256Ready (Get-PropertyOrDefault -Object $draftPackageSource -Name "runtimeNupkgSha256" -DefaultValue "")
$draftSmokeLogSha256Ready = Test-Sha256Ready (Get-PropertyOrDefault -Object $draftCommand -Name "logSha256" -DefaultValue "")
$validationStdoutSummaryReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "stdoutSummaryReady" -DefaultValue $false)
$validationStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "stderrSummaryReady" -DefaultValue $false)
$validationStdoutStderrSummariesReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "stdoutStderrSummariesReady" -DefaultValue $false)
$draftNoProjectReference = [bool](Get-PropertyOrDefault -Object $draftPackageSource -Name "noProjectReference" -DefaultValue $false)
$draftSmokeStatus = [string](Get-PropertyOrDefault -Object $draftResults -Name "smokeStatus" -DefaultValue "missing-draft-smoke-status")
$ownerHandoffState = [string](Get-PropertyOrDefault -Object $externalRuntimeOwnerHandoff -Name "handoffState" -DefaultValue "missing-external-runtime-proof-owner-handoff")
$ownerHandoffActionStatus = [string](Get-PropertyOrDefault -Object $externalRuntimeOwnerHandoff -Name "ownerActionStatus" -DefaultValue "missing-owner-action-status")
$packageInventoryRecordKind = [string](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "recordKind" -DefaultValue "missing-release-candidate-package-inventory")
$packageInventoryPackageCount = [int](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "packageCount" -DefaultValue 0)
$packageInventorySha256Ready = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "sha256Ready" -DefaultValue $false)
$packageInventoryCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$packageInventoryCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "canCloseReleaseIssue" -DefaultValue $false)
$releasePackageProofState = [string](Get-PropertyOrDefault -Object $releasePackageProof -Name "proofState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releasePackageProof -Name "packageInventoryState" -DefaultValue "missing-release-package-proof-bundle")))
$releasePackageProofCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releasePackageProof -Name "canPromoteRuntimeProof" -DefaultValue $false)
$releasePackageProofCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $releasePackageProof -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$releasePackageProofCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releasePackageProof -Name "canCloseReleaseIssue" -DefaultValue $false)
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "reviewState" -DefaultValue ([string](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageInventoryState" -DefaultValue "missing-final-package-review-bundle")))
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$finalPackageReviewCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canCloseReleaseIssue" -DefaultValue $false)
$localFeedConsumerRunStatus = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RunStatus" -DefaultValue "missing-local-feed-consumer-summary")
$localFeedConsumerRuntimePackageKey = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RuntimePackageKey" -DefaultValue "missing-runtime-package-key")
$localFeedConsumerRestoreSourceMode = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RestoreSourceMode" -DefaultValue "missing-restore-source-mode")
$localFeedConsumerUsesProjectReference = [bool](Get-PropertyOrDefault -Object $localFeedConsumer -Name "UsesProjectReference" -DefaultValue $true)

$compatibleHostRequired = -not ($canPromoteRuntimeProof -and $isRuntimeExecutionEvidence)
$promotionBlockedReason = if ($canPromoteRuntimeProof -and $isRuntimeExecutionEvidence) {
  "none"
}
else {
  "blocked-by-cuda-driver is not smoke passed"
}

$runbookState = if ($canPromoteRuntimeProof -and $isRuntimeExecutionEvidence) {
  "runtime-proof-ready"
}
else {
  "owner-action-required"
}

$inputTemplatePath = "artifacts/final-release/external-runtime-proof-record.input-template.json"
$realRecordPath = "artifacts/final-release/external-runtime-proof-record.json"
$smokeLogPath = "artifacts/final-release/external-runtime-proof/$RuntimePackageKey/package-consumer-smoke.log"
$consumerSummaryPath = "artifacts/package-consumer/package-consumer-validation-summary.json"
$managedNupkgPath = "artifacts/final-release/packages/JYPPX.TensorRtSharp.*.nupkg"
$runtimeNupkgPath = "artifacts/final-release/packages/JYPPX.TensorRtSharp.runtime.$RuntimePackageKey.*.nupkg"

$generateTemplateCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey"
$smokeCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -SmokeRuntimePackageKey $RuntimePackageKey -RunSmoke -KeepConsumerOutput"
$smokeLogHashCommand = "Get-FileHash -LiteralPath `"$smokeLogPath`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash"
$managedPackageHashCommand = "Get-FileHash -LiteralPath `"<downloaded-managed-nupkg>`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash"
$runtimePackageHashCommand = "Get-FileHash -LiteralPath `"<downloaded-runtime-nupkg>`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash"
$copyRecordCommand = "Copy-Item -LiteralPath $inputTemplatePath -Destination $realRecordPath"
$validateCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath $realRecordPath -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
$ownerInputTemplatePath = "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json"
$ownerInputRealInputPath = "artifacts/final-release/package-consumer-runtime-proof-owner-input.json"
$ownerInputValidationPath = "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json"
$ownerInputImportPath = "artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json"
$ownerInputTemplateCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey"
$copyOwnerInputTemplateCommand = "Copy-Item -LiteralPath $ownerInputTemplatePath -Destination $ownerInputRealInputPath"
$ownerInputStrictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -InputPath $ownerInputRealInputPath -Strict"
$ownerInputImportCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 -InputPath $ownerInputRealInputPath -Strict"
$refreshBundleCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1 -RuntimePackageKey $RuntimePackageKey"

$ownerRuntimeSmokeRunbookState = if ($canPromoteRuntimeProof -and $isRuntimeExecutionEvidence) {
  "runtime-smoke-proof-ready"
}
else {
  "blocked-owner-compatible-host-runtime-smoke"
}

$forbiddenRuntimeSmokeSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "Smoke=not-requested",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "build-only",
  "dry-run",
  "template-only"
)

$ownerInputArtifacts = @(
  [pscustomobject]@{
    id = "release-candidate-package-inventory"
    path = "artifacts/final-release/release-candidate-package-inventory.json"
    state = "$packageInventoryRecordKind; packageCount=$packageInventoryPackageCount; sha256Ready=$packageInventorySha256Ready"
    boundary = "Local package inventory records package identity and hashes for owner review only; it is not public channel proof, runtime proof, or post-publish proof."
  }
  [pscustomobject]@{
    id = "release-package-proof-bundle"
    path = "artifacts/final-release/release-package-proof-bundle.json"
    state = "$releasePackageProofState; canPromoteRuntimeProof=$releasePackageProofCanPromoteRuntimeProof; canUseAsPublicPackageProof=$releasePackageProofCanUseAsPublicPackageProof"
    boundary = "Release package proof aggregates local package layout and feed evidence only; it cannot promote runtime proof or close a release issue."
  }
  [pscustomobject]@{
    id = "final-package-review-bundle"
    path = "artifacts/final-release/final-package-review-bundle.json"
    state = "$finalPackageReviewState; packageCount=$finalPackageReviewPackageCount; canUseAsPublicPackageProof=$finalPackageReviewCanUseAsPublicPackageProof"
    boundary = "Final package review is owner-facing review input; it is not owner authorization, public package proof, runtime proof, or post-publish proof."
  }
  [pscustomobject]@{
    id = "local-feed-consumer-summary"
    path = "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json"
    state = "$localFeedConsumerRunStatus; runtimePackageKey=$localFeedConsumerRuntimePackageKey; restoreSourceMode=$localFeedConsumerRestoreSourceMode; usesProjectReference=$localFeedConsumerUsesProjectReference"
    boundary = "Local feed consumer evidence can prove restore/build/dependency-probe behavior only; it is not post-publish proof or runtime execution proof."
  }
)

$steps = @(
  New-RunbookStep `
    -Id "refresh-input-template" `
    -Title "Refresh the real proof input template" `
    -Command $generateTemplateCommand `
    -ExpectedEvidence $inputTemplatePath `
    -ProofBoundary "The input template is not proof and must keep templateOnly=true until copied and filled."
  New-RunbookStep `
    -Id "run-package-consumer-smoke" `
    -Title "Run package consumer smoke on a compatible CUDA host" `
    -Command $smokeCommand `
    -ExpectedEvidence "$consumerSummaryPath; $smokeLogPath; smokeStatus=passed; exitCode=0" `
    -ProofBoundary "blocked-by-cuda-driver, dependency-probe-only, and build-only outputs are not smoke passed."
  New-RunbookStep `
    -Id "hash-smoke-log" `
    -Title "Hash the preserved smoke log" `
    -Command $smokeLogHashCommand `
    -ExpectedEvidence "64-character SHA256 copied into command.logSha256" `
    -ProofBoundary "A log path without a matching SHA256 cannot promote runtime proof."
  New-RunbookStep `
    -Id "hash-managed-package" `
    -Title "Hash the consumed managed nupkg" `
    -Command $managedPackageHashCommand `
    -ExpectedEvidence "$managedNupkgPath; 64-character SHA256 copied into packageSource.managedNupkgSha256" `
    -ProofBoundary "A local build artifact or unchecked package hash is not public package proof."
  New-RunbookStep `
    -Id "hash-runtime-package" `
    -Title "Hash the consumed runtime nupkg" `
    -Command $runtimePackageHashCommand `
    -ExpectedEvidence "$runtimeNupkgPath; 64-character SHA256 copied into packageSource.runtimeNupkgSha256" `
    -ProofBoundary "The runtime nupkg must match the exact runtime package key."
  New-RunbookStep `
    -Id "review-stdout-stderr" `
    -Title "Review stdout and stderr summaries" `
    -Command "Review $smokeLogPath and fill results.stdoutSummary plus results.stderrSummary; use no-stderr-emitted only when stderr is actually empty." `
    -ExpectedEvidence "results.stdoutSummary and results.stderrSummary are both non-empty" `
    -ProofBoundary "A log hash alone cannot promote runtime proof without reviewed stdout/stderr summaries."
  New-RunbookStep `
    -Id "fill-real-record" `
    -Title "Fill the real external runtime proof record" `
    -Command $copyRecordCommand `
    -ExpectedEvidence $realRecordPath `
    -ProofBoundary "Only a reviewed recordKind=external-runtime-proof-record, templateOnly=false record can be validated as proof."
  New-RunbookStep `
    -Id "validate-real-record" `
    -Title "Validate with existing log and promotion gate" `
    -Command $validateCommand `
    -ExpectedEvidence "ValidationState=runtime-proof-ready; canPromoteRuntimeProof=True; isRuntimeExecutionEvidence=True" `
    -ProofBoundary "The -FailOnNotProof gate must pass before release evidence can treat it as runtime proof."
  New-RunbookStep `
    -Id "export-owner-input-template" `
    -Title "Export package consumer runtime proof owner input template" `
    -Command $ownerInputTemplateCommand `
    -ExpectedEvidence $ownerInputTemplatePath `
    -ProofBoundary "The owner input template is a fillable contract only and remains non-proof until real external evidence validates."
  New-RunbookStep `
    -Id "fill-owner-input" `
    -Title "Copy and fill owner runtime smoke input" `
    -Command $copyOwnerInputTemplateCommand `
    -ExpectedEvidence $ownerInputRealInputPath `
    -ProofBoundary "The copied owner input must be filled with real public package source, clean external consumer paths, logs, hashes, host metadata, and stdout/stderr summaries."
  New-RunbookStep `
    -Id "validate-owner-input-strict" `
    -Title "Run strict owner input validator" `
    -Command $ownerInputStrictValidationCommand `
    -ExpectedEvidence "$ownerInputValidationPath; validationState=real-runtime-proof only after all owner fields are real" `
    -ProofBoundary "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict rejects templates, local feed, ProjectReference, direct .nupkg, missing hashes, and smoke placeholders."
  New-RunbookStep `
    -Id "import-owner-input-strict" `
    -Title "Import owner-filled runtime proof input" `
    -Command $ownerInputImportCommand `
    -ExpectedEvidence $ownerInputImportPath `
    -ProofBoundary "Importing owner input is still a validator bridge; it does not publish packages or replace external-runtime-proof-record validation."
  New-RunbookStep `
    -Id "refresh-release-evidence" `
    -Title "Refresh release evidence after proof validation" `
    -Command $refreshBundleCommand `
    -ExpectedEvidence "artifacts/final-release/release-evidence-bundle.json" `
    -ProofBoundary "Refreshing the bundle without a promotable proof keeps the release blocked."
)

$requiredRecordFields = @(
  New-RequiredRecordField -Path "recordKind" -ExpectedValue "external-runtime-proof-record" -Why "Distinguishes a real owner-filled record from draft/template/example records."
  New-RequiredRecordField -Path "templateOnly" -ExpectedValue "false" -Why "Templates cannot be promoted."
  New-RequiredRecordField -Path "proofClassification" -ExpectedValue "package-consumer-runtime" -Why "Only clean package consumer runtime proof can satisfy the release gate."
  New-RequiredRecordField -Path "runtimePackageKey" -ExpectedValue $RuntimePackageKey -Why "The proof must target the release runtime package key."
  New-RequiredRecordField -Path "packageSource.runtimePackageKey" -ExpectedValue $RuntimePackageKey -Why "The consumed runtime package must match the release target."
  New-RequiredRecordField -Path "packageSource.managedNupkgSha256" -ExpectedValue "64-character SHA256" -Why "The exact managed package consumed by the clean project must be auditable."
  New-RequiredRecordField -Path "packageSource.runtimeNupkgSha256" -ExpectedValue "64-character SHA256" -Why "The exact runtime package consumed by the clean project must be auditable."
  New-RequiredRecordField -Path "packageSource.noProjectReference" -ExpectedValue "true" -Why "The consumer must prove package consumption, not project reference execution."
  New-RequiredRecordField -Path "host.gpuName / host.nvidiaDriverVersion / host.cudaRuntimeVersion / host.tensorRtRuntimeVersion" -ExpectedValue "non-empty" -Why "The compatible host and runtime stack must be reviewable."
  New-RequiredRecordField -Path "command.exitCode" -ExpectedValue "0" -Why "The smoke command must have succeeded."
  New-RequiredRecordField -Path "command.logPath" -ExpectedValue $smokeLogPath -Why "The validator must be able to locate the preserved smoke log."
  New-RequiredRecordField -Path "command.logSha256" -ExpectedValue "64-character SHA256 matching the log" -Why "The smoke output must be tamper-evident."
  New-RequiredRecordField -Path "results.stdoutSummary" -ExpectedValue "non-empty reviewed stdout summary" -Why "The smoke output must be reviewed, not only hashed."
  New-RequiredRecordField -Path "results.stderrSummary" -ExpectedValue "non-empty reviewed stderr summary or no-stderr-emitted" -Why "The stderr channel must be accounted for explicitly."
  New-RequiredRecordField -Path "results.smokeStatus" -ExpectedValue "passed" -Why "blocked-by-cuda-driver is not smoke passed."
  New-RequiredRecordField -Path "results.nativeAssetsCopied" -ExpectedValue "true" -Why "The package must copy bridge/vendor runtime assets into the clean consumer output."
  New-RequiredRecordField -Path "isDependencyProbeOnly" -ExpectedValue "false" -Why "Dependency probes cannot satisfy runtime execution evidence."
  New-RequiredRecordField -Path "isRuntimeExecutionEvidence" -ExpectedValue "true" -Why "The real record must declare the proof boundary explicitly."
  New-RequiredRecordField -Path "canPromoteRuntimeProof" -ExpectedValue "true" -Why "The validator uses this as part of the promotion gate."
)

$requiredOwnerInputFields = @(
  "cleanExternalConsumerRoot",
  "consumerProjectPath",
  "publicPackageSource",
  "managedPackageId",
  "managedPackageVersion",
  "managedNupkgPath",
  "managedNupkgSha256",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "runtimeNupkgPath",
  "runtimeNupkgSha256",
  "ownerName",
  "machineName",
  "hostOs",
  "hostArchitecture",
  "gpuName",
  "cudaDriverVersion",
  "cudaDriverSupportedRuntime",
  "cudaRuntimeVersion",
  "cudnnVersion",
  "tensorRtVersion",
  "tensorRtLine",
  "restoreCommand",
  "buildCommand",
  "smokeCommand",
  "exitCode",
  "startedAtUtc",
  "finishedAtUtc",
  "dependencyProbeStatus",
  "smokeStatus",
  "nativeAssetsCopied",
  "smokeLogPath",
  "smokeLogSha256",
  "stdoutSummary",
  "stderrSummary"
)

$nonProofBoundaries = @(
  "This runbook does not publish packages.",
  "This runbook does not approve public release.",
  "external-runtime-proof-owner-handoff is a guide, not runtime proof.",
  "external-runtime-proof-record-draft is a draft, not runtime proof.",
  "template-only, input-template, and example-not-for-publication records are not runtime proof.",
  "build-only and dependency-probe-only records are not runtime execution proof.",
  "blocked-by-cuda-driver is not smoke passed.",
  "ready-needs-manual-approval is not public release approval.",
  "runtimePackageKeyMatches=true without consumed package hashes and matching log SHA256 remains owner-action-required.",
  "stdoutSummaryReady=false or stderrSummaryReady=false remains owner-action-required.",
  "real-model-runtime and synthetic-input-runtime do not replace package-consumer-runtime proof.",
  "release-candidate-package-inventory, release-package-proof-bundle, final-package-review-bundle, and local-feed-consumer-summary are owner inputs only and cannot promote runtime proof.",
  "package-consumer-runtime-proof-owner-input is an owner-filled contract and cannot replace Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof.",
  "Smoke=not-requested is not runtime smoke proof."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "compatible-host-runtime-proof-runbook"
  runbookState = $runbookState
  ownerRuntimeSmokeRunbookState = $ownerRuntimeSmokeRunbookState
  ownerRuntimeSmokeBlocker = "blocked-owner-compatible-host-runtime-smoke"
  runtimePackageKey = $RuntimePackageKey
  compatibleHostRequired = $compatibleHostRequired
  performsPublish = $false
  approvesPublicRelease = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $canPromoteRuntimeProof
  isRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
  promotionBlockedReason = $promotionBlockedReason
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofBlockerCategory = $runtimeProofBlockerCategory
  externalRuntimeProofState = $validationState
  externalRuntimeProofClassification = $proofClassification
  externalRuntimeProofFailedProofItemCount = $failedProofItemCount
  externalRuntimeProofDraftState = $draftProofState
  externalRuntimeProofDraftClassification = $draftProofClassification
  externalRuntimeProofDraftCanPromoteRuntimeProof = $draftCanPromoteRuntimeProof
  externalRuntimeProofDraftRuntimeExecutionEvidence = $draftRuntimeExecutionEvidence
  draftManagedNupkgSha256Ready = $draftManagedNupkgSha256Ready
  draftRuntimeNupkgSha256Ready = $draftRuntimeNupkgSha256Ready
  draftSmokeLogSha256Ready = $draftSmokeLogSha256Ready
  externalRuntimeProofStdoutSummaryReady = $validationStdoutSummaryReady
  externalRuntimeProofStderrSummaryReady = $validationStderrSummaryReady
  externalRuntimeProofStdoutStderrSummariesReady = $validationStdoutStderrSummariesReady
  draftNoProjectReference = $draftNoProjectReference
  draftSmokeStatus = $draftSmokeStatus
  ownerHandoffState = $ownerHandoffState
  ownerHandoffActionStatus = $ownerHandoffActionStatus
  releaseEvidenceBundleState = $bundleState
  releaseCandidatePackageInventoryState = $packageInventoryRecordKind
  releaseCandidatePackageInventoryPackageCount = $packageInventoryPackageCount
  releaseCandidatePackageInventorySha256Ready = $packageInventorySha256Ready
  releaseCandidatePackageInventoryCanUseAsPublicPackageProof = $packageInventoryCanUseAsPublicPackageProof
  releaseCandidatePackageInventoryCanCloseReleaseIssue = $packageInventoryCanCloseReleaseIssue
  releasePackageProofState = $releasePackageProofState
  releasePackageProofCanPromoteRuntimeProof = $releasePackageProofCanPromoteRuntimeProof
  releasePackageProofCanUseAsPublicPackageProof = $releasePackageProofCanUseAsPublicPackageProof
  releasePackageProofCanCloseReleaseIssue = $releasePackageProofCanCloseReleaseIssue
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  finalPackageReviewCanCloseReleaseIssue = $finalPackageReviewCanCloseReleaseIssue
  localFeedConsumerRunStatus = $localFeedConsumerRunStatus
  localFeedConsumerRuntimePackageKey = $localFeedConsumerRuntimePackageKey
  localFeedConsumerRestoreSourceMode = $localFeedConsumerRestoreSourceMode
  localFeedConsumerUsesProjectReference = $localFeedConsumerUsesProjectReference
  expectedArtifacts = [pscustomobject]@{
    inputTemplatePath = $inputTemplatePath
    realRecordPath = $realRecordPath
    smokeLogPath = $smokeLogPath
    packageConsumerSummaryPath = $consumerSummaryPath
    managedNupkgPath = $managedNupkgPath
    runtimeNupkgPath = $runtimeNupkgPath
    ownerInputTemplatePath = $ownerInputTemplatePath
    ownerInputRealInputPath = $ownerInputRealInputPath
    ownerInputValidationPath = $ownerInputValidationPath
    ownerInputImportPath = $ownerInputImportPath
    validationPath = "artifacts/final-release/external-runtime-proof-validation.json"
    releaseEvidenceBundlePath = "artifacts/final-release/release-evidence-bundle.json"
    releaseCandidatePackageInventoryPath = "artifacts/final-release/release-candidate-package-inventory.json"
    releasePackageProofBundlePath = "artifacts/final-release/release-package-proof-bundle.json"
    finalPackageReviewBundlePath = "artifacts/final-release/final-package-review-bundle.json"
    localFeedConsumerSummaryPath = "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json"
  }
  commands = [pscustomobject]@{
    generateInputTemplate = $generateTemplateCommand
    runPackageConsumerSmoke = $smokeCommand
    computeSmokeLogSha256 = $smokeLogHashCommand
    computeManagedNupkgSha256 = $managedPackageHashCommand
    computeRuntimeNupkgSha256 = $runtimePackageHashCommand
    copyInputTemplateToRealRecord = $copyRecordCommand
    validateFilledRecord = $validateCommand
    exportOwnerInputTemplate = $ownerInputTemplateCommand
    copyOwnerInputTemplateToRealInput = $copyOwnerInputTemplateCommand
    validateOwnerInputStrict = $ownerInputStrictValidationCommand
    importOwnerInputStrict = $ownerInputImportCommand
    refreshReleaseEvidenceBundle = $refreshBundleCommand
  }
  steps = @($steps)
  requiredRecordFields = @($requiredRecordFields)
  requiredOwnerInputFields = @($requiredOwnerInputFields)
  forbiddenRuntimeSmokeSubstitutes = @($forbiddenRuntimeSmokeSubstitutes)
  ownerInputArtifacts = @($ownerInputArtifacts)
  nonProofBoundaries = @($nonProofBoundaries)
  sourceEvidence = @(
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-record.draft.json",
    "artifacts/final-release/external-runtime-proof-owner-handoff.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-candidate-package-inventory.json",
    "artifacts/final-release/release-package-proof-bundle.json",
    "artifacts/final-release/final-package-review-bundle.json",
    "artifacts/final-release/final-release-dry-run-summary.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json"
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "compatible-host-runtime-proof-runbook.json"
$markdownPath = Join-Path $outputRoot "compatible-host-runtime-proof-runbook.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Compatible Host Runtime Proof Runbook")
$lines.Add("")
$lines.Add("- runbook state: ``$runbookState``")
$lines.Add("- owner runtime smoke runbook state: ``$ownerRuntimeSmokeRunbookState``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- compatible host required: ``$compatibleHostRequired``")
$lines.Add("- performs publish: ``false``")
$lines.Add("- approves public release: ``false``")
$lines.Add("- can publish publicly: ``false``")
$lines.Add("- can close release issue: ``false``")
$lines.Add("- can promote runtime proof: ``$canPromoteRuntimeProof``")
$lines.Add("- runtime execution evidence: ``$isRuntimeExecutionEvidence``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- external proof state: ``$validationState``")
$lines.Add("- external proof classification: ``$proofClassification``")
$lines.Add("- external proof failed proof item count: ``$failedProofItemCount``")
$lines.Add("- draft state: ``$draftProofState``")
$lines.Add("- draft managed nupkg SHA256 ready: ``$draftManagedNupkgSha256Ready``")
$lines.Add("- draft runtime nupkg SHA256 ready: ``$draftRuntimeNupkgSha256Ready``")
$lines.Add("- draft smoke log SHA256 ready: ``$draftSmokeLogSha256Ready``")
$lines.Add("- draft no ProjectReference: ``$draftNoProjectReference``")
$lines.Add("- draft smoke status: ``$draftSmokeStatus``")
$lines.Add("- owner handoff state: ``$ownerHandoffState``")
$lines.Add("- release evidence bundle state: ``$bundleState``")
$lines.Add("- release candidate package inventory: ``$packageInventoryRecordKind``; package count: ``$packageInventoryPackageCount``; SHA256 ready: ``$packageInventorySha256Ready``; public proof: ``$packageInventoryCanUseAsPublicPackageProof``")
$lines.Add("- release package proof bundle: ``$releasePackageProofState``; can promote runtime proof: ``$releasePackageProofCanPromoteRuntimeProof``")
$lines.Add("- final package review bundle: ``$finalPackageReviewState``; package count: ``$finalPackageReviewPackageCount``; public proof: ``$finalPackageReviewCanUseAsPublicPackageProof``")
$lines.Add("- local feed consumer summary: ``$localFeedConsumerRunStatus``; runtime package key: ``$localFeedConsumerRuntimePackageKey``; restore source mode: ``$localFeedConsumerRestoreSourceMode``; uses ProjectReference: ``$localFeedConsumerUsesProjectReference``")
$lines.Add("- promotion blocked reason: $promotionBlockedReason")
$lines.Add("")
$lines.Add("## Commands")
$lines.Add("")
$lines.Add('```powershell')
$lines.Add($generateTemplateCommand)
$lines.Add($smokeCommand)
$lines.Add($smokeLogHashCommand)
$lines.Add($managedPackageHashCommand)
$lines.Add($runtimePackageHashCommand)
$lines.Add($copyRecordCommand)
$lines.Add($validateCommand)
$lines.Add($ownerInputTemplateCommand)
$lines.Add($copyOwnerInputTemplateCommand)
$lines.Add($ownerInputStrictValidationCommand)
$lines.Add($ownerInputImportCommand)
$lines.Add($refreshBundleCommand)
$lines.Add('```')
$lines.Add("")
$lines.Add("## Execution Steps")
$lines.Add("")
$lines.Add("| ID | Title | Command | Expected evidence | Proof boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($step in $steps) {
  $lines.Add("| ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.title) | ``$(ConvertTo-MarkdownCell $step.command)`` | $(ConvertTo-MarkdownCell $step.expectedEvidence) | $(ConvertTo-MarkdownCell $step.proofBoundary) |")
}
$lines.Add("")
$lines.Add("## Required Record Fields")
$lines.Add("")
$lines.Add("| Field | Expected value | Why |")
$lines.Add("| --- | --- | --- |")
foreach ($field in $requiredRecordFields) {
  $lines.Add("| ``$($field.path)`` | ``$(ConvertTo-MarkdownCell $field.expectedValue)`` | $(ConvertTo-MarkdownCell $field.why) |")
}
$lines.Add("")
$lines.Add("## Required Owner Input Fields")
$lines.Add("")
foreach ($field in $requiredOwnerInputFields) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Forbidden Runtime Smoke Substitutes")
$lines.Add("")
foreach ($substitute in $forbiddenRuntimeSmokeSubstitutes) {
  $lines.Add("- ``$substitute``")
}
$lines.Add("")
$lines.Add("## Owner Input Artifacts")
$lines.Add("")
$lines.Add("| ID | Path | State | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($artifact in $ownerInputArtifacts) {
  $lines.Add("| ``$($artifact.id)`` | ``$($artifact.path)`` | $(ConvertTo-MarkdownCell $artifact.state) | $(ConvertTo-MarkdownCell $artifact.boundary) |")
}
$lines.Add("")
$lines.Add("## Non-Proof Boundaries")
$lines.Add("")
foreach ($boundary in $nonProofBoundaries) {
  $lines.Add("- $boundary")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Compatible host runtime proof runbook written to $jsonPath"
Write-Host "Compatible host runtime proof runbook written to $markdownPath"
Write-Host "RunbookState=$runbookState CompatibleHostRequired=$compatibleHostRequired RuntimeProofStatus=$runtimeProofStatus CanPromoteRuntimeProof=$canPromoteRuntimeProof IsRuntimeExecutionEvidence=$isRuntimeExecutionEvidence PromotionBlockedReason=$promotionBlockedReason"
