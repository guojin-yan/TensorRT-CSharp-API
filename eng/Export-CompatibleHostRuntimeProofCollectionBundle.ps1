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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-CollectionStep {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Command,
    [string]$ExpectedArtifact,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    command = $Command
    expectedArtifact = $ExpectedArtifact
    boundary = $Boundary
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

function New-PreflightChecklistItem {
  param(
    [string]$Id,
    [string]$Required,
    [string]$Why
  )

  [pscustomobject]@{
    id = $Id
    required = $Required
    why = $Why
  }
}

$runbook = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$externalRuntimeProofDraft = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record.draft.json"
$externalRuntimeOwnerHandoff = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-owner-handoff.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$releasePackageProof = Read-JsonOrNull "artifacts\final-release\release-package-proof-bundle.json"
$releaseCandidatePackageInventory = Read-JsonOrNull "artifacts\final-release\release-candidate-package-inventory.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"
$localFeedConsumer = Read-JsonOrNull "artifacts\local-feed-consumer\local-nuget-feed-consumer-summary.json"

$runbookCommands = Get-PropertyOrDefault -Object $runbook -Name "commands" -DefaultValue $null
$runbookExpectedArtifacts = Get-PropertyOrDefault -Object $runbook -Name "expectedArtifacts" -DefaultValue $null
$runbookState = [string](Get-PropertyOrDefault -Object $runbook -Name "runbookState" -DefaultValue "missing-compatible-host-runtime-proof-runbook")
$runbookCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $runbook -Name "canPromoteRuntimeProof" -DefaultValue $false)
$runbookRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $runbook -Name "isRuntimeExecutionEvidence" -DefaultValue $false)
$runbookPerformsPublish = [bool](Get-PropertyOrDefault -Object $runbook -Name "performsPublish" -DefaultValue $false)
$runbookApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $runbook -Name "approvesPublicRelease" -DefaultValue $false)
$runbookPromotionBlockedReason = [string](Get-PropertyOrDefault -Object $runbook -Name "promotionBlockedReason" -DefaultValue "missing-compatible-host-runbook-blocked-reason")
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $runbook -Name "runtimeProofStatus" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "runtimeProofStatus" -DefaultValue "missing-runtime-proof-status")))
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $runbook -Name "externalRuntimeProofState" -DefaultValue ([string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")))
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $runbook -Name "externalRuntimeProofClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")))
$externalRuntimeProofFailedProofItemCount = [int](Get-PropertyOrDefault -Object $runbook -Name "externalRuntimeProofFailedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)))
$packageConsumerSmokeStatus = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeResult" -DefaultValue "missing-package-consumer-smoke")
$packageConsumerEvidenceKind = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "EvidenceKind" -DefaultValue "missing-package-consumer-evidence-kind")
$packageConsumerRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $packageConsumer -Name "IsRuntimeExecutionEvidence" -DefaultValue $false)
$packageConsumerDependencyProbeOnly = [bool](Get-PropertyOrDefault -Object $packageConsumer -Name "IsDependencyProbeOnly" -DefaultValue $true)
$releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$releasePackageProofState = [string](Get-PropertyOrDefault -Object $releasePackageProof -Name "proofState" -DefaultValue "missing-release-package-proof-bundle")
$releasePackageProofCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releasePackageProof -Name "canPromoteRuntimeProof" -DefaultValue $false)
$releasePackageProofCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $releasePackageProof -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$packageInventoryRecordKind = [string](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "recordKind" -DefaultValue "missing-release-candidate-package-inventory")
$packageInventoryPackageCount = [int](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "packageCount" -DefaultValue 0)
$packageInventorySha256Ready = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "sha256Ready" -DefaultValue $false)
$packageInventoryCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "reviewState" -DefaultValue ([string](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageInventoryState" -DefaultValue "missing-final-package-review-bundle")))
$finalPackageReviewPackageCount = [int](Get-PropertyOrDefault -Object $finalPackageReview -Name "packageCount" -DefaultValue 0)
$finalPackageReviewCanUseAsPublicPackageProof = [bool](Get-PropertyOrDefault -Object $finalPackageReview -Name "canUseAsPublicPackageProof" -DefaultValue $false)
$localFeedConsumerRunStatus = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RunStatus" -DefaultValue "missing-local-feed-consumer-summary")
$localFeedConsumerRuntimePackageKey = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RuntimePackageKey" -DefaultValue "missing-runtime-package-key")
$localFeedConsumerRestoreSourceMode = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RestoreSourceMode" -DefaultValue "missing-restore-source-mode")
$localFeedConsumerUsesProjectReference = [bool](Get-PropertyOrDefault -Object $localFeedConsumer -Name "UsesProjectReference" -DefaultValue $true)

$inputTemplatePath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "inputTemplatePath" -DefaultValue "artifacts/final-release/external-runtime-proof-record.input-template.json")
$realRecordPath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "realRecordPath" -DefaultValue "artifacts/final-release/external-runtime-proof-record.json")
$smokeLogPath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "smokeLogPath" -DefaultValue "artifacts/final-release/external-runtime-proof/$RuntimePackageKey/package-consumer-smoke.log")
$packageConsumerSummaryPath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "packageConsumerSummaryPath" -DefaultValue "artifacts/package-consumer/package-consumer-validation-summary.json")
$validationPath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "validationPath" -DefaultValue "artifacts/final-release/external-runtime-proof-validation.json")

$exportRunbookCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofRunbook.ps1 -RuntimePackageKey $RuntimePackageKey"
$generateTemplateCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "generateInputTemplate" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey")
$smokeCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "runPackageConsumerSmoke" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -SmokeRuntimePackageKey $RuntimePackageKey -RunSmoke -KeepConsumerOutput")
$smokeLogHashCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "computeSmokeLogSha256" -DefaultValue "Get-FileHash -LiteralPath `"$smokeLogPath`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash")
$managedPackageHashCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "computeManagedNupkgSha256" -DefaultValue "Get-FileHash -LiteralPath `"<downloaded-managed-nupkg>`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash")
$runtimePackageHashCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "computeRuntimeNupkgSha256" -DefaultValue "Get-FileHash -LiteralPath `"<downloaded-runtime-nupkg>`" -Algorithm SHA256 | Select-Object -ExpandProperty Hash")
$copyRecordCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "copyInputTemplateToRealRecord" -DefaultValue "Copy-Item -LiteralPath $inputTemplatePath -Destination $realRecordPath")
$validateCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "validateFilledRecord" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath $realRecordPath -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof")
$ownerInputTemplatePath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "ownerInputTemplatePath" -DefaultValue "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json")
$ownerInputRealInputPath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "ownerInputRealInputPath" -DefaultValue "artifacts/final-release/package-consumer-runtime-proof-owner-input.json")
$ownerInputValidationPath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "ownerInputValidationPath" -DefaultValue "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json")
$ownerInputImportPath = [string](Get-PropertyOrDefault -Object $runbookExpectedArtifacts -Name "ownerInputImportPath" -DefaultValue "artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json")
$ownerInputTemplateCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "exportOwnerInputTemplate" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PackageConsumerRuntimeProofOwnerInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey")
$copyOwnerInputTemplateCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "copyOwnerInputTemplateToRealInput" -DefaultValue "Copy-Item -LiteralPath $ownerInputTemplatePath -Destination $ownerInputRealInputPath")
$ownerInputStrictValidationCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "validateOwnerInputStrict" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -InputPath $ownerInputRealInputPath -Strict")
$ownerInputImportCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "importOwnerInputStrict" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 -InputPath $ownerInputRealInputPath -Strict")
$refreshReleaseEvidenceCommand = [string](Get-PropertyOrDefault -Object $runbookCommands -Name "refreshReleaseEvidenceBundle" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1 -RuntimePackageKey $RuntimePackageKey")
$refreshOwnerFacingCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerApprovalInputTemplate.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerDecisionRecord.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePublishExecutionChecklist.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleasePromotionIssueRecord.ps1"
$ownerRuntimeSmokeRunbookState = [string](Get-PropertyOrDefault -Object $runbook -Name "ownerRuntimeSmokeRunbookState" -DefaultValue "blocked-owner-compatible-host-runtime-smoke")

$copyableExecutionOrder = @(
  $exportRunbookCommand,
  $generateTemplateCommand,
  $smokeCommand,
  $smokeLogHashCommand,
  $managedPackageHashCommand,
  $runtimePackageHashCommand,
  $copyRecordCommand,
  $validateCommand,
  $ownerInputTemplateCommand,
  $copyOwnerInputTemplateCommand,
  $ownerInputStrictValidationCommand,
  $ownerInputImportCommand,
  $refreshReleaseEvidenceCommand,
  $refreshOwnerFacingCommand
)

$operatorQuickStart = @(
  "Use a compatible NVIDIA driver / CUDA runtime / TensorRT runtime host for $RuntimePackageKey.",
  "Run the copyable execution order from the repository root without changing proof flags by hand.",
  "Keep the package consumer smoke log and copy its SHA256 into command.logSha256.",
  "Hash the consumed managed and runtime nupkg files from the source that the clean consumer restored.",
  "Copy the input template to external-runtime-proof-record.json, set templateOnly=false, and fill only real host/run/package evidence.",
  "Run the validator with -RequireExistingLog -FailOnNotProof; stop if it fails.",
  "Export package-consumer-runtime-proof-owner-input.template.json, copy it to the owner input path, fill real public package and compatible-host smoke fields, then run Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict.",
  "Import the owner input only after strict validation succeeds; do not use template-only input as proof.",
  "Refresh release evidence and owner-facing records only after the real record validates."
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

$preflightChecklist = @(
  New-PreflightChecklistItem -Id "compatible-gpu-host" -Required "NVIDIA driver can run CUDA/TensorRT runtime for $RuntimePackageKey" -Why "Current local blocker is driver/runtime mismatch; blocked-by-cuda-driver is not smoke passed."
  New-PreflightChecklistItem -Id "clean-package-consumer" -Required "Smoke command consumes nupkg packages without ProjectReference" -Why "The release gate requires package-consumer-runtime proof."
  New-PreflightChecklistItem -Id "preserved-smoke-log" -Required "Smoke log path exists and can be hashed" -Why "The validator requires an existing log and matching command.logSha256."
  New-PreflightChecklistItem -Id "managed-package-hash" -Required "Managed nupkg SHA256 is captured" -Why "The exact managed package consumed by the clean project must be auditable."
  New-PreflightChecklistItem -Id "runtime-package-hash" -Required "Runtime nupkg SHA256 is captured" -Why "The exact runtime package consumed by the clean project must match the runtime key."
  New-PreflightChecklistItem -Id "filled-real-record" -Required "external-runtime-proof-record.json has recordKind=external-runtime-proof-record and templateOnly=false" -Why "Templates, drafts, examples, and collection bundles are non-proof."
  New-PreflightChecklistItem -Id "proof-validator" -Required "Test-ExternalRuntimeProofRecord.ps1 passes with -RequireExistingLog -FailOnNotProof" -Why "Only this validation can promote package-consumer-runtime proof."
  New-PreflightChecklistItem -Id "owner-input-strict-validator" -Required "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict passes on the owner-filled input" -Why "Owner input must prove clean external consumer path, public package source, no ProjectReference, package/log SHA256, host metadata, and stdout/stderr summaries."
)

$collectionState = if ($runbookCanPromoteRuntimeProof -and $runbookRuntimeExecutionEvidence) {
  "runtime-proof-ready-but-collection-bundle-non-proof"
}
else {
  "owner-action-required"
}

$requiredRecordFieldsFromRunbook = @()
if ($runbook -and $runbook.PSObject.Properties.Name -contains "requiredRecordFields") {
  $requiredRecordFieldsFromRunbook = @($runbook.requiredRecordFields)
}

$requiredOwnerInputFieldsFromRunbook = @()
if ($runbook -and $runbook.PSObject.Properties.Name -contains "requiredOwnerInputFields") {
  $requiredOwnerInputFieldsFromRunbook = @($runbook.requiredOwnerInputFields)
}

if ($requiredRecordFieldsFromRunbook.Count -eq 0) {
  $requiredRecordFieldsFromRunbook = @(
    New-RequiredRecordField -Path "recordKind" -ExpectedValue "external-runtime-proof-record" -Why "Real proof must be an owner-filled record."
    New-RequiredRecordField -Path "templateOnly" -ExpectedValue "false" -Why "Template, draft, and example records cannot be promoted."
    New-RequiredRecordField -Path "proofClassification" -ExpectedValue "package-consumer-runtime" -Why "Only clean package consumer runtime proof can satisfy the release gate."
    New-RequiredRecordField -Path "runtimePackageKey" -ExpectedValue $RuntimePackageKey -Why "Proof must match the release runtime package key."
    New-RequiredRecordField -Path "packageSource.runtimePackageKey" -ExpectedValue $RuntimePackageKey -Why "The consumed runtime nupkg must match the target."
    New-RequiredRecordField -Path "command.exitCode" -ExpectedValue "0" -Why "The smoke command must succeed."
    New-RequiredRecordField -Path "command.logSha256" -ExpectedValue "64-character SHA256 matching the log" -Why "The smoke log must be traceable."
    New-RequiredRecordField -Path "results.stdoutSummary" -ExpectedValue "non-empty reviewed stdout summary" -Why "The smoke output must be reviewed, not only hashed."
    New-RequiredRecordField -Path "results.stderrSummary" -ExpectedValue "non-empty reviewed stderr summary or no-stderr-emitted" -Why "The stderr channel must be accounted for explicitly."
    New-RequiredRecordField -Path "results.smokeStatus" -ExpectedValue "passed" -Why "blocked-by-cuda-driver is not smoke passed."
    New-RequiredRecordField -Path "isRuntimeExecutionEvidence" -ExpectedValue "true" -Why "Real runtime execution evidence must be explicit."
    New-RequiredRecordField -Path "canPromoteRuntimeProof" -ExpectedValue "true" -Why "The validator must pass -FailOnNotProof."
  )
}

$collectionSteps = @(
  New-CollectionStep -Id "refresh-runbook" -Title "Refresh compatible-host runbook" -Command $exportRunbookCommand -ExpectedArtifact "artifacts/final-release/compatible-host-runtime-proof-runbook.json" -Boundary "Runbook refresh is still guidance, not runtime proof."
  New-CollectionStep -Id "refresh-input-template" -Title "Refresh external proof input template" -Command $generateTemplateCommand -ExpectedArtifact $inputTemplatePath -Boundary "Input template remains templateOnly=true until copied and filled."
  New-CollectionStep -Id "run-package-consumer-smoke" -Title "Run clean package consumer smoke on compatible host" -Command $smokeCommand -ExpectedArtifact "$packageConsumerSummaryPath; $smokeLogPath" -Boundary "blocked-by-cuda-driver and dependency-probe-only are not smoke passed."
  New-CollectionStep -Id "hash-smoke-log" -Title "Hash preserved smoke log" -Command $smokeLogHashCommand -ExpectedArtifact "command.logSha256" -Boundary "A log path without a matching SHA256 cannot promote runtime proof."
  New-CollectionStep -Id "hash-managed-package" -Title "Hash consumed managed nupkg" -Command $managedPackageHashCommand -ExpectedArtifact "packageSource.managedNupkgSha256" -Boundary "Unchecked local artifacts are not package proof."
  New-CollectionStep -Id "hash-runtime-package" -Title "Hash consumed runtime nupkg" -Command $runtimePackageHashCommand -ExpectedArtifact "packageSource.runtimeNupkgSha256" -Boundary "Runtime package hash must match the exact runtime package key."
  New-CollectionStep -Id "review-stdout-stderr" -Title "Review stdout/stderr summaries" -Command "Review $smokeLogPath and fill results.stdoutSummary plus results.stderrSummary; use no-stderr-emitted only when stderr is empty." -ExpectedArtifact "results.stdoutSummary; results.stderrSummary" -Boundary "A log hash without reviewed stdout/stderr summaries cannot promote runtime proof."
  New-CollectionStep -Id "fill-real-record" -Title "Copy template and fill real proof record" -Command $copyRecordCommand -ExpectedArtifact $realRecordPath -Boundary "Only a reviewed recordKind=external-runtime-proof-record, templateOnly=false record can be proof."
  New-CollectionStep -Id "validate-filled-record" -Title "Validate existing log and proof promotion gate" -Command $validateCommand -ExpectedArtifact $validationPath -Boundary "-FailOnNotProof must pass before release evidence can treat the record as proof."
  New-CollectionStep -Id "export-owner-input-template" -Title "Export package consumer runtime proof owner input template" -Command $ownerInputTemplateCommand -ExpectedArtifact $ownerInputTemplatePath -Boundary "Owner input templates are fillable contracts, not runtime proof."
  New-CollectionStep -Id "fill-owner-input" -Title "Copy and fill owner runtime smoke input" -Command $copyOwnerInputTemplateCommand -ExpectedArtifact $ownerInputRealInputPath -Boundary "Owner input must be filled with real public package source, clean external consumer paths, logs, hashes, host metadata, and stdout/stderr summaries."
  New-CollectionStep -Id "validate-owner-input-strict" -Title "Validate owner input strictly" -Command $ownerInputStrictValidationCommand -ExpectedArtifact $ownerInputValidationPath -Boundary "Strict owner input validation rejects local feed, ProjectReference, direct .nupkg, Smoke=not-requested, dependency-probe-only, and blocked-by-cuda-driver substitutes."
  New-CollectionStep -Id "import-owner-input-strict" -Title "Import owner-filled runtime proof input" -Command $ownerInputImportCommand -ExpectedArtifact $ownerInputImportPath -Boundary "Owner input import is a validator bridge only; it does not publish or replace external-runtime-proof-record validation."
  New-CollectionStep -Id "refresh-release-evidence" -Title "Refresh release and owner-facing evidence" -Command "$refreshReleaseEvidenceCommand; $refreshOwnerFacingCommand" -ExpectedArtifact "release evidence, owner approval, decision, checklist, and promotion issue records" -Boundary "Refreshing records without promotable proof keeps release blocked."
)

$safetyNotes = @(
  "compatible-host-runtime-proof-collection-bundle is an external execution collection bundle, not runtime proof.",
  "This collection bundle does not publish packages.",
  "This collection bundle does not approve public release.",
  "This collection bundle keeps canPromoteRuntimeProof=false even when it lists commands.",
  "Only external-runtime-proof-record.json validated with -RequireExistingLog -FailOnNotProof can promote package-consumer-runtime proof.",
  "stdoutSummary and stderrSummary must both be reviewed; no-stderr-emitted is acceptable only when stderr is actually empty.",
  "blocked-by-cuda-driver is not smoke passed.",
  "dependency-probe-only and build-only package consumer outputs are not runtime execution proof.",
  "Smoke=not-requested is not runtime smoke proof.",
  "Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict must pass on real owner input before package-consumer-runtime owner input can be imported.",
  "runbook, handoff, checklist, template, draft, example, and collection bundle artifacts remain non-proof until a real filled record validates."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "compatible-host-runtime-proof-collection-bundle"
  collectionState = $collectionState
  runtimePackageKey = $RuntimePackageKey
  compatibleHostRequired = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionEvidence = $false
  runbookState = $runbookState
  ownerRuntimeSmokeRunbookState = $ownerRuntimeSmokeRunbookState
  runbookCanPromoteRuntimeProof = $runbookCanPromoteRuntimeProof
  runbookRuntimeExecutionEvidence = $runbookRuntimeExecutionEvidence
  runbookPerformsPublish = $runbookPerformsPublish
  runbookApprovesPublicRelease = $runbookApprovesPublicRelease
  runtimeProofStatus = $runtimeProofStatus
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
  externalRuntimeProofFailedProofItemCount = $externalRuntimeProofFailedProofItemCount
  packageConsumerSmokeStatus = $packageConsumerSmokeStatus
  packageConsumerEvidenceKind = $packageConsumerEvidenceKind
  packageConsumerRuntimeExecutionEvidence = $packageConsumerRuntimeExecutionEvidence
  packageConsumerDependencyProbeOnly = $packageConsumerDependencyProbeOnly
  releaseEvidenceBundleState = $releaseEvidenceBundleState
  releasePackageProofState = $releasePackageProofState
  releasePackageProofCanPromoteRuntimeProof = $releasePackageProofCanPromoteRuntimeProof
  releasePackageProofCanUseAsPublicPackageProof = $releasePackageProofCanUseAsPublicPackageProof
  releaseCandidatePackageInventoryState = $packageInventoryRecordKind
  releaseCandidatePackageInventoryPackageCount = $packageInventoryPackageCount
  releaseCandidatePackageInventorySha256Ready = $packageInventorySha256Ready
  releaseCandidatePackageInventoryCanUseAsPublicPackageProof = $packageInventoryCanUseAsPublicPackageProof
  finalPackageReviewState = $finalPackageReviewState
  finalPackageReviewPackageCount = $finalPackageReviewPackageCount
  finalPackageReviewCanUseAsPublicPackageProof = $finalPackageReviewCanUseAsPublicPackageProof
  localFeedConsumerRunStatus = $localFeedConsumerRunStatus
  localFeedConsumerRuntimePackageKey = $localFeedConsumerRuntimePackageKey
  localFeedConsumerRestoreSourceMode = $localFeedConsumerRestoreSourceMode
  localFeedConsumerUsesProjectReference = $localFeedConsumerUsesProjectReference
  promotionBlockedReason = $runbookPromotionBlockedReason
  expectedArtifacts = [pscustomobject]@{
    inputTemplatePath = $inputTemplatePath
    realRecordPath = $realRecordPath
    smokeLogPath = $smokeLogPath
    packageConsumerSummaryPath = $packageConsumerSummaryPath
    validationPath = $validationPath
    ownerInputTemplatePath = $ownerInputTemplatePath
    ownerInputRealInputPath = $ownerInputRealInputPath
    ownerInputValidationPath = $ownerInputValidationPath
    ownerInputImportPath = $ownerInputImportPath
    releaseCandidatePackageInventoryPath = "artifacts/final-release/release-candidate-package-inventory.json"
    releasePackageProofBundlePath = "artifacts/final-release/release-package-proof-bundle.json"
    finalPackageReviewBundlePath = "artifacts/final-release/final-package-review-bundle.json"
    localFeedConsumerSummaryPath = "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json"
    collectionBundleJsonPath = "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json"
    collectionBundleMarkdownPath = "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md"
  }
  commands = [pscustomobject]@{
    exportCompatibleHostRuntimeProofRunbook = $exportRunbookCommand
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
    refreshReleaseEvidenceBundle = $refreshReleaseEvidenceCommand
    refreshOwnerFacingRecords = $refreshOwnerFacingCommand
  }
  operatorQuickStart = @($operatorQuickStart)
  preflightChecklist = @($preflightChecklist)
  copyableExecutionOrder = @($copyableExecutionOrder)
  collectionSteps = @($collectionSteps)
  requiredRecordFields = @($requiredRecordFieldsFromRunbook)
  requiredOwnerInputFields = @($requiredOwnerInputFieldsFromRunbook)
  ownerInputArtifacts = @($ownerInputArtifacts)
  sourceEvidence = @(
    "artifacts/final-release/compatible-host-runtime-proof-runbook.json",
    "artifacts/final-release/external-runtime-proof-record.input-template.json",
    "artifacts/final-release/external-runtime-proof-record.draft.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/external-runtime-proof-owner-handoff.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-candidate-package-inventory.json",
    "artifacts/final-release/release-package-proof-bundle.json",
    "artifacts/final-release/final-package-review-bundle.json"
  )
  safetyNotes = @($safetyNotes)
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "compatible-host-runtime-proof-collection-bundle.json"
$markdownPath = Join-Path $outputRoot "compatible-host-runtime-proof-collection-bundle.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Compatible Host Runtime Proof Collection Bundle")
$lines.Add("")
$lines.Add("- collection state: ``$collectionState``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- compatible host required: ``true``")
$lines.Add("- performs publish: ``false``")
$lines.Add("- approves public release: ``false``")
$lines.Add("- can publish publicly: ``false``")
$lines.Add("- can close release issue: ``false``")
$lines.Add("- can promote runtime proof: ``false``")
$lines.Add("- runtime execution evidence: ``false``")
$lines.Add("- runbook state: ``$runbookState``")
$lines.Add("- owner runtime smoke runbook state: ``$ownerRuntimeSmokeRunbookState``")
$lines.Add("- runbook can promote runtime proof: ``$runbookCanPromoteRuntimeProof``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- external proof state: ``$externalRuntimeProofState``")
$lines.Add("- external proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- package consumer smoke status: ``$packageConsumerSmokeStatus``")
$lines.Add("- package consumer evidence kind: ``$packageConsumerEvidenceKind``")
$lines.Add("- release candidate package inventory: ``$packageInventoryRecordKind``; package count: ``$packageInventoryPackageCount``; SHA256 ready: ``$packageInventorySha256Ready``; public proof: ``$packageInventoryCanUseAsPublicPackageProof``")
$lines.Add("- release package proof bundle: ``$releasePackageProofState``; can promote runtime proof: ``$releasePackageProofCanPromoteRuntimeProof``")
$lines.Add("- final package review bundle: ``$finalPackageReviewState``; package count: ``$finalPackageReviewPackageCount``; public proof: ``$finalPackageReviewCanUseAsPublicPackageProof``")
$lines.Add("- local feed consumer summary: ``$localFeedConsumerRunStatus``; runtime package key: ``$localFeedConsumerRuntimePackageKey``; restore source mode: ``$localFeedConsumerRestoreSourceMode``; uses ProjectReference: ``$localFeedConsumerUsesProjectReference``")
$lines.Add("- promotion blocked reason: $runbookPromotionBlockedReason")
$lines.Add("")
$lines.Add("## One-Pass External Host Commands")
$lines.Add("")
$lines.Add("Run these commands from the repository root on a compatible CUDA/TensorRT host. Stop on the first failure; do not rewrite a failed smoke or validator result as proof.")
$lines.Add("")
$lines.Add('```powershell')
foreach ($command in $copyableExecutionOrder) {
  $lines.Add($command)
}
$lines.Add('```')
$lines.Add("")
$lines.Add("## Operator Quick Start")
$lines.Add("")
foreach ($item in $operatorQuickStart) {
  $lines.Add("- $item")
}
$lines.Add("")
$lines.Add("## Preflight Checklist")
$lines.Add("")
$lines.Add("| ID | Required input | Why |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $preflightChecklist) {
  $lines.Add("| ``$($item.id)`` | $(ConvertTo-MarkdownCell $item.required) | $(ConvertTo-MarkdownCell $item.why) |")
}
$lines.Add("")
$lines.Add("## Collection Steps")
$lines.Add("")
$lines.Add("| ID | Title | Command | Expected artifact | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($step in $collectionSteps) {
  $lines.Add("| ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.title) | ``$(ConvertTo-MarkdownCell $step.command)`` | $(ConvertTo-MarkdownCell $step.expectedArtifact) | $(ConvertTo-MarkdownCell $step.boundary) |")
}
$lines.Add("")
$lines.Add("## Required Real Record Fields")
$lines.Add("")
$lines.Add("| Field | Expected value | Why |")
$lines.Add("| --- | --- | --- |")
foreach ($field in $requiredRecordFieldsFromRunbook) {
  $lines.Add("| ``$($field.path)`` | ``$(ConvertTo-MarkdownCell $field.expectedValue)`` | $(ConvertTo-MarkdownCell $field.why) |")
}
$lines.Add("")
$lines.Add("## Required Owner Input Fields")
$lines.Add("")
foreach ($field in $requiredOwnerInputFieldsFromRunbook) {
  $lines.Add("- ``$field``")
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
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Compatible host runtime proof collection bundle written to $jsonPath"
Write-Host "Compatible host runtime proof collection bundle written to $markdownPath"
Write-Host "CollectionState=$collectionState RuntimeProofStatus=$runtimeProofStatus CanPromoteRuntimeProof=False IsRuntimeExecutionEvidence=False PerformsPublish=False ApprovesPublicRelease=False"
