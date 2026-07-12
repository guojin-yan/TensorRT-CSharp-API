[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
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

function New-DecisionItem {
  param(
    [string]$Id,
    [string]$Title,
    [string]$CurrentStatus,
    [string]$RequiredOwnerDecision,
    [string]$Evidence,
    [string]$CannotClaim
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    currentStatus = $CurrentStatus
    defaultDecision = "pending-release-owner-approval"
    requiredOwnerDecision = $RequiredOwnerDecision
    evidence = $Evidence
    cannotClaim = $CannotClaim
  }
}

$finalRelease = Read-JsonOrNull "artifacts\final-release\final-release-dry-run-summary.json"
$releaseChecklist = Read-JsonOrNull "artifacts\release-candidate\release-candidate-checklist.json"
$bilingualAudit = Read-JsonOrNull "artifacts\api-doc-audit\public-api-bilingual-documentation-audit.json"
$linuxStatus = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-execution-status.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$externalRuntimeProof = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$compatibleHostRunbook = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-runbook.json"
$compatibleHostCollectionBundle = Read-JsonOrNull "artifacts\final-release\compatible-host-runtime-proof-collection-bundle.json"
$postPublish = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$overallStatus = if ($finalRelease) { [string]$finalRelease.overallStatus } else { "missing-final-release-dry-run" }
$blockingIssueCount = if ($finalRelease) { [int]$finalRelease.blockingIssueCount } else { -1 }
$manualApprovalCount = if ($finalRelease) { [int]$finalRelease.manualApprovalCount } else { -1 }
$warningCount = if ($finalRelease) { [int]$finalRelease.warningCount } else { -1 }
$smokeStatus = if ($finalRelease) { [string]$finalRelease.packageConsumerSmokeStatus } else { "missing" }
$runtimeProofStatus = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofStatus" -and -not [string]::IsNullOrWhiteSpace([string]$finalRelease.runtimeProofStatus)) { [string]$finalRelease.runtimeProofStatus } else { $smokeStatus }
$runtimeProofRequiredForRelease = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "runtimeProofRequiredForRelease") { [bool]$finalRelease.runtimeProofRequiredForRelease } else { -not [string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase) }
$allowRuntimeSmokeBlocked = if ($finalRelease -and $finalRelease.PSObject.Properties.Name -contains "allowRuntimeSmokeBlocked") { [bool]$finalRelease.allowRuntimeSmokeBlocked } else { $false }
$runtimeProofBlockerOwnerActionStatus = if ([string]::Equals($runtimeProofStatus, "ready", [System.StringComparison]::OrdinalIgnoreCase)) { "resolved" } else { "owner-action-required" }
$runtimeProofBlockerCategory = switch ($runtimeProofStatus) {
  "ready" { "none"; break }
  "blocked-by-cuda-driver" { "cuda-driver-runtime-compatibility"; break }
  "blocked-by-application-control" { "application-control-policy"; break }
  "not-requested" { "runtime-smoke-not-requested"; break }
  default { "runtime-proof-incomplete"; break }
}
$runtimeProofOwnerCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -RunSmoke -AllowSmokeFailure"
$compatibleHostRunbookCommands = Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "commands" -DefaultValue $null
$compatibleHostRunbookState = [string](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "runbookState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookState" -DefaultValue "missing-compatible-host-runtime-proof-runbook")))
$compatibleHostRunbookCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "compatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookCompatibleHostRequired" -DefaultValue $true)))
$compatibleHostRunbookPerformsPublish = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookPerformsPublish" -DefaultValue $false)))
$compatibleHostRunbookApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "approvesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookApprovesPublicRelease" -DefaultValue $false)))
$compatibleHostRunbookCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostRunbookRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostRunbookPromotionBlockedReason = [string](Get-PropertyOrDefault -Object $compatibleHostRunbook -Name "promotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookPromotionBlockedReason" -DefaultValue "blocked-by-cuda-driver is not smoke passed")))
$compatibleHostRunbookRunPackageConsumerSmokeCommand = [string](Get-PropertyOrDefault -Object $compatibleHostRunbookCommands -Name "runPackageConsumerSmoke" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookRunPackageConsumerSmokeCommand" -DefaultValue $runtimeProofOwnerCommand)))
$compatibleHostRunbookValidateFilledRecordCommand = [string](Get-PropertyOrDefault -Object $compatibleHostRunbookCommands -Name "validateFilledRecord" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofRunbookValidateFilledRecordCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof")))
$compatibleHostCollectionBundleCommands = Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "commands" -DefaultValue $null
$compatibleHostCollectionBundleState = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "collectionState" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleState" -DefaultValue "missing-compatible-host-runtime-proof-collection-bundle")))
$compatibleHostCollectionBundleCompatibleHostRequired = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "compatibleHostRequired" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired" -DefaultValue $true)))
$compatibleHostCollectionBundlePerformsPublish = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "performsPublish" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePerformsPublish" -DefaultValue $false)))
$compatibleHostCollectionBundleApprovesPublicRelease = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "approvesPublicRelease" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease" -DefaultValue $false)))
$compatibleHostCollectionBundleCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof" -DefaultValue $false)))
$compatibleHostCollectionBundleRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "isRuntimeExecutionEvidence" -DefaultValue ([bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence" -DefaultValue $false)))
$compatibleHostCollectionBundlePromotionBlockedReason = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundle -Name "promotionBlockedReason" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason" -DefaultValue "blocked-by-cuda-driver is not smoke passed")))
$compatibleHostCollectionBundleRunPackageConsumerSmokeCommand = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundleCommands -Name "runPackageConsumerSmoke" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand" -DefaultValue $compatibleHostRunbookRunPackageConsumerSmokeCommand)))
$compatibleHostCollectionBundleValidateFilledRecordCommand = [string](Get-PropertyOrDefault -Object $compatibleHostCollectionBundleCommands -Name "validateFilledRecord" -DefaultValue ([string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand" -DefaultValue $compatibleHostRunbookValidateFilledRecordCommand)))
$externalRuntimeProofStateFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofState" -DefaultValue "missing-external-runtime-proof-record-template") } else { "missing-external-runtime-proof-validation" }
$externalRuntimeProofClassificationFallback = if ($externalRuntimeProofValidation) { [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification") } elseif ($externalRuntimeProof) { [string](Get-PropertyOrDefault -Object $externalRuntimeProof -Name "proofClassification" -DefaultValue "missing-proof-classification") } else { "missing-proof-classification" }
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofState" -DefaultValue $externalRuntimeProofStateFallback)
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofClassification" -DefaultValue $externalRuntimeProofClassificationFallback)
$externalRuntimeProofRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofPackageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofPackageSourceRuntimePackageKeyMatches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)))
$externalRuntimeProofConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$externalRuntimeProofSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$externalRuntimeProofHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)))
$externalRuntimeProofCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)))
$externalRuntimeProofManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$externalRuntimeProofLogSha256FormatReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256FormatReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256FormatReady" -DefaultValue $false)))
$externalRuntimeProofLogSha256Matches = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofLogSha256Matches" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "logSha256Matches" -DefaultValue $false)))
$externalRuntimeProofFailedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "externalRuntimeProofFailedProofItemCount" -DefaultValue ([int](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "failedProofItemCount" -DefaultValue -1)))
$externalRuntimeProofCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPromoteRuntimeProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)))
$externalRuntimeExecutionEvidence = $externalRuntimeProofCanPromoteRuntimeProof
$externalRuntimeProofOwnerActionStatus = if ($externalRuntimeProofCanPromoteRuntimeProof) { "resolved" } else { "owner-action-required" }
$postPublishVerificationState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishVerificationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublish -Name "validationState" -DefaultValue "missing-post-publish-validation")))
$postPublishProofClassification = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassification" -DefaultValue ([string](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")))
$postPublishProofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishProofClassificationPromotable" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassificationPromotable" -DefaultValue $false)))
$postPublishManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishManagedNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "managedNupkgSha256Ready" -DefaultValue $false)))
$postPublishRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRuntimeNupkgSha256Ready" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "runtimeNupkgSha256Ready" -DefaultValue $false)))
$postPublishConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishConsumerProjectIdentityReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "consumerProjectIdentityReady" -DefaultValue $false)))
$postPublishSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishSmokeCommandRuntimeKeyReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)))
$postPublishHostReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishHostReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "hostReady" -DefaultValue $false)))
$postPublishCommandsReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishCommandsReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "commandsReady" -DefaultValue $false)))
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishStdoutStderrSummaryReady" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutStderrSummaryReady" -DefaultValue $false)))
$isPostPublishVerificationProof = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "isPostPublishVerificationProof" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "isPostPublishVerificationProof" -DefaultValue $false)))
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue ([bool](Get-PropertyOrDefault -Object $postPublish -Name "canCloseReleaseIssue" -DefaultValue $false)))
$signingStatus = if ($finalRelease) { [string]$finalRelease.signingStatus } else { "missing" }
$callbackProof = if ($finalRelease) { [bool]$finalRelease.realCallbackRuntimeProof } else { $false }
$bilingualFindingCount = if ($bilingualAudit) { [int]$bilingualAudit.findingCount } else { -1 }
$linuxProofState = if ($linuxStatus -and [bool]$linuxStatus.isLinux -and [string]$linuxStatus.status -eq "linux-runner-ready-for-real-validation") { "runner-ready-but-proof-not-recorded" } elseif ($linuxStatus) { "handoff-only" } else { "missing-handoff" }

$decisions = @(
  New-DecisionItem `
    -Id "signing-trust" `
    -Title "Signing and trust" `
    -CurrentStatus $signingStatus `
    -RequiredOwnerDecision "Decide whether this RC may remain unsigned, or provide package/AuthentiCode signing evidence before public publication." `
    -Evidence "artifacts/final-release/final-release-dry-run-summary.json; docs/articles/zh-cn/signing-and-trust-policy.md" `
    -CannotClaim "Do not treat unsigned-or-not-requested as signed release output."
  New-DecisionItem `
    -Id "release-channel" `
    -Title "Release channel" `
    -CurrentStatus "undecided" `
    -RequiredOwnerDecision "Choose nuget.org, GitHub Packages, private feed, GitHub Release assets, or a staged combination with rollback rules." `
    -Evidence "docs/articles/zh-cn/nuget-and-github-packages-release-guide.md" `
    -CannotClaim "Do not claim a public release until the chosen channel is actually pushed and verified."
  New-DecisionItem `
    -Id "nvidia-redistribution" `
    -Title "NVIDIA redistribution" `
    -CurrentStatus "pending-legal-or-owner-review" `
    -RequiredOwnerDecision "Confirm CUDA, cuDNN, and TensorRT redistribution terms, package size, and allowed hosting channel." `
    -Evidence "docs/articles/zh-cn/signing-and-trust-policy.md; pack/runtime/runtime-packages.manifest.json" `
    -CannotClaim "Do not publish NVIDIA runtime components publicly without redistribution approval."
  New-DecisionItem `
    -Id "linux-runner-proof" `
    -Title "Linux runner proof" `
    -CurrentStatus $linuxProofState `
    -RequiredOwnerDecision "Assign a Linux x64 runner and require CMake build, runtime asset collection, runtime nupkg, and package consumer evidence." `
    -Evidence "artifacts/linux-dry-run/$LinuxRuntimePackageKey" `
    -CannotClaim "Do not treat Windows-generated Linux handoff as real Linux runner proof."
  New-DecisionItem `
    -Id "runtime-smoke" `
    -Title "Full package runtime smoke" `
    -CurrentStatus ("packageConsumerSmokeStatus=" + $smokeStatus + "; runtimeProofStatus=" + $runtimeProofStatus + "; runtimeProofRequiredForRelease=" + $runtimeProofRequiredForRelease + "; allowRuntimeSmokeBlocked=" + $allowRuntimeSmokeBlocked + "; ownerAction=" + $runtimeProofBlockerOwnerActionStatus + "; blockerCategory=" + $runtimeProofBlockerCategory + "; compatibleHostRunbookState=" + $compatibleHostRunbookState + "; compatibleHostCollectionBundleState=" + $compatibleHostCollectionBundleState + "; externalRuntimeProofState=" + $externalRuntimeProofState + "; externalRuntimeProofClassification=" + $externalRuntimeProofClassification + "; externalRuntimeProofRuntimePackageKeyMatches=" + $externalRuntimeProofRuntimePackageKeyMatches + "; externalRuntimeProofPackageSourceRuntimePackageKeyMatches=" + $externalRuntimeProofPackageSourceRuntimePackageKeyMatches + "; externalRuntimeProofManagedNupkgSha256Ready=" + $externalRuntimeProofManagedNupkgSha256Ready + "; externalRuntimeProofRuntimeNupkgSha256Ready=" + $externalRuntimeProofRuntimeNupkgSha256Ready + "; externalRuntimeProofLogSha256FormatReady=" + $externalRuntimeProofLogSha256FormatReady + "; externalRuntimeProofLogSha256Matches=" + $externalRuntimeProofLogSha256Matches + "; externalRuntimeProofFailedProofItemCount=" + $externalRuntimeProofFailedProofItemCount + "; externalRuntimeProofOwnerAction=" + $externalRuntimeProofOwnerActionStatus) `
    -RequiredOwnerDecision "Accept the current environment blocker for RC documentation or rerun on a CUDA-compatible machine before public promotion. External proof must match runtimePackageKey, packageSource.runtimePackageKey, clean consumerProjectName/consumerProjectPath, complete CUDA/TensorRT/cuDNN host metadata, downloaded managed/runtime nupkg SHA256, smokeCommand with --runtime-package-key, and include a validated smoke logSha256. Follow compatible-host-runtime-proof-runbook.json or compatible-host-runtime-proof-collection-bundle.json and run: $compatibleHostCollectionBundleRunPackageConsumerSmokeCommand" `
    -Evidence "artifacts/package-readiness/runtime-package-readiness-summary.json; artifacts/package-consumer/package-consumer-validation-summary.json; artifacts/final-release/external-runtime-proof-validation.json; artifacts/final-release/compatible-host-runtime-proof-runbook.json; artifacts/final-release/compatible-host-runtime-proof-runbook.md; artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json; artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md" `
    -CannotClaim "Do not treat blocked-by-cuda-driver, dependency-probe-only, runtime-deserialization-dependency-diagnostics, runtime-key-mismatched, package-source-key-mismatched, missing-package-hash, missing-log-hash, compatible-host-runtime-proof-runbook, compatible-host-runtime-proof-collection-bundle, runtimeProofRequiredForRelease=true, or allowRuntimeSmokeBlocked=true as smoke passed or API proof."
  New-DecisionItem `
    -Id "post-publish-verification" `
    -Title "Post-publish verification" `
    -CurrentStatus ("verificationState=" + $postPublishVerificationState + "; classification=" + $postPublishProofClassification + "; promotable=" + $postPublishProofClassificationPromotable + "; managedNupkgSha256Ready=" + $postPublishManagedNupkgSha256Ready + "; runtimeNupkgSha256Ready=" + $postPublishRuntimeNupkgSha256Ready + "; consumerProjectIdentityReady=" + $postPublishConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $postPublishSmokeCommandRuntimeKeyReady + "; hostReady=" + $postPublishHostReady + "; commandsReady=" + $postPublishCommandsReady + "; stdoutStderrSummaryReady=" + $postPublishStdoutStderrSummaryReady + "; isProof=" + $isPostPublishVerificationProof + "; canCloseReleaseIssue=" + $canCloseReleaseIssue) `
    -RequiredOwnerDecision "After a real channel push, attach a validated post-publish verification record before closing the release issue." `
    -Evidence "artifacts/final-release/post-publish-verification-validation.json; docs/articles/zh-cn/post-publish-verification-record.md" `
    -CannotClaim "Do not treat template-only, draft, missing package hashes, missing clean consumer identity, missing host metadata, missing command capture, missing --runtime-package-key smoke command, or missing stdout/stderr summaries as post-publish verified."
  New-DecisionItem `
    -Id "callback-proof" `
    -Title "Real callback runtime proof" `
    -CurrentStatus ("realCallbackRuntimeProof=" + $callbackProof) `
    -RequiredOwnerDecision "Keep callback proof false for this RC, or require full package consumer evidence with InvocationCount>0 before promotion." `
    -Evidence "artifacts/release-candidate/release-candidate-readiness-summary.json; docs/articles/zh-cn/real-callback-runtime-evidence-schema.md" `
    -CannotClaim "Do not treat precheck, schema-ready, design gate, or InvocationCount=0 as real callback runtime proof."
)

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  templateOnly = $true
  approvalState = "pending-release-owner-approval"
  canPublishPublicly = $false
  overallStatus = $overallStatus
  blockingIssueCount = $blockingIssueCount
  manualApprovalCount = $manualApprovalCount
  warningCount = $warningCount
  bilingualDocumentationFindingCount = $bilingualFindingCount
  packageConsumerSmokeStatus = $smokeStatus
  runtimeProofStatus = $runtimeProofStatus
  runtimeProofRequiredForRelease = $runtimeProofRequiredForRelease
  allowRuntimeSmokeBlocked = $allowRuntimeSmokeBlocked
  runtimeProofBlockerOwnerActionStatus = $runtimeProofBlockerOwnerActionStatus
  runtimeProofBlockerCategory = $runtimeProofBlockerCategory
  runtimeProofOwnerCommand = $runtimeProofOwnerCommand
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
  externalRuntimeProofRuntimePackageKeyMatches = $externalRuntimeProofRuntimePackageKeyMatches
  externalRuntimeProofPackageSourceRuntimePackageKeyMatches = $externalRuntimeProofPackageSourceRuntimePackageKeyMatches
  externalRuntimeProofConsumerProjectIdentityReady = $externalRuntimeProofConsumerProjectIdentityReady
  externalRuntimeProofSmokeCommandRuntimeKeyReady = $externalRuntimeProofSmokeCommandRuntimeKeyReady
  externalRuntimeProofHostReady = $externalRuntimeProofHostReady
  externalRuntimeProofCommandsReady = $externalRuntimeProofCommandsReady
  externalRuntimeProofManagedNupkgSha256Ready = $externalRuntimeProofManagedNupkgSha256Ready
  externalRuntimeProofRuntimeNupkgSha256Ready = $externalRuntimeProofRuntimeNupkgSha256Ready
  externalRuntimeProofLogSha256FormatReady = $externalRuntimeProofLogSha256FormatReady
  externalRuntimeProofLogSha256Matches = $externalRuntimeProofLogSha256Matches
  externalRuntimeProofFailedProofItemCount = $externalRuntimeProofFailedProofItemCount
  externalRuntimeProofOwnerActionStatus = $externalRuntimeProofOwnerActionStatus
  externalRuntimeProofCanPromoteRuntimeProof = $externalRuntimeProofCanPromoteRuntimeProof
  externalRuntimeExecutionEvidence = $externalRuntimeExecutionEvidence
  postPublishVerificationState = $postPublishVerificationState
  postPublishProofClassification = $postPublishProofClassification
  postPublishProofClassificationPromotable = $postPublishProofClassificationPromotable
  postPublishManagedNupkgSha256Ready = $postPublishManagedNupkgSha256Ready
  postPublishRuntimeNupkgSha256Ready = $postPublishRuntimeNupkgSha256Ready
  postPublishConsumerProjectIdentityReady = $postPublishConsumerProjectIdentityReady
  postPublishSmokeCommandRuntimeKeyReady = $postPublishSmokeCommandRuntimeKeyReady
  postPublishHostReady = $postPublishHostReady
  postPublishCommandsReady = $postPublishCommandsReady
  postPublishStdoutStderrSummaryReady = $postPublishStdoutStderrSummaryReady
  isPostPublishVerificationProof = $isPostPublishVerificationProof
  canCloseReleaseIssue = $canCloseReleaseIssue
  compatibleHostRequired = $compatibleHostRunbookCompatibleHostRequired -or $compatibleHostCollectionBundleCompatibleHostRequired
  compatibleHostRuntimeProofRunbookState = $compatibleHostRunbookState
  compatibleHostRuntimeProofRunbookCompatibleHostRequired = $compatibleHostRunbookCompatibleHostRequired
  compatibleHostRuntimeProofRunbookPerformsPublish = $compatibleHostRunbookPerformsPublish
  compatibleHostRuntimeProofRunbookApprovesPublicRelease = $compatibleHostRunbookApprovesPublicRelease
  compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof = $compatibleHostRunbookCanPromoteRuntimeProof
  compatibleHostRuntimeProofRunbookRuntimeExecutionEvidence = $compatibleHostRunbookRuntimeExecutionEvidence
  compatibleHostRuntimeProofRunbookPromotionBlockedReason = $compatibleHostRunbookPromotionBlockedReason
  compatibleHostRuntimeProofRunbookRunPackageConsumerSmokeCommand = $compatibleHostRunbookRunPackageConsumerSmokeCommand
  compatibleHostRuntimeProofRunbookValidateFilledRecordCommand = $compatibleHostRunbookValidateFilledRecordCommand
  compatibleHostRuntimeProofCollectionBundleState = $compatibleHostCollectionBundleState
  compatibleHostRuntimeProofCollectionBundleCompatibleHostRequired = $compatibleHostCollectionBundleCompatibleHostRequired
  compatibleHostRuntimeProofCollectionBundlePerformsPublish = $compatibleHostCollectionBundlePerformsPublish
  compatibleHostRuntimeProofCollectionBundleApprovesPublicRelease = $compatibleHostCollectionBundleApprovesPublicRelease
  compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof = $compatibleHostCollectionBundleCanPromoteRuntimeProof
  compatibleHostRuntimeProofCollectionBundleRuntimeExecutionEvidence = $compatibleHostCollectionBundleRuntimeExecutionEvidence
  compatibleHostRuntimeProofCollectionBundlePromotionBlockedReason = $compatibleHostCollectionBundlePromotionBlockedReason
  compatibleHostRuntimeProofCollectionBundleRunPackageConsumerSmokeCommand = $compatibleHostCollectionBundleRunPackageConsumerSmokeCommand
  compatibleHostRuntimeProofCollectionBundleValidateFilledRecordCommand = $compatibleHostCollectionBundleValidateFilledRecordCommand
  realCallbackRuntimeProof = $callbackProof
  signingStatus = $signingStatus
  linuxProofState = $linuxProofState
  checklistItemCount = if ($releaseChecklist) { @($releaseChecklist).Count } else { 0 }
  decisions = $decisions
  sourceEvidence = @(
    "artifacts/final-release/compatible-host-runtime-proof-runbook.json",
    "artifacts/final-release/compatible-host-runtime-proof-runbook.md",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
    "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.md",
    "artifacts/final-release/final-release-dry-run-summary.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json"
  )
  safetyNotes = @(
    "This template does not publish packages.",
    "ready-needs-manual-approval is not a public release approval.",
    "Linux handoff is not Linux runner proof.",
    "blocked-by-cuda-driver is not API proof.",
    "runtime-deserialization-dependency-diagnostics is not runtime execution proof.",
    "runtime proof blocker owner action must be resolved by compatible package-consumer-runtime evidence before promotion.",
    "external runtime proof remains owner-action-required until runtimePackageKey, packageSource.runtimePackageKey, clean consumer identity, --runtime-package-key smoke command, host CUDA/TensorRT/cuDNN metadata, nupkg SHA256, and logSha256 validation are ready.",
    "post-publish verification remains not closeable until package hashes, clean consumer identity, host metadata, command capture, --runtime-package-key smoke command, stdout/stderr summaries, and validator proof are ready.",
    "compatible-host-runtime-proof-runbook is an owner-action runbook, not runtime proof.",
    "compatible-host-runtime-proof-collection-bundle is an external execution package, not runtime proof, publication approval, or package push.",
    "compatibleHostRuntimeProofRunbookCanPromoteRuntimeProof=false keeps runtime proof owner action required.",
    "compatibleHostRuntimeProofCollectionBundleCanPromoteRuntimeProof=false keeps runtime proof owner action required.",
    "runtimeProofRequiredForRelease=true means public release still needs compatible runtime proof or explicit owner disposition.",
    "allowRuntimeSmokeBlocked=true records dry-run intent only; it is not smoke passed.",
    "realCallbackRuntimeProof=false must remain visible until InvocationCount>0 evidence exists."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-owner-decision-template.json"
$markdownPath = Join-Path $outputRoot "release-owner-decision-template.md"

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Owner Decision Template")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Linux runtime key: ``$LinuxRuntimePackageKey``")
$lines.Add("")
$lines.Add("Approval state: ``pending-release-owner-approval``")
$lines.Add("")
$lines.Add("This template is a release-owner decision record. It does not publish packages and does not turn dry-run evidence into release proof.")
$lines.Add("")
$lines.Add("## Current Evidence Snapshot")
$lines.Add("")
$lines.Add("- overall status: ``$overallStatus``")
$lines.Add("- blocking issues: $blockingIssueCount")
$lines.Add("- manual approvals: $manualApprovalCount")
$lines.Add("- warnings: $warningCount")
$lines.Add("- bilingual documentation findings: $bilingualFindingCount")
$lines.Add("- package consumer smoke: ``$smokeStatus``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- runtime proof required for release: ``$runtimeProofRequiredForRelease``")
$lines.Add("- allow runtime smoke blocked: ``$allowRuntimeSmokeBlocked``")
$lines.Add("- runtime proof blocker owner action: ``$runtimeProofBlockerOwnerActionStatus``")
$lines.Add("- runtime proof blocker category: ``$runtimeProofBlockerCategory``")
$lines.Add("- runtime proof suggested command: ``$runtimeProofOwnerCommand``")
$lines.Add("- external runtime proof state: ``$externalRuntimeProofState``")
$lines.Add("- external runtime proof classification: ``$externalRuntimeProofClassification``")
$lines.Add("- external runtime proof runtime key matches: ``$externalRuntimeProofRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof package source runtime key matches: ``$externalRuntimeProofPackageSourceRuntimePackageKeyMatches``")
$lines.Add("- external runtime proof consumer project identity ready: ``$externalRuntimeProofConsumerProjectIdentityReady``")
$lines.Add("- external runtime proof smoke command runtime key ready: ``$externalRuntimeProofSmokeCommandRuntimeKeyReady``")
$lines.Add("- external runtime proof host ready: ``$externalRuntimeProofHostReady``")
$lines.Add("- external runtime proof commands ready: ``$externalRuntimeProofCommandsReady``")
$lines.Add("- external runtime proof managed nupkg SHA256 ready: ``$externalRuntimeProofManagedNupkgSha256Ready``")
$lines.Add("- external runtime proof runtime nupkg SHA256 ready: ``$externalRuntimeProofRuntimeNupkgSha256Ready``")
$lines.Add("- external runtime proof log SHA256 format ready: ``$externalRuntimeProofLogSha256FormatReady``")
$lines.Add("- external runtime proof log SHA256 matches: ``$externalRuntimeProofLogSha256Matches``")
$lines.Add("- external runtime proof failed proof item count: ``$externalRuntimeProofFailedProofItemCount``")
$lines.Add("- external runtime proof owner action: ``$externalRuntimeProofOwnerActionStatus``")
$lines.Add("- post-publish verification state: ``$postPublishVerificationState``")
$lines.Add("- post-publish proof classification: ``$postPublishProofClassification``")
$lines.Add("- post-publish proof classification promotable: ``$postPublishProofClassificationPromotable``")
$lines.Add("- post-publish managed nupkg SHA256 ready: ``$postPublishManagedNupkgSha256Ready``")
$lines.Add("- post-publish runtime nupkg SHA256 ready: ``$postPublishRuntimeNupkgSha256Ready``")
$lines.Add("- post-publish consumer project identity ready: ``$postPublishConsumerProjectIdentityReady``")
$lines.Add("- post-publish smoke command runtime key ready: ``$postPublishSmokeCommandRuntimeKeyReady``")
$lines.Add("- post-publish host ready: ``$postPublishHostReady``")
$lines.Add("- post-publish commands ready: ``$postPublishCommandsReady``")
$lines.Add("- post-publish stdout/stderr summary ready: ``$postPublishStdoutStderrSummaryReady``")
$lines.Add("- post-publish verification proof: ``$isPostPublishVerificationProof``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- compatible host runbook state: ``$compatibleHostRunbookState``")
$lines.Add("- compatible host runbook can promote runtime proof: ``$compatibleHostRunbookCanPromoteRuntimeProof``")
$lines.Add("- compatible host runbook runtime execution evidence: ``$compatibleHostRunbookRuntimeExecutionEvidence``")
$lines.Add("- compatible host collection bundle state: ``$compatibleHostCollectionBundleState``")
$lines.Add("- compatible host collection bundle can promote runtime proof: ``$compatibleHostCollectionBundleCanPromoteRuntimeProof``")
$lines.Add("- compatible host collection bundle runtime execution evidence: ``$compatibleHostCollectionBundleRuntimeExecutionEvidence``")
$lines.Add("- compatible host collection bundle smoke command: ``$compatibleHostCollectionBundleRunPackageConsumerSmokeCommand``")
$lines.Add("- compatible host collection bundle validation command: ``$compatibleHostCollectionBundleValidateFilledRecordCommand``")
$lines.Add("- real callback runtime proof: ``$callbackProof``")
$lines.Add("- signing status: ``$signingStatus``")
$lines.Add("- Linux proof state: ``$linuxProofState``")
$lines.Add("")
$lines.Add("## Required Decisions")
$lines.Add("")
foreach ($decision in $decisions) {
  $lines.Add("### $($decision.title)")
  $lines.Add("")
  $lines.Add("- id: ``$($decision.id)``")
  $lines.Add("- current status: ``$($decision.currentStatus)``")
  $lines.Add("- default decision: ``$($decision.defaultDecision)``")
  $lines.Add("- required owner decision: $($decision.requiredOwnerDecision)")
  $lines.Add("- evidence: ``$($decision.evidence)``")
  $lines.Add("- cannot claim: $($decision.cannotClaim)")
  $lines.Add("- owner decision: [ ] approve [ ] defer [ ] block")
  $lines.Add("- owner notes:")
  $lines.Add("")
}
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $summary.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release owner decision template written to $jsonPath"
Write-Host "Release owner decision template written to $markdownPath"
