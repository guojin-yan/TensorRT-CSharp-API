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

function Normalize-PostPublishRequiredEvidence {
  param([AllowNull()][object]$Evidence)

  $items = @($Evidence | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
  foreach ($requiredField in @("noLocalPackageSource", "noLocalNupkgPackageReference")) {
    if ($items -notcontains $requiredField) {
      $items += $requiredField
    }
  }

  return @($items)
}

function New-ProofItem {
  param(
    [string]$Id,
    [string]$ProofClass,
    [string]$CurrentState,
    [string]$ExpectedInputTemplate,
    [string]$ExpectedValidatedRecord,
    [string[]]$RequiredArtifacts,
    [string]$ValidatorCommand,
    [string[]]$RequiredLogFields,
    [string[]]$RequiredSha256Fields,
    [string]$ArticleSlug,
    [int]$ArticleTopicId,
    [string]$OwnerAction,
    [string]$CompatibleHostRequirement,
    [string]$LocalCollectability,
    [string]$Blocker,
    [string[]]$NonSubstituteProofKinds
  )

  [pscustomobject]@{
    id = $Id
    proofClass = $ProofClass
    currentState = $CurrentState
    expectedInputTemplate = $ExpectedInputTemplate
    expectedValidatedRecord = $ExpectedValidatedRecord
    requiredArtifacts = $RequiredArtifacts
    validatorCommand = $ValidatorCommand
    requiredLogFields = $RequiredLogFields
    requiredSha256Fields = $RequiredSha256Fields
    articleSlug = $ArticleSlug
    articleTopicId = $ArticleTopicId
    ownerAction = $OwnerAction
    compatibleHostRequirement = $CompatibleHostRequirement
    localCollectability = $LocalCollectability
    blocker = $Blocker
    nonSubstituteProofKinds = $NonSubstituteProofKinds
    passed = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
  }
}

$freeze = Read-JsonOrNull "artifacts\final-release\release-freeze-final-verification.json"
$preflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$evidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$aliasClosure = Read-JsonOrNull "artifacts\interface-coverage\deferred-btier-alias-proof-closure-record.json"
$packageConsumerProofPack = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-execution-pack.json"
$packageConsumerProofPackValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-pack-validation.json"
$linuxRunnerProofPack = Read-JsonOrNull "artifacts\final-release\linux-runner-proof-execution-pack.json"
$linuxRunnerProofPackValidation = Read-JsonOrNull "artifacts\final-release\linux-runner-proof-pack-validation.json"
$realCaseProofPack = Read-JsonOrNull "artifacts\final-release\real-case-proof-execution-pack.json"
$realCaseEvidenceValidation = Read-JsonOrNull "artifacts\final-release\real-case-evidence-record-validation.json"

$preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$freezeState = [string](Get-PropertyOrDefault -Object $freeze -Name "verificationState" -DefaultValue "missing-release-freeze-final-verification")
$externalProofState = [string](Get-PropertyOrDefault -Object $evidence -Name "externalRuntimeProofState" -DefaultValue "missing-external-runtime-proof-validation")
$linuxProof = [bool](Get-PropertyOrDefault -Object $evidence -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleProof = [bool](Get-PropertyOrDefault -Object $evidence -Name "sampleRunEvidenceCanPromoteRealModelRuntime" -DefaultValue $false)
$postPublishProof = [bool](Get-PropertyOrDefault -Object $evidence -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishRequiredEvidence = @(Get-PropertyOrDefault -Object $preflight -Name "postPublishRequiredEvidence" -DefaultValue @())
if ($postPublishRequiredEvidence.Count -eq 0) {
  $postPublishRequiredEvidence = @(
    "selectedChannel",
    "channelSourceUri",
    "publishedPackageUrl",
    "managedPackageUrl",
    "runtimePackageUrl",
    "managedNupkgSha256",
    "runtimeNupkgSha256",
    "cleanConsumerRootOutsideRepository",
    "consumerProjectPath",
    "noProjectReference",
    "noLocalPackageSource",
    "noLocalNupkgPackageReference",
    "restoreLogPath",
    "nativeAssetListingSha256",
    "dependencyProbeLogPath",
    "dependencyProbeLogSha256",
    "runtimeSmokeLogPath",
    "runtimeSmokeLogSha256",
    "runtimeSmokePassed",
    "runtimeSmokeExitCode",
    "stdoutSummary",
    "stderrSummary",
    "hostMetadata")
}
$postPublishRequiredEvidence = Normalize-PostPublishRequiredEvidence -Evidence $postPublishRequiredEvidence
$aliasClosureCount = [int](Get-PropertyOrDefault -Object $aliasClosure -Name "closureCandidateCount" -DefaultValue 0)
$packageConsumerProofPackState = [string](Get-PropertyOrDefault -Object $packageConsumerProofPack -Name "packState" -DefaultValue "missing-package-consumer-runtime-proof-execution-pack")
$packageConsumerProofPackMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $packageConsumerProofPack -Name "missingOwnerInputCount" -DefaultValue 0)
$packageConsumerProofPackCanPromote = [bool](Get-PropertyOrDefault -Object $packageConsumerProofPack -Name "canPromotePackageConsumerRuntime" -DefaultValue $false)
$packageConsumerProofPackValidationState = [string](Get-PropertyOrDefault -Object $packageConsumerProofPackValidation -Name "validationState" -DefaultValue "missing-package-consumer-runtime-proof-pack-validation")
$packageConsumerProofPackValidationCanPromote = [bool](Get-PropertyOrDefault -Object $packageConsumerProofPackValidation -Name "canPromotePackageConsumerRuntime" -DefaultValue $false)
$linuxRunnerProofPackState = [string](Get-PropertyOrDefault -Object $linuxRunnerProofPack -Name "packState" -DefaultValue "missing-linux-runner-proof-execution-pack")
$linuxRunnerProofPackMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $linuxRunnerProofPack -Name "missingOwnerInputCount" -DefaultValue 0)
$linuxRunnerProofPackCanPromote = [bool](Get-PropertyOrDefault -Object $linuxRunnerProofPack -Name "canPromoteLinuxRunnerProof" -DefaultValue $false)
$linuxRunnerProofPackValidationState = [string](Get-PropertyOrDefault -Object $linuxRunnerProofPackValidation -Name "validationState" -DefaultValue "missing-linux-runner-proof-pack-validation")
$linuxRunnerProofPackValidationCanPromote = [bool](Get-PropertyOrDefault -Object $linuxRunnerProofPackValidation -Name "canPromoteLinuxRunnerProof" -DefaultValue $false)
$realCaseProofPackState = [string](Get-PropertyOrDefault -Object $realCaseProofPack -Name "packState" -DefaultValue "missing-real-case-proof-execution-pack")
$realCaseProofCaseCount = [int](Get-PropertyOrDefault -Object $realCaseProofPack -Name "caseCount" -DefaultValue 0)
$realCaseProofBlockedCaseCount = [int](Get-PropertyOrDefault -Object $realCaseProofPack -Name "blockedCaseCount" -DefaultValue 0)
$realCaseProofMissingOwnerInputCount = [int](Get-PropertyOrDefault -Object $realCaseProofPack -Name "missingOwnerInputCount" -DefaultValue 0)
$realCaseProofCanPromote = [bool](Get-PropertyOrDefault -Object $realCaseProofPack -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$realCaseEvidenceValidatorState = [string](Get-PropertyOrDefault -Object $realCaseEvidenceValidation -Name "validationState" -DefaultValue "missing-real-case-evidence-validation")
$realCaseEvidenceValidatorCanPromote = [bool](Get-PropertyOrDefault -Object $realCaseEvidenceValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)

$commonNonSubstitutes = @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "managed-readiness",
  "managed-readiness-only",
  "callback-allocator-readiness-snapshot",
  "CallbackAllocatorReadinessSnapshot",
  "TensorRtCallbackAllocatorReadinessSnapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only",
  "B-tier alias proof closure record",
  "deferred safety triage",
  "release freeze final verification"
)

$proofItems = @(
  New-ProofItem `
    -Id "owner-authorization" `
    -ProofClass "owner-authorization" `
    -CurrentState "freezeState=$freezeState; preflightState=$preflightState; owner authorization remains required" `
    -ExpectedInputTemplate "artifacts/final-release/release-owner-approval-input-template.json" `
    -ExpectedValidatedRecord "artifacts/final-release/release-owner-approval-input-record.json" `
    -RequiredArtifacts @("release-owner-approval-input-record.json", "release-owner-decision-record.json", "owner-authorized-publish-command-plan.json", "owner-authorized-publish-command-plan-validation.json") `
    -ValidatorCommand "Test-ReleaseOwnerApprovalInput.ps1 + Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -RequiredLogFields @("ownerName", "ownerDecisionId", "approvalTimestampUtc", "targetChannel", "rollbackPlan", "credentialHandlingAcknowledged", "nvidiaRedistributionApproval") `
    -RequiredSha256Fields @("approvedCommandPlanSha256", "approvedProofBundleSha256", "ownerApprovalInputSha256", "ownerAuthorizedPublishCommandPlanSha256") `
    -ArticleSlug "release-owner-action-checklist-final-hold.md" `
    -ArticleTopicId 100 `
    -OwnerAction "Owner must provide explicit non-template approval and manually materialize publish commands." `
    -CompatibleHostRequirement "Human owner decision; no CUDA host requirement." `
    -LocalCollectability "not-collectable-locally-without-owner" `
    -Blocker "owner-action-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes

  New-ProofItem `
    -Id "package-consumer-runtime" `
    -ProofClass "package-consumer-runtime" `
    -CurrentState "externalRuntimeProofState=$externalProofState; runtimePackageKey=$RuntimePackageKey" `
    -ExpectedInputTemplate "artifacts/final-release/external-runtime-proof-record-template.json" `
    -ExpectedValidatedRecord "artifacts/final-release/external-runtime-proof-record.json" `
    -RequiredArtifacts @("external-runtime-proof-record.json", "clean consumer project outside repository", "restore/build/runtime smoke log", "nupkg SHA256 list") `
    -ValidatorCommand "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredLogFields @("restoreLogPath", "buildLogPath", "runtimeSmokeLogPath", "stdoutSummary", "stderrSummary", "cleanConsumerRoot", "noProjectReference", "noLocalPackageSource", "noLocalNupkgPackageReference") `
    -RequiredSha256Fields @("managedNupkgSha256", "runtimeNupkgSha256", "runtimeSmokeLogSha256") `
    -ArticleSlug "compatible-host-proof-backfill-package.md" `
    -ArticleTopicId 86 `
    -OwnerAction "Run clean external package consumer smoke on a compatible CUDA/TensorRT host." `
    -CompatibleHostRequirement "Windows x64 host with matching CUDA/TensorRT/cuDNN and accessible GPU for $RuntimePackageKey." `
    -LocalCollectability "blocked-unless-compatible-host-and-real-package-logs-exist" `
    -Blocker "compatible-host-runtime-proof-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes

  New-ProofItem `
    -Id "package-consumer-proof-pack" `
    -ProofClass "package-consumer-runtime" `
    -CurrentState "packState=$packageConsumerProofPackState; validationState=$packageConsumerProofPackValidationState; missingOwnerInputCount=$packageConsumerProofPackMissingOwnerInputCount" `
    -ExpectedInputTemplate "artifacts/final-release/package-consumer-runtime-proof-execution-pack.json" `
    -ExpectedValidatedRecord "artifacts/final-release/external-runtime-proof-record.json" `
    -RequiredArtifacts @("package-consumer-runtime-proof-execution-pack.json", "external-runtime-proof-record.json", "clean consumer project outside repository", "restore/build/runtime smoke log", "nupkg SHA256 list") `
    -ValidatorCommand "Test-PackageConsumerRuntimeProofPack.ps1 + Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredLogFields @("restoreLogPath", "buildLogPath", "runtimeSmokeLogPath", "stdoutSummary", "stderrSummary", "cleanConsumerRoot", "noProjectReference", "noLocalPackageSource", "noLocalNupkgPackageReference") `
    -RequiredSha256Fields @("managedNupkgSha256", "runtimeNupkgSha256", "runtimeSmokeLogSha256") `
    -ArticleSlug "package-consumer-runtime-proof-playbook.md" `
    -ArticleTopicId 86 `
    -OwnerAction "Fill package-consumer-runtime proof pack and external runtime proof record from a clean external consumer on a compatible host." `
    -CompatibleHostRequirement "Windows x64 CUDA/TensorRT/cuDNN host with real package source and GPU runtime smoke capability for $RuntimePackageKey." `
    -LocalCollectability "blocked-unless-clean-external-consumer-real-package-logs-and-host-metadata-exist" `
    -Blocker "package-consumer-owner-evidence-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes

  New-ProofItem `
    -Id "linux-runner-proof" `
    -ProofClass "linux-runner-proof" `
    -CurrentState "linuxRuntimePackageKey=$LinuxRuntimePackageKey; isRealLinuxRunnerProof=$linuxProof" `
    -ExpectedInputTemplate "artifacts/final-release/linux-runner-evidence-record.template.json" `
    -ExpectedValidatedRecord "artifacts/final-release/linux-runner-evidence-record.json" `
    -RequiredArtifacts @("linux-runner-evidence-record.json", "Linux command log", "CUDA/TensorRT/cuDNN host metadata") `
    -ValidatorCommand "Test-LinuxRunnerEvidenceRecord.ps1" `
    -RequiredLogFields @("linuxRunnerCommand", "linuxRunnerLogPath", "stdoutSummary", "stderrSummary", "hostOs", "gpuName", "cudaVersion", "tensorRtVersion") `
    -RequiredSha256Fields @("linuxRunnerLogSha256") `
    -ArticleSlug "blog-linux-runner-evidence-guide.md" `
    -ArticleTopicId 94 `
    -OwnerAction "Run Linux runner on a real Linux x64 CUDA/TensorRT host and copy back validated evidence." `
    -CompatibleHostRequirement "Linux x64 compatible CUDA/TensorRT host for $LinuxRuntimePackageKey." `
    -LocalCollectability "not-collectable-on-current-windows-only-run" `
    -Blocker "real-linux-host-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes

  New-ProofItem `
    -Id "linux-runner-proof-pack" `
    -ProofClass "linux-runner-proof" `
    -CurrentState "packState=$linuxRunnerProofPackState; validationState=$linuxRunnerProofPackValidationState; missingOwnerInputCount=$linuxRunnerProofPackMissingOwnerInputCount" `
    -ExpectedInputTemplate "artifacts/final-release/linux-runner-proof-execution-pack.json" `
    -ExpectedValidatedRecord "artifacts/final-release/linux-runner-evidence-record.json" `
    -RequiredArtifacts @("linux-runner-proof-execution-pack.json", "linux-runner-evidence-record.json", "Linux command log", "runtime package SHA256", "CUDA/TensorRT/cuDNN host metadata") `
    -ValidatorCommand "Test-LinuxRunnerProofPack.ps1 + Test-LinuxRunnerEvidenceRecord.ps1" `
    -RequiredLogFields @("linuxRunnerCommand", "linuxRunnerLogPath", "stdoutSummary", "stderrSummary", "hostOs", "kernelVersion", "gpuName", "cudaVersion", "tensorRtVersion") `
    -RequiredSha256Fields @("runtimePackageSha256", "linuxRunnerLogSha256") `
    -ArticleSlug "linux-runner-evidence-checklist.md" `
    -ArticleTopicId 94 `
    -OwnerAction "Fill Linux runner proof pack and evidence record from a real Linux x64 CUDA/TensorRT host." `
    -CompatibleHostRequirement "Linux x64 compatible CUDA/TensorRT host for $LinuxRuntimePackageKey." `
    -LocalCollectability "not-collectable-on-current-windows-only-run" `
    -Blocker "linux-runner-owner-evidence-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes

  New-ProofItem `
    -Id "real-model-runtime" `
    -ProofClass "real-model-runtime" `
    -CurrentState "sampleRunEvidenceCanPromoteRealModelRuntime=$sampleProof; aliasClosureCandidateCount=$aliasClosureCount" `
    -ExpectedInputTemplate "artifacts/user-acceptance/sample-run-evidence-record.yolovision.template.json" `
    -ExpectedValidatedRecord "artifacts/user-acceptance/sample-run-evidence-record.json" `
    -RequiredArtifacts @("Classification real model assets", "YoloVision real model assets", "sample-run-evidence-record.json", "sample runner logs") `
    -ValidatorCommand "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -RequiredLogFields @("sampleRunCommand", "sampleRunLogPath", "stdoutSummary", "stderrSummary", "proofClassification", "canPromoteRealModelRuntime") `
    -RequiredSha256Fields @("modelSha256", "labelsSha256", "inputAssetSha256", "sampleRunLogSha256") `
    -ArticleSlug "real-model-owner-backfill-checklist.md" `
    -ArticleTopicId 90 `
    -OwnerAction "Provide real Classification/YoloVision assets and collect sample runtime logs." `
    -CompatibleHostRequirement "CUDA/TensorRT host capable of running Classification and YoloVision samples." `
    -LocalCollectability "blocked-unless-real-model-assets-and-compatible-runtime-exist" `
    -Blocker "real-model-assets-and-runtime-proof-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes

  New-ProofItem `
    -Id "real-case-proof-pack" `
    -ProofClass "real-model-runtime" `
    -CurrentState "packState=$realCaseProofPackState; caseCount=$realCaseProofCaseCount; blockedCaseCount=$realCaseProofBlockedCaseCount; validatorState=$realCaseEvidenceValidatorState" `
    -ExpectedInputTemplate "artifacts/final-release/real-case-evidence-record-template.json" `
    -ExpectedValidatedRecord "artifacts/final-release/real-case-evidence-record.json" `
    -RequiredArtifacts @("real-case-proof-execution-pack.json", "real-case-evidence-record-template.json", "real-case-evidence-record.json", "YoloVision/OnnxToEngine/TensorRtExec real logs and screenshots") `
    -ValidatorCommand "Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json" `
    -RequiredLogFields @("commandLine", "stdoutLogPath", "stderrLogPath", "stdoutSummary", "stderrSummary", "hostOs", "gpuName", "cudaVersion", "tensorRtVersion") `
    -RequiredSha256Fields @("onnxSha256", "engineSha256", "inputArtifactSha256", "outputArtifactSha256") `
    -ArticleSlug "yolovision-real-asset-walkthrough.md" `
    -ArticleTopicId 90 `
    -OwnerAction "Fill real case evidence records for YoloVision, OnnxToEngine, and TensorRtExec with real assets, logs, screenshots, hashes, and host metadata." `
    -CompatibleHostRequirement "CUDA/TensorRT host capable of building external ONNX and running selected real sample cases." `
    -LocalCollectability "blocked-unless-real-case-assets-logs-hashes-and-host-metadata-exist" `
    -Blocker "real-case-owner-evidence-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes

  New-ProofItem `
    -Id "post-publish-verification" `
    -ProofClass "post-publish-verification" `
    -CurrentState "isPostPublishVerificationProof=$postPublishProof; publication not performed by automation" `
    -ExpectedInputTemplate "artifacts/final-release/post-publish-verification-record.template.json" `
    -ExpectedValidatedRecord "artifacts/final-release/post-publish-verification-record.json" `
    -RequiredArtifacts @("post-publish-verification-record.json", "real channel package URL", "published package URL", "clean external consumer root outside repository", "runtime smoke log/hash", "host metadata") `
    -ValidatorCommand "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredLogFields @("selectedChannel", "channelSourceUri", "publishedPackageUrl", "managedPackageUrl", "runtimePackageUrl", "cleanConsumerRootOutsideRepository", "consumerProjectPath", "noProjectReference", "noLocalPackageSource", "noLocalNupkgPackageReference", "restoreLogPath", "dependencyProbeLogPath", "runtimeSmokeLogPath", "runtimeSmokePassed", "runtimeSmokeExitCode", "stdoutSummary", "stderrSummary", "hostMetadata") `
    -RequiredSha256Fields @("managedNupkgSha256", "runtimeNupkgSha256", "nativeAssetListingSha256", "dependencyProbeLogSha256", "runtimeSmokeLogSha256") `
    -ArticleSlug "post-publish-verification-record.md" `
    -ArticleTopicId 88 `
    -OwnerAction "After owner-approved publication, validate the real channel package from a clean external consumer." `
    -CompatibleHostRequirement "Post-publication clean consumer host with package source access and runtime smoke capability." `
    -LocalCollectability "blocked-until-owner-approved-publication-exists" `
    -Blocker "post-publish-timing-required" `
    -NonSubstituteProofKinds $commonNonSubstitutes
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-runtime-proof-execution-matrix"
  matrixState = "blocked-real-proof-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isReleaseProofComplete = $false
  sourceArtifacts = @(
    "artifacts/final-release/release-freeze-final-verification.json",
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-owner-approval-input-template.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/owner-authorized-publish-command-plan.json",
    "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
    "artifacts/interface-coverage/deferred-btier-alias-proof-closure-record.json",
    "artifacts/final-release/external-runtime-proof-record-template.json",
    "artifacts/final-release/package-consumer-runtime-proof-execution-pack.json",
    "artifacts/final-release/package-consumer-runtime-proof-pack-validation.json",
    "artifacts/final-release/linux-runner-evidence-record.template.json",
    "artifacts/final-release/linux-runner-proof-execution-pack.json",
    "artifacts/final-release/linux-runner-proof-pack-validation.json",
    "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
    "artifacts/user-acceptance/onnx-engine-build-evidence-sidecar-audit.json",
    "artifacts/final-release/real-model-and-package-proof-input-package.json",
    "artifacts/final-release/real-case-proof-execution-pack.json",
    "artifacts/final-release/real-case-evidence-record-template.json",
    "artifacts/final-release/real-case-evidence-record-validation.json",
    "artifacts/final-release/post-publish-verification-record-template.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/callback-runtime-proof-execution-pack.json",
    "artifacts/final-release/callback-runtime-proof-execution-pack-validation.json"
  )
  nonSubstituteProofKinds = $commonNonSubstitutes
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidence.Count
  proofItemCount = $proofItems.Count
  proofItems = $proofItems
  blockedProofItemCount = @($proofItems | Where-Object { -not $_.passed }).Count
  ownerProofTemplateCoverageState = "covered-by-eight-template-paths"
  validatorCoverageState = "covered-by-eight-validator-commands"
  ownerProofExecutionArticleCoverageState = "covered-by-article-slug-map"
  expectedInputTemplateCount = @($proofItems | Where-Object { -not [string]::IsNullOrWhiteSpace($_.expectedInputTemplate) }).Count
  expectedValidatedRecordCount = @($proofItems | Where-Object { -not [string]::IsNullOrWhiteSpace($_.expectedValidatedRecord) }).Count
  validatorCommandCount = @($proofItems | Where-Object { -not [string]::IsNullOrWhiteSpace($_.validatorCommand) }).Count
  articleMappedProofItemCount = @($proofItems | Where-Object { -not [string]::IsNullOrWhiteSpace($_.articleSlug) -and $_.articleTopicId -gt 0 }).Count
  packageConsumerProofExecutionPackPath = "artifacts/final-release/package-consumer-runtime-proof-execution-pack.json"
  packageConsumerProofPackState = $packageConsumerProofPackState
  packageConsumerProofPackValidationState = $packageConsumerProofPackValidationState
  packageConsumerProofMissingOwnerInputCount = $packageConsumerProofPackMissingOwnerInputCount
  packageConsumerProofPackCanPromote = ($packageConsumerProofPackCanPromote -and $packageConsumerProofPackValidationCanPromote)
  linuxRunnerProofExecutionPackPath = "artifacts/final-release/linux-runner-proof-execution-pack.json"
  linuxRunnerProofPackState = $linuxRunnerProofPackState
  linuxRunnerProofPackValidationState = $linuxRunnerProofPackValidationState
  linuxRunnerProofMissingOwnerInputCount = $linuxRunnerProofPackMissingOwnerInputCount
  linuxRunnerProofPackCanPromote = ($linuxRunnerProofPackCanPromote -and $linuxRunnerProofPackValidationCanPromote)
  realCaseProofExecutionPackPath = "artifacts/final-release/real-case-proof-execution-pack.json"
  realCaseEvidenceRecordTemplatePath = "artifacts/final-release/real-case-evidence-record-template.json"
  realCaseEvidenceValidatorCommand = "Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json"
  realCaseProofPackState = $realCaseProofPackState
  realCaseProofCaseCount = $realCaseProofCaseCount
  realCaseProofBlockedCaseCount = $realCaseProofBlockedCaseCount
  realCaseProofMissingOwnerInputCount = $realCaseProofMissingOwnerInputCount
  realCaseProofCanPromote = ($realCaseProofCanPromote -and $realCaseEvidenceValidatorCanPromote)
  boundary = "Runtime proof execution matrix is a blocked execution plan. It does not perform publish actions and does not convert templates, B-tier alias closure, local feed, ProjectReference, managed-readiness, CallbackAllocatorReadinessSnapshot, precheck-only, dry-run-only, or schema-only evidence into release proof."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-runtime-proof-execution-matrix.json"
$markdownPath = Join-Path $artifactRoot "release-runtime-proof-execution-matrix.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$proofRows = $proofItems | ForEach-Object {
  $required = (@($_.requiredArtifacts) -join "<br>").Replace("|", "\|")
  $logs = (@($_.requiredLogFields) -join "<br>").Replace("|", "\|")
  $hashes = (@($_.requiredSha256Fields) -join "<br>").Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.proofClass)`` | ``$($_.passed)`` | $($_.currentState.Replace("|", "\|")) | ``$($_.expectedInputTemplate)`` | ``$($_.expectedValidatedRecord)`` | ``$($_.validatorCommand)`` | $logs | $hashes | $required | ``$($_.articleTopicId)`` / ``$($_.articleSlug)`` | $($_.blocker.Replace("|", "\|")) |"
}

$markdown = @"
# Release Runtime Proof Execution Matrix

生成时间：$($record.generatedAtUtc)

## 总结

``recordKind=release-runtime-proof-execution-matrix``，``matrixState=blocked-real-proof-required``。该矩阵是 release proof 的真实执行清单，不执行发布、不上传包、不关闭 release issue。

| 项目 | 值 |
|---|---|
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isReleaseProofComplete | ``False`` |
| proof item count | ``$($record.proofItemCount)`` |
| blocked proof item count | ``$($record.blockedProofItemCount)`` |
| owner proof template coverage state | ``$($record.ownerProofTemplateCoverageState)`` |
| validator coverage state | ``$($record.validatorCoverageState)`` |
| article coverage state | ``$($record.ownerProofExecutionArticleCoverageState)`` |
| expected input template count | ``$($record.expectedInputTemplateCount)`` |
| expected validated record count | ``$($record.expectedValidatedRecordCount)`` |
| validator command count | ``$($record.validatorCommandCount)`` |
| article mapped proof item count | ``$($record.articleMappedProofItemCount)`` |
| package consumer proof pack state | ``$($record.packageConsumerProofPackState)`` |
| package consumer missing owner input count | ``$($record.packageConsumerProofMissingOwnerInputCount)`` |
| Linux runner proof pack state | ``$($record.linuxRunnerProofPackState)`` |
| Linux runner missing owner input count | ``$($record.linuxRunnerProofMissingOwnerInputCount)`` |
| real case proof pack state | ``$($record.realCaseProofPackState)`` |
| real case proof case count | ``$($record.realCaseProofCaseCount)`` |
| real case blocked case count | ``$($record.realCaseProofBlockedCaseCount)`` |
| real case missing owner input count | ``$($record.realCaseProofMissingOwnerInputCount)`` |

## Proof Items

| ID | Proof class | Passed | Current state | Input template | Validated record | Validator | Required logs | Required SHA256 | Required artifacts | Article | Blocker |
|---|---|---|---|---|---|---|---|---|---|---|---|
$($proofRows -join "`r`n")

## Boundary

$($record.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release runtime proof execution matrix written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ProofItemCount=$($record.proofItemCount)"
Write-Output "BlockedProofItemCount=$($record.blockedProofItemCount)"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
