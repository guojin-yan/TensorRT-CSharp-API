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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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

function Write-Utf8File {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject
  )

  $content = @($InputObject) -join [Environment]::NewLine
  [System.IO.File]::WriteAllText($LiteralPath, $content + [Environment]::NewLine, $script:utf8)
}

function New-BlockerLane {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string]$CurrentState,
    [string]$OwnerNextAction,
    [string[]]$RequiredEvidence,
    [string[]]$Validators,
    [string[]]$Rejects,
    [string]$Boundary
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    currentState = $CurrentState
    ownerActionRequired = $true
    readyForPromotion = $false
    requiredEvidence = @($RequiredEvidence)
    validators = @($Validators)
    rejects = @($Rejects)
    ownerNextAction = $OwnerNextAction
    boundary = $Boundary
  }
}

function Test-RepositoryFileExists {
  param([string]$RelativePath)

  return Test-Path -LiteralPath (Join-Path $RepositoryRoot $RelativePath) -PathType Leaf
}

$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$closeDashboard = Read-JsonOrNull "artifacts\final-release\final-release-close-blocker-dashboard.json"
$cleanClosureValidation = Read-JsonOrNull "artifacts\final-release\clean-consumer-external-proof-closure-pack-validation.json"
$cleanProofExecutionBundle = Read-JsonOrNull "artifacts\final-release\clean-consumer-proof-execution-bundle.json"
$cleanPublicPackageConsumerProofGapReport = Read-JsonOrNull "artifacts\final-release\clean-public-package-consumer-proof-gap-report.json"
$postPublishPreflight = Read-JsonOrNull "artifacts\final-release\final-post-publish-clean-consumer-proof-preflight.json"
$postPublishCandidateValidation = Read-JsonOrNull "artifacts\final-release\final-post-publish-clean-consumer-proof-candidate-validation.json"
$postPublishOwnerInputValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-owner-input-validation.json"
$trt11BridgeRuntimeProof = Read-JsonOrNull "artifacts\package-consumer\bridge-runtime\win-x64-trt11.0-cuda13.2-cudnn9.22\bridge-package-runtime-consumer-proof.json"
$trt11RootCauseReport = Read-JsonOrNull "artifacts\final-release\trt11-runtime-smoke-root-cause-report.json"
$trt11DllResolutionReport = Read-JsonOrNull "artifacts\final-release\trt11-runtime-dll-resolution-report.json"
$trt10VsTrt11BridgeRuntimeDiagnosticDiff = Read-JsonOrNull "artifacts\final-release\trt10-vs-trt11-bridge-runtime-diagnostic-diff.json"
$yoloVisionLicenseApprovalValidation = Read-JsonOrNull "artifacts\yolovision\reference-assets\asset-license-approval-validation.json"
$deferredBTierWorkPackage = Read-JsonOrNull "artifacts\interface-coverage\deferred-btier-implementation-work-package.json"
$deferredSafetyTriage = Read-JsonOrNull "artifacts\interface-coverage\deferred-candidate-safety-triage.json"
$projectQualityShardRunbookReady = (Test-RepositoryFileExists "artifacts\test-analysis\project-quality-shard-runbook.md") -and (Test-RepositoryFileExists "docs\articles\zh-cn\project-quality-sharded-gate.md")
$projectQualityShardRunbookState = if ($projectQualityShardRunbookReady) { "runbook-ready-non-proof" } else { "runbook-required" }

$sourceStates = [pscustomobject]@{
  releaseEvidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
  releaseCanPublishPublicly = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canPublishPublicly" -DefaultValue $false)
  releaseCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canPromoteRuntimeProof" -DefaultValue $false)
  releaseCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canCloseReleaseIssue" -DefaultValue $false)
  releaseIsPackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isPackageConsumerRuntimeProof" -DefaultValue $false)
  releaseIsRealModelRuntimeProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isRealModelRuntimeProof" -DefaultValue $false)
  closeDashboardState = [string](Get-PropertyOrDefault -Object $closeDashboard -Name "dashboardState" -DefaultValue "missing-final-release-close-blocker-dashboard")
  closeDashboardBlockedBlockerCount = [int](Get-PropertyOrDefault -Object $closeDashboard -Name "blockedBlockerCount" -DefaultValue -1)
  cleanClosureValidationState = [string](Get-PropertyOrDefault -Object $cleanClosureValidation -Name "validationState" -DefaultValue "missing-clean-consumer-external-proof-closure-pack-validation")
  cleanProofExecutionBundleState = [string](Get-PropertyOrDefault -Object $cleanProofExecutionBundle -Name "bundleState" -DefaultValue "missing-clean-consumer-proof-execution-bundle")
  cleanPublicPackageConsumerProofGapReportState = [string](Get-PropertyOrDefault -Object $cleanPublicPackageConsumerProofGapReport -Name "reportState" -DefaultValue "missing-clean-public-package-consumer-proof-gap-report")
  cleanPublicPackageConsumerProofGapCount = [int](Get-PropertyOrDefault -Object $cleanPublicPackageConsumerProofGapReport -Name "gapCount" -DefaultValue -1)
  cleanPublicPackageConsumerProofOwnerActionRequiredCount = [int](Get-PropertyOrDefault -Object $cleanPublicPackageConsumerProofGapReport -Name "ownerActionRequiredCount" -DefaultValue -1)
  cleanPublicPackageConsumerProofOwnerInputFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $cleanPublicPackageConsumerProofGapReport -Name "ownerInputFailedActionRequiredCount" -DefaultValue -1)
  cleanPublicPackageConsumerProofCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $cleanPublicPackageConsumerProofGapReport -Name "canPromoteRuntimeProof" -DefaultValue $false)
  postPublishPreflightState = [string](Get-PropertyOrDefault -Object $postPublishPreflight -Name "preflightState" -DefaultValue "missing-final-post-publish-clean-consumer-proof-preflight")
  postPublishPreflightBlockedProofCandidateCount = [int](Get-PropertyOrDefault -Object $postPublishPreflight -Name "blockedProofCandidateCount" -DefaultValue -1)
  postPublishPreflightFailedActionRequiredCount = [int](Get-PropertyOrDefault -Object $postPublishPreflight -Name "failedActionRequiredCount" -DefaultValue -1)
  postPublishCandidateValidationState = [string](Get-PropertyOrDefault -Object $postPublishCandidateValidation -Name "validationState" -DefaultValue "missing-final-post-publish-clean-consumer-proof-candidate-validation")
  postPublishOwnerInputValidationState = [string](Get-PropertyOrDefault -Object $postPublishOwnerInputValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-owner-input-validation")
  trt11ProofClassification = [string](Get-PropertyOrDefault -Object $trt11BridgeRuntimeProof -Name "proofClassification" -DefaultValue "missing-trt11-bridge-runtime-proof")
  trt11SmokeStatus = [string](Get-PropertyOrDefault -Object $trt11BridgeRuntimeProof -Name "smokeStatus" -DefaultValue "missing-smoke-status")
  trt11ExitCode = [int](Get-PropertyOrDefault -Object $trt11BridgeRuntimeProof -Name "exitCode" -DefaultValue -1)
  trt11CanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $trt11BridgeRuntimeProof -Name "canPromoteRuntimeProof" -DefaultValue $false)
  trt11RootCauseReportState = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "reportState" -DefaultValue "missing-trt11-runtime-smoke-root-cause-report")
  trt11RootCauseFailureSignature = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "failureSignature" -DefaultValue "missing-failure-signature")
  trt11RootCauseCategory = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "rootCauseCategory" -DefaultValue "missing-root-cause-category")
  trt11RootCauseSubcategory = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "rootCauseSubcategory" -DefaultValue "missing-root-cause-subcategory")
  trt11RootCauseCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "canPromoteRuntimeProof" -DefaultValue $false)
  trt11RootCauseCudaPreflightAvailable = [bool](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "cudaPreflightAvailable" -DefaultValue $false)
  trt11RootCauseCudaPreflightAttempted = [bool](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "cudaPreflightAttempted" -DefaultValue $false)
  trt11RootCauseCudaPreflightDriverVersion = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "cudaPreflightDriverVersion" -DefaultValue "")
  trt11RootCauseCudaPreflightRuntimeVersion = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "cudaPreflightRuntimeVersion" -DefaultValue "")
  trt11RootCauseCudaPreflightDeviceCount = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "cudaPreflightDeviceCount" -DefaultValue "")
  trt11RootCauseCudaPreflightInitStatus = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "cudaPreflightInitStatus" -DefaultValue "")
  trt11RootCauseCudaPreflightCanAttemptTensorRtRuntimeCreate = [bool](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "cudaPreflightCanAttemptTensorRtRuntimeCreate" -DefaultValue $false)
  trt11RootCauseNativeCreateRuntimeDiagnosticAvailable = [bool](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "nativeCreateRuntimeDiagnosticAvailable" -DefaultValue $false)
  trt11RootCauseNativeCreateRuntimeAttempted = [bool](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "nativeCreateRuntimeAttempted" -DefaultValue $false)
  trt11RootCauseNativeCreateRuntimeReturnedNull = [bool](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "nativeCreateRuntimeReturnedNull" -DefaultValue $false)
  trt11RootCauseNativeCreateRuntimeLastStatus = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "nativeCreateRuntimeLastStatus" -DefaultValue "")
  trt11RootCauseNativeCreateRuntimePhase = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "nativeCreateRuntimePhase" -DefaultValue "")
  trt11RootCauseNativeCreateRuntimeLoggerMessageCount = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "nativeCreateRuntimeLoggerMessageCount" -DefaultValue "")
  trt11RootCauseNativeCreateRuntimeLastLoggerMessage = [string](Get-PropertyOrDefault -Object $trt11RootCauseReport -Name "nativeCreateRuntimeLastLoggerMessage" -DefaultValue "")
  trt11DllResolutionReportState = [string](Get-PropertyOrDefault -Object $trt11DllResolutionReport -Name "reportState" -DefaultValue "missing-trt11-runtime-dll-resolution-report")
  trt11DllResolutionMissingRequiredDllGroupCount = [int](Get-PropertyOrDefault -Object $trt11DllResolutionReport -Name "missingRequiredDllGroupCount" -DefaultValue -1)
  trt11DllResolutionDuplicateDllGroupCount = [int](Get-PropertyOrDefault -Object $trt11DllResolutionReport -Name "duplicateDllGroupCount" -DefaultValue -1)
  trt11DllResolutionOwnerActionRequired = [bool](Get-PropertyOrDefault -Object $trt11DllResolutionReport -Name "ownerActionRequired" -DefaultValue $true)
  trt11DllResolutionCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $trt11DllResolutionReport -Name "canPromoteRuntimeProof" -DefaultValue $false)
  trt10VsTrt11BridgeRuntimeDiagnosticDiffState = [string](Get-PropertyOrDefault -Object $trt10VsTrt11BridgeRuntimeDiagnosticDiff -Name "reportState" -DefaultValue "missing-trt10-vs-trt11-bridge-runtime-diagnostic-diff")
  trt10VsTrt11BridgeRuntimeDiagnosticDiffDifferingFieldCount = [int](Get-PropertyOrDefault -Object $trt10VsTrt11BridgeRuntimeDiagnosticDiff -Name "differingFieldCount" -DefaultValue 0)
  trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt10SmokePassed = [bool](Get-PropertyOrDefault -Object $trt10VsTrt11BridgeRuntimeDiagnosticDiff -Name "trt10SmokePassed" -DefaultValue $false)
  trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt11SmokeFailed = [bool](Get-PropertyOrDefault -Object $trt10VsTrt11BridgeRuntimeDiagnosticDiff -Name "trt11SmokeFailed" -DefaultValue $false)
  trt10VsTrt11BridgeRuntimeDiagnosticDiffFailureSignature = [string](Get-PropertyOrDefault -Object $trt10VsTrt11BridgeRuntimeDiagnosticDiff -Name "trt11FailureSignature" -DefaultValue "missing-failure-signature")
  trt10VsTrt11BridgeRuntimeDiagnosticDiffCanPromoteRuntimeProof = [bool](Get-PropertyOrDefault -Object $trt10VsTrt11BridgeRuntimeDiagnosticDiff -Name "canPromoteRuntimeProof" -DefaultValue $false)
  yoloVisionLicenseApprovalState = [string](Get-PropertyOrDefault -Object $yoloVisionLicenseApprovalValidation -Name "validationState" -DefaultValue "missing-yolovision-reference-asset-license-approval-validation")
  yoloVisionLicenseOwnerActionRequiredCount = [int](Get-PropertyOrDefault -Object $yoloVisionLicenseApprovalValidation -Name "ownerActionRequiredCount" -DefaultValue -1)
  yoloVisionAllAssetsApproved = [bool](Get-PropertyOrDefault -Object $yoloVisionLicenseApprovalValidation -Name "allAssetsApproved" -DefaultValue $false)
  deferredBTierWorkPackageState = [string](Get-PropertyOrDefault -Object $deferredBTierWorkPackage -Name "workPackageState" -DefaultValue "missing-deferred-btier-implementation-work-package")
  deferredBTierWorkItemCount = [int](Get-PropertyOrDefault -Object $deferredBTierWorkPackage -Name "workItemCount" -DefaultValue 0)
  deferredBTierWorkItemTargetCount = [int](Get-PropertyOrDefault -Object $deferredBTierWorkPackage -Name "workItemTargetCount" -DefaultValue 0)
  deferredBTierClosedWorkItemCount = [int](Get-PropertyOrDefault -Object $deferredBTierWorkPackage -Name "closedWorkItemCount" -DefaultValue 0)
  deferredBTierRemainingWorkItemCount = [int](Get-PropertyOrDefault -Object $deferredBTierWorkPackage -Name "remainingWorkItemCount" -DefaultValue 0)
  deferredSafetyTriageState = [string](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "recordKind" -DefaultValue "missing-deferred-candidate-safety-triage")
  deferredSafetyTriageTotalRows = [int](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "totalTriageRowCount" -DefaultValue 0)
  projectQualityShardRunbookState = $projectQualityShardRunbookState
  projectQualityShardRunbookReady = $projectQualityShardRunbookReady
}

$blockers = @(
  New-BlockerLane -Order 1 -Id "clean-public-package-consumer-proof" -Title "Clean public package consumer runtime proof" -CurrentState "$($sourceStates.cleanPublicPackageConsumerProofGapReportState); gaps=$($sourceStates.cleanPublicPackageConsumerProofGapCount); ownerActionRequired=$($sourceStates.cleanPublicPackageConsumerProofOwnerActionRequiredCount); ownerInputFailedActionRequired=$($sourceStates.cleanPublicPackageConsumerProofOwnerInputFailedActionRequiredCount)" -OwnerNextAction "Run repository-external clean consumer against public package source, capture restore/build/smoke logs and hashes, then run strict package-consumer-runtime validators." -RequiredEvidence @(
    "clean public package consumer proof gap report",
    "consumer path outside repository",
    "no ProjectReference",
    "no local feed as public proof",
    "no direct nupkg reference",
    "package id/version/source",
    "managed/runtime package hashes",
    "runtime smoke stdout/stderr logs",
    "smoke log SHA256",
    "host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
    "owner review identity"
  ) -Validators @(
    "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof",
    "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
    "eng/Export-ReleaseEvidenceBundle.ps1"
  ) -Rejects @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "build-only",
    "DependencyProbe-only",
    "bridge-only compatible-host smoke"
  ) -Boundary "This lane is not proof until strict validators accept real repository-external clean consumer logs, hashes, host metadata, package metadata, and owner review."
  New-BlockerLane -Order 2 -Id "post-publish-clean-consumer-proof" -Title "Post-publish clean consumer proof" -CurrentState ([string]$sourceStates.postPublishPreflightState) -OwnerNextAction "After public package publication, run a fresh repository-external clean consumer restore/build/smoke from the public source and import the post-publish proof record." -RequiredEvidence @(
    "public package source after publication",
    "downloaded package hashes",
    "post-publish restore/build/smoke logs",
    "post-publish runtime smoke log SHA256",
    "post-publish host metadata",
    "owner confirmation that pre-publish smoke is not reused"
  ) -Validators @(
    "eng/Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict",
    "eng/Test-PostPublishCleanConsumerRealProofGate.ps1 -Strict",
    "eng/Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"
  ) -Rejects @(
    "pre-publish smoke reused as post-publish proof",
    "local feed",
    "template",
    "dry-run",
    "dashboard-only evidence"
  ) -Boundary "Post-publish proof is a separate after-publication owner gate and cannot be satisfied by pre-publish smoke or local package sources."
  New-BlockerLane -Order 3 -Id "yolovision-asset-license-approval" -Title "YoloVision reference asset license approval" -CurrentState ([string]$sourceStates.yoloVisionLicenseApprovalState) -OwnerNextAction "Fill owner license approval record for model, labels, and image assets with license URI, redistribution decision, public repository decision, evidence URI, notes, signature, and approval scope." -RequiredEvidence @(
    "ownerName",
    "ownerSignature",
    "approvalDateUtc",
    "approvalScope",
    "per-asset ownerLicenseName",
    "per-asset ownerLicenseUri",
    "per-asset ownerRedistributionApproved",
    "per-asset ownerPublicRepositoryApproved",
    "per-asset ownerApprovalEvidenceUri",
    "per-asset ownerApprovalNotes"
  ) -Validators @(
    "eng/Test-YoloVisionAssetLicenseApprovalRecord.ps1",
    "eng/Export-ReleaseEvidenceBundle.ps1"
  ) -Rejects @(
    "hash-ready local asset as license approval",
    "TensorRT SLA as YOLO asset license approval",
    "acquisition success as redistribution approval"
  ) -Boundary "Asset SHA256 and acquisition success do not approve redistribution; owner-approved model, labels, and image license evidence is required."
  New-BlockerLane -Order 4 -Id "trt11-runtime-smoke-root-cause" -Title "TRT11 CUDA 13.2 runtime smoke root cause" -CurrentState "$($sourceStates.trt11ProofClassification); smoke=$($sourceStates.trt11SmokeStatus); exitCode=$($sourceStates.trt11ExitCode); rootCause=$($sourceStates.trt11RootCauseCategory); subcategory=$($sourceStates.trt11RootCauseSubcategory); signature=$($sourceStates.trt11RootCauseFailureSignature); cudaPreflight=$($sourceStates.trt11RootCauseCudaPreflightAvailable)/$($sourceStates.trt11RootCauseCudaPreflightAttempted)/driver=$($sourceStates.trt11RootCauseCudaPreflightDriverVersion)/runtime=$($sourceStates.trt11RootCauseCudaPreflightRuntimeVersion)/devices=$($sourceStates.trt11RootCauseCudaPreflightDeviceCount)/init=$($sourceStates.trt11RootCauseCudaPreflightInitStatus)/canAttemptTensorRT=$($sourceStates.trt11RootCauseCudaPreflightCanAttemptTensorRtRuntimeCreate); createDiag=$($sourceStates.trt11RootCauseNativeCreateRuntimeDiagnosticAvailable)/$($sourceStates.trt11RootCauseNativeCreateRuntimeAttempted)/$($sourceStates.trt11RootCauseNativeCreateRuntimeReturnedNull)/$($sourceStates.trt11RootCauseNativeCreateRuntimeLastStatus); phase=$($sourceStates.trt11RootCauseNativeCreateRuntimePhase); loggerMessages=$($sourceStates.trt11RootCauseNativeCreateRuntimeLoggerMessageCount); dllResolution=$($sourceStates.trt11DllResolutionReportState); missingDllGroups=$($sourceStates.trt11DllResolutionMissingRequiredDllGroupCount); duplicateDllGroups=$($sourceStates.trt11DllResolutionDuplicateDllGroupCount); diff=$($sourceStates.trt10VsTrt11BridgeRuntimeDiagnosticDiffState); diffFields=$($sourceStates.trt10VsTrt11BridgeRuntimeDiagnosticDiffDifferingFieldCount)" -OwnerNextAction "Investigate createInferRuntime null return by checking CUDA preflight driver/runtime/device/init markers, native create-runtime diagnostic markers, copied TensorRT logger message fields, vendor DLL order, CUDA/TensorRT/cuDNN compatibility, plugin initialization, dependency loading, and runtime version mismatch." -RequiredEvidence @(
    "TRT11 runtime smoke stdout/stderr",
    "TRT11 runtime smoke root-cause report",
    "TRT11 native create-runtime diagnostic snapshot",
    "TRT11 runtime DLL resolution report",
    "TRT10/TRT11 bridge runtime diagnostic diff",
    "PATH/DLL resolution order",
    "CUDA 13.2 runtime/toolkit metadata",
    "TensorRT 11 runtime metadata",
    "cuDNN 9.22 metadata",
    "dependency preflight output",
    "CUDA preflight output",
    "root-cause classification or passed runtime proof"
  ) -Validators @(
    "eng/Test-BridgePackageRuntimeConsumer.ps1 without -AllowRuntimeSmokeFailure",
    "cmake --preset win-x64-trt11-cuda13-release",
    "cmake --build --preset win-x64-trt11-cuda13-release --parallel"
  ) -Rejects @(
    "failed compatible-host attempt as runtime proof",
    "build success as runtime proof",
    "dependency probe as smoke pass"
  ) -Boundary "TRT11 failed compatible-host attempt is useful evidence but cannot promote runtime proof until smokeStatus=passed and strict proof conditions hold."
  New-BlockerLane -Order 5 -Id "deferred-readonly-implementation-batch" -Title "Deferred readonly/API design-gate implementation batch" -CurrentState "$($sourceStates.deferredBTierWorkPackageState); workItems=$($sourceStates.deferredBTierWorkItemCount)/$($sourceStates.deferredBTierWorkItemTargetCount); closed=$($sourceStates.deferredBTierClosedWorkItemCount); remaining=$($sourceStates.deferredBTierRemainingWorkItemCount)" -OwnerNextAction "Do not repeat btier-001 through btier-$($sourceStates.deferredBTierClosedWorkItemCount.ToString('D3')). Select a newly audited candidate or a separately evidenced runtime/model gap, then require native implementation, generated bindings, high-level wrapper, docs, smoke/quality tests, and version guards." -RequiredEvidence @(
    "deferred B-tier work-item proof closure ledger",
    "manifest entry with correct version guard",
    "native implementation",
    "generated interop",
    "pointer-free high-level wrapper",
    "docs or article update",
    "ProjectQuality test",
    "interface coverage refresh"
  ) -Validators @(
    "eng/Generate-Bindings.ps1",
    "eng/Test-BindingGeneratorOutputs.ps1",
    "eng/Export-InterfaceCoverageMatrix.ps1",
    "dotnet build .\\TensorRtSharp.sln -c Debug --no-restore"
  ) -Rejects @(
    "deleting deferred rows",
    "manifest-only completion",
    "public raw IntPtr creator/recorder/allocator",
    "callback trampoline without owner proof",
    "borrowed pointer with unclear lifetime"
  ) -Boundary "The $($sourceStates.deferredBTierClosedWorkItemCount) ledgered B-tier items have source-quality proof closure, not runtime or release proof. New API rows become real only after native/source, wrapper, version guard, docs, and tests agree."
  New-BlockerLane -Order 6 -Id "project-quality-shard-runbook" -Title "ProjectQuality shard runbook and gate" -CurrentState $projectQualityShardRunbookState -OwnerNextAction "Keep shard runbook and zh-cn article current when new release-heavy tests are added; continue recording passed TRX hashes and refreshing class coverage." -RequiredEvidence @(
    "artifacts/test-analysis/project-quality-shard-runbook.md",
    "docs/articles/zh-cn/project-quality-sharded-gate.md",
    "commands for class-level shards",
    "release-heavy test isolation guidance"
  ) -Validators @(
    "dotnet test .\\tests\\JYPPX.ProjectQuality.Tests\\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter <targeted-class>"
  ) -Rejects @(
    "claiming shard coverage is whole-suite pass",
    "claiming test inventory alone is release proof"
  ) -Boundary "Shard coverage is quality evidence and execution guidance only; it is not runtime proof, post-publish proof, or publish approval."
)

$record = [pscustomobject]@{
  schemaVersion = 1
  recordKind = "final-proof-readiness-blocker-dashboard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  dashboardState = "blocked-final-proof-readiness-owner-action-required"
  blockerCount = $blockers.Count
  ownerActionRequiredCount = @($blockers | Where-Object { $_.ownerActionRequired }).Count
  readyForPromotionCount = @($blockers | Where-Object { $_.readyForPromotion }).Count
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceStates = $sourceStates
  blockers = $blockers
  nextBatchRecommendedOrder = @(
    "clean-public-package-consumer-proof",
    "post-publish-clean-consumer-proof",
    "trt11-runtime-smoke-root-cause",
    "deferred-readonly-implementation-batch",
    "project-quality-shard-runbook",
    "yolovision-asset-license-approval"
  )
  sourceArtifacts = @(
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/final-release-close-blocker-dashboard.json",
    "artifacts/final-release/clean-consumer-external-proof-closure-pack-validation.json",
    "artifacts/final-release/clean-consumer-proof-execution-bundle.json",
    "artifacts/final-release/clean-public-package-consumer-proof-gap-report.json",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-preflight.json",
    "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json",
    "artifacts/final-release/post-publish-verification-owner-input-validation.json",
    "artifacts/package-consumer/bridge-runtime/win-x64-trt11.0-cuda13.2-cudnn9.22/bridge-package-runtime-consumer-proof.json",
    "artifacts/final-release/trt11-runtime-smoke-root-cause-report.json",
    "artifacts/final-release/trt11-runtime-dll-resolution-report.json",
    "artifacts/final-release/trt10-vs-trt11-bridge-runtime-diagnostic-diff.json",
    "artifacts/yolovision/reference-assets/asset-license-approval-validation.json",
    "artifacts/interface-coverage/deferred-btier-implementation-work-package.json",
    "artifacts/interface-coverage/deferred-btier-work-item-proof-closure-ledger.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json",
    "artifacts/test-analysis/project-quality-shard-runbook.md",
    "docs/articles/zh-cn/project-quality-sharded-gate.md"
  )
  boundary = "This dashboard aggregates final proof readiness blockers only. It does not run runtime smoke, does not publish packages, does not approve release close, does not promote package-consumer-runtime proof, and does not replace owner-supplied real evidence with validators."
}

$jsonPath = Join-Path $OutputRoot "final-proof-readiness-blocker-dashboard.json"
$markdownPath = Join-Path $OutputRoot "final-proof-readiness-blocker-dashboard.md"

$json = $record | ConvertTo-Json -Depth 16
Write-Utf8File -LiteralPath $jsonPath -InputObject $json

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Final Proof Readiness Blocker Dashboard")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| dashboardState | ``$($record.dashboardState)`` |")
$lines.Add("| blockerCount | ``$($record.blockerCount)`` |")
$lines.Add("| ownerActionRequiredCount | ``$($record.ownerActionRequiredCount)`` |")
$lines.Add("| readyForPromotionCount | ``$($record.readyForPromotionCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("| canPublishPublicly | ``$($record.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Source States")
$lines.Add("")
$lines.Add("- release evidence: ``$($sourceStates.releaseEvidenceBundleState)``; canPromoteRuntimeProof=``$($sourceStates.releaseCanPromoteRuntimeProof)``; canPublishPublicly=``$($sourceStates.releaseCanPublishPublicly)``; canCloseReleaseIssue=``$($sourceStates.releaseCanCloseReleaseIssue)``")
$lines.Add("- close dashboard: ``$($sourceStates.closeDashboardState)``; blockedBlockers=``$($sourceStates.closeDashboardBlockedBlockerCount)``")
$lines.Add("- clean proof execution bundle: ``$($sourceStates.cleanProofExecutionBundleState)``")
$lines.Add("- Clean public package consumer proof gap report: ``$($sourceStates.cleanPublicPackageConsumerProofGapReportState)``; gaps=``$($sourceStates.cleanPublicPackageConsumerProofGapCount)``; ownerActionRequired=``$($sourceStates.cleanPublicPackageConsumerProofOwnerActionRequiredCount)``; ownerInputFailedActionRequired=``$($sourceStates.cleanPublicPackageConsumerProofOwnerInputFailedActionRequiredCount)``; canPromoteRuntimeProof=``$($sourceStates.cleanPublicPackageConsumerProofCanPromoteRuntimeProof)``")
$lines.Add("- post-publish preflight: ``$($sourceStates.postPublishPreflightState)``; blockedCandidates=``$($sourceStates.postPublishPreflightBlockedProofCandidateCount)``; ownerActions=``$($sourceStates.postPublishPreflightFailedActionRequiredCount)``")
$lines.Add("- TRT11 runtime smoke: ``$($sourceStates.trt11ProofClassification)``; smoke=``$($sourceStates.trt11SmokeStatus)``; exitCode=``$($sourceStates.trt11ExitCode)``")
$lines.Add("- TRT11 root cause report: ``$($sourceStates.trt11RootCauseReportState)``; signature=``$($sourceStates.trt11RootCauseFailureSignature)``; category=``$($sourceStates.trt11RootCauseCategory)``; subcategory=``$($sourceStates.trt11RootCauseSubcategory)``; cudaPreflight=``$($sourceStates.trt11RootCauseCudaPreflightAvailable)/$($sourceStates.trt11RootCauseCudaPreflightAttempted)/driver=$($sourceStates.trt11RootCauseCudaPreflightDriverVersion)/runtime=$($sourceStates.trt11RootCauseCudaPreflightRuntimeVersion)/devices=$($sourceStates.trt11RootCauseCudaPreflightDeviceCount)/init=$($sourceStates.trt11RootCauseCudaPreflightInitStatus)``; createDiag=``$($sourceStates.trt11RootCauseNativeCreateRuntimeDiagnosticAvailable)/$($sourceStates.trt11RootCauseNativeCreateRuntimeAttempted)/$($sourceStates.trt11RootCauseNativeCreateRuntimeReturnedNull)/$($sourceStates.trt11RootCauseNativeCreateRuntimeLastStatus)``; phase=``$(ConvertTo-MarkdownCell $sourceStates.trt11RootCauseNativeCreateRuntimePhase)``; loggerMessages=``$(ConvertTo-MarkdownCell $sourceStates.trt11RootCauseNativeCreateRuntimeLoggerMessageCount)``; canPromoteRuntimeProof=``$($sourceStates.trt11RootCauseCanPromoteRuntimeProof)``")
$lines.Add("- TRT11 DLL resolution report: ``$($sourceStates.trt11DllResolutionReportState)``; missingRequiredDllGroups=``$($sourceStates.trt11DllResolutionMissingRequiredDllGroupCount)``; duplicateDllGroups=``$($sourceStates.trt11DllResolutionDuplicateDllGroupCount)``; ownerActionRequired=``$($sourceStates.trt11DllResolutionOwnerActionRequired)``; canPromoteRuntimeProof=``$($sourceStates.trt11DllResolutionCanPromoteRuntimeProof)``")
$lines.Add("- TRT10 vs TRT11 bridge diagnostic diff: ``$($sourceStates.trt10VsTrt11BridgeRuntimeDiagnosticDiffState)``; differingFields=``$($sourceStates.trt10VsTrt11BridgeRuntimeDiagnosticDiffDifferingFieldCount)``; trt10Passed=``$($sourceStates.trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt10SmokePassed)``; trt11Failed=``$($sourceStates.trt10VsTrt11BridgeRuntimeDiagnosticDiffTrt11SmokeFailed)``; canPromoteRuntimeProof=``$($sourceStates.trt10VsTrt11BridgeRuntimeDiagnosticDiffCanPromoteRuntimeProof)``")
$lines.Add("- YoloVision asset license approval: ``$($sourceStates.yoloVisionLicenseApprovalState)``; ownerActions=``$($sourceStates.yoloVisionLicenseOwnerActionRequiredCount)``; allAssetsApproved=``$($sourceStates.yoloVisionAllAssetsApproved)``")
$lines.Add("- deferred B-tier work package: ``$($sourceStates.deferredBTierWorkPackageState)``; workItems=``$($sourceStates.deferredBTierWorkItemCount)/$($sourceStates.deferredBTierWorkItemTargetCount)``")
$lines.Add("- ProjectQuality shard runbook: ``$($sourceStates.projectQualityShardRunbookState)``; ready=``$($sourceStates.projectQualityShardRunbookReady)``")
$lines.Add("")
$lines.Add("## Blocker Lanes")
$lines.Add("")
$lines.Add("| Order | ID | Current State | Owner Next Action | Boundary |")
$lines.Add("| ---: | --- | --- | --- | --- |")
foreach ($blocker in $blockers) {
  $lines.Add("| $($blocker.order) | ``$($blocker.id)`` | ``$(ConvertTo-MarkdownCell $blocker.currentState)`` | $(ConvertTo-MarkdownCell $blocker.ownerNextAction) | $(ConvertTo-MarkdownCell $blocker.boundary) |")
}
$lines.Add("")
$lines.Add("## Next Batch Recommended Order")
$lines.Add("")
foreach ($id in $record.nextBatchRecommendedOrder) {
  $lines.Add("- ``$id``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)

Write-Utf8File -LiteralPath $markdownPath -InputObject $lines

Write-Host "Final proof readiness blocker dashboard written to $jsonPath"
Write-Host "Final proof readiness blocker dashboard written to $markdownPath"
