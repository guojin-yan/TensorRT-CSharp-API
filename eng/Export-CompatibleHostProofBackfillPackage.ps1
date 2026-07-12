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

function New-BackfillStep {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Phase,
    [string]$Command,
    [string]$InputArtifact,
    [string]$OutputArtifact,
    [string]$RequiredFields,
    [string]$Validator,
    [string]$CurrentState,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    phase = $Phase
    command = $Command
    inputArtifact = $InputArtifact
    outputArtifact = $OutputArtifact
    requiredFields = $RequiredFields
    validator = $Validator
    currentState = $CurrentState
    ownerAction = "owner-action-required"
    performsPublish = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

function New-SampleAssetRequirement {
  param(
    [string]$SampleName,
    [string]$TaskScope,
    [string]$Template,
    [string]$CurrentState
  )

  [pscustomobject]@{
    sampleName = $SampleName
    taskScope = $TaskScope
    template = $Template
    currentState = $CurrentState
    requiredAssets = @(
      "modelPath/modelSha256",
      "labelsPath/labelsSha256",
      "inputAssetPath/inputAssetSha256",
      "model license and redistribution notes",
      "preprocessedInputTensorPath/preprocessedInputTensorSha256 when --input-data is used",
      "TensorRtExec build report and evidence sidecar",
      "sampleRunCommand",
      "sampleRunLogPath/sampleRunLogSha256",
      "stdoutSummary or stderrSummary",
      "proofClassification=real-model-runtime",
      "canPromoteRealModelRuntime=true"
    )
    forbiddenPromotion = "package-consumer-runtime"
    performsPublish = $false
    boundary = "Sample run evidence can promote only to real-model-runtime; it cannot replace package-consumer-runtime release proof."
  }
}

$ownerPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$finalGap = Read-JsonOrNull "artifacts\final-release\release-candidate-final-gap-review.json"
$preflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$externalValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$externalInputDraft = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record.input-template.json"
$externalRecordTemplate = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-record-template.json"
$linuxValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$sampleAssetAudit = Read-JsonOrNull "artifacts\user-acceptance\sample-asset-manifest-audit.json"
$realModelOwnerHandoff = Read-JsonOrNull "artifacts\user-acceptance\real-model-owner-handoff.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$cleanConsumerScan = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-project-scan.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"

$ownerPackageState = [string](Get-PropertyOrDefault -Object $ownerPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$finalGapState = [string](Get-PropertyOrDefault -Object $finalGap -Name "reviewState" -DefaultValue "missing-final-gap-review")
$preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$preflightFailedItemCount = [int](Get-PropertyOrDefault -Object $preflight -Name "failedItemCount" -DefaultValue -1)
$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$externalState = [string](Get-PropertyOrDefault -Object $externalValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalClassification = [string](Get-PropertyOrDefault -Object $externalValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$externalCanPromote = [bool](Get-PropertyOrDefault -Object $externalValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalFailedProofItemCount = [int](Get-PropertyOrDefault -Object $externalValidation -Name "failedProofItemCount" -DefaultValue -1)
$linuxState = [string](Get-PropertyOrDefault -Object $linuxValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxProof = [bool](Get-PropertyOrDefault -Object $linuxValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleRunState = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleCanPromoteRealModel = [bool](Get-PropertyOrDefault -Object $sampleRunValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$sampleManifestCount = [int](Get-PropertyOrDefault -Object $sampleAssetAudit -Name "manifestCount" -DefaultValue -1)
$sampleManifestErrorCount = [int](Get-PropertyOrDefault -Object $sampleAssetAudit -Name "errorCount" -DefaultValue -1)
$realModelHandoffState = [string](Get-PropertyOrDefault -Object $realModelOwnerHandoff -Name "handoffState" -DefaultValue "missing-real-model-owner-handoff")
$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$cleanConsumerScanState = [string](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "scanState" -DefaultValue "missing-clean-consumer-project-scan")
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)
$articleMatrix = Get-PropertyOrDefault -Object $finalGap -Name "articleMatrix" -DefaultValue $null
$articleCount = [int](Get-PropertyOrDefault -Object $articleMatrix -Name "articleCount" -DefaultValue -1)

$externalRuntimeRequiredFields = @(
  "recordKind=external-runtime-proof-record",
  "templateOnly=false",
  "proofClassification=package-consumer-runtime",
  "runtimePackageKey=$RuntimePackageKey",
  "managedNupkgSha256/runtimeNupkgSha256",
  "packageSource.runtimePackageKey",
  "clean consumer project identity",
  "host owner/machine/os/gpu/driver/CUDA/TensorRT/cuDNN metadata",
  "restore/build/dependency probe/runtime smoke commands",
  "smokeCommand includes --runtime-package-key $RuntimePackageKey",
  "stdoutSummary and stderrSummary",
  "smokeLogPath/smokeLogSha256",
  "validator-promoted runtime proof state from Test-ExternalRuntimeProofRecord.ps1"
)

$postPublishRequiredFields = @(
  "recordKind=post-publish-verification-record",
  "templateOnly=false",
  "postPublishProofClassification=post-publish-package-consumer-runtime",
  "selectedChannel/channelSourceUri",
  "managedPackageId/runtimePackageId/version",
  "managedPackageUrl/runtimePackageUrl",
  "downloaded managed/runtime nupkg SHA256 and timestamp",
  "clean consumer root outside repository",
  "no ProjectReference",
  "native assets copied",
  "dependency probe and runtime smoke logs",
  "stdoutSummary and stderrSummary",
  "all log SHA256 values match",
  "validator-promoted close readiness from Test-PostPublishVerificationRecord.ps1"
)

$ownerProofTracks = @(
  [pscustomobject]@{
    trackId = "package-consumer-runtime"
    proofClass = "package-consumer-runtime"
    inputJsonPath = "artifacts/final-release/external-runtime-proof-record.json"
    inputTemplatePath = "artifacts/final-release/external-runtime-proof-record.input-template.json"
    requiredOwnerInputs = $externalRuntimeRequiredFields
    logFields = @("command.logPath", "results.stdoutSummary", "results.stderrSummary", "results.failureDiagnostic")
    sha256Fields = @("packageSource.managedNupkgSha256", "packageSource.runtimeNupkgSha256", "command.logSha256")
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
    expectedArtifacts = @(
      "artifacts/final-release/external-runtime-proof-record.json",
      "artifacts/final-release/external-runtime-proof-validation.json",
      "artifacts/final-release/external-runtime-proof-validation.md"
    )
    promotionBlockers = @(
      "bridge-only package consumer log",
      "Skipped=True",
      "dependency-probe-only",
      "WrapperSurfaceEvidenceKind=compile-surface-proof",
      "IsRuntimeExecutionProof=False",
      "ProjectReference",
      "mismatched log SHA256"
    )
    currentState = "validationState=$externalState; canPromoteRuntimeProof=$externalCanPromote"
    canPromoteProof = $externalCanPromote
    performsPublish = $false
    canCloseReleaseIssue = $false
  }
  [pscustomobject]@{
    trackId = "linux-runner-proof"
    proofClass = "linux-runner-proof"
    inputJsonPath = "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json"
    inputTemplatePath = "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record-template.json"
    requiredOwnerInputs = @(
      "recordKind=linux-runner-evidence-record",
      "linuxRuntimePackageKey=$LinuxRuntimePackageKey",
      "real Linux x64 compatible host metadata",
      "CUDA/TensorRT/cuDNN runtime metadata",
      "runner command and log path",
      "runner log SHA256",
      "isRealLinuxRunnerProof=true only after validator proves it"
    )
    logFields = @("runnerLogPath", "stdoutSummary", "stderrSummary")
    sha256Fields = @("runnerLogSha256")
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey"
    expectedArtifacts = @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json",
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
    )
    promotionBlockers = @(
      "Windows handoff for Linux proof",
      "template-only record",
      "blocked-by-cuda-driver",
      "missing Linux runner log SHA256"
    )
    currentState = "validationState=$linuxState; isRealLinuxRunnerProof=$linuxProof"
    canPromoteProof = $linuxProof
    performsPublish = $false
    canCloseReleaseIssue = $false
  }
  [pscustomobject]@{
    trackId = "real-model-runtime"
    proofClass = "real-model-runtime"
    inputJsonPath = "artifacts/user-acceptance/sample-run-evidence-record.json"
    inputTemplatePath = "artifacts/user-acceptance/sample-run-evidence-record.template.json"
    requiredOwnerInputs = @(
      "Classification model/labels/input hashes and license notes",
      "YoloVision model/labels/input hashes and license notes",
      "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom coverage notes where applicable",
      "det/cls/seg/obb/pose/sem task evidence",
      "TensorRtExec build report and evidence sidecar",
      "sample runner log path and SHA256",
      "proofClassification=real-model-runtime",
      "canPromoteRealModelRuntime=true only after validator proves it"
    )
    logFields = @("sampleRunLogPath", "stdoutSummary", "stderrSummary", "TensorRtExec build report")
    sha256Fields = @("modelSha256", "labelsSha256", "inputAssetSha256", "preprocessedInputTensorSha256", "sampleRunLogSha256")
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    expectedArtifacts = @(
      "samples/assets/classification-assets.json",
      "samples/assets/yolovision-assets.json",
      "artifacts/user-acceptance/sample-run-evidence-record.json",
      "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
    )
    promotionBlockers = @(
      "build-only",
      "parse-only",
      "sidecar-only",
      "package-consumer-runtime classification",
      "missing model/input/license hashes"
    )
    currentState = "sampleRunEvidence=$sampleRunState; canPromoteRealModelRuntime=$sampleCanPromoteRealModel"
    canPromoteProof = $sampleCanPromoteRealModel
    performsPublish = $false
    canCloseReleaseIssue = $false
  }
  [pscustomobject]@{
    trackId = "post-publish-verification"
    proofClass = "post-publish-package-consumer-runtime"
    inputJsonPath = "artifacts/final-release/post-publish-verification-record.json"
    inputTemplatePath = "artifacts/final-release/post-publish-verification-record.input-draft.json"
    requiredOwnerInputs = $postPublishRequiredFields
    logFields = @("restoreLogPath", "nativeAssetListingPath", "dependencyProbeLogPath", "smokeLogPath", "stdoutSummary", "stderrSummary")
    sha256Fields = @("managedNupkgSha256", "runtimeNupkgSha256", "restoreLogSha256", "nativeAssetListingSha256", "dependencyProbeLogSha256", "smokeLogSha256")
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof"
    expectedArtifacts = @(
      "artifacts/final-release/post-publish-verification-record.json",
      "artifacts/final-release/post-publish-verification-validation.json",
      "artifacts/final-release/post-publish-verification-validation.md"
    )
    promotionBlockers = @(
      "template",
      "draft",
      "local feed",
      "ProjectReference",
      "bridge-only package consumer log",
      "dependency-probe-only",
      "mismatched log SHA256",
      "missing execution steps"
    )
    currentState = "validationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof"
    canPromoteProof = $postPublishProof
    performsPublish = $false
    canCloseReleaseIssue = $false
  }
)

$backfillSteps = @(
  New-BackfillStep `
    -Id "refresh-owner-release-execution-package" `
    -Title "Refresh owner release execution package" `
    -Phase "preflight-refresh" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerReleaseExecutionPackage.ps1" `
    -InputArtifact "artifacts/final-release/release-candidate-final-gap-review.json" `
    -OutputArtifact "artifacts/final-release/owner-release-execution-package.json" `
    -RequiredFields "performsPublish=false; canCloseReleaseIssue=false; manual publish placeholders only" `
    -Validator "ReleaseCandidateReadinessTests.OwnerReleaseExecutionPackageKeepsPublishActionsManualAndProofBoundariesExplicit" `
    -CurrentState "packageState=$ownerPackageState; preflightState=$preflightState; failedItemCount=$preflightFailedItemCount" `
    -Boundary "Owner package is guidance only and does not execute dotnet nuget push."

  New-BackfillStep `
    -Id "prepare-external-runtime-proof-record" `
    -Title "Prepare package-consumer-runtime external proof record" `
    -Phase "compatible-host-prepublish" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordInputTemplate.ps1" `
    -InputArtifact "artifacts/final-release/external-runtime-proof-record-template.json" `
    -OutputArtifact "artifacts/final-release/external-runtime-proof-record.input-template.json" `
    -RequiredFields ($externalRuntimeRequiredFields -join "; ") `
    -Validator "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -CurrentState "validationState=$externalState; proofClassification=$externalClassification; failedProofItemCount=$externalFailedProofItemCount; canPromoteRuntimeProof=$externalCanPromote" `
    -Boundary "template, draft, local feed, ProjectReference, DependencyProbe, build-only, parse-only, sidecar-only, and blocked-by-cuda-driver cannot substitute package-consumer-runtime proof."

  New-BackfillStep `
    -Id "collect-linux-runner-evidence" `
    -Title "Collect Linux compatible host runner evidence" `
    -Phase "compatible-host-linux" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey" `
    -InputArtifact "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record-template.json" `
    -OutputArtifact "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json" `
    -RequiredFields "Linux host metadata; runtime package key; command log; validator output; CUDA/TensorRT/cuDNN versions" `
    -Validator "Test-LinuxRunnerEvidenceRecord.ps1" `
    -CurrentState "validationState=$linuxState; isRealLinuxRunnerProof=$linuxProof" `
    -Boundary "Windows handoff, template-only records, and blocked-by-cuda-driver are not Linux runner proof."

  New-BackfillStep `
    -Id "collect-real-model-sample-evidence" `
    -Title "Collect Classification/YoloVision real-model-runtime evidence" `
    -Phase "real-model-samples" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -InputArtifact "samples/assets/*.template.json" `
    -OutputArtifact "artifacts/user-acceptance/sample-run-evidence-record-validation.json" `
    -RequiredFields "model/labels/input hashes; license; TensorRtExec sidecar; sample runner log; stdout/stderr summary; canPromoteRealModelRuntime=true" `
    -Validator "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1" `
    -CurrentState "handoffState=$realModelHandoffState; sampleRunEvidence=$sampleRunState; manifestCount=$sampleManifestCount; errorCount=$sampleManifestErrorCount; canPromoteRealModelRuntime=$sampleCanPromoteRealModel" `
    -Boundary "Classification/YoloVision sample proof can promote only to real-model-runtime and cannot claim package-consumer-runtime."

  New-BackfillStep `
    -Id "prepare-post-publish-verification" `
    -Title "Prepare post-publish clean consumer verification" `
    -Phase "post-publish" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -InputArtifact "artifacts/final-release/post-publish-verification-record-template.json" `
    -OutputArtifact "artifacts/final-release/post-publish-verification-validation.json" `
    -RequiredFields ($postPublishRequiredFields -join "; ") `
    -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -CurrentState "validationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof; cleanConsumerScanState=$cleanConsumerScanState" `
    -Boundary "Post-publish verification requires a real channel publish; local feed, ProjectReference, draft, helper scan, and collection package cannot close the release issue."

  New-BackfillStep `
    -Id "refresh-release-close-preflight" `
    -Title "Refresh release close preflight after proof backfill" `
    -Phase "release-close-review" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1" `
    -InputArtifact "artifacts/final-release/release-evidence-bundle.json" `
    -OutputArtifact "artifacts/final-release/release-close-preflight.json" `
    -RequiredFields "all proof items pass before canCloseReleaseIssue can become true" `
    -Validator "Export-ReleaseClosePreflight.ps1" `
    -CurrentState "preflightState=$preflightState; failedItemCount=$preflightFailedItemCount" `
    -Boundary "Preflight aggregates proof state but is not owner authorization, runtime proof, post-publish proof, or package push."
)

$sampleRequirements = @(
  New-SampleAssetRequirement `
    -SampleName "Classification" `
    -TaskScope "classification/top-k classifier with user-provided ONNX and labels" `
    -Template "samples/assets/classification-assets.template.json" `
    -CurrentState "candidate-not-downloaded; template-only; owner-action-required"

  New-SampleAssetRequirement `
    -SampleName "YoloVision" `
    -TaskScope "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom; det/cls/seg/obb/pose/sem; det、cls、seg、obb、pose、sem" `
    -Template "samples/assets/yolovision-assets.template.json" `
    -CurrentState "candidate-not-downloaded; template-only; owner-action-required"
)

$nonSubstituteProofKinds = @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "DependencyProbe",
  "build-only",
  "parse-only",
  "sidecar-only",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "bridge-only package consumer log",
  "bridge-only wrapper surface",
  "Skipped=True",
  "WrapperSurfaceEvidenceKind=compile-surface-proof",
  "IsRuntimeExecutionProof=False",
  "mismatched log SHA256",
  "Parser/ParserRefitter diagnostic snapshots",
  "copied managed diagnostic snapshot",
  "owner-action-required without real logs",
  "Windows handoff for Linux proof"
)

$packageState = if (
  $externalCanPromote -and
  $linuxProof -and
  $sampleCanPromoteRealModel -and
  $postPublishProof -and
  [string]::Equals($preflightState, "ready-to-close", [System.StringComparison]::OrdinalIgnoreCase)
) {
  "ready-for-owner-close-review"
}
else {
  "owner-action-required"
}

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "compatible-host-proof-backfill-package"
  packageState = $packageState
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  requiresCompatibleHost = $true
  requiresHumanOwner = $true
  ownerReleaseExecutionPackageState = $ownerPackageState
  finalGapReviewState = $finalGapState
  releaseEvidenceBundleState = $releaseEvidenceState
  preflightState = $preflightState
  preflightFailedItemCount = $preflightFailedItemCount
  staleFindingCount = $staleFindingCount
  articleCount = $articleCount
  externalRuntimeProof = [ordered]@{
    validationState = $externalState
    proofClassification = $externalClassification
    canPromoteRuntimeProof = $externalCanPromote
    failedProofItemCount = $externalFailedProofItemCount
    requiredFields = $externalRuntimeRequiredFields
  }
  linuxRunnerProof = [ordered]@{
    validationState = $linuxState
    isRealLinuxRunnerProof = $linuxProof
  }
  sampleEvidence = [ordered]@{
    handoffState = $realModelHandoffState
    validationState = $sampleRunState
    manifestCount = $sampleManifestCount
    errorCount = $sampleManifestErrorCount
    canPromoteRealModelRuntime = $sampleCanPromoteRealModel
    requirements = $sampleRequirements
  }
  postPublishVerification = [ordered]@{
    validationState = $postPublishState
    isPostPublishVerificationProof = $postPublishProof
    cleanConsumerScanState = $cleanConsumerScanState
    requiredFields = $postPublishRequiredFields
  }
  ownerProofFinalBackfillTracks = $ownerProofTracks
  requiredOwnerInputs = @($ownerProofTracks | ForEach-Object { $_.requiredOwnerInputs } | Select-Object -Unique)
  validatorCommands = @($ownerProofTracks | ForEach-Object { $_.validatorCommand } | Select-Object -Unique)
  expectedArtifacts = @($ownerProofTracks | ForEach-Object { $_.expectedArtifacts } | Select-Object -Unique)
  promotionBlockers = @($ownerProofTracks | ForEach-Object { $_.promotionBlockers } | Select-Object -Unique)
  backfillSteps = $backfillSteps
  nonSubstituteProofKinds = $nonSubstituteProofKinds
  sourceArtifacts = @(
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/release-candidate-final-gap-review.json",
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-project-scan.json",
    "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
    "artifacts/user-acceptance/sample-asset-manifest-audit.json",
    "artifacts/user-acceptance/real-model-owner-handoff.json",
    "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
  )
  safetyNotes = @(
    "This package is owner guidance only and does not publish packages.",
    "performsPublish=false and canCloseReleaseIssue=false remain fixed until real owner proof exists.",
    "package-consumer-runtime proof must come from a clean consumer runtime smoke on a compatible host.",
    "Classification/YoloVision sample evidence can promote only to real-model-runtime.",
    "post-publish verification requires a real channel publish and clean consumer package identity.",
    "blocked-by-cuda-driver is an environment blocker, not smoke passed.",
    "ProjectReference, local feed, build-only, parse-only, and sidecar-only cannot substitute release proof."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "compatible-host-proof-backfill-package.json"
$markdownPath = Join-Path $artifactRoot "compatible-host-proof-backfill-package.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$stepRows = $backfillSteps | ForEach-Object {
  $step = $_
  "| ``$($step.id)`` | $($step.phase) | $($step.currentState) | ``$($step.validator)`` | $($step.boundary) |"
}

$sampleRows = $sampleRequirements | ForEach-Object {
  $sample = $_
  "| ``$($sample.sampleName)`` | $($sample.taskScope) | ``$($sample.template)`` | $($sample.currentState) | $($sample.boundary) |"
}

$trackRows = $ownerProofTracks | ForEach-Object {
  $track = $_
  "| ``$($track.trackId)`` | ``$($track.proofClass)`` | ``$($track.inputJsonPath)`` | ``$($track.validatorCommand)`` | $($track.currentState) |"
}

$requiredOwnerInputLines = $record.requiredOwnerInputs | ForEach-Object { "- ``$_``" }
$validatorCommandLines = $record.validatorCommands | ForEach-Object { "- ``$_``" }
$expectedArtifactLines = $record.expectedArtifacts | ForEach-Object { "- ``$_``" }
$promotionBlockerLines = $record.promotionBlockers | ForEach-Object { "- ``$_``" }
$externalFieldLines = $externalRuntimeRequiredFields | ForEach-Object { "- ``$_``" }
$postPublishFieldLines = $postPublishRequiredFields | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }

$markdown = @"
# Compatible Host Proof Backfill Package

生成时间：$($record.generatedAtUtc)

## 总结

该回填包把 Owner Release Execution Package 进一步拆成 compatible host proof、Linux runner proof、Classification/YoloVision real-model-runtime proof 和 post-publish verification proof 的可执行材料清单。它是 owner guidance，不执行真实发布，不伪造 proof。``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前状态

| 项目 | 当前值 |
|---|---|
| packageState | ``$packageState`` |
| ownerReleaseExecutionPackageState | ``$ownerPackageState`` |
| finalGapReviewState | ``$finalGapState`` |
| releaseEvidenceBundleState | ``$releaseEvidenceState`` |
| preflightState | ``$preflightState`` |
| preflightFailedItemCount | ``$preflightFailedItemCount`` |
| runtimePackageKey | ``$RuntimePackageKey`` |
| linuxRuntimePackageKey | ``$LinuxRuntimePackageKey`` |
| staleFindingCount | ``$staleFindingCount`` |
| articleCount | ``$articleCount`` |

## Compatible Host 回填步骤

| ID | 阶段 | 当前状态 | Validator | 边界 |
|---|---|---|---|---|
$($stepRows -join "`r`n")

## Owner Proof Final Backfill Tracks

| Track | Proof class | Input JSON | Validator command | Current state |
|---|---|---|---|---|
$($trackRows -join "`r`n")

## Required Owner Inputs

$($requiredOwnerInputLines -join "`r`n")

## Validator Commands

$($validatorCommandLines -join "`r`n")

## Expected Artifacts

$($expectedArtifactLines -join "`r`n")

## Promotion Blockers

$($promotionBlockerLines -join "`r`n")

## External Runtime Proof 必填字段

$($externalFieldLines -join "`r`n")

## Post-Publish Verification 必填字段

$($postPublishFieldLines -join "`r`n")

## Classification / YoloVision 样例实证

| Sample | Task scope | Template | 当前状态 | 边界 |
|---|---|---|---|---|
$($sampleRows -join "`r`n")

YoloVision 支持 YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom，以及 det/cls/seg/obb/pose/sem（即 det、cls、seg、obb、pose、sem）。真实样例 proof 仍需要 owner 提供模型、labels、输入资产、license、SHA256、TensorRtExec sidecar、sample runner log 和 sample-run-evidence record。样例 proof 只能晋级 real-model-runtime，不能晋级 package-consumer-runtime。

## 不可替代材料

$($nonSubstituteLines -join "`r`n")

## 来源材料

$($sourceLines -join "`r`n")

## Safety Notes

$($safetyLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Compatible host proof backfill package written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackageState=$packageState"
Write-Output "PerformsPublish=False"
Write-Output "CanCloseReleaseIssue=False"
