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

function Get-ArrayOrEmpty {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue @()
  if ($null -eq $value) {
    return @()
  }

  return @($value)
}

function New-GapItem {
  param(
    [string]$GapId,
    [string]$Title,
    [string]$ProofClass,
    [string]$SourcePreflightItem,
    [string]$CurrentState,
    [string[]]$RequiredRealInputs,
    [string]$Validator,
    [string]$OwnerAction,
    [string]$Boundary,
    [string[]]$RelatedArtifacts,
    [string[]]$NonSubstitutes
  )

  [pscustomobject]@{
    gapId = $GapId
    title = $Title
    proofClass = $ProofClass
    sourcePreflightItem = $SourcePreflightItem
    passed = $false
    currentState = $CurrentState
    requiredRealInputs = $RequiredRealInputs
    validator = $Validator
    ownerAction = $OwnerAction
    boundary = $Boundary
    relatedArtifacts = $RelatedArtifacts
    nonSubstitutes = $NonSubstitutes
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$preflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$promotionIssue = Read-JsonOrNull "artifacts\final-release\release-promotion-issue-record.json"
$inputPackage = Read-JsonOrNull "artifacts\final-release\real-model-and-package-proof-input-package.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$deferredSafetyTriage = Read-JsonOrNull "artifacts\interface-coverage\deferred-candidate-safety-triage.json"

$preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$preflightFailedItemCount = [int](Get-PropertyOrDefault -Object $preflight -Name "failedItemCount" -DefaultValue -1)
$preflightCanClose = [bool](Get-PropertyOrDefault -Object $preflight -Name "canCloseReleaseIssue" -DefaultValue $false)
$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$releaseEvidenceComplete = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isReleaseEvidenceComplete" -DefaultValue $false)
$promotionState = [string](Get-PropertyOrDefault -Object $promotionIssue -Name "promotionState" -DefaultValue "missing-release-promotion-issue-record")
$inputPackageState = [string](Get-PropertyOrDefault -Object $inputPackage -Name "packageState" -DefaultValue "missing-real-model-and-package-proof-input-package")
$ownerReleaseExecutionPackageState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$oneScreenReleaseHoldChecklist = @(Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "oneScreenReleaseHoldChecklist" -DefaultValue @())
$oneScreenReleaseHoldChecklistCount = if ($oneScreenReleaseHoldChecklist.Count -gt 0) {
  $oneScreenReleaseHoldChecklist.Count
}
else {
  [int](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "oneScreenReleaseHoldChecklistCount" -DefaultValue 0)
}

$externalProofState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofState" -DefaultValue "missing-external-runtime-proof-validation")
$externalCanPromote = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canPromoteRuntimeProof" -DefaultValue $false)
$externalFailedProofItemCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "externalRuntimeProofFailedProofItemCount" -DefaultValue -1)
$ownerPlanState = [string](Get-PropertyOrDefault -Object $preflight -Name "ownerPlanValidationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $promotionIssue -Name "ownerApprovalInputValidationStatus" -DefaultValue "missing-owner-authorization")))
$linuxProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "sampleRunEvidenceValidationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleCanPromote = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "sampleRunEvidenceCanPromoteRealModelRuntime" -DefaultValue $false)
$postPublishState = [string](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishVerificationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishProof = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidence -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishRequiredEvidence = @(
  Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishRequiredEvidence" -DefaultValue @(
    Get-PropertyOrDefault -Object $preflight -Name "postPublishRequiredEvidence" -DefaultValue @()
  )
)
$postPublishRequiredEvidence = Normalize-PostPublishRequiredEvidence -Evidence $postPublishRequiredEvidence
$postPublishRequiredEvidenceCount = if ($postPublishRequiredEvidence.Count -gt 0) {
  $postPublishRequiredEvidence.Count
}
else {
  [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "postPublishRequiredEvidenceCount" -DefaultValue (
    [int](Get-PropertyOrDefault -Object $preflight -Name "postPublishRequiredEvidenceCount" -DefaultValue 0)
  ))
}
$releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount = [int](Get-PropertyOrDefault -Object $releaseEvidence -Name "releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount" -DefaultValue $postPublishRequiredEvidenceCount)
$deferredSafetyTriageKind = [string](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "triageKind" -DefaultValue "missing-deferred-candidate-safety-triage")
$deferredSafetyTriageTotalRows = [int](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "totalTriageRowCount" -DefaultValue 0)
$deferredSafetyTierSummaries = Get-ArrayOrEmpty -Object $deferredSafetyTriage -Name "tierSummaries"
$deferredSafetyTierAImmediateSafeCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "A - immediate-safe" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTierBSafeAlternativeCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "B - safe-alternative-or-alias" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTierCDesignGateCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "C - design-gate-required" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTierDKeepDeferredCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "D - keep-deferred" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTriageState = if ($deferredSafetyTriageKind -eq "deferred-candidate-safety-triage") { "triage-ready-planning-input-only" } else { "missing-deferred-safety-triage" }
$deferredSafetyTriageProofBoundary = "Deferred safety triage is planning and boundary disclosure only: A rows are implementation candidates, B rows are safe-alternative/alias proof closures, C rows require design gates, and D rows must stay deferred; none are package-consumer-runtime proof, owner approval, post-publish verification, or permission to delete deferred records."

$commonNonSubstitutes = @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "helper",
  "build-only",
  "parse-only",
  "sidecar-only",
  "DependencyProbe",
  "dependency-probe-only",
  "Skipped=True",
  "blocked-by-cuda-driver",
  "bridge-only package consumer log",
  "bridge-only wrapper surface",
  "WrapperSurfaceEvidenceKind=compile-surface-proof",
  "IsRuntimeExecutionProof=False",
  "mismatched log SHA256",
  "Parser/ParserRefitter diagnostic snapshots",
  "copied managed diagnostic snapshot",
  "Windows handoff for Linux proof",
  "owner-action-required without validator pass",
  "deferred safety triage",
  "safe-alternative-or-alias planning input",
  "design-gate-required planning input",
  "keep-deferred boundary disclosure"
)

$gapItems = @(
  New-GapItem `
    -GapId "owner-authorization" `
    -Title "Owner authorization and manual publish command materialization" `
    -ProofClass "owner-authorization" `
    -SourcePreflightItem "owner-authorized-command-plan" `
    -CurrentState "ownerPlanValidationState=$ownerPlanState; promotionState=$promotionState; canPublishPublicly=false" `
    -RequiredRealInputs @(
      "explicit non-template owner approval input record",
      "owner-approved channel choice",
      "manual command materialization outside automation",
      "NVIDIA redistribution disposition",
      "stale claim audit remains clean"
    ) `
    -Validator "Test-ReleaseOwnerApprovalInput.ps1 + Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -OwnerAction "Provide explicit owner authorization while keeping publish commands manual and outside these scripts." `
    -Boundary "Owner guidance and placeholder commands do not authorize publication and do not execute dotnet nuget push." `
    -RelatedArtifacts @(
      "artifacts/final-release/release-owner-approval-input-validation.json",
      "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
      "artifacts/final-release/release-promotion-issue-record.json"
    ) `
    -NonSubstitutes $commonNonSubstitutes

  New-GapItem `
    -GapId "package-consumer-runtime" `
    -Title "Package consumer runtime proof on compatible host" `
    -ProofClass "package-consumer-runtime" `
    -SourcePreflightItem "external-runtime-proof-record" `
    -CurrentState "externalRuntimeProofState=$externalProofState; canPromoteRuntimeProof=$externalCanPromote; failedProofItemCount=$externalFailedProofItemCount; preflightCanClose=$preflightCanClose" `
    -RequiredRealInputs @(
      "managed/runtime nupkg SHA256 from exact release candidate artifacts",
      "runtimePackageKey=$RuntimePackageKey",
      "clean consumer project outside repository",
      "host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
      "restore/build/dependency probe/runtime smoke logs",
      "stdoutSummary and stderrSummary from reviewed real log",
      "smokeLogPath/smokeLogSha256 from an existing log",
      "no ProjectReference"
    ) `
    -Validator "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -OwnerAction "Run clean package consumer smoke on a compatible CUDA/TensorRT host and validate the filled external-runtime-proof-record.json." `
    -Boundary "blocked-by-cuda-driver is an environment blocker, not smoke passed; bridge-only wrapper surface, WrapperSurfaceEvidenceKind=compile-surface-proof, IsRuntimeExecutionProof=False, Parser/ParserRefitter diagnostic snapshots, input package, and collection package are not runtime proof." `
    -RelatedArtifacts @(
      "artifacts/final-release/external-runtime-proof-record.input-template.json",
      "artifacts/final-release/external-runtime-proof-validation.json",
      "artifacts/final-release/real-model-and-package-proof-input-package.json"
    ) `
    -NonSubstitutes $commonNonSubstitutes

  New-GapItem `
    -GapId "linux-runner-proof" `
    -Title "Linux runner proof" `
    -ProofClass "linux-runner-proof" `
    -SourcePreflightItem "full-acceptance-close-readiness" `
    -CurrentState "linuxRuntimePackageKey=$LinuxRuntimePackageKey; isRealLinuxRunnerProof=$linuxProof" `
    -RequiredRealInputs @(
      "real Linux x64 runner host metadata",
      "runtime package key $LinuxRuntimePackageKey",
      "command log and validator output",
      "CUDA/TensorRT/cuDNN versions from runner host",
      "real runner artifact captured from Linux host"
    ) `
    -Validator "Test-LinuxRunnerEvidenceRecord.ps1" `
    -OwnerAction "Collect Linux runner evidence on a real Linux x64 runner instead of using Windows handoff or template-only files." `
    -Boundary "Windows handoff, template-only record, and dry-run-only evidence are not Linux runner proof." `
    -RelatedArtifacts @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record-template.json",
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
    ) `
    -NonSubstitutes $commonNonSubstitutes

  New-GapItem `
    -GapId "real-model-runtime" `
    -Title "Classification and YoloVision real model runtime proof" `
    -ProofClass "real-model-runtime" `
    -SourcePreflightItem "full-acceptance-close-readiness" `
    -CurrentState "sampleRunEvidenceValidationState=$sampleState; canPromoteRealModelRuntime=$sampleCanPromote; inputPackageState=$inputPackageState" `
    -RequiredRealInputs @(
      "Classification model/labels/input/license/SHA256",
      "YoloVision model/labels/input/license/SHA256",
      "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom metadata",
      "det/cls/seg/obb/pose/sem and det、cls、seg、obb、pose、sem task notes",
      "TensorRtExec sidecar and build report",
      "sample runner command and real sample runner log",
      "stdoutSummary and stderrSummary from reviewed real sample log",
      "sample-run-evidence record with proofClassification=real-model-runtime"
    ) `
    -Validator "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -OwnerAction "Provide real Classification/YoloVision assets and validate sample-run-evidence after running the sample runner." `
    -Boundary "Real model sample proof can promote only to real-model-runtime; it never replaces package-consumer-runtime release proof." `
    -RelatedArtifacts @(
      "samples/assets/classification-assets.template.json",
      "samples/assets/yolovision-assets.template.json",
      "artifacts/user-acceptance/sample-run-evidence-record.template.json",
      "artifacts/final-release/real-model-and-package-proof-input-package.json"
    ) `
    -NonSubstitutes $commonNonSubstitutes

  New-GapItem `
    -GapId "post-publish-verification" `
    -Title "Post-publish verification proof" `
    -ProofClass "post-publish verification" `
    -SourcePreflightItem "post-publish-verification-record" `
    -CurrentState "postPublishVerificationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof; canCloseReleaseIssue=$postPublishCanClose" `
    -RequiredRealInputs @(
      $postPublishRequiredEvidence
      "real channel package URL and selected release channel",
      "downloaded managed/runtime nupkg SHA256",
      "clean consumer root outside repository",
      "no ProjectReference",
      "native assets listing from downloaded packages",
      "dependency probe log",
      "runtime smoke log",
      "stdoutSummary and stderrSummary from reviewed real post-publish logs"
    ) `
    -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -OwnerAction "After owner-approved real publication, validate a clean external consumer against the real channel packages." `
    -Boundary "Post-publish verification requires real channel packages; local feed, ProjectReference, helper scan, draft, or input package cannot close the release issue." `
    -RelatedArtifacts @(
      "artifacts/final-release/post-publish-verification-record.input-draft.json",
      "artifacts/final-release/post-publish-verification-validation.json",
      "artifacts/final-release/real-model-and-package-proof-input-package.json"
    ) `
    -NonSubstitutes $commonNonSubstitutes
)

$failedPreflightItems = Get-ArrayOrEmpty -Object $preflight -Name "preflightItems" | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) }
$canCloseReleaseIssue = $false
$dashboardState = if ($preflightState -eq "ready-for-release-close-owner-review" -and @($gapItems).Count -eq 0) { "ready-for-release-close-owner-review" } else { "blocked-real-proof-required" }

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-close-gap-dashboard"
  dashboardState = $dashboardState
  packageState = "owner-action-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $canCloseReleaseIssue
  ownerActionStatus = "owner-action-required"
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isRealModelRuntimeProof = $false
  isPostPublishVerificationProof = $false
  preflightState = $preflightState
  preflightFailedItemCount = $preflightFailedItemCount
  releaseEvidenceBundleState = $releaseEvidenceState
  isReleaseEvidenceComplete = $releaseEvidenceComplete
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidenceCount
  releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount = $releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount
  releasePromotionIssueState = $promotionState
  realModelAndPackageProofInputPackageState = $inputPackageState
  ownerReleaseExecutionPackageState = $ownerReleaseExecutionPackageState
  oneScreenReleaseHoldChecklist = @($oneScreenReleaseHoldChecklist)
  oneScreenReleaseHoldChecklistCount = $oneScreenReleaseHoldChecklistCount
  deferredSafetyTriageState = $deferredSafetyTriageState
  deferredSafetyTriageKind = $deferredSafetyTriageKind
  deferredSafetyTriageTotalRows = $deferredSafetyTriageTotalRows
  deferredSafetyTierAImmediateSafeCount = $deferredSafetyTierAImmediateSafeCount
  deferredSafetyTierBSafeAlternativeCount = $deferredSafetyTierBSafeAlternativeCount
  deferredSafetyTierCDesignGateCount = $deferredSafetyTierCDesignGateCount
  deferredSafetyTierDKeepDeferredCount = $deferredSafetyTierDKeepDeferredCount
  deferredSafetyTriageIsReleaseProof = $false
  deferredSafetyTriageCanPublishPublicly = $false
  deferredSafetyTriageCanCloseReleaseIssue = $false
  deferredSafetyTriageProofBoundary = $deferredSafetyTriageProofBoundary
  failedPreflightItems = $failedPreflightItems
  gapCount = @($gapItems).Count
  gapItems = $gapItems
  sourceEvidence = @(
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-promotion-issue-record.json",
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/real-model-and-package-proof-input-package.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json"
  )
  sourceArtifacts = @(
    "artifacts/final-release/release-close-preflight.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-promotion-issue-record.json",
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/owner-release-execution-package.md",
    "artifacts/final-release/real-model-and-package-proof-input-package.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.md"
  )
  nonSubstituteProofKinds = $commonNonSubstitutes
  safetyNotes = @(
    "Release close gap dashboard is owner guidance only and does not publish packages.",
    "performsPublish=false, canPublishPublicly=false, and canCloseReleaseIssue=false remain fixed for this dashboard.",
    "package-consumer-runtime requires a clean package consumer runtime smoke on a compatible host.",
    "real-model-runtime requires real Classification/YoloVision assets, hashes, licenses, and sample runner logs.",
    "post-publish verification requires a real channel package, downloaded hashes, clean consumer logs, and -FailOnNotProof validation.",
    "blocked-by-cuda-driver is an environment blocker, not smoke passed.",
    $deferredSafetyTriageProofBoundary,
    "Bridge-only package consumer logs, bridge-only wrapper surface evidence, WrapperSurfaceEvidenceKind=compile-surface-proof, IsRuntimeExecutionProof=False, Skipped=True, dependency-probe-only, mismatched log SHA256, Windows handoff for Linux proof, and copied Parser/ParserRefitter diagnostic snapshots cannot substitute runtime proof.",
    "ProjectReference, local feed, template, draft, runbook, collection package, input package, build-only, parse-only, and sidecar-only cannot substitute proof."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "release-close-gap-dashboard.json"
$markdownPath = Join-Path $artifactRoot "release-close-gap-dashboard.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$gapRows = $gapItems | ForEach-Object {
  "| ``$($_.gapId)`` | ``$($_.proofClass)`` | ``$($_.passed)`` | $($_.currentState.Replace("|", "\|")) | ``$($_.validator)`` | $($_.boundary.Replace("|", "\|")) |"
}
$oneScreenReleaseHoldRows = $oneScreenReleaseHoldChecklist | ForEach-Object {
  $item = $_
  $id = ([string](Get-PropertyOrDefault -Object $item -Name "id" -DefaultValue "")).Replace("|", "\|")
  $blocker = ([string](Get-PropertyOrDefault -Object $item -Name "ownerVisibleBlocker" -DefaultValue "")).Replace("|", "\|")
  $currentState = ([string](Get-PropertyOrDefault -Object $item -Name "currentState" -DefaultValue "")).Replace("|", "\|")
  $nextAction = ([string](Get-PropertyOrDefault -Object $item -Name "ownerNextAction" -DefaultValue "")).Replace("|", "\|")
  $validator = ([string](Get-PropertyOrDefault -Object $item -Name "validatorCommand" -DefaultValue "")).Replace("|", "\|")
  "| ``$id`` | $blocker | $currentState | $nextAction | ``$validator`` |"
}
$sourceLines = $record.sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $commonNonSubstitutes | ForEach-Object { "- ``$_``" }
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }
$postPublishRequiredEvidenceLines = $postPublishRequiredEvidence | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release Close Gap Dashboard

生成时间：$($record.generatedAtUtc)

## 总结

该 dashboard 汇总 release close 前仍然阻塞的真实 proof gap。它是 owner guidance，不执行发布、不上传包、不伪造 proof。``recordKind=release-close-gap-dashboard``，``dashboardState=$dashboardState``，``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前状态

| 项目 | 当前值 |
|---|---|
| preflightState | ``$preflightState`` |
| preflightFailedItemCount | ``$preflightFailedItemCount`` |
| releaseEvidenceBundleState | ``$releaseEvidenceState`` |
| isReleaseEvidenceComplete | ``$releaseEvidenceComplete`` |
| releasePromotionIssueState | ``$promotionState`` |
| realModelAndPackageProofInputPackageState | ``$inputPackageState`` |
| deferredSafetyTriageState | ``$deferredSafetyTriageState`` |
| deferredSafetyTriageTotalRows | ``$deferredSafetyTriageTotalRows`` |
| deferredSafetyTierAImmediateSafeCount | ``$deferredSafetyTierAImmediateSafeCount`` |
| deferredSafetyTierBSafeAlternativeCount | ``$deferredSafetyTierBSafeAlternativeCount`` |
| deferredSafetyTierCDesignGateCount | ``$deferredSafetyTierCDesignGateCount`` |
| deferredSafetyTierDKeepDeferredCount | ``$deferredSafetyTierDKeepDeferredCount`` |
| postPublishRequiredEvidenceCount | ``$postPublishRequiredEvidenceCount`` |
| releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount | ``$releaseRuntimeProofExecutionMatrixPostPublishRequiredEvidenceCount`` |
| gapCount | ``$($gapItems.Count)`` |
| ownerReleaseExecutionPackageState | ``$ownerReleaseExecutionPackageState`` |
| oneScreenReleaseHoldChecklistCount | ``$oneScreenReleaseHoldChecklistCount`` |

## One-Screen Release Hold Checklist

This table is inherited from ``owner-release-execution-package`` and is owner guidance only. It does not execute publish, promote runtime proof, or close the release issue.

| ID | Owner visible blocker | Current state | Owner next action | Validator |
|---|---|---|---|---|
$($oneScreenReleaseHoldRows -join "`r`n")

## Gap Items

| Gap | Proof class | Passed | Current state | Validator | Boundary |
|---|---|---|---|---|---|
$($gapRows -join "`r`n")

## Owner Action Notes

- ``package-consumer-runtime`` 只能来自 compatible host 上的 clean package consumer runtime smoke。
- Bridge-only wrapper surface evidence remains ``WrapperSurfaceEvidenceKind=compile-surface-proof`` and ``IsRuntimeExecutionProof=False``; ``Skipped=True``、``dependency-probe-only``、ONNX Parser diagnostic snapshot、ONNX ParserRefitter diagnostic snapshot 和 copied managed diagnostic snapshot 都不能替代 clean package consumer runtime proof。
- ``real-model-runtime`` 只能来自真实 Classification/YoloVision 模型、labels、输入资产、license、SHA256 和 sample runner log。
- YoloVision proof 范围固定覆盖 ``YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom``，以及 ``det/cls/seg/obb/pose/sem``（``det、cls、seg、obb、pose、sem``）；这些样例 proof 只能晋级 ``real-model-runtime``，不能替代 ``package-consumer-runtime``。
- ``post-publish verification`` 只能来自真实渠道 package、下载 hash、clean consumer restore/build/probe/smoke 日志和 ``Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof``。
- Linux runner proof 需要真实 Linux x64 runner 产物，Windows handoff 不能替代。
- ``blocked-by-cuda-driver`` 是环境阻塞，不是 smoke passed。
- ``deferred-candidate-safety-triage`` 是 boundary disclosure / planning input：A/B/C/D 分层不会让 ``canPublishPublicly``、``canCloseReleaseIssue`` 或 runtime proof 晋级，也不是删除 deferred 记录的许可。

## Post-Publish Required Evidence

$($postPublishRequiredEvidenceLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Safety Notes

$($safetyLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release close gap dashboard written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "DashboardState=$dashboardState"
Write-Output "GapCount=$($gapItems.Count)"
Write-Output "PerformsPublish=False"
Write-Output "CanCloseReleaseIssue=False"
