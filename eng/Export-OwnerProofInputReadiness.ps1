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

function New-OwnerInputContract {
  param(
    [string]$BlockerId,
    [string]$ProofClass,
    [string]$Title,
    [string]$CurrentState,
    [string[]]$RequiredInputFiles,
    [string[]]$TemplateFiles,
    [string[]]$ReplaceOrFillInstructions,
    [string]$ValidatorCommand,
    [string[]]$SuccessCriteria,
    [string[]]$NonSubstitutes,
    [string[]]$ExpectedOutputArtifacts,
    [string[]]$SourceGuidance
  )

  [pscustomobject]@{
    blockerId = $BlockerId
    proofClass = $ProofClass
    title = $Title
    currentState = $CurrentState
    ready = $false
    contractState = "blocked-real-owner-input-required"
    requiredInputFiles = $RequiredInputFiles
    templateFiles = $TemplateFiles
    replaceOrFillInstructions = $ReplaceOrFillInstructions
    validatorCommand = $ValidatorCommand
    successCriteria = $SuccessCriteria
    nonSubstitutes = $NonSubstitutes
    expectedOutputArtifacts = $ExpectedOutputArtifacts
    sourceGuidance = $SourceGuidance
    canPromoteOnPass = $false
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    ownerAction = "owner-action-required"
  }
}

$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$compatibleHostProofExecutionPack = Read-JsonOrNull "artifacts\final-release\compatible-host-proof-execution-pack.json"
$releaseProofReadinessSnapshot = Read-JsonOrNull "artifacts\final-release\release-proof-readiness-snapshot.json"
$realModelAndPackageProofInputPackage = Read-JsonOrNull "artifacts\final-release\real-model-and-package-proof-input-package.json"
$externalRuntimeProofCollectionPackage = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-collection-package.json"
$postPublishVerificationCollectionPackage = Read-JsonOrNull "artifacts\final-release\post-publish-verification-collection-package.json"
$ownerApprovalValidation = Read-JsonOrNull "artifacts\final-release\release-owner-approval-input-validation.json"
$authorizedPlanValidation = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan-validation.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$linuxRunnerValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$postPublishVerificationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$ownerPackageState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$compatibleHostPackState = [string](Get-PropertyOrDefault -Object $compatibleHostProofExecutionPack -Name "packageState" -DefaultValue "missing-compatible-host-proof-execution-pack")
$releaseProofReadinessState = [string](Get-PropertyOrDefault -Object $releaseProofReadinessSnapshot -Name "readinessState" -DefaultValue "missing-release-proof-readiness-snapshot")
$realModelInputPackageState = [string](Get-PropertyOrDefault -Object $realModelAndPackageProofInputPackage -Name "packageState" -DefaultValue "missing-real-model-and-package-proof-input-package")
$externalRuntimeCollectionState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofCollectionPackage -Name "packageState" -DefaultValue "missing-external-runtime-proof-collection-package")
$postPublishCollectionState = [string](Get-PropertyOrDefault -Object $postPublishVerificationCollectionPackage -Name "packageState" -DefaultValue "missing-post-publish-verification-collection-package")

$ownerApprovalState = [string](Get-PropertyOrDefault -Object $ownerApprovalValidation -Name "validationState" -DefaultValue "missing-release-owner-approval-input-validation")
$ownerApprovalCanPublish = [bool](Get-PropertyOrDefault -Object $ownerApprovalValidation -Name "canPublishPublicly" -DefaultValue $false)
$authorizedPlanState = [string](Get-PropertyOrDefault -Object $authorizedPlanValidation -Name "validationState" -DefaultValue "missing-owner-authorized-publish-command-plan-validation")
$authorizedPlanCanMaterialize = [bool](Get-PropertyOrDefault -Object $authorizedPlanValidation -Name "canMaterializeExecutableCommands" -DefaultValue $false)

$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$externalRuntimeCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)

$linuxRunnerState = [string](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxRunnerProof = [bool](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)

$sampleRunState = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-record-validation")
$sampleRunClassification = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$sampleCanPromoteRealModelRuntime = [bool](Get-PropertyOrDefault -Object $sampleRunValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)

$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishClassification = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")
$postPublishIsProof = [bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "canCloseReleaseIssue" -DefaultValue $false)

$commonNonSubstitutes = @(
  "template",
  "draft",
  "checklist",
  "dashboard",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "DependencyProbe",
  "dependency-probe-only",
  "sidecar-only",
  "build-only",
  "parse-only",
  "Skipped=True",
  "blocked-by-cuda-driver"
)

$contracts = @(
  New-OwnerInputContract `
    -BlockerId "owner-authorization" `
    -ProofClass "owner-authorization" `
    -Title "Owner authorization and manual publish command approval" `
    -CurrentState "ownerApprovalValidationState=$ownerApprovalState; ownerApprovalCanPublish=$ownerApprovalCanPublish; authorizedPlanValidationState=$authorizedPlanState; authorizedPlanCanMaterialize=$authorizedPlanCanMaterialize" `
    -RequiredInputFiles @(
      "artifacts/final-release/release-owner-approval-input.json",
      "artifacts/final-release/owner-authorized-publish-command-plan.json"
    ) `
    -TemplateFiles @(
      "artifacts/final-release/release-owner-approval-input-template.json",
      "artifacts/final-release/release-owner-approval-input-record.example.json",
      "artifacts/final-release/owner-authorized-publish-command-plan.json"
    ) `
    -ReplaceOrFillInstructions @(
      "Fill a non-template owner approval input with ownerName, ownerDecisionId, approvalTimestampUtc, targetChannel, package identities, rollback plan, and credential handling.",
      "Validate the owner approval input without changing publish gates directly.",
      "Review owner-authorized-publish-command-plan.json and keep publish commands manual until all proof validators pass."
    ) `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -SuccessCriteria @(
      "release-owner-approval-input-validation.json reports canPublishPublicly only from a real non-template owner input.",
      "owner-authorized-publish-command-plan-validation.json reports canMaterializeExecutableCommands from approved manual commands.",
      "No generated checklist or example record is treated as authorization."
    ) `
    -NonSubstitutes ($commonNonSubstitutes + @("example owner approval", "pending owner decision")) `
    -ExpectedOutputArtifacts @(
      "artifacts/final-release/release-owner-approval-input-validation.json",
      "artifacts/final-release/owner-authorized-publish-command-plan-validation.json"
    ) `
    -SourceGuidance @(
      "artifacts/final-release/owner-release-execution-package.json",
      "artifacts/final-release/release-publish-execution-checklist.json"
    )

  New-OwnerInputContract `
    -BlockerId "package-consumer-runtime" `
    -ProofClass "package-consumer-runtime" `
    -Title "Compatible-host clean package consumer runtime proof" `
    -CurrentState "validationState=$externalRuntimeProofState; proofClassification=$externalRuntimeProofClassification; canPromoteRuntimeProof=$externalRuntimeCanPromote; collectionState=$externalRuntimeCollectionState" `
    -RequiredInputFiles @(
      "artifacts/final-release/external-runtime-proof-record.json",
      "clean external consumer project outside this repository",
      "real runtime smoke log with SHA256"
    ) `
    -TemplateFiles @(
      "artifacts/final-release/external-runtime-proof-record.input-template.json",
      "artifacts/final-release/external-runtime-proof-record.example.json",
      "artifacts/final-release/external-runtime-proof-collection-package.json",
      "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json"
    ) `
    -ReplaceOrFillInstructions @(
      "Run restore/build/probe/runtime smoke from a clean consumer that references the exact release candidate packages.",
      "Capture host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata and package SHA256 values.",
      "Fill external-runtime-proof-record.json with real log path, stdout/stderr summary, and matching log SHA256."
    ) `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof" `
    -SuccessCriteria @(
      "proofClassification is package-consumer-runtime.",
      "canPromoteRuntimeProof is true only after real compatible-host runtime execution.",
      "ProjectReference, local feed, dependency-probe-only, and blocked-by-cuda-driver evidence are rejected."
    ) `
    -NonSubstitutes ($commonNonSubstitutes + @("bridge-only log", "WrapperSurfaceEvidenceKind=compile-surface-proof", "IsRuntimeExecutionProof=False")) `
    -ExpectedOutputArtifacts @(
      "artifacts/final-release/external-runtime-proof-record.json",
      "artifacts/final-release/external-runtime-proof-validation.json",
      "artifacts/final-release/external-runtime-proof-validation.md"
    ) `
    -SourceGuidance @(
      "artifacts/final-release/compatible-host-proof-execution-pack.json",
      "artifacts/final-release/external-runtime-proof-collection-package.json"
    )

  New-OwnerInputContract `
    -BlockerId "linux-runner-proof" `
    -ProofClass "linux-runner-proof" `
    -Title "Real Linux x64 runner proof" `
    -CurrentState "validationState=$linuxRunnerState; isRealLinuxRunnerProof=$linuxRunnerProof; linuxRuntimePackageKey=$LinuxRuntimePackageKey" `
    -RequiredInputFiles @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json",
      "real Linux x64 runner command log with SHA256"
    ) `
    -TemplateFiles @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.template.json",
      "docs/articles/zh-cn/linux-runner-evidence-record-schema.md",
      "docs/articles/zh-cn/linux-runner-evidence-checklist.md"
    ) `
    -ReplaceOrFillInstructions @(
      "Execute the Linux runner flow on a real Linux x64 CUDA/TensorRT host.",
      "Record runtime package key, CUDA/TensorRT/cuDNN versions, native copy/build evidence, command log, and SHA256.",
      "Copy the filled Linux runner evidence record back under artifacts/linux-dry-run/$LinuxRuntimePackageKey."
    ) `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey" `
    -SuccessCriteria @(
      "linux-runner-evidence-validation.json reports isRealLinuxRunnerProof=true.",
      "The evidence comes from a Linux x64 runner, not Windows dry-run or handoff guidance.",
      "Runtime package key and host metadata match the target Linux package."
    ) `
    -NonSubstitutes ($commonNonSubstitutes + @("Windows handoff", "dry run summary", "Linux template-only record")) `
    -ExpectedOutputArtifacts @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json",
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
    ) `
    -SourceGuidance @(
      "artifacts/final-release/compatible-host-proof-execution-pack.json",
      "docs/articles/zh-cn/linux-runner-evidence-checklist.md"
    )

  New-OwnerInputContract `
    -BlockerId "real-model-runtime" `
    -ProofClass "real-model-runtime" `
    -Title "Classification and YoloVision real model runtime proof" `
    -CurrentState "validationState=$sampleRunState; proofClassification=$sampleRunClassification; canPromoteRealModelRuntime=$sampleCanPromoteRealModelRuntime; inputPackageState=$realModelInputPackageState" `
    -RequiredInputFiles @(
      "samples/assets/classification-assets.json",
      "samples/assets/yolovision-assets.json",
      "artifacts/user-acceptance/sample-run-evidence-record.json",
      "real model/sample runner log with SHA256"
    ) `
    -TemplateFiles @(
      "artifacts/final-release/real-model-and-package-proof-input-package.json",
      "artifacts/user-acceptance/sample-run-evidence-record.input-template.json"
    ) `
    -ReplaceOrFillInstructions @(
      "Provide real Classification/YoloVision model, labels, input tensor, license note, and SHA256 values.",
      "Run TensorRtExec or sample runner on real assets and capture runtime log plus sidecar.",
      "Fill sample-run-evidence-record.json with proofClassification=real-model-runtime."
    ) `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog" `
    -SuccessCriteria @(
      "sample asset manifests contain real model/input/license SHA256 values.",
      "sample-run-evidence-record-validation.json reports canPromoteRealModelRuntime=true.",
      "Build-only, parse-only, sidecar-only, and model candidate lists are rejected."
    ) `
    -NonSubstitutes ($commonNonSubstitutes + @("model candidate list", "support matrix", "missing model hash", "missing license hash")) `
    -ExpectedOutputArtifacts @(
      "samples/assets/classification-assets.json",
      "samples/assets/yolovision-assets.json",
      "artifacts/user-acceptance/sample-run-evidence-record.json",
      "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
    ) `
    -SourceGuidance @(
      "artifacts/final-release/real-model-and-package-proof-input-package.json",
      "artifacts/final-release/compatible-host-proof-execution-pack.json"
    )

  New-OwnerInputContract `
    -BlockerId "post-publish-verification" `
    -ProofClass "post-publish-verification" `
    -Title "Post-publish clean consumer verification from public channel" `
    -CurrentState "validationState=$postPublishState; proofClassification=$postPublishClassification; isPostPublishVerificationProof=$postPublishIsProof; canCloseReleaseIssue=$postPublishCanClose; collectionState=$postPublishCollectionState" `
    -RequiredInputFiles @(
      "artifacts/final-release/post-publish-verification-record.json",
      "real public package URL",
      "clean external consumer restore/build/probe/smoke log with SHA256"
    ) `
    -TemplateFiles @(
      "artifacts/final-release/post-publish-verification-record-template.json",
      "artifacts/final-release/post-publish-verification-record.input-draft.json",
      "artifacts/final-release/post-publish-verification-collection-package.json"
    ) `
    -ReplaceOrFillInstructions @(
      "After owner-approved public publish, restore the public package from the real channel in a clean external consumer.",
      "Capture public package URL, downloaded nupkg SHA256, host metadata, restore/build/probe/smoke logs, stdout/stderr summary, and log SHA256.",
      "Fill post-publish-verification-record.json and validate it with FailOnNotProof."
    ) `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof" `
    -SuccessCriteria @(
      "isPostPublishVerificationProof=true only after public channel package verification.",
      "canCloseReleaseIssue remains false until a real post-publish proof record passes.",
      "Local feed, ProjectReference, input draft, and collection package evidence are rejected."
    ) `
    -NonSubstitutes ($commonNonSubstitutes + @("post-publish template", "post-publish backfill plan", "post-publish input draft", "pre-publish local package")) `
    -ExpectedOutputArtifacts @(
      "artifacts/final-release/post-publish-verification-record.json",
      "artifacts/final-release/post-publish-verification-validation.json",
      "artifacts/final-release/post-publish-verification-validation.md"
    ) `
    -SourceGuidance @(
      "artifacts/final-release/post-publish-verification-collection-package.json",
      "artifacts/final-release/release-publish-execution-checklist.json"
    )
)

$sourceEvidence = @(
  "artifacts/final-release/owner-release-execution-package.json",
  "artifacts/final-release/compatible-host-proof-execution-pack.json",
  "artifacts/final-release/release-proof-readiness-snapshot.json",
  "artifacts/final-release/real-model-and-package-proof-input-package.json",
  "artifacts/final-release/external-runtime-proof-collection-package.json",
  "artifacts/final-release/post-publish-verification-collection-package.json",
  "artifacts/final-release/release-owner-approval-input-validation.json",
  "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
  "artifacts/final-release/external-runtime-proof-validation.json",
  "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
  "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
  "artifacts/final-release/post-publish-verification-validation.json"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "owner-proof-input-readiness"
  readinessState = "blocked-real-proof-input-required"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  contractCount = $contracts.Count
  readyContractCount = 0
  blockedContractCount = $contracts.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  requiresHumanOwner = $true
  requiresCompatibleHost = $true
  ownerReleaseExecutionPackageState = $ownerPackageState
  compatibleHostProofExecutionPackState = $compatibleHostPackState
  releaseProofReadinessSnapshotState = $releaseProofReadinessState
  realModelAndPackageProofInputPackageState = $realModelInputPackageState
  externalRuntimeProofCollectionPackageState = $externalRuntimeCollectionState
  postPublishVerificationCollectionPackageState = $postPublishCollectionState
  ownerInputContracts = $contracts
  sourceEvidence = $sourceEvidence
  nonSubstituteProofKinds = $commonNonSubstitutes
  boundary = "This artifact makes real owner proof inputs executable and auditable, but it is not proof, not publication approval, not package push, and not release-close approval."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-proof-input-readiness.json"
$markdownPath = Join-Path $artifactRoot "owner-proof-input-readiness.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Proof Input Readiness")
$lines.Add("")
$lines.Add("Generated: ``$($record.generatedAtUtc)``")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- readiness state: ``$($record.readinessState)``")
$lines.Add("- contract count: ``$($record.contractCount)``")
$lines.Add("- ready contract count: ``$($record.readyContractCount)``")
$lines.Add("- blocked contract count: ``$($record.blockedContractCount)``")
$lines.Add("- performs publish: ``$($record.performsPublish)``")
$lines.Add("- can publish publicly: ``$($record.canPublishPublicly)``")
$lines.Add("- can close release issue: ``$($record.canCloseReleaseIssue)``")
$lines.Add("- requires human owner: ``$($record.requiresHumanOwner)``")
$lines.Add("- requires compatible host: ``$($record.requiresCompatibleHost)``")
$lines.Add("")
$lines.Add("## Owner Input Contracts")
$lines.Add("")
$lines.Add("| blockerId | proofClass | contractState | validator | currentState |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($contract in $contracts) {
  $lines.Add("| ``$($contract.blockerId)`` | ``$($contract.proofClass)`` | ``$($contract.contractState)`` | ``$($contract.validatorCommand.Replace("|", "\|"))`` | $($contract.currentState.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Required Files")
$lines.Add("")
foreach ($contract in $contracts) {
  $lines.Add("### ``$($contract.blockerId)``")
  foreach ($item in $contract.requiredInputFiles) {
    $lines.Add("- required: ``$item``")
  }
  foreach ($item in $contract.templateFiles) {
    $lines.Add("- template/guidance: ``$item``")
  }
}
$lines.Add("")
$lines.Add("## Non-Substitutes")
$lines.Add("")
foreach ($item in $commonNonSubstitutes) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("## Source Evidence")
$lines.Add("")
foreach ($item in $sourceEvidence) {
  $lines.Add("- ``$item``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.boundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner proof input readiness written to $jsonPath"
Write-Host "Owner proof input readiness written to $markdownPath"
