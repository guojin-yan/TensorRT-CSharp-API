[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

function New-BackfillItem {
  param(
    [string]$Id,
    [string]$ProofClass,
    [string]$Title,
    [string]$CurrentState,
    [string]$FirstCommand,
    [string]$ValidatorCommand,
    [string[]]$RequiredRealInputs,
    [string[]]$ExpectedArtifacts,
    [string[]]$SourceArtifacts,
    [string]$BlockerReason
  )

  [pscustomobject]@{
    id = $Id
    proofClass = $ProofClass
    title = $Title
    currentState = $CurrentState
    firstCommand = $FirstCommand
    validatorCommand = $ValidatorCommand
    requiredRealInputs = $RequiredRealInputs
    requiredRealInputCount = @($RequiredRealInputs).Count
    expectedArtifacts = $ExpectedArtifacts
    sourceArtifacts = $SourceArtifacts
    cannotUse = $script:NonSubstituteProofKinds
    nonSubstituteProofKinds = $script:NonSubstituteProofKinds
    blockerReason = $BlockerReason
    ownerAction = "owner-action-required"
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$ownerProofInputValidation = Read-JsonOrNull "artifacts\final-release\release-owner-proof-input-record-validation.json"
$ownerAuthorizedPlanValidation = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan-validation.json"
$externalRuntimeValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$linuxRunnerValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$sampleRunValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$sampleAssetAudit = Read-JsonOrNull "artifacts\user-acceptance\sample-asset-manifest-audit.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$releaseIssueCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"
$ownerReleaseExecutionPackage = Read-JsonOrNull "artifacts\final-release\owner-release-execution-package.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"

$ownerProofInputState = [string](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "validationState" -DefaultValue "missing-release-owner-proof-input-record-validation")
$ownerProofInputCanPromote = [bool](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "canPromoteOwnerProofInput" -DefaultValue $false)
$ownerPlanState = [string](Get-PropertyOrDefault -Object $ownerAuthorizedPlanValidation -Name "validationState" -DefaultValue "missing-owner-authorized-publish-command-plan-validation")
$ownerPlanCanMaterialize = [bool](Get-PropertyOrDefault -Object $ownerAuthorizedPlanValidation -Name "canMaterializeExecutableCommands" -DefaultValue $false)
$externalRuntimeState = [string](Get-PropertyOrDefault -Object $externalRuntimeValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalRuntimeCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$linuxRunnerState = [string](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxRunnerProof = [bool](Get-PropertyOrDefault -Object $linuxRunnerValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$sampleRunState = [string](Get-PropertyOrDefault -Object $sampleRunValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleRunProof = [bool](Get-PropertyOrDefault -Object $sampleRunValidation -Name "isRealSampleRunProof" -DefaultValue $false)
$sampleAssetState = [string](Get-PropertyOrDefault -Object $sampleAssetAudit -Name "auditState" -DefaultValue "missing-sample-asset-manifest-audit")
$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$preflightState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$preflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)
$preflightCanClose = [bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)
$bundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$bundleCanPublish = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canPublishPublicly" -DefaultValue $false)
$bundleCanClose = [bool](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "canCloseReleaseIssue" -DefaultValue $false)
$closeRecordState = [string](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")
$closeRecordCanPromote = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "canPromoteReleaseIssueCloseRecord" -DefaultValue $false)
$closeRecordCanClose = [bool](Get-PropertyOrDefault -Object $releaseIssueCloseValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$ownerPackageState = [string](Get-PropertyOrDefault -Object $ownerReleaseExecutionPackage -Name "packageState" -DefaultValue "missing-owner-release-execution-package")
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)

$script:NonSubstituteProofKinds = @(
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "direct .nupkg reference",
  "local package source",
  "helper scan",
  "build-only",
  "parse-only",
  "sidecar-only",
  "dependency-probe-only",
  "Skipped=True",
  "blocked-by-cuda-driver",
  "bridge-only package consumer log",
  "bridge-only wrapper surface",
  "WrapperSurfaceEvidenceKind=compile-surface-proof",
  "IsRuntimeExecutionProof=False",
  "mismatched log SHA256",
  "managed-readiness",
  "readiness snapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only",
  "template-only release issue close record",
  "preflight-only release issue close record",
  "release-issue-close-record-template.json",
  "Windows handoff for Linux proof"
)

$backfillItems = @(
  New-BackfillItem `
    -Id "owner-authorization" `
    -ProofClass "owner-authorization" `
    -Title "Owner authorization and publish command approval" `
    -CurrentState "ownerProofInputValidationState=$ownerProofInputState; ownerProofInputCanPromote=$ownerProofInputCanPromote; ownerAuthorizedPlanValidationState=$ownerPlanState; ownerPlanCanMaterializeExecutableCommands=$ownerPlanCanMaterialize" `
    -FirstCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerProofInputRecordTemplate.ps1" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1 -InputPath artifacts/final-release/release-owner-proof-input-record.json -RequireExistingLogs -FailOnNotProof" `
    -RequiredRealInputs @(
      "ownerAuthorization.ownerName",
      "ownerAuthorization.ownerDecisionId",
      "selectedChannel.channelSourceUri",
      "packages.managed.nupkgSha256",
      "packages.runtime.nupkgSha256",
      "runtimeEvidence.runtimeSmokeLogSha256",
      "hostMetadata.tensorRtRuntimeVersion",
      "acknowledgements.hashesComputedFromReferencedFiles",
      "no local feed, ProjectReference, direct .nupkg, or placeholder command"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/release-owner-proof-input-record.json",
      "artifacts/final-release/release-owner-proof-input-record-validation.json",
      "artifacts/final-release/owner-authorized-publish-command-plan-validation.json"
    ) `
    -SourceArtifacts @(
      "artifacts/final-release/release-owner-proof-input-record-template.json",
      "artifacts/final-release/release-owner-proof-input-record-validation.json",
      "artifacts/final-release/owner-authorized-publish-command-plan-validation.json"
    ) `
    -BlockerReason "Owner authorization and command approval are still template/placeholder guidance until validator-passing real owner fields exist."

  New-BackfillItem `
    -Id "package-consumer-runtime" `
    -ProofClass "package-consumer-runtime" `
    -Title "Compatible host package consumer runtime proof" `
    -CurrentState "externalRuntimeValidationState=$externalRuntimeState; externalRuntimeCanPromoteRuntimeProof=$externalRuntimeCanPromote; runtimePackageKey=$RuntimePackageKey" `
    -FirstCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-CompatibleHostRuntimeProofCollectionBundle.ps1 -RuntimePackageKey $RuntimePackageKey" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof" `
    -RequiredRealInputs @(
      "clean consumer root outside repository",
      "managed/runtime package identities from the selected channel",
      "managed/runtime nupkg SHA256",
      "runtime package key",
      "compatible host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
      "restore/build/runtime smoke commands",
      "stdoutSummary and stderrSummary",
      "existing runtime smoke log with matching SHA256",
      "no local feed, ProjectReference, bridge-only log, Skipped=True, dependency-probe-only, or copied diagnostic snapshot"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/external-runtime-proof-record.json",
      "artifacts/final-release/external-runtime-proof-validation.json"
    ) `
    -SourceArtifacts @(
      "artifacts/final-release/external-runtime-proof-record-template.json",
      "artifacts/final-release/external-runtime-proof-validation.json",
      "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json"
    ) `
    -BlockerReason "Package-consumer-runtime remains blocked until a real compatible-host smoke log and hash-matching validation pass."

  New-BackfillItem `
    -Id "linux-runner-proof" `
    -ProofClass "linux-runner-proof" `
    -Title "Linux x64 runtime runner proof" `
    -CurrentState "linuxRunnerValidationState=$linuxRunnerState; isRealLinuxRunnerProof=$linuxRunnerProof; linuxRuntimePackageKey=$LinuxRuntimePackageKey" `
    -FirstCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-LinuxRunnerEvidenceRecordTemplate.ps1 -RuntimePackageKey $LinuxRuntimePackageKey" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey -RequireExistingLog -FailOnNotProof" `
    -RequiredRealInputs @(
      "real Linux x64 runner identity",
      "Linux runtime package key",
      "Linux restore/build/smoke commands",
      "Linux host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
      "existing Linux runner smoke log with matching SHA256",
      "no Windows handoff, runbook, or container planning record as proof"
    ) `
    -ExpectedArtifacts @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json",
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
    ) `
    -SourceArtifacts @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
      "docs/articles/zh-cn/linux-runner-evidence-record-schema.md"
    ) `
    -BlockerReason "Linux proof requires a real Linux runner record; Windows handoff and runbooks remain non-proof."

  New-BackfillItem `
    -Id "real-model-runtime" `
    -ProofClass "real-model-runtime" `
    -Title "Classification and YoloVision real model runtime proof" `
    -CurrentState "sampleRunValidationState=$sampleRunState; isRealSampleRunProof=$sampleRunProof; sampleAssetAuditState=$sampleAssetState; yoloVisionScope=YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom; taskScope=det/cls/seg/obb/pose/sem" `
    -FirstCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-SampleAssetManifestTemplate.ps1" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredRealInputs @(
      "model source URL and license",
      "model SHA256",
      "input image or tensor SHA256",
      "labels or class metadata SHA256",
      "TensorRtExec build sidecar",
      "sample runner command and log SHA256",
      "Classification or YoloVision task metadata",
      "YoloVision family coverage: YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom",
      "YoloVision task coverage: det/cls/seg/obb/pose/sem"
    ) `
    -ExpectedArtifacts @(
      "artifacts/user-acceptance/sample-asset-manifest.json",
      "artifacts/user-acceptance/sample-run-evidence-record.json",
      "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
    ) `
    -SourceArtifacts @(
      "artifacts/user-acceptance/sample-asset-manifest-audit.json",
      "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
      "samples/YoloVision/README.md",
      "samples/Classification/README.md"
    ) `
    -BlockerReason "Real-model-runtime requires real assets and logs; build-only sidecars and synthetic examples do not promote release proof."

  New-BackfillItem `
    -Id "post-publish-verification" `
    -ProofClass "post-publish-verification" `
    -Title "Post-publish clean consumer verification" `
    -CurrentState "postPublishValidationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof; postPublishCanCloseReleaseIssue=$postPublishCanClose" `
    -FirstCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordTemplate.ps1" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof" `
    -RequiredRealInputs @(
      "selected channel and channel source URI",
      "published managed/runtime package URLs",
      "downloaded managed/runtime nupkg SHA256",
      "clean consumer root outside repository",
      "no ProjectReference",
      "restore/build/dependency probe/runtime smoke commands",
      "runtime smoke command containing --runtime-package-key",
      "restore/native asset/dependency probe/smoke log SHA256",
      "stdoutSummary and stderrSummary",
      "host metadata and reviewer identity"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/post-publish-verification-record.json",
      "artifacts/final-release/post-publish-verification-validation.json"
    ) `
    -SourceArtifacts @(
      "artifacts/final-release/post-publish-verification-record-template.json",
      "artifacts/final-release/post-publish-verification-validation.json",
      "artifacts/final-release/post-publish-clean-consumer-project-scan.json"
    ) `
    -BlockerReason "Post-publish verification can only happen after real publication from the selected channel; helper scans and drafts are not proof."

  New-BackfillItem `
    -Id "release-issue-close-record" `
    -ProofClass "release-issue-close-record" `
    -Title "Final release issue close record" `
    -CurrentState "releaseIssueCloseRecordValidationState=$closeRecordState; closeRecordCanPromote=$closeRecordCanPromote; closeRecordCanCloseReleaseIssue=$closeRecordCanClose; releaseClosePreflightState=$preflightState; releaseClosePreflightFailedItemCount=$preflightFailedItemCount; preflightCanCloseReleaseIssue=$preflightCanClose; releaseEvidenceBundleState=$bundleState; bundleCanCloseReleaseIssue=$bundleCanClose; staleFindingCount=$staleFindingCount" `
    -FirstCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordTemplate.ps1" `
    -ValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady" `
    -RequiredRealInputs @(
      "releaseIssue.url",
      "ownerDecision.finalCloseDecision",
      "postPublishVerification.isPostPublishVerificationProof",
      "postPublishVerification.canCloseReleaseIssue",
      "releaseClosePreflight.canCloseReleaseIssue",
      "staleReleaseClaims.findingCount=0",
      "releaseEvidenceBundle.sha256",
      "rollbackPlan.summary",
      "rollbackPlan.packageYankOrDeprecatePlan",
      "owner/reviewer identity and timestamp"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/release-issue-close-record.json",
      "artifacts/final-release/release-issue-close-record-validation.json"
    ) `
    -SourceArtifacts @(
      "artifacts/final-release/release-issue-close-record-template.json",
      "artifacts/final-release/release-issue-close-record-validation.json",
      "artifacts/final-release/release-close-preflight.json",
      "artifacts/final-release/release-evidence-bundle.json",
      "artifacts/final-release/stale-release-claims-audit.json"
    ) `
    -BlockerReason "The final close record stays blocked until real proof, preflight, bundle hash, rollback plan, and owner final close decision all validate."
)

$readyItemCount = @($backfillItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $false) }).Count
$packageState = if ($readyItemCount -eq $backfillItems.Count) { "ready-for-owner-close-review" } else { "blocked-real-proof-required" }

$sourceArtifacts = @(
  "artifacts/final-release/release-owner-proof-input-record-validation.json",
  "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
  "artifacts/final-release/external-runtime-proof-validation.json",
  "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json",
  "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
  "artifacts/user-acceptance/sample-asset-manifest-audit.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/final-release/release-close-preflight.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/release-issue-close-record-validation.json",
  "artifacts/final-release/owner-release-execution-package.json"
)

$record = [pscustomobject]@{
  recordKind = "owner-proof-backfill-execution-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packageState = $packageState
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  ownerReleaseExecutionPackageState = $ownerPackageState
  releaseClosePreflightState = $preflightState
  releaseClosePreflightFailedItemCount = $preflightFailedItemCount
  releaseIssueCloseRecordValidationState = $closeRecordState
  releaseIssueCloseRecordCanPromote = $closeRecordCanPromote
  releaseIssueCloseRecordCanCloseReleaseIssue = $closeRecordCanClose
  releaseEvidenceBundleState = $bundleState
  releaseEvidenceBundleCanPublishPublicly = $bundleCanPublish
  releaseEvidenceBundleCanCloseReleaseIssue = $bundleCanClose
  staleReleaseClaimsFindingCount = $staleFindingCount
  backfillItemCount = @($backfillItems).Count
  readyBackfillItemCount = $readyItemCount
  backfillItems = $backfillItems
  nonSubstituteProofKinds = $script:NonSubstituteProofKinds
  sourceArtifacts = $sourceArtifacts
  ownerBackfillOrder = @(
    "fill release-owner-proof-input-record.json",
    "validate owner authorization and manual publish command plan",
    "collect package-consumer-runtime proof on compatible host",
    "collect Linux runner proof on real Linux x64 runner",
    "collect Classification/YoloVision real-model-runtime proof",
    "owner manually executes publish commands outside automation only after authorization",
    "collect post-publish clean consumer proof from selected channel",
    "refresh release-close-preflight and release-evidence-bundle",
    "fill and validate release-issue-close-record.json"
  )
  safetyNotes = @(
    "This pack is owner guidance only and does not execute publish commands.",
    "performsPublish=false, canPublishPublicly=false, and canCloseReleaseIssue=false are fixed until real owner proof exists.",
    "Post-publish verification alone must not close the release issue.",
    "Release close preflight, evidence bundle, readiness snapshots, and release issue close record templates are non-substitute proof.",
    "Only validator-passing real proof records with existing logs and matching SHA256 values can move the release toward owner close review."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-proof-backfill-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "owner-proof-backfill-execution-pack.md"

Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)
$itemRows = $backfillItems | ForEach-Object {
  $state = ([string]$_.currentState).Replace("|", "\|")
  $validator = ([string]$_.validatorCommand).Replace("|", "\|")
  $blocker = ([string]$_.blockerReason).Replace("|", "\|")
  "| ``$($_.id)`` | ``$($_.proofClass)`` | $state | ``$validator`` | $blocker |"
}
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $script:NonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$orderLines = $record.ownerBackfillOrder | ForEach-Object { "- $_" }
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }

$markdown = @"
# Owner Proof Backfill Execution Pack

生成时间：$($record.generatedAtUtc)

## 总结

该执行包把 owner 真实 proof 回填路径集中到一个文件：owner authorization、``package-consumer-runtime``、Linux runner proof、``real-model-runtime``、post-publish verification 和 release issue close record。它只做 owner guidance，不执行 ``dotnet nuget push``、不上传 GitHub Packages、不关闭 release issue。``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前状态

| 项目 | 当前值 |
|---|---|
| packageState | ``$packageState`` |
| ownerReleaseExecutionPackageState | ``$ownerPackageState`` |
| releaseClosePreflightState | ``$preflightState`` |
| releaseClosePreflightFailedItemCount | ``$preflightFailedItemCount`` |
| releaseIssueCloseRecordValidationState | ``$closeRecordState`` |
| releaseIssueCloseRecordCanPromote | ``$closeRecordCanPromote`` |
| releaseEvidenceBundleState | ``$bundleState`` |
| backfillItemCount | ``$($backfillItems.Count)`` |
| readyBackfillItemCount | ``$readyItemCount`` |

## Backfill Items

| ID | Proof class | Current state | Validator | Blocker reason |
|---|---|---|---|---|
$($itemRows -join "`r`n")

## Owner Backfill Order

$($orderLines -join "`r`n")

## Release Issue Close Boundary

``release-issue-close-record-template.json``、schema-only、preflight-only、readiness snapshot、release close preflight 和 release evidence bundle 都不能关闭 release issue。最终关闭前必须回填真实 ``release-issue-close-record.json``，并通过 ``Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady``；该记录必须引用真实 post-publish proof、release close preflight、stale claim audit、release evidence bundle SHA256、rollback plan 和 owner final close decision。

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Source Artifacts

$($sourceLines -join "`r`n")

## Safety Notes

$($safetyLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner proof backfill execution pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackageState=$packageState"
Write-Output "BackfillItemCount=$($backfillItems.Count)"
Write-Output "PerformsPublish=False"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
