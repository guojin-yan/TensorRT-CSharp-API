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

function New-ExecutionStep {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Phase,
    [string]$Command,
    [string]$RequiredEvidence,
    [string]$Validator,
    [string]$CurrentState,
    [string]$Boundary,
    [string[]]$RequiredOwnerInputs = @(),
    [string[]]$ValidatorCommands = @(),
    [string[]]$ExpectedArtifacts = @(),
    [string[]]$PromotionBlockers = @()
  )

  if ($RequiredOwnerInputs.Count -eq 0) {
    $RequiredOwnerInputs = @($RequiredEvidence)
  }

  if ($ValidatorCommands.Count -eq 0) {
    $ValidatorCommands = @($Command)
  }

  [pscustomobject]@{
    id = $Id
    title = $Title
    phase = $Phase
    command = $Command
    requiredEvidence = $RequiredEvidence
    validator = $Validator
    requiredOwnerInputs = $RequiredOwnerInputs
    validatorCommands = $ValidatorCommands
    expectedArtifacts = $ExpectedArtifacts
    promotionBlockers = $PromotionBlockers
    currentState = $CurrentState
    ownerAction = "owner-action-required"
    performsPublish = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

function New-ManualPublishPlaceholder {
  param(
    [string]$Id,
    [string]$Title,
    [string]$CommandTemplate,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    commandTemplate = $CommandTemplate
    commandState = "owner-manual-command-placeholder"
    requiresOwnerAuthorization = $true
    performsPublish = $false
    executedByThisScript = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

$finalGap = Read-JsonOrNull "artifacts\final-release\release-candidate-final-gap-review.json"
$preflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$ownerPlanValidation = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan-validation.json"
$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$postPublishVerificationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$cleanConsumerScan = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-project-scan.json"
$sampleRunEvidenceValidation = Read-JsonOrNull "artifacts\user-acceptance\sample-run-evidence-record-validation.json"
$sampleAssetManifestAudit = Read-JsonOrNull "artifacts\user-acceptance\sample-asset-manifest-audit.json"
$linuxValidation = Read-JsonOrNull "artifacts\linux-dry-run\$LinuxRuntimePackageKey\linux-runner-evidence-validation.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"

$preflightState = [string](Get-PropertyOrDefault -Object $preflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$preflightFailedItemCount = [int](Get-PropertyOrDefault -Object $preflight -Name "failedItemCount" -DefaultValue -1)
$finalGapState = [string](Get-PropertyOrDefault -Object $finalGap -Name "reviewState" -DefaultValue "missing-final-gap-review")
$releaseEvidenceState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$ownerPlanState = [string](Get-PropertyOrDefault -Object $ownerPlanValidation -Name "validationState" -DefaultValue "missing-owner-authorized-publish-command-plan-validation")
$externalProofState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalProofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$externalProofCanPromote = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "canPromoteRuntimeProof" -DefaultValue $false)
$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishRequiredEvidence = @(
  Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRequiredEvidence" -DefaultValue @(
    Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishRequiredEvidence" -DefaultValue @()
  )
)
$postPublishRequiredEvidence = Normalize-PostPublishRequiredEvidence -Evidence $postPublishRequiredEvidence
$postPublishRequiredEvidenceCount = if ($postPublishRequiredEvidence.Count -gt 0) {
  $postPublishRequiredEvidence.Count
}
else {
  [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRequiredEvidenceCount" -DefaultValue (
    [int](Get-PropertyOrDefault -Object $postPublishVerificationValidation -Name "postPublishRequiredEvidenceCount" -DefaultValue 0)
  ))
}
$cleanConsumerScanState = [string](Get-PropertyOrDefault -Object $cleanConsumerScan -Name "scanState" -DefaultValue "missing-clean-consumer-project-scan")
$sampleRunState = [string](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "validationState" -DefaultValue "missing-sample-run-evidence-validation")
$sampleCanPromoteRealModel = [bool](Get-PropertyOrDefault -Object $sampleRunEvidenceValidation -Name "canPromoteRealModelRuntime" -DefaultValue $false)
$sampleManifestErrorCount = [int](Get-PropertyOrDefault -Object $sampleAssetManifestAudit -Name "errorCount" -DefaultValue -1)
$linuxState = [string](Get-PropertyOrDefault -Object $linuxValidation -Name "validationState" -DefaultValue "missing-linux-runner-evidence-validation")
$linuxProof = [bool](Get-PropertyOrDefault -Object $linuxValidation -Name "isRealLinuxRunnerProof" -DefaultValue $false)
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)
$articleCount = [int](Get-PropertyOrDefault -Object (Get-PropertyOrDefault -Object $finalGap -Name "articleMatrix" -DefaultValue $null) -Name "articleCount" -DefaultValue -1)

$executionSteps = @(
  New-ExecutionStep `
    -Id "refresh-stale-release-claims" `
    -Title "Refresh stale release claims audit" `
    -Phase "pre-publish-proof-collection" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1" `
    -RequiredEvidence "artifacts/final-release/stale-release-claims-audit.json with findingCount=0" `
    -Validator "Test-StaleReleaseClaims.ps1" `
    -CurrentState "findingCount=$staleFindingCount" `
    -Boundary "The stale claim audit is a guardrail, not publication approval or release proof." `
    -RequiredOwnerInputs @(
      "release-facing README/docs/articles/artifacts with no premature publication or proof completion claims",
      "stale-release-claims-audit.json with findingCount=0"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/stale-release-claims-audit.json",
      "artifacts/final-release/stale-release-claims-audit.md"
    ) `
    -PromotionBlockers @(
      "published to NuGet claim before owner proof",
      "release issue can be closed claim before validators pass"
    )

  New-ExecutionStep `
    -Id "collect-external-runtime-proof" `
    -Title "Collect package-consumer-runtime proof on compatible host" `
    -Phase "pre-publish-proof-collection" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredEvidence "external-runtime-proof-record.json with proofClassification=package-consumer-runtime, matching package hashes, clean consumer identity, smoke command, stdout/stderr summary, and log SHA256" `
    -Validator "Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -CurrentState "validationState=$externalProofState; proofClassification=$externalProofClassification; canPromoteRuntimeProof=$externalProofCanPromote" `
    -Boundary "local feed, ProjectReference, DependencyProbe, build-only, dependency-probe-only, blocked-by-cuda-driver, sidecar-only, managed-readiness, CallbackAllocatorReadinessSnapshot, precheck-only, dry-run-only, and schema-only cannot substitute package-consumer-runtime proof." `
    -RequiredOwnerInputs @(
      "managed/runtime nupkg SHA256 from the exact release candidate artifacts",
      "runtimePackageKey=$RuntimePackageKey",
      "clean consumer project outside this repository",
      "host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
      "restore/build/dependency probe/runtime smoke logs",
      "stdoutSummary and stderrSummary reviewed from real logs",
      "smoke log path plus matching 64-character SHA256",
      "no ProjectReference"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/external-runtime-proof-record.json",
      "artifacts/final-release/external-runtime-proof-validation.json",
      "artifacts/final-release/external-runtime-proof-validation.md"
    ) `
    -PromotionBlockers @(
      "bridge-only package consumer log",
      "Skipped=True",
      "dependency-probe-only",
      "managed-readiness",
      "CallbackAllocatorReadinessSnapshot",
      "precheck-only",
      "dry-run-only",
      "schema-only",
      "WrapperSurfaceEvidenceKind=compile-surface-proof",
      "IsRuntimeExecutionProof=False",
      "ProjectReference",
      "mismatched log SHA256"
    )

  New-ExecutionStep `
    -Id "collect-real-model-runtime-proof" `
    -Title "Collect Classification/YoloVision real-model-runtime proof" `
    -Phase "pre-publish-proof-collection" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1" `
    -RequiredEvidence "model, labels, input tensor, license note, SHA256 values, TensorRtExec build sidecar, sample runner log, and sample-run-evidence record" `
    -Validator "Test-SampleAssetManifest.ps1 + Test-SampleRunEvidenceRecord.ps1" `
    -CurrentState "sampleRunEvidence=$sampleRunState; canPromoteRealModelRuntime=$sampleCanPromoteRealModel; sampleManifestErrorCount=$sampleManifestErrorCount" `
    -Boundary "Classification/YoloVision sample evidence can promote only to real-model-runtime; it cannot claim package-consumer-runtime." `
    -RequiredOwnerInputs @(
      "Classification real model asset, labels, license note, and SHA256 values",
      "YoloVision real assets covering YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom where available",
      "YoloVision task evidence for det/cls/seg/obb/pose/sem",
      "TensorRtExec build sidecar for the real model",
      "sample runner log path plus matching SHA256",
      "sample-run-evidence record with proofClassification=real-model-runtime"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    ) `
    -ExpectedArtifacts @(
      "samples/assets/classification-assets.json",
      "samples/assets/yolovision-assets.json",
      "artifacts/user-acceptance/sample-run-evidence-record.json",
      "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
    ) `
    -PromotionBlockers @(
      "build-only",
      "parse-only",
      "sidecar-only",
      "missing model/input/license hashes",
      "package-consumer-runtime classification"
    )

  New-ExecutionStep `
    -Id "collect-linux-runner-proof" `
    -Title "Collect Linux compatible host runner proof" `
    -Phase "pre-publish-proof-collection" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey" `
    -RequiredEvidence "Linux runner record, runtime package key, logs, validator output, and compatible CUDA/TensorRT host metadata" `
    -Validator "Test-LinuxRunnerEvidenceRecord.ps1" `
    -CurrentState "validationState=$linuxState; isRealLinuxRunnerProof=$linuxProof" `
    -Boundary "Windows handoff, template-only records, and blocked-by-cuda-driver are not Linux runner proof." `
    -RequiredOwnerInputs @(
      "real Linux x64 compatible host run",
      "linuxRuntimePackageKey=$LinuxRuntimePackageKey",
      "Linux runner evidence record",
      "host CUDA/TensorRT/cuDNN metadata",
      "runner log path plus matching SHA256"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey"
    ) `
    -ExpectedArtifacts @(
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-record.json",
      "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
    ) `
    -PromotionBlockers @(
      "Windows handoff for Linux proof",
      "template-only record",
      "blocked-by-cuda-driver"
    )

  New-ExecutionStep `
    -Id "validate-owner-proof-input-record" `
    -Title "Validate owner proof input record before owner authorization" `
    -Phase "owner-proof-input" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1" `
    -RequiredEvidence "release-owner-proof-input-record-validation.json with selected channel, package URL/hash, clean consumer, runtime log/hash, stdout/stderr summary, host metadata, and no local feed/ProjectReference/direct .nupkg substitutions" `
    -Validator "Test-ReleaseOwnerProofInputRecord.ps1" `
    -CurrentState "template/default validation remains blocked until owner-filled input exists" `
    -Boundary "Owner proof input validates concrete owner-filled evidence fields only; it does not publish packages or close the release issue." `
    -RequiredOwnerInputs @(
      "release-owner-proof-input-record.json",
      "owner authorization identity and decision id",
      "selected channel URL",
      "managed/runtime package URL and SHA256",
      "clean consumer project outside the repo",
      "restore/native asset/dependency probe/runtime smoke logs and SHA256",
      "stdout/stderr summary",
      "host metadata"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseOwnerProofInputRecordTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1 -InputPath artifacts/final-release/release-owner-proof-input-record.json -RequireExistingLogs -FailOnNotProof"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/release-owner-proof-input-record-template.json",
      "artifacts/final-release/release-owner-proof-input-record-validation.json"
    ) `
    -PromotionBlockers @(
      "template",
      "schema-only",
      "managed-readiness",
      "local feed",
      "ProjectReference",
      "direct .nupkg reference",
      "missing log SHA256"
    )

  New-ExecutionStep `
    -Id "validate-owner-authorization" `
    -Title "Validate owner authorization before manual publish" `
    -Phase "owner-authorization" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -RequiredEvidence "owner-authorized-publish-command-plan-validation.json with explicit owner authorization and no placeholder-only command materialization" `
    -Validator "Test-OwnerAuthorizedPublishCommandPlan.ps1" `
    -CurrentState "validationState=$ownerPlanState" `
    -Boundary "No script in this package executes or authorizes publication; owner authorization remains an external human decision." `
    -RequiredOwnerInputs @(
      "explicit non-template owner approval input",
      "release channel selection",
      "NVIDIA redistribution disposition",
      "owner-reviewed package identities and SHA256 hashes",
      "manual publish command materialization outside automation"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/release-owner-approval-input-validation.json",
      "artifacts/final-release/owner-authorized-publish-command-plan-validation.json"
    ) `
    -PromotionBlockers @(
      "template",
      "draft",
      "owner-action-required without validator pass",
      "placeholder-only command materialization"
    )

  New-ExecutionStep `
    -Id "post-publish-clean-consumer-scan" `
    -Title "Scan clean post-publish consumer project" `
    -Phase "post-publish-proof-collection" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1 -ConsumerProject <owner-clean-consumer.csproj>" `
    -RequiredEvidence "clean consumer scan with no ProjectReference and package identity from a real public or configured release channel" `
    -Validator "Test-PostPublishCleanConsumerProject.ps1" `
    -CurrentState "scanState=$cleanConsumerScanState" `
    -Boundary "Clean consumer scanning is helper evidence until paired with a real post-publish verification proof record." `
    -RequiredOwnerInputs @(
      "clean consumer .csproj outside this repository",
      "PackageReference-only consumption with no ProjectReference",
      "real public or configured release channel package source"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1 -ConsumerProject <owner-clean-consumer.csproj>"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/post-publish-clean-consumer-project-scan.json",
      "artifacts/final-release/post-publish-clean-consumer-project-scan.md"
    ) `
    -PromotionBlockers @(
      "helper scan without post-publish verification record",
      "ProjectReference",
      "local feed"
    )

  New-ExecutionStep `
    -Id "validate-post-publish-verification" `
    -Title "Validate post-publish verification proof" `
    -Phase "post-publish-proof-collection" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -RequiredEvidence "post-publish-verification-record.json with real channel package identity, downloaded nupkg hashes, clean consumer restore/build/smoke logs, stdout/stderr summaries, and matching SHA256 values" `
    -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof" `
    -CurrentState "validationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof" `
    -Boundary "template, draft, local feed, ProjectReference, and helper scans cannot substitute post-publish verification proof." `
    -RequiredOwnerInputs @(
      $postPublishRequiredEvidence
      "real published channel URL/source",
      "downloaded managed/runtime nupkg SHA256 values and timestamped SHA256 source notes",
      "cleanConsumerRoot outside this repository",
      "no ProjectReference",
      "restore/build/native asset/dependency probe/runtime smoke logs",
      "stdoutSummary and stderrSummary reviewed from real post-publish logs",
      "matching SHA256 values for every recorded log"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/post-publish-verification-record.json",
      "artifacts/final-release/post-publish-verification-validation.json",
      "artifacts/final-release/post-publish-verification-validation.md"
    ) `
    -PromotionBlockers @(
      "template",
      "draft",
      "local feed",
      "ProjectReference",
      "bridge-only package consumer log",
      "dependency-probe-only",
      "mismatched log SHA256",
      "missing execution steps"
    )

  New-ExecutionStep `
    -Id "refresh-close-preflight" `
    -Title "Refresh final release close preflight" `
    -Phase "release-close-review" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1" `
    -RequiredEvidence "release-close-preflight.json with all proof items passing before release issue closure" `
    -Validator "Export-ReleaseClosePreflight.ps1" `
    -CurrentState "preflightState=$preflightState; failedItemCount=$preflightFailedItemCount" `
    -Boundary "Release close preflight aggregates proof gaps; it is not owner authorization, runtime proof, post-publish proof, or package push." `
    -RequiredOwnerInputs @(
      "external runtime proof validator passing",
      "Linux runner proof validator passing",
      "real-model-runtime proof validator passing",
      "owner authorization validator passing",
      "post-publish verification validator passing"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseClosePreflight.ps1"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/release-close-preflight.json",
      "artifacts/final-release/release-close-preflight.md"
    ) `
    -PromotionBlockers @(
      "any failed preflight item",
      "owner-action-required",
      "blocked-real-proof-required"
    )

  New-ExecutionStep `
    -Id "validate-release-issue-close-record" `
    -Title "Validate final release issue close record" `
    -Phase "release-close-review" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1" `
    -RequiredEvidence "release-issue-close-record-validation.json with non-template owner final close decision, real post-publish proof, release close preflight pass, stale claim audit findingCount=0, evidence bundle SHA256 match, and rollback plan" `
    -Validator "Test-ReleaseIssueCloseRecord.ps1" `
    -CurrentState "template/default validation remains blocked until real owner-filled close record and post-publish proof exist" `
    -Boundary "Release issue close record validates final owner close evidence only; it does not publish packages or close the issue by itself." `
    -RequiredOwnerInputs @(
      "release-issue-close-record.json",
      "release issue id and URL",
      "owner identity, approval timestamp, and final close decision",
      "selected channel",
      "managed/runtime package URL and SHA256",
      "post-publish verification validation path",
      "release close preflight path",
      "stale release claims audit path",
      "release evidence bundle SHA256",
      "rollback/yanking/deprecation plan"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseIssueCloseRecordTemplate.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady"
    ) `
    -ExpectedArtifacts @(
      "artifacts/final-release/release-issue-close-record-template.json",
      "artifacts/final-release/release-issue-close-record-validation.json"
    ) `
    -PromotionBlockers @(
      "template",
      "schema-only",
      "missing owner final close decision",
      "missing post-publish proof",
      "release close preflight blocked",
      "stale release claims",
      "mismatched evidence bundle SHA256"
    )
)

$publishPlaceholders = @(
  New-ManualPublishPlaceholder `
    -Id "nuget-org-managed-package" `
    -Title "Manual NuGet.org managed package push placeholder" `
    -CommandTemplate "dotnet nuget push <owner-reviewed-managed.nupkg> --api-key <owner-secret> --source https://api.nuget.org/v3/index.json" `
    -Boundary "This package records the owner command template only; it never runs dotnet nuget push and cannot prove publication."

  New-ManualPublishPlaceholder `
    -Id "nuget-org-runtime-packages" `
    -Title "Manual NuGet.org runtime package push placeholder" `
    -CommandTemplate "dotnet nuget push <owner-reviewed-runtime.nupkg> --api-key <owner-secret> --source https://api.nuget.org/v3/index.json" `
    -Boundary "Runtime package publication requires owner authorization and channel policy review; this script does not publish."

  New-ManualPublishPlaceholder `
    -Id "github-packages" `
    -Title "Manual GitHub Packages upload placeholder" `
    -CommandTemplate "dotnet nuget push <owner-reviewed.nupkg> --api-key <owner-secret> --source <owner-github-packages-source>" `
    -Boundary "GitHub Packages upload remains a manual owner action and is not executed by this package."

  New-ManualPublishPlaceholder `
    -Id "github-release-assets" `
    -Title "Manual GitHub Release asset upload placeholder" `
    -CommandTemplate "gh release upload <tag> <owner-reviewed-artifacts> --repo <owner-repo>" `
    -Boundary "GitHub Release upload remains a manual owner action and is not executed by this package."
)

$oneScreenReleaseHoldChecklist = @(
  [pscustomobject]@{
    id = "owner-authorization"
    ownerVisibleBlocker = "Owner authorization is still pending."
    currentState = "ownerPlanValidationState=$ownerPlanState"
    ownerNextAction = "Fill the non-template owner approval input, select the release channel, record NVIDIA redistribution disposition, and rerun owner validators."
    firstCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1"
    requiredRealInputs = @(
      "release-owner-proof-input-record.json",
      "non-template owner approval input",
      "owner-approved release channel",
      "NVIDIA redistribution disposition",
      "owner-reviewed package identities and SHA256 hashes"
    )
    cannotUse = @("template", "draft", "placeholder-only command materialization", "owner-action-required without validator pass")
    performsPublish = $false
    canCloseReleaseIssue = $false
  }

  [pscustomobject]@{
    id = "package-consumer-runtime"
    ownerVisibleBlocker = "Package-consumer-runtime proof is missing."
    currentState = "validationState=$externalProofState; proofClassification=$externalProofClassification; canPromoteRuntimeProof=$externalProofCanPromote"
    ownerNextAction = "Run the release candidate packages in a clean consumer on a compatible CUDA/TensorRT host and validate external-runtime-proof-record.json."
    firstCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath artifacts/final-release/external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
    requiredRealInputs = @(
      "clean consumer project outside this repository",
      "managed/runtime nupkg SHA256 values",
      "host OS/GPU/driver/CUDA/TensorRT/cuDNN metadata",
      "restore/build/dependency probe/runtime smoke logs",
      "stdoutSummary/stderrSummary and matching log SHA256"
    )
    cannotUse = @("local feed", "ProjectReference", "DependencyProbe", "dependency-probe-only", "blocked-by-cuda-driver", "bridge-only package consumer log", "mismatched log SHA256")
    performsPublish = $false
    canCloseReleaseIssue = $false
  }

  [pscustomobject]@{
    id = "real-model-runtime"
    ownerVisibleBlocker = "Classification/YoloVision real-model-runtime proof is missing."
    currentState = "validationState=$sampleRunEvidenceState; canPromoteRealModelRuntime=$sampleCanPromoteRealModel; sampleAssetManifestErrorCount=$sampleManifestErrorCount"
    ownerNextAction = "Backfill real Classification and YoloVision assets, run sample evidence, and validate sample-run-evidence-record.json."
    firstCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleAssetManifest.ps1"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    requiredRealInputs = @(
      "Classification real model asset, labels, license note, and SHA256 values",
      "YoloVision assets for YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom where available",
      "det/cls/seg/obb/pose/sem task evidence",
      "TensorRtExec sidecar/build report",
      "sample runner log path plus matching SHA256"
    )
    cannotUse = @("build-only", "parse-only", "sidecar-only", "missing model/input/license hashes", "package-consumer-runtime classification")
    performsPublish = $false
    canCloseReleaseIssue = $false
  }

  [pscustomobject]@{
    id = "linux-runner-proof"
    ownerVisibleBlocker = "Linux runner proof is missing."
    currentState = "validationState=$linuxState; isRealLinuxRunnerProof=$linuxProof"
    ownerNextAction = "Run Linux runner evidence on a real Linux x64 compatible CUDA/TensorRT host and attach the validation output."
    firstCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $LinuxRuntimePackageKey"
    requiredRealInputs = @(
      "real Linux x64 compatible host run",
      "linuxRuntimePackageKey=$LinuxRuntimePackageKey",
      "host CUDA/TensorRT/cuDNN metadata",
      "runner log path plus matching SHA256"
    )
    cannotUse = @("Windows handoff for Linux proof", "template-only record", "blocked-by-cuda-driver")
    performsPublish = $false
    canCloseReleaseIssue = $false
  }

  [pscustomobject]@{
    id = "post-publish-verification"
    ownerVisibleBlocker = "Post-publish verification proof is missing."
    currentState = "validationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof"
    ownerNextAction = "After owner-approved publication, verify a clean external consumer against the real channel packages and validate post-publish-verification-record.json."
    firstCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishCleanConsumerProject.ps1 -ConsumerProject <owner-clean-consumer.csproj>"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof"
    requiredRealInputs = @(
      $postPublishRequiredEvidence
      "real published channel URL/source",
      "downloaded managed/runtime nupkg SHA256 values",
      "cleanConsumerRoot outside this repository",
      "restore/build/native asset/dependency probe/runtime smoke logs",
      "stdoutSummary/stderrSummary and matching SHA256 values"
    )
    cannotUse = @("template", "draft", "local feed", "ProjectReference", "bridge-only package consumer log", "dependency-probe-only", "mismatched log SHA256")
    performsPublish = $false
    canCloseReleaseIssue = $false
  }
)

$mustNotSubstitute = @(
  "helper",
  "template",
  "draft",
  "runbook",
  "collection package",
  "input package",
  "local inventory",
  "local feed",
  "ProjectReference",
  "DependencyProbe",
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
  "bridge-only package consumer log",
  "bridge-only wrapper surface",
  "Skipped=True",
  "WrapperSurfaceEvidenceKind=compile-surface-proof",
  "IsRuntimeExecutionProof=False",
  "mismatched log SHA256",
  "Parser/ParserRefitter diagnostic snapshots",
  "copied managed diagnostic snapshot",
  "build-only",
  "parse-only",
  "sidecar-only",
  "Windows handoff for Linux proof"
)

$sourceArtifacts = @(
  "artifacts/final-release/release-candidate-final-gap-review.json",
  "artifacts/final-release/release-close-preflight.json",
  "artifacts/final-release/release-evidence-bundle.json",
  "artifacts/final-release/callback-runtime-proof-execution-pack.json",
  "artifacts/final-release/callback-runtime-proof-execution-pack-validation.json",
  "artifacts/final-release/owner-action-required.md",
  "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
  "artifacts/final-release/release-owner-proof-input-record-template.json",
  "artifacts/final-release/release-owner-proof-input-record-validation.json",
  "artifacts/final-release/release-issue-close-record-template.json",
  "artifacts/final-release/release-issue-close-record-validation.json",
  "artifacts/final-release/external-runtime-proof-validation.json",
  "artifacts/final-release/compatible-host-runtime-proof-collection-bundle.json",
  "artifacts/final-release/real-model-and-package-proof-input-package.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/final-release/release-package-proof-bundle.json",
  "artifacts/final-release/final-package-review-bundle.json",
  "artifacts/final-release/post-publish-clean-consumer-project-scan.json",
  "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
  "artifacts/user-acceptance/sample-asset-manifest-audit.json",
  "artifacts/linux-dry-run/$LinuxRuntimePackageKey/linux-runner-evidence-validation.json"
)

$packageState = if (
  [string]::Equals($preflightState, "ready-to-close", [System.StringComparison]::OrdinalIgnoreCase) -and
  $externalProofCanPromote -and
  $postPublishProof -and
  $linuxProof
) {
  "ready-for-owner-close-review"
}
else {
  "blocked-real-proof-required"
}

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "owner-release-execution-package"
  packageState = $packageState
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  ownerActionStatus = "owner-action-required"
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isRealModelRuntimeProof = $false
  isPostPublishVerificationProof = $false
  requiresHumanOwner = $true
  finalGapReviewState = $finalGapState
  releaseEvidenceBundleState = $releaseEvidenceState
  preflightState = $preflightState
  preflightFailedItemCount = $preflightFailedItemCount
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidenceCount
  articleCount = $articleCount
  oneScreenReleaseHoldChecklist = $oneScreenReleaseHoldChecklist
  oneScreenReleaseHoldChecklistCount = @($oneScreenReleaseHoldChecklist).Count
  executionSteps = $executionSteps
  manualPublishPlaceholders = $publishPlaceholders
  proofBackfillOrder = @(
    "refresh stale claim audit",
    "fill and validate release-owner-proof-input-record.json",
    "collect package-consumer-runtime proof on compatible host",
    "collect Classification/YoloVision real-model-runtime proof",
    "collect Linux runner proof",
    "validate explicit owner authorization",
    "owner manually executes publish commands outside this script",
    "scan clean post-publish consumer project",
    "validate post-publish verification proof",
    "refresh release close preflight",
    "fill and validate release-issue-close-record.json"
  )
  mustNotSubstitute = $mustNotSubstitute
  requiredOwnerInputs = @($executionSteps | ForEach-Object { $_.requiredOwnerInputs } | Select-Object -Unique)
  validatorCommands = @($executionSteps | ForEach-Object { $_.validatorCommands } | Select-Object -Unique)
  expectedArtifacts = @($executionSteps | ForEach-Object { $_.expectedArtifacts } | Select-Object -Unique)
  promotionBlockers = @($executionSteps | ForEach-Object { $_.promotionBlockers } | Select-Object -Unique)
  sourceEvidence = $sourceArtifacts
  sourceArtifacts = $sourceArtifacts
  safetyNotes = @(
    "This package is owner guidance only and does not publish packages.",
    "release-owner-proof-input-record-validation is a concrete owner input validator, but it is still blocked until a real owner-filled record with matching log hashes exists.",
    "dotnet nuget push, GitHub Packages upload, and GitHub Release upload are manual owner placeholders only.",
    "package-consumer-runtime proof requires a real clean consumer runtime smoke record and strict validator pass.",
    "real-model-runtime proof for Classification/YoloVision requires real model assets, hashes, licenses, runner logs, and sample evidence records.",
    "post-publish verification proof requires a real channel package identity and downloaded package hashes.",
    "release-issue-close-record-validation remains blocked until owner final close decision, release close preflight, stale audit, post-publish proof, and evidence bundle SHA256 all validate.",
    "blocked-by-cuda-driver is an environment blocker, not smoke passed and not API proof.",
    "TrtexecAlignmentStatus=parse-only remains a parse/report boundary for advanced TensorRtExec options.",
    "bridge-only package consumer log, bridge-only wrapper surface, Skipped=True, WrapperSurfaceEvidenceKind=compile-surface-proof, IsRuntimeExecutionProof=False, mismatched log SHA256, Parser/ParserRefitter diagnostic snapshots, copied managed diagnostic snapshot, managed-readiness, CallbackAllocatorReadinessSnapshot, precheck-only, dry-run-only, schema-only, and template-only release issue close records remain non-substitute proof kinds."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-release-execution-package.json"
$markdownPath = Join-Path $artifactRoot "owner-release-execution-package.md"

Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 8)
$stepRows = $executionSteps | ForEach-Object {
  $step = $_
  "| ``$($step.id)`` | $($step.phase) | $($step.currentState) | ``$($step.validator)`` | $($step.boundary) |"
}

$placeholderRows = $publishPlaceholders | ForEach-Object {
  $placeholder = $_
  "| ``$($placeholder.id)`` | ``$($placeholder.commandTemplate)`` | ``$($placeholder.commandState)`` | $($placeholder.boundary) |"
}

$oneScreenReleaseHoldRows = $oneScreenReleaseHoldChecklist | ForEach-Object {
  $item = $_
  "| ``$($item.id)`` | $($item.ownerVisibleBlocker) | $($item.currentState) | $($item.ownerNextAction) | ``$($item.firstCommand)`` | ``$($item.validatorCommand)`` |"
}

$nonSubstituteLines = $mustNotSubstitute | ForEach-Object { "- ``$_``" }
$requiredOwnerInputLines = $record.requiredOwnerInputs | ForEach-Object { "- ``$_``" }
$validatorCommandLines = $record.validatorCommands | ForEach-Object { "- ``$_``" }
$expectedArtifactLines = $record.expectedArtifacts | ForEach-Object { "- ``$_``" }
$promotionBlockerLines = $record.promotionBlockers | ForEach-Object { "- ``$_``" }
$sourceLines = $sourceArtifacts | ForEach-Object { "- ``$_``" }
$safetyLines = $record.safetyNotes | ForEach-Object { "- $_" }
$postPublishRequiredEvidenceLines = $postPublishRequiredEvidence | ForEach-Object { "- ``$_``" }

$markdown = @"
# Owner Release Execution Package

生成时间：$($record.generatedAtUtc)

## 总结

该执行包把最终总检中的 blocker 转成 owner 可执行、可回填、可验证的顺序。它是 owner guidance，不是 owner authorization、runtime proof、post-publish proof、release close approval 或 package push。``performsPublish=false``，``canPublishPublicly=false``，``canCloseReleaseIssue=false``。

## 当前状态

| 项目 | 当前值 |
|---|---|
| packageState | ``$packageState`` |
| finalGapReviewState | ``$finalGapState`` |
| releaseEvidenceBundleState | ``$releaseEvidenceState`` |
| preflightState | ``$preflightState`` |
| preflightFailedItemCount | ``$preflightFailedItemCount`` |
| postPublishRequiredEvidenceCount | ``$postPublishRequiredEvidenceCount`` |
| runtimePackageKey | ``$RuntimePackageKey`` |
| linuxRuntimePackageKey | ``$LinuxRuntimePackageKey`` |
| articleCount | ``$articleCount`` |
| oneScreenReleaseHoldChecklistCount | ``$($oneScreenReleaseHoldChecklist.Count)`` |

## 一屏 Release Hold 清单

这张表是 owner 当前最短执行面：它只汇总 blocker、下一步动作、首个命令和 validator，不执行发布、不上传包、不把 guidance 当 proof。

| Blocker | Owner visible blocker | 当前状态 | Owner 下一步 | First command | Validator |
|---|---|---|---|---|---|
$($oneScreenReleaseHoldRows -join "`r`n")

## Owner 执行顺序

| ID | 阶段 | 当前状态 | Validator | 边界 |
|---|---|---|---|---|
$($stepRows -join "`r`n")

## Post-Publish Required Evidence

$($postPublishRequiredEvidenceLines -join "`r`n")

## 手动发布占位命令

这些命令模板仅用于 owner 手动执行前审阅和材料化。本脚本不会执行 ``dotnet nuget push``、GitHub Packages 上传或 GitHub Release 上传。

| ID | Command template | State | 边界 |
|---|---|---|---|
$($placeholderRows -join "`r`n")

## Proof 回填顺序

1. refresh stale claim audit。
2. collect package-consumer-runtime proof on compatible host。
3. collect Classification/YoloVision real-model-runtime proof。
4. collect Linux runner proof。
5. validate explicit owner authorization。
6. owner manually executes publish commands outside this script。
7. scan clean post-publish consumer project。
8. validate post-publish verification proof。
9. refresh release close preflight。
10. fill and validate release-issue-close-record.json。

## Required Owner Inputs

$($requiredOwnerInputLines -join "`r`n")

## Validator Commands

$($validatorCommandLines -join "`r`n")

## Expected Artifacts

$($expectedArtifactLines -join "`r`n")

## Promotion Blockers

$($promotionBlockerLines -join "`r`n")

## 不可替代材料

$($nonSubstituteLines -join "`r`n")

## 来源材料

$($sourceLines -join "`r`n")

## Safety Notes

$($safetyLines -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Owner release execution package written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackageState=$packageState"
Write-Output "PerformsPublish=False"
Write-Output "CanCloseReleaseIssue=False"
