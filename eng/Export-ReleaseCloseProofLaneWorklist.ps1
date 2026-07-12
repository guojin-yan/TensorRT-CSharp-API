[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

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

function New-ProofLane {
  param(
    [string]$Id,
    [string]$ProofKind,
    [string]$State,
    [string[]]$SourceArtifacts,
    [string[]]$ValidatorCommands,
    [string[]]$RequiredOwnerActions,
    [string[]]$RequiredFields,
    [string[]]$ForbiddenSubstitutes,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    proofKind = $ProofKind
    laneState = $State
    sourceArtifacts = @($SourceArtifacts)
    validatorCommands = @($ValidatorCommands)
    requiredOwnerActions = @($RequiredOwnerActions)
    requiredFields = @($RequiredFields)
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    blocked = $true
    boundary = $Boundary
  }
}

$yoloValidation = Read-JsonOrNull "artifacts\user-acceptance\yolovision-real-asset-owner-proof-input-validation.json"
$yoloImport = Read-JsonOrNull "artifacts\user-acceptance\yolovision-real-asset-owner-proof-import-report.json"
$packageOwnerValidation = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-owner-input-validation.json"
$packageWorklist = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-worklist.json"
$releaseCloseWorklist = Read-JsonOrNull "artifacts\final-release\release-close-proof-worklist.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"

$commonForbidden = @(
  "build-only",
  "dry-run",
  "template",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "TensorRtExec report",
  "YoloVision matrix",
  "OnnxToEngine report",
  "readonly diagnostics",
  "screenshot",
  "sidecar-only report",
  "skipped run",
  "blocked-by-cuda-driver"
)

$lanes = @(
  New-ProofLane `
    -Id "real-model-runtime" `
    -ProofKind "real-model-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $yoloValidation -Name "validationState" -DefaultValue "owner-action-required")) `
    -SourceArtifacts @(
      "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
      "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
      "artifacts/user-acceptance/yolovision-real-asset-owner-proof-import-report.json",
      "artifacts/user-acceptance/yolovision-real-asset-owner-sample-run-evidence.candidate.json"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    ) `
    -RequiredOwnerActions @(
      "Fill real ONNX, labels, input image, tensor, TensorRtExec report, engine, YoloVision run log, output JSON, SHA256, host metadata, and owner review.",
      "Preserve expected evidence lines including YoloVision Passed=True.",
      "Promote only after strict sample-run evidence validation passes with existing logs."
    ) `
    -RequiredFields @("runLogPath", "runLogSha256", "outputJsonPath", "outputJsonSha256", "hostOs", "gpuName", "driverVersion", "cudaVersion", "tensorRtVersion", "ownerReviewer", "ownerReviewedAtUtc") `
    -ForbiddenSubstitutes $commonForbidden `
    -Boundary "sample-run-evidence candidate is not package-consumer-runtime proof; real-model-runtime requires real logs, output JSON, hashes, host metadata, and owner review."
  New-ProofLane `
    -Id "package-consumer-runtime" `
    -ProofKind "package-consumer-runtime" `
    -State ([string](Get-PropertyOrDefault -Object $packageOwnerValidation -Name "validationState" -DefaultValue ([string](Get-PropertyOrDefault -Object $packageWorklist -Name "worklistState" -DefaultValue "owner-action-required")))) `
    -SourceArtifacts @(
      "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json",
      "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
      "artifacts/final-release/package-consumer-runtime-proof-worklist.json",
      "artifacts/final-release/package-consumer-runtime-proof-record.json"
    ) `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof"
    ) `
    -RequiredOwnerActions @(
      "Run clean external consumer outside the repository with a real public package source.",
      "Record publicPackageSource, package id/version, runtime package version, host OS, GPU, driver, CUDA, TensorRT, exitCode=0, stdout/stderr, and log hash.",
      "Confirm no local feed, no ProjectReference, and no direct .nupkg substitute was used."
    ) `
    -RequiredFields @("publicPackageSource", "managedPackageId", "managedPackageVersion", "runtimePackageVersion", "hostOs", "gpuName", "cudaDriverVersion", "cudaRuntimeVersion", "tensorRtVersion", "runtimeSmokeExitCode", "stdoutSummary", "stderrSummary", "runtimeSmokeLogSha256") `
    -ForbiddenSubstitutes (@($commonForbidden) + @("sample-run-evidence", "real-model-runtime")) `
    -Boundary "package-consumer-runtime is a separate release proof lane and cannot be replaced by sample-run-evidence, local feed, ProjectReference, direct .nupkg, build-only, dry-run, or template records."
  New-ProofLane `
    -Id "post-publish-verification" `
    -ProofKind "post-publish-verification" `
    -State ([string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "owner-action-required")) `
    -SourceArtifacts @("artifacts/final-release/post-publish-verification-record.json", "artifacts/final-release/post-publish-verification-validation.json") `
    -ValidatorCommands @("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -FailOnNotProof") `
    -RequiredOwnerActions @(
      "Publish through the real owner-approved public package channel.",
      "Record public package source, package URLs, hashes, install command, run command, logs, host metadata, and owner review.",
      "Do not use package-consumer-runtime pre-publish proof as post-publish verification."
    ) `
    -RequiredFields @("publicPackageSource", "publishedPackageUrl", "installLogPath", "installLogSha256", "runLogPath", "runLogSha256", "hostOs", "ownerReviewer", "ownerReviewedAtUtc") `
    -ForbiddenSubstitutes (@($commonForbidden) + @("package-consumer-runtime", "pre-publish validation")) `
    -Boundary "post-publish verification requires real public channel publish and clean install/run logs; package-consumer-runtime cannot replace it."
  New-ProofLane `
    -Id "public-owner-confirmation" `
    -ProofKind "public-owner-confirmation" `
    -State ([string](Get-PropertyOrDefault -Object $releaseCloseWorklist -Name "worklistState" -DefaultValue "owner-action-required")) `
    -SourceArtifacts @("artifacts/final-release/release-close-proof-worklist.json", "artifacts/final-release/release-issue-close-record-owner-input.json", "artifacts/final-release/release-issue-final-close-decision.json") `
    -ValidatorCommands @(
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecordOwnerInput.ps1 -Strict",
      "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueFinalCloseDecision.ps1 -Strict"
    ) `
    -RequiredOwnerActions @(
      "Review real-model-runtime, package-consumer-runtime, and post-publish verification lanes.",
      "Record release issue id, final owner decision, rollback owner, rollback trigger, and rollback plan.",
      "Close only after strict release close validation passes."
    ) `
    -RequiredFields @("releaseIssueId", "releaseIssueUrl", "finalOwnerDecision", "rollbackOwner", "rollbackTrigger", "rollbackPlan", "ownerReviewer", "ownerReviewedAtUtc") `
    -ForbiddenSubstitutes $commonForbidden `
    -Boundary "public owner confirmation is not proof by itself; it can only close the release after all real proof lanes pass strict validation."
)

$worklist = [ordered]@{
  recordKind = "release-close-proof-lane-worklist"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  worklistState = "blocked-real-owner-public-postpublish-proof-required"
  laneCount = $lanes.Count
  blockedLaneCount = ($lanes | Where-Object { $_.blocked }).Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  sourceArtifacts = @(
    "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
    "artifacts/user-acceptance/yolovision-real-asset-owner-proof-import-report.json",
    "artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json",
    "artifacts/final-release/package-consumer-runtime-proof-worklist.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/release-close-proof-worklist.json"
  )
  lanes = @($lanes)
  boundary = "Release close requires separate real-model-runtime, package-consumer-runtime, post-publish verification, and public owner confirmation lanes. sample-run-evidence candidates cannot replace package-consumer-runtime proof. Templates, candidates, sidecars, local feed, ProjectReference, direct .nupkg, build-only, dry-run, skipped, and blocked-by-cuda-driver records are not proof."
}

$jsonPath = Join-Path $OutputRoot "release-close-proof-lane-worklist.json"
$markdownPath = Join-Path $OutputRoot "release-close-proof-lane-worklist.md"
$worklist | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$laneRows = foreach ($lane in $lanes) {
  "| ``$(ConvertTo-MarkdownCell $lane.id)`` | ``$(ConvertTo-MarkdownCell $lane.proofKind)`` | ``$(ConvertTo-MarkdownCell $lane.laneState)`` | ``True`` |"
}

$markdown = @"
# Release Close Proof Lane Worklist

Generated at: ``$($worklist.generatedAtUtc)``

## Summary

- recordKind: ``$($worklist.recordKind)``
- worklistState: ``$($worklist.worklistState)``
- laneCount: ``$($worklist.laneCount)``
- blockedLaneCount: ``$($worklist.blockedLaneCount)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromoteRuntimeProof: ``False``

## Lanes

| Lane | Proof Kind | State | Blocked |
| --- | --- | --- | --- |
$($laneRows -join "`r`n")

## Boundary

$($worklist.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release close proof lane worklist written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "WorklistState=$($worklist.worklistState) LaneCount=$($worklist.laneCount) BlockedLaneCount=$($worklist.blockedLaneCount)"
