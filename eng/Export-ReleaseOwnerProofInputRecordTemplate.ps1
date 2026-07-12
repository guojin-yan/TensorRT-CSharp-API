[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Set-ContentWithRetry {
  param(
    [Parameter(Mandatory = $true)][string]$LiteralPath,
    [Parameter(Mandatory = $true)][object]$Value,
    [int]$RetryCount = 8,
    [int]$DelayMilliseconds = 125
  )

  for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
    try {
      Set-Content -LiteralPath $LiteralPath -Value $Value -Encoding utf8
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $RetryCount) { throw }
      Start-Sleep -Milliseconds ($DelayMilliseconds * $attempt)
    }
    catch [System.UnauthorizedAccessException] {
      if ($attempt -eq $RetryCount) { throw }
      Start-Sleep -Milliseconds ($DelayMilliseconds * $attempt)
    }
  }
}

$requiredOwnerInputFields = @(
  "ownerAuthorization.ownerName",
  "ownerAuthorization.ownerDecisionId",
  "ownerAuthorization.approvalTimestampUtc",
  "ownerAuthorization.targetChannel",
  "ownerAuthorization.releaseIssueUrl",
  "selectedChannel.channelName",
  "selectedChannel.channelSourceUri",
  "packages.managed.packageId",
  "packages.managed.packageVersion",
  "packages.managed.packageUrl",
  "packages.managed.nupkgSha256",
  "packages.runtime.packageId",
  "packages.runtime.packageVersion",
  "packages.runtime.packageUrl",
  "packages.runtime.nupkgSha256",
  "cleanConsumer.cleanConsumerRoot",
  "cleanConsumer.consumerProjectPath",
  "cleanConsumer.noProjectReference",
  "cleanConsumer.noLocalPackageSource",
  "cleanConsumer.noLocalNupkgPackageReference",
  "runtimeEvidence.restoreLogPath",
  "runtimeEvidence.restoreLogSha256",
  "runtimeEvidence.nativeAssetListingPath",
  "runtimeEvidence.nativeAssetListingSha256",
  "runtimeEvidence.dependencyProbeLogPath",
  "runtimeEvidence.dependencyProbeLogSha256",
  "runtimeEvidence.runtimeSmokeLogPath",
  "runtimeEvidence.runtimeSmokeLogSha256",
  "runtimeEvidence.runtimeSmokeExitCode",
  "runtimeEvidence.runtimeSmokePassed",
  "runtimeEvidence.stdoutSummary",
  "runtimeEvidence.stderrSummary",
  "hostMetadata.osDescription",
  "hostMetadata.gpuName",
  "hostMetadata.driverVersion",
  "hostMetadata.cudaDriverVersion",
  "hostMetadata.cudaRuntimeVersion",
  "hostMetadata.tensorRtRuntimeVersion",
  "hostMetadata.tensorRtRuntimeLine",
  "hostMetadata.cudnnVersion",
  "acknowledgements.notTemplate",
  "acknowledgements.noLocalFeed",
  "acknowledgements.noProjectReference",
  "acknowledgements.noDirectNupkgReference",
  "acknowledgements.notManagedReadiness",
  "acknowledgements.realRuntimeSmokeExecuted",
  "acknowledgements.hashesComputedFromReferencedFiles"
)

$nonSubstituteProofKinds = @(
  "template",
  "draft",
  "example",
  "runbook",
  "dashboard",
  "readiness snapshot",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "direct .nupkg reference",
  "dependency-probe-only",
  "build-only",
  "parse-only",
  "sidecar-only",
  "blocked-by-cuda-driver",
  "managed-readiness",
  "managed-readiness-only",
  "callback-allocator-readiness-snapshot",
  "CallbackAllocatorReadinessSnapshot",
  "TensorRtCallbackAllocatorReadinessSnapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-owner-proof-input-record-template"
  recordState = "template-only"
  proofClassification = "template-only"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRealOwnerProofInput = $false
  canPromoteOwnerProofInput = $false
  validationTarget = "artifacts/final-release/release-owner-proof-input-record.json"
  requiredOwnerInputFields = $requiredOwnerInputFields
  requiredOwnerInputFieldCount = $requiredOwnerInputFields.Count
  ownerAuthorization = [ordered]@{
    ownerName = ""
    ownerDecisionId = ""
    approvalTimestampUtc = $null
    targetChannel = ""
    releaseIssueUrl = ""
    approvedProofBundleSha256 = ""
    rollbackPlanUri = ""
    approvalRationale = ""
  }
  selectedChannel = [ordered]@{
    channelName = ""
    channelSourceUri = ""
    publishedAtUtc = $null
    packageSourceNotes = ""
  }
  packages = [ordered]@{
    managed = [ordered]@{
      packageId = "JYPPX.TensorRT.CSharp.API"
      packageVersion = ""
      packageUrl = ""
      nupkgSha256 = ""
      sha256Source = ""
    }
    runtime = [ordered]@{
      packageId = ""
      packageVersion = ""
      packageUrl = ""
      nupkgSha256 = ""
      sha256Source = ""
      runtimePackageKey = $RuntimePackageKey
    }
  }
  cleanConsumer = [ordered]@{
    cleanConsumerRoot = ""
    consumerProjectName = ""
    consumerProjectPath = ""
    noProjectReference = $false
    noLocalPackageSource = $false
    noLocalNupkgPackageReference = $false
    restoreSourceSummary = ""
  }
  runtimeEvidence = [ordered]@{
    restoreLogPath = ""
    restoreLogSha256 = ""
    nativeAssetListingPath = ""
    nativeAssetListingSha256 = ""
    dependencyProbeLogPath = ""
    dependencyProbeLogSha256 = ""
    runtimeSmokeCommand = ""
    runtimeSmokeLogPath = ""
    runtimeSmokeLogSha256 = ""
    runtimeSmokeExitCode = $null
    runtimeSmokePassed = $false
    stdoutSummary = ""
    stderrSummary = ""
  }
  hostMetadata = [ordered]@{
    osDescription = ""
    architecture = ""
    gpuName = ""
    driverVersion = ""
    cudaDriverVersion = ""
    cudaRuntimeVersion = ""
    tensorRtRuntimeVersion = ""
    tensorRtRuntimeLine = ""
    cudnnVersion = ""
  }
  acknowledgements = [ordered]@{
    notTemplate = $false
    noLocalFeed = $false
    noProjectReference = $false
    noDirectNupkgReference = $false
    notManagedReadiness = $false
    realRuntimeSmokeExecuted = $false
    hashesComputedFromReferencedFiles = $false
  }
  nonSubstituteProofKinds = $nonSubstituteProofKinds
  validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerProofInputRecord.ps1 -InputPath artifacts/final-release/release-owner-proof-input-record.json -RequireExistingLogs -FailOnNotProof"
  boundary = "This template is owner-filled input material only. It does not publish packages, authorize publication, prove runtime execution, or close the release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "release-owner-proof-input-record-template.json"
$markdownPath = Join-Path $artifactRoot "release-owner-proof-input-record-template.md"

Set-ContentWithRetry -LiteralPath $jsonPath -Value ($record | ConvertTo-Json -Depth 12)

$fieldLines = $requiredOwnerInputFields | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$markdown = @"
# Release Owner Proof Input Record Template

生成时间：$($record.generatedAtUtc)

## Summary

- record state: ``$($record.recordState)``
- proof classification: ``$($record.proofClassification)``
- performs publish: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- validation target: ``$($record.validationTarget)``

## Required Owner Input Fields

$($fieldLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Validator

````powershell
$($record.validatorCommand)
````

## Boundary

$($record.boundary)
"@

Set-ContentWithRetry -LiteralPath $markdownPath -Value $markdown

Write-Output "Release owner proof input record template written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "RecordState=$($record.recordState)"
Write-Output "RequiredOwnerInputFieldCount=$($record.requiredOwnerInputFieldCount)"
Write-Output "CanPublishPublicly=False"
