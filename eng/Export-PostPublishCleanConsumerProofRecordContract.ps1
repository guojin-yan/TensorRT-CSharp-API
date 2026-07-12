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

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function New-RequiredField {
  param(
    [string]$Name,
    [string]$Description,
    [string[]]$ForbiddenSubstitutes = @()
  )

  [pscustomobject]@{
    name = $Name
    description = $Description
    required = $true
    valueState = "owner-input-required"
    ready = $false
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
  }
}

$requiredFields = @(
  New-RequiredField -Name "cleanExternalConsumer.root" -Description "Repository-external clean consumer workspace root." -ForbiddenSubstitutes @("repo sample", "ProjectReference", "in-repository path")
  New-RequiredField -Name "cleanExternalConsumer.projectPath" -Description "Repository-external clean consumer project path." -ForbiddenSubstitutes @("repo sample", "ProjectReference", "in-repository path")
  New-RequiredField -Name "cleanExternalConsumer.restoreLogPath" -Description "Owner captured restore log path." -ForbiddenSubstitutes @("summary-only", "missing log")
  New-RequiredField -Name "cleanExternalConsumer.restoreLogSha256" -Description "SHA256 of clean consumer restore log." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "cleanExternalConsumer.buildLogPath" -Description "Owner captured build log path." -ForbiddenSubstitutes @("summary-only", "missing log")
  New-RequiredField -Name "cleanExternalConsumer.buildLogSha256" -Description "SHA256 of clean consumer build log." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "cleanExternalConsumer.smokeLogPath" -Description "Owner captured runtime smoke log path." -ForbiddenSubstitutes @("build-only", "summary-only", "missing log")
  New-RequiredField -Name "cleanExternalConsumer.smokeLogSha256" -Description "SHA256 of clean consumer runtime smoke log." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "cleanExternalConsumer.stdoutLogPath" -Description "Owner captured stdout log path." -ForbiddenSubstitutes @("summary-only", "missing log")
  New-RequiredField -Name "cleanExternalConsumer.stdoutLogSha256" -Description "SHA256 of stdout log." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "cleanExternalConsumer.stderrLogPath" -Description "Owner captured stderr log path, or explicit no-stderr-emitted token." -ForbiddenSubstitutes @("summary-only")
  New-RequiredField -Name "cleanExternalConsumer.stderrLogSha256" -Description "SHA256 of stderr log, or explicit no-stderr-emitted token." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "hostMetadata.osDescription" -Description "Clean consumer host OS description." -ForbiddenSubstitutes @("unknown host")
  New-RequiredField -Name "hostMetadata.architecture" -Description "Clean consumer host architecture." -ForbiddenSubstitutes @("unknown host")
  New-RequiredField -Name "hostMetadata.gpuName" -Description "Clean consumer GPU model." -ForbiddenSubstitutes @("dependency probe only")
  New-RequiredField -Name "hostMetadata.cudaDriverVersion" -Description "CUDA driver version on clean consumer host." -ForbiddenSubstitutes @("dependency probe only")
  New-RequiredField -Name "hostMetadata.cudaRuntimeVersion" -Description "CUDA runtime version used by clean consumer." -ForbiddenSubstitutes @("dependency probe only")
  New-RequiredField -Name "hostMetadata.cudnnVersion" -Description "cuDNN version used by clean consumer." -ForbiddenSubstitutes @("dependency probe only")
  New-RequiredField -Name "hostMetadata.tensorRtVersion" -Description "TensorRT version used by clean consumer." -ForbiddenSubstitutes @("dependency probe only")
  New-RequiredField -Name "hostMetadata.tensorRtLine" -Description "TensorRT major line used by clean consumer, such as TRT8/TRT10/TRT11." -ForbiddenSubstitutes @("dependency probe only")
  New-RequiredField -Name "ownerReview.reviewer" -Description "Owner reviewer for post-publish clean consumer proof." -ForbiddenSubstitutes @("template", "draft")
  New-RequiredField -Name "ownerReview.reviewedAtUtc" -Description "Owner review timestamp for post-publish clean consumer proof." -ForbiddenSubstitutes @("template", "draft")
  New-RequiredField -Name "ownerReview.approvalState" -Description "Owner approval/block decision for post-publish clean consumer proof." -ForbiddenSubstitutes @("template", "draft")
  New-RequiredField -Name "forbiddenSubstituteCounts.projectReferenceCount" -Description "ProjectReference count from clean consumer scan, expected 0." -ForbiddenSubstitutes @("ProjectReference")
  New-RequiredField -Name "forbiddenSubstituteCounts.localFeedReferenceCount" -Description "Local feed count from clean consumer scan, expected 0." -ForbiddenSubstitutes @("local feed")
  New-RequiredField -Name "forbiddenSubstituteCounts.directNupkgReferenceCount" -Description "Direct nupkg count from clean consumer scan, expected 0." -ForbiddenSubstitutes @("direct nupkg")
  New-RequiredField -Name "forbiddenSubstituteCounts.buildOnlyCount" -Description "Build-only evidence count, expected 0." -ForbiddenSubstitutes @("build-only")
  New-RequiredField -Name "forbiddenSubstituteCounts.dependencyProbeOnlyCount" -Description "Dependency-probe-only evidence count, expected 0." -ForbiddenSubstitutes @("dependency probe")
  New-RequiredField -Name "forbiddenSubstituteCounts.blockedByDriverOnlyCount" -Description "Blocked-by-driver-only evidence count, expected 0." -ForbiddenSubstitutes @("blocked-by-driver")
  New-RequiredField -Name "cleanConsumerProjectPath" -Description "Repository-external clean consumer project path or archive." -ForbiddenSubstitutes @("repo sample", "ProjectReference")
  New-RequiredField -Name "packageSourceUrl" -Description "Public package source used by clean consumer restore." -ForbiddenSubstitutes @("local feed")
  New-RequiredField -Name "restoredPackageId" -Description "Package id resolved by clean consumer restore." -ForbiddenSubstitutes @("ProjectReference")
  New-RequiredField -Name "restoredPackageVersion" -Description "Package version resolved by clean consumer restore." -ForbiddenSubstitutes @("local build version")
  New-RequiredField -Name "restoredPackageSha256" -Description "SHA256 of restored public package." -ForbiddenSubstitutes @("direct nupkg")
  New-RequiredField -Name "restoreCommand" -Description "Owner captured restore command." -ForbiddenSubstitutes @("runbook")
  New-RequiredField -Name "buildCommand" -Description "Owner captured build command." -ForbiddenSubstitutes @("dry-run")
  New-RequiredField -Name "smokeCommand" -Description "Owner captured smoke command with runtime package key." -ForbiddenSubstitutes @("build-only")
  New-RequiredField -Name "smokeExitCode" -Description "Exit code of clean consumer runtime smoke." -ForbiddenSubstitutes @("skipped")
  New-RequiredField -Name "stdoutPath" -Description "Path to smoke stdout log." -ForbiddenSubstitutes @("summary-only")
  New-RequiredField -Name "stdoutSha256" -Description "SHA256 of smoke stdout log." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "stderrPath" -Description "Path to smoke stderr log." -ForbiddenSubstitutes @("summary-only")
  New-RequiredField -Name "stderrSha256" -Description "SHA256 of smoke stderr log." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "hostOs" -Description "Clean consumer host operating system." -ForbiddenSubstitutes @("unknown host")
  New-RequiredField -Name "cudaDriverVersion" -Description "CUDA driver version on clean consumer host." -ForbiddenSubstitutes @("dependency probe only")
  New-RequiredField -Name "runtimePackageKey" -Description "Runtime package key used by smoke command." -ForbiddenSubstitutes @("placeholder key")
  New-RequiredField -Name "forbiddenSubstituteScan" -Description "Explicit no ProjectReference/local feed/direct nupkg scan result." -ForbiddenSubstitutes @("local feed", "ProjectReference", "direct nupkg")
)

$record = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-proof-record-contract"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  contractState = "blocked-post-publish-clean-consumer-proof-record-required"
  requiredFieldCount = $requiredFields.Count
  blockedRequiredFieldCount = $requiredFields.Count
  readyRequiredFieldCount = 0
  requiredFields = @($requiredFields)
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json",
    "artifacts/final-release/strict-close-ready-convergence-dashboard-validation.json"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "This contract waits for a real repository-external clean consumer proof record. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-record-contract.json"
$markdownPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-record-contract.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Post-Publish Clean Consumer Proof Record Contract",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| contractState | ``$($record.contractState)`` |",
  "| requiredFieldCount | ``$($record.requiredFieldCount)`` |",
  "| blockedRequiredFieldCount | ``$($record.blockedRequiredFieldCount)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Required Fields",
  "",
  "| Field | State | Description |",
  "| --- | --- | --- |"
)

foreach ($field in $requiredFields) {
  $markdown += "| $($field.name) | ``$($field.valueState)`` | $($field.description) |"
}

$markdown += @("", "## Boundary", "", $record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish clean consumer proof record contract written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ContractState=$($record.contractState) RequiredFields=$($record.requiredFieldCount) Blocked=$($record.blockedRequiredFieldCount)"
