[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputPath = "artifacts\final-release\release-proof-owner-backfill-summary-validation.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function New-LaneSummary {
  param(
    [string]$Id,
    [string]$ExpectedRecordPath,
    [string]$ValidatorOutputPath,
    [string]$StrictValidatorCommand,
    [string[]]$RequiredLogs,
    [string[]]$RequiredHashFields,
    [string[]]$RequiredHostMetadata
  )

  $recordFullPath = Resolve-RepoPath -Path $ExpectedRecordPath
  $validatorFullPath = Resolve-RepoPath -Path $ValidatorOutputPath

  $missing = @()
  if (-not (Test-Path -LiteralPath $recordFullPath -PathType Leaf)) {
    $missing += "expected record missing: $ExpectedRecordPath"
  }
  if (-not (Test-Path -LiteralPath $validatorFullPath -PathType Leaf)) {
    $missing += "validator output missing: $ValidatorOutputPath"
  }
  foreach ($log in $RequiredLogs) {
    if (-not (Test-Path -LiteralPath (Resolve-RepoPath -Path $log) -PathType Leaf)) {
      $missing += "required log missing: $log"
    }
  }

  [pscustomobject]@{
    id = $Id
    expectedRecordPath = $ExpectedRecordPath
    recordExists = [bool](Test-Path -LiteralPath $recordFullPath -PathType Leaf)
    validatorOutputArchivePath = $ValidatorOutputPath
    validatorOutputExists = [bool](Test-Path -LiteralPath $validatorFullPath -PathType Leaf)
    requiredLogFiles = @($RequiredLogs)
    requiredHashFields = @($RequiredHashFields)
    requiredHostMetadata = @($RequiredHostMetadata)
    strictValidatorCommand = $StrictValidatorCommand
    missingInputs = @($missing)
    currentValidationState = "blocked-owner-action-required"
    canPromote = $false
  }
}

$realModelRuntimeLane = New-LaneSummary `
  -Id "real-model-runtime" `
  -ExpectedRecordPath "artifacts/final-release/real-case-evidence-record.json" `
  -ValidatorOutputPath "artifacts/final-release/owner-evidence/real-model-runtime/validator-output.log" `
  -StrictValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealCaseEvidenceRecord.ps1 -RecordPath .\artifacts\final-release\real-case-evidence-record.json -FailOnNotProof" `
  -RequiredLogs @(
    "artifacts/final-release/owner-evidence/real-model-runtime/det/stdout.log",
    "artifacts/final-release/owner-evidence/real-model-runtime/cls/stdout.log",
    "artifacts/final-release/owner-evidence/real-model-runtime/seg/stdout.log",
    "artifacts/final-release/owner-evidence/real-model-runtime/obb/stdout.log",
    "artifacts/final-release/owner-evidence/real-model-runtime/pose/stdout.log",
    "artifacts/final-release/owner-evidence/real-model-runtime/sem/stdout.log"
  ) `
  -RequiredHashFields @("onnxSha256", "engineSha256", "inputArtifactSha256", "outputArtifactSha256", "stdoutLogSha256", "stderrLogSha256") `
  -RequiredHostMetadata @("hostOs", "gpuName", "nvidiaDriverVersion", "cudaVersion", "tensorRtVersion")

$packageConsumerRuntimeLane = New-LaneSummary `
  -Id "package-consumer-runtime" `
  -ExpectedRecordPath "artifacts/final-release/package-consumer-runtime-proof-record.json" `
  -ValidatorOutputPath "artifacts/final-release/owner-evidence/package-consumer-runtime/validator-output.log" `
  -StrictValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -InputPath .\artifacts\final-release\package-consumer-runtime-proof-record.json -Strict -RequireExistingLog -FailOnNotProof" `
  -RequiredLogs @(
    "artifacts/final-release/owner-evidence/package-consumer-runtime/package-consumer-restore.log",
    "artifacts/final-release/owner-evidence/package-consumer-runtime/package-consumer-build.log",
    "artifacts/final-release/owner-evidence/package-consumer-runtime/package-consumer-smoke.stdout.log",
    "artifacts/final-release/owner-evidence/package-consumer-runtime/package-consumer-smoke.stderr.log"
  ) `
  -RequiredHashFields @("managedNupkgSha256", "runtimeNupkgSha256", "smokeLogSha256") `
  -RequiredHostMetadata @("ownerName", "machineName", "gpuName", "cudaRuntimeVersion", "tensorRtVersion")

$postPublishVerificationLane = New-LaneSummary `
  -Id "post-publish-verification" `
  -ExpectedRecordPath "artifacts/final-release/post-publish-verification-record.json" `
  -ValidatorOutputPath "artifacts/final-release/owner-evidence/post-publish-verification/validator-output.log" `
  -StrictValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath .\artifacts\final-release\post-publish-verification-record.json -Strict -RequireExistingLog -FailOnNotProof" `
  -RequiredLogs @(
    "artifacts/final-release/owner-evidence/post-publish-verification/post-publish-restore.log",
    "artifacts/final-release/owner-evidence/post-publish-verification/post-publish-runtime-smoke.log"
  ) `
  -RequiredHashFields @("managedNupkgSha256", "runtimeNupkgSha256", "smokeLogSha256") `
  -RequiredHostMetadata @("ownerName", "machineName", "gpuName", "tensorRtRuntimeVersion")

$releaseIssueCloseLane = New-LaneSummary `
  -Id "release-issue-close" `
  -ExpectedRecordPath "artifacts/final-release/release-issue-close-record.json" `
  -ValidatorOutputPath "artifacts/final-release/owner-evidence/release-issue-close/release-close-validator-output.log" `
  -StrictValidatorCommand "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath .\artifacts\final-release\release-issue-close-record.json -FailOnNotCloseReady" `
  -RequiredLogs @(
    "artifacts/final-release/owner-evidence/release-issue-close/real-model-validator-output.log",
    "artifacts/final-release/owner-evidence/release-issue-close/package-consumer-validator-output.log",
    "artifacts/final-release/owner-evidence/release-issue-close/post-publish-validator-output.log"
  ) `
  -RequiredHashFields @("evidenceBundleSha256", "managedPackageSha256", "runtimePackageSha256") `
  -RequiredHostMetadata @("real-model-runtime host metadata", "package-consumer-runtime host metadata", "post-publish-verification host metadata")

$lanes = @(
  $realModelRuntimeLane,
  $packageConsumerRuntimeLane,
  $postPublishVerificationLane,
  $releaseIssueCloseLane
)

$summary = [pscustomobject]@{
  recordKind = "release-proof-owner-backfill-summary-validation"
  generatedAtLocal = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
  validationState = "blocked-owner-action-required"
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  performsPublish = $false
  approvesPublicRelease = $false
  proofBoundary = "Summary runner is read-only. It does not create proof, generate hashes, mark lanes passed, publish packages, or close release issues."
  lanes = $lanes
}

$outputFullPath = Resolve-RepoPath -Path $OutputPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputFullPath) | Out-Null
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $outputFullPath -Encoding utf8
$summary | ConvertTo-Json -Depth 12
