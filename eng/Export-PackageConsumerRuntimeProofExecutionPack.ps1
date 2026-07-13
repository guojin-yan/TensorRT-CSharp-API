[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
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

$missingOwnerInputs = @(
  "clean external consumer root outside repository",
  "managed package source path or URL",
  "runtime package source path or URL",
  "managed nupkg SHA256",
  "runtime nupkg SHA256",
  "restore log path",
  "build log path",
  "runtime smoke log path",
  "runtime smoke log SHA256",
  "native asset listing path",
  "native asset listing SHA256",
  "stdout summary from runtime smoke log",
  "stderr summary from runtime smoke log",
  "runtime smoke exit code",
  "no ProjectReference confirmation",
  "no local package source confirmation",
  "no local nupkg package reference confirmation",
  "host OS",
  "host architecture",
  "GPU name",
  "NVIDIA driver version",
  "CUDA runtime version",
  "TensorRT runtime version",
  "cuDNN version when applicable",
  "owner review"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "package-consumer-runtime-proof-execution-pack"
  packState = "blocked-owner-action-required"
  runtimePackageKey = $RuntimePackageKey
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromotePackageConsumerRuntime = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  cleanConsumerRoot = "owner-action-required-outside-repository"
  repositoryRoot = $RepositoryRoot
  packageSourceKind = "owner-action-required-local-feed-or-real-channel"
  packageSourcePathOrUrl = "owner-action-required"
  managedNupkgPath = "owner-action-required"
  runtimeNupkgPath = "owner-action-required"
  managedNupkgSha256 = "owner-action-required"
  runtimeNupkgSha256 = "owner-action-required"
  restoreLogPath = "owner-action-required"
  buildLogPath = "owner-action-required"
  runtimeSmokeLogPath = "owner-action-required"
  runtimeSmokeLogSha256 = "owner-action-required"
  nativeAssetListingPath = "owner-action-required"
  nativeAssetListingSha256 = "owner-action-required"
  stdoutSummary = "owner-action-required"
  stderrSummary = "owner-action-required"
  runtimeSmokeExitCode = "owner-action-required"
  noProjectReferenceRequired = $true
  noProjectReference = "owner-action-required"
  noLocalPackageSource = "owner-action-required"
  noLocalNupkgPackageReference = "owner-action-required"
  hostMetadata = [ordered]@{
    hostOs = "owner-action-required"
    hostArchitecture = "owner-action-required"
    gpuName = "owner-action-required"
    nvidiaDriverVersion = "owner-action-required"
    cudaRuntimeVersion = "owner-action-required"
    tensorRtRuntimeVersion = "owner-action-required"
    cudnnVersion = "owner-action-required-when-applicable"
  }
  requiredCommands = @(
    "dotnet nuget locals all --clear",
    "dotnet new console -n TensorRtSharpPackageConsumerProof",
    "dotnet add .\TensorRtSharpPackageConsumerProof\TensorRtSharpPackageConsumerProof.csproj package JYPPX.TensorRT.CSharp.API --version <owner-version> --source <owner-package-source>",
    "dotnet restore .\TensorRtSharpPackageConsumerProof\TensorRtSharpPackageConsumerProof.csproj",
    "dotnet build .\TensorRtSharpPackageConsumerProof\TensorRtSharpPackageConsumerProof.csproj -c Release",
    "dotnet run --project .\TensorRtSharpPackageConsumerProof\TensorRtSharpPackageConsumerProof.csproj -c Release -- --runtime-package-key $RuntimePackageKey",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File <repo>\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath <filled-record> -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
  )
  expectedEvidenceRecordTemplate = "artifacts/final-release/external-runtime-proof-record-template.json"
  expectedValidatedEvidenceRecord = "artifacts/final-release/external-runtime-proof-record.json"
  validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofPack.ps1 -PackPath .\artifacts\final-release\package-consumer-runtime-proof-execution-pack.json"
  upstreamValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -InputPath .\artifacts\final-release\external-runtime-proof-record.json -RuntimePackageKey $RuntimePackageKey -RequireExistingLog -FailOnNotProof"
  ownerActionRequired = $true
  missingOwnerInputCount = $missingOwnerInputs.Count
  missingOwnerInputs = $missingOwnerInputs
  nonSubstituteProofKinds = @(
    "local feed",
    "ProjectReference",
    "dependency-probe-only",
    "build-only",
    "sidecar-only",
    "template",
    "runbook",
    "blocked-by-cuda-driver",
    "managed-readiness-only",
    "precheck-only",
    "dry-run-only",
    "schema-only",
    "CallbackAllocatorReadinessSnapshot"
  )
  proofBoundary = "This execution pack is owner guidance only. Local feeds, ProjectReference consumers, dependency probes, build-only reports, templates, runbooks, blocked-by-cuda-driver records, managed-readiness-only, precheck-only, dry-run-only, schema-only, and CallbackAllocatorReadinessSnapshot cannot promote package-consumer-runtime proof."
  promotionBlockers = @(
    "clean external consumer has not been provided",
    "real package source and nupkg hashes have not been provided",
    "restore/build/runtime smoke logs have not been provided",
    "native asset listing and SHA256 have not been provided",
    "runtime smoke stdout/stderr summaries and exit code have not been provided",
    "host CUDA/TensorRT/cuDNN/GPU metadata has not been provided",
    "no ProjectReference evidence has not been confirmed",
    "no local package source and no local nupkg package reference evidence has not been confirmed"
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-execution-pack.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$commandLines = $record.requiredCommands | ForEach-Object { "- ``$_``" }
$missingLines = $record.missingOwnerInputs | ForEach-Object { "- ``$_``" }
$blockerLines = $record.promotionBlockers | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $record.nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }

$markdown = @"
# Package Consumer Runtime Proof Execution Pack

生成时间：$($record.generatedAtUtc)

## Summary

- record kind: ``$($record.recordKind)``
- pack state: ``$($record.packState)``
- runtime package key: ``$($record.runtimePackageKey)``
- performs publish: ``False``
- performs runtime execution: ``False``
- can promote package-consumer-runtime: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- missing owner input count: ``$($record.missingOwnerInputCount)``

## Required Commands

$($commandLines -join "`r`n")

## Missing Owner Inputs

$($missingLines -join "`r`n")

## Promotion Blockers

$($blockerLines -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Boundary

$($record.proofBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Package consumer runtime proof execution pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackState=$($record.packState)"
Write-Output "MissingOwnerInputCount=$($record.missingOwnerInputCount)"
Write-Output "CanPublishPublicly=False"
