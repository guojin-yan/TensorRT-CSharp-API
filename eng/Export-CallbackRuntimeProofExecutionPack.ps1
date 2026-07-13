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
  "compatible CUDA/TensorRT host",
  "runtime package key",
  "callback scenario name",
  "real package consumer root outside repository",
  "runtime smoke command",
  "runtime smoke log path",
  "runtime smoke log SHA256",
  "callback invocation marker",
  "InvocationCount greater than zero",
  "IsRealCallbackRuntimeProof=true evidence from validator",
  "stdout summary from runtime smoke log",
  "stderr summary from runtime smoke log",
  "host OS/GPU/driver/CUDA/TensorRT metadata",
  "owner review"
)

$nonSubstitutes = @(
  "managed-readiness",
  "managed-readiness-only",
  "callback-allocator-readiness-snapshot",
  "CallbackAllocatorReadinessSnapshot",
  "TensorRtCallbackAllocatorReadinessSnapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only",
  "blocked-by-cuda-driver",
  "dependency-probe-only",
  "build-only",
  "template",
  "runbook"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "callback-runtime-proof-execution-pack"
  packState = "blocked-owner-action-required"
  runtimePackageKey = $RuntimePackageKey
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRealCallbackRuntime = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  realCallbackRuntimeProof = $false
  isRealCallbackRuntimeProof = $false
  runtimeEvidenceKind = "owner-action-required"
  callbackScenario = "owner-action-required"
  cleanConsumerRoot = "owner-action-required-outside-repository"
  runtimeSmokeCommand = "owner-action-required"
  runtimeSmokeLogPath = "owner-action-required"
  runtimeSmokeLogSha256 = "owner-action-required"
  invocationMarker = "owner-action-required"
  invocationCount = "owner-action-required-greater-than-zero"
  stdoutSummary = "owner-action-required"
  stderrSummary = "owner-action-required"
  hostMetadata = [ordered]@{
    hostOs = "owner-action-required"
    gpuName = "owner-action-required"
    nvidiaDriverVersion = "owner-action-required"
    cudaRuntimeVersion = "owner-action-required"
    tensorRtRuntimeVersion = "owner-action-required"
  }
  expectedEvidenceSchema = "docs/articles/zh-cn/real-callback-runtime-evidence-schema.md"
  expectedReadinessSnapshot = "CallbackAllocatorReadinessSnapshot=managed-readiness"
  validatorCommand = "owner must validate real callback runtime smoke with IsRealCallbackRuntimeProof=true and InvocationCount>0"
  ownerActionRequired = $true
  missingOwnerInputCount = $missingOwnerInputs.Count
  missingOwnerInputs = $missingOwnerInputs
  nonSubstituteProofKinds = $nonSubstitutes
  proofBoundary = "This execution pack is owner guidance only. TensorRtCallbackAllocatorReadinessSnapshot, RuntimeEvidenceKind=managed-readiness, precheck-only, dry-run-only, schema-only, dependency-probe-only, build-only, template, runbook, and blocked-by-cuda-driver cannot promote real callback runtime proof."
  promotionBlockers = @(
    "compatible host runtime smoke has not been provided",
    "callback invocation marker has not been provided",
    "InvocationCount>0 has not been proven",
    "runtime smoke log path and SHA256 have not been provided",
    "IsRealCallbackRuntimeProof=true validator evidence has not been provided"
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "callback-runtime-proof-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "callback-runtime-proof-execution-pack.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$missingLines = $record.missingOwnerInputs | ForEach-Object { "- ``$_``" }
$nonSubstituteLines = $record.nonSubstituteProofKinds | ForEach-Object { "- ``$_``" }
$blockerLines = $record.promotionBlockers | ForEach-Object { "- ``$_``" }

$markdown = @"
# Callback Runtime Proof Execution Pack

生成时间：$($record.generatedAtUtc)

## Summary

- record kind: ``$($record.recordKind)``
- pack state: ``$($record.packState)``
- runtime package key: ``$($record.runtimePackageKey)``
- performs publish: ``False``
- performs runtime execution: ``False``
- can promote real callback runtime: ``False``
- is real callback runtime proof: ``False``
- missing owner input count: ``$($record.missingOwnerInputCount)``

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

Write-Output "Callback runtime proof execution pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackState=$($record.packState)"
Write-Output "MissingOwnerInputCount=$($record.missingOwnerInputCount)"
Write-Output "CanPromoteRealCallbackRuntime=False"
