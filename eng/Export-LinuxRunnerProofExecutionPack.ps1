[CmdletBinding()]
param(
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

$missingOwnerInputs = @(
  "real Linux x64 host",
  "target distro",
  "kernel version",
  "GPU name",
  "NVIDIA driver version",
  "CUDA runtime version",
  "TensorRT runtime version",
  "cuDNN version when applicable",
  "runtime package SHA256",
  "Linux runner command",
  "Linux runner log path",
  "Linux runner log SHA256",
  "stdout summary",
  "stderr summary",
  "owner review"
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "linux-runner-proof-execution-pack"
  packState = "blocked-owner-action-required"
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  targetDistro = "Ubuntu 22.04 or owner-documented compatible Linux"
  targetArch = "linux-x64"
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteLinuxRunnerProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  runtimePackageSha256 = "owner-action-required"
  linuxRunnerCommand = "owner-action-required"
  linuxRunnerLogPath = "owner-action-required"
  linuxRunnerLogSha256 = "owner-action-required"
  hostOs = "owner-action-required"
  kernelVersion = "owner-action-required"
  gpuName = "owner-action-required"
  nvidiaDriverVersion = "owner-action-required"
  cudaVersion = "owner-action-required"
  tensorRtVersion = "owner-action-required"
  cudnnVersion = "owner-action-required-if-applicable"
  requiredCommands = @(
    "uname -a",
    "nvidia-smi",
    "dotnet --info",
    "sha256sum <runtime-nupkg>",
    "dotnet run --project <linux-runner-project-or-clean-consumer> -- --runtime-package-key $LinuxRuntimePackageKey",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File <repo>/eng/Test-LinuxRunnerEvidenceRecord.ps1 -InputPath <filled-record>"
  )
  expectedEvidenceRecordTemplate = "artifacts/final-release/linux-runner-evidence-record.template.json"
  expectedValidatedEvidenceRecord = "artifacts/final-release/linux-runner-evidence-record.json"
  validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerProofPack.ps1 -PackPath .\artifacts\final-release\linux-runner-proof-execution-pack.json"
  upstreamValidatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -InputPath .\artifacts\final-release\linux-runner-evidence-record.json"
  ownerActionRequired = $true
  missingOwnerInputCount = $missingOwnerInputs.Count
  missingOwnerInputs = $missingOwnerInputs
  proofBoundary = "This execution pack is owner guidance only. Windows dry-runs, WSL without real GPU/CUDA/TensorRT validation, templates, runbooks, and blocked-by-cuda-driver records cannot promote Linux runner proof."
  promotionBlockers = @(
    "real Linux host evidence has not been provided",
    "runtime package SHA256 has not been provided",
    "Linux runner log has not been provided",
    "CUDA/TensorRT/cuDNN/GPU/driver metadata has not been provided",
    "owner review has not been provided"
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "linux-runner-proof-execution-pack.json"
$markdownPath = Join-Path $artifactRoot "linux-runner-proof-execution-pack.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$commandLines = $record.requiredCommands | ForEach-Object { "- ``$_``" }
$missingLines = $record.missingOwnerInputs | ForEach-Object { "- ``$_``" }
$blockerLines = $record.promotionBlockers | ForEach-Object { "- ``$_``" }

$markdown = @"
# Linux Runner Proof Execution Pack

生成时间：$($record.generatedAtUtc)

## Summary

- record kind: ``$($record.recordKind)``
- pack state: ``$($record.packState)``
- Linux runtime package key: ``$($record.linuxRuntimePackageKey)``
- target distro: ``$($record.targetDistro)``
- target arch: ``$($record.targetArch)``
- performs publish: ``False``
- performs runtime execution: ``False``
- can promote Linux runner proof: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- missing owner input count: ``$($record.missingOwnerInputCount)``

## Required Commands

$($commandLines -join "`r`n")

## Missing Owner Inputs

$($missingLines -join "`r`n")

## Promotion Blockers

$($blockerLines -join "`r`n")

## Boundary

$($record.proofBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Linux runner proof execution pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PackState=$($record.packState)"
Write-Output "MissingOwnerInputCount=$($record.missingOwnerInputCount)"
Write-Output "CanPublishPublicly=False"
