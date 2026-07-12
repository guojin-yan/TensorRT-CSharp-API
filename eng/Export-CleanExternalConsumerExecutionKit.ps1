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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$executionSteps = @(
  [pscustomobject]@{
    stepId = "01-create-clean-root-outside-repository"
    ownerAction = "Create a new consumer workspace outside the repository root."
    requiredEvidence = @("absolute clean external root", "directory creation timestamp", "repository path comparison")
  },
  [pscustomobject]@{
    stepId = "02-restore-from-public-source"
    ownerAction = "Restore the managed package from the selected public package source."
    requiredEvidence = @("public source URL", "published package version", "restore command", "restore stdout/stderr")
  },
  [pscustomobject]@{
    stepId = "03-build-clean-consumer"
    ownerAction = "Build the clean consumer project without repository references."
    requiredEvidence = @("build command", "build exitCode=0", "build stdout/stderr")
  },
  [pscustomobject]@{
    stepId = "04-run-runtime-smoke"
    ownerAction = "Run the runtime smoke path with the selected runtime package key."
    requiredEvidence = @("run command", "smoke command", "exitCode=0", "stdout/stderr", "smoke log path", "smoke log SHA256")
  },
  [pscustomobject]@{
    stepId = "05-capture-host-and-package-metadata"
    ownerAction = "Capture host, GPU, NVIDIA runtime, managed package, native bridge, and runtime package hashes."
    requiredEvidence = @("GPU name", "driver version", "CUDA version", "TensorRT version", "cuDNN version", ".NET SDK version", "managed/native/runtime SHA256")
  },
  [pscustomobject]@{
    stepId = "06-run-strict-validator"
    ownerAction = "Run strict package consumer proof validator against the completed owner input."
    requiredEvidence = @("validator command", "validator output", "validation JSON", "zero failed blocker count")
  }
)

$kit = [pscustomobject]@{
  recordKind = "clean-external-consumer-execution-kit"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  kitState = "blocked-owner-clean-external-consumer-runtime-evidence-required"
  cleanRootRequirement = "Clean external consumer root must be outside the repository."
  repositoryRoot = $RepositoryRoot
  disallowedSubstitutes = @(
    "project-reference substitute",
    "local package-feed substitute",
    "direct nupkg substitute",
    "template-only record",
    "dashboard-only record",
    "build-only report",
    "dry-run record"
  )
  publicPackageSourceRequired = $true
  requiredCommands = [pscustomobject]@{
    scaffold = "dotnet new console -n TensorRtSharpPublicConsumer"
    addPackage = "dotnet add package JYPPX.TensorRT.CSharp.API --version <published-version> --source <public-package-source>"
    restore = "dotnet restore"
    build = "dotnet build -c Release"
    smoke = "dotnet run -c Release -- --runtime-package-key win-x64-trt11.0-cuda13.2-cudnn9.22"
    validator = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  }
  requiredExitCode = 0
  requiredOutputEvidence = @(
    "restore stdout/stderr summary",
    "build stdout/stderr summary",
    "smoke stdout/stderr summary",
    "smoke log path",
    "smoke log SHA256",
    "native assets copied flag",
    "dependency probe status"
  )
  runtimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22"
  requiredPackageHashes = @(
    "managedPackageSha256",
    "nativeBridgeSha256",
    "runtimePackageSha256",
    "smokeLogSha256"
  )
  requiredHostMetadata = @(
    "machineName",
    "osVersion",
    "gpuName",
    "nvidiaDriverVersion",
    "cudaRuntimeVersion",
    "tensorRtVersion",
    "cudnnVersion",
    "dotnetSdkVersion"
  )
  executionSteps = @($executionSteps)
  ownerInputArtifact = "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json"
  expectedRecord = "artifacts/final-release/package-consumer-runtime-proof-record.json"
  validatorCommand = "eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof"
  performsPublish = $false
  performsRuntimeExecution = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  isPackageConsumerRuntimeProof = $false
  boundary = "This kit is an execution checklist and owner evidence contract. It does not perform publication or runtime execution and does not promote package consumer evidence."
}

$jsonPath = Join-Path $OutputRoot "clean-external-consumer-execution-kit.json"
$markdownPath = Join-Path $OutputRoot "clean-external-consumer-execution-kit.md"
$kit | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$stepRows = foreach ($step in $executionSteps) {
  $requiredEvidenceText = ($step.requiredEvidence) -join "; "
  "| ``$(ConvertTo-MarkdownCell $step.stepId)`` | $(ConvertTo-MarkdownCell $step.ownerAction) | $(ConvertTo-MarkdownCell $requiredEvidenceText) |"
}

$markdown = @"
# Clean External Consumer Execution Kit

Generated at: ``$($kit.generatedAtUtc)``

## Summary

- kitState: ``$($kit.kitState)``
- cleanRootRequirement: ``$($kit.cleanRootRequirement)``
- publicPackageSourceRequired: ``True``
- requiredExitCode: ``0``
- runtimePackageKey: ``$($kit.runtimePackageKey)``
- performsPublish: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- canPromotePackageConsumerRuntime: ``False``

## Steps

| Step | Owner Action | Required Evidence |
| --- | --- | --- |
$($stepRows -join "`r`n")

## Disallowed Substitutes

$(@($kit.disallowedSubstitutes | ForEach-Object { "- ``$_``" }) -join "`r`n")

## Boundary

$($kit.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Clean external consumer execution kit written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "KitState=$($kit.kitState) StepCount=$(@($executionSteps).Count)"
