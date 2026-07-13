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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Step {
  param([int]$Order, [string]$Id, [string]$Title, [string]$Command, [string]$Evidence, [string]$Boundary)

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    command = $Command
    expectedEvidence = $Evidence
    ownerActionRequired = $true
    performsPublish = $false
    performsRuntimeExecutionInAutomation = $false
    canPromoteRuntimeProof = $false
    boundary = $Boundary
  }
}

$steps = @(
  New-Step 1 "create-external-root" "Create repository-external workspace" 'New-Item -ItemType Directory -Force -Path "<owner-external-root>\TensorRtSharpCleanConsumer"; Set-Location "<owner-external-root>\TensorRtSharpCleanConsumer"' "external workspace absolute path" "Must be outside the TensorRtSharp repository."
  New-Step 2 "create-clean-consumer-project" "Create clean console project" 'dotnet new console -n TensorRtSharpCleanConsumer --framework net8.0' "cleanConsumerCsprojPath and csproj SHA256" "No ProjectReference is allowed."
  New-Step 3 "add-managed-package" "Add managed package from real source" 'dotnet add .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj package JYPPX.TensorRtSharp --version <owner-package-version> --source <owner-real-package-source-url>' "package source URL and managed package identity" "Local feed and direct .nupkg are forbidden substitutes."
  New-Step 4 "add-runtime-package" "Add runtime package from real source" 'dotnet add .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj package JYPPX.TensorRtSharp.Native.<runtime-key> --version <owner-package-version> --source <owner-real-package-source-url>' "runtime package identity and runtime key" "Local feed and direct .nupkg are forbidden substitutes."
  New-Step 5 "restore-with-log" "Restore and capture log" 'dotnet restore .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj *> .\restore.log' "restoreLogPath and restoreLogSha256" "Restore must come from a real package source."
  New-Step 6 "build-with-log" "Build and capture log" 'dotnet build .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj -c Release --no-restore *> .\build.log' "buildLogPath and buildLogSha256" "Build-only output cannot promote runtime proof."
  New-Step 7 "run-smoke-with-stdout-stderr" "Run smoke and capture stdout/stderr" 'dotnet run --project .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj -c Release --no-build 1> .\smoke.stdout.log 2> .\smoke.stderr.log; $LASTEXITCODE | Set-Content .\smoke.exitcode.txt' "runLogPath, smoke stdout/stderr logs, and exitCode" "Runtime proof requires exitCode=0 plus real stdout/stderr and log hashes."
  New-Step 8 "collect-native-assets" "List restored native assets" 'Get-ChildItem -Recurse -File .\TensorRtSharpCleanConsumer\bin\Release | Where-Object { $_.Name -match "jyppx|tensorrt|cuda|cudnn|\.dll$|\.so$" } | Select-Object FullName,Length | ConvertTo-Json -Depth 4 | Set-Content .\native-assets.json' "nativeAssetListingPath and nativeAssetListingSha256" "Native asset listing alone is not runtime proof."
  New-Step 9 "compute-sha256" "Compute SHA256 for logs and packages" 'Get-ChildItem .\restore.log,.\build.log,.\smoke.stdout.log,.\smoke.stderr.log,.\native-assets.json -File | Get-FileHash -Algorithm SHA256 | ConvertTo-Json -Depth 4 | Set-Content .\sha256-manifest.json' "SHA256 manifest for every evidence file" "Hash manifest must match existing files in strict validation."
  New-Step 10 "capture-host-metadata" "Capture host metadata" '[pscustomobject]@{ os=[Environment]::OSVersion.VersionString; arch=[Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString(); machineName=$env:COMPUTERNAME; dotnetSdk=(dotnet --version); gpuName="<owner-gpu-name>"; nvidiaDriver="<owner-nvidia-driver>"; cudaRuntimeToolkit="<owner-cuda-runtime-toolkit>"; tensorrt="<owner-tensorrt-version>"; cudnn="<owner-cudnn-version>" } | ConvertTo-Json | Set-Content .\host-metadata.json' "hostMetadataPath and hostMetadataSha256" "Host metadata without runtime smoke cannot promote proof."
  New-Step 11 "fill-owner-input" "Fill import JSON" 'Copy-Item "<repo>\artifacts\final-release\external-clean-consumer-execution-result.template.json" .\external-clean-consumer-execution-result.owner.json; <owner-fill-all-paths-hashes-and-confirmations>' "owner input JSON with no placeholders" "Template remains non-proof until filled with real evidence."
  New-Step 12 "run-strict-import" "Run strict repository validator" 'pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Import-ExternalCleanConsumerExecutionResult.ps1" -OwnerInputPath .\external-clean-consumer-execution-result.owner.json -RequireExistingFiles -RequireHashMatch; pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Test-ExternalCleanConsumerExecutionResult.ps1" -Strict -RequireExistingFiles -RequireHashMatch -FailOnNotProof' "strict validator JSON/MD" "Strict validation can create a proof candidate only from real evidence; this command pack itself remains non-proof."
)

$forbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "direct nupkg",
  "pre-publish smoke reused as post-publish proof",
  "local smoke",
  "build-only",
  "dependency-probe",
  "dashboard",
  "runbook",
  "template",
  "candidate",
  "command pack"
)

$ownerFields = @(
  "repositoryExternalWorkspaceRoot",
  "cleanConsumerCsprojPath",
  "packageSourceUrl",
  "managedPackageId",
  "managedPackageVersion",
  "managedPackageSha256",
  "runtimePackageId",
  "runtimePackageVersion",
  "runtimePackageKey",
  "runtimePackageSha256",
  "restoreLogPath",
  "restoreLogSha256",
  "buildLogPath",
  "buildLogSha256",
  "runLogPath",
  "runLogSha256",
  "smokeStdoutPath",
  "smokeStdoutSha256",
  "smokeStderrPath",
  "smokeStderrSha256",
  "nativeAssetListingPath",
  "nativeAssetListingSha256",
  "hostMetadata"
)

$record = [pscustomobject]@{
  recordKind = "external-clean-consumer-owner-command-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packState = "blocked-external-clean-consumer-owner-command-pack-required"
  stepCount = $steps.Count
  ownerFieldCount = $ownerFields.Count
  forbiddenSubstituteCount = $forbiddenSubstitutes.Count
  steps = @($steps)
  ownerFields = @($ownerFields)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner-copyable command guidance only. It does not execute runtime smoke in automation, does not publish packages, is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "external-clean-consumer-owner-command-pack.json"
$markdownPath = Join-Path $OutputRoot "external-clean-consumer-owner-command-pack.md"
$record | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($step in $steps) {
  "| ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.title) | ``$(ConvertTo-MarkdownCell $step.command)`` | $(ConvertTo-MarkdownCell $step.expectedEvidence) |"
}

$markdown = @"
# External CleanConsumer Owner Command Pack

| Field | Value |
|---|---|
| packState | ``$($record.packState)`` |
| stepCount | ``$($record.stepCount)`` |
| ownerActionRequired | ``$($record.ownerActionRequired)`` |
| passed | ``$($record.passed)`` |
| performsRuntimeExecution | ``$($record.performsRuntimeExecution)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |

## Owner Commands

| ID | Title | Command | Expected Evidence |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($record.boundary)
"@

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
