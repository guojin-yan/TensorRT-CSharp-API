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

function New-RunbookStep {
  param(
    [int]$Order,
    [string]$Id,
    [string]$Title,
    [string[]]$Commands,
    [string[]]$RequiredEvidence,
    [string]$Boundary
  )

  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    commands = @($Commands)
    requiredEvidence = @($RequiredEvidence)
    ownerActionRequired = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    boundary = $Boundary
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$forbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "candidate",
  "draft",
  "dashboard",
  "dry-run",
  "dependency-probe-only",
  "blocked-by-cuda-driver",
  "build-only",
  "template-only"
)

$steps = @(
  New-RunbookStep -Order 1 -Id "create-repository-external-workspace" -Title "Create repository-external clean consumer workspace" -Commands @(
    'New-Item -ItemType Directory -Force C:\trtsharp-clean-consumer | Out-Null',
    'Set-Location C:\trtsharp-clean-consumer',
    'dotnet new console --framework net8.0'
  ) -RequiredEvidence @(
    "absolute workspace path outside TensorRtSharp4.0",
    "generated .csproj path",
    "confirmation that no ProjectReference points back to the repository"
  ) -Boundary "Workspace creation is setup only; it is not runtime proof."
  New-RunbookStep -Order 2 -Id "configure-public-or-owner-approved-source" -Title "Configure owner-approved package source" -Commands @(
    'dotnet nuget add source <public-or-owner-approved-package-source-url> --name trtsharp-proof-source',
    'dotnet nuget list source'
  ) -RequiredEvidence @(
    "package source URL",
    "package source name",
    "owner confirmation that local feed is not used for post-publish proof"
  ) -Boundary "Source configuration is owner input only; local feed cannot substitute post-publish proof."
  New-RunbookStep -Order 3 -Id "install-managed-and-runtime-packages" -Title "Install managed API and runtime packages" -Commands @(
    'dotnet add package JYPPX.TensorRtSharp --version <managed-version> --source <package-source-url>',
    'dotnet add package JYPPX.CudaSharp --version <managed-version> --source <package-source-url>',
    'dotnet add package JYPPX.TensorRtSharp.Native.<runtime-key> --version <runtime-version> --source <package-source-url>'
  ) -RequiredEvidence @(
    "managed package id/version",
    "runtime package id/version/runtime key",
    "resolved package source",
    "nupkg path and SHA256 for each consumed package"
  ) -Boundary "Package install metadata is required, but restore/install alone is not package-consumer-runtime proof."
  New-RunbookStep -Order 4 -Id "restore-build-and-capture-logs" -Title "Restore and build the clean consumer" -Commands @(
    'dotnet restore --no-cache --force-evaluate *> package-consumer-restore.log',
    'dotnet build -c Release --no-restore *> package-consumer-build.log'
  ) -RequiredEvidence @(
    "restore log path and SHA256",
    "build log path and SHA256",
    "restore/build exit code 0"
  ) -Boundary "Restore/build are build-only evidence and cannot promote runtime proof."
  New-RunbookStep -Order 5 -Id "run-runtime-smoke-and-capture-streams" -Title "Run runtime smoke and capture stdout/stderr" -Commands @(
    'dotnet run -c Release --no-build -- --runtime-smoke *> package-consumer-smoke.stdout.log 2> package-consumer-smoke.stderr.log',
    'Get-Content package-consumer-smoke.stdout.log,package-consumer-smoke.stderr.log | Set-Content package-consumer-merged-transcript.log'
  ) -RequiredEvidence @(
    "stdoutPath",
    "stderrPath",
    "mergedTranscriptPath",
    "runtime smoke exitCode=0",
    "passed=true only after real runtime command succeeds"
  ) -Boundary "Runtime smoke contributes to proof only after owner input import and strict validators accept existing logs and hashes."
  New-RunbookStep -Order 6 -Id "hash-packages-logs-and-validator-output" -Title "Hash packages, logs, and validator output" -Commands @(
    'Get-FileHash -Algorithm SHA256 <managed-nupkg-path>,<runtime-nupkg-path>',
    'Get-FileHash -Algorithm SHA256 package-consumer-restore.log,package-consumer-build.log,package-consumer-smoke.stdout.log,package-consumer-smoke.stderr.log,package-consumer-merged-transcript.log',
    'Get-FileHash -Algorithm SHA256 <validator-output-path>'
  ) -RequiredEvidence @(
    "packageIdentity.nupkgSha256",
    "stdoutSha256",
    "stderrSha256",
    "mergedTranscriptSha256",
    "validatorOutputSha256"
  ) -Boundary "Hashes authenticate artifacts but cannot prove execution by themselves."
  New-RunbookStep -Order 7 -Id "capture-host-and-owner-review" -Title "Capture host metadata and owner review" -Commands @(
    'dotnet --info',
    'nvidia-smi',
    'Write-Output "<ownerReviewer> <ownerReviewTimestampUtc> <runtimePackageKey>"'
  ) -RequiredEvidence @(
    "hostMetadata.os/arch/gpu",
    "driverVersion",
    "cudaVersion",
    "tensorRtVersion",
    "dotnetVersion",
    "ownerReviewer",
    "ownerReviewTimestampUtc"
  ) -Boundary "Host metadata and owner review are traceability fields, not proof by themselves."
  New-RunbookStep -Order 8 -Id "fill-owner-external-result-input" -Title "Fill owner external result input" -Commands @(
    'Copy-Item artifacts/final-release/owner-external-proof-execution-result.input.template.json artifacts/final-release/owner-external-proof-execution-result.input.json',
    'Fill the matching resultInputs[] item with paths, hashes, exitCode=0, passed=true, owner review, and nonSubstituteConfirmations'
  ) -RequiredEvidence @(
    "owner-external-proof-execution-result.input.json",
    "resultInputs[].resultInputId matched to template",
    "10+ nonSubstituteConfirmations"
  ) -Boundary "Filled owner input is strict-validator input only until import and validators pass."
  New-RunbookStep -Order 9 -Id "import-and-run-strict-validator-chain" -Title "Import owner result and run strict validator chain" -Commands @(
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealExternalProofRecordImportValidator.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealExternalProofRecordImportValidator.ps1 -Strict',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordCandidateFromOwnerResultImport.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict'
  ) -RequiredEvidence @(
    "owner external proof execution result import validation",
    "real external proof record import validator",
    "candidate bridge output for strict validator input"
  ) -Boundary "Import and candidate bridge remain non-proof until strict real proof validators promote concrete records."
)

$record = [pscustomobject]@{
  recordKind = "clean-external-package-consumer-owner-runbook"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runbookState = "blocked-owner-clean-external-package-consumer-execution-required"
  runbookPurpose = "owner-executable clean external package consumer proof collection"
  stepCount = $steps.Count
  blockedStepCount = $steps.Count
  steps = @($steps)
  requiredInputTarget = "artifacts/final-release/owner-external-proof-execution-result.input.json"
  fillableTemplate = "artifacts/final-release/owner-external-proof-execution-result.input.template.json"
  forbiddenSubstitutes = $forbiddenSubstitutes
  requiredResultInputFields = @(
    "resultInputId",
    "packageIdentity.nupkgPath",
    "packageIdentity.nupkgSha256",
    "stdoutPath",
    "stdoutSha256",
    "stderrPath",
    "stderrSha256",
    "mergedTranscriptPath",
    "mergedTranscriptSha256",
    "validatorOutputPath",
    "validatorOutputSha256",
    "exitCode",
    "passed",
    "ownerReviewer",
    "ownerReviewTimestampUtc",
    "nonSubstituteConfirmations"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This runbook is owner execution guidance only. It is not runtime proof, post-publish proof, publish approval, package push, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "clean-external-package-consumer-owner-runbook.json"
$markdownPath = Join-Path $OutputRoot "clean-external-package-consumer-owner-runbook.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$stepRows = foreach ($step in $steps) {
  "| ``$(ConvertTo-MarkdownCell $step.id)`` | ``$($step.order)`` | $(ConvertTo-MarkdownCell $step.title) | $(ConvertTo-MarkdownCell $step.boundary) |"
}
$commandLines = foreach ($step in $steps) {
  "### $($step.order). $($step.title)"
  ""
  foreach ($command in $step.commands) { "- ``$command``" }
  ""
}

$markdown = @"
# Clean External Package Consumer Owner Runbook

该 runbook 是 Owner 在仓库外 clean consumer 中采集 package-consumer-runtime proof 输入的执行清单。它不会发布包，不会晋级 proof，也不会关闭 release issue。

| Field | Value |
|---|---|
| runbookState | ``$($record.runbookState)`` |
| stepCount | ``$($record.stepCount)`` |
| requiredInputTarget | ``$($record.requiredInputTarget)`` |
| fillableTemplate | ``$($record.fillableTemplate)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Steps

| Step | Order | Title | Boundary |
|---|---:|---|---|
$($stepRows -join "`r`n")

## Commands

$($commandLines -join "`r`n")

## Boundary

$($record.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean external package consumer owner runbook written to $jsonPath"
Write-Host "RunbookState=$($record.runbookState) Steps=$($record.stepCount) PerformsPublish=False CanPromoteRuntimeProof=False CanCloseReleaseIssue=False"
