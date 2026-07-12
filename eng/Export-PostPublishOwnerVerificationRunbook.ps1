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

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function New-RunbookStep {
  param([int]$Order, [string]$Id, [string]$Title, [string[]]$Commands, [string[]]$RequiredEvidence, [string]$Boundary)
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
  "pre-publish package-consumer proof",
  "candidate",
  "draft",
  "dashboard",
  "dry-run",
  "blocked-by-cuda-driver",
  "build-only",
  "template-only"
)

$steps = @(
  New-RunbookStep -Order 1 -Id "confirm-owner-public-publish-completed" -Title "Confirm owner-approved public publish is complete" -Commands @(
    'Record selected package channel, package URLs, package ids, versions, and publish timestamp',
    'Do not run dotnet nuget push from this automation runbook'
  ) -RequiredEvidence @(
    "selected public or owner-approved package channel",
    "published package URLs",
    "managed/runtime package identity",
    "owner publish authorization record"
  ) -Boundary "This runbook does not publish; it only records owner-supplied public publish evidence."
  New-RunbookStep -Order 2 -Id "create-post-publish-clean-consumer" -Title "Create post-publish clean consumer outside repository" -Commands @(
    'New-Item -ItemType Directory -Force C:\trtsharp-post-publish-consumer | Out-Null',
    'Set-Location C:\trtsharp-post-publish-consumer',
    'dotnet new console --framework net8.0'
  ) -RequiredEvidence @(
    "repository-external clean consumer path",
    "consumer project hash",
    "no ProjectReference"
  ) -Boundary "Clean project creation is setup only and cannot prove post-publish availability."
  New-RunbookStep -Order 3 -Id "restore-from-public-channel" -Title "Restore packages from public channel" -Commands @(
    'dotnet nuget add source <public-package-source-url> --name trtsharp-post-publish-proof-source',
    'dotnet add package JYPPX.TensorRtSharp --version <published-version> --source <public-package-source-url>',
    'dotnet add package JYPPX.TensorRtSharp.Native.<runtime-key> --version <published-version> --source <public-package-source-url>',
    'dotnet restore --no-cache --force-evaluate *> post-publish-restore.log'
  ) -RequiredEvidence @(
    "public package source URL",
    "restore log path and SHA256",
    "package URL and downloaded nupkg SHA256",
    "no local feed or direct .nupkg"
  ) -Boundary "Post-publish proof requires public-channel restore; local feed and direct package files are forbidden substitutes."
  New-RunbookStep -Order 4 -Id "run-post-publish-runtime-smoke" -Title "Run post-publish runtime smoke" -Commands @(
    'dotnet build -c Release --no-restore *> post-publish-build.log',
    'dotnet run -c Release --no-build -- --runtime-smoke *> post-publish-smoke.stdout.log 2> post-publish-smoke.stderr.log',
    'Get-Content post-publish-restore.log,post-publish-build.log,post-publish-smoke.stdout.log,post-publish-smoke.stderr.log | Set-Content post-publish-merged-transcript.log'
  ) -RequiredEvidence @(
    "build log path and SHA256",
    "runtime smoke stdout/stderr paths and SHA256",
    "merged transcript path and SHA256",
    "exitCode=0 and passed=true"
  ) -Boundary "Only a real post-publish runtime smoke from the public channel can contribute to post-publish proof after strict validation."
  New-RunbookStep -Order 5 -Id "capture-post-publish-host-and-assets" -Title "Capture host, native asset, and dependency evidence" -Commands @(
    'dotnet --info',
    'nvidia-smi',
    'Get-ChildItem -Recurse .\bin\Release | Out-File post-publish-native-asset-listing.log',
    'Get-FileHash -Algorithm SHA256 post-publish-*.log'
  ) -RequiredEvidence @(
    "host metadata",
    "native asset listing",
    "dependency/runtime package identity",
    "log and package SHA256 values"
  ) -Boundary "Host and asset listings are support evidence, not post-publish proof alone."
  New-RunbookStep -Order 6 -Id "fill-owner-result-input-and-validate" -Title "Fill owner result input and validate post-publish lane" -Commands @(
    'Copy-Item artifacts/final-release/owner-external-proof-execution-result.input.template.json artifacts/final-release/owner-external-proof-execution-result.input.json',
    'Fill the post-publish-verification resultInputs[] item with public-channel paths, hashes, exitCode=0, passed=true, owner review, and nonSubstituteConfirmations',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCloseRealProofImportBridge.ps1',
    'pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCloseRealProofImportBridge.ps1 -Strict'
  ) -RequiredEvidence @(
    "post-publish resultInputs[] item",
    "owner external proof execution result import validation",
    "release close real proof import bridge validation",
    "strict post-publish validator output"
  ) -Boundary "Owner input and bridge output remain non-proof until strict post-publish validators accept real public-channel evidence."
)

$record = [pscustomobject]@{
  recordKind = "post-publish-owner-verification-runbook"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runbookState = "blocked-owner-post-publish-verification-required"
  runbookPurpose = "owner-executable post-publish clean consumer verification guidance"
  stepCount = $steps.Count
  blockedStepCount = $steps.Count
  steps = @($steps)
  requiredInputTarget = "artifacts/final-release/owner-external-proof-execution-result.input.json"
  fillableTemplate = "artifacts/final-release/owner-external-proof-execution-result.input.template.json"
  forbiddenSubstitutes = $forbiddenSubstitutes
  requiredPublicChannelEvidence = @(
    "public package source URL",
    "published package URL",
    "downloaded nupkg SHA256",
    "restore/build/runtime logs",
    "stdout/stderr/merged transcript",
    "validator output",
    "host metadata",
    "owner review"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This runbook is post-publish owner guidance only. It cannot publish, prove post-publish verification, promote runtime proof, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "post-publish-owner-verification-runbook.json"
$markdownPath = Join-Path $OutputRoot "post-publish-owner-verification-runbook.md"
$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$stepRows = foreach ($step in $steps) { "| ``$(ConvertTo-MarkdownCell $step.id)`` | ``$($step.order)`` | $(ConvertTo-MarkdownCell $step.title) | $(ConvertTo-MarkdownCell $step.boundary) |" }
$commandLines = foreach ($step in $steps) {
  "### $($step.order). $($step.title)"
  ""
  foreach ($command in $step.commands) { "- ``$command``" }
  ""
}

$markdown = @"
# Post Publish Owner Verification Runbook

该 runbook 指导 Owner 在真实公开发布之后采集 post-publish clean consumer verification 输入。它不会执行发布，不会把输入模板晋级为 proof，也不会关闭 release issue。

| Field | Value |
|---|---|
| runbookState | ``$($record.runbookState)`` |
| stepCount | ``$($record.stepCount)`` |
| requiredInputTarget | ``$($record.requiredInputTarget)`` |
| fillableTemplate | ``$($record.fillableTemplate)`` |
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

Write-Host "Post publish owner verification runbook written to $jsonPath"
Write-Host "RunbookState=$($record.runbookState) Steps=$($record.stepCount) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
