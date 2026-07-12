[CmdletBinding()]
param(
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

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ManualCommand {
  param(
    [string]$Id,
    [string]$Phase,
    [string]$Command,
    [string]$OwnerAction,
    [string]$RequiredProof,
    [string]$Validator
  )

  [pscustomobject]@{
    id = $Id
    phase = $Phase
    command = $Command
    notExecutedByAutomation = $true
    placeholderOnly = $true
    commandMaterializationState = "owner-manual-placeholder-only"
    materializedExecutableCommand = ""
    ownerExecutionOnly = $true
    modelExecutionForbidden = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    ownerAction = $OwnerAction
    requiredProof = $RequiredProof
    validator = $Validator
    boundary = "Manual command handoff only. This command is not executed by automation, is not package push proof, and is not release close approval."
  }
}

$freezeManifestValidation = Read-JsonOrNull "artifacts\final-release\release-candidate-final-freeze-manifest-validation.json"
$releaseEvidence = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$publicPackageValidation = Read-JsonOrNull "artifacts\final-release\public-package-proof-owner-input-validation.json"
$postPublishConfirmationValidation = Read-JsonOrNull "artifacts\final-release\post-publish-proof-owner-confirmation-validation.json"
$finalPostPublishValidation = Read-JsonOrNull "artifacts\final-release\final-post-publish-audit-pack-validation.json"
$strictCloseValidation = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$manualCommands = @(
  New-ManualCommand `
    -Id "review-freeze-manifest" `
    -Phase "pre-publish-review" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFinalFreezeManifest.ps1 -Strict" `
    -OwnerAction "Owner reviews final freeze manifest, artifact hashes, and non-proof boundaries before any public command is copied." `
    -RequiredProof "Freeze manifest validation artifact and human review notes." `
    -Validator "Test-ReleaseCandidateFinalFreezeManifest.ps1 -Strict"

  New-ManualCommand `
    -Id "nuget-public-push-placeholder" `
    -Phase "owner-public-publish" `
    -Command "dotnet nuget push <PACKAGE_PATH> --api-key <NUGET_API_KEY> --source https://api.nuget.org/v3/index.json --skip-duplicate" `
    -OwnerAction "Owner manually replaces placeholders and runs this outside automation only after authorization and package hash review." `
    -RequiredProof "Public NuGet package URL, package version, published timestamp, and nupkg SHA256 entered into public-package-proof-owner-input." `
    -Validator "Test-PublicPackageProofOwnerInput.ps1 -Strict"

  New-ManualCommand `
    -Id "github-package-push-placeholder" `
    -Phase "owner-public-publish" `
    -Command "dotnet nuget push <PACKAGE_PATH> --api-key <GITHUB_TOKEN> --source https://nuget.pkg.github.com/<OWNER>/index.json --skip-duplicate" `
    -OwnerAction "Owner manually chooses whether GitHub Packages is an approved secondary channel and records URL/hash evidence." `
    -RequiredProof "GitHub Packages URL, package version, published timestamp, and nupkg SHA256 if used." `
    -Validator "Test-PublicPackageProofOwnerInput.ps1 -Strict"

  New-ManualCommand `
    -Id "post-publish-clean-consumer-smoke-placeholder" `
    -Phase "post-publish-proof" `
    -Command "dotnet new console -n <CLEAN_CONSUMER>; dotnet add <CLEAN_CONSUMER> package JYPPX.TensorRtSharp --version <PUBLIC_VERSION>; dotnet run --project <CLEAN_CONSUMER>" `
    -OwnerAction "Owner runs clean consumer restore/build/runtime smoke from the public package source and records log path/hash/host metadata." `
    -RequiredProof "Clean consumer restore/build/runtime log, SHA256, package source list, host metadata, and no local feed evidence." `
    -Validator "Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof"

  New-ManualCommand `
    -Id "strict-close-validator-placeholder" `
    -Phase "release-close" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseIssueCloseRecord.ps1 -InputPath artifacts/final-release/release-issue-close-record.json -FailOnNotCloseReady" `
    -OwnerAction "Owner runs strict close validator only after all public package and post-publish proof records are real and validated." `
    -RequiredProof "Ready release issue close record with real proof references and post-publish verification." `
    -Validator "Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
)

$freezeValidationState = [string](Get-PropertyOrDefault -Object $freezeManifestValidation -Name "validationState" -DefaultValue "missing-release-candidate-final-freeze-manifest-validation")
$publicPackageValidationState = [string](Get-PropertyOrDefault -Object $publicPackageValidation -Name "validationState" -DefaultValue "missing-public-package-proof-owner-input-validation")
$postPublishConfirmationState = [string](Get-PropertyOrDefault -Object $postPublishConfirmationValidation -Name "validationState" -DefaultValue "missing-post-publish-proof-owner-confirmation-validation")
$finalPostPublishState = [string](Get-PropertyOrDefault -Object $finalPostPublishValidation -Name "validationState" -DefaultValue "missing-final-post-publish-audit-pack-validation")
$strictCloseState = [string](Get-PropertyOrDefault -Object $strictCloseValidation -Name "validationState" -DefaultValue "missing-release-issue-close-record-validation")

$manualPrerequisites = @(
  [pscustomobject]@{ id = "freeze-manifest"; state = $freezeValidationState; requiredState = "release-candidate-final-freeze-manifest-ready-for-owner-handoff"; ready = [string]::Equals($freezeValidationState, "release-candidate-final-freeze-manifest-ready-for-owner-handoff", [StringComparison]::Ordinal) }
  [pscustomobject]@{ id = "public-package-proof-owner-input"; state = $publicPackageValidationState; requiredState = "public-package-proof-owner-input-ready"; ready = [string]::Equals($publicPackageValidationState, "public-package-proof-owner-input-ready", [StringComparison]::Ordinal) }
  [pscustomobject]@{ id = "post-publish-proof-owner-confirmation"; state = $postPublishConfirmationState; requiredState = "post-publish-proof-owner-confirmation-ready"; ready = [string]::Equals($postPublishConfirmationState, "post-publish-proof-owner-confirmation-ready", [StringComparison]::Ordinal) }
  [pscustomobject]@{ id = "final-post-publish-audit-pack"; state = $finalPostPublishState; requiredState = "final-post-publish-audit-ready"; ready = [string]::Equals($finalPostPublishState, "final-post-publish-audit-ready", [StringComparison]::Ordinal) }
  [pscustomobject]@{ id = "strict-release-close-validator"; state = $strictCloseState; requiredState = "ready-for-owner-release-issue-close"; ready = [string]::Equals($strictCloseState, "ready-for-owner-release-issue-close", [StringComparison]::Ordinal) }
)

$blockedPrerequisites = @($manualPrerequisites | Where-Object { -not [bool]$_.ready })

$record = [pscustomobject]@{
  recordKind = "public-publish-owner-manual-command-handoff"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  handoffState = "blocked-owner-public-publish-required"
  commandCount = $manualCommands.Count
  manualPrerequisiteCount = $manualPrerequisites.Count
  blockedManualPrerequisiteCount = $blockedPrerequisites.Count
  manualPrerequisites = $manualPrerequisites
  manualCommands = $manualCommands
  notExecutedByAutomation = $true
  sourceArtifacts = @(
    "artifacts/final-release/release-candidate-final-freeze-manifest-validation.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/public-package-proof-owner-input-validation.json",
    "artifacts/final-release/post-publish-proof-owner-confirmation-validation.json",
    "artifacts/final-release/final-post-publish-audit-pack-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Public publish owner manual command handoff provides copyable placeholders only. It never executes dotnet nuget push, never publishes packages, and is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "public-publish-owner-manual-command-handoff.json"
$markdownPath = Join-Path $artifactRoot "public-publish-owner-manual-command-handoff.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$commandRows = $record.manualCommands | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | $(ConvertTo-MarkdownCell $_.phase) | ``$($_.notExecutedByAutomation)`` | ``$($_.performsPublish)`` | $(ConvertTo-MarkdownCell $_.validator) |"
}

$prereqRows = $record.manualPrerequisites | ForEach-Object {
  "| $(ConvertTo-MarkdownCell $_.id) | $(ConvertTo-MarkdownCell $_.state) | $(ConvertTo-MarkdownCell $_.requiredState) | ``$($_.ready)`` |"
}

$markdown = @"
# Public Publish Owner Manual Command Handoff

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| handoffState | ``$($record.handoffState)`` |
| commandCount | ``$($record.commandCount)`` |
| manualPrerequisiteCount | ``$($record.manualPrerequisiteCount)`` |
| blockedManualPrerequisiteCount | ``$($record.blockedManualPrerequisiteCount)`` |
| notExecutedByAutomation | ``$($record.notExecutedByAutomation)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Manual Prerequisites

| Prerequisite | Current State | Required State | Ready |
|---|---|---|---:|
$($prereqRows -join "`r`n")

## Manual Commands

| Command | Phase | Not Executed By Automation | Performs Publish | Validator |
|---|---|---:|---:|---|
$($commandRows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish owner manual command handoff written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "HandoffState=$($record.handoffState) Commands=$($record.commandCount) BlockedPrerequisites=$($record.blockedManualPrerequisiteCount)"
