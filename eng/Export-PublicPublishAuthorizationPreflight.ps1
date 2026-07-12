[CmdletBinding()]
param(
  [string]$OwnerImportPacketPath = "artifacts/final-release/owner-real-evidence-import-packet.json",
  [string]$OwnerImportPacketValidationPath = "artifacts/final-release/owner-real-evidence-import-packet-validation.json",
  [string]$FinalActionMapPath = "artifacts/final-release/final-publish-action-required-evidence-map.json",
  [string]$ReleaseCloseStrictOrderPath = "artifacts/final-release/release-close-strict-proof-execution-order.json",
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-Requirement {
  param(
    [string]$Id,
    [string]$Category,
    [string]$RequiredEvidence,
    [string[]]$SourceLanes,
    [string[]]$Validators,
    [string]$OwnerAction,
    [string[]]$ForbiddenSubstitutes
  )

  [pscustomobject]@{
    id = $Id
    category = $Category
    state = "blocked-owner-real-evidence-required"
    requiredEvidence = $RequiredEvidence
    sourceLanes = @($SourceLanes)
    validators = @($Validators)
    ownerAction = $OwnerAction
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromotePackageConsumerRuntime = $false
    canPromoteRuntimeProof = $false
  }
}

$packet = Read-JsonOrNull -Path $OwnerImportPacketPath
$packetValidation = Read-JsonOrNull -Path $OwnerImportPacketValidationPath
$finalActionMap = Read-JsonOrNull -Path $FinalActionMapPath
$strictOrder = Read-JsonOrNull -Path $ReleaseCloseStrictOrderPath

$forbiddenSubstitutes = @(
  "template-only record",
  "dashboard-only record",
  "build-only report",
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "dry-run record",
  "screenshot-only evidence",
  "article readiness map"
)

$requirements = @(
  New-Requirement -Id "public-package-source-and-identity" -Category "public-package-channel" -RequiredEvidence "Real public package source, package id, published version, public URL, and owner reviewer." -SourceLanes @("package-consumer-runtime-owner-proof-required", "post-publish-verification-owner-proof-required") -Validators @("eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof", "eng/Test-PostPublishVerificationRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof") -OwnerAction "Owner supplies public package source and immutable package identity from the selected GitHub/NuGet route." -ForbiddenSubstitutes $forbiddenSubstitutes
  New-Requirement -Id "real-package-hashes" -Category "package-hash-chain" -RequiredEvidence "Managed package SHA256, native bridge SHA256, runtime package SHA256, downloaded package hash manifest, and matching source file paths." -SourceLanes @("package-consumer-runtime-owner-proof-required", "owner-external-proof-result-import-owner-proof-required") -Validators @("eng/Test-OwnerRealInputHashAndPathValidator.ps1 -Strict", "eng/Test-OwnerExternalProofExecutionResultImport.ps1 -Strict") -OwnerAction "Owner supplies real package files and hash manifest; every hash must match an existing file." -ForbiddenSubstitutes $forbiddenSubstitutes
  New-Requirement -Id "clean-external-consumer-smoke" -Category "package-consumer-runtime" -RequiredEvidence "Clean external consumer root outside the repository, restore/build/smoke commands, exitCode=0, stdout/stderr, smoke log path, smoke log SHA256, host metadata." -SourceLanes @("package-consumer-runtime-owner-proof-required") -Validators @("eng/Test-PackageConsumerRuntimeProofRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof") -OwnerAction "Owner runs restore/build/smoke in an external clean consumer using only the public package source." -ForbiddenSubstitutes $forbiddenSubstitutes
  New-Requirement -Id "post-publish-public-install-run" -Category "post-publish-verification" -RequiredEvidence "Public channel install transcript, public run transcript, smoke log, smoke log SHA256, stdout/stderr summary, host metadata, owner review." -SourceLanes @("post-publish-verification-owner-proof-required") -Validators @("eng/Test-PostPublishVerificationRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof") -OwnerAction "After real publication, Owner installs and runs from the public channel and records reviewed logs/hashes." -ForbiddenSubstitutes $forbiddenSubstitutes
  New-Requirement -Id "owner-review-and-rollback-plan" -Category "owner-release-decision" -RequiredEvidence "Owner reviewer, reviewedAtUtc, owner decision, rollback/deprecation plan reference, release issue close/keep-open decision." -SourceLanes @("post-publish-verification-owner-proof-required", "owner-result-candidate-bridge-real-proof-required") -Validators @("eng/Test-PostPublishVerificationRecord.ps1 -Strict -RequireExistingLog -FailOnNotProof", "eng/Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict") -OwnerAction "Owner reviews public-channel evidence and records rollback/deprecation and release close decision inputs." -ForbiddenSubstitutes $forbiddenSubstitutes
  New-Requirement -Id "release-close-final-decision" -Category "release-close-authorization" -RequiredEvidence "All six final action-required lanes pass strict validation, final evidence bundle is reviewed, stale claim audit is clean, and Owner final close decision is present." -SourceLanes @("final-owner-real-input-template-pack-owner-input-required", "real-model-runtime-owner-proof-required", "package-consumer-runtime-owner-proof-required", "post-publish-verification-owner-proof-required", "owner-external-proof-result-import-owner-proof-required", "owner-result-candidate-bridge-real-proof-required") -Validators @("eng/Test-FinalPublishProofGate.ps1 -Strict", "eng/Test-StaleReleaseClaims.ps1", "eng/Export-ReleaseCandidateFinalEvidenceFreeze.ps1") -OwnerAction "Owner authorizes final public publish and close only after every real proof lane and strict gate is accepted." -ForbiddenSubstitutes $forbiddenSubstitutes
)

$preflight = [pscustomobject]@{
  recordKind = "public-publish-authorization-preflight"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  preflightState = "blocked-owner-public-publish-authorization-required"
  sourceOwnerImportPacket = $OwnerImportPacketPath
  sourceOwnerImportPacketValidation = $OwnerImportPacketValidationPath
  sourceFinalActionMap = $FinalActionMapPath
  sourceReleaseCloseStrictOrder = $ReleaseCloseStrictOrderPath
  ownerImportLaneCount = if ($null -ne $packet) { [int]$packet.laneCount } else { 0 }
  ownerImportValidationState = if ($null -ne $packetValidation) { [string]$packetValidation.validationState } else { "missing" }
  finalActionRequiredCount = if ($null -ne $finalActionMap) { [int]$finalActionMap.actionRequiredCount } else { 0 }
  releaseCloseStrictExecutionStepCount = if ($null -ne $strictOrder) { [int]$strictOrder.executionStepCount } else { 0 }
  requirementCount = @($requirements).Count
  blockedRequirementCount = @($requirements).Count
  requirements = @($requirements)
  commandPolicy = [pscustomobject]@{
    executesDotnetNugetPush = $false
    uploadsGitHubReleaseAssets = $false
    storesTokens = $false
    closesReleaseIssue = $false
    commandBoundary = "This preflight describes authorization requirements only. It does not run dotnet nuget push, upload GitHub Release assets, store tokens, or close release issues."
  }
  performsPublish = $false
  performsRuntimeExecution = $false
  uploadsGitHubReleaseAssets = $false
  executesDotnetNugetPush = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromotePackageConsumerRuntime = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "This preflight is a non-executing authorization checklist. It cannot become publish approval or close approval until real Owner evidence and strict validators pass."
}

$jsonPath = Join-Path $OutputRoot "public-publish-authorization-preflight.json"
$markdownPath = Join-Path $OutputRoot "public-publish-authorization-preflight.md"
$preflight | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($requirement in $requirements) {
  "| ``$(ConvertTo-MarkdownCell $requirement.id)`` | ``$(ConvertTo-MarkdownCell $requirement.category)`` | ``$(ConvertTo-MarkdownCell $requirement.state)`` | ``False`` |"
}

$markdown = @"
# Public Publish Authorization Preflight

Generated at: ``$($preflight.generatedAtUtc)``

## Summary

- preflightState: ``$($preflight.preflightState)``
- ownerImportLaneCount: ``$($preflight.ownerImportLaneCount)``
- requirementCount: ``$($preflight.requirementCount)``
- blockedRequirementCount: ``$($preflight.blockedRequirementCount)``
- performsPublish: ``False``
- executesDotnetNugetPush: ``False``
- uploadsGitHubReleaseAssets: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``

## Requirements

| Requirement | Category | State | Can Publish |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Boundary

$($preflight.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Public publish authorization preflight written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "PreflightState=$($preflight.preflightState) RequirementCount=$($preflight.requirementCount)"
