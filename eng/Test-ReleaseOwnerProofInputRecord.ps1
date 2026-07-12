[CmdletBinding()]
param(
  [string]$InputPath = ".\artifacts\final-release\release-owner-proof-input-record.json",
  [switch]$RequireExistingLogs,
  [switch]$FailOnNotProof,
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

function Resolve-RepoPath {
  param([string]$Path)
  if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepoPath $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Test-TextReady {
  param([AllowNull()][object]$Value)
  return -not [string]::IsNullOrWhiteSpace([string]$Value)
}

function Test-Sha256 {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match '^[A-Fa-f0-9]{64}$'
}

function Test-LogHashMatches {
  param(
    [AllowNull()][object]$Path,
    [AllowNull()][object]$ExpectedSha256
  )

  if (-not (Test-Sha256 $ExpectedSha256)) { return $false }
  $resolved = Resolve-RepoPath ([string]$Path)
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $false }
  $actual = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
  return [string]::Equals($actual, ([string]$ExpectedSha256).ToLowerInvariant(), [System.StringComparison]::Ordinal)
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Detail,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    detail = $Detail
    boundary = $Boundary
  }
}

$inputFullPath = Resolve-RepoPath $InputPath
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-ReleaseOwnerProofInputRecordTemplate.ps1") -RepositoryRoot $RepositoryRoot | Out-Null
  $inputFullPath = Join-Path $RepositoryRoot "artifacts\final-release\release-owner-proof-input-record-template.json"
}

$record = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$recordState = [string](Get-PropertyOrDefault -Object $record -Name "recordState" -DefaultValue "")
$isTemplate = $recordKind -like "*template*" -or $recordState -eq "template-only"

$owner = Get-PropertyOrDefault -Object $record -Name "ownerAuthorization" -DefaultValue $null
$channel = Get-PropertyOrDefault -Object $record -Name "selectedChannel" -DefaultValue $null
$packages = Get-PropertyOrDefault -Object $record -Name "packages" -DefaultValue $null
$managedPackage = Get-PropertyOrDefault -Object $packages -Name "managed" -DefaultValue $null
$runtimePackage = Get-PropertyOrDefault -Object $packages -Name "runtime" -DefaultValue $null
$consumer = Get-PropertyOrDefault -Object $record -Name "cleanConsumer" -DefaultValue $null
$runtime = Get-PropertyOrDefault -Object $record -Name "runtimeEvidence" -DefaultValue $null
$hostMetadata = Get-PropertyOrDefault -Object $record -Name "hostMetadata" -DefaultValue $null
$ack = Get-PropertyOrDefault -Object $record -Name "acknowledgements" -DefaultValue $null

$ownerAuthorizationReady =
  (Test-TextReady (Get-PropertyOrDefault -Object $owner -Name "ownerName" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $owner -Name "ownerDecisionId" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $owner -Name "approvalTimestampUtc" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $owner -Name "targetChannel" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $owner -Name "releaseIssueUrl" -DefaultValue ""))

$selectedChannelReady =
  (Test-TextReady (Get-PropertyOrDefault -Object $channel -Name "channelName" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $channel -Name "channelSourceUri" -DefaultValue ""))

$packageIdentityReady =
  (Test-TextReady (Get-PropertyOrDefault -Object $managedPackage -Name "packageId" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $managedPackage -Name "packageVersion" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $managedPackage -Name "packageUrl" -DefaultValue "")) -and
  (Test-Sha256 (Get-PropertyOrDefault -Object $managedPackage -Name "nupkgSha256" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtimePackage -Name "packageId" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtimePackage -Name "packageVersion" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtimePackage -Name "packageUrl" -DefaultValue "")) -and
  (Test-Sha256 (Get-PropertyOrDefault -Object $runtimePackage -Name "nupkgSha256" -DefaultValue ""))

$cleanConsumerReady =
  (Test-TextReady (Get-PropertyOrDefault -Object $consumer -Name "cleanConsumerRoot" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $consumer -Name "consumerProjectPath" -DefaultValue "")) -and
  [bool](Get-PropertyOrDefault -Object $consumer -Name "noProjectReference" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $consumer -Name "noLocalPackageSource" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $consumer -Name "noLocalNupkgPackageReference" -DefaultValue $false)

$runtimeHashesReady =
  (Test-Sha256 (Get-PropertyOrDefault -Object $runtime -Name "restoreLogSha256" -DefaultValue "")) -and
  (Test-Sha256 (Get-PropertyOrDefault -Object $runtime -Name "nativeAssetListingSha256" -DefaultValue "")) -and
  (Test-Sha256 (Get-PropertyOrDefault -Object $runtime -Name "dependencyProbeLogSha256" -DefaultValue "")) -and
  (Test-Sha256 (Get-PropertyOrDefault -Object $runtime -Name "runtimeSmokeLogSha256" -DefaultValue ""))

$runtimeTextReady =
  (Test-TextReady (Get-PropertyOrDefault -Object $runtime -Name "restoreLogPath" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtime -Name "nativeAssetListingPath" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtime -Name "dependencyProbeLogPath" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtime -Name "runtimeSmokeLogPath" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtime -Name "stdoutSummary" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $runtime -Name "stderrSummary" -DefaultValue ""))

$runtimeSmokeExitCode = Get-PropertyOrDefault -Object $runtime -Name "runtimeSmokeExitCode" -DefaultValue $null
$runtimeEvidenceReady =
  $runtimeTextReady -and
  $runtimeHashesReady -and
  [bool](Get-PropertyOrDefault -Object $runtime -Name "runtimeSmokePassed" -DefaultValue $false) -and
  ($null -ne $runtimeSmokeExitCode) -and
  ([int]$runtimeSmokeExitCode -eq 0)

$hostMetadataReady =
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "osDescription" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "gpuName" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "driverVersion" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "cudaDriverVersion" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "cudaRuntimeVersion" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "tensorRtRuntimeVersion" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "tensorRtRuntimeLine" -DefaultValue "")) -and
  (Test-TextReady (Get-PropertyOrDefault -Object $hostMetadata -Name "cudnnVersion" -DefaultValue ""))

$acknowledgementsReady =
  [bool](Get-PropertyOrDefault -Object $ack -Name "notTemplate" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $ack -Name "noLocalFeed" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $ack -Name "noProjectReference" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $ack -Name "noDirectNupkgReference" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $ack -Name "notManagedReadiness" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $ack -Name "realRuntimeSmokeExecuted" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $ack -Name "hashesComputedFromReferencedFiles" -DefaultValue $false)

$logHashesMatch = $false
if ($RequireExistingLogs) {
  $logHashesMatch =
    (Test-LogHashMatches -Path (Get-PropertyOrDefault -Object $runtime -Name "restoreLogPath" -DefaultValue "") -ExpectedSha256 (Get-PropertyOrDefault -Object $runtime -Name "restoreLogSha256" -DefaultValue "")) -and
    (Test-LogHashMatches -Path (Get-PropertyOrDefault -Object $runtime -Name "nativeAssetListingPath" -DefaultValue "") -ExpectedSha256 (Get-PropertyOrDefault -Object $runtime -Name "nativeAssetListingSha256" -DefaultValue "")) -and
    (Test-LogHashMatches -Path (Get-PropertyOrDefault -Object $runtime -Name "dependencyProbeLogPath" -DefaultValue "") -ExpectedSha256 (Get-PropertyOrDefault -Object $runtime -Name "dependencyProbeLogSha256" -DefaultValue "")) -and
    (Test-LogHashMatches -Path (Get-PropertyOrDefault -Object $runtime -Name "runtimeSmokeLogPath" -DefaultValue "") -ExpectedSha256 (Get-PropertyOrDefault -Object $runtime -Name "runtimeSmokeLogSha256" -DefaultValue ""))
}

$items = @(
  New-ValidationItem -Id "not-template-record" -Passed (-not $isTemplate) -Detail "recordKind=$recordKind; recordState=$recordState" -Boundary "Templates and examples cannot become owner proof input."
  New-ValidationItem -Id "owner-authorization-fields" -Passed $ownerAuthorizationReady -Detail "owner name, decision id, timestamp, target channel, and release issue URL are required." -Boundary "A vague approval flag is not owner authorization."
  New-ValidationItem -Id "selected-channel-fields" -Passed $selectedChannelReady -Detail "selected channel name and source URI are required." -Boundary "Local or unspecified channels cannot prove publication path."
  New-ValidationItem -Id "package-url-and-hash-fields" -Passed $packageIdentityReady -Detail "managed/runtime package id, version, URL, and SHA256 are required." -Boundary "A local nupkg path or missing hash is not selected-channel proof."
  New-ValidationItem -Id "clean-consumer-boundary" -Passed $cleanConsumerReady -Detail "clean consumer must reject ProjectReference, local package sources, and direct .nupkg references." -Boundary "Repository-local consumers cannot substitute post-publish proof."
  New-ValidationItem -Id "runtime-evidence-fields" -Passed $runtimeEvidenceReady -Detail "restore/native asset/dependency probe/runtime smoke logs, hashes, exit code, and summaries are required." -Boundary "Dependency probe or build-only evidence is not runtime smoke proof."
  New-ValidationItem -Id "host-metadata-fields" -Passed $hostMetadataReady -Detail "OS/GPU/driver/CUDA/TensorRT/cuDNN metadata are required." -Boundary "Runtime proof without host metadata is not reproducible."
  New-ValidationItem -Id "owner-acknowledgements" -Passed $acknowledgementsReady -Detail "Owner must acknowledge non-template, no local feed/reference, real runtime smoke, and computed hashes." -Boundary "Readiness snapshots and schema-only records cannot substitute proof."
)

if ($RequireExistingLogs) {
  $items += New-ValidationItem -Id "referenced-log-hashes-match" -Passed $logHashesMatch -Detail "RequireExistingLogs was specified; referenced log paths must exist and match SHA256 fields." -Boundary "Mismatched or missing logs cannot promote owner proof input."
}

$failedItems = @($items | Where-Object { -not [bool]$_.passed })
$canPromoteOwnerProofInput =
  -not $isTemplate -and
  $ownerAuthorizationReady -and
  $selectedChannelReady -and
  $packageIdentityReady -and
  $cleanConsumerReady -and
  $runtimeEvidenceReady -and
  $hostMetadataReady -and
  $acknowledgementsReady -and
  ((-not $RequireExistingLogs) -or $logHashesMatch) -and
  $failedItems.Count -eq 0

$proofClassification = if ($canPromoteOwnerProofInput) {
  "real-owner-proof-input"
} elseif ($isTemplate) {
  "template-only"
} else {
  "blocked-owner-input-required"
}

$validationState = if ($canPromoteOwnerProofInput) {
  "real-owner-proof-input-valid"
} elseif ($isTemplate) {
  "blocked-template-only"
} else {
  "blocked-owner-input-required"
}

$nonSubstitutes = @(
  "template",
  "draft",
  "example",
  "dashboard",
  "readiness snapshot",
  "collection package",
  "local feed",
  "ProjectReference",
  "direct .nupkg reference",
  "dependency-probe-only",
  "build-only",
  "parse-only",
  "sidecar-only",
  "managed-readiness",
  "CallbackAllocatorReadinessSnapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only"
)

$validation = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-owner-proof-input-record-validation"
  sourceInputPath = $inputFullPath
  validationState = $validationState
  proofClassification = $proofClassification
  isTemplate = $isTemplate
  isRealOwnerProofInput = $canPromoteOwnerProofInput
  canPromoteOwnerProofInput = $canPromoteOwnerProofInput
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  requireExistingLogs = [bool]$RequireExistingLogs
  validationItemCount = $items.Count
  failedValidationItemCount = $failedItems.Count
  validationItems = $items
  ownerAuthorizationReady = $ownerAuthorizationReady
  selectedChannelReady = $selectedChannelReady
  packageIdentityReady = $packageIdentityReady
  cleanConsumerReady = $cleanConsumerReady
  runtimeEvidenceReady = $runtimeEvidenceReady
  hostMetadataReady = $hostMetadataReady
  acknowledgementsReady = $acknowledgementsReady
  referencedLogHashesMatch = $logHashesMatch
  nonSubstituteProofKinds = $nonSubstitutes
  boundary = "This validator can validate owner-filled proof input only. It does not publish packages, authorize publication by itself, promote runtime proof, or close the release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "release-owner-proof-input-record-validation.json"
$markdownPath = Join-Path $artifactRoot "release-owner-proof-input-record-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $items | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | $(([string]$_.detail).Replace("|", "\|")) | $(([string]$_.boundary).Replace("|", "\|")) |"
}
$nonSubstituteLines = $nonSubstitutes | ForEach-Object { "- ``$_``" }
$markdown = @"
# Release Owner Proof Input Record Validation

- validation state: ``$validationState``
- proof classification: ``$proofClassification``
- is real owner proof input: ``$canPromoteOwnerProofInput``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- failed validation item count: ``$($failedItems.Count)``

## Validation Items

| ID | Passed | Detail | Boundary |
| --- | --- | --- | --- |
$($rows -join "`r`n")

## Non-Substitute Proof Kinds

$($nonSubstituteLines -join "`r`n")

## Boundary

$($validation.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release owner proof input record validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState"
Write-Output "ProofClassification=$proofClassification"
Write-Output "CanPromoteOwnerProofInput=$canPromoteOwnerProofInput"
Write-Output "CanPublishPublicly=False"

if ($FailOnNotProof -and -not $canPromoteOwnerProofInput) {
  throw "Release owner proof input record is not real proof input. validationState=$validationState failedValidationItemCount=$($failedItems.Count)"
}
