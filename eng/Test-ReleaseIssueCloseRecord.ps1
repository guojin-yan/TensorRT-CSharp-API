[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$InputPath = ".\artifacts\final-release\release-issue-close-record.json",
  [switch]$FailOnNotCloseReady
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepoPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $null
  }

  $fullPath = Resolve-RepoPath -Path $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $fullPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrNull {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  if ($null -eq $Object) { return $null }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $null
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  $value = Get-PropertyOrNull -Object $Object -Name $Name
  if ($null -eq $value) { return $DefaultValue }
  return $value
}

function Test-Truthy {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return $false }
  if ($Value -is [bool]) { return [bool]$Value }
  $text = [string]$Value
  return [string]::Equals($text, "true", [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-NonEmpty {
  param([AllowNull()][object]$Value)

  return -not [string]::IsNullOrWhiteSpace([string]$Value)
}

function Get-NestedPropertyOrNull {
  param(
    [AllowNull()][object]$Object,
    [string[]]$Path
  )

  $current = $Object
  foreach ($part in $Path) {
    if ($null -eq $current) { return $null }
    if ($current.PSObject.Properties.Name -notcontains $part) { return $null }
    $current = $current.PSObject.Properties[$part].Value
  }

  return $current
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

function Get-FileSha256OrNull {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $null
  }

  $fullPath = Resolve-RepoPath -Path $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    return $null
  }

  return (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
}

$inputFullPath = Resolve-RepoPath -Path $InputPath
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  $inputFullPath = Join-Path $RepositoryRoot "artifacts\final-release\release-issue-close-record-template.json"
}

if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) {
  throw "Release issue close record input not found. Run Export-ReleaseIssueCloseRecordTemplate.ps1 first."
}

$record = Get-Content -LiteralPath $inputFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseClosePreflight = Read-JsonOrNull "artifacts\final-release\release-close-preflight.json"
$staleAudit = Read-JsonOrNull "artifacts\final-release\stale-release-claims-audit.json"
$ownerProofInputValidation = Read-JsonOrNull "artifacts\final-release\release-owner-proof-input-record-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$recordState = [string](Get-PropertyOrDefault -Object $record -Name "recordState" -DefaultValue "")
$proofClassification = [string](Get-PropertyOrDefault -Object $record -Name "proofClassification" -DefaultValue "")
$isTemplate = $recordKind -eq "release-issue-close-record-template" -or $recordState -eq "template-only" -or $proofClassification -eq "template-only"
$declaredCanPublishPublicly = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canPublishPublicly")
$declaredCanCloseReleaseIssue = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canCloseReleaseIssue")
$declaredPromoteCloseRecord = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canPromoteReleaseIssueCloseRecord")

$releaseIssueId = Get-NestedPropertyOrNull -Object $record -Path @("releaseIssue", "id")
$releaseIssueUrl = Get-NestedPropertyOrNull -Object $record -Path @("releaseIssue", "url")
$ownerName = Get-NestedPropertyOrNull -Object $record -Path @("ownerDecision", "ownerName")
$ownerApprovalTimestamp = Get-NestedPropertyOrNull -Object $record -Path @("ownerDecision", "approvalTimestampUtc")
$ownerFinalCloseDecision = Get-NestedPropertyOrNull -Object $record -Path @("ownerDecision", "finalCloseDecision")
$ownerCloseReason = Get-NestedPropertyOrNull -Object $record -Path @("ownerDecision", "closeDecisionReason")
$ownerAcknowledgedNoProofNoClose = Test-Truthy (Get-NestedPropertyOrNull -Object $record -Path @("ownerDecision", "noRealProofNoCloseAcknowledged"))
$selectedChannelName = Get-NestedPropertyOrNull -Object $record -Path @("selectedChannel", "name")
$selectedChannelSourceUri = Get-NestedPropertyOrNull -Object $record -Path @("selectedChannel", "sourceUri")
$managedPackageUrl = Get-NestedPropertyOrNull -Object $record -Path @("packages", "managed", "packageUrl")
$managedPackageSha256 = Get-NestedPropertyOrNull -Object $record -Path @("packages", "managed", "nupkgSha256")
$runtimePackageUrl = Get-NestedPropertyOrNull -Object $record -Path @("packages", "runtime", "packageUrl")
$runtimePackageSha256 = Get-NestedPropertyOrNull -Object $record -Path @("packages", "runtime", "nupkgSha256")
$bundleHashDeclared = Get-NestedPropertyOrNull -Object $record -Path @("releaseEvidenceBundle", "sha256")
$rollbackSummary = Get-NestedPropertyOrNull -Object $record -Path @("rollbackPlan", "summary")
$rollbackPackagePlan = Get-NestedPropertyOrNull -Object $record -Path @("rollbackPlan", "packageYankOrDeprecatePlan")

$postPublishState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
$postPublishProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
$postPublishCanClose = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
$preflightState = [string](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "preflightState" -DefaultValue "missing-release-close-preflight")
$preflightCanClose = [bool](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "canCloseReleaseIssue" -DefaultValue $false)
$preflightFailedItemCount = [int](Get-PropertyOrDefault -Object $releaseClosePreflight -Name "failedItemCount" -DefaultValue -1)
$staleFindingCount = [int](Get-PropertyOrDefault -Object $staleAudit -Name "findingCount" -DefaultValue -1)
$ownerProofState = [string](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "validationState" -DefaultValue "missing-release-owner-proof-input-record-validation")
$ownerProofPromote = [bool](Get-PropertyOrDefault -Object $ownerProofInputValidation -Name "canPromoteOwnerProofInput" -DefaultValue $false)
$evidenceBundleState = [string](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "bundleState" -DefaultValue "missing-release-evidence-bundle")
$evidenceBundleHashActual = Get-FileSha256OrNull -Path "artifacts\final-release\release-evidence-bundle.json"
$bundleHashMatches = (Test-NonEmpty $bundleHashDeclared) -and (Test-NonEmpty $evidenceBundleHashActual) -and [string]::Equals([string]$bundleHashDeclared, [string]$evidenceBundleHashActual, [System.StringComparison]::OrdinalIgnoreCase)

$ownerDecisionReady =
  (Test-NonEmpty $releaseIssueId) -and
  (Test-NonEmpty $releaseIssueUrl) -and
  (Test-NonEmpty $ownerName) -and
  (Test-NonEmpty $ownerApprovalTimestamp) -and
  [string]::Equals([string]$ownerFinalCloseDecision, "close-release-issue", [System.StringComparison]::OrdinalIgnoreCase) -and
  (Test-NonEmpty $ownerCloseReason) -and
  $ownerAcknowledgedNoProofNoClose
$selectedChannelReady = (Test-NonEmpty $selectedChannelName) -and (Test-NonEmpty $selectedChannelSourceUri)
$packageIdentityReady = (Test-NonEmpty $managedPackageUrl) -and (Test-NonEmpty $managedPackageSha256) -and (Test-NonEmpty $runtimePackageUrl) -and (Test-NonEmpty $runtimePackageSha256)
$rollbackPlanReady = (Test-NonEmpty $rollbackSummary) -and (Test-NonEmpty $rollbackPackagePlan)

$items = @(
  New-ValidationItem -Id "not-template-record" -Passed (-not $isTemplate -and $recordKind -eq "release-issue-close-record") -Detail "recordKind=$recordKind; recordState=$recordState; proofClassification=$proofClassification" -Boundary "Templates and examples cannot close the release issue."
  New-ValidationItem -Id "owner-final-close-decision" -Passed $ownerDecisionReady -Detail "releaseIssueId=$releaseIssueId; ownerName=$ownerName; finalCloseDecision=$ownerFinalCloseDecision; acknowledgedNoProofNoClose=$ownerAcknowledgedNoProofNoClose" -Boundary "Final close requires explicit owner identity, timestamp, release issue, close decision, reason, and no-proof acknowledgement."
  New-ValidationItem -Id "selected-channel-fields" -Passed $selectedChannelReady -Detail "selectedChannel=$selectedChannelName; sourceUri=$selectedChannelSourceUri" -Boundary "A release issue cannot close against an unspecified package channel."
  New-ValidationItem -Id "package-url-and-hash-fields" -Passed $packageIdentityReady -Detail "managedPackageUrlReady=$(Test-NonEmpty $managedPackageUrl); runtimePackageUrlReady=$(Test-NonEmpty $runtimePackageUrl)" -Boundary "Final close requires real managed/runtime package URLs and SHA256 values."
  New-ValidationItem -Id "owner-proof-input-promoted" -Passed $ownerProofPromote -Detail "validationState=$ownerProofState; canPromoteOwnerProofInput=$ownerProofPromote" -Boundary "Owner proof input template/readiness/schema records cannot satisfy final close."
  New-ValidationItem -Id "post-publish-verification-proof" -Passed ($postPublishProof -and $postPublishCanClose) -Detail "validationState=$postPublishState; isPostPublishVerificationProof=$postPublishProof; canCloseReleaseIssue=$postPublishCanClose" -Boundary "Only real post-publish validation proof can close the release issue."
  New-ValidationItem -Id "release-close-preflight-passed" -Passed $preflightCanClose -Detail "preflightState=$preflightState; failedItemCount=$preflightFailedItemCount; canCloseReleaseIssue=$preflightCanClose" -Boundary "Final close requires release-close-preflight canCloseReleaseIssue=true."
  New-ValidationItem -Id "stale-release-claims-clean" -Passed ($staleFindingCount -eq 0) -Detail "findingCount=$staleFindingCount" -Boundary "Stale release claims block close even if other evidence passes."
  New-ValidationItem -Id "evidence-bundle-sha256-matches" -Passed $bundleHashMatches -Detail "bundleState=$evidenceBundleState; declaredSha256=$bundleHashDeclared; actualSha256=$evidenceBundleHashActual" -Boundary "Final close must pin the evidence bundle hash."
  New-ValidationItem -Id "rollback-plan-ready" -Passed $rollbackPlanReady -Detail "rollbackSummaryReady=$(Test-NonEmpty $rollbackSummary); packagePlanReady=$(Test-NonEmpty $rollbackPackagePlan)" -Boundary "Final close needs rollback/yank/deprecate guidance."
  New-ValidationItem -Id "declared-close-flags" -Passed ($declaredCanPublishPublicly -and $declaredCanCloseReleaseIssue -and $declaredPromoteCloseRecord) -Detail "declaredCanPublishPublicly=$declaredCanPublishPublicly; declaredCanCloseReleaseIssue=$declaredCanCloseReleaseIssue; declaredPromoteCloseRecord=$declaredPromoteCloseRecord" -Boundary "Owner must explicitly declare close readiness only after all real proof gates pass."
)

$failedItems = @($items | Where-Object { -not [bool]$_.passed })
$canCloseReleaseIssue = $failedItems.Count -eq 0
$canPromoteReleaseIssueCloseRecord = $canCloseReleaseIssue -and -not $isTemplate

if ($canPromoteReleaseIssueCloseRecord) {
  $validationState = "ready-for-owner-release-issue-close"
  $validatedProofClassification = "real-release-issue-close-proof"
}
elseif ($isTemplate) {
  $validationState = "blocked-template-only"
  $validatedProofClassification = "template-only"
}
else {
  $validationState = "blocked-real-proof-required"
  $validatedProofClassification = "blocked-real-proof-required"
}

$nonSubstitutes = @(
  "template",
  "draft",
  "schema-only",
  "dry-run-only",
  "precheck-only",
  "managed-readiness",
  "readiness snapshot",
  "collection package",
  "owner handoff",
  "local feed",
  "ProjectReference",
  "direct .nupkg reference",
  "dependency-probe-only",
  "bridge-only package consumer log",
  "CallbackAllocatorReadinessSnapshot",
  "blocked-by-cuda-driver",
  "missing package URL",
  "missing package SHA256",
  "missing post-publish proof",
  "missing owner final close decision",
  "mismatched evidence bundle SHA256"
)

$validation = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-issue-close-record-validation"
  sourceInputPath = $inputFullPath
  validationState = $validationState
  proofClassification = $validatedProofClassification
  isTemplate = $isTemplate
  canPromoteReleaseIssueCloseRecord = $canPromoteReleaseIssueCloseRecord
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $canCloseReleaseIssue
  validationItemCount = $items.Count
  failedValidationItemCount = $failedItems.Count
  validationItems = $items
  releaseIssueId = $releaseIssueId
  releaseIssueUrl = $releaseIssueUrl
  ownerFinalCloseDecision = $ownerFinalCloseDecision
  selectedChannel = $selectedChannelName
  selectedChannelSourceUri = $selectedChannelSourceUri
  ownerProofInputValidationState = $ownerProofState
  ownerProofInputCanPromote = $ownerProofPromote
  postPublishVerificationState = $postPublishState
  isPostPublishVerificationProof = $postPublishProof
  postPublishCanCloseReleaseIssue = $postPublishCanClose
  releaseClosePreflightState = $preflightState
  releaseClosePreflightFailedItemCount = $preflightFailedItemCount
  releaseClosePreflightCanCloseReleaseIssue = $preflightCanClose
  staleReleaseClaimsFindingCount = $staleFindingCount
  releaseEvidenceBundleState = $evidenceBundleState
  releaseEvidenceBundleSha256Declared = $bundleHashDeclared
  releaseEvidenceBundleSha256Actual = $evidenceBundleHashActual
  releaseEvidenceBundleSha256Matches = $bundleHashMatches
  rollbackPlanReady = $rollbackPlanReady
  nonSubstituteProofKinds = $nonSubstitutes
  boundary = "This validator only validates a final owner-filled close record against real release evidence. It does not publish packages, upload artifacts, or close an issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "release-issue-close-record-validation.json"
$markdownPath = Join-Path $artifactRoot "release-issue-close-record-validation.md"

$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $items | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | $(([string]$_.detail).Replace("|", "\|")) | $(([string]$_.boundary).Replace("|", "\|")) |"
}
$nonSubstituteLines = $nonSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# Release Issue Close Record Validation

- validation state: ``$validationState``
- proof classification: ``$validatedProofClassification``
- can promote release issue close record: ``$canPromoteReleaseIssueCloseRecord``
- performs publish: ``False``
- canPublishPublicly=false
- canCloseReleaseIssue: ``$canCloseReleaseIssue``
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

Write-Output "Release issue close record validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$validationState"
Write-Output "CanPromoteReleaseIssueCloseRecord=$canPromoteReleaseIssueCloseRecord"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=$canCloseReleaseIssue"
Write-Output "FailedValidationItemCount=$($failedItems.Count)"

if ($FailOnNotCloseReady -and -not $canPromoteReleaseIssueCloseRecord) {
  throw "Release issue close record is not close-ready. validationState=$validationState failedValidationItemCount=$($failedItems.Count)"
}
