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

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function New-BridgeLane {
  param([string]$Id, [AllowNull()][object]$Record, [string]$StateProperty, [string]$DefaultState, [string]$OwnerAction, [string]$Validator)
  $state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
  $blockedReasons = New-Object System.Collections.Generic.List[string]
  if ([string]::IsNullOrWhiteSpace($state) -or $state -like "missing-*") { $blockedReasons.Add("missing-validation-artifact") | Out-Null }
  if ($state -match "blocked|missing|required|invalid|draft|candidate|template") { $blockedReasons.Add("real-owner-proof-or-strict-validation-required") | Out-Null }
  if ($state -notmatch "ready|passed|close-ready|proof") { $blockedReasons.Add("not-ready-for-release-close") | Out-Null }
  [pscustomobject]@{
    id = $Id
    state = $state
    ready = $false
    ownerAction = $OwnerAction
    validator = $Validator
    validatorCommand = $Validator
    sourceArtifactStateProperty = $StateProperty
    blockedReason = (@($blockedReasons.ToArray()) -join "; ")
    blockedReasons = @($blockedReasons.ToArray())
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    isPostPublishProof = $false
    boundary = "Release close bridge lane only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$publicPublishDraftValidation = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-record-draft-validation.json"
$cleanConsumerDraftValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-record-draft-validation.json"
$forbiddenScanValidation = Read-JsonOrNull "artifacts\final-release\public-publish-forbidden-substitute-scan-validation.json"
$postPublishVerification = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$releaseIssueCloseStrictOwnerDecisionImport = Read-JsonOrNull "artifacts\final-release\release-issue-close-strict-owner-decision-import-validation.json"
$releaseIssueCloseRecord = Read-JsonOrNull "artifacts\final-release\release-issue-close-record-validation.json"

$lanes = @(
  New-BridgeLane -Id "public-publish-real-result-record" -Record $publicPublishDraftValidation -StateProperty "validationState" -DefaultState "missing-public-publish-real-result-record-draft-validation" -OwnerAction "Fill real public package URL, SHA256, timestamp, transcript, owner account, reviewer, and rollback review." -Validator "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict"
  New-BridgeLane -Id "post-publish-clean-consumer-proof-record" -Record $cleanConsumerDraftValidation -StateProperty "validationState" -DefaultState "missing-post-publish-clean-consumer-proof-record-draft-validation" -OwnerAction "Fill repository-external restore/build/smoke logs, hashes, host metadata, and no-substitute confirmations." -Validator "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict"
  New-BridgeLane -Id "forbidden-substitute-scan" -Record $forbiddenScanValidation -StateProperty "validationState" -DefaultState "missing-public-publish-forbidden-substitute-scan-validation" -OwnerAction "Confirm no local feed, ProjectReference, direct nupkg, template, dry-run, dashboard, audit pack, or local-only scan was substituted." -Validator "eng\Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict"
  New-BridgeLane -Id "post-publish-verification" -Record $postPublishVerification -StateProperty "validationState" -DefaultState "missing-post-publish-verification-validation" -OwnerAction "Import or rerun post-publish verification against the real public package." -Validator "eng\Test-PostPublishVerification.ps1"
  New-BridgeLane -Id "strict-owner-decision-import" -Record $releaseIssueCloseStrictOwnerDecisionImport -StateProperty "validationState" -DefaultState "missing-release-issue-close-strict-owner-decision-import-validation" -OwnerAction "Import final owner close decision after real records are filled." -Validator "eng\Test-ReleaseIssueCloseStrictOwnerDecisionImport.ps1 -Strict"
  New-BridgeLane -Id "release-issue-close-record" -Record $releaseIssueCloseRecord -StateProperty "validationState" -DefaultState "missing-release-issue-close-record-validation" -OwnerAction "Run final strict close validator only after all real proof records pass." -Validator "eng\Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady"
)

$blockedLanes = @($lanes | Where-Object { -not [bool]$_.ready })
$forbiddenSubstituteMarkers = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "build-only",
  "dry-run",
  "candidate",
  "dashboard",
  "blocked-by-driver",
  "audit pack",
  "template",
  "draft"
)
$summary = [pscustomobject]@{
  bridgeInputOnly = $true
  blockedLaneCount = $blockedLanes.Count
  readyLaneCount = 0
  blockedLanes = @($blockedLanes | ForEach-Object { [string]$_.id })
  proofPromotionAllowed = $false
  publishAllowed = $false
  releaseCloseAllowed = $false
}

$record = [pscustomobject]@{
  recordKind = "release-close-real-proof-import-bridge"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bridgeState = "blocked-release-close-real-proof-import-required"
  laneCount = $lanes.Count
  blockedLaneCount = $blockedLanes.Count
  readyLaneCount = 0
  forbiddenSubstituteMarkers = $forbiddenSubstituteMarkers
  summary = $summary
  bridgeLanes = @($lanes)
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-real-result-record-draft-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-proof-record-draft-validation.json",
    "artifacts/final-release/public-publish-forbidden-substitute-scan-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/release-issue-close-strict-owner-decision-import-validation.json",
    "artifacts/final-release/release-issue-close-record-validation.json"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "This bridge maps real proof import blockers for owner action only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-close-real-proof-import-bridge.json"
$markdownPath = Join-Path $OutputRoot "release-close-real-proof-import-bridge.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Release Close Real Proof Import Bridge",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| bridgeState | ``$($record.bridgeState)`` |",
  "| laneCount | ``$($record.laneCount)`` |",
  "| blockedLaneCount | ``$($record.blockedLaneCount)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Bridge Lanes",
  "",
  "| Lane | State | Owner Action | Validator |",
  "| --- | --- | --- | --- |"
)

foreach ($lane in $lanes) {
  $markdown += "| $($lane.id) | ``$($lane.state)`` | $($lane.ownerAction) | ``$($lane.validator)`` |"
}

$markdown += @("", "## Boundary", "", $record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release close real proof import bridge written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "BridgeState=$($record.bridgeState) Lanes=$($record.laneCount) Blocked=$($record.blockedLaneCount)"
