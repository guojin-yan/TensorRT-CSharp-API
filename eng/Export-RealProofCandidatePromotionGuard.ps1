[CmdletBinding()]
param(
  [string]$StrictRecordPath = "artifacts\final-release\real-proof-input-candidate-strict-record.json",
  [string]$DeltaPackPath = "artifacts\final-release\owner-real-proof-field-delta-pack.json",
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-GuardRequirement {
  param([string]$Name, [string]$Rule, [bool]$Passed)

  [pscustomobject]@{
    name = $Name
    rule = $Rule
    passed = $Passed
    required = $true
    status = if ($Passed) { "promotion-guard-requirement-ready" } else { "blocked-promotion-guard-requirement" }
  }
}

$resolvedStrictRecordPath = Resolve-InputPath -Path $StrictRecordPath
$resolvedDeltaPackPath = Resolve-InputPath -Path $DeltaPackPath
if (-not (Test-Path -LiteralPath $resolvedStrictRecordPath -PathType Leaf)) {
  throw "Real proof input candidate strict record not found: $resolvedStrictRecordPath"
}
if (-not (Test-Path -LiteralPath $resolvedDeltaPackPath -PathType Leaf)) {
  throw "Owner real proof field delta pack not found: $resolvedDeltaPackPath"
}

$strictRecord = Get-Content -LiteralPath $resolvedStrictRecordPath -Raw -Encoding utf8 | ConvertFrom-Json
$deltaPack = Get-Content -LiteralPath $resolvedDeltaPackPath -Raw -Encoding utf8 | ConvertFrom-Json
$candidates = @(Get-PropertyOrDefault -Object $strictRecord -Name "strictCandidateRecords" -DefaultValue @())
$deltas = @(Get-PropertyOrDefault -Object $deltaPack -Name "fieldDeltas" -DefaultValue @())

$guardItems = @()
foreach ($candidate in $candidates) {
  $candidateId = [string](Get-PropertyOrDefault -Object $candidate -Name "candidateId" -DefaultValue "unknown-candidate")
  $fieldContracts = @(Get-PropertyOrDefault -Object $candidate -Name "fieldContracts" -DefaultValue @())
  $blockedFieldContracts = @($fieldContracts | Where-Object { -not [bool]$_.ready })
  $candidateDeltas = @($deltas | Where-Object { [string]$_.candidateId -eq $candidateId })
  $blockedDeltas = @($candidateDeltas | Where-Object { -not [bool]$_.ready })

  $requirements = @(
    New-GuardRequirement -Name "allFieldContractsReady" -Rule "Every strict candidate field contract must be ready." -Passed ($blockedFieldContracts.Count -eq 0 -and $fieldContracts.Count -gt 0)
    New-GuardRequirement -Name "allOwnerDeltasClosed" -Rule "Every Owner field delta must be complete and ready." -Passed ($blockedDeltas.Count -eq 0 -and $candidateDeltas.Count -gt 0)
    New-GuardRequirement -Name "noSubstituteEvidence" -Rule "No template, candidate, hash-only, local feed, ProjectReference, direct nupkg, or DependencyProbe substitute may satisfy proof." -Passed $false
    New-GuardRequirement -Name "strictValidatorRequired" -Rule "A later real proof record validator must pass before proof promotion." -Passed $false
  )

  $blockedRequirementCount = @($requirements | Where-Object { -not [bool]$_.passed }).Count
  $promotionAllowed = $blockedRequirementCount -eq 0

  $guardItems += [pscustomobject]@{
    candidateId = $candidateId
    proofLane = [string](Get-PropertyOrDefault -Object $candidate -Name "proofLane" -DefaultValue "unknown-proof-lane")
    guardState = if ($promotionAllowed) { "candidate-promotion-review-ready" } else { "blocked-real-proof-candidate-promotion-not-allowed" }
    fieldContractCount = $fieldContracts.Count
    blockedFieldContractCount = $blockedFieldContracts.Count
    ownerDeltaCount = $candidateDeltas.Count
    blockedOwnerDeltaCount = $blockedDeltas.Count
    requirements = $requirements
    blockedRequirementCount = $blockedRequirementCount
    promotionAllowed = $promotionAllowed
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    boundary = "This guard item can only allow candidate review. It cannot promote runtime proof, publish packages, verify post-publish state, or close the release issue."
  }
}

$promotionAllowedCandidateCount = @($guardItems | Where-Object { [bool]$_.promotionAllowed }).Count
$blockedCandidateCount = @($guardItems | Where-Object { -not [bool]$_.promotionAllowed }).Count

$recordOut = [pscustomobject]@{
  recordKind = "real-proof-candidate-promotion-guard"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  strictRecordPath = $resolvedStrictRecordPath
  deltaPackPath = $resolvedDeltaPackPath
  guardState = "blocked-real-proof-candidate-promotion-not-allowed"
  candidateCount = $guardItems.Count
  promotionAllowedCandidateCount = $promotionAllowedCandidateCount
  blockedCandidateCount = $blockedCandidateCount
  guardItems = @($guardItems)
  sourceArtifacts = @(
    "artifacts/final-release/real-proof-input-candidate-strict-record.json",
    "artifacts/final-release/real-proof-input-candidate-strict-record-validation.json",
    "artifacts/final-release/owner-real-proof-field-delta-pack.json",
    "artifacts/final-release/owner-real-proof-field-delta-pack-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This promotion guard prevents strict candidates from being mistaken for proof. It does not promote proof, publish packages, verify post-publish state, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "real-proof-candidate-promotion-guard.json"
$markdownPath = Join-Path $OutputRoot "real-proof-candidate-promotion-guard.md"
$recordOut | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real Proof Candidate Promotion Guard")
$lines.Add("")
$lines.Add("`real-proof-candidate-promotion-guard` 汇总 strict candidate 和 Owner delta 的提升条件，防止 candidate 被误判为 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| guardState | ``$(ConvertTo-MarkdownCell $recordOut.guardState)`` |")
$lines.Add("| candidateCount | ``$($recordOut.candidateCount)`` |")
$lines.Add("| promotionAllowedCandidateCount | ``$($recordOut.promotionAllowedCandidateCount)`` |")
$lines.Add("| blockedCandidateCount | ``$($recordOut.blockedCandidateCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Guard Items")
$lines.Add("")
$lines.Add("| Candidate | Lane | State | Blocked Fields | Blocked Deltas | Blocked Requirements |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($item in $guardItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.candidateId) | $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.guardState) | ``$($item.blockedFieldContractCount)`` | ``$($item.blockedOwnerDeltaCount)`` | ``$($item.blockedRequirementCount)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof candidate promotion guard written to $jsonPath"
Write-Host "Real proof candidate promotion guard markdown written to $markdownPath"
Write-Host "GuardState=$($recordOut.guardState) Candidates=$($recordOut.candidateCount) Blocked=$($recordOut.blockedCandidateCount) PromotionAllowed=$($recordOut.promotionAllowedCandidateCount)"
