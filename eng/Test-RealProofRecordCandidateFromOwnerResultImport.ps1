[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-record-candidate-from-owner-result-import.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof record candidate from owner result import not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$candidateItems = @(Get-PropertyOrDefault -Object $record -Name "candidateItems" -DefaultValue @())
$strictReadyCandidates = @($candidateItems | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "strictValidatorInputReady" -DefaultValue $false) })
$postPublishCandidates = @($candidateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "proofLane" -DefaultValue "") -eq "post-publish-verification" })
$packageConsumerCandidates = @($candidateItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "proofLane" -DefaultValue "") -eq "package-consumer-runtime" })
$summary = Get-PropertyOrDefault -Object $record -Name "summary" -DefaultValue $null
$laneGroups = @((Get-PropertyOrDefault -Object $record -Name "laneGroups" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "real-proof-record-candidate-from-owner-result-import") -Severity "blocker" -Detail "recordKind must identify the owner-result import candidate bridge.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-count-projected" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "candidateCount" -DefaultValue -1) -eq $candidateItems.Count -and [int](Get-PropertyOrDefault -Object $record -Name "strictValidatorReadyCandidateCount" -DefaultValue -1) -eq $strictReadyCandidates.Count) -Severity "blocker" -Detail "Candidate counts must match projected ready contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "only-ready-contracts-project" -Passed ($candidateItems.Count -eq [int](Get-PropertyOrDefault -Object $record -Name "sourceReadyCandidateContractCount" -DefaultValue -1)) -Severity "blocker" -Detail "Only readyForPromotionGuard contracts may become candidates.")) | Out-Null
$items.Add((New-ValidationItem -Id "post-publish-lane-separated" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "postPublishVerificationCandidateCount" -DefaultValue -1) -eq $postPublishCandidates.Count -and [int](Get-PropertyOrDefault -Object $record -Name "packageConsumerRuntimeCandidateCount" -DefaultValue -1) -eq $packageConsumerCandidates.Count) -Severity "blocker" -Detail "Post-publish candidates must remain separate from package-consumer-runtime candidates.")) | Out-Null
$items.Add((New-ValidationItem -Id "strict-validator-input-summary" -Passed ($null -ne $summary -and [bool](Get-PropertyOrDefault -Object $summary -Name "strictValidatorInputOnly" -DefaultValue $false) -and -not [bool](Get-PropertyOrDefault -Object $summary -Name "proofPromotionAllowed" -DefaultValue $true) -and [int](Get-PropertyOrDefault -Object $summary -Name "candidateCount" -DefaultValue -1) -eq $candidateItems.Count -and [int](Get-PropertyOrDefault -Object $summary -Name "strictValidatorReadyCandidateCount" -DefaultValue -1) -eq $strictReadyCandidates.Count) -Severity "blocker" -Detail "Candidate bridge must expose a non-promotable strict-validator input summary.")) | Out-Null
$items.Add((New-ValidationItem -Id "lane-groups-projected" -Passed (($candidateItems.Count -eq 0 -and $laneGroups.Count -eq 0) -or ($candidateItems.Count -gt 0 -and $laneGroups.Count -gt 0)) -Severity "blocker" -Detail "Candidate bridge must group candidates by proof lane when candidates exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Candidate bridge must not promote runtime, post-publish, or release-close proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-or-close" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -Severity "blocker" -Detail "Candidate bridge must not publish or close release issue.")) | Out-Null

foreach ($candidate in $candidateItems) {
  $id = [string](Get-PropertyOrDefault -Object $candidate -Name "proofRecordCandidateId" -DefaultValue "unknown-candidate")
  $fileEvidenceChecks = @(Get-PropertyOrDefault -Object $candidate -Name "fileEvidenceChecks" -DefaultValue @())
  $sourceOwnerResultRow = Get-PropertyOrDefault -Object $candidate -Name "sourceOwnerResultRow" -DefaultValue $null
  $resultArtifactPaths = @(Get-PropertyOrDefault -Object $candidate -Name "resultArtifactPaths" -DefaultValue @())
  $hashProof = @(Get-PropertyOrDefault -Object $candidate -Name "hashProof" -DefaultValue @())
  $shapeReady = -not [string]::IsNullOrWhiteSpace($id) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $candidate -Name "proofLane" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $candidate -Name "resultInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $candidate -Name "executionInputId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $candidate -Name "candidateId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $candidate -Name "runtimePackageKey" -DefaultValue "")) -and
    $fileEvidenceChecks.Count -ge 5 -and
    $null -ne $sourceOwnerResultRow -and
    $resultArtifactPaths.Count -ge 1 -and
    $hashProof.Count -ge 5 -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $candidate -Name "resultArtifactPath" -DefaultValue "")) -and
    [bool](Get-PropertyOrDefault -Object $candidate -Name "ownerReviewReady" -DefaultValue $false) -and
    [bool](Get-PropertyOrDefault -Object $candidate -Name "nonSubstituteConfirmationsReady" -DefaultValue $false) -and
    -not [bool](Get-PropertyOrDefault -Object $candidate -Name "canPromoteRuntimeProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $candidate -Name "isRuntimeExecutionProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $candidate -Name "isPostPublishProof" -DefaultValue $true) -and
    -not [bool](Get-PropertyOrDefault -Object $candidate -Name "canCloseReleaseIssue" -DefaultValue $true)

  $items.Add((New-ValidationItem -Id "$id-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each candidate must inherit identity, file evidence checks, owner review, non-substitute status, and false proof flags.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-strict-validator-ready" -Passed ([bool](Get-PropertyOrDefault -Object $candidate -Name "strictValidatorInputReady" -DefaultValue $false)) -Severity "blocker" -Detail "Projected candidates must be ready only as strict validator inputs.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-still-not-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $candidate -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $candidate -Name "isPostPublishProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $candidate -Name "canCloseReleaseIssue" -DefaultValue $true)) -Severity "blocker" -Detail "Projected candidate must remain non-proof.")) | Out-Null
}

if ($postPublishCandidates.Count -gt 0) {
  foreach ($candidate in $postPublishCandidates) {
    $id = [string](Get-PropertyOrDefault -Object $candidate -Name "proofRecordCandidateId" -DefaultValue "post-publish-candidate")
    $boundary = [string](Get-PropertyOrDefault -Object $candidate -Name "postPublishBoundary" -DefaultValue "")
    $items.Add((New-ValidationItem -Id "$id-post-publish-boundary" -Passed ($boundary -match "package-consumer-runtime cannot substitute") -Severity "blocker" -Detail "Post-publish candidate must explicitly reject package-consumer-runtime substitution.")) | Out-Null
  }
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-real-proof-record-candidate-from-owner-result-import" } elseif ($candidateItems.Count -gt 0) { "owner-result-import-candidate-ready-for-strict-validator" } else { "blocked-owner-result-import-candidate-required" }

$validation = [pscustomobject]@{
  recordKind = "real-proof-record-candidate-from-owner-result-import-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateCount = $candidateItems.Count
  strictValidatorReadyCandidateCount = $strictReadyCandidates.Count
  packageConsumerRuntimeCandidateCount = $packageConsumerCandidates.Count
  postPublishVerificationCandidateCount = $postPublishCandidates.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = 0
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates only the owner-result-import candidate bridge. It is not runtime proof, post-publish proof, publication approval, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-record-candidate-from-owner-result-import-validation.json"
$markdownPath = Join-Path $OutputRoot "real-proof-record-candidate-from-owner-result-import-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Real Proof Record Candidate From Owner Result Import Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateCount | ``$($validation.candidateCount)`` |
| strictValidatorReadyCandidateCount | ``$($validation.strictValidatorReadyCandidateCount)`` |
| packageConsumerRuntimeCandidateCount | ``$($validation.packageConsumerRuntimeCandidateCount)`` |
| postPublishVerificationCandidateCount | ``$($validation.postPublishVerificationCandidateCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isPostPublishProof | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join [Environment]::NewLine)

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof record candidate from owner result import validation written to $jsonPath"
Write-Host "Real proof record candidate from owner result import validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState Candidates=$($validation.candidateCount) FailedBlockers=$($validation.failedBlockerCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Real proof record candidate from owner result import validation failed with $($failedBlockers.Count) blocker(s)."
}
