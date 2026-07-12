[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-input-candidate-strict-record.json",
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
  throw "Real proof input candidate strict record not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$candidateState = [string](Get-PropertyOrDefault -Object $record -Name "candidateState" -DefaultValue "")
$candidates = @(Get-PropertyOrDefault -Object $record -Name "strictCandidateRecords" -DefaultValue @())
$requiredProofLanes = @(Get-PropertyOrDefault -Object $record -Name "requiredProofLanes" -DefaultValue @())
$forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $record -Name "forbiddenSubstitutes" -DefaultValue @())

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "real-proof-input-candidate-strict-record") -Severity "blocker" -Detail "recordKind must be real-proof-input-candidate-strict-record.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-state" -Passed ($candidateState -eq "blocked-real-proof-input-candidate-required") -Severity "blocker" -Detail "Default strict candidate record must remain blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-count" -Passed ($candidates.Count -eq 6) -Severity "blocker" -Detail "Strict candidate record must include exactly 6 candidate records.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-candidate-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyCandidateCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default strict candidate record must not claim ready candidates.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-candidate-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedCandidateCount" -DefaultValue -1) -eq $candidates.Count) -Severity "blocker" -Detail "Default strict candidate record must keep all candidates blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Strict candidate record must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Strict candidate record must not publish or approve public publication.")) | Out-Null

foreach ($lane in @("package-consumer-runtime", "post-publish-verification", "linux-runner-proof", "real-model-runtime", "release-close-owner-input", "strict-close-validation")) {
  $items.Add((New-ValidationItem -Id "lane-$lane" -Passed ($requiredProofLanes -contains $lane -and @($candidates | Where-Object { [string]$_.proofLane -eq $lane }).Count -eq 1) -Severity "blocker" -Detail "Required proof lane must be present exactly once: $lane.")) | Out-Null
}

foreach ($required in @("local feed", "ProjectReference", "direct .nupkg", "DependencyProbe", "build-only", "template", "Windows handoff for Linux proof", "hash-only audit")) {
  $items.Add((New-ValidationItem -Id "forbidden-$($required.Replace(' ', '-').Replace('.', '').ToLowerInvariant())" -Passed ($forbiddenSubstitutes -contains $required) -Severity "blocker" -Detail "Forbidden substitute must be listed: $required.")) | Out-Null
}

foreach ($candidate in $candidates) {
  $candidateId = [string](Get-PropertyOrDefault -Object $candidate -Name "candidateId" -DefaultValue "unknown-candidate")
  $fieldContracts = @(Get-PropertyOrDefault -Object $candidate -Name "fieldContracts" -DefaultValue @())
  $blockedFieldCount = [int](Get-PropertyOrDefault -Object $candidate -Name "blockedFieldCount" -DefaultValue -1)
  $blockedFields = @($fieldContracts | Where-Object { -not [bool]$_.ready })

  $items.Add((New-ValidationItem -Id "$candidateId-state-blocked" -Passed ([string](Get-PropertyOrDefault -Object $candidate -Name "candidateState" -DefaultValue "") -eq "blocked-real-proof-input-candidate-required") -Severity "blocker" -Detail "Default candidate must remain blocked until owner evidence is complete.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-field-contract-shape" -Passed ($fieldContracts.Count -ge 7) -Severity "blocker" -Detail "Each candidate must include strict field contracts.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-blocked-field-count" -Passed ($blockedFieldCount -eq 0 -and $blockedFields.Count -eq 0) -Severity "action-required" -Detail "Owner must fill every blocked field contract before candidate review.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-not-ready-for-owner-review" -Passed (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "readyForOwnerReview" -DefaultValue $true)) -Severity "blocker" -Detail "Default candidate must not claim readyForOwnerReview.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-not-ready-for-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "readyForPromotion" -DefaultValue $true)) -Severity "blocker" -Detail "Default candidate must not claim readyForPromotion.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-no-proof-flags" -Passed (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $candidate -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $candidate -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $candidate -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Candidate proof flags must remain false.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-real-proof-input-candidate-strict-record"
}
else {
  "blocked-real-proof-input-candidate-required"
}

$validation = [pscustomobject]@{
  recordKind = "real-proof-input-candidate-strict-record-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateCount = $candidates.Count
  blockedCandidateCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCandidateCount" -DefaultValue $candidates.Count)
  readyCandidateCount = [int](Get-PropertyOrDefault -Object $record -Name "readyCandidateCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates strict candidate record shape only. It is not runtime proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-input-candidate-strict-record-validation.json"
$markdownPath = Join-Path $OutputRoot "real-proof-input-candidate-strict-record-validation.md"
$validation | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Real Proof Input Candidate Strict Record Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateCount | ``$($validation.candidateCount)`` |
| blockedCandidateCount | ``$($validation.blockedCandidateCount)`` |
| readyCandidateCount | ``$($validation.readyCandidateCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isReleaseCloseProof | ``False`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---|---|---|
$($rows -join [Environment]::NewLine)

## Boundary

$($validation.boundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof input candidate strict record validation written to $jsonPath"
Write-Host "Real proof input candidate strict record validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState Candidates=$($validation.candidateCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Real proof input candidate strict record validation failed with $($failedBlockers.Count) blocker(s)."
}
