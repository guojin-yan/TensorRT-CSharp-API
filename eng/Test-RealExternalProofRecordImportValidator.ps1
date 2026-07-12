[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-external-proof-record-import-validator.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
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

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real external proof record import validator not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$validatorState = [string](Get-PropertyOrDefault -Object $record -Name "validatorState" -DefaultValue "")
$contracts = @(Get-PropertyOrDefault -Object $record -Name "candidateContracts" -DefaultValue @())
$blockedContracts = @($contracts | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "contractState" -DefaultValue "") -eq "blocked-real-external-proof-record-import-required" })
$readyContracts = @($contracts | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "readyForPromotionGuard" -DefaultValue $false) })
$fileMissingCount = ($contracts | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "fileMissingCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $fileMissingCount) { $fileMissingCount = 0 }
$invalidSha256Count = ($contracts | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "invalidSha256Count" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $invalidSha256Count) { $invalidSha256Count = 0 }
$hashMismatchCount = ($contracts | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "hashMismatchCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $hashMismatchCount) { $hashMismatchCount = 0 }
$outsideAllowedEvidenceRootCount = ($contracts | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "outsideAllowedEvidenceRootCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $outsideAllowedEvidenceRootCount) { $outsideAllowedEvidenceRootCount = 0 }
$forbiddenSubstituteFindingCount = ($contracts | ForEach-Object { [int](Get-PropertyOrDefault -Object $_ -Name "forbiddenSubstituteFindingCount" -DefaultValue 0) } | Measure-Object -Sum).Sum
if ($null -eq $forbiddenSubstituteFindingCount) { $forbiddenSubstituteFindingCount = 0 }

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "real-external-proof-record-import-validator") -Severity "blocker" -Detail "recordKind must be real-external-proof-record-import-validator.")) | Out-Null
$items.Add((New-ValidationItem -Id "validator-state" -Passed ($validatorState -eq "blocked-real-external-proof-record-import-required" -or $validatorState -eq "real-external-proof-record-import-ready") -Severity "blocker" -Detail "Validator state must be blocked or ready.")) | Out-Null
$items.Add((New-ValidationItem -Id "contract-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "candidateContractCount" -DefaultValue 0) -eq 6 -and $contracts.Count -eq 6) -Severity "blocker" -Detail "Validator must cover 6 proof contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-contract-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedCandidateContractCount" -DefaultValue -1) -eq $blockedContracts.Count) -Severity "blocker" -Detail "Blocked contract count must match contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-contract-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyCandidateContractCount" -DefaultValue -1) -eq $readyContracts.Count) -Severity "blocker" -Detail "Ready contract count must match contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "evidence-failure-counts-projected" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "fileMissingCount" -DefaultValue -1) -eq $fileMissingCount -and [int](Get-PropertyOrDefault -Object $record -Name "invalidSha256Count" -DefaultValue -1) -eq $invalidSha256Count -and [int](Get-PropertyOrDefault -Object $record -Name "hashMismatchCount" -DefaultValue -1) -eq $hashMismatchCount -and [int](Get-PropertyOrDefault -Object $record -Name "outsideAllowedEvidenceRootCount" -DefaultValue -1) -eq $outsideAllowedEvidenceRootCount -and [int](Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteFindingCount" -DefaultValue -1) -eq $forbiddenSubstituteFindingCount) -Severity "blocker" -Detail "Validator must project classified evidence failures inherited from owner result import contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Validator must not promote proof by itself.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-or-close" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -Severity "blocker" -Detail "Validator must not publish or close release issue.")) | Out-Null

foreach ($contract in $contracts) {
  $contractId = [string](Get-PropertyOrDefault -Object $contract -Name "candidateContractId" -DefaultValue "unknown-contract")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contract -Name "resultImportItemId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contract -Name "proofLane" -DefaultValue "")) -and
    @((Get-PropertyOrDefault -Object $contract -Name "forbiddenSubstitutes" -DefaultValue @())).Count -ge 8 -and
    ($contract.PSObject.Properties.Name -contains "fileMissingCount") -and
    ($contract.PSObject.Properties.Name -contains "invalidSha256Count") -and
    ($contract.PSObject.Properties.Name -contains "hashMismatchCount") -and
    ($contract.PSObject.Properties.Name -contains "outsideAllowedEvidenceRootCount") -and
    ($contract.PSObject.Properties.Name -contains "forbiddenSubstituteFindingCount") -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contract -Name "strictValidatorCommand" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$contractId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each contract must include lane identity, classified evidence counts, forbidden substitutes, and strict validator command.")) | Out-Null
  if (-not [bool](Get-PropertyOrDefault -Object $contract -Name "readyForPromotionGuard" -DefaultValue $false)) {
    $items.Add((New-ValidationItem -Id "$contractId-real-proof-import-required" -Passed $false -Severity "action-required" -Detail "Owner result import must be complete before this contract can feed promotion guard.")) | Out-Null
  }
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not $_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-real-external-proof-record-import-validator"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-real-external-proof-record-import-required"
}
else {
  "real-external-proof-record-import-ready"
}

$validation = [pscustomobject]@{
  recordKind = "real-external-proof-record-import-validator-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateContractCount = $contracts.Count
  blockedCandidateContractCount = $blockedContracts.Count
  readyCandidateContractCount = $readyContracts.Count
  fileMissingCount = [int]$fileMissingCount
  invalidSha256Count = [int]$invalidSha256Count
  hashMismatchCount = [int]$hashMismatchCount
  outsideAllowedEvidenceRootCount = [int]$outsideAllowedEvidenceRootCount
  forbiddenSubstituteFindingCount = [int]$forbiddenSubstituteFindingCount
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "This validates real external proof import contracts only. It does not promote runtime proof, publish packages, or close release issue."
}

$jsonPath = Join-Path $OutputRoot "real-external-proof-record-import-validator-validation.json"
$markdownPath = Join-Path $OutputRoot "real-external-proof-record-import-validator-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Real External Proof Record Import Validator Validation

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateContractCount | ``$($validation.candidateContractCount)`` |
| fileMissingCount | ``$($validation.fileMissingCount)`` |
| invalidSha256Count | ``$($validation.invalidSha256Count)`` |
| hashMismatchCount | ``$($validation.hashMismatchCount)`` |
| outsideAllowedEvidenceRootCount | ``$($validation.outsideAllowedEvidenceRootCount)`` |
| forbiddenSubstituteFindingCount | ``$($validation.forbiddenSubstituteFindingCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |

## Boundary

$($validation.safetyBoundary)
"@
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real external proof record import validator validation written to $jsonPath"
Write-Host "Real external proof record import validator validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) FailedActionRequired=$($failedActionRequired.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Real external proof record import validator validation failed with $($failedBlockers.Count) blocker(s)."
}
