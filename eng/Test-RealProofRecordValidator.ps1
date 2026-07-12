[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-record-validator.json",
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

function Test-NonEmptyArray {
  param([AllowNull()][object[]]$Items)

  return @($Items).Count -gt 0
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof record validator not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$contracts = @(Get-PropertyOrDefault -Object $record -Name "validatorContracts" -DefaultValue @())
$blockedContracts = @($contracts | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "contractState" -DefaultValue "") -eq "blocked-real-proof-record-validation-input-required" })
$readyContracts = @($contracts | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "validatorReady" -DefaultValue $false) })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "real-proof-record-validator") -Severity "blocker" -Detail "recordKind must be real-proof-record-validator.")) | Out-Null
$items.Add((New-ValidationItem -Id "validator-state" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "validatorState" -DefaultValue "") -eq "blocked-real-proof-record-validation-input-required") -Severity "blocker" -Detail "Validator record must remain blocked until real proof inputs are provided.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "candidateCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Validator must cover 6 candidates.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-contract-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "blockedValidatorContractCount" -DefaultValue -1) -eq $blockedContracts.Count -and $blockedContracts.Count -ge 6) -Severity "blocker" -Detail "Default validator must keep all candidate contracts blocked.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-contract-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "readyValidatorContractCount" -DefaultValue -1) -eq $readyContracts.Count -and $readyContracts.Count -eq 0) -Severity "blocker" -Detail "Default validator must not claim ready contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Validator contract must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Validator contract must not publish or approve public publication.")) | Out-Null

foreach ($contract in $contracts) {
  $candidateId = [string](Get-PropertyOrDefault -Object $contract -Name "candidateId" -DefaultValue "unknown-candidate")
  $shapeReady = -not [string]::IsNullOrWhiteSpace($candidateId) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contract -Name "proofLane" -DefaultValue "")) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $contract -Name "requiredRuntimeEvidence" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $contract -Name "requiredHostMetadata" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $contract -Name "requiredPackageIdentity" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $contract -Name "requiredCommandCapture" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $contract -Name "requiredLogHash" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $contract -Name "requiredValidatorOutput" -DefaultValue @())) -and
    (Test-NonEmptyArray -Items @(Get-PropertyOrDefault -Object $contract -Name "forbiddenSubstitutes" -DefaultValue @())) -and
    [bool](Get-PropertyOrDefault -Object $contract -Name "ownerReviewRequired" -DefaultValue $false) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contract -Name "promotionGuardState" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contract -Name "validatorCommand" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contract -Name "proofBoundary" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$candidateId-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each validator contract must include all required proof input fields and boundaries.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$candidateId-owner-action-required" -Passed ([bool](Get-PropertyOrDefault -Object $contract -Name "validatorReady" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must provide real proof input and satisfy promotion guard before validator contract can become ready.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) { "invalid-real-proof-record-validator" } else { "blocked-real-proof-record-validation-input-required" }

$validation = [pscustomobject]@{
  recordKind = "real-proof-record-validator-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateCount = [int](Get-PropertyOrDefault -Object $record -Name "candidateCount" -DefaultValue 0)
  blockedValidatorContractCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedValidatorContractCount" -DefaultValue 0)
  readyValidatorContractCount = [int](Get-PropertyOrDefault -Object $record -Name "readyValidatorContractCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates the real proof record validator contract shape only. It is not runtime proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-record-validator-validation.json"
$markdownPath = Join-Path $OutputRoot "real-proof-record-validator-validation.md"
$validation | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Real Proof Record Validator Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateCount | ``$($validation.candidateCount)`` |
| blockedValidatorContractCount | ``$($validation.blockedValidatorContractCount)`` |
| readyValidatorContractCount | ``$($validation.readyValidatorContractCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| canPromoteRuntimeProof | ``False`` |
| canPublishPublicly | ``False`` |
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

Write-Host "Real proof record validator validation written to $jsonPath"
Write-Host "Real proof record validator validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState Candidates=$($validation.candidateCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Real proof record validator validation failed with $($failedBlockers.Count) blocker(s)."
}
