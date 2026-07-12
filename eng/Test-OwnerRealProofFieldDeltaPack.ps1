[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-real-proof-field-delta-pack.json",
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
  throw "Owner real proof field delta pack not found: $resolvedInputPath"
}

$pack = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$deltas = @(Get-PropertyOrDefault -Object $pack -Name "fieldDeltas" -DefaultValue @())
$blockedDeltas = @($deltas | Where-Object { [string]$_.deltaState -eq "blocked-owner-real-proof-field-delta-required" })

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $pack -Name "recordKind" -DefaultValue "") -eq "owner-real-proof-field-delta-pack") -Severity "blocker" -Detail "recordKind must be owner-real-proof-field-delta-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "delta-state" -Passed ([string](Get-PropertyOrDefault -Object $pack -Name "deltaState" -DefaultValue "") -eq "blocked-owner-real-proof-field-delta-required") -Severity "blocker" -Detail "Delta pack must remain blocked while Owner field deltas are incomplete.")) | Out-Null
$items.Add((New-ValidationItem -Id "candidate-count" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "candidateCount" -DefaultValue 0) -eq 6) -Severity "blocker" -Detail "Delta pack must cover 6 candidates.")) | Out-Null
$items.Add((New-ValidationItem -Id "field-delta-count" -Passed ($deltas.Count -ge 6) -Severity "blocker" -Detail "Delta pack must include blocked field deltas.")) | Out-Null
$items.Add((New-ValidationItem -Id "blocked-field-contract-count" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "blockedFieldContractCount" -DefaultValue -1) -eq $blockedDeltas.Count -and $blockedDeltas.Count -ge 6) -Severity "blocker" -Detail "Blocked field contract count must match blocked deltas.")) | Out-Null
$items.Add((New-ValidationItem -Id "ready-field-contract-count" -Passed ([int](Get-PropertyOrDefault -Object $pack -Name "readyFieldContractCount" -DefaultValue -1) -eq 0) -Severity "blocker" -Detail "Default delta pack must not claim ready field contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-proof-promotion" -Passed (-not [bool](Get-PropertyOrDefault -Object $pack -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "isRuntimeExecutionProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "isReleaseCloseProof" -DefaultValue $true)) -Severity "blocker" -Detail "Delta pack must not promote proof or close readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not [bool](Get-PropertyOrDefault -Object $pack -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $pack -Name "canPublishPublicly" -DefaultValue $true)) -Severity "blocker" -Detail "Delta pack must not publish or approve public publication.")) | Out-Null

foreach ($delta in $deltas) {
  $id = [string](Get-PropertyOrDefault -Object $delta -Name "fieldDeltaId" -DefaultValue "unknown-delta")
  $shapeReady = -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $delta -Name "candidateId" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $delta -Name "fieldName" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $delta -Name "ownerAction" -DefaultValue "")) -and
    -not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $delta -Name "targetValidationCommand" -DefaultValue ""))
  $items.Add((New-ValidationItem -Id "$id-shape" -Passed $shapeReady -Severity "blocker" -Detail "Each field delta must include candidate, field, owner action, and validation command.")) | Out-Null
  $items.Add((New-ValidationItem -Id "$id-owner-action-required" -Passed ([bool](Get-PropertyOrDefault -Object $delta -Name "ready" -DefaultValue $false)) -Severity "action-required" -Detail "Owner must complete field delta before candidate promotion review.")) | Out-Null
}

$failedBlockers = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($items | Where-Object { -not [bool]$_.passed -and $_.severity -eq "action-required" })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-owner-real-proof-field-delta-pack"
}
else {
  "blocked-owner-real-proof-field-delta-required"
}

$validation = [pscustomobject]@{
  recordKind = "owner-real-proof-field-delta-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  candidateCount = [int](Get-PropertyOrDefault -Object $pack -Name "candidateCount" -DefaultValue 0)
  fieldDeltaCount = $deltas.Count
  blockedFieldContractCount = [int](Get-PropertyOrDefault -Object $pack -Name "blockedFieldContractCount" -DefaultValue 0)
  readyFieldContractCount = [int](Get-PropertyOrDefault -Object $pack -Name "readyFieldContractCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  boundary = "This validates Owner field deltas only. It is not runtime proof, not publication approval, and not release close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-real-proof-field-delta-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-real-proof-field-delta-pack-validation.md"
$validation | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validation.validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $($_.detail.Replace('|', '\|')) |"
}

$markdown = @"
# Owner Real Proof Field Delta Pack Validation

生成时间：$($validation.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($validation.validationState)`` |
| candidateCount | ``$($validation.candidateCount)`` |
| fieldDeltaCount | ``$($validation.fieldDeltaCount)`` |
| blockedFieldContractCount | ``$($validation.blockedFieldContractCount)`` |
| readyFieldContractCount | ``$($validation.readyFieldContractCount)`` |
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

Write-Host "Owner real proof field delta pack validation written to $jsonPath"
Write-Host "Owner real proof field delta pack validation markdown written to $markdownPath"
Write-Host "ValidationState=$validationState FieldDeltas=$($validation.fieldDeltaCount) FailedBlockers=$($validation.failedBlockerCount) FailedActionRequired=$($validation.failedActionRequiredCount)"

if ($Strict -and $failedBlockers.Count -gt 0) {
  throw "Owner real proof field delta pack validation failed with $($failedBlockers.Count) blocker(s)."
}
